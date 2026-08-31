"""
User-facing configuration for model function code generation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Literal

CodegenMode = Literal["off", "auto", "subprocess", "inprocess", "build-only", "load-only"]

VALID_MODES: tuple[str, ...] = ("off", "auto", "subprocess", "inprocess", "build-only", "load-only")


@dataclass(frozen=True)
class CodegenOptions:
    """
    How a model's CasADi Functions should be compiled to a shared library.

    The defaults are chosen for a large model: compile in a subprocess (so the parent never
    holds the symbolic graphs), at ``-O1`` (``-O3`` on generated code of this size costs
    minutes for a few percent), without ``-march=native`` (which would make the cache
    non-portable between machines).

    Parameters
    ----------
    cache_dir: str | None
        Where compiled libraries live. ``None`` resolves to ``$BIOPTIM_CACHE_DIR``, then to
        the platform user cache directory. Never the current working directory.
    opt_level: int
        The ``-O`` level handed to the compiler. 0 compiles fast and runs ~2-4x slower;
        1 is the measured sweet spot; 2 and 3 cost a lot of wall time on large generated C.
    extra_flags: tuple[str, ...]
        Additional compiler flags. These are folded into the cache key, so a build with
        ``-march=native`` never gets reused by a build without it.
    jobs: int | None
        Parallel compiler processes. ``None`` means ``os.cpu_count()``.
    mode: str
        - ``"off"``: no code generation at all (the historical behaviour).
        - ``"auto"``: load from cache if possible, otherwise build in a subprocess.
        - ``"subprocess"``: same as auto, but never fall back to building in-process.
        - ``"inprocess"``: build in this process. Simpler to debug, but the peak RAM of the
          cold run is as bad as it is today, because glibc does not return the arena.
        - ``"build-only"``: build the library, then stop. For pre-warming a cache.
        - ``"load-only"``: use the cache if it is there, otherwise fall back to plain
          uncompiled Functions without ever invoking a compiler.
    emit_jacobian: bool
        Whether to emit ``jac_<name>`` alongside the function. This is what gives CasADi the
        exact Jacobian sparsity of the external; without it every call site contributes a
        dense block. Only turn it off for a function whose Jacobian is genuinely dense and
        very expensive to code-generate.
    rebuild: bool
        Ignore any existing cache entry and rebuild it.
    keep_sources: bool
        Keep the generated ``.c`` files next to the library, for inspection.
    cache_tag: str | None
        An arbitrary string folded into the cache key, to force a separate cache entry.
    max_cache_bytes: int
        Soft cap on the cache directory. Oldest entries are pruned past this on write.
    verbose: bool
        Print cache hits, misses and compiler invocations.
    """

    cache_dir: str | None = None
    opt_level: int = 1
    extra_flags: tuple[str, ...] = ()
    jobs: int | None = None
    mode: CodegenMode = "auto"
    emit_jacobian: bool = True
    rebuild: bool = False
    keep_sources: bool = False
    cache_tag: str | None = None
    max_cache_bytes: int = 2 * 1024**3
    verbose: bool = False

    def __post_init__(self):
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {self.mode!r}")
        if not isinstance(self.opt_level, int) or not 0 <= self.opt_level <= 3:
            raise ValueError(f"opt_level must be an int in [0, 3], got {self.opt_level!r}")
        if self.jobs is not None and (not isinstance(self.jobs, int) or self.jobs < 1):
            raise ValueError(f"jobs must be a positive int or None, got {self.jobs!r}")
        if not isinstance(self.extra_flags, tuple):
            # A list is the natural thing for a user to pass; normalize so the dataclass
            # stays hashable and the cache key stays stable.
            object.__setattr__(self, "extra_flags", tuple(self.extra_flags))

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def n_jobs(self) -> int:
        return self.jobs if self.jobs is not None else (os.cpu_count() or 1)

    @property
    def may_compile(self) -> bool:
        """Whether this mode is allowed to invoke a compiler."""
        return self.mode in ("auto", "subprocess", "inprocess", "build-only")

    @property
    def compiler_flags(self) -> tuple[str, ...]:
        return ("-fPIC", f"-O{self.opt_level}") + tuple(self.extra_flags)

    def with_rebuild(self, value: bool = True) -> "CodegenOptions":
        return replace(self, rebuild=value)

    @classmethod
    def resolve(cls, codegen: "bool | CodegenOptions | None") -> "CodegenOptions":
        """
        Turn whatever the user passed as ``codegen=`` into a CodegenOptions.

        ``None`` consults ``$BIOPTIM_CODEGEN`` so a whole script can be switched over
        without editing it.
        """
        if isinstance(codegen, CodegenOptions):
            options = codegen
        elif codegen is True:
            options = cls()
        elif codegen is False:
            options = cls(mode="off")
        elif codegen is None:
            enabled = os.environ.get("BIOPTIM_CODEGEN", "").strip().lower() in ("1", "true", "yes", "on")
            options = cls() if enabled else cls(mode="off")
        else:
            raise TypeError(f"codegen must be a bool, a CodegenOptions or None, got {type(codegen).__name__}")

        if os.environ.get("BIOPTIM_CODEGEN_REBUILD", "").strip().lower() in ("1", "true", "yes", "on"):
            options = options.with_rebuild()
        return options
