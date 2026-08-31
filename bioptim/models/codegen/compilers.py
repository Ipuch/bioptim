"""
Compiler discovery and invocation for generated CasADi C code.

Nothing here is bioptim-specific: it turns ``.c`` files into one shared library, in
parallel, and reports failures with enough of the compiler's own output to be actionable.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path


class CompilerNotFoundError(RuntimeError):
    """No usable C compiler on PATH."""


class CompilationError(RuntimeError):
    """The compiler ran and refused the input."""


@lru_cache(maxsize=1)
def find_compiler() -> str:
    """The C compiler to use, honouring ``$CC`` first."""
    candidates = [os.environ.get("CC"), "gcc", "clang", "cc"]
    for candidate in candidates:
        if candidate and shutil.which(candidate):
            return candidate
    raise CompilerNotFoundError(
        "bioptim codegen needs a C compiler but found none on PATH. "
        "Install gcc or clang, or set $CC, or pass codegen=CodegenOptions(mode='load-only')."
    )


@lru_cache(maxsize=1)
def compiler_identity() -> str:
    """A short string identifying the toolchain, folded into the cache key.

    A library built by one compiler for one target must never be served to another.
    """
    try:
        compiler = find_compiler()
    except CompilerNotFoundError:
        return "none"

    parts = [Path(compiler).name]
    for flag in ("-dumpversion", "-dumpmachine"):
        try:
            result = subprocess.run([compiler, flag], capture_output=True, text=True, timeout=10)
            parts.append(result.stdout.strip() or "?")
        except (OSError, subprocess.SubprocessError):
            parts.append("?")
    return "-".join(parts)


def library_suffix() -> str:
    system = platform.system()
    if system == "Windows":
        return ".dll"
    if system == "Darwin":
        return ".dylib"
    return ".so"


def _run(cmd: list[str], what: str) -> None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as e:
        raise CompilationError(f"could not run the compiler for {what}: {e}") from e
    if result.returncode != 0:
        # The generated C is machine-written, so a failure here is a bioptim bug, not a
        # user error. Keep enough output to actually diagnose it.
        stderr = result.stderr.strip()
        if len(stderr) > 4000:
            stderr = stderr[:4000] + "\n... (truncated)"
        raise CompilationError(f"compiling {what} failed:\n$ {' '.join(cmd)}\n{stderr}")


def compile_objects(c_files: list[Path], flags: tuple[str, ...], jobs: int, verbose: bool = False) -> list[Path]:
    """
    Compile each ``.c`` to a ``.o``, in parallel.

    Splitting per function rather than emitting one huge translation unit is what keeps the
    wall time tolerable: on a whole-body model the Jacobian of ``forward_dynamics`` is the
    overwhelming majority of the generated code, so compiling the rest alongside it is free.
    """
    objects = [c_file.with_suffix(".o") for c_file in c_files]

    def _compile(pair: tuple[Path, Path]) -> None:
        c_file, obj = pair
        if verbose:
            print(f"[bioptim.codegen] compiling {c_file.name}")
        _run([find_compiler(), "-c", *flags, str(c_file), "-o", str(obj)], c_file.name)

    if not c_files:
        return []

    with ThreadPoolExecutor(max_workers=max(1, min(jobs, len(c_files)))) as pool:
        # list() so exceptions from the workers propagate here rather than being dropped.
        list(pool.map(_compile, zip(c_files, objects)))
    return objects


def link_library(objects: list[Path], output: Path, verbose: bool = False) -> Path:
    """Link the objects into one shared library."""
    if not objects:
        raise CompilationError("nothing to link: no object files were produced")
    if verbose:
        print(f"[bioptim.codegen] linking {output.name} from {len(objects)} object(s)")

    cmd = [find_compiler(), "-shared"]
    if platform.system() == "Darwin":
        cmd += ["-Wl,-undefined,dynamic_lookup"]
    cmd += [str(o) for o in objects] + ["-o", str(output)]
    if platform.system() != "Windows":
        # CasADi's generated code calls into libm.
        cmd.append("-lm")
    _run(cmd, output.name)
    return output
