"""
Turn recorded model Functions into one compiled shared library.

The derivative set, and why it is exactly this
----------------------------------------------
``casadi.external()`` resolves derivatives by *symbol name* in the library. What is present
decides what the solver can do, and the combinations are not interchangeable:

    f only                      -> no derivatives at all
    f, jac_f                    -> gradient and sparse Jacobian, but no exact Hessian
    f, adj1_f, fwd1_adj1_f      -> exact Hessian, but the Jacobian is assumed DENSE
    f, jac_f, adj1_f, fwd1_adj1_f  -> everything, with exact sparsity

The third row is what a previous attempt at this shipped, and it is why that attempt was
slow: ``jac_<name>_sparsity_out`` is the only channel through which CasADi can learn an
external's Jacobian sparsity, so without it every model call contributes a fully dense
block to the NLP Jacobian. This module always emits the fourth row.

Expanding before code generation
--------------------------------
``CodeGenerator.add()`` accepts an MX Function, but the C it produces is pathological --
measured on one whole-body ``forward_dynamics``: 7.09 MB of C that gcc needs 140 s to
compile at ``-O0``, against 0.4 MB and 0.7 s for the same function expanded to SX. So
expansion is mandatory here, and it is cheap (~0.03 s per function, no measurable RAM).

Where the work happens
----------------------
Building the derivatives and the C is itself expensive, and glibc does not return the
arena afterwards (measured: RSS stays 85 MB above baseline after every reference is dropped
and ``malloc_trim`` is called). So by default the whole step runs in a **subprocess** that
exits, and the parent only ever loads the finished library.

The functions reach that subprocess through CasADi's own serialization
(``Function.save`` / ``Function.load``) rather than by rebuilding the model there. That
keeps the child independent of the backend: it needs no biorbd, no pinocchio, no model file,
and no knowledge of which dynamics mixin produced the function.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
import warnings
from pathlib import Path
from typing import Any

from .cache import ModelFunctionCache, _signature_of
from .compilers import CompilationError, compile_objects, library_suffix, link_library
from .options import CodegenOptions

SPEC_NAME = "spec.json"
RESULT_NAME = "result.json"
FUNCTION_SUFFIX = ".casadi"


def _release_memory() -> None:
    """Drop what Python is holding, then ask glibc to return the arena.

    ``malloc_trim`` recovers only part of it -- the allocator fragments badly under CasADi's
    graph construction -- which is precisely why the cold build wants a subprocess.
    """
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _renamed(fun, symbol: str):
    """
    Return an SX copy of ``fun`` named ``symbol``.

    CasADi has no ``Function.rename``, so the function is rebuilt around its own inputs.
    The rename has to happen *before* the derivatives are taken, so that they come out as
    ``jac_<symbol>`` / ``adj1_<symbol>`` and ``external()`` can find them.
    """
    from casadi import Function

    expanded = fun.expand() if fun.class_name() == "MXFunction" else fun
    args = expanded.sx_in()
    return Function(symbol, args, expanded.call(args), expanded.name_in(), expanded.name_out())


def generate_sources(symbols: list[str], load, build_dir: Path, options: CodegenOptions) -> tuple[list[Path], dict]:
    """
    Code-generate one ``.c`` per function, plus its derivatives.

    ``load(symbol)`` returns the Function to compile, so this works the same whether the
    caller holds the functions in memory or reads them back from disk in a subprocess.

    One CodeGenerator per function rather than one for all of them: it caps peak RAM at the
    largest single function instead of the sum, and it is what lets the compiler run in
    parallel afterwards.
    """
    from casadi import CodeGenerator

    build_dir.mkdir(parents=True, exist_ok=True)
    c_files: list[Path] = []
    signatures: dict[str, dict] = {}

    for symbol in sorted(symbols):
        fun = load(symbol)
        if options.verbose:
            print(f"[bioptim.codegen] generating {symbol} ({fun.n_instructions():,} instructions)")

        renamed = _renamed(fun, symbol)
        signatures[symbol] = {"signature": _signature_of(renamed), "method": symbol}

        generator = CodeGenerator(f"{symbol}.c", {"with_header": False})
        generator.add(renamed)
        if options.emit_jacobian:
            generator.add(renamed.jacobian())
        adjoint = renamed.reverse(1)
        generator.add(adjoint)
        generator.add(adjoint.forward(1))
        generator.generate(str(build_dir) + os.sep)

        c_files.append(build_dir / f"{symbol}.c")

        # Drop every reference to this function's graph before starting the next one.
        del renamed, adjoint, generator, fun
        _release_memory()

    return c_files, signatures


def _digest_sources(c_files: list[Path]) -> str:
    digest = hashlib.sha256()
    for c_file in sorted(c_files):
        digest.update(c_file.name.encode())
        digest.update(c_file.read_bytes())
    return digest.hexdigest()


def _compile_and_link(c_files: list[Path], staging: Path, options: CodegenOptions) -> tuple[Path, str]:
    """Compile the generated C into one library inside ``staging``. Returns (path, digest)."""
    sources_digest = _digest_sources(c_files)
    # The library is named after the content that produced it, so publishing never has to
    # overwrite a file another process may already have mapped into memory.
    library_name = f"model.{sources_digest[:16]}{library_suffix()}"

    objects = compile_objects(c_files, options.compiler_flags, options.n_jobs, options.verbose)
    library = link_library(objects, staging / library_name, options.verbose)
    return library, sources_digest


# --------------------------------------------------------------------------------------
# in-process
# --------------------------------------------------------------------------------------
def _build_inprocess(cache: ModelFunctionCache, recorded: dict[str, Any], staging: Path) -> tuple[Path, dict, str]:
    options = cache.options

    def load(symbol):
        return recorded[symbol]

    c_files, signatures = generate_sources(list(recorded), load, staging, options)
    library, digest = _compile_and_link(c_files, staging, options)
    if options.keep_sources:
        _keep_sources(cache, c_files)
    return library, signatures, digest


# --------------------------------------------------------------------------------------
# subprocess
# --------------------------------------------------------------------------------------
def _child_environment() -> dict[str, str]:
    """
    The environment for the build subprocess, with this interpreter's import path carried over.

    A child inherits environment variables but not ``sys.path``, so a bioptim that was found
    through a path manipulation -- running from a checkout, an editable install, a test
    harness -- would simply not be importable in the child. Propagating sys.path through
    PYTHONPATH makes the subprocess work wherever the parent does.
    """
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    entries = [p for p in sys.path if p] + ([existing] if existing else [])
    # dict.fromkeys keeps first-seen order while removing duplicates.
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(entries))
    return env


def _build_in_subprocess(cache: ModelFunctionCache, recorded: dict[str, Any], staging: Path) -> tuple[Path, dict, str]:
    """
    Hand the functions to a child process, which generates, compiles and exits.

    The parent writes each Function out with CasADi's serialization and immediately drops
    its reference, so it never holds the derivatives or the generated C -- and, more to the
    point, never grows an allocator arena it cannot give back.
    """
    options = cache.options
    staging.mkdir(parents=True, exist_ok=True)

    for symbol in list(recorded):
        recorded[symbol].save(str(staging / f"{symbol}{FUNCTION_SUFFIX}"))
        recorded[symbol] = None
    recorded.clear()
    _release_memory()

    spec = {
        "staging": str(staging),
        "symbols": sorted(p.stem for p in staging.glob(f"*{FUNCTION_SUFFIX}")),
        "options": {
            "opt_level": options.opt_level,
            "extra_flags": list(options.extra_flags),
            "jobs": options.n_jobs,
            "emit_jacobian": options.emit_jacobian,
            "verbose": options.verbose,
        },
    }
    spec_path = staging / SPEC_NAME
    spec_path.write_text(json.dumps(spec))

    if options.verbose:
        print(f"[bioptim.codegen] building {len(spec['symbols'])} function(s) in a subprocess")

    completed = subprocess.run(
        [sys.executable, "-m", "bioptim.models.codegen", str(spec_path)],
        capture_output=not options.verbose,
        text=True,
        env=_child_environment(),
    )
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()[-2000:] if completed.stderr else "(no output captured)"
        raise CompilationError(f"the codegen subprocess failed (exit {completed.returncode}):\n{detail}")

    result = json.loads((staging / RESULT_NAME).read_text())
    library = Path(result["library"])
    if options.keep_sources:
        _keep_sources(cache, [Path(p) for p in result["c_files"]])
    return library, result["signatures"], result["sources_digest"]


def _keep_sources(cache: ModelFunctionCache, c_files: list[Path]) -> None:
    sources_dir = cache.directory / "src"
    sources_dir.mkdir(exist_ok=True)
    for c_file in c_files:
        if c_file.exists():
            shutil.copy2(c_file, sources_dir / c_file.name)


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------
def build(cache: ModelFunctionCache, recorded: dict[str, Any]) -> Path | None:
    """
    Generate, compile, link and publish a library for ``recorded``.

    Returns the published library path, or None if there was nothing to do. Failures are
    warned about rather than raised: a model that cannot be compiled must still run.
    """
    options = cache.options
    if not recorded:
        return cache.library_path
    if not options.may_compile:
        if options.verbose:
            print(
                f"[bioptim.codegen] mode={options.mode!r} may not compile; "
                f"{len(recorded)} function(s) left uncompiled"
            )
        return cache.library_path

    cache.directory.mkdir(parents=True, exist_ok=True)
    staging = cache.directory / f".build-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    in_subprocess = options.mode in ("auto", "subprocess", "build-only")

    try:
        try:
            if in_subprocess:
                library, signatures, digest = _build_in_subprocess(cache, recorded, staging)
            else:
                library, signatures, digest = _build_inprocess(cache, recorded, staging)
        except (CompilationError, OSError) as e:
            if options.mode != "auto" or not in_subprocess or not recorded:
                raise
            # "auto" promises a library, not a particular way of producing one.
            warnings.warn(f"bioptim codegen: the build subprocess failed, retrying in-process.\n{e}")
            library, signatures, digest = _build_inprocess(cache, recorded, staging)

        published = cache.directory / library.name
        if not published.exists():
            os.replace(library, published)

        # Carry forward anything an earlier build produced, so a partial rebuild does not
        # drop functions that are still perfectly good.
        merged = cache.previously_compiled()
        merged.update(signatures)
        cache.write_manifest(published, merged, digest)

        cache.prune()
        return published

    except (CompilationError, RuntimeError, OSError) as e:
        warnings.warn(
            f"bioptim codegen: could not build a compiled library for {cache.fast_key}; "
            f"falling back to uncompiled CasADi Functions.\n{e}"
        )
        return None
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        _release_memory()


def _main(spec_path: str) -> int:
    """Child-process entry point: read the spec, generate, compile, write the result."""
    from casadi import Function

    spec = json.loads(Path(spec_path).read_text())
    staging = Path(spec["staging"])
    options = CodegenOptions(**spec["options"])

    def load(symbol):
        return Function.load(str(staging / f"{symbol}{FUNCTION_SUFFIX}"))

    c_files, signatures = generate_sources(spec["symbols"], load, staging, options)
    library, digest = _compile_and_link(c_files, staging, options)

    (staging / RESULT_NAME).write_text(
        json.dumps(
            {
                "library": str(library),
                "signatures": signatures,
                "sources_digest": digest,
                "c_files": [str(c) for c in c_files],
            }
        )
    )
    return 0

