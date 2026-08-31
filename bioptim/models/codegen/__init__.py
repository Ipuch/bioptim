"""
Compile a model's CasADi Functions to a shared library and load them back with
``casadi.external()``.

Why
---
Every bioptim model query returns a ``casadi.Function`` holding a symbolic graph, and every
call site inlines that graph again: an RK4 integrator over ``ns`` shooting nodes replicates
the rigid-body dynamics ``4 * ns`` times before the solver has taken a single derivative.
On a 34-dof whole-body model that turns a 20-node problem's constraint Jacobian into 150+
million instructions and better than 14 GB of RAM.

Compiling each Function to C and loading it back as an external collapses the graph to one
opaque node per call site, and runs the arithmetic as native code instead of through
CasADi's interpreter.

Use
---
    from bioptim import TorqueBiorbdModel, CodegenOptions

    model = TorqueBiorbdModel("whole_body.bioMod", codegen=True)
    model = TorqueBiorbdModel("whole_body.bioMod", codegen=CodegenOptions(opt_level=0, verbose=True))

The first run builds and caches the library; later runs load it and never construct the
graph at all. ``BIOPTIM_CODEGEN=1`` switches a whole script over without editing it.

This is a large-model feature. On a pendulum it is slightly *slower* than letting CasADi
inline and common-subexpression-eliminate the graph.
"""

from .builder import build
from .cache import ModelFunctionCache, clear_cache, default_cache_root, compute_fast_key, symbol_for
from .hooks import (
    attach_codegen,
    flush_codegen,
    maybe_expand,
    uses_compiled_functions,
    warn_codegen_cold_run,
    warn_incompatible_options,
)
from .options import CodegenOptions

# `clear_cache` is too generic a name to sit in the top-level bioptim namespace.
clear_codegen_cache = clear_cache

__all__ = [
    "CodegenOptions",
    "ModelFunctionCache",
    "attach_codegen",
    "build",
    "clear_cache",
    "clear_codegen_cache",
    "compute_fast_key",
    "default_cache_root",
    "flush_codegen",
    "maybe_expand",
    "uses_compiled_functions",
    "symbol_for",
    "warn_codegen_cold_run",
    "warn_incompatible_options",
]
