"""
The two calls a model backend makes: attach a cache at construction, flush it when the
set of functions the problem actually needs is known.
"""

from __future__ import annotations

import warnings
from typing import Any

from .builder import build
from .cache import ModelFunctionCache
from .options import CodegenOptions


def attach_codegen(model: Any, codegen: "bool | CodegenOptions | None") -> None:
    """
    Give ``model`` a ``_codegen`` handle, or ``None`` if code generation is off.

    Must be called after the model's geometry is loaded and after any constructor-time
    mutation of it, because the cache key fingerprints the live model.
    """
    options = CodegenOptions.resolve(codegen)
    model._codegen = ModelFunctionCache(model, options) if options.enabled else None


def flush_codegen(model: Any) -> None:
    """
    Compile whatever functions this model had to build, and swap them for the compiled ones.

    Safe to call repeatedly and on a model with no cache: it is a no-op unless something was
    recorded. The functions already handed out to a partially built OCP are *not* retracted
    -- on a cold run the problem keeps the uncompiled graphs it was built with, and the
    compiled library benefits the next run. That is the deal record-then-batch makes: we
    cannot know which functions a problem needs until it has asked for them.
    """
    cache: ModelFunctionCache | None = getattr(model, "_codegen", None)
    if cache is None or not cache.has_pending:
        return

    # Another process may be building the very same library (MultiStart runs N workers that
    # each construct their own model). Whoever gets the lock builds; the others find the
    # manifest waiting for them.
    with cache.lock():
        cache._read_manifest()
        recorded = cache.take_recorded()
        still_missing = {symbol: fun for symbol, fun in recorded.items() if cache.load(symbol) is None}
        if not still_missing:
            _adopt_compiled(model, cache)
            return
        build(cache, still_missing)

    _adopt_compiled(model, cache)


def _adopt_compiled(model: Any, cache: ModelFunctionCache) -> None:
    """
    Replace the model's cached graphs with the compiled externals where one now exists.

    This matters beyond tidiness: anything that asks the model for a function *after* the
    flush -- penalties added later, a second OCP built on the same model -- then gets the
    compiled version instead of rebuilding the graph.
    """
    cached = getattr(model, "_cached_functions", None)
    if not cached:
        return
    for key in list(cached):
        compiled = cache.load(cache.symbol_for(key))
        if compiled is not None:
            cached[key] = compiled


def uses_compiled_functions(*sources: Any) -> bool:
    """
    Whether any of ``sources`` is working with compiled model functions.

    Accepts whatever is in scope at the call site -- an OptimalControlProgram, an NLP, a
    PenaltyController or a model -- because the ``.expand()`` calls this guards are spread
    across the dynamics, penalty and constraint layers, which each hold a different handle.
    """
    for source in sources:
        if source is None:
            continue
        if getattr(source, "codegen_active", False):
            return True
        for owner in (getattr(source, "ocp", None), getattr(source, "get_nlp", None)):
            if owner is not None and getattr(owner, "codegen_active", False):
                return True
        model = getattr(source, "model", None)
        cache = getattr(model, "_codegen", None) if model is not None else getattr(source, "_codegen", None)
        if cache is not None and cache.is_warm:
            return True
    return False


def maybe_expand(fun, *sources: Any):
    """
    ``fun.expand()``, unless compiled model functions are in play.

    Expanding a graph that contains a compiled external does not fail -- CasADi 3.7 allows
    call nodes inside SX -- it just routes every call through the SX interpreter instead of
    leaving it an opaque node, which measures around 37x slower. So these expansions, which
    are unconditional today, have to become conditional.
    """
    if uses_compiled_functions(*sources):
        return fun
    return fun.expand()


def warn_incompatible_options(ocp_use_sx: bool, expand_dynamics: bool) -> None:
    """
    Tell the user, once, which of their settings code generation is overriding.

    Compiled externals are correct inside an SX graph but roughly 37x slower there than in
    an MX one, because every call is marshalled through the SX interpreter rather than
    being an opaque node. So ``use_sx`` and ``expand_dynamics`` are forced off; the results
    are identical either way, only the speed differs.
    """
    overridden = []
    if ocp_use_sx:
        overridden.append("use_sx=True -> False")
    if expand_dynamics:
        overridden.append("expand_dynamics=True -> False")
    if not overridden:
        return
    warnings.warn(
        "bioptim codegen: compiled model functions are much faster in an MX graph than in an "
        "expanded SX one, so the following were overridden: " + ", ".join(overridden) + ". "
        "The solution is unchanged; set codegen=False to keep your original settings.",
        stacklevel=3,
    )


def warn_codegen_cold_run() -> None:
    """
    Tell the user this run pays for the compilation and the next one collects on it.

    Worth saying out loud, because the cold run looks like nothing happened: it runs exactly
    as it would without codegen, and only the library left behind is different.
    """
    warnings.warn(
        "bioptim codegen: no compiled library for this model yet, so this run behaves exactly as it would "
        "without codegen and compiles the functions it needed at the end. Run the same script again to use "
        "the compiled library.",
        stacklevel=3,
    )
