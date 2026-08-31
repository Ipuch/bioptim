"""
On-disk cache of compiled model CasADi Functions.

The contract, and why it is shaped this way
-------------------------------------------
A model's rigid-body Functions are expensive in two different ways. Building them costs RAM
(the symbolic graph), and *using* them costs far more RAM, because every call site inlines
the graph again -- an RK4 integrator over ``ns`` shooting nodes replicates it
``4 * ns`` times before the solver has even taken a derivative.

Compiling each Function to C and loading it back with ``casadi.external()`` collapses both:
the graph never exists in Python, and each call site becomes one opaque node.

For that to be a win rather than a trap, two things must hold.

1. The shared library must export ``jac_<name>`` and not only the adjoint. ``jac_`` is the
   only channel through which CasADi can learn an external's Jacobian *sparsity*; without
   it, every model call contributes a fully dense block to the NLP Jacobian.
2. The cache key must be exact. Two variants of the same method (``forward_dynamics`` with
   and without contact) have identical signatures, so a key built from the signature alone
   silently loads the wrong function.

This module owns the key and the on-disk layout. ``builder.py`` owns the compilation.

Layout
------
    <root>/<fast_key>/manifest.json
                     /model.<content_hash>.so
                     /src/<symbol>.c          (only when keep_sources)
                     /.lock
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import weakref
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .options import CodegenOptions

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1
LOCK_NAME = ".lock"


# --------------------------------------------------------------------------------------
# cache root
# --------------------------------------------------------------------------------------
def default_cache_root() -> Path:
    """Where compiled libraries live when the user does not say.

    Deliberately not the current working directory: bioptim already drops an ``nlp.c``
    there when ``c_compile=True``, and that is a wart we do not want to repeat.
    """
    env = os.environ.get("BIOPTIM_CACHE_DIR")
    if env:
        return Path(env).expanduser() / "codegen"

    try:
        from platformdirs import user_cache_dir

        return Path(user_cache_dir("bioptim")) / "codegen"
    except ImportError:
        return Path.home() / ".cache" / "bioptim" / "codegen"


def clear_cache(cache_dir: str | Path | None = None) -> None:
    """Remove every compiled library. Running processes keep the libraries they mapped."""
    root = Path(cache_dir) if cache_dir is not None else default_cache_root()
    if root.exists():
        shutil.rmtree(root)


# --------------------------------------------------------------------------------------
# fingerprinting
# --------------------------------------------------------------------------------------
def _hash_file(path: str | Path) -> str:
    """Hash a model file by *content*, not by path.

    Two runs pointing at the same path must not share a cache entry if the file changed,
    and two runs pointing at copies of one file should share it.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_file_digests(model: Any) -> list[str]:
    """Content hashes of every file backing this model.

    ``MultiBiorbdModel.path`` returns a pair of lists rather than a string, so flatten
    whatever shape comes back and hash each readable file.
    """
    paths: list[str] = []

    def _collect(value):
        if value is None:
            return
        if isinstance(value, (str, os.PathLike)):
            paths.append(str(value))
        elif isinstance(value, (list, tuple)):
            for item in value:
                _collect(item)

    _collect(getattr(model, "path", None))

    digests = []
    for path in paths:
        try:
            # Content only, deliberately not the file name: two copies of one model must
            # share a cache entry, and a renamed-but-edited model must not.
            digests.append(_hash_file(path))
        except OSError:
            # A model built in memory rather than from a file. Fall back to the path
            # string; the numeric fingerprint below still guards against a stale hit.
            digests.append(f"unreadable:{path}")
    return sorted(digests)


def _as_numbers(value) -> tuple[float, ...] | None:
    """
    Coerce a backend quantity to plain floats, or None if it is not numeric.

    biorbd returns its scalars as CasADi MX expressions (``Vector3d.to_mx()``), so this has
    to evaluate them; pinocchio returns numpy arrays directly.
    """
    try:
        import numpy as np
        from casadi import DM, MX, SX, evalf

        if hasattr(value, "to_mx"):
            value = value.to_mx()
        if isinstance(value, (MX, SX, DM)):
            value = evalf(value)
        array = np.asarray(value, dtype=float).ravel()
        if array.size == 0 or not np.all(np.isfinite(array)):
            return None
        return tuple(round(float(v), 12) for v in array)
    except Exception:
        # A symbolic value that cannot be evaluated (a model parameterised by an
        # optimisation variable). It is not part of the identity of the compiled code.
        return None


def _backend_scalars(model: Any) -> list[tuple[str, Any]]:
    """Numeric properties of the underlying biorbd/pinocchio model, if they can be read."""
    underlying = getattr(model, "model", None) or getattr(model, "_model", None)
    if underlying is None:
        return []

    found: list[tuple[str, Any]] = []

    # biorbd: getGravity()/mass() return MX-backed objects. pinocchio: gravity.linear is
    # a numpy array and there is no mass(), so the inertias are summed instead.
    for label, getter in (
        ("gravity", lambda: underlying.getGravity()),
        ("gravity", lambda: underlying.gravity.linear),
        ("mass", lambda: underlying.mass()),
        ("mass", lambda: sum(inertia.mass for inertia in underlying.inertias)),
    ):
        try:
            numbers = _as_numbers(getter())
        except Exception:
            continue
        if numbers is not None:
            found.append((label, numbers))

    return found


def _numeric_fingerprint(model: Any) -> list[str]:
    """Cheap scalars describing the *live* model, computed without touching CasADi.

    This exists because ``set_gravity`` and friends mutate the underlying model without
    invalidating ``_cached_functions``. A key built only from the file contents would
    happily serve a library compiled under a different gravity.
    """
    items: list[str] = []

    def _record(label: str, value):
        try:
            items.append(f"{label}={value}")
        except Exception:
            pass

    for attr in ("nb_q", "nb_qdot", "nb_qddot", "nb_tau", "nb_root", "nb_segments", "nb_markers", "nb_muscles"):
        value = getattr(model, attr, None)
        if isinstance(value, int):
            _record(attr, value)

    for attr in ("name_dofs", "marker_names", "muscle_names", "rigid_contact_names", "soft_contact_names"):
        value = getattr(model, attr, None)
        if isinstance(value, (list, tuple)):
            _record(attr, ",".join(str(v) for v in value))

    # Gravity and mass are the two things most likely to be changed after construction --
    # set_gravity mutates the underlying model without invalidating _cached_functions, so
    # a key blind to it would happily serve a library compiled under different gravity.
    for label, value in _backend_scalars(model):
        _record(label, value)

    contact_types = getattr(model, "contact_types", None)
    if contact_types:
        _record("contact_types", ",".join(sorted(str(c) for c in contact_types)))

    friction = getattr(model, "friction_coefficients", None)
    if friction is not None:
        try:
            import numpy as np

            _record("friction", np.asarray(friction).round(12).tobytes().hex())
        except Exception:
            pass

    # Holonomic constraints are attached after __init__ by set_holonomic_configuration, and
    # they change every partitioned function the model generates.
    holonomic = getattr(model, "_holonomic_constraints", None)
    if holonomic is not None:
        _record("holonomic_n", len(holonomic))
        _record("holonomic_dependent", tuple(getattr(model, "_dependent_joint_index", ()) or ()))
        _record("holonomic_independent", tuple(getattr(model, "_independent_joint_index", ()) or ()))
        _record("holonomic_stab", (getattr(model, "stabilization", None), getattr(model, "alpha", None), getattr(model, "beta", None)))
        for i, constraint in enumerate(holonomic):
            # The constraint is a CasADi Function; its name and shape identify it well
            # enough to separate different holonomic configurations.
            try:
                _record(f"holonomic_{i}", f"{constraint.name()}:{constraint.size_in(0)}:{constraint.size_out(0)}")
            except Exception:
                _record(f"holonomic_{i}", repr(constraint)[:80])

    external_force_set = getattr(model, "external_force_set", None)
    if external_force_set is not None:
        _record("efs_type", type(external_force_set).__name__)
        _record("efs_n", getattr(external_force_set, "nb_external_forces", "?"))
        _record("efs_ncomp", getattr(external_force_set, "nb_external_forces_components", "?"))

    # Sub-models of a MultiBiorbdModel each contribute their own fingerprint.
    sub_models = getattr(model, "models", None)
    if isinstance(sub_models, (list, tuple)):
        for i, sub in enumerate(sub_models):
            items.append(f"sub{i}[" + "|".join(_numeric_fingerprint(sub)) + "]")

    return items


def _toolchain_fingerprint(options: CodegenOptions) -> list[str]:
    """Everything about the toolchain that changes the ABI or the generated code."""
    import casadi

    items = [
        f"manifest_version={MANIFEST_VERSION}",
        f"casadi={casadi.__version__}",
        f"python={sys.version_info.major}.{sys.version_info.minor}",
        f"machine={platform.machine()}",
        f"system={platform.system()}",
        f"flags={' '.join(options.compiler_flags)}",
        f"emit_jacobian={options.emit_jacobian}",
    ]

    try:
        from ...misc.__version__ import __version__ as bioptim_version

        items.append(f"bioptim={bioptim_version}")
    except Exception:
        pass

    for module_name in ("biorbd_casadi", "pinocchio"):
        module = sys.modules.get(module_name)
        if module is not None:
            items.append(f"{module_name}={getattr(module, '__version__', 'unknown')}")

    from .compilers import compiler_identity

    items.append(f"compiler={compiler_identity()}")
    return items


def compute_fast_key(model: Any, options: CodegenOptions) -> str:
    """
    The cache directory name.

    Everything folded in here is computable without constructing a single CasADi graph --
    that is the whole point, since a cache hit must never pay the cost we are avoiding.
    """
    parts: list[str] = []
    parts.append(f"class={type(model).__module__}.{type(model).__qualname__}")
    parts.append("mro=" + ",".join(f"{c.__module__}.{c.__qualname__}" for c in type(model).__mro__[:-1]))
    parts.extend(f"file={d}" for d in _model_file_digests(model))
    parts.extend(f"model={item}" for item in _numeric_fingerprint(model))
    parts.extend(f"tool={item}" for item in _toolchain_fingerprint(options))
    if options.cache_tag:
        parts.append(f"tag={options.cache_tag}")

    return hashlib.blake2s("\n".join(parts).encode("utf-8"), digest_size=16).hexdigest()


def symbol_for(cache_key: tuple, fast_key: str) -> str:
    """
    The C symbol name for one ``cache_function`` key.

    ``cache_key`` is the decorator's own key, ``(method_name, args, frozenset(kwargs))``.
    Folding it into the symbol is what stops ``forward_dynamics(with_contact=True)`` and
    ``forward_dynamics(with_contact=False)`` -- which have identical signatures -- from
    colliding, which is the bug that sank the previous attempt at this.
    """
    method_name = cache_key[0]
    payload = repr((cache_key, fast_key)).encode("utf-8")
    digest = hashlib.blake2s(payload, digest_size=6).hexdigest()
    # Keep the symbol a valid C identifier regardless of the method name.
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in str(method_name))
    return f"bioptim_{safe}_{digest}"


# --------------------------------------------------------------------------------------
# locking
# --------------------------------------------------------------------------------------
@contextmanager
def _exclusive_lock(lock_path: Path, timeout: float | None = None) -> Iterator[bool]:
    """
    Hold an exclusive advisory lock on ``lock_path``.

    MultiStart runs N worker processes that each build their own model, so without this
    every worker would spend the full compile time on the same library. Yields True if the
    lock was acquired, False if it was not (only possible when a timeout is given).
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:
        # Windows: fall back to running unlocked. The atomic publish below still keeps the
        # cache consistent; the cost is duplicated work, not corruption.
        yield True
        return

    with open(lock_path, "w") as handle:
        if timeout is None:
            fcntl.flock(handle, fcntl.LOCK_EX)
            acquired = True
        else:
            import time

            deadline = time.monotonic() + timeout
            acquired = False
            while time.monotonic() < deadline:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except BlockingIOError:
                    time.sleep(0.05)
        try:
            yield acquired
        finally:
            if acquired:
                fcntl.flock(handle, fcntl.LOCK_UN)


# --------------------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------------------
def _signature_of(fun) -> dict:
    """The shape of a Function, as stored in the manifest and re-checked on load."""
    return {
        "n_in": fun.n_in(),
        "n_out": fun.n_out(),
        "name_in": [fun.name_in(i) for i in range(fun.n_in())],
        "name_out": [fun.name_out(i) for i in range(fun.n_out())],
        "size_in": [list(fun.size_in(i)) for i in range(fun.n_in())],
        "size_out": [list(fun.size_out(i)) for i in range(fun.n_out())],
    }


class ModelFunctionCache:
    """
    Per-model handle on the compiled-function cache.

    One of these is attached to a model as ``model._codegen``. ``cache_function`` asks it
    for a compiled function before falling back to building the graph, and hands it every
    function it did have to build so they can be compiled at the next flush.
    """

    def __init__(self, model: Any, options: CodegenOptions):
        self._options = options
        self._model_repr = f"{type(model).__name__}({getattr(model, 'path', None)})"
        # Weak, so the model is not kept alive by its own cache.
        self._model_ref = weakref.ref(model)
        self._root = (Path(options.cache_dir).expanduser() if options.cache_dir else default_cache_root()).resolve()
        self._fast_key: str | None = None
        self._dir: Path | None = None
        self._loaded: dict[str, Any] = {}
        self._recorded: dict[str, Any] = {}
        self._recorded_keys: dict[str, tuple] = {}
        self._manifest: dict | None = None
        self._library_path: Path | None = None

    def _resolve(self) -> None:
        """
        Compute the cache key, on first use rather than at construction.

        Deliberately lazy: a model is not finished when ``__init__`` returns. Gravity can be
        changed afterwards, and a HolonomicBiorbdModel does not learn its constraints until
        ``set_holonomic_configuration`` is called -- both change the generated code. Keying
        at construction time would quietly serve a library built for a different model.
        """
        if self._fast_key is not None:
            return
        model = self._model_ref()
        if model is None:
            raise RuntimeError("the model owning this codegen cache has been garbage collected")

        self._fast_key = compute_fast_key(model, self._options)
        self._dir = self._root / self._fast_key

        if self._options.rebuild and self._dir.exists():
            self._log(f"rebuild requested, discarding {self._dir}")
            shutil.rmtree(self._dir, ignore_errors=True)

        self._read_manifest()

    # -- plumbing ----------------------------------------------------------------------
    def reset_key(self) -> None:
        """
        Forget the computed key, so the next use re-fingerprints the model.

        Needed when a model is reconfigured after construction in a way that changes the
        code it generates -- set_holonomic_configuration is the case that matters. Anything
        already recorded is dropped: it belonged to the previous configuration.
        """
        self._fast_key = None
        self._dir = None
        self._manifest = None
        self._library_path = None
        self._loaded.clear()
        self._recorded.clear()
        self._recorded_keys.clear()

    def _log(self, message: str) -> None:
        if self._options.verbose:
            print(f"[bioptim.codegen] {message}")

    @property
    def options(self) -> CodegenOptions:
        return self._options

    @property
    def enabled(self) -> bool:
        return self._options.enabled

    @property
    def fast_key(self) -> str:
        self._resolve()
        return self._fast_key

    @property
    def directory(self) -> Path:
        self._resolve()
        return self._dir

    @property
    def library_path(self) -> Path | None:
        self._resolve()
        return self._library_path

    @property
    def has_pending(self) -> bool:
        return bool(self._recorded)

    @property
    def is_warm(self) -> bool:
        """
        Whether this cache can already serve compiled functions.

        This gates whether the problem is forced into MX. Forcing MX with nothing compiled
        yet is the worst of both worlds: the graph stays small but every evaluation goes
        through CasADi's interpreter as a call node -- measured at 94 s per constraint
        Jacobian on a 34-dof model, against 0.56 s once the same functions are compiled.
        """
        self._resolve()
        return self._manifest is not None and bool(self._manifest.get("functions"))

    def symbol_for(self, cache_key: tuple) -> str:
        self._resolve()
        return symbol_for(cache_key, self._fast_key)

    # -- manifest ----------------------------------------------------------------------
    def _read_manifest(self) -> None:
        manifest_path = self._dir / MANIFEST_NAME
        if not manifest_path.exists():
            self._log(f"cold cache at {self._dir}")
            return
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            warnings.warn(f"bioptim codegen: ignoring unreadable manifest {manifest_path}: {e}")
            return
        if manifest.get("manifest_version") != MANIFEST_VERSION:
            self._log(f"manifest version mismatch at {manifest_path}, treating as cold")
            return

        library = self._dir / manifest.get("library", "")
        if not library.exists():
            warnings.warn(f"bioptim codegen: manifest at {manifest_path} references a missing library {library}")
            return

        self._manifest = manifest
        self._library_path = library
        self._log(f"warm cache: {len(manifest.get('functions', {}))} function(s) in {library.name}")

    def write_manifest(self, library: Path, signatures: dict[str, dict], sources_digest: str) -> None:
        """Publish a manifest describing a freshly built library."""
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "fast_key": self._fast_key,
            "model": self._model_repr,
            "library": library.name,
            "sources_sha256": sources_digest,
            "functions": signatures,
        }
        target = self._dir / MANIFEST_NAME
        tmp = self._dir / f".{MANIFEST_NAME}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(manifest, indent=2))
        os.replace(tmp, target)
        self._manifest = manifest
        self._library_path = library

    # -- the two calls cache_function makes --------------------------------------------
    def load(self, symbol: str):
        """
        Return the compiled Function for ``symbol``, or None if the cache cannot serve it.

        Returning None is the normal cold-cache path, not an error: the caller builds the
        graph itself and records it for the next flush.
        """
        self._resolve()
        if symbol in self._loaded:
            return self._loaded[symbol]
        if self._manifest is None or self._library_path is None:
            return None

        expected = self._manifest.get("functions", {}).get(symbol)
        if expected is None:
            self._log(f"miss: {symbol} is not in the manifest")
            return None

        from casadi import external

        try:
            fun = external(symbol, str(self._library_path))
        except Exception as e:
            warnings.warn(f"bioptim codegen: could not load {symbol} from {self._library_path}: {e}")
            self._invalidate()
            return None

        # Second-level check. The fast key is thorough but it is a hash of things we
        # believed mattered; this compares what the library actually exports against what
        # the manifest promised, which catches whatever the key missed.
        actual = _signature_of(fun)
        if actual != expected["signature"]:
            warnings.warn(
                f"bioptim codegen: signature mismatch for {symbol} in {self._library_path}; rebuilding.\n"
                f"  expected {expected['signature']}\n  got      {actual}"
            )
            self._invalidate()
            return None

        self._loaded[symbol] = fun
        self._log(f"hit: {symbol}")
        return fun

    def record(self, cache_key: tuple, symbol: str, fun) -> None:
        """Queue a freshly built Function for compilation at the next flush."""
        if symbol in self._recorded:
            return
        self._recorded[symbol] = fun
        self._recorded_keys[symbol] = cache_key
        self._log(f"recorded: {symbol} ({cache_key[0]}, {fun.n_instructions():,} instructions)")

    def _invalidate(self) -> None:
        self._manifest = None
        self._library_path = None
        self._loaded.clear()

    def take_recorded(self) -> dict[str, Any]:
        recorded, self._recorded = self._recorded, {}
        self._recorded_keys = {}
        return recorded

    def previously_compiled(self) -> dict[str, dict]:
        """Signatures already in the manifest, so a partial rebuild does not lose them."""
        if self._manifest is None:
            return {}
        return dict(self._manifest.get("functions", {}))

    @contextmanager
    def lock(self, timeout: float | None = None) -> Iterator[bool]:
        with _exclusive_lock(self._dir / LOCK_NAME, timeout=timeout) as acquired:
            yield acquired

    def prune(self) -> None:
        """Keep the cache under ``max_cache_bytes``, dropping least-recently-used entries."""
        if not self._root.exists():
            return
        entries = []
        total = 0
        for child in self._root.iterdir():
            if not child.is_dir():
                continue
            size = sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
            total += size
            entries.append((child.stat().st_mtime, size, child))
        if total <= self._options.max_cache_bytes:
            return
        for _, size, child in sorted(entries):
            if child == self._dir:
                # Never prune the entry this process is using; it is mapped into memory.
                continue
            shutil.rmtree(child, ignore_errors=True)
            total -= size
            self._log(f"pruned {child.name} ({size / 1024**2:.1f} MB)")
            if total <= self._options.max_cache_bytes:
                break
