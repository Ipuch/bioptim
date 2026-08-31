# Compiling model functions (`codegen`)

## What it does

Every model query — `forward_dynamics()`, `markers()`, `mass_matrix()` — returns a
`casadi.Function` holding a symbolic graph. Every place bioptim *uses* one of those
functions inlines the graph again: an RK4 integrator over `ns` shooting nodes replicates the
rigid-body dynamics `4 * ns` times, and the solver then differentiates all of it.

That last step is what breaks. On the 34-dof whole-body model, at only **two** shooting
nodes, the forward graph is modest (constraint vector: 742 k instructions) but
`jacobian(g, v)` over 273 variables needs more than 8.5 GB and fails.

`codegen` compiles each model function to C once, caches the shared library, and loads it
back with `casadi.external()`. The symbolic graph is then never built: each call site
becomes one opaque node, the derivatives are precompiled into the library, and the
arithmetic runs as native code instead of through CasADi's interpreter.

```python
from bioptim import TorqueBiorbdModel

model = TorqueBiorbdModel("whole_body.bioMod", codegen=True)
```

## When it helps — and when it does not

Measured at `n_shooting=20`, RK4, against the current default (`use_sx=True,
expand_dynamics=True`):

| model | dof | `jac_g` eval | peak RSS | `jac_g` graph |
|---|---|---|---|---|
| pendulum | 2 | 0.65 → 2.25 ms (**3.5× slower**) | 162 → 150 MB | 245 478 → 382 |
| triple_pendulum | 3 | 4.85 → 4.27 ms | 296 → 167 MB | 1 494 197 → 622 |
| trunk_and_2arm | 8 | 89.6 → 12.3 ms (**7.3× faster**) | 2067 → 181 MB (**11×**) | 20 123 515 → 1 362 |
| wholebody | 34 | **cannot build** → 564 ms | **>16 GB** → 493 MB | — → 6 312 |

End to end, on the 8-dof model (torque-driven, RK4, IPOPT with exact Hessian) both
configurations converge in the **same number of iterations** to the **same optimum**:

| n_shooting | solve | peak RSS | cost |
|---|---|---|---|
| 20 | 56.07 → 2.49 s (**22×**) | 7346 → 274 MB (**27×**) | 813522.431102232 / ...231 |
| 30 | 72.73 → 2.52 s (**29×**) | 10932 → 342 MB (**32×**) | identical |
| 40 | aborts above 16 GB → 3.65 s | >16 GB → 411 MB | — |

**The crossover is around 3 degrees of freedom.** Below it, codegen is slower: the fixed
cost of an external call exceeds the cost of a small function, and CasADi's inlining plus
common-subexpression elimination wins. Above it, both RAM and time improve, and the gap
widens quickly. This is why `codegen` is opt-in and not a default.

## Warming the cache

The set of functions a problem needs is only known once the problem has asked for them, so a
cold cache cannot help the run that fills it:

- **First run** — behaves exactly as it would without `codegen`, records which functions it
  needed, and compiles them at the end. It warns you that this is what happened.
- **Every run after** — loads the library and never builds a model graph at all.

### On a large model, warm the cache explicitly

That first run is a *baseline-configuration* run, and on a large model the baseline is
exactly what does not fit — an 8-dof model at 30 shooting nodes needs more than 12 GB. So do
not warm a large model by running your problem once. Ask the model for the functions
directly, with `mode="build-only"`, and never build the OCP:

```python
from bioptim import TorqueBiorbdModel, CodegenOptions

model = TorqueBiorbdModel("trunk_and_2arm.bioMod", codegen=CodegenOptions(mode="build-only"))
model.forward_dynamics(with_contact=False)   # the functions your dynamics will use
model.reshape_qdot()
model.codegen_flush()
```

That took 4.9 s for the 8-dof model above, against a cold full run that cannot complete at
all. Note the arguments matter: `forward_dynamics()` and `forward_dynamics(with_contact=False)`
are different cache entries, so warm the exact call your dynamics makes. If you warm the wrong
one, nothing breaks — that function is simply compiled on the next flush instead.

### Where the compiling happens

In a **subprocess**, by default. That is not cosmetic: building the derivatives and generating
the C costs memory that glibc does not return to the OS afterwards (measured on the whole-body
model: RSS stays **+187 MB** in-process, against **+1 MB** via a subprocess).

## What it forces off

A compiled function is fast inside an **MX** graph, where it stays one opaque node. Inside an
SX graph it is still correct, but every call is marshalled through the SX interpreter — about
37× slower. So when a compiled library is in use, bioptim overrides, warning once:

| setting | forced to |
|---|---|
| `use_sx=True` | `False` |
| `expand_dynamics=True` | `False` |
| penalty `expand=True` | `False` |
| shake-tree expansion | off |

The solution is unchanged; only the representation is. Set `codegen=False` to keep your own
settings.

**ACADOS is incompatible** and raises. It generates its own C from an SX graph and cannot
resolve the external's symbols.

## Options

```python
from bioptim import CodegenOptions

model = TorqueBiorbdModel("model.bioMod", codegen=CodegenOptions(
    cache_dir=None,      # None -> $BIOPTIM_CACHE_DIR -> platform user cache dir
    opt_level=1,         # 0 compiles fast and runs ~2-4x slower; 1 is the sweet spot
    extra_flags=(),      # folded into the cache key, so builds never mix
    jobs=None,           # parallel compiler processes; None -> os.cpu_count()
    mode="auto",         # auto | subprocess | inprocess | build-only | load-only
    emit_jacobian=True,  # see "Why jac_ matters" below
    rebuild=False,
    keep_sources=False,
    cache_tag=None,
    verbose=False,
))
```

`BIOPTIM_CODEGEN=1` switches a whole script over without editing it, and
`BIOPTIM_CODEGEN_REBUILD=1` forces a rebuild.

### Cost of the first run

Compilation is dominated by the Jacobian of `forward_dynamics` — on the whole-body model it
is 1.04 M instructions, about 85 % of the total, and takes roughly 280 s at `-O1`. Use
`opt_level=0` to trade runtime speed for a much faster build while iterating.

## Where the library lives

```
<cache root>/<key>/manifest.json
                  /model.<content hash>.so
                  /src/<symbol>.c        (only with keep_sources=True)
```

The cache root is `CodegenOptions.cache_dir`, else `$BIOPTIM_CACHE_DIR/codegen`, else the
platform user cache directory (`~/.cache/bioptim/codegen` on Linux). Never the working
directory.

Clear it with:

```python
from bioptim import clear_codegen_cache
clear_codegen_cache()
```

## When the cache is invalidated

The key is computed on first use — not at construction, because a model is not finished when
`__init__` returns — and covers the model file **contents** (not its path), the model class
and its MRO, contact types, external forces, holonomic configuration, gravity and mass as
actually set, the CasADi/bioptim/backend versions, and the exact compiler and flags. A
renamed copy of a model shares its entry; an edited one does not.

Each entry then verifies, on load, that the library's real signatures match what the manifest
promised, and rebuilds if not.

Concurrent processes (for example `MultiStart`) take an exclusive lock, so exactly one of
them compiles and the rest wait and load.

## Caveats

- `.n_instructions()` raises on a compiled function — there is no graph to count. Code that
  introspects model functions needs to handle that.
- A model function's `.name()` becomes `bioptim_<method>_<hash>` and `.class_name()` becomes
  `"External"`.
- `Solver.IPOPT(c_compile=True)` alongside `codegen` is **untested**. It would compile the
  whole NLP to C, which would then have to resolve the model library's symbols, and that is
  not wired up. It could not be tested here because `c_compile=True` fails in this
  environment on its own, with or without `codegen` (the conda CasADi build does not ship the
  loadable `libcasadi_nlpsol_Ipopt.so` that its `Importer` path needs). Treat the combination
  as unsupported until someone can verify it.
