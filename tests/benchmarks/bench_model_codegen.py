"""
Benchmark harness for the model-function code generation work.

This is deliberately *not* a pytest test: it is a measurement tool meant to be run by hand
(and in CI as a nightly job) to produce the before/after numbers that justify — or refute —
compiling the model's CasADi Functions to a shared library.

It reports, for a given model:
  - RSS after importing bioptim, after loading the model, and after building the OCP
  - per-model-function graph size (``n_instructions``) and construction cost
  - the size and sparsity of the NLP constraint Jacobian ``jac_g``
  - wall time of the OCP construction, and of a few numeric evaluations of ``jac_g``

Usage
-----
    python tests/benchmarks/bench_model_codegen.py bioptim/examples/models/pendulum.bioMod
    python tests/benchmarks/bench_model_codegen.py bioptim/examples/models/wholebody_model.bioMod \
        --n-shooting 20 --no-use-sx --no-expand

Beware: the default (``--use-sx``, ``--expand``) configuration on wholebody_model.bioMod is
what the codegen work exists to fix. It needs upwards of 14 GB of RAM and takes minutes.
Start with ``--n-shooting 20`` and watch the machine.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from pathlib import Path

# Run straight from a checkout without installing.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MB.

    ``ru_maxrss`` is a high-water mark, so this never goes down. That is the number that
    matters here: it is what decides whether the machine swaps.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def cap_memory(limit_mb: int) -> None:
    """
    Refuse to allocate past ``limit_mb``, dying with MemoryError instead of the machine.

    This harness exists to measure a configuration that is known to need better than 14 GB,
    so an unbounded run can take the whole desktop down with it. A hard RLIMIT_AS turns
    "the machine froze" into "this configuration needs more than the budget", which is the
    measurement we actually wanted.
    """
    limit_bytes = limit_mb * 1024 * 1024
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        limit_bytes = min(limit_bytes, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, hard))
    print(f"[memory cap: {limit_mb} MB of address space]")


def available_mb() -> float:
    """MemAvailable, i.e. what can be allocated without pushing the machine into swap."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("inf")


class Report:
    """Accumulates measurements and prints them as an aligned table plus optional JSON."""

    def __init__(self, verbose: bool = True):
        self.rows: list[tuple[str, object]] = []
        self.verbose = verbose

    def add(self, label: str, value: object) -> None:
        self.rows.append((label, value))
        if self.verbose:
            print(f"  {label:<44} {value}")

    def section(self, title: str) -> None:
        if self.verbose:
            print(f"\n=== {title} ===")

    def as_dict(self) -> dict:
        return {label: value for label, value in self.rows}


def _fmt(value: float, unit: str, decimals: int = 1) -> str:
    return f"{value:.{decimals}f} {unit}"


def graph_size(fun) -> int | None:
    """Instruction count of a Function, or None if it has no graph (a compiled External)."""
    try:
        return fun.n_instructions()
    except RuntimeError:
        return None


def bench_model_functions(model, report: Report) -> None:
    """Build the model Functions one at a time and report the size of each graph."""
    report.section("model functions")

    # The no-argument accessors that a torque- or marker-driven OCP actually reaches for.
    # Methods taking arguments (rt, marker, ...) are skipped: their argument space is
    # unbounded, so there is no single graph to measure.
    candidates = (
        "forward_dynamics",
        "inverse_dynamics",
        "mass_matrix",
        "non_linear_effects",
        "reshape_qdot",
        "markers",
        "markers_velocities",
        "center_of_mass",
        "center_of_mass_velocity",
    )

    total_instructions = 0
    for name in candidates:
        method = getattr(model, name, None)
        if method is None:
            continue
        before = peak_rss_mb()
        tic = time.perf_counter()
        try:
            fun = method()
        except (NotImplementedError, RuntimeError) as e:
            report.add(f"{name}", f"skipped ({type(e).__name__}: {str(e)[:60]})")
            continue
        elapsed = time.perf_counter() - tic
        # A compiled function has no graph to count -- that is the point of compiling it.
        n_instructions = graph_size(fun)
        total_instructions += n_instructions or 0
        size = f"{n_instructions:>10,} instr" if n_instructions is not None else f"{'compiled':>10}     "
        report.add(
            f"{name}",
            f"{size}  {fun.class_name():<12} "
            f"build {_fmt(elapsed, 's', 3)}  peak +{_fmt(peak_rss_mb() - before, 'MB')}",
        )

    report.add("total model instructions", f"{total_instructions:,}")


def bench_ocp(model_path: str, args, report: Report) -> None:
    """Build a torque-driven OCP on the given model and measure the NLP graph."""
    from casadi import jacobian, vertcat, Function
    import numpy as np

    from bioptim import (
        OptimalControlProgram,
        DynamicsOptions,
        BoundsList,
        InitialGuessList,
        Objective,
        ObjectiveFcn,
        OdeSolver,
        TorqueBiorbdModel,
        CodegenOptions,
    )

    codegen = (
        CodegenOptions(cache_dir=args.cache_dir, opt_level=args.opt_level, verbose=args.verbose)
        if args.codegen
        else None
    )

    report.section("environment")
    import casadi

    report.add("casadi", casadi.__version__)
    report.add("rss after imports", _fmt(peak_rss_mb(), "MB"))

    report.section(f"model: {model_path}")
    tic = time.perf_counter()
    rss_before_model = peak_rss_mb()
    bio_model = TorqueBiorbdModel(model_path, codegen=codegen)
    report.add("model load", _fmt(time.perf_counter() - tic, "s", 2))
    report.add("rss after model load", _fmt(peak_rss_mb(), "MB"))
    report.add("rss delta for model", _fmt(peak_rss_mb() - rss_before_model, "MB"))
    report.add("nb_q / nb_qdot / nb_tau", f"{bio_model.nb_q} / {bio_model.nb_qdot} / {bio_model.nb_tau}")
    report.add("nb_markers / nb_contacts", f"{bio_model.nb_markers} / {bio_model.nb_contacts}")

    bench_model_functions(bio_model, report)

    # Build the OCP on a *fresh* model so the measurements above do not pollute it.
    del bio_model
    gc.collect()

    report.section(
        f"ocp: n_shooting={args.n_shooting} use_sx={args.use_sx} "
        f"expand_dynamics={args.expand} n_threads={args.n_threads}"
    )
    bio_model = TorqueBiorbdModel(model_path, codegen=codegen)

    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["q"][:, 0] = 0
    x_bounds["qdot"][:, 0] = 0

    x_init = InitialGuessList()
    x_init["q"] = [0] * bio_model.nb_q
    x_init["qdot"] = [0] * bio_model.nb_qdot

    n_tau = bio_model.nb_tau
    u_bounds = BoundsList()
    u_bounds["tau"] = [-300] * n_tau, [300] * n_tau
    u_init = InitialGuessList()
    u_init["tau"] = [0] * n_tau

    rss_before_ocp = peak_rss_mb()
    tic = time.perf_counter()
    ocp = OptimalControlProgram(
        bio_model,
        args.n_shooting,
        args.final_time,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4(), expand_dynamics=args.expand),
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau"),
        use_sx=args.use_sx,
        n_threads=args.n_threads,
    )
    build_time = time.perf_counter() - tic
    report.add("ocp build time", _fmt(build_time, "s", 2))
    report.add("rss after ocp build", _fmt(peak_rss_mb(), "MB"))
    report.add("rss delta for ocp", _fmt(peak_rss_mb() - rss_before_ocp, "MB"))

    if args.skip_jacobian:
        return

    # The decisive measurement: the constraint Jacobian is what IPOPT rebuilds and
    # re-evaluates at every iteration, and it is where the graph explosion shows up.
    #
    # ocp.g is empty until solve time -- the penalty vector is only assembled by the solver
    # interface. Go through the same path generic_solve() does, so we measure what IPOPT
    # actually sees rather than a lookalike.
    from bioptim.interfaces.interface_utils import _shake_penalties_tree
    from bioptim.interfaces.ipopt_interface import IpoptInterface

    report.section("constraint jacobian")
    interface = IpoptInterface(ocp)
    v = ocp.variables_vector
    v_bounds = ocp.bounds_vectors
    raw_g, _ = interface.dispatch_bounds()
    g = _shake_penalties_tree(ocp, raw_g, v, v_bounds, args.expand)
    report.add("n_variables / n_constraints", f"{v.shape[0]} / {g.shape[0]}")

    rss_before_jac = peak_rss_mb()
    tic = time.perf_counter()
    jac = jacobian(g, v)
    jac_fun = Function("jac_g", [v], [jac])
    jac_build = time.perf_counter() - tic
    nnz = jac.nnz()
    density = nnz / (jac.shape[0] * jac.shape[1]) if jac.shape[0] * jac.shape[1] else 0.0
    report.add("jac_g build time", _fmt(jac_build, "s", 2))
    jac_instr = graph_size(jac_fun)
    report.add("jac_g n_instructions", f"{jac_instr:,}" if jac_instr is not None else "compiled")
    report.add("jac_g shape / nnz / density", f"{jac.shape} / {nnz:,} / {density:.4f}")
    report.add("rss after jac_g", _fmt(peak_rss_mb(), "MB"))
    report.add("rss delta for jac_g", _fmt(peak_rss_mb() - rss_before_jac, "MB"))

    v0 = np.zeros(v.shape[0])
    tic = time.perf_counter()
    for _ in range(args.n_evals):
        jac_fun(v0)
    eval_time = (time.perf_counter() - tic) / args.n_evals
    report.add(f"jac_g eval (mean of {args.n_evals})", _fmt(eval_time * 1000, "ms", 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_path", help="path to a .bioMod file")
    parser.add_argument("--n-shooting", type=int, default=20)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--n-threads", type=int, default=1)
    parser.add_argument("--n-evals", type=int, default=5, help="numeric evaluations of jac_g to average over")
    parser.add_argument("--use-sx", dest="use_sx", action="store_true", default=True)
    parser.add_argument("--no-use-sx", dest="use_sx", action="store_false")
    parser.add_argument("--expand", dest="expand", action="store_true", default=True)
    parser.add_argument("--no-expand", dest="expand", action="store_false")
    parser.add_argument("--skip-jacobian", action="store_true", help="stop after building the OCP")
    parser.add_argument("--codegen", action="store_true", help="compile the model functions to a shared library")
    parser.add_argument("--cache-dir", default=None, help="where compiled libraries live")
    parser.add_argument("--opt-level", type=int, default=1, help="compiler -O level")
    parser.add_argument("--verbose", action="store_true", help="print codegen cache activity")
    parser.add_argument("--json", type=Path, default=None, help="also write the report as JSON here")
    parser.add_argument(
        "--max-memory-mb",
        type=int,
        default=8192,
        help="hard cap on address space; the run dies with MemoryError rather than taking the machine down",
    )
    args = parser.parse_args()

    # Leave the machine enough headroom to stay usable, whatever the user asked for.
    headroom_mb = 4096
    usable = available_mb() - headroom_mb
    if args.max_memory_mb > usable:
        print(f"[capping at {usable:.0f} MB: only {available_mb():.0f} MB available, keeping {headroom_mb} MB free]")
        args.max_memory_mb = int(usable)
    cap_memory(args.max_memory_mb)

    report = Report()
    try:
        bench_ocp(args.model_path, args, report)
    except (MemoryError, RuntimeError) as e:
        # CasADi surfaces an allocation failure as a RuntimeError wrapping std::bad_alloc
        # rather than as a Python MemoryError, so both have to be treated as "out of budget".
        if not isinstance(e, MemoryError) and "bad_alloc" not in str(e):
            raise
        print(
            f"\n=== ABORTED: this configuration needs more than the {args.max_memory_mb} MB budget ===\n"
            f"    That is the measurement, not a failure of the harness. Re-run with a smaller\n"
            f"    --n-shooting, with --no-use-sx --no-expand, or with --codegen."
        )
        raise SystemExit(2)

    print(f"\n=== peak rss for the whole run: {_fmt(peak_rss_mb(), 'MB')} ===")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.as_dict(), indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
