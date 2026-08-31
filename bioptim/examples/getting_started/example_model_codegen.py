"""
Compiling a model's CasADi functions to a shared library.

Every model query (forward_dynamics, markers, ...) is a CasADi Function holding a symbolic
graph. Every place that function is used inlines the graph again: an RK4 integrator over
`ns` shooting nodes replicates the rigid-body dynamics `4 * ns` times, and then the solver
differentiates all of it. On a large model that is what makes an OCP run out of RAM.

Passing `codegen=True` compiles each model function to C once, caches the shared library,
and loads it back with `casadi.external()`. From then on the graph is never built: each call
site is a single opaque node, and the arithmetic runs as native code.

Measured on the 34-dof whole-body model at 20 shooting nodes, the constraint Jacobian goes
from 152 million instructions (which needs more than 16 GB and does not finish) to 6,312
instructions in 493 MB.

This is a large-model feature. Below roughly 3 degrees of freedom it is *slower* than letting
CasADi inline and common-subexpression-eliminate the graph, so it is opt-in.

Run this file twice. The first run compiles; the second uses the library.
"""

import time

from bioptim import (
    BoundsList,
    CodegenOptions,
    DynamicsOptions,
    InitialGuessList,
    Objective,
    ObjectiveFcn,
    OdeSolver,
    OptimalControlProgram,
    Solver,
    TorqueBiorbdModel,
)
from bioptim.examples.utils import ExampleUtils


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    codegen: bool | CodegenOptions = True,
) -> OptimalControlProgram:
    """
    Build a torque-driven OCP whose model functions are compiled.

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod file
    final_time: float
        The duration of the movement
    n_shooting: int
        The number of shooting points
    codegen: bool | CodegenOptions
        True for the defaults, or a CodegenOptions to choose the cache directory, the
        optimisation level, and where the compilation happens.

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = TorqueBiorbdModel(biorbd_model_path, codegen=codegen)

    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, [0, -1]] = 0
    x_bounds["q"][1, -1] = 3.14
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0

    x_init = InitialGuessList()
    x_init["q"] = [0] * bio_model.nb_q
    x_init["qdot"] = [0] * bio_model.nb_qdot

    u_bounds = BoundsList()
    u_bounds["tau"] = [-100] * bio_model.nb_tau, [100] * bio_model.nb_tau
    u_bounds["tau"][1, :] = 0

    u_init = InitialGuessList()
    u_init["tau"] = [0] * bio_model.nb_tau

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=DynamicsOptions(ode_solver=OdeSolver.RK4()),
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau"),
    )


def main():
    """
    Build and solve the same problem twice, to show the difference the cached library makes.

    The first run behaves exactly as it would without codegen -- there is nothing compiled
    to use yet -- and compiles what it needed at the end. The second run loads the library
    and never builds a symbolic graph for the model at all.
    """

    model_path = ExampleUtils.folder + "/models/pendulum.bioMod"

    # A CodegenOptions is only needed to override the defaults. `codegen=True` is enough for
    # the common case; the library then lands in the platform user cache directory.
    options = CodegenOptions(verbose=True)

    for run in ("first (compiles)", "second (uses the compiled library)"):
        print(f"\n=== {run} ===")
        tic = time.perf_counter()
        ocp = prepare_ocp(model_path, final_time=1.0, n_shooting=30, codegen=options)
        build_time = time.perf_counter() - tic

        # codegen_active says whether *this* problem was built on compiled functions. On the
        # first run it is False: there was nothing compiled yet, so the problem was built
        # exactly as it would be without codegen. (The model's own functions are swapped for
        # the compiled ones after the build, ready for whatever is asked of it next.)
        print(f"  build            : {build_time:.2f} s")
        print(f"  used compiled    : {ocp.codegen_active}")

        solver = Solver.IPOPT(show_online_optim=False)
        solver.set_print_level(0)
        sol = ocp.solve(solver)
        print(f"  cost             : {float(sol.cost):.8f}")

    print(
        "\nThe compiled library is reused by every later run, and by every other process, "
        "until the model file, the model configuration, casadi, bioptim or the compiler flags change."
    )


if __name__ == "__main__":
    main()
