"""

"""
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    PenaltyController,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    OdeSolver,
    OdeSolverBase,
    Node,
    Solver,
    CostType,
    MultinodeObjectiveList, PhaseTransitionList, PhaseTransitionFcn, MultinodeConstraintList, MultinodeConstraintFcn,
    NonLinearProgram, DynamicsEvaluation, DynamicsFunctions, ConfigureProblem, Constraint, Axis, InitialGuessList,
    InterpolationType, ParameterList, ParameterObjectiveList,
)
import numpy as np
from custom_package.fes_dynamics import FesDynamicsFcn
from sandbox.custom_package.fes_objectives import FesObjective

# PROBLEM PARAMETERS
# t_stim = 1.0  # stimulation period (total)
t_phase = round(1 / 50, 3)  # period a stimulation pulse
n_nodes = 1
f_min, f_max, f_init = 0, 1000, 0
cn_min, cn_max, cn_init = 0.0, 2.0, 0.0
tau_min, tau_max, tau_init = -0.0, 0.0, 0.0
fes_min, fes_max, fes_init = 0.0002, 0.0008, 0.0002  # todo: verificar scala?
t_stim = 10 * t_phase  # stimulation period (total)
n_phase_off = 5         # todo: this value should be calculated in regard to the cycling range


def prepare_ocp(
        biorbd_model_path: str,
        ode_solver: OdeSolverBase = OdeSolver.RK8(),
        phase_period: float = 0,
        n_phase: int = 0,
        n_shooting: int = 0,
        assume_phase_dynamics: bool = True,
) -> OptimalControlProgram:

    # Problem parameters and BiorbdModel
    n_phase_total = n_phase + n_phase_off
    stim = [1] * n_phase + [0] * n_phase_off

    bio_model = tuple(BiorbdModel(biorbd_model_path) for _ in range(n_phase_total))
    n_shoot = tuple(n_shooting for _ in range(n_phase_total))
    final_time = [phase_period] * n_phase_total

    # Dynamics
    dynamics = DynamicsList()
    for phase_index in range(n_phase_total):
        extra_parameters = {'phase_index': phase_index, 'stim': stim, 't_phase': t_phase}
        dynamics.add(FesDynamicsFcn.FES_DRIVEN,
                     expand=False,
                     **extra_parameters)

    # Constraints
    constraints = ConstraintList()

    # Constraints
    multinode_constraints = MultinodeConstraintList()
    for i in range(n_phase_total):
        multinode_constraints.add(
            MultinodeConstraintFcn.CONTROLS_EQUALITY,
            nodes_phase=(i, i),
            nodes=(Node.START, Node.END),
            key="all",
        )

    # Add objective functions
    objective_functions = ObjectiveList()
    for i in range(n_phase_total):
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=10, phase=i)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=i)
        objective_functions.add(
            FesObjective.custom_func_track_torque,
            custom_type=ObjectiveFcn.Mayer,
            node=Node.ALL_SHOOTING, # all node expect the last one (Node.END) as this is related to the control i think.
            target=np.ones((bio_model[i].nb_tau, n_shoot[i])) * 20,  # automatically handled for every cost functions
            quadratic=True,
            weight=1e10,
            phase=i,
        )
    # Path constraint
    x_bounds = BoundsList()
    for i in range(n_phase_total):
        x_bounds.add("q", bounds=bio_model[i].bounds_from_ranges("q"), phase=i)
        x_bounds.add("qdot", bounds=bio_model[i].bounds_from_ranges("qdot"), phase=i)

        if i == 0:
            x_bounds.add("cn",
                         min_bound=[cn_min] * bio_model[i].nb_muscles,
                         max_bound=[cn_min] * bio_model[i].nb_muscles,
                         phase=i)
            x_bounds.add("f",
                         min_bound=[f_min] * bio_model[i].nb_muscles,
                         max_bound=[f_min] * bio_model[i].nb_muscles,
                         phase=i)
        else:
            x_bounds.add("cn",
                         min_bound=[cn_min] * bio_model[i].nb_muscles,
                         max_bound=[cn_max] * bio_model[i].nb_muscles,
                         phase=i)
            x_bounds.add("f",
                         min_bound=[f_min] * bio_model[i].nb_muscles,
                         max_bound=[f_max] * bio_model[i].nb_muscles,
                         phase=i)

    x_init = InitialGuessList()
    # x_init["q"] = [1.57] * bio_model[0].nb_q
    # x_init["qdot"] = [0] * bio_model[0].nb_q
    x_init.add("cn", [cn_init] * bio_model[0].nb_muscles)
    x_init.add("f", [f_init] * bio_model[0].nb_muscles)

    # Define control path constraint
    u_bounds = BoundsList()
    for i in range(n_phase_total):
        u_bounds.add("tau",
                     min_bound=[tau_min] * bio_model[i].nb_tau,
                     max_bound=[tau_max] * bio_model[i].nb_tau,
                     phase=i)

    u_init = InitialGuessList()
    u_init.add("tau", [tau_init] * bio_model[0].nb_tau, phase=0)

    # Define phase transitions
    phase_transitions = PhaseTransitionList()
    for i in range(n_phase_total - 1):
        phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=i)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shoot,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective_functions,
        constraints=constraints,
        multinode_constraints=multinode_constraints,
        phase_transitions=phase_transitions,
        ode_solver=ode_solver,
        assume_phase_dynamics=assume_phase_dynamics,
        n_threads=10,
    )


def main():
    """
    Defines a multiphase ocp and animate the results
    """
    # --- Prepare the optimal control program --- #
    model_path = "models/arm26_one_muscle.bioMod"
    # model_path = "models/leg6of9musc_clean_RF.bioMod"
    n_phases = int(t_stim / t_phase)  # number of stimulation corresponding to phases

    ocp = prepare_ocp(biorbd_model_path=model_path,
                      phase_period=t_phase,
                      n_shooting=n_nodes,
                      n_phase=n_phases)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))
    sol.graphs()

    # --- Show results --- #
    # sol.print_cost()
    # sol.animate()


if __name__ == "__main__":
    main()
