"""
A very simple yet meaningful optimal control program consisting in a pendulum starting downward and ending upward
while requiring the minimum of generalized forces. The solver is only allowed to move the pendulum sideways.

This simple example is a good place to start investigating bioptim as it describes the most common dynamics out there
(the joint torque driven), it defines an objective function and some boundaries and initial guesses

During the optimization process, the graphs are updated real-time (even though it is a bit too fast and short to really
appreciate it). Finally, once it finished optimizing, it animates the model using the optimal solution
"""

from typing import Callable

from bioptim import (
    OptimalControlProgram,
    DynamicsOptions,
    BoundsList,
    InitialGuessList,
    ObjectiveFcn,
    Objective,
    OdeSolver,
    OdeSolverBase,
    CostType,
    Solver,
    TorqueBiorbdModel,
    ControlType,
    PhaseDynamics,
    OnlineOptim,
    OrderingStrategy,
    TorqueDynamics,
    ParameterList,
    ExternalForceSetTimeSeries,
    ExternalForceSetVariables,
    ContactType,
    FatigueList,
    ObjectiveList,
)

from bioptim.examples.utils import ExampleUtils
import numpy as np
import pin_actuation as pa
from pinocchio import casadi as cpin
from pinocchio_model import PinocchioModel


class TorquePinModel(PinocchioModel, TorqueDynamics):
    def __init__(
        self,
        bio_model: cpin.Model,
        friction_coefficients: np.ndarray = None,
        parameters: ParameterList = None,
        external_force_set: ExternalForceSetTimeSeries | ExternalForceSetVariables = None,
        contact_types: list[ContactType] | tuple[ContactType] = (),
        fatigue: FatigueList = None,
        visual_model=None,
        collision_model=None,
        **kwargs,
    ):
        super().__init__(
            bio_model=bio_model,
            friction_coefficients=friction_coefficients,
            parameters=parameters,
            external_force_set=external_force_set,
            visual_model=visual_model,
            collision_model=collision_model,
            **kwargs,
        )
        self.fatigue = None

    def serialize(self) -> tuple[Callable, dict]:
        return TorquePinModel, dict(
            bio_model=self.path,
            friction_coefficients=self.friction_coefficients,
            external_force_set=self.external_force_set,
            contact_types=self.contact_types,
        )


def prepare_ocp(
    biomodel: "BioModel",
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolverBase = OdeSolver.RK4(n_integration_steps=2),
    use_sx: bool = True,
    n_threads: int = 1,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
    control_type: ControlType = ControlType.CONSTANT,
    ordering_strategy: OrderingStrategy = OrderingStrategy.TIME_MAJOR,
    visual_model=None,
    collision_model=None,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    ode_solver: OdeSolverBase = OdeSolver.RK4()
        Which type of OdeSolver to use
    use_sx: bool
        If the SX variable should be used instead of MX (can be extensive on RAM)
    n_threads: int
        The number of threads to use in the paralleling (1 = no parallel computing)
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. PhaseDynamics.ONE_PER_NODE should also be used when multi-node penalties with more than 3 nodes or with COLLOCATION (cx_intermediate_list) are added to the OCP.
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)
    control_type: ControlType
        The type of the controls
    visual_model: pin.GeometryModel
        Visual geometry model for rendering
    collision_model: pin.GeometryModel
        Collision geometry model for rendering

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = TorquePinModel(biomodel, visual_model=visual_model, collision_model=collision_model)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[0, 1], weight=10
    )  # shoulder rotations<
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[6, 7, 8], weight=10
    )  # shoulder rotations<
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=5)  # Minimize final velocity
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", weight=1000, target=np.array([0 for _ in range(bio_model.nb_q)])
    )  # Minimize final position

    # DynamicsOptions
    dynamics = DynamicsOptions(
        ode_solver=ode_solver,
        expand_dynamics=expand_dynamics,
        phase_dynamics=phase_dynamics,
    )

    # Path bounds
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, 0] = 0  # Start and end at 0...
    # x_bounds["q"][1, -1] = 3.14  # ...but end with pendulum 180 degrees rotated
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"].min[:, :] = -31.4
    x_bounds["qdot"].max[:, :] = 31.4
    # x_bounds["qdot"][:, [0, -1]] = 0  # Start and end without any velocity

    # Initial guess (optional since it is 0, we show how to initialize anyway)
    x_init = InitialGuessList()
    x_init["q"] = [0] * bio_model.nb_q
    x_init["qdot"] = [0] * bio_model.nb_qdot

    # Define control path bounds
    n_tau = bio_model.nb_tau
    u_bounds = BoundsList()
    u_bounds["tau"] = [-100] * n_tau, [100] * n_tau  # Limit the strength of the pendulum to (-100 to 100)...
    # u_bounds["tau"][1, :] = 0  # ...but remove the capability to actively rotate

    # Initial guess (optional since it is 0, we show how to initialize anyway)
    u_init = InitialGuessList()
    u_init["tau"] = [0] * n_tau

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        control_type=control_type,
        use_sx=use_sx,
        n_threads=n_threads,
        ordering_strategy=ordering_strategy,
    )


def main():
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """

    model_dir = "ThoracoscapularShoulderModel/ThoracoscapularShoulderModelSubject_no_root_reducedmarker_set_xyz.osim"
    geom_path = "ThoracoscapularShoulderModel/Geometry"

    # Load model
    model, visual_model, collision_model = pa.build_pinocchio_model_from_opensim(
        str(model_dir), str(geom_path), build_visual=True, build_collision=True
    )
    print(f"   Model created with {model.nq} configuration variables and {model.nv} degrees of freedom.")
    model.gravity.linear = np.array([0.0, -9.81, 0.0])
    cmodel = cpin.Model(model)
    import pinocchio as pin

    pin_model_again = pin.Model(cmodel)

    ocp = prepare_ocp(
        biomodel=cmodel,
        final_time=0.1,
        n_shooting=10,
        n_threads=1,
        visual_model=visual_model,
        collision_model=collision_model,
    )

    # --- Live plots --- #
    ocp.add_plot_penalty(CostType.ALL)  # This will display the objectives and constraints at the current iteration
    # ocp.add_plot_check_conditioning()  # This will display the conditioning of the problem at the current iteration
    # ocp.add_plot_ipopt_outputs()  # This will display the solver's output at the current iteration

    # --- Saving the solver's output during the optimization --- #
    # path_to_results = "temporary_results/"
    # result_file_name = "pendulum"
    # nb_iter_save = 10  # Save the solver's output every 10 iterations
    # ocp.save_intermediary_ipopt_iterations(
    #     path_to_results, result_file_name, nb_iter_save
    # )  # This will save the solver's output at each iteration

    # --- If one is interested in checking the conditioning of the problem, they can uncomment the following line --- #
    # ocp.check_conditioning()

    # --- Print ocp structure --- #
    ocp.print(to_console=False, to_graph=False)

    # --- Solve the ocp --- #
    # Default is OnlineOptim.MULTIPROCESS on Linux, OnlineOptim.MULTIPROCESS_SERVER on Windows and None on MacOS
    # To see the graphs on MacOS, one must run the server manually (see resources/plotting_server.py)
    solver = Solver.IPOPT(online_optim=OnlineOptim.DEFAULT)
    solver._max_iter = 0

    # # Show the constraints Jacobian sparsity
    # # If one wants to see the constraints Jacobian sparsity pattern, they can uncomment the following line
    # ocp.show_constraints_jacobian_sparsity(solver)
    sol = ocp.solve(solver)

    # # --- Warm restart --- #
    # # If one is interested in restarting an optimization from a previous solution with different objective or constraint
    # # functions, they can do as follows:
    #
    # # We first swap the objective (list_index=0) to a minimize state (instead minimize control)
    # ocp.update_objectives(Objective(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", list_index=0))
    # # Then, we resolve the ocp from the last solution (warm_start=sol).
    # # Please note that sending the same "solver" will greatly improve launch time as the internal structure of the
    # # solver is already initialized for the non-changed parts of the problem
    # sol = ocp.solve(solver, warm_start=sol)

    # --- Show the results graph --- #
    sol.print_cost()
    # sol.graphs(show_bounds=True, save_name="results.png")

    # --- Animate the solution --- #
    # viewer = "bioviz"
    viewer = "pyorerun"
    sol.animate(n_frames=0, viewer=viewer, show_now=True)

    # # --- Saving the solver's output after the optimization --- #
    # Here is an example of how we recommend to save the solution. Please note that sol.ocp is not picklable and that sol will be loaded using the current bioptim version, not the version at the time of the generation of the results.
    # import pickle
    # import git
    # from datetime import date
    #
    # # Save the version of bioptim and the date of the optimization for future reference
    # repo = git.Repo(search_parent_directories=True)
    # commit_id = str(repo.commit())
    # branch = str(repo.active_branch)
    # tag = repo.git.describe("--tags")
    # bioptim_version = repo.git.version_info
    # git_date = repo.git.log("-1", "--format=%cd")
    # version_dic = {
    #     "commit_id": commit_id,
    #     "git_date": git_date,
    #     "branch": branch,
    #     "tag": tag,
    #     "bioptim_version": bioptim_version,
    #     "date_of_the_optimization": date.today().strftime("%b-%d-%Y-%H-%M-%S"),
    # }
    #
    # q = sol.decision_states(to_merge=SolutionMerge.NODES)["q"]
    # qdot = sol.decision_states(to_merge=SolutionMerge.NODES)["qdot"]
    # tau = sol.decision_controls(to_merge=SolutionMerge.NODES)["tau"]
    #
    # # Do everything you need with the solution here before we delete ocp
    # integrated_sol = sol.integrate(to_merge=SolutionMerge.NODES)
    # q_integrated = integrated_sol["q"]
    # qdot_integrated = integrated_sol["qdot"]
    #
    # # Save the output of the optimization
    # with open("pendulum_data.pkl", "wb") as file:
    #     data = {"q": q,
    #             "qdot": qdot,
    #             "tau": tau,
    #             "real_time_to_optimize": sol.real_time_to_optimize,
    #             "version": version_dic,
    #             "q_integrated": q_integrated,
    #             "qdot_integrated": qdot_integrated}
    #     pickle.dump(data, file)
    #
    # # Save the solution for future use, you will only need to do sol.ocp = prepare_ocp() to get the same solution object as above.
    # with open("pendulum_sol.pkl", "wb") as file:
    #     del sol.ocp
    #     pickle.dump(sol, file)


if __name__ == "__main__":
    main()
