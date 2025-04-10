"""
This is a clone of the pendulum example, but using the PinocchioModel interface.
It shows how to use bioptim with models loaded from URDF via Pinocchio.

Differences from pendulum.py:
- Uses PinocchioModel instead of BiorbdModel.
- Loads the model from a URDF file.
- Assumes a simple single-joint pendulum URDF structure.
- Bounds and initial guesses are adapted for a single DoF.
"""

from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    BoundsList,
    InitialGuessList,
    ObjectiveFcn,
    Objective,
    OdeSolver,
    OdeSolverBase,
    CostType,
    Solver,
    PinocchioModel,  # Import PinocchioModel
    ControlType,
    PhaseDynamics,
)


def prepare_ocp(
    pinocchio_model_path: str,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    use_sx: bool = True,
    n_threads: int = 1,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
    control_type: ControlType = ControlType.CONSTANT,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    pinocchio_model_path: str
        The path to the pinocchio model (URDF)
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
    expand_dynamics: bool
        If the dynamics function should be expanded.
    control_type: ControlType
        The type of the controls

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = PinocchioModel(pinocchio_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Dynamics
    dynamics = Dynamics(
        DynamicsFcn.TORQUE_DRIVEN, ode_solver=ode_solver, expand_dynamics=expand_dynamics, phase_dynamics=phase_dynamics
    )

    # Path bounds
    # Assuming a single DoF pendulum (rotation)
    # Use model limits if defined in URDF, otherwise set manually
    # PinocchioModel.bounds_from_ranges can be used if limits are set in URDF <limit> tag
    try:
        x_bounds = BoundsList()
        # Note: PinocchioModel.bounds_from_ranges assumes 'q' and 'qdot' variable names
        x_bounds["q"] = bio_model.bounds_from_ranges("q") # Gets bounds from URDF joint limits
        x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot") # Gets velocity limits
    except KeyError: # Handle case where URDF might lack limits
        print("Warning: Bounds not fully defined in URDF, using manual bounds.")
        x_bounds = BoundsList()
        limit_range = (-3.15, 3.15) # Example manual range
        velocity_limit = 10        # Example manual limit
        x_bounds["q"] = [limit_range[0]] * bio_model.nb_q, [limit_range[1]] * bio_model.nb_q
        x_bounds["qdot"] = [-velocity_limit] * bio_model.nb_qdot, [velocity_limit] * bio_model.nb_qdot

    # Enforce initial and final position/velocity
    x_bounds["q"][:, [0, -1]] = 0  # Start and end position at 0 rad
    x_bounds["q"][0, -1] = 3.14   # End at pi rad (upward) - Assumes 1st DoF is rotation
    x_bounds["qdot"][:, [0, -1]] = 0  # Start and end without any velocity

    # Initial guess (optional since it is 0, we show how to initialize anyway)
    x_init = InitialGuessList()
    x_init["q"] = [0] * bio_model.nb_q
    x_init["qdot"] = [0] * bio_model.nb_qdot

    # Define control path bounds
    n_tau = bio_model.nb_tau # Should be 1 for single joint pendulum
    # Use effort limits from URDF if available, otherwise manual
    effort_limit = 100 # Example manual limit
    try:
         tau_limits = bio_model.model.effortLimit
         u_bounds = BoundsList()
         u_bounds["tau"] = -tau_limits, tau_limits
    except AttributeError:
         print("Warning: Effort limits not found in model, using manual bounds for control.")
         u_bounds = BoundsList()
         u_bounds["tau"] = [-effort_limit] * n_tau, [effort_limit] * n_tau

    # Initial guess (optional since it is 0, we show how to initialize anyway)
    u_init = InitialGuessList()
    u_init["tau"] = [0] * n_tau

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        use_sx=use_sx,
        n_threads=n_threads,
        control_type=control_type,
    )


def main():
    """
    If pendulum_pinocchio is run as a script, it will perform the optimization and animates it
    """

    # Define the path to your URDF file
    # Make sure this path is correct relative to where you run the script
    urdf_path = "models/pendulum.urdf"

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(pinocchio_model_path=urdf_path, final_time=1.5, n_shooting=30, n_threads=1)

    # --- Add plots --- #
    ocp.add_plot_penalty(CostType.ALL)

    # --- Print ocp structure --- #
    ocp.print(to_console=False, to_graph=False)

    # --- Solve the ocp --- #
    # Note: Online plotting might depend on the chosen viewer backend for PinocchioModel
    sol = ocp.solve(Solver.IPOPT(show_online_optim=True))

    # --- Show the results --- #
    sol.print_cost()
    # sol.graphs(show_bounds=True)

    # --- Animate the solution --- #
    # Note: Animation for PinocchioModel relies on an external viewer setup.
    # The default 'animate' might try bioviz. You may need a specific viewer like meshcat.
    # For now, let's call animate, but it might require viewer setup.
    print("Attempting animation...")
    print("Note: PinocchioModel animation requires a compatible viewer (e.g., meshcat-python or bioviz with conversion).")
    print("Ensure viewer is installed and running if necessary.")
    try:
        sol.animate(n_frames=100) # Request 100 frames for interpolation
    except Exception as e:
        print(f"Animation failed: {e}")


if __name__ == "__main__":
    main() 