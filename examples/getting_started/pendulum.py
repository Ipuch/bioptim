"""
A very simple yet meaningful optimal control program consisting in a pendulum starting downward and ending upward
while requiring the minimum of generalized forces. The solver is only allowed to move the pendulum sideways.

This simple example is a good place to start investigating bioptim as it describes the most common dynamics out there
(the joint torque driven), it defines an objective function and some boundaries and initial guesses

During the optimization process, the graphs are updated real-time (even though it is a bit too fast and short to really
appreciate it). Finally, once it finished optimizing, it animates the model using the optimal solution
"""

import biorbd

from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    Bounds,
    QAndQDotBounds,
    InitialGuess,
    ShowResult,
    ObjectiveFcn,
    Objective,
)


def prepare_ocp(biorbd_model_path: str, final_time: float, n_shooting: int) -> OptimalControlProgram:
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

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    # Load a biorbd model
    biorbd_model = biorbd.Model(biorbd_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE_DERIVATIVE)  # Minimize the generalized forces

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    n_tau = biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -100, 100, 0

    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[:, [0, -1]] = 0  # Initial and final position and velocities are 0 for all the degrees of freedom
    x_bounds[1, -1] = 3.14  # Except for the final position of the rotation that is half turn rotated

    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds[0, :] = 0  # Remove the capability of the solver to actively rotate the model

    # Initial guess
    x_init = InitialGuess([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))
    u_init = InitialGuess([tau_init] * n_tau)

    # Prepare the ocp
    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions=objective_functions,
    )


if __name__ == "__main__":
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(biorbd_model_path="pendulum.bioMod", final_time=3, n_shooting=100)

    # --- Solve the ocp --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show the results in a bioviz animation --- #
    result = ShowResult(ocp, sol)
    result.animate()
