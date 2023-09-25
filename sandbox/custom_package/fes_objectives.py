import numpy as np
from bioptim.limits import penalty
from casadi import MX
from bioptim import PenaltyController


class FesObjective:

    @staticmethod
    def custom_func_track_torque(controller: PenaltyController) -> MX:
        # controller is a high level interface to access to many feature of the optimal control problem
        # everything is hidden in it,
        # we had to make that kind of interface to support multinode constraints and objective in several projects
        # It is a bit more complicated to understand at first
        # but it is more flexible and more powerful at the end of the day

        muscle_joint_torque = controller.model.muscle_joint_torque_from_muscle_forces(controller.states["f"].mx,
                                                                            controller.states["q"].mx,
                                                                            controller.states["qdot"].mx)

        # Don't put your target here. All objective function essentially support a target value, it is automatically
        # handled later on. So here we just want to return the expression that will be used to compute the value to be
        # tracked.


        # muscle_joint_torque is MX graph but as we need to handle MX and SX,
        # we need automatic conversion that is done by mx_to_cx, that turn it into a casadi function automatically
        # this is now a standard way to send the expression to bioptim,
        return controller.mx_to_cx(
            "muscle_joint_torque",
            muscle_joint_torque,
            controller.states["q"],
            controller.states["f"],
            controller.states["qdot"],
        )

