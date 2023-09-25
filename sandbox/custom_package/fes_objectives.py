import numpy as np
from bioptim.limits import penalty
from casadi import MX
from bioptim import PenaltyController


class FesObjective:

    @staticmethod
    def custom_func_track_torque(controller: PenaltyController,
                                 # first_marker: str,
                                 # second_marker: float,
                                 extra_value: list) -> MX:

        joint_tau = controller.model.muscle_joint_torque_from_muscle_forces(controller.states["f"].mx,
                                                                            controller.states["q"].mx,
                                                                            controller.states["qdot"].mx)

        return joint_tau - target_value

