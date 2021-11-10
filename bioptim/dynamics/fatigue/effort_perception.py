from casadi import if_else, fabs

from .muscle_fatigue import MuscleFatigue
from .tau_fatigue import TauFatigue
from ...misc.enums import VariableType


class EffortPerception(MuscleFatigue):
    """
    A placeholder for fatigue dynamics.
    """

    def __init__(self, effort_threshold: float, effort_factor: float, scaling=10):
        """
        Parameters
        ----------
        effort_threshold: float
            The activation level at which the structure starts to build/relax the effort perception
        effort_factor: float
            Effort perception build up rate
        scaling: float
            The scaling factor to the max value
        """

        super(EffortPerception, self).__init__(scaling=scaling)
        self.effort_threshold = effort_threshold
        self.effort_factor = effort_factor
        self.abs_effort_factor = fabs(self.effort_threshold)
        self.abs_scaling = self.scaling if self.scaling >= 0 else -self.scaling

    @staticmethod
    def suffix(variable_type: VariableType) -> tuple:
        if variable_type == VariableType.STATES:
            return ("mf",)
        else:
            return ("",)

    @staticmethod
    def color() -> tuple:
        return ("tab:brown",)

    def default_state_only(self) -> bool:
        return True

    def default_initial_guess(self) -> tuple:
        return (0,)

    def default_bounds(self, variable_type: VariableType) -> tuple:
        return (0,), (1,)

    def apply_dynamics(self, target_load, *states):
        effort = states[0]
        abs_target_load = fabs(target_load)

        delta_load = (abs_target_load - self.abs_effort_factor) * if_else(
            abs_target_load > self.abs_effort_factor,
            1 / (self.abs_scaling - self.abs_effort_factor),
            1 / self.abs_effort_factor,
        )
        mf_long_dot = self.effort_factor * if_else(delta_load > 0, 1 - effort, effort) * delta_load
        return mf_long_dot

    @staticmethod
    def dynamics_suffix() -> str:
        return ""


class TauEffortPerception(TauFatigue):
    """
    A placeholder for fatigue dynamics.
    """

    def __init__(self, minus: EffortPerception, plus: EffortPerception, **kwargs):
        """
        Parameters
        ----------
        minus: EffortPerception
            The Michaud model for the negative tau
        plus: EffortPerception
            The Michaud model for the positive tau
        """

        super(TauEffortPerception, self).__init__(minus=minus, plus=plus, state_only=True, **kwargs)

    @staticmethod
    def default_state_only():
        return True

    @staticmethod
    def dynamics_suffix() -> str:
        return "effort"