"""
This script implements a custom dynamics to work with bioptim.
"""
from casadi import MX, SX, exp, vertcat, casadi
import numpy as np
from bioptim.misc.fcn_enum import FcnEnum

from bioptim import (
    OptimalControlProgram,
    NonLinearProgram,
    ConfigureProblem,
    DynamicsEvaluation,
    DynamicsFunctions
)

# Model parameters
tauc = 0.011  # 11ms
a_scale = 250  #
pd0 = 0.000086  # average value
pdt = 0.000138  # average value
km = 0.18
tau1 = 0.044
tau2 = 0.018
stim_amp = 1  # 100e-3
r0 = 5
pwd = 0.0004

class DingModelPulseWidth:

    def __init__(self, custom_time=False):
        self.custom_time = custom_time

    @staticmethod
    def fes_system_dynamics(
            time: MX | SX,
            states: MX | SX,
            controls: MX | SX,
            parameters: MX | SX,
            stochastic_variables: MX | SX,
            nlp: NonLinearProgram,
            nb_phases=None,
            **extra_parameters: list[MX] | list[SX]
    ) -> DynamicsEvaluation:

        # STATE VARIABLES
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        cn = DynamicsFunctions.get(nlp.states["cn"], states)
        f = DynamicsFunctions.get(nlp.states["f"], states)
        # CONTROL VARIABLES
        joints_tau = DynamicsFunctions.get(nlp.controls["tau"], controls)
        # Access the available data from extra_arguments
        phase_index = extra_parameters.get('phase_index')
        stim = extra_parameters.get('stim')
        t_phase = extra_parameters.get('t_phase')
        # retrieve time
        t = time

        # DYNAMICS
        sum_var_list = []
        for m in range(nlp.model.nb_muscles):
            sum_var = 0
            t -= t_phase
            for i in range(1, phase_index):
                if i == 1:
                    ri = 1
                else:
                    ri = 1 + (r0 - 1) * exp(-t_phase / tauc)
                if stim[i] == 1:
                    sum_var += ri * exp(-(t - i * t_phase) / tauc)
                else:
                    first_off_index = next((j for j, value in enumerate(stim) if value == 0), None)
                    sum_var += ri * exp(-(t - (first_off_index - 1) * t_phase) / tauc)
            sum_var_list.append(sum_var)
        cn_dot = (1 / tauc) * casadi.vertcat(*sum_var_list) - (cn / tauc)
        a = stim[phase_index] * stim_amp * a_scale * (1 - exp(-(pwd - pd0) / pdt))
        f_dot = a * (cn / (km + cn)) - (f / (tau1 + tau2 * (cn / (km + cn))))

        muscles_tau = nlp.model.muscle_joint_torque_from_muscle_forces(f, q, qdot)
        ddq = nlp.model.forward_dynamics(q, qdot, joints_tau + muscles_tau)

        sum_var_list.clear()
        return DynamicsEvaluation(dxdt=vertcat(qdot, ddq, cn_dot, f_dot), defects=None)

    def configure_fes_dynamics(self,
                               ocp: OptimalControlProgram,
                               nlp: NonLinearProgram,
                               **extra_parameters):
        # control variables
        ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)

        # state variables
        # nlp.parameters = ocp.parameters
        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_new_variable("cn",  #
                                                nlp.model.muscle_names,
                                                ocp,
                                                nlp,
                                                as_states=True,
                                                as_controls=False)
        ConfigureProblem.configure_new_variable("f",  # force
                                                nlp.model.muscle_names,
                                                ocp,
                                                nlp,
                                                as_states=True,
                                                as_controls=False)

        ConfigureProblem.configure_dynamics_function(ocp,
                                                     nlp,
                                                     DingModelPulseWidth.fes_system_dynamics,
                                                     **extra_parameters)


class FesDynamicsFcn(FcnEnum):
    """
    Selection of valid dynamics functions
    """
    FES_DRIVEN = (DingModelPulseWidth().configure_fes_dynamics,)
