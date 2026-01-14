from typing import Callable, Any

import pinocchio as pin
from pinocchio import casadi as cpin
import numpy as np
import casadi
from casadi import SX, MX, vertcat, horzcat, Function, DM, norm_fro

from bioptim.models.protocols.biomodel import BioModel
from bioptim.models.utils import cache_function
from bioptim.limits.path_conditions import Bounds
from bioptim.misc.mapping import BiMapping, BiMappingList


class PinocchioModel(BioModel):
    """
    This class wraps the pinocchio model and allows the user to call the pinocchio functions from the biomodel protocol
    """

    def __init__(
        self,
        bio_model: str | cpin.Model,
        friction_coefficients: np.ndarray = None,
        parameters: Any = None,
        external_force_set: Any = None,
        visual_model: "pin.GeometryModel" = None,
        collision_model: "pin.GeometryModel" = None,
        **kwargs,
    ):
        super().__init__()

        # Create CasADi model for symbolic computations
        self.model = bio_model
        self.cdata = self.model.createData()

        # Store geometry models for visualization
        self.visual_model = visual_model
        self.collision_model = collision_model

        self._friction_coefficients = friction_coefficients
        self.parameters = parameters if parameters is not None else SX.sym("parameters_placeholder", 0, 1)
        self.external_force_set = external_force_set

        self.external_forces = SX.sym(
            "external_forces_mx",
            self.external_force_set.nb_external_forces_components if self.external_force_set else 0,
            1,
        )

        self._symbolic_variables()
        self._cached_functions = {}

    @classmethod
    def from_pinocchio_object(
        cls,
        model: pin.Model,
        friction_coefficients: np.ndarray = None,
        parameters: Any = None,
        external_force_set: Any = None,
        visual_model: "pin.GeometryModel" = None,
        collision_model: "pin.GeometryModel" = None,
        **kwargs,
    ) -> "PinocchioModel":
        """Create a PinocchioModel from an existing pinocchio Model object"""
        cmodel = cpin.Model(model)
        return cls(
            bio_model=cmodel,
            friction_coefficients=friction_coefficients,
            parameters=parameters,
            external_force_set=external_force_set,
            visual_model=visual_model,
            collision_model=collision_model,
            **kwargs,
        )

    def _symbolic_variables(self):
        """Declaration of MX variables of the right shape for the creation of CasADi Functions"""
        # self.q = MX.sym("q_mx", self.nb_q, 1)
        # self.qdot = MX.sym("qdot_mx", self.nb_qdot, 1)
        # self.qddot = MX.sym("qddot_mx", self.nb_qddot, 1)
        # self.tau = MX.sym("tau_mx", self.nb_tau, 1)

        self.q = SX.sym("q_sx", self.nb_q, 1)
        self.qdot = SX.sym("qdot_sx", self.nb_qdot, 1)
        self.qddot = SX.sym("qddot_sx", self.nb_qddot, 1)
        self.tau = SX.sym("tau_sx", self.nb_tau, 1)

    # ==================== Core Properties ====================

    @property
    def name(self) -> str:
        return self.model.name

    @property
    def nb_q(self) -> int:
        return self.model.nq

    @property
    def nb_qdot(self) -> int:
        return self.model.nv

    @property
    def nb_qddot(self) -> int:
        return self.model.nv

    @property
    def nb_tau(self) -> int:
        return self.model.nv

    @property
    def nb_dof(self) -> int:
        return self.model.nv

    @property
    def nb_segments(self) -> int:
        return self.model.nbodies

    @property
    def nb_markers(self) -> int:
        # Count only OP_FRAME type frames as markers
        count = 0
        for i in range(self.model.nframes):
            if self.model.frames[i].type == pin.FrameType.OP_FRAME:
                count += 1
        return count if count > 0 else self.model.nframes

    @property
    def nb_quaternions(self) -> int:
        """Count joints that use quaternion representation"""
        count = 0
        for i in range(self.model.njoints):
            joint = self.model.joints[i]
            # Spherical and FreeFlyer joints use quaternions
            if "Spherical" in joint.shortname() or "FreeFlyer" in joint.shortname():
                count += 1
        return count

    @property
    def nb_root(self) -> int:
        """Pinocchio doesn't have an explicit root concept like biorbd"""
        return 0

    @property
    def nb_ligaments(self) -> int:
        """Ligaments not supported in Pinocchio"""
        return 0

    @property
    def nb_passive_joint_torques(self) -> int:
        """Passive joint torques not supported in Pinocchio"""
        return 0

    @property
    def nb_rigid_contacts(self) -> int:
        return 0

    @property
    def nb_soft_contacts(self) -> int:
        return 0

    @property
    def nb_muscles(self) -> int:
        return 0

    @property
    def name_dofs(self) -> tuple[str, ...]:
        name_dofs = []
        joint_names = tuple(self.model.names)
        for i in range(1, self.model.njoints):
            joint = self.model.joints[i]
            for j in range(joint.nv):
                name_dofs.append(f"{joint_names[i]}_{j}")
        return tuple(name_dofs)

    @property
    def marker_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.model.frames)

    @property
    def segments(self) -> tuple:
        """Return segment/body names"""
        return tuple(self.model.names)

    @property
    def rigid_contact_names(self) -> tuple[str, ...]:
        return ()

    @property
    def contact_types(self):
        return ()

    @property
    def soft_contact_names(self) -> tuple[str, ...]:
        return ()

    @property
    def muscle_names(self) -> tuple[str, ...]:
        return ()

    @property
    def friction_coefficients(self) -> MX | SX | np.ndarray:
        return self._friction_coefficients

    def set_friction_coefficients(self, new_friction_coefficients) -> None:
        if isinstance(new_friction_coefficients, (DM, np.ndarray)) and np.any(new_friction_coefficients < 0):
            raise ValueError("Friction coefficients must be positive")
        self._friction_coefficients = new_friction_coefficients

    # ==================== Index Methods ====================

    def marker_index(self, name) -> int:
        return self.model.getFrameId(name)

    def segment_index(self, name) -> int:
        return self.model.getBodyId(name)

    # ==================== Gravity ====================

    @cache_function
    def gravity(self) -> Function:
        """Get the current gravity applied to the model"""
        gravity_vector = DM(self.model.gravity.linear)
        return Function("gravity", [], [gravity_vector], [], ["gravity"])

    def set_gravity(self, new_gravity) -> None:
        self.model.gravity.linear = np.array(new_gravity).flatten()
        self.cmodel.gravity.linear = np.array(new_gravity).flatten()
        # Clear cache for gravity
        self._cached_functions = {k: v for k, v in self._cached_functions.items() if "gravity" not in k[0]}

    # ==================== Mass ====================

    @cache_function
    def mass(self) -> Function:
        """Get the total mass of the model"""
        total_mass = pin.computeTotalMass(self.model)
        return Function("mass", [], [DM(total_mass)], [], ["mass"])

    # ==================== Dynamics ====================

    @cache_function
    def forward_dynamics(self, with_contact: bool = False) -> Function:
        if with_contact:
            raise NotImplementedError("Forward dynamics with contact not yet implemented for PinocchioModel")

        q = self.q
        qdot = self.qdot
        tau = self.tau

        qddot = cpin.aba(self.model, self.cdata, q, qdot, tau)

        return Function(
            "forward_dynamics",
            [q, qdot, tau, self.external_forces, self.parameters],
            [qddot],
            ["q", "qdot", "tau", "external_forces", "parameters"],
            ["qddot"],
        )

    @cache_function
    def inverse_dynamics(self, with_contact: bool = False) -> Function:
        if with_contact:
            raise NotImplementedError("Inverse dynamics with contact not yet implemented for PinocchioModel")

        q = self.q
        qdot = self.qdot
        qddot = self.qddot

        tau = cpin.rnea(self.model, self.cdata, q, qdot, qddot)

        return Function(
            "inverse_dynamics",
            [q, qdot, qddot],
            [tau],
            ["q", "qdot", "qddot"],
            ["tau"],
        )

    @cache_function
    def mass_matrix(self) -> Function:
        q = self.q
        M = cpin.crba(self.model, self.cdata, q)

        return Function(
            "mass_matrix",
            [q],
            [M],
            ["q"],
            ["mass_matrix"],
        )

    @cache_function
    def non_linear_effects(self) -> Function:
        q = self.q
        qdot = self.qdot
        nle = cpin.nonLinearEffects(self.model, self.cdata, q, qdot)
        return Function(
            "non_linear_effects",
            [q, qdot],
            [nle],
            ["q", "qdot"],
            ["nle"],
        )

    # ==================== Kinematics ====================

    @cache_function
    def marker(self, marker_index: int, reference_frame_idx: int = None) -> Function:
        q = self.q
        cpin.forwardKinematics(self.model, self.cdata, q)
        cpin.updateFramePlacements(self.model, self.cdata)

        pos = self.cdata.oMf[marker_index].translation

        if reference_frame_idx is not None:
            # Transform to reference frame
            ref_oMf = self.cdata.oMf[reference_frame_idx]
            pos_homogeneous = vertcat(pos, 1)
            # ref_oMf.inverse() * pos
            inv_rot = ref_oMf.rotation.T
            inv_trans = -inv_rot @ ref_oMf.translation
            pos = inv_rot @ pos + inv_trans

        return Function(
            f"marker_{marker_index}",
            [q],
            [pos],
            ["q"],
            ["marker_position"],
        )

    @cache_function
    def markers(self) -> Function:
        q = self.q
        cpin.forwardKinematics(self.model, self.cdata, q)
        cpin.updateFramePlacements(self.model, self.cdata)

        # Collect all frame positions
        positions = []
        for i in range(self.model.nframes):
            positions.append(self.cdata.oMf[i].translation)

        markers_matrix = horzcat(*positions) if positions else MX.zeros(3, 0)

        return Function(
            "markers",
            [q],
            [markers_matrix],
            ["q"],
            ["markers"],
        )

    @cache_function
    def markers_velocities(self, reference_index=None) -> Function:
        q = self.q
        qdot = self.qdot

        cpin.forwardKinematics(self.model, self.cdata, q, qdot)
        cpin.updateFramePlacements(self.model, self.cdata)

        velocities = []
        for i in range(self.model.nframes):
            frame = self.model.frames[i]
            # Get velocity in local frame then convert to world
            v_local = cpin.getFrameVelocity(self.model, self.cdata, i, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            velocities.append(v_local.linear)

        vel_matrix = horzcat(*velocities) if velocities else MX.zeros(3, 0)

        return Function(
            "markers_velocities",
            [q, qdot],
            [vel_matrix],
            ["q", "qdot"],
            ["markers_velocities"],
        )

    @cache_function
    def marker_velocity(self, marker_index: int) -> Function:
        q = self.q
        qdot = self.qdot

        cpin.forwardKinematics(self.model, self.cdata, q, qdot)
        cpin.updateFramePlacements(self.model, self.cdata)

        v_local = cpin.getFrameVelocity(self.model, self.cdata, marker_index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        vel = v_local.linear

        return Function(
            "marker_velocity",
            [q, qdot],
            [vel],
            ["q", "qdot"],
            ["marker_velocity"],
        )

    @cache_function
    def markers_accelerations(self, reference_index=None) -> Function:
        q = self.q
        qdot = self.qdot
        qddot = self.qddot

        cpin.forwardKinematics(self.model, self.cdata, q, qdot, qddot)
        cpin.updateFramePlacements(self.model, self.cdata)

        accelerations = []
        for i in range(self.model.nframes):
            a_local = cpin.getFrameClassicalAcceleration(
                self.model, self.cdata, i, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            accelerations.append(a_local.linear)

        acc_matrix = horzcat(*accelerations) if accelerations else MX.zeros(3, 0)

        return Function(
            "markers_accelerations",
            [q, qdot, qddot],
            [acc_matrix],
            ["q", "qdot", "qddot"],
            ["markers_accelerations"],
        )

    @cache_function
    def marker_acceleration(self, marker_index: int) -> Function:
        q = self.q
        qdot = self.qdot
        qddot = self.qddot

        cpin.forwardKinematics(self.model, self.cdata, q, qdot, qddot)
        cpin.updateFramePlacements(self.model, self.cdata)

        a_local = cpin.getFrameClassicalAcceleration(
            self.model, self.cdata, marker_index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        acc = a_local.linear

        return Function(
            "marker_acceleration",
            [q, qdot, qddot],
            [acc],
            ["q", "qdot", "qddot"],
            ["marker_acceleration"],
        )

    # ==================== Center of Mass ====================

    @cache_function
    def center_of_mass(self) -> Function:
        q = self.q
        com = cpin.centerOfMass(self.model, self.cdata, q)
        return Function(
            "center_of_mass",
            [q],
            [com],
            ["q"],
            ["center_of_mass"],
        )

    @cache_function
    def center_of_mass_velocity(self) -> Function:
        q = self.q
        qdot = self.qdot
        cpin.centerOfMass(self.model, self.cdata, q, qdot)
        com_vel = self.cdata.vcom[0]  # Velocity of COM of whole model
        return Function(
            "center_of_mass_velocity",
            [q, qdot],
            [com_vel],
            ["q", "qdot"],
            ["center_of_mass_velocity"],
        )

    @cache_function
    def center_of_mass_acceleration(self) -> Function:
        q = self.q
        qdot = self.qdot
        qddot = self.qddot
        cpin.centerOfMass(self.model, self.cdata, q, qdot, qddot)
        com_acc = self.cdata.acom[0]  # Acceleration of COM of whole model
        return Function(
            "center_of_mass_acceleration",
            [q, qdot, qddot],
            [com_acc],
            ["q", "qdot", "qddot"],
            ["center_of_mass_acceleration"],
        )

    @cache_function
    def angular_momentum(self) -> Function:
        q = self.q
        qdot = self.qdot
        cpin.computeCentroidalMomentum(self.model, self.cdata, q, qdot)
        # hg is spatial momentum [angular; linear]
        ang_mom = self.cdata.hg.angular
        return Function(
            "angular_momentum",
            [q, qdot],
            [ang_mom],
            ["q", "qdot"],
            ["angular_momentum"],
        )

    # ==================== Utility Methods ====================

    @cache_function
    def rotation_matrix_to_euler_angles(self, sequence: str) -> Function:
        """Convert rotation matrix to euler angles using specified sequence"""
        R = MX.sym("R", 3, 3)

        # Map sequence to Pinocchio-style extraction
        # This is a simple implementation; more robust would use proper intrinsic rotations
        if sequence.lower() == "xyz":
            # Extract XYZ Euler angles from rotation matrix
            angles = vertcat(
                casadi.atan2(R[2, 1], R[2, 2]),  # roll
                casadi.atan2(-R[2, 0], casadi.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)),  # pitch
                casadi.atan2(R[1, 0], R[0, 0]),  # yaw
            )
        elif sequence.lower() == "zyx":
            angles = vertcat(
                casadi.atan2(R[1, 0], R[0, 0]),
                casadi.atan2(-R[2, 0], casadi.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)),
                casadi.atan2(R[2, 1], R[2, 2]),
            )
        else:
            # Default to XYZ
            angles = vertcat(
                casadi.atan2(R[2, 1], R[2, 2]),
                casadi.atan2(-R[2, 0], casadi.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)),
                casadi.atan2(R[1, 0], R[0, 0]),
            )

        return Function(
            "rotation_matrix_to_euler_angles",
            [R],
            [angles],
            ["rotation_matrix"],
            ["euler_angles"],
        )

    @cache_function
    def reshape_qdot(self) -> Function:
        """
        Reshape qdot if needed (e.g., for quaternions).
        For Pinocchio, we typically don't need special handling.
        """
        q = self.q
        qdot = self.qdot
        # Identity transform for now
        return Function(
            "reshape_qdot",
            [q, qdot, self.parameters],
            [qdot],
            ["q", "qdot", "parameters"],
            ["qdot_reshaped"],
        )

    @cache_function
    def normalize_state_quaternions(self) -> Function:
        """Normalize quaternions in the state vector"""
        q = self.q
        q_normalized = MX(q)

        # Find joints with quaternion representation and normalize
        for i in range(1, self.model.njoints):  # Skip universe joint
            joint = self.model.joints[i]
            if "Spherical" in joint.shortname() or "FreeFlyer" in joint.shortname():
                idx_q = joint.idx_q
                if "FreeFlyer" in joint.shortname():
                    # FreeFlyer: translation (3) + quaternion (4)
                    quat_start = idx_q + 3
                    quat = q_normalized[quat_start : quat_start + 4]
                    quat_norm = quat / norm_fro(quat)
                    q_normalized[quat_start : quat_start + 4] = quat_norm
                else:
                    # Spherical: quaternion (4)
                    quat = q_normalized[idx_q : idx_q + 4]
                    quat_norm = quat / norm_fro(quat)
                    q_normalized[idx_q : idx_q + 4] = quat_norm

        return Function(
            "normalize_state_quaternions",
            [q],
            [q_normalized],
            ["q"],
            ["q_normalized"],
        )

    @cache_function
    def lagrangian(self) -> Function:
        """Compute the Lagrangian L = T - V (kinetic - potential energy)"""
        q = self.q
        qdot = self.qdot

        # Kinetic energy: 0.5 * qdot.T @ M @ qdot
        M = cpin.crba(self.model, self.cdata, q)
        T = 0.5 * casadi.dot(qdot, M @ qdot)

        # Potential energy
        cpin.computePotentialEnergy(self.model, self.cdata, q)
        V = self.cdata.potential_energy

        L = T - V

        return Function(
            "lagrangian",
            [q, qdot],
            [L],
            ["q", "qdot"],
            ["lagrangian"],
        )

    # ==================== Bounds ====================

    def bounds_from_ranges(self, variables: str | list[str], mapping: BiMapping | BiMappingList = None) -> Bounds:
        """Create bounds from the model's position/velocity limits"""
        # out = Bounds()

        if isinstance(variables, str):
            variables = [variables]

        for var in variables:
            if var in ["q", "q_joints"]:
                min_vals = DM(self.model.lowerPositionLimit).toarray().squeeze()
                max_vals = DM(self.model.upperPositionLimit).toarray().squeeze()
            elif var in ["qdot", "qdot_joints"]:
                min_vals = -DM(self.model.velocityLimit).toarray().squeeze()
                max_vals = DM(self.model.velocityLimit).toarray().squeeze()
            elif var in ["qddot", "qddot_joints"]:
                # No explicit acceleration limits in Pinocchio, use large bounds
                min_vals = [-1000.0] * self.nb_qddot
                max_vals = [1000.0] * self.nb_qddot
            else:
                raise ValueError(f"Unknown variable type: {var}")

        return Bounds(var, min_bound=min_vals, max_bound=max_vals)

    # ==================== Not Implemented / Stubs ====================

    def copy(self):
        return PinocchioModel(self.model)

    def serialize(self) -> tuple[Callable, dict]:
        return PinocchioModel, dict(bio_model=self.model.name)

    @cache_function
    def rt(self, rt_index) -> Function:
        raise NotImplementedError("rt() not implemented for PinocchioModel")

    @cache_function
    def torque(self) -> Function:
        raise NotImplementedError("torque() not implemented for PinocchioModel")

    @cache_function
    def forward_dynamics_free_floating_base(self) -> Function:
        raise NotImplementedError("forward_dynamics_free_floating_base() not implemented for PinocchioModel")

    def reorder_qddot_root_joints(self) -> Function:
        raise NotImplementedError("reorder_qddot_root_joints() not implemented for PinocchioModel")

    @cache_function
    def contact_forces_from_constrained_forward_dynamics(self) -> Function:
        raise NotImplementedError(
            "contact_forces_from_constrained_forward_dynamics() not implemented for PinocchioModel"
        )

    @cache_function
    def qdot_from_impact(self) -> Function:
        raise NotImplementedError("qdot_from_impact() not implemented for PinocchioModel")

    @cache_function
    def muscle_activation_dot(self) -> Function:
        raise NotImplementedError("muscle_activation_dot() not implemented for PinocchioModel")

    @cache_function
    def muscle_joint_torque(self) -> Function:
        raise NotImplementedError("muscle_joint_torque() not implemented for PinocchioModel")

    @cache_function
    def muscle_length_jacobian(self) -> Function:
        raise NotImplementedError("muscle_length_jacobian() not implemented for PinocchioModel")

    @cache_function
    def muscle_velocity(self) -> Function:
        raise NotImplementedError("muscle_velocity() not implemented for PinocchioModel")

    @cache_function
    def tau_max(self) -> Function:
        raise NotImplementedError("tau_max() not implemented for PinocchioModel")

    @cache_function
    def rigid_contact_acceleration(self, contact_index, contact_axis) -> Function:
        raise NotImplementedError("rigid_contact_acceleration() not implemented for PinocchioModel")

    @cache_function
    def soft_contact_forces(self) -> Function:
        raise NotImplementedError("soft_contact_forces() not implemented for PinocchioModel")

    @cache_function
    def rigid_contact_forces(self) -> Function:
        raise NotImplementedError("rigid_contact_forces() not implemented for PinocchioModel")

    @cache_function
    def passive_joint_torque(self) -> Function:
        raise NotImplementedError("passive_joint_torque() not implemented for PinocchioModel")

    @cache_function
    def ligament_joint_torque(self) -> Function:
        raise NotImplementedError("ligament_joint_torque() not implemented for PinocchioModel")

    @cache_function
    def partitioned_forward_dynamics(self, q_u, qdot_u, q_v_init, tau) -> Function:
        raise NotImplementedError("partitioned_forward_dynamics() not implemented for PinocchioModel")

    def to_pyorerun_model(self):
        """Create a pyorerun PinocchioModel for visualization."""
        import pyorerun
        import pinocchio as pin

        # Convert cpin.Model to pin.Model for visualization
        pin_model = pin.Model(self.model)
        return pyorerun.PinocchioModel.from_pinocchio_object(
            pin_model,
            visual_model=self.visual_model,
            collision_model=self.collision_model,
        )

    @property
    def pyorerun_marker_names(self) -> list[str]:
        """Get marker names formatted for pyorerun visualization."""
        return list(self.marker_names)

    @staticmethod
    def animate(
        ocp,
        solution,
        show_now: bool = True,
        show_tracked_markers: bool = False,
        viewer: str = "pyorerun",
        n_frames: int = 0,
        **kwargs,
    ):
        if viewer == "pyorerun":
            from bioptim.models.viewer_pyorerun import animate_with_pyorerun

            return animate_with_pyorerun(ocp, solution, show_now, show_tracked_markers, **kwargs)
