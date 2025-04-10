from typing import Callable, Any

import casadi as ca
import numpy as np
import pinocchio as pin

from ..protocols.biomodel import BioModel
from ..utils import cache_function
from ...limits.path_conditions import Bounds
from ...misc.mapping import BiMapping, BiMappingList


class PinocchioModel(BioModel):
    """
    Implementation of the BioModel protocol for models loaded via Pinocchio.

    Attributes
    ----------
    model: pin.Model
        The pinocchio model itself
    data: pin.Data
        The pinocchio data structure associated with the model
    path: str
        Path to the model file (e.g., URDF)
    _gravity: ca.MX
        The gravity vector
    """

    def __init__(self, path: str):
        """
        Load a model from a URDF file using Pinocchio.

        Parameters
        ----------
        path: str
            The path to the URDF file.
        """
        try:
            self.model = pin.buildModelFromUrdf(path)
        except Exception as e:
            raise ValueError(f"Could not load model from URDF: {path}") from e

        self.data = self.model.createData()
        self.path = path
        self._gravity = ca.MX(self.model.gravity.linear)  # Store gravity vector

        # Basic check for floating base (Pinocchio convention adds root_joint)
        # This influences q vs nv dimensions
        self._has_floating_base = self.model.joints[1].shortname() == "JointModelFreeFlyer"

        # Prepare symbolic variables for CasADi functions
        self._q_sym = ca.MX.sym("q", self.nb_q, 1)
        self._qdot_sym = ca.MX.sym("qdot", self.nb_qdot, 1)  # Corresponds to nv
        self._qddot_sym = ca.MX.sym("qddot", self.nb_qdot, 1)  # Corresponds to nv
        self._tau_sym = ca.MX.sym("tau", self.nb_tau, 1)  # Corresponds to nv

        # # Pinocchio CasADi interface (requires pinocchio built with CasADi support)
        # if not hasattr(pin, "casadi"):
        #     raise ImportError("Pinocchio was not built with CasADi support. Please rebuild Pinocchio with CasADi.")

        self.pinocchio_casadi = pin.casadi

    @property
    def name(self) -> str:
        """Get the name of the model"""
        return self.model.name

    def copy(self):
        """copy the model by reloading one"""
        return PinocchioModel(self.path)

    def serialize(self) -> tuple[Callable, dict]:
        """transform the class into a save and load format"""
        return PinocchioModel, dict(path=self.path)

    @property
    def friction_coefficients(self) -> ca.Function:
        """Get the coefficient of friction to apply to specified elements in the dynamics"""
        # Pinocchio stores friction in model.friction (vector of size nv)
        # Wrap it in a CasADi function that returns this constant vector
        friction_val = ca.MX(self.model.friction)
        return ca.Function("friction_coefficients", [], [friction_val], [], ["friction"])()  # Call to return MX

    @cache_function
    def gravity(self) -> ca.Function:
        """Get the current gravity applied to the model as a Function returning MX"""
        return ca.Function("gravity", [], [self._gravity], [], ["gravity_vec"])

    def set_gravity(self, new_gravity: np.ndarray | ca.MX | ca.SX):
        """Set the gravity vector"""
        if isinstance(new_gravity, np.ndarray):
            new_gravity = ca.MX(new_gravity)
        if not isinstance(new_gravity, (ca.MX, ca.SX)) or new_gravity.shape != (3, 1):
            raise ValueError("new_gravity must be a 3x1 CasADi MX or SX symbol or numpy array")
        self.model.gravity.linear = new_gravity  # Update Pinocchio model's gravity
        self._gravity = ca.MX(new_gravity)  # Update stored CasADi gravity
        # Invalidate cache for functions that depend on gravity (e.g., inverse_dynamics, nle)
        # Pinocchio CasADi functions might implicitly capture the model state at creation.
        # Re-creating them or ensuring they use symbolic gravity might be needed.
        # For now, assume functions created *after* set_gravity use the new value.

    @property
    def nb_tau(self) -> int:
        """Get the number of generalized forces (actuated joints, nv)"""
        return self.model.nv

    @property
    def nb_segments(self) -> int:
        """Get the number of bodies (including universe)"""
        # Pinocchio uses 'bodies' or 'frames'. model.nbodies includes universe.
        # model.njoints roughly corresponds to segments in biorbd.
        return self.model.njoints

    def segment_index(self, segment_name) -> int:
        """Get the segment index from its name (using Pinocchio's joint names)"""
        if self.model.existJointName(segment_name):
            return self.model.getJointId(segment_name)
        else:
            # Fallback to frame names if joints don't match
            if self.model.existFrameName(segment_name):
                frame_id = self.model.getFrameId(segment_name)
                # Return the joint associated with this frame's parent
                return self.model.frames[frame_id].parent
            raise ValueError(f"Segment/Joint/Frame '{segment_name}' not found in Pinocchio model.")

    @property
    def nb_quaternions(self) -> int:
        """Get the number of quaternion base joints"""
        # A bit simplistic, assumes only the root can be a free flyer quaternion
        return 1 if self._has_floating_base else 0

    @property
    def nb_dof(self) -> int:
        """Get the number of DoF (nv)"""
        return self.model.nv

    @property
    def nb_q(self) -> int:
        """Get the number of Generalized coordinates (nq)"""
        return self.model.nq

    @property
    def nb_qdot(self) -> int:
        """Get the number of Generalized velocities (nv)"""
        return self.model.nv

    @property
    def nb_qddot(self) -> int:
        """Get the number of Generalized accelerations (nv)"""
        return self.model.nv

    @property
    def nb_root(self) -> int:
        """Get the number of root DoF"""
        # In Pinocchio, root joint has index 1 if it exists
        if self.model.njoints > 1:
            # Free flyer has 6 DoF (nv=6), Planar has 3 DoF (nv=3)
            # Check joint type of joint 1
            joint1_type = self.model.joints[1].shortname()
            if "FreeFlyer" in joint1_type:
                return 6
            if "Planar" in joint1_type:
                return 3
            # Fixed base models might have a different root convention, or start joints at 1
            # Assume 0 if not explicitly floating/planar root
            return 0  # Or check if model.joints[0] has DoFs?
        return 0

    @property
    def segments(self) -> tuple:
        """Get all segments (using Pinocchio's joints)"""
        # Not a direct equivalent. Returning joint names.
        return tuple(self.model.names[1:])  # Skip 'universe'

    @cache_function
    def rotation_matrix_to_euler_angles(self, rot_mat_sym: ca.MX | ca.SX, sequence: str) -> ca.Function:
        """
        Get the Euler angles from a rotation matrix, in the sequence specified
        args: rotation matrix (symbolic 3x3)
        """
        # Pinocchio has pin.rpy.matrixToRpy, but need a CasADi version.
        # Implement Euler angle extraction symbolically using CasADi.
        # Example for ZYX sequence (roll, pitch, yaw)
        R = rot_mat_sym

        if sequence.upper() == "ZYX":
            sy = ca.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            singular = sy < 1e-6

            x = ca.arctan2(R[2, 1], R[2, 2])
            y = ca.arctan2(-R[2, 0], sy)
            z = ca.arctan2(R[1, 0], R[0, 0])

            # Handle singularity (pitch = +/- 90 degrees)
            # Not perfectly handled here, might need CasADi conditionals if required
            # if singular:
            #    x = atan2(-R[1,2], R[1,1])
            #    y = atan2(-R[2,0], sy)
            #    z = 0

            euler_angles = ca.vertcat(x, y, z)  # Roll, Pitch, Yaw
        else:
            raise NotImplementedError(f"Euler sequence '{sequence}' not implemented symbolically.")

        return ca.Function("rot_mat_to_euler", [rot_mat_sym], [euler_angles], ["rot_mat"], ["euler"])

    @cache_function
    def mass(self) -> ca.Function:
        """Get the mass of the model"""
        # Compute total mass once
        total_mass = sum(inertia.mass for inertia in self.model.inertias[1:])  # Skip universe
        return ca.Function("mass", [], [ca.MX(total_mass)], [], ["mass"])

    @cache_function
    def rt(self, rt_index: int) -> ca.Function:
        """
        Get the rototrans matrix of a frame (treat frame as RT target) that is placed on the model
        args: q
        """
        if rt_index < 0 or rt_index >= self.model.nframes:
            raise ValueError(f"rt_index (frame index) {rt_index} out of bounds.")

        # Use Pinocchio's CasADi interface for forward kinematics
        frame_pose_func = self.pinocchio_casadi.computeFrameJacobian(self.model, self._q_sym, rt_index)

        # This computes Jacobian, not pose. Need Frame Placement.
        # Let's try pin.updateFramePlacement + extract

        # Need a CasADi function for frame placement
        # Option 1: If pinocchio.casadi.updateFramePlacement exists (it might not directly)
        # Option 2: Reimplement frame placement symbolically or use a different approach.

        # Fallback: Use marker function if frames align with markers.
        # For now, raise NotImplementedError until Pinocchio's CasADi support for this is confirmed/implemented.
        raise NotImplementedError("Symbolic RT (Frame Pose) function not implemented for Pinocchio yet.")
        # Placeholder logic:
        # pose_func = self.pinocchio_casadi.some_frame_placement_function(self.model, self._q_sym, rt_index)
        # return ca.Function(f"rt_{rt_index}", [self._q_sym], [pose_func], ["q"], ["rt_matrix"])

    @cache_function
    def center_of_mass(self) -> ca.Function:
        """Get the center of mass of the model"""
        com_expr = self.pinocchio_casadi.centerOfMass(self.model, self._q_sym)
        return ca.Function("center_of_mass", [self._q_sym], [com_expr], ["q"], ["com"])

    @cache_function
    def center_of_mass_velocity(self) -> ca.Function:
        """Get the center of mass velocity of the model"""
        com_vel_expr = self.pinocchio_casadi.centerOfMassVelocity(self.model, self._q_sym, self._qdot_sym)
        return ca.Function(
            "center_of_mass_velocity", [self._q_sym, self._qdot_sym], [com_vel_expr], ["q", "qdot"], ["com_vel"]
        )

    @cache_function
    def center_of_mass_acceleration(self) -> ca.Function:
        """Get the center of mass acceleration of the model"""
        com_acc_expr = self.pinocchio_casadi.centerOfMassAcceleration(
            self.model, self._q_sym, self._qdot_sym, self._qddot_sym
        )
        return ca.Function(
            "center_of_mass_acceleration",
            [self._q_sym, self._qdot_sym, self._qddot_sym],
            [com_acc_expr],
            ["q", "qdot", "qddot"],
            ["com_acc"],
        )

    @cache_function
    def angular_momentum(self) -> ca.Function:
        """Get the angular momentum of the model"""
        # Need to compute centroidal momentum matrix first
        # H = Ag(q) * v
        Ag_expr = self.pinocchio_casadi.computeCentroidalMomentumTimeVariation(self.model, self._q_sym, self._qdot_sym)
        # This computes dAg/dt * v, not Ag. Need Ag directly.
        # Option: computeCentroidalMap
        Ag_map = self.pinocchio_casadi.computeCentroidalMap(self.model, self._q_sym)
        H_expr = Ag_map @ self._qdot_sym  # Centroidal momentum H = [linear, angular]
        angular_momentum_expr = H_expr[3:]  # Extract angular part

        return ca.Function(
            "angular_momentum", [self._q_sym, self._qdot_sym], [angular_momentum_expr], ["q", "qdot"], ["ang_mom"]
        )

    @cache_function
    def reshape_qdot(self, q_sym: ca.MX | ca.SX, qdot_sym: ca.MX | ca.SX) -> ca.Function:
        """
        Map velocity vector (qdot_sym, nv) to tangent vector of configuration (nq).
        For free flyers, this involves mapping angular velocity to quaternion derivatives.
        If nq == nv, returns qdot_sym directly.
        args: q_sym (nq), qdot_sym (nv)
        """
        if self.nb_q == self.nb_qdot:
            # If dimensions match, assume qdot is already the time derivative of q
            return ca.Function("reshape_qdot", [q_sym, qdot_sym], [qdot_sym], ["q", "qdot_in"], ["qdot_out"])
        elif self._has_floating_base:
            # Use pinocchio.integrate to get the idea, need symbolic version
            # dq = pinocchio.integrate(model, q_zero, v*dt) - q_zero / dt
            # Pinocchio CasADi might offer pin.casadi.integrate or similar.
            # Placeholder: Need symbolic integration or velocity mapping function.
            # For now, assume bioptim handles this internally or raises error if nq!=nv.
            # Let's return qdot_sym but WARN, as this isn't strictly correct for free flyers.
            print(
                "Warning: reshape_qdot for floating base Pinocchio model is approximate. Bioptim assumes qdot has dimension nq."
            )
            # Ideally, bioptim should handle the q/v difference. If it expects nq output:
            # Need a function qdot_nv_to_qdot_nq(q_nq, qdot_nv) -> qdot_nq
            raise NotImplementedError("Symbolic velocity mapping (nv -> nq) for floating base not implemented.")
            # return ca.Function("reshape_qdot", [q_sym, qdot_sym], [qdot_sym_mapped_to_nq], ...)
        else:
            # Other cases where nq != nv (e.g., specific joints)?
            raise NotImplementedError(
                f"reshape_qdot not implemented for model where nq ({self.nb_q}) != nv ({self.nb_qdot})"
            )

    @property
    def name_dof(self) -> tuple[str, ...]:
        """Get the name of the degrees of freedom (velocity dim: nv)"""
        # Pinocchio's model.names includes 'universe'. Velocity names correspond to joints.
        # However, the velocity vector 'v' aligns with tangent space, need names for that.
        # Often joint names map directly to velocity components after the root.
        # Let's return joint names, excluding universe.
        return tuple(self.model.names[1:])

    @property
    def contact_names(self) -> tuple[str, ...]:
        """Get the name of the contacts (use frames designated as contacts)"""
        # Requires a convention, e.g., frames named "contact_*"
        contact_frames = []
        for frame in self.model.frames:
            if frame.name.startswith("contact_"):
                contact_frames.append(frame.name)
        return tuple(contact_frames)

    @property
    def nb_soft_contacts(self) -> int:
        """Get the number of soft contacts (Not directly supported by base Pinocchio)"""
        return 0  # Assume no soft contacts unless extended

    @property
    def soft_contact_names(self) -> tuple[str, ...]:
        """Get the soft contact names"""
        return ()

    @property
    def muscle_names(self) -> tuple[str, ...]:
        """Get the muscle names (Not supported by Pinocchio)"""
        return ()

    @cache_function
    def torque(self, activation_sym: ca.MX | ca.SX, q_sym: ca.MX | ca.SX, qdot_sym: ca.MX | ca.SX) -> ca.Function:
        """Get the muscle torque (Not supported by Pinocchio)"""
        # Return zero torque matching actuator dimension (nv)
        zero_tau = ca.MX.zeros(self.nb_tau, 1)
        return ca.Function("muscle_torque", [activation_sym, q_sym, qdot_sym], [zero_tau], ["a", "q", "qdot"], ["tau"])

    @cache_function
    def forward_dynamics_free_floating_base(
        self, q_sym: ca.MX | ca.SX, qdot_sym: ca.MX | ca.SX, qddot_joints_sym: ca.MX | ca.SX
    ) -> ca.Function:
        """compute the free floating base forward dynamics"""
        # Pinocchio's ABA computes full qddot. Need function that takes only joint acc.
        # pinocchio.computeFreeFlyerMassInverse perhaps?
        # Or use constraint dynamics formulation?
        # Defer implementation - This requires specific Pinocchio algorithms.
        raise NotImplementedError("forward_dynamics_free_floating_base not implemented for PinocchioModel.")

    def reorder_qddot_root_joints(self, qddot_root: ca.MX | ca.SX, qddot_joints: ca.MX | ca.SX) -> ca.Function:
        """Reorder qddot from root and joints (assumes root is first self.nb_root DoFs)"""
        qddot_full = ca.vertcat(qddot_root, qddot_joints)
        return ca.Function("reorder_qddot", [qddot_root, qddot_joints], [qddot_full])

    @cache_function
    def forward_dynamics(self, with_contact=False) -> ca.Function:
        """compute the forward dynamics (qddot = aba(q, v, tau))"""
        if with_contact:
            # Pinocchio's contact dynamics (e.g., pin.forwardDynamics with constraints)
            # needs more setup (contact models, etc.) - Defer for now.
            raise NotImplementedError("Forward dynamics with contact not implemented for PinocchioModel.")

        qddot_expr = self.pinocchio_casadi.aba(self.model, self._q_sym, self._qdot_sym, self._tau_sym)
        inputs = [self._q_sym, self._qdot_sym, self._tau_sym]
        input_names = ["q", "qdot", "tau"]
        # TODO: Handle external forces if protocol requires it as input here
        # external_forces_sym = ca.MX.sym(...)
        # qddot_expr = self.pinocchio_casadi.aba(..., fext=external_forces_sym)
        # inputs.append(external_forces_sym)
        # input_names.append("external_forces")
        return ca.Function("forward_dynamics", inputs, [qddot_expr], input_names, ["qddot"])

    @cache_function
    def inverse_dynamics(self) -> ca.Function:
        """compute the inverse dynamics (tau = rnea(q, v, a))"""
        tau_expr = self.pinocchio_casadi.rnea(self.model, self._q_sym, self._qdot_sym, self._qddot_sym)
        inputs = [self._q_sym, self._qdot_sym, self._qddot_sym]
        input_names = ["q", "qdot", "qddot"]
        # TODO: Handle external forces if protocol requires it as input here
        # external_forces_sym = ca.MX.sym(...)
        # tau_expr = self.pinocchio_casadi.rnea(..., fext=external_forces_sym)
        # inputs.append(external_forces_sym)
        # input_names.append("external_forces")
        return ca.Function("inverse_dynamics", inputs, [tau_expr], input_names, ["tau"])

    @cache_function
    def contact_forces_from_constrained_forward_dynamics(self) -> ca.Function:
        """compute the contact forces (requires contact dynamics setup)"""
        raise NotImplementedError(
            "contact_forces_from_constrained_forward_dynamics not implemented for PinocchioModel."
        )

    @cache_function
    def qdot_from_impact(self) -> ca.Function:
        """compute the post-impact velocities (requires impulse dynamics setup)"""
        raise NotImplementedError("qdot_from_impact not implemented for PinocchioModel.")

    @cache_function
    def muscle_activation_dot(
        self, muscle_excitations_sym: ca.MX | ca.SX, muscle_activations_sym: ca.MX | ca.SX
    ) -> ca.Function:
        """Get the activation derivative (Not supported)"""
        # Return zero derivative matching number of muscles (0)
        zero_dot = ca.MX()  # Empty matrix
        return ca.Function("muscle_activation_dot", [muscle_excitations_sym, muscle_activations_sym], [zero_dot])

    @cache_function
    def muscle_joint_torque(
        self, muscle_states_sym: ca.MX | ca.SX, q_sym: ca.MX | ca.SX, qdot_sym: ca.MX | ca.SX
    ) -> ca.Function:
        """Get the muscular joint torque (Not supported)"""
        zero_tau = ca.MX.zeros(self.nb_tau, 1)
        return ca.Function("muscle_joint_torque", [muscle_states_sym, q_sym, qdot_sym], [zero_tau])

    @cache_function
    def muscle_length_jacobian(self) -> ca.Function:
        """Get the muscle length jacobian (Not supported)"""
        # Return empty jacobian
        zero_jac = ca.MX()
        return ca.Function("muscle_length_jacobian", [self._q_sym], [zero_jac])

    @cache_function
    def muscle_velocity(self) -> ca.Function:
        """Get the muscle velocity (Not supported)"""
        zero_vel = ca.MX()
        return ca.Function("muscle_velocity", [self._q_sym, self._qdot_sym], [zero_vel])

    @cache_function
    def markers(self) -> ca.Function:
        """Get the markers of the model (positions of all frames)"""
        # Compute positions of all frames symbolically
        all_frame_positions = []
        # Requires symbolic forwardKinematics and updateFramePlacements if possible
        # Alternative: Loop through frame IDs and call a symbolic frame_pose function
        # For now, assume pinocchio casadi provides a way to get all frame poses
        try:
            # This hypothetical function computes placement of *all* frames
            all_poses_expr = self.pinocchio_casadi.updateFramePlacements(self.model, self._q_sym)
            # Extract translation part for each frame
            for frame_id in range(self.model.nframes):
                # Need a symbolic way to get frame pose from all_poses_expr or recompute
                # frame_pose = symbolic_get_frame_pose(all_poses_expr, frame_id) # Hypothetical
                # all_frame_positions.append(frame_pose[:3, 3]) # Extract translation

                # Workaround: Recompute each frame individually (less efficient)
                frame_pos = self.pinocchio_casadi.framePlacement(self.model, self._q_sym, frame_id).translation()
                all_frame_positions.append(frame_pos)

        except AttributeError:
            raise NotImplementedError("Symbolic computation of all frame placements not found/implemented.")

        markers_expr = ca.horzcat(*all_frame_positions)  # Shape (3, nframes)
        return ca.Function("markers", [self._q_sym], [markers_expr], ["q"], ["marker_pos"])

    @property
    def nb_markers(self) -> int:
        """Get the number of markers (using frames as markers)"""
        return self.model.nframes

    def marker_index(self, name: str) -> int:
        """Get the index of a marker (using frame name)"""
        if self.model.existFrameName(name):
            return self.model.getFrameId(name)
        else:
            raise ValueError(f"Marker (frame) '{name}' not found in Pinocchio model.")

    @property
    def marker_names(self) -> tuple[str, ...]:
        """Get the marker names (using frame names)"""
        return tuple(frame.name for frame in self.model.frames)

    # --- Velocity/Acceleration ---
    # These require symbolic Jacobians and their time derivatives. Pinocchio CasADi provides these.

    @cache_function
    def frame_velocity(self, frame_id: int, reference_frame=pin.LOCAL) -> ca.Function:
        """Get the velocity of one frame"""
        frame_vel_expr = self.pinocchio_casadi.getFrameVelocity(
            self.model, self._q_sym, self._qdot_sym, frame_id, reference_frame
        )
        return ca.Function(
            f"frame_velocity_{frame_id}", [self._q_sym, self._qdot_sym], [frame_vel_expr], ["q", "qdot"], ["frame_vel"]
        )

    @cache_function
    def frame_acceleration(self, frame_id: int, reference_frame=pin.LOCAL) -> ca.Function:
        """Get the acceleration of one frame"""
        frame_acc_expr = self.pinocchio_casadi.getFrameAcceleration(
            self.model, self._q_sym, self._qdot_sym, self._qddot_sym, frame_id, reference_frame
        )
        return ca.Function(
            f"frame_acceleration_{frame_id}",
            [self._q_sym, self._qdot_sym, self._qddot_sym],
            [frame_acc_expr],
            ["q", "qdot", "qddot"],
            ["frame_acc"],
        )

    @cache_function
    def markers_velocities(self, reference_frame=pin.LOCAL) -> ca.Function:
        """Get the velocities of all markers (frames)"""
        all_frame_vels = []
        for frame_id in range(self.model.nframes):
            # Requires symbolic getFrameVelocity
            frame_vel_expr = self.pinocchio_casadi.getFrameVelocity(
                self.model, self._q_sym, self._qdot_sym, frame_id, reference_frame
            ).linear()  # Linear velocity part
            all_frame_vels.append(frame_vel_expr)
        markers_vel_expr = ca.horzcat(*all_frame_vels)  # Shape (3, nframes)
        return ca.Function(
            "markers_velocities", [self._q_sym, self._qdot_sym], [markers_vel_expr], ["q", "qdot"], ["marker_vel"]
        )

    @cache_function
    def marker_velocity(self, marker_index: int, reference_frame=pin.LOCAL) -> ca.Function:
        """Get the linear velocity of one marker (frame)"""
        frame_vel_expr = self.pinocchio_casadi.getFrameVelocity(
            self.model, self._q_sym, self._qdot_sym, marker_index, reference_frame
        ).linear()
        return ca.Function(
            f"marker_velocity_{marker_index}",
            [self._q_sym, self._qdot_sym],
            [frame_vel_expr],
            ["q", "qdot"],
            ["marker_vel"],
        )

    @cache_function
    def markers_accelerations(self, reference_frame=pin.LOCAL) -> ca.Function:
        """Get the accelerations of all markers (frames)"""
        all_frame_accs = []
        for frame_id in range(self.model.nframes):
            # Requires symbolic getFrameAcceleration
            frame_acc_expr = self.pinocchio_casadi.getFrameAcceleration(
                self.model, self._q_sym, self._qdot_sym, self._qddot_sym, frame_id, reference_frame
            ).linear()  # Linear acceleration part
            all_frame_accs.append(frame_acc_expr)
        markers_acc_expr = ca.horzcat(*all_frame_accs)  # Shape (3, nframes)
        return ca.Function(
            "markers_accelerations",
            [self._q_sym, self._qdot_sym, self._qddot_sym],
            [markers_acc_expr],
            ["q", "qdot", "qddot"],
            ["marker_acc"],
        )

    @cache_function
    def marker_acceleration(self, marker_index: int, reference_frame=pin.LOCAL) -> ca.Function:
        """Get the linear acceleration of one marker (frame)"""
        frame_acc_expr = self.pinocchio_casadi.getFrameAcceleration(
            self.model, self._q_sym, self._qdot_sym, self._qddot_sym, marker_index, reference_frame
        ).linear()
        return ca.Function(
            f"marker_acceleration_{marker_index}",
            [self._q_sym, self._qdot_sym, self._qddot_sym],
            [frame_acc_expr],
            ["q", "qdot", "qddot"],
            ["marker_acc"],
        )

    # --- End Velocity/Acceleration ---

    @property
    def nb_rigid_contacts(self) -> int:
        """Get the number of rigid contacts (using frame convention)"""
        return len(self.contact_names)

    @cache_function
    def tau_max(self) -> ca.Function:
        """Get the maximum torque (use model.effortLimit)"""
        tau_max_val = ca.MX(self.model.effortLimit)
        # Need to make sure this returns a function taking q, qdot even if limit is constant
        return ca.Function("tau_max", [self._q_sym, self._qdot_sym], [tau_max_val], ["q", "qdot"], ["tau_max"])

    @cache_function
    def rigid_contact_acceleration(self, contact_index: int, contact_axis: int) -> ca.Function:
        """Get the rigid contact acceleration (requires contact dynamics setup)"""
        # Need frame acceleration and projection onto axis
        frame_acc_func = self.frame_acceleration(
            contact_index, reference_frame=pin.LOCAL
        )  # Acceleration in local frame
        frame_acc_expr = frame_acc_func(self._q_sym, self._qdot_sym, self._qddot_sym)

        # Project onto axis (e.g., axis 0 is X, 1 is Y, 2 is Z in local frame)
        contact_acc_expr = frame_acc_expr[contact_axis]

        return ca.Function(
            f"rigid_contact_acceleration_{contact_index}_{contact_axis}",
            [self._q_sym, self._qdot_sym, self._qddot_sym],
            [contact_acc_expr],
            ["q", "qdot", "qddot"],
            ["contact_acc"],
        )

    @cache_function
    def soft_contact_forces(self) -> ca.Function:
        """Get the soft contact forces (Not supported)"""
        zero_force = ca.MX()
        return ca.Function("soft_contact_forces", [self._q_sym, self._qdot_sym], [zero_force])

    @cache_function
    def normalize_state_quaternions(self, q_sym: ca.MX | ca.SX) -> ca.Function:
        """Normalize the quaternions of the state"""
        if not self._has_floating_base:
            # No quaternions to normalize (assuming fixed base or non-quaternion root)
            return ca.Function("normalize_state_quaternions", [q_sym], [q_sym])

        # Assumes quaternion is at indices 3, 4, 5, 6 (Pinocchio convention for free flyer q)
        quat_indices = [3, 4, 5, 6]
        q_normalized = ca.MX(q_sym)  # Make a mutable copy
        quaternion = q_normalized[quat_indices]
        normalized_quaternion = quaternion / ca.norm_2(quaternion)
        q_normalized[quat_indices] = normalized_quaternion

        return ca.Function("normalize_state_quaternions", [q_sym], [q_normalized], ["q_in"], ["q_out"])

    @cache_function
    def rigid_contact_forces(self) -> ca.Function:
        """Get rigid contact forces (requires constrained dynamics)"""
        raise NotImplementedError(
            "rigid_contact_forces requires solving constrained dynamics, not directly available via this protocol method for PinocchioModel."
        )

    @cache_function
    def passive_joint_torque(self) -> ca.Function:
        """Get the passive joint torque (use model.damping * qdot)"""
        passive_tau_expr = ca.diag(self.model.damping) @ self._qdot_sym
        return ca.Function(
            "passive_joint_torque", [self._q_sym, self._qdot_sym], [passive_tau_expr], ["q", "qdot"], ["passive_tau"]
        )

    @cache_function
    def ligament_joint_torque(self) -> ca.Function:
        """Get the ligament joint torque (Not directly supported)"""
        zero_tau = ca.MX.zeros(self.nb_tau, 1)
        return ca.Function("ligament_joint_torque", [self._q_sym, self._qdot_sym], [zero_tau])

    def bounds_from_ranges(self, variables: str | list[str], mapping: BiMapping | BiMappingList = None) -> Bounds:
        """Create bounds from ranges of the model"""
        if isinstance(variables, str):
            variables = [variables]

        nq = self.nb_q
        nv = self.nb_qdot  # nv = qdot dim

        # Get limits from pinocchio model
        q_min = ca.DM(self.model.lowerPositionLimit)
        q_max = ca.DM(self.model.upperPositionLimit)
        qdot_min = -ca.DM(self.model.velocityLimit)
        qdot_max = ca.DM(self.model.velocityLimit)
        qddot_min = -ca.DM(self.model.effortLimit)  # Use effort limits as proxy? Needs review.
        qddot_max = ca.DM(self.model.effortLimit)

        # Handle potential infinite bounds from pinocchio (often large numbers)
        inf = float("inf")
        q_min[ca.is_inf(q_min)] = -inf
        q_max[ca.is_inf(q_max)] = inf
        qdot_min[ca.is_inf(qdot_min)] = -inf
        qdot_max[ca.is_inf(qdot_max)] = inf
        qddot_min[ca.is_inf(qddot_min)] = -inf
        qddot_max[ca.is_inf(qddot_max)] = inf

        min_bounds = ca.DM()
        max_bounds = ca.DM()

        for var in variables:
            if var == "q":
                min_b = q_min
                max_b = q_max
            elif var == "qdot":
                # Bioptim expects qdot bounds to have nq rows if var=='qdot' but model limits are nv.
                # Need careful mapping, especially for floating base.
                # Simple approach: Assume mapping applies, or use nv rows if qdot refers to velocity vector.
                # Let's assume qdot here refers to the velocity vector v (nv dimensions).
                min_b = qdot_min
                max_b = qdot_max
                if mapping and "qdot" in mapping:  # Apply mapping if provided for qdot
                    min_b = mapping["qdot"].map(min_b)
                    max_b = mapping["qdot"].map(max_b)
                elif mapping and "q" in mapping and "qdot" not in mapping:  # Apply q map if only one provided
                    min_b = mapping["q"].map(min_b)
                    max_b = mapping["q"].map(max_b)

            elif var == "qddot":
                min_b = qddot_min
                max_b = qddot_max
                # Apply mapping similarly if needed
                if mapping and "qddot" in mapping:
                    min_b = mapping["qddot"].map(min_b)
                    max_b = mapping["qddot"].map(max_b)
                elif mapping and "q" in mapping and "qddot" not in mapping:  # Apply q map
                    min_b = mapping["q"].map(min_b)
                    max_b = mapping["q"].map(max_b)
            else:
                raise ValueError(f"Unknown variable '{var}' for bounds_from_ranges")

            min_bounds = ca.vertcat(min_bounds, min_b)
            max_bounds = ca.vertcat(max_bounds, max_b)

        bounds = Bounds(min_bound=np.array(min_bounds).flatten(), max_bound=np.array(max_bounds).flatten())
        return bounds

    @cache_function
    def lagrangian(self) -> ca.Function:
        """Compute the Lagrangian L = K - P"""
        # Use Pinocchio CasADi functions for kinetic and potential energy
        K = self.pinocchio_casadi.computeKineticEnergy(self.model, self._q_sym, self._qdot_sym)
        P = self.pinocchio_casadi.computePotentialEnergy(self.model, self._q_sym)
        L = K - P
        return ca.Function("lagrangian", [self._q_sym, self._qdot_sym], [L], ["q", "qdot"], ["L"])

    @cache_function
    def partitioned_forward_dynamics(self, q_u, qdot_u, q_v_init, tau) -> ca.Function:
        """Forward dynamics for independent joints (Not directly supported by base Pinocchio ABA/RNEA)"""
        # This requires algorithms like constrained DAE solvers or specific Featherstone methods.
        raise NotImplementedError("partitioned_forward_dynamics not implemented for PinocchioModel.")

    @staticmethod
    def animate(ocp, solution: "SolutionData", show_now: bool = True, **kwargs: Any) -> None | list:
        """
        Animate a solution using an external viewer like meshcat-python or bioviz.

        Note: Requires a viewer to be installed and configured.
        """
        # Option 1: Use bioviz if model is convertible or via intermediate format
        # Option 2: Use meshcat-python (common with Pinocchio)
        try:
            # Attempt to use a generic viewer utility or bioviz
            from ..models.biorbd.viewer_utils import display_solution_from_app  # Re-use Biorbd's viewer?

            # This might require solution data format to be compatible
            return display_solution_from_app(ocp, solution.integrate(), **kwargs)

        except ImportError:
            print("Warning: Animation requires 'bioviz' or another viewer configured. Skipping animation.")
            return None
        except Exception as e:
            print(f"Warning: Animation failed. Error: {e}. Skipping animation.")
            return None
