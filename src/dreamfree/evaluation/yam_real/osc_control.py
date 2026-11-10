"""Offline-capable Cartesian torque control; no CAN imports or hardware side effects."""

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

if __package__:
    from .common import EefTargetRejected
    from .eef_control import DRIVER_LIMITS, WristKinematics, decode_rotation
else:
    from common import EefTargetRejected
    from eef_control import DRIVER_LIMITS, WristKinematics, decode_rotation

CONTRACT_PATH = Path(__file__).with_name("osc_i2rt_contract.json")
MODEL_FILES = {
    "crank_4310": "yam.xml",
    "linear_4310": "yam_4310_linear.xml",
    "linear_3507": "yam_lw_gripper.xml",
}
HISTORY_CONTRACT = "osc_accepted_cartesian_targets_v1"


class OscFault(RuntimeError):
    """Latched deployment fault; callers must disable motors and end the session."""


def vector(value, size, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values.")
    return result


def rotation_vector(rotation):
    """Principal SO(3) logarithm, in the base frame; stable near zero and pi."""
    import mujoco

    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.asarray(rotation, dtype=np.float64).reshape(9))
    if quat[0] < 0:
        quat = -quat
    sine = np.linalg.norm(quat[1:])
    return 2 * quat[1:] if sine < 1e-10 else quat[1:] * (2 * np.arctan2(sine, quat[0]) / sine)


def rotation_exp(value):
    angle = np.linalg.norm(value)
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = value / angle
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew


def encode_pose(position, rotation, aperture):
    return np.concatenate((position, rotation[:, 0], rotation[:, 1], [aperture]))


@dataclass
class OscConfig:
    # These are simulation examples, not hardware-tuned gains or safety ratings.
    kp: tuple = (100.0, 100.0, 100.0, 60.0, 60.0, 60.0)
    kd: tuple = (20.0, 20.0, 20.0, 16.0, 16.0, 16.0)
    torque_limit: tuple = (12.0, 12.0, 12.0, 3.0, 3.0, 3.0, 0.6)
    torque_rate: tuple = (300.0, 300.0, 300.0, 80.0, 80.0, 80.0, 10.0)
    joint_lower: tuple = tuple(DRIVER_LIMITS[:, 0])
    joint_upper: tuple = tuple(DRIVER_LIMITS[:, 1])
    joint_margin: float = 0.05
    joint_speed_limit: float = 2.0
    stopping_acceleration: float = 5.0
    position_step: float = 0.01
    rotation_step: float = 0.08
    target_linear_speed: float = 0.15
    target_angular_speed: float = 1.0
    gripper_speed: float = 1.0
    gripper_margin: float = 0.05
    max_position_error: float = 0.08
    max_orientation_error: float = 0.6
    gripper_kp: float = 2.0
    gripper_kd: float = 0.1
    task_length_scale: float = 0.15
    inertia_damping: float = 0.01
    singularity_threshold: float = 0.04
    rate_hz: float = 100.0
    feedback_timeout: float = 0.03
    target_timeout: float = 3.0
    startup_timeout: float = 30.0
    max_period: float = 0.04
    max_consecutive_misses: int = 3
    saturation_timeout: float = 0.5
    can_timeout: float = 0.0015

    def __post_init__(self):
        for name, size in (
            ("kp", 6),
            ("kd", 6),
            ("torque_limit", 7),
            ("torque_rate", 7),
            ("joint_lower", 6),
            ("joint_upper", 6),
        ):
            value = vector(getattr(self, name), size, name)
            setattr(self, name, value)
            if name not in ("joint_lower", "joint_upper") and np.any(value <= 0):
                raise ValueError(f"{name} must be positive.")
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, np.ndarray):
                continue
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{item.name} must be finite and positive.")
        if type(self.max_consecutive_misses) is not int:
            raise ValueError("max_consecutive_misses must be an integer.")
        if (
            np.any(self.joint_lower < DRIVER_LIMITS[:, 0])
            or np.any(self.joint_upper > DRIVER_LIMITS[:, 1])
            or np.any(self.joint_lower + 2 * self.joint_margin >= self.joint_upper)
        ):
            raise ValueError(
                "OSC joint bounds must fit the pinned driver limits with a positive margin."
            )
        if self.max_period <= 1 / self.rate_hz or self.feedback_timeout < 1 / self.rate_hz:
            raise ValueError(
                "max_period and feedback_timeout must accommodate the requested servo period."
            )
        if self.inertia_damping > 1:
            raise ValueError("inertia_damping must not exceed the singularity damping cap of 1.")
        # DM4340/DM4310 wire-format bounds, not continuous hardware torque ratings.
        if np.any(self.torque_limit > [28, 28, 28, 10, 10, 10, 10]):
            raise ValueError("Torque bounds exceed the supported motor protocol ranges.")
        if np.any(self.torque_limit < np.array([28, 28, 28, 10, 10, 10, 10]) / 4095):
            raise ValueError(
                "Torque bounds must accommodate the motor protocol half-quantum around zero."
            )

    def as_dict(self):
        return {
            f.name: (
                getattr(self, f.name).tolist()
                if isinstance(getattr(self, f.name), np.ndarray)
                else getattr(self, f.name)
            )
            for f in fields(self)
        }


class WristDynamics:
    """Six-DOF pinned I2RT dynamics with the training wrist frame, per arm base.

    Meshes are omitted only after checking the original XML hash. Every moving
    body has an explicit inertial; no inertia is inferred from omitted geometry.
    The gripper mass is lumped into link_6, as in the pinned driver model.
    """

    def __init__(self, xml, *, model_file, gravity, training_model):
        import mujoco

        if mujoco.__version__ != "3.8.1":
            raise ValueError(f"OSC validated API requires mujoco==3.8.1, got {mujoco.__version__}.")
        contract = json.loads(CONTRACT_PATH.read_text())
        key = "robot_models/yam/" + model_file
        digest = hashlib.sha256(xml.encode()).hexdigest()
        if digest != contract["source_sha256"].get(key):
            raise ValueError(f"OSC dynamics model hash mismatch: {model_file}.")
        root = ET.fromstring(xml)
        bodies = root.findall(".//body")
        if len(bodies) != 6 or any(b.find("inertial") is None for b in bodies):
            raise ValueError("OSC requires the pinned six-body explicit-inertia model.")
        for parent in root.iter():
            for child in list(parent):
                if child.tag in ("asset", "geom"):
                    parent.remove(child)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        if self.model.nq != 6 or self.model.nv != 6 or self.model.nu != 0:
            raise ValueError(
                "OSC requires exactly six hinge coordinates and direct applied torques."
            )
        if [self.model.joint(i).name for i in range(6)] != [f"joint{i}" for i in range(1, 7)]:
            raise ValueError("OSC dynamics joint order mismatch.")
        self.model.jnt_limited[:] = False
        self.model.opt.gravity[:] = vector(gravity, 3, "gravity in arm base")
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)
        self.gravity_data = mujoco.MjData(self.model)
        self.site = self.model.site("tcp_site").id
        self.body = self.model.site_bodyid[self.site]
        self.fk = WristKinematics(training_model)
        self.metadata = dict(
            model_file=model_file,
            model_sha256=digest,
            gravity=self.model.opt.gravity.tolist(),
            gravity_factor=1.0,
            driver_gravity_enabled=False,
            driver_pd_enabled=False,
            frame="per_arm_base",
            site="tcp_site",
            gripper_model="fixed_lumped_inertia",
        )
        # Check actual transforms, including the wrist orientation, before hardware opens.
        for q in (np.zeros(6), np.array([0.2, 0.7, 1.1, -0.6, 0.3, -0.2])):
            pose, *_ = self.evaluate(q, np.zeros(6))
            if not np.allclose(pose, self.fk.forward(q), atol=1e-12, rtol=0):
                raise ValueError("OSC dynamics FK disagrees with the trained tcp_site frame.")

    def evaluate(self, q, dq):
        import mujoco

        self.data.qpos[:] = vector(q, 6, "joint position")
        self.data.qvel[:] = vector(dq, 6, "joint velocity")
        mujoco.mj_forward(self.model, self.data)
        pose = np.eye(4)
        pose[:3, 3] = self.data.site_xpos[self.site]
        pose[:3, :3] = self.data.site_xmat[self.site].reshape(3, 3)
        jac, jac_dot = np.empty((6, 6)), np.empty((6, 6))
        mujoco.mj_jacSite(self.model, self.data, jac[:3], jac[3:], self.site)
        mujoco.mj_jacDot(self.model, self.data, jac_dot[:3], jac_dot[3:], pose[:3, 3], self.body)
        mass = np.empty((6, 6))
        mujoco.mj_fullM(self.model, mass, self.data.qM)
        bias = self.data.qfrc_bias.copy()
        if not all(np.isfinite(a).all() for a in (pose, jac, jac_dot, mass, bias)):
            raise OscFault("Nonfinite MuJoCo dynamics.")
        return pose, jac, jac_dot @ self.data.qvel, mass, bias

    def gravity(self, q):
        import mujoco

        self.gravity_data.qpos[:] = vector(q, 6, "joint position")
        self.gravity_data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.gravity_data)
        return self.gravity_data.qfrc_bias.copy()


class ArmOsc:
    """Pose/velocity feedback -> operational-space wrench -> joint torque."""

    def __init__(self, dynamics, config, gripper_limits):
        self.dynamics, self.config = dynamics, config
        self.gripper_limits = vector(
            gripper_limits, 2, "calibrated [closed, open] gripper endpoints"
        )
        if abs(np.diff(self.gripper_limits)[0]) < 0.1:
            raise ValueError(
                "OSC requires distinct calibrated gripper endpoints; no auto-calibration."
            )
        self.previous_torque = np.zeros(7)
        self.saturated_for = 0.0

    def validate_state(self, q, dq):
        cfg = self.config
        q, dq = vector(q, 7, "state"), vector(dq, 7, "velocity")
        lower, upper = cfg.joint_lower + cfg.joint_margin, cfg.joint_upper - cfg.joint_margin
        stopping_q = q[:6] + dq[:6] * np.abs(dq[:6]) / (2 * cfg.stopping_acceleration)
        if (
            np.any(q[:6] <= lower)
            or np.any(q[:6] >= upper)
            or np.any(stopping_q <= lower)
            or np.any(stopping_q >= upper)
            or np.any(np.abs(dq[:6]) > cfg.joint_speed_limit)
        ):
            raise OscFault("OSC joint limit, stopping-distance, or joint-speed bound violated.")
        return q, dq

    def target_error(self, pose, target):
        return np.r_[
            target[:3] - pose[:3, 3], rotation_vector(decode_rotation(target[3:9]) @ pose[:3, :3].T)
        ]

    def error_exceeds_bound(self, error):
        return (
            np.linalg.norm(error[:3]) > self.config.max_position_error
            or np.linalg.norm(error[3:]) > self.config.max_orientation_error
        )

    def validate_target(self, q, dq, target):
        """Admission check only; never mutate torque/saturation state or send commands."""
        q, _ = self.validate_state(q, dq)
        target = vector(target, 10, "EEF target")
        error = self.target_error(self.dynamics.fk.forward(q[:6]), target)
        if self.error_exceeds_bound(error):
            raise EefTargetRejected(
                f"OSC proposed target exceeds Cartesian bounds: "
                f"position_error={np.linalg.norm(error[:3]):.6f} m, "
                f"orientation_error={np.linalg.norm(error[3:]):.6f} rad."
            )

    def compute(self, q, dq, target, dt):
        cfg = self.config
        q, dq = self.validate_state(q, dq)
        target = vector(target, 10, "EEF target")
        if not np.isfinite(dt) or not 0 < dt <= cfg.max_period:
            raise OscFault("Invalid or missed OSC servo period.")
        if not 0 <= target[9] <= 1:
            raise OscFault("OSC target aperture must be in [0,1].")
        pose, jac, jdot_dq, mass, bias = self.dynamics.evaluate(q[:6], dq[:6])
        error = self.target_error(pose, target)
        if self.error_exceeds_bound(error):
            raise OscFault("OSC Cartesian tracking-error bound violated.")
        scale = np.array(
            [1.0, 1.0, 1.0, cfg.task_length_scale, cfg.task_length_scale, cfg.task_length_scale]
        )
        js = scale[:, None] * jac
        # Regularized operational inertia; damping increases near kinematic singularities.
        sigma = np.linalg.svd(js, compute_uv=False)[-1]
        damping = cfg.inertia_damping * max(1.0, cfg.singularity_threshold / max(sigma, 1e-8))
        damping = min(damping, 1.0)
        inverse_inertia = js @ np.linalg.solve(mass, js.T)
        acceleration = cfg.kp * error - cfg.kd * (jac @ dq[:6])
        wrench = np.linalg.solve(
            inverse_inertia + damping**2 * np.eye(6), scale * (acceleration - jdot_dq)
        )
        # bias contains C(q,dq)+g(q) exactly once; no MotorChainRobot compensation runs.
        arm_torque = js.T @ wrench + bias
        span = self.gripper_limits[1] - self.gripper_limits[0]
        grip_torque = cfg.gripper_kp * span * (target[9] - q[6]) - cfg.gripper_kd * span * dq[6]
        requested = np.r_[arm_torque, grip_torque]
        if not np.isfinite(requested).all():
            raise OscFault("Nonfinite OSC torque.")
        bounded = np.clip(requested, -cfg.torque_limit, cfg.torque_limit)
        torque = np.clip(
            bounded,
            self.previous_torque - cfg.torque_rate * dt,
            self.previous_torque + cfg.torque_rate * dt,
        )
        limited = not np.allclose(torque, requested, atol=1e-10, rtol=0)
        self.saturated_for = self.saturated_for + dt if limited else 0.0
        if self.saturated_for > cfg.saturation_timeout:
            raise OscFault(
                "OSC torque/torque-rate saturation persisted beyond configured duration."
            )
        self.previous_torque = torque.copy()
        report = dict(
            position_error_m=float(np.linalg.norm(error[:3])),
            orientation_error_rad=float(np.linalg.norm(error[3:])),
            minimum_scaled_singular_value=float(sigma),
            inertia_damping=float(damping),
            torque_limited=limited,
            requested_torque=requested.tolist(),
            torque=torque.tolist(),
        )
        return torque, report


def limit_target(action, previous, dt, config):
    """Limit a policy update once; servo holds this exact accepted target until replaced.

    Returns the commanded EEF history entry and limiting diagnostics, never a measured pose.
    """
    requested = vector(action, 20, "EEF action")
    previous = vector(previous, 20, "previous EEF target")
    if not np.isfinite(dt) or dt <= 0:
        raise OscFault("OSC target update interval must be positive.")
    dt = min(dt, 1 / 30)  # Inference pauses must not allow a larger first target jump.
    accepted, reports = [], []
    for start in (0, 10):
        raw, old = requested[start : start + 10], previous[start : start + 10]
        rotation, old_rotation = decode_rotation(raw[3:9]), decode_rotation(old[3:9])
        dp = raw[:3] - old[:3]
        dr = rotation_vector(rotation @ old_rotation.T)
        pscale = min(
            1.0,
            min(config.position_step, config.target_linear_speed * dt)
            / max(np.linalg.norm(dp), 1e-12),
        )
        rscale = min(
            1.0,
            min(config.rotation_step, config.target_angular_speed * dt)
            / max(np.linalg.norm(dr), 1e-12),
        )
        grip = np.clip(raw[9], 0, 1)
        grip = np.clip(grip, old[9] - config.gripper_speed * dt, old[9] + config.gripper_speed * dt)
        value = encode_pose(old[:3] + pscale * dp, rotation_exp(rscale * dr) @ old_rotation, grip)
        accepted.extend(value)
        reports.append(
            dict(
                position_scale=float(pscale),
                rotation_scale=float(rscale),
                gripper_limited=bool(grip != raw[9]),
                rotation_orthonormalized=bool(
                    not np.allclose(raw[3:9], np.r_[rotation[:, 0], rotation[:, 1]], atol=1e-6)
                ),
            )
        )
    return np.asarray(accepted, np.float64), reports
