"""Standalone NumPy wrist IK for ABC's pinned YAM hardware and EEF20 policy.

No hardware imports or I/O occur during construction or solving. Both arms must
solve before the caller publishes the resulting 14D joint/gripper command.
"""

from __future__ import annotations

import hashlib
import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

if __package__:
    from .common import EEF_MODEL_SHA256, EefTargetRejected
else:
    from common import EEF_MODEL_SHA256, EefTargetRejected

I2RT_REVISION = "852f6fff33970fe9b882c53e6b28c66172c97291"
# get_yam_robot's explicit overrides, including its 0.15-radian buffer.
# https://github.com/ArthurAllshire/i2rt-drivers/blob/852f6fff33970fe9b882c53e6b28c66172c97291/i2rt/robots/get_robot.py
DRIVER_LIMITS = np.array(
    [[-2.767, 3.28], [-0.15, 3.80], [-0.15, 3.28], [-1.72, 1.72], [-1.72, 1.72], [-2.24, 2.24]]
)
DRIVER_HASHES = {
    "get_robot.py": "c2f3c76caa0d5fb6cf85cc2e1884a217c382b37608315514e2949c9ac672b325",
    "motor_chain_robot.py": "be610bd3a4da816fbb03293e513cb3d45476d5950b21a197b55c07b705d5aa16",
}


def validate_driver():
    """Verify the limit/command implementation without opening CAN devices."""
    spec = importlib.util.find_spec("i2rt")
    if spec is None or not spec.origin:
        raise RuntimeError(
            "EEF execution requires ABC's pinned i2rt driver in the workstation environment."
        )
    root = Path(spec.origin).parent
    for name, digest in DRIVER_HASHES.items():
        path = root / "robots" / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(
                f"EEF driver contract mismatch: {path}; expected i2rt {I2RT_REVISION}. "
                "Revalidate the configured joint limits before supporting another driver."
            )
    return dict(revision=I2RT_REVISION, root=str(root), joint_limits=DRIVER_LIMITS.tolist())


def _fixed_transform(element):
    result = np.eye(4)
    result[:3, 3] = np.fromstring(element.get("pos", "0 0 0"), sep=" ")
    quat = np.fromstring(element.get("quat", "1 0 0 0"), sep=" ")
    w, x, y, z = quat / np.linalg.norm(quat)
    result[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    return result


def decode_rotation(columns):
    """Recover a right-handed frame by orthonormalizing two predicted columns.

    Args:
        columns: Shape (6,), ordered R00,R10,R20,R01,R11,R21.

    Returns:
        Rotation matrix (3, 3), mapping wrist coordinates into the arm base.

    Raises:
        ValueError: The columns are nonfinite, near zero, or collinear.
    """
    columns = np.asarray(columns, dtype=np.float64)
    if columns.shape != (6,) or not np.isfinite(columns).all():
        raise ValueError("EEF orientation must contain six finite rotation-column values.")
    first, second = columns[:3], columns[3:]
    first_norm, second_norm = np.linalg.norm(first), np.linalg.norm(second)
    if min(first_norm, second_norm) < 1e-6:
        raise EefTargetRejected("Degenerate EEF rotation: near-zero column.")
    first = first / first_norm
    second = second / second_norm
    second = second - first * np.dot(first, second)
    norm = np.linalg.norm(second)
    if norm < 1e-6:
        raise EefTargetRejected("Degenerate EEF rotation: collinear columns.")
    second /= norm
    return np.column_stack((first, second, np.cross(first, second)))


class WristKinematics:
    """The dataset's exact six-hinge chain, ending at tcp_site in each arm base."""

    def __init__(self, model_path):
        path = Path(model_path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != EEF_MODEL_SHA256:
            raise ValueError(f"EEF wrist model hash mismatch: {path}.")
        root = ET.parse(path).getroot()
        node = root.find("worldbody/body[@name='arm']")
        self.fixed = []
        for index in range(1, 7):
            node = node.find(f"body[@name='link_{index}']")
            self.fixed.append(_fixed_transform(node))
        self.tool = _fixed_transform(node.find("site[@name='tcp_site']"))

    def forward(self, joints, *, jacobian=False):
        """Evaluate the wrist pose and optionally its geometric derivatives.

        Args:
            joints: Six joint angles in radians, shape (6,).
            jacobian: Also return position (3, 6) and rotation (9, 6) derivatives.

        Returns:
            Base-to-wrist transform (4, 4), or that transform and both Jacobians.
        """
        joints = np.asarray(joints, dtype=np.float64)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("Wrist FK requires six finite joint angles.")
        transform, origins, axes = np.eye(4), [], []
        for fixed, q in zip(self.fixed, joints):
            transform = transform @ fixed
            origins.append(transform[:3, 3].copy())
            axes.append(transform[:3, 2].copy())
            c, s = np.cos(q), np.sin(q)
            rotation = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            transform = transform @ rotation
        pose = transform @ self.tool
        if not jacobian:
            return pose
        axes = np.asarray(axes)
        position_jac = np.cross(axes, pose[:3, 3] - np.asarray(origins)).T
        # dR/dq = skew(axis_in_base) @ R, flattened in row-major order.
        rotation_jac = np.stack(
            [np.cross(axis, pose[:3, :3].T).T.reshape(-1) for axis in axes], axis=1
        )
        return pose, position_jac, rotation_jac

    def encode_command(self, command):
        """Express an issued joint target in the policy's action-history space.

        Args:
            command: Shape (14,), left then right six joints and gripper aperture.

        Returns:
            Float32 action (20,): position, two rotation columns, and aperture per arm.
        """
        command = np.asarray(command, dtype=np.float64)
        if command.shape != (14,) or not np.isfinite(command).all():
            raise ValueError("EEF history encoding requires a finite 14D issued joint command.")
        arms = []
        for start in (0, 7):
            pose = self.forward(command[start : start + 6])
            arms.append(
                np.concatenate(
                    (pose[:3, 3], pose[:3, 0], pose[:3, 1], command[start + 6 : start + 7])
                )
            )
        return np.concatenate(arms).astype(np.float32)


class EefController:
    """Bounded local IK; failure aborts, with no seed-return or target projection fallback."""

    def __init__(
        self,
        model_path,
        *,
        max_joint_delta=0.1,
        max_tracking_error=0.35,
        position_tolerance=0.005,
        orientation_tolerance=0.05,
        max_iterations=80,
    ):
        values = (max_joint_delta, max_tracking_error, position_tolerance, orientation_tolerance)
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError(
                "EEF IK motion bounds and pose tolerances must be finite and positive."
            )
        if type(max_iterations) is not int or max_iterations <= 0:
            raise ValueError("EEF IK iterations must be a positive integer.")
        self.kinematics = WristKinematics(model_path)
        self.max_joint_delta, self.max_tracking_error = max_joint_delta, max_tracking_error
        self.position_tolerance, self.orientation_tolerance = (
            position_tolerance,
            orientation_tolerance,
        )
        self.max_iterations = max_iterations

    @property
    def contract(self):
        return dict(
            solver="bounded_damped_least_squares",
            model_sha256=EEF_MODEL_SHA256,
            i2rt_revision=I2RT_REVISION,
            driver_source_sha256=DRIVER_HASHES.copy(),
            joint_limits=DRIVER_LIMITS.tolist(),
            max_joint_delta=self.max_joint_delta,
            max_tracking_error=self.max_tracking_error,
            position_tolerance=self.position_tolerance,
            orientation_tolerance=self.orientation_tolerance,
            max_iterations=self.max_iterations,
            failure_behavior="abort_before_publish",
            collision_checking=False,
        )

    def _solve_arm(self, target, previous, measured):
        """Find a nearby joint target satisfying pose and joint-space bounds.

        Args:
            target: Absolute wrist action (10,) in this arm's base; aperture is unused.
            previous: Last issued joint target (6,), used as the initial IK seed.
            measured: Last received joint observation (6,), bounding tracking error.

        Returns:
            Float32 joint target (6,) and model-based pose errors/iteration count.

        Raises:
            ValueError: Rotation, joint bounds, or final pose tolerances are invalid.
        """
        rotation = decode_rotation(target[3:9])
        # Keep float32 rounding strictly inside the downstream driver's limits.
        lower = np.maximum(DRIVER_LIMITS[:, 0] + 1e-6, previous - self.max_joint_delta)
        upper = np.minimum(DRIVER_LIMITS[:, 1] - 1e-6, previous + self.max_joint_delta)
        lower = np.maximum(lower, measured - self.max_tracking_error)
        upper = np.minimum(upper, measured + self.max_tracking_error)
        if np.any(lower >= upper):
            raise EefTargetRejected(
                "EEF IK has no joint interval satisfying limits, step bound, and tracking bound."
            )
        # Bounds are part of the solve; no post-solve clipping of an accepted target.
        q = np.clip(previous, lower, upper)
        weight, damping = 0.1, 1e-5

        def residual(pose):
            return np.concatenate(
                (target[:3] - pose[:3, 3], weight * (rotation - pose[:3, :3]).ravel())
            )

        for iteration in range(self.max_iterations):
            pose, position_jac, rotation_jac = self.kinematics.forward(q, jacobian=True)
            error = residual(pose)
            # This objective stopping threshold is separate from final pose acceptance.
            if np.linalg.norm(error) < 1e-8:
                break
            # Twelve residuals (3 position + 9 rotation entries) constrain six joints.
            jac = np.concatenate((position_jac, weight * rotation_jac), axis=0)
            delta = np.linalg.solve(jac.T @ jac + damping * np.eye(6), jac.T @ error)
            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                candidate = np.clip(q + scale * delta, lower, upper)
                if (
                    np.linalg.norm(residual(self.kinematics.forward(candidate)))
                    < np.linalg.norm(error) - 1e-12
                ):
                    q, accepted = candidate, True
                    damping = max(1e-8, damping * 0.5)
                    break
            if not accepted:
                damping *= 10
                if damping > 1e3:
                    break
        # Validate the exact float32 command the follower will receive.
        q = q.astype(np.float32)
        pose = self.kinematics.forward(q)
        position_error = float(np.linalg.norm(pose[:3, 3] - target[:3]))
        orientation_error = float(
            np.arccos(np.clip((np.trace(rotation.T @ pose[:3, :3]) - 1) / 2, -1, 1))
        )
        if not np.isfinite(q).all() or not np.isfinite([position_error, orientation_error]).all():
            raise ValueError("EEF IK produced nonfinite joints or pose errors.")
        if (
            np.any(q < lower - 1e-6)
            or np.any(q > upper + 1e-6)
            or position_error > self.position_tolerance
            or orientation_error > self.orientation_tolerance
        ):
            raise EefTargetRejected(
                f"EEF IK target rejected: position_error={position_error:.6f} m "
                f"(max {self.position_tolerance}), orientation_error={orientation_error:.6f} rad "
                f"(max {self.orientation_tolerance}); target unreachable within configured bounds "
                "or local solver did not converge. No command sent."
            )
        return q, dict(
            position_error_m=position_error,
            orientation_error_rad=orientation_error,
            iterations=iteration + 1,
        )

    def command(self, action, *, previous, measured):
        """Solve both arms before returning a command for the caller to publish.

        Args:
            action: Denormalized policy action (20,), with apertures already in [0, 1].
            previous: Previous issued command (14,); use measured state for the first step.
            measured: Most recent received joint/gripper observation (14,).

        Returns:
            Joint command (14,), its FK action-history encoding (20,), and IK diagnostics.
            These describe commanded targets; they do not assert measured pose tracking.
        """
        action, previous, measured = (
            np.asarray(value, dtype=np.float64) for value in (action, previous, measured)
        )
        if action.shape != (20,) or previous.shape != (14,) or measured.shape != (14,):
            raise ValueError(
                "EEF control requires action20, previous joint command14, and measured joint state14."
            )
        if not all(np.isfinite(value).all() for value in (action, previous, measured)):
            raise ValueError("EEF control inputs must be finite.")
        if np.any(action[[9, 19]] < 0) or np.any(action[[9, 19]] > 1):
            raise ValueError("EEF grippers must be bounded to [0,1] before IK.")
        command, diagnostics = np.empty(14, np.float32), {}
        for side, joint_start, eef_start in (("left", 0, 0), ("right", 7, 10)):
            try:
                q, report = self._solve_arm(
                    action[eef_start : eef_start + 10],
                    previous[joint_start : joint_start + 6],
                    measured[joint_start : joint_start + 6],
                )
            except EefTargetRejected as exc:
                raise EefTargetRejected(f"{side} arm: {exc}") from exc
            except ValueError as exc:
                raise ValueError(f"{side} arm: {exc}") from exc
            command[joint_start : joint_start + 6] = q
            command[joint_start + 6] = action[eef_start + 9]
            diagnostics[side] = report
        return command, self.kinematics.encode_command(command), diagnostics


def controller_from_args(args):
    return EefController(
        Path(args.abc_root) / "abc_sim/models/assets/i2rt_yam/yam.xml",
        max_joint_delta=args.max_joint_delta if args.max_joint_delta is not None else 0.1,
        max_tracking_error=args.eef_max_tracking_error,
        position_tolerance=args.eef_position_tolerance,
        orientation_tolerance=args.eef_orientation_tolerance,
        max_iterations=args.eef_ik_iterations,
    )
