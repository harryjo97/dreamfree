"""NumPy-only contracts, also usable when copied to the robot workstation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROTOCOL = "dreamfree_yam_standard_v1"
ABC_REVISION = "d0832d12651d1b260a652861a14648dc5f3660c7"
CAMERAS = ("top", "left", "right")
JOINTS = tuple(range(6)) + tuple(range(7, 13))
WARMUP_INSTRUCTION = "optimized inference warmup"
EEF_MODEL_SHA256 = "a797c4c27d1c074eb24b04d9cf13cef9a66589f135ebf35bb22f2be3b86f7d28"


class EefTargetRejected(ValueError):
    """A proposed EEF target failed admission before publication; eligible for replanning."""


def action_contract(dataset_name):
    """Describe policy outputs separately from the always-14D robot interface."""
    if dataset_name not in ("yam_abc130k", "yam_abc130k_eef"):
        raise ValueError(f"Unsupported YAM deployment dataset: {dataset_name!r}.")
    eef = dataset_name == "yam_abc130k_eef"
    result = dict(
        dataset_name=dataset_name,
        action_space="eef" if eef else "joint",
        action_dim=20 if eef else 14,
        proprio_dim=14,
        action_representation="absolute_wrist_pose_rot6d_continuous_apertures"
        if eef
        else "absolute_joints_continuous_apertures",
    )
    if eef:
        result["eef_kinematics"] = dict(
            model_sha256=EEF_MODEL_SHA256,
            frame="per_arm_base",
            site="tcp_site",
            position_units="meters",
            rotation="first_two_columns",
            arm_order=["left", "right"],
            gripper_indices=[9, 19],
        )
    return result


def write_json(path, value):
    path = Path(path)
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def validate_server(
    info, *, expected_checkpoint=None, expected_variant=None, replan_steps=8, action_space="joint"
):
    if action_space not in ("joint", "eef"):
        raise ValueError(f"Unsupported action space: {action_space!r}.")
    contract = action_contract("yam_abc130k_eef" if action_space == "eef" else "yam_abc130k")
    expected = dict(
        protocol=PROTOCOL,
        **contract,
        action_horizon=32,
        fps=30,
        camera_size_hw=[480, 640],
        cameras=list(CAMERAS),
        text_mode="encoder",
    )
    for key, value in expected.items():
        if info.get(key) != value:
            raise ValueError(f"YAM server {key}: expected {value!r}, got {info.get(key)!r}.")
    if info.get("variant") not in ("standard", "action_history"):
        raise ValueError("Server must identify its Stage 2 variant.")
    history = info.get("action_history_horizon")
    if type(history) is not int or (
        history <= 0 if info["variant"] == "action_history" else history != 0
    ):
        raise ValueError("Server action-history horizon disagrees with its variant.")
    if history + 1 + info["action_horizon"] > 1024:
        raise ValueError("Server action-history query length exceeds the RoPE cache.")
    if expected_checkpoint and info.get("checkpoint") != str(expected_checkpoint).rstrip("/"):
        raise ValueError(
            f"Server checkpoint mismatch: {info.get('checkpoint')!r} != {str(expected_checkpoint)!r}."
        )
    if expected_variant and info["variant"] != expected_variant:
        raise ValueError(
            f"Server variant mismatch: expected {expected_variant!r}, got {info['variant']!r}."
        )
    if not 1 <= replan_steps <= info["action_horizon"] or replan_steps != info.get("replan_steps"):
        raise ValueError(
            "Client --open-loop-horizon must match the server's --replan-steps (1..32)."
        )
    optimization = info.get("inference_optimization", {})
    if optimization.get("enabled") and not optimization.get("warmup_complete"):
        raise ValueError("Optimized YAM server has not completed warmup.")


def validate_profile(profile):
    if list(profile.robots) != ["left", "right"]:
        raise ValueError("YAM profile robots must be ordered left, right (seven values each).")
    if set(profile.cameras) != set(CAMERAS):
        raise ValueError("YAM profile requires top, left, right cameras.")
    for name, camera in profile.cameras.items():
        if (camera.height, camera.width, camera.fps) != (480, 640, 30):
            raise ValueError(f"Camera {name} must be 640x480 RGB at 30 FPS.")
    for name, robot in profile.robots.items():
        state = np.asarray(robot.init_q, dtype=np.float32)
        if state.shape != (7,) or not np.isfinite(state).all() or not 0 <= state[6] <= 1:
            raise ValueError(
                f"Robot {name} init_q must have six finite radians and aperture in [0,1]."
            )
        if robot.follower.control_rate < 30:
            raise ValueError(f"Robot {name} follower must support at least 30 Hz.")


def validate_observation(observation, *, allow_preprocessed=True):
    if __package__:
        from .image_preprocessing import PACKED_IMAGE_KEY, PREPROCESSING_KEY, validate_packed_image
    else:
        from image_preprocessing import PACKED_IMAGE_KEY, PREPROCESSING_KEY, validate_packed_image

    state = np.asarray(observation.get("state"), dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("YAM observation state must contain 14 finite values.")
    packed = PREPROCESSING_KEY in observation or PACKED_IMAGE_KEY in observation
    if packed:
        if not allow_preprocessed:
            raise ValueError("Expected raw YAM images; refusing repeated preprocessing.")
        validate_packed_image(observation)
    for name in CAMERAS:
        if not packed:
            image = observation.get("images", {}).get(name)
            if (
                not isinstance(image, np.ndarray)
                or image.dtype != np.uint8
                or image.shape != (3, 480, 640)
            ):
                raise ValueError(f"YAM camera {name} must be uint8 RGB CHW (3,480,640).")
        if not observation.get("masks", {}).get(name, False):
            raise ValueError(f"YAM camera {name} is unavailable.")
    if not isinstance(observation.get("prompt"), str) or not observation["prompt"].strip():
        raise ValueError("A nonempty task instruction is required.")
    return state


def executable_chunk(actions, horizon=32, *, action_space="joint"):
    if action_space not in ("joint", "eef"):
        raise ValueError(f"Unsupported action space: {action_space!r}.")
    dim, indices = (20, [9, 19]) if action_space == "eef" else (14, [6, 13])
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape != (horizon, dim) or not np.isfinite(actions).all():
        raise ValueError(f"Expected finite ({horizon},{dim}) action chunk, got {actions.shape}.")
    result = actions.copy()
    grip = result[:, indices]
    clipped = int(np.count_nonzero((grip < 0) | (grip > 1)))
    result[:, indices] = np.clip(grip, 0, 1)
    return result, clipped


def validate_joint_step(command, previous, max_delta):
    if max_delta is None:
        return
    delta = np.abs(np.asarray(command)[list(JOINTS)] - np.asarray(previous)[list(JOINTS)])
    if np.any(delta > max_delta):
        joint = JOINTS[int(np.argmax(delta))]
        raise ValueError(
            f"Joint {joint} command delta {delta.max():.4f} rad exceeds {max_delta:.4f}; no command sent."
        )
