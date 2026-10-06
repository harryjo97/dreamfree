"""Run an DreamFree Stage 2 policy on a physical Panda DROID robot.

This client is compatible with Python 3.7 and does not require installing DreamFree.
Copy image_preprocessing.py beside it for the default CPU PyTorch resize path.
"""

import argparse
import contextlib
import datetime
import ipaddress
import json
import numbers
import os
import pickle
import signal
import socket
import struct
import time
from io import BytesIO
from typing import Mapping

import numpy as np

DROID_CONTROL_FREQUENCY_HZ = 15
HEADER_STRUCT = struct.Struct("!Q")
PICKLE_PROTOCOL = 4
MAX_MESSAGE_BYTES = 256 * 1024 * 1024
MAX_ACTION_NOISE_SEED = (1 << 63) - 1
RPC_PROTOCOL_VERSION = 4
GRIPPER_PREDICTION_REPRESENTATION = "continuous_denormalized"
GRIPPER_BINARIZATION = "client_threshold_gt_0.5"
DROID_GRIPPER_NORMALIZED_COMMAND_SCALE = 1.0
DEFAULT_DROID_GRIPPER_CLOSE_THRESHOLD = 0.5
NATIVE_POLICY_RUNTIME = "dreamfree_stage2_student_native_optimized"
SUPPORTED_CHECKPOINT_TYPES = ("dreamfree_stage2",)
ISSUED_ACTION_HISTORY_KEY = "deployment/issued_action_history"
ISSUED_ACTION_HISTORY_IS_PAD_KEY = "deployment/issued_action_history_is_pad"
ACTION_HISTORY_POLICY_RUNTIMES = ()
SUPPORTED_POLICY_RUNTIMES = ("dreamfree_stage2_student_native_optimized",)
SERVER_NAME_BY_POLICY_RUNTIME = {
    "dreamfree_stage2_student_native_optimized": "dreamfree_droid_stage2_real_optimized"
}
OBSERVATION_KEYS = (
    "observation/exterior_image_1_left",
    "observation/exterior_image_2_left",
    "observation/wrist_image_left",
    "observation/joint_position",
    "observation/gripper_position",
)
PANDA_JOINT_MIN_RAD = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float32
)
PANDA_JOINT_MAX_RAD = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float32
)
PANDA_JOINT_VELOCITY_MAX_RAD_S = np.array(
    [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], dtype=np.float32
)
DEFAULT_MAX_SAFETY_REGENERATIONS = 3
DEFAULT_ACTION_SAFETY_MODE = "reject-and-regenerate"
ACTION_SAFETY_MODES = (DEFAULT_ACTION_SAFETY_MODE, "clip", "none")


class UnsafeActionError(RuntimeError):
    """A predicted action violates an explicit Panda motion safety limit."""


class _CompatUnpickler(pickle.Unpickler):
    """Unpickle NumPy 2 payloads in the older NumPy used by DROID."""

    def find_class(self, module, name):
        if module == "numpy._core" or module.startswith("numpy._core."):
            compat_module = module.replace("numpy._core", "numpy.core", 1)
            try:
                return super().find_class(compat_module, name)
            except ModuleNotFoundError:
                pass
        return super().find_class(module, name)


def _loads(payload):
    return _CompatUnpickler(BytesIO(payload)).load()


def _recvall(sock, num_bytes):
    chunks = []
    remaining = int(num_bytes)
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("Socket closed while receiving RPC payload.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(sock, payload):
    data = pickle.dumps(payload, protocol=PICKLE_PROTOCOL)
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError(
            "RPC payload exceeds the maximum size: bytes={} maximum={}".format(
                len(data), MAX_MESSAGE_BYTES
            )
        )
    sock.sendall(HEADER_STRUCT.pack(len(data)) + data)


def recv_message(sock):
    header = _recvall(sock, HEADER_STRUCT.size)
    num_bytes = HEADER_STRUCT.unpack(header)[0]
    if num_bytes <= 0 or num_bytes > MAX_MESSAGE_BYTES:
        raise ValueError(
            "Invalid RPC payload size: bytes={} allowed=[1,{}]".format(num_bytes, MAX_MESSAGE_BYTES)
        )
    return _loads(_recvall(sock, num_bytes))


class PolicyRpcClient(object):
    def __init__(self, host, port, timeout_sec=600.0):
        self.host = str(host)
        self.port = int(port)
        self._sock = socket.create_connection((self.host, self.port), timeout=float(timeout_sec))
        self._sock.settimeout(float(timeout_sec))

    def request(self, payload):
        send_message(self._sock, payload)
        response = recv_message(self._sock)
        if isinstance(response, dict) and response.get("ok") is False:
            raise RuntimeError(str(response.get("error", "Policy RPC request failed.")))
        if isinstance(response, dict) and response.get("ok") is True and ("data" in response):
            return response["data"]
        return response

    def infer(self, observation, instruction):
        return self.request(
            {"endpoint": "infer", "observation": observation, "instruction": str(instruction)}
        )

    def info(self):
        return self.request({"endpoint": "info"})

    def reset(self, episode_seed=None):
        request = {"endpoint": "reset"}
        if episode_seed is not None:
            request["episode_seed"] = episode_seed
        return self.request(request)

    def close(self):
        self._sock.close()


def configure_image_preprocessing(client, mode, server_info, transport_dtype=None):
    if mode == "server":
        if transport_dtype is not None:
            raise ValueError("--image-transport-dtype requires --image-preprocessing client.")
        client.image_preprocessor = None
        return {"mode": "server"}
    if mode != "client":
        raise ValueError("Unknown image preprocessing mode: {!r}".format(mode))
    try:
        from dreamfree.evaluation.droid_real.image_preprocessing import ClientImagePreprocessor
    except ModuleNotFoundError as exc:
        if exc.name not in (
            "experiments",
            "dreamfree.evaluation.droid_real",
            "dreamfree.evaluation.droid_real.image_preprocessing",
        ):
            raise
        try:
            from image_preprocessing import ClientImagePreprocessor
        except ModuleNotFoundError as standalone_exc:
            raise RuntimeError(
                "Copy image_preprocessing.py beside run_dreamfree_stage2_robot.py on the robot laptop."
            ) from standalone_exc
    selected_dtype = "bf16" if transport_dtype is None else transport_dtype
    client.image_preprocessor = ClientImagePreprocessor(client, server_info, selected_dtype)
    return client.image_preprocessor.metadata


def resolve_image_transport_args(parser, args):
    if args.image_preprocessing == "server":
        if args.image_transport_dtype is not None:
            parser.error("--image-transport-dtype requires --image-preprocessing client.")
    elif args.image_transport_dtype is None:
        args.image_transport_dtype = "bf16"


def prepare_inference_request(client, observation, client_timing=None):
    preprocessor = getattr(client, "image_preprocessor", None)
    if preprocessor is None:
        return observation
    return preprocessor.prepare(observation, client_timing=client_timing)


@contextlib.contextmanager
def delay_keyboard_interrupt():
    """Delay Ctrl+C until a request or response has completed."""
    interrupted = [False]
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        del signum, frame
        interrupted[0] = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted[0]:
            raise KeyboardInterrupt


def _require_loopback_host(host):
    if host == "localhost":
        return
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError(
            "--remote-host must be localhost or a loopback IP because this pickle RPC must only be used through an SSH tunnel; got {!r}.".format(
                host
            )
        )
    if not address.is_loopback:
        raise ValueError(
            "--remote-host must be a loopback IP because this pickle RPC must only be used through an SSH tunnel; got {!r}.".format(
                host
            )
        )


def _copy_camera_rgb(image, mode="contiguous"):
    if mode == "contiguous":
        return np.ascontiguousarray(image[..., :3][..., ::-1])
    if mode != "channels":
        raise ValueError("Unknown camera RGB copy mode: {!r}.".format(mode))
    result = np.empty(image.shape[:2] + (3,), dtype=image.dtype)
    result[..., 0] = image[..., 2]
    result[..., 1] = image[..., 1]
    result[..., 2] = image[..., 0]
    return result


def _camera_image(image_observations, camera_id, rgb_copy="contiguous"):
    matches = [
        key
        for key in image_observations
        if str(camera_id) in str(key) and "left" in str(key).lower()
    ]
    if len(matches) != 1:
        raise KeyError(
            "Expected exactly one left stereo image for camera ID {!r}, found keys={}. Check the IDs with ZED_Explorer.".format(
                camera_id, matches
            )
        )
    image = np.asarray(image_observations[matches[0]])
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(
            "Camera {!r} image must have shape [H,W,>=3], got {}.".format(camera_id, image.shape)
        )
    return _copy_camera_rgb(image, rgb_copy)


def extract_droid_observation(args, obs_dict, client_timing=None):
    if client_timing is not None:
        client_timing.clear()
    with record_client_time(client_timing, "client_preprocess_seconds"):
        camera_ids = [args.left_camera_id, args.right_camera_id, args.wrist_camera_id]
        if len(set(camera_ids)) != 3:
            raise ValueError("Left, right, and wrist camera IDs must be distinct.")
        if not isinstance(obs_dict, Mapping):
            raise TypeError("DROID observation must be a mapping.")
        image_observations = obs_dict.get("image")
        if not isinstance(image_observations, Mapping):
            raise KeyError("DROID observation is missing mapping key `image`.")
        rgb_copy = getattr(args, "camera_rgb_copy", "contiguous")
        left_image = _camera_image(image_observations, args.left_camera_id, rgb_copy)
        right_image = _camera_image(image_observations, args.right_camera_id, rgb_copy)
        wrist_image = _camera_image(image_observations, args.wrist_camera_id, rgb_copy)
        robot_state = obs_dict.get("robot_state")
        if not isinstance(robot_state, Mapping):
            raise KeyError("DROID observation is missing mapping key `robot_state`.")
        joint_position = np.asarray(robot_state.get("joint_positions"), dtype=np.float32).reshape(
            -1
        )
        if joint_position.shape != (7,) or not np.isfinite(joint_position).all():
            raise ValueError(
                "robot_state.joint_positions must be 7 finite values, got shape={}.".format(
                    joint_position.shape
                )
            )
        gripper_position = np.asarray(
            [robot_state.get("gripper_position")], dtype=np.float32
        ).reshape(-1)
        if gripper_position.shape != (1,) or not np.isfinite(gripper_position).all():
            raise ValueError(
                "robot_state.gripper_position must be one finite value, got shape={}.".format(
                    gripper_position.shape
                )
            )
        request = {
            "observation/exterior_image_1_left": left_image,
            "observation/exterior_image_2_left": right_image,
            "observation/wrist_image_left": wrist_image,
            "observation/joint_position": joint_position,
            "observation/gripper_position": gripper_position,
        }
        views = {"left": left_image, "right": right_image, "wrist": wrist_image}
        return (request, views)


def validate_server_info(
    info,
    expected_checkpoint,
    expected_action_horizon,
    open_loop_horizon,
    expected_policy_runtime=NATIVE_POLICY_RUNTIME,
    expected_action_noise_seed=None,
    expected_checkpoint_type=None,
):
    if not isinstance(info, Mapping):
        raise TypeError("Policy server info response must be a mapping.")
    expected_server_name = SERVER_NAME_BY_POLICY_RUNTIME.get(expected_policy_runtime)
    if expected_server_name is None:
        raise ValueError(
            "Unsupported expected policy runtime: {!r}.".format(expected_policy_runtime)
        )
    expect_random_action_noise = expected_action_noise_seed is not None
    expected = {
        "server": expected_server_name,
        "rpc_protocol_version": RPC_PROTOCOL_VERSION,
        "serialization": "pickle_protocol_4",
        "transport_security": "localhost_only_use_ssh_tunnel",
        "policy_runtime": expected_policy_runtime,
        "evaluation_mode": expected_policy_runtime,
        "random_action_noise": expect_random_action_noise,
        "action_noise_seed": expected_action_noise_seed,
        "requires_episode_seed_on_reset": expect_random_action_noise,
        "load_text_encoder": True,
        "action_space": "joint_position",
        "action_representation": "absolute",
        "gripper_action_space": "position",
        "gripper_prediction_representation": GRIPPER_PREDICTION_REPRESENTATION,
        "gripper_binarization": GRIPPER_BINARIZATION,
        "action_dim": 8,
        "control_frequency_hz": DROID_CONTROL_FREQUENCY_HZ,
        "action_horizon": int(expected_action_horizon),
        "replan_steps": int(open_loop_horizon),
    }
    if expected_checkpoint_type is not None:
        expected["checkpoint_type"] = str(expected_checkpoint_type)
    mismatches = []
    for key, expected_value in expected.items():
        if info.get(key) != expected_value:
            mismatches.append(
                "{}: expected {!r}, got {!r}".format(key, expected_value, info.get(key))
            )
    optimization = info.get("optimization_contract")
    if not isinstance(optimization, Mapping):
        mismatches.append("Missing optimized inference contract.")
    else:
        for key, expected_value in {
            "attention_policy": "sdpa_fused_no_math",
            "action_cross_kv_cache": True,
            "compile_dit": True,
            "warmup_complete": True,
        }.items():
            if optimization.get(key) != expected_value:
                mismatches.append(
                    "optimization_contract.{} must be {!r}".format(key, expected_value)
                )
    server_checkpoint = info.get("checkpoint_path")
    if not isinstance(server_checkpoint, str) or os.path.realpath(
        server_checkpoint
    ) != os.path.realpath(expected_checkpoint):
        mismatches.append(
            "checkpoint_path: expected {!r}, got {!r}".format(
                os.path.realpath(expected_checkpoint), server_checkpoint
            )
        )
    server_keys = info.get("observation_keys")
    if not isinstance(server_keys, (list, tuple)) or tuple(server_keys) != OBSERVATION_KEYS:
        mismatches.append(
            "observation_keys: expected {!r}, got {!r}".format(OBSERVATION_KEYS, server_keys)
        )
    if mismatches:
        raise RuntimeError(
            "Policy server contract validation failed:\n  - " + "\n  - ".join(mismatches)
        )


def validate_reset_response(response, episode_seed):
    if not isinstance(response, Mapping):
        raise TypeError("Policy reset response must be a mapping.")
    expected = {"status": "reset", "episode_seed": episode_seed}
    mismatches = []
    for key, expected_value in expected.items():
        if response.get(key) != expected_value:
            mismatches.append(
                "{}: expected {!r}, got {!r}".format(key, expected_value, response.get(key))
            )
    derived_seed = response.get("action_noise_derived_seed")
    if episode_seed is None:
        if derived_seed is not None:
            mismatches.append(
                "action_noise_derived_seed: expected None, got {!r}".format(derived_seed)
            )
    elif (
        isinstance(derived_seed, bool)
        or not isinstance(derived_seed, int)
        or derived_seed < 0
        or (derived_seed > MAX_ACTION_NOISE_SEED)
    ):
        mismatches.append(
            "action_noise_derived_seed: expected an integer in [0,{}], got {!r}".format(
                MAX_ACTION_NOISE_SEED, derived_seed
            )
        )
    if mismatches:
        raise RuntimeError(
            "Policy reset contract validation failed:\n  - " + "\n  - ".join(mismatches)
        )


def coerce_action_chunk(result, expected_horizon):
    if not isinstance(result, Mapping) or "actions" not in result:
        raise TypeError("Policy inference response must contain an `actions` field.")
    actions = np.asarray(result["actions"], dtype=np.float32)
    expected_shape = (int(expected_horizon), 8)
    if actions.shape != expected_shape:
        raise ValueError(
            "Policy action chunk must have shape {}, got {}.".format(expected_shape, actions.shape)
        )
    if not np.isfinite(actions).all():
        raise ValueError("Policy action chunk contains NaN or infinite values.")
    return actions


def print_predicted_action_chunk(
    actions, open_loop_horizon, timestep=None, action_safety_mode=DEFAULT_ACTION_SAFETY_MODE
):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 8:
        raise ValueError("Action chunk must have shape [T,8].")
    open_loop_horizon = int(open_loop_horizon)
    if open_loop_horizon <= 0 or open_loop_horizon > actions.shape[0]:
        raise ValueError(
            "open_loop_horizon must be in [1,{}], got {}.".format(
                actions.shape[0], open_loop_horizon
            )
        )
    location = "observe-only inference"
    if timestep is not None:
        location = "rollout timestep {}".format(int(timestep))
    if action_safety_mode == DEFAULT_ACTION_SAFETY_MODE:
        description = "Safety-validated predicted action chunk"
        handling = ""
    elif action_safety_mode == "clip":
        description = "Raw predicted action chunk"
        handling = "; joint targets will be clipped to position and step-rate bounds at execution"
    elif action_safety_mode == "none":
        description = "Raw predicted action chunk"
        handling = "; joint targets will be executed without motion safety checks or clipping"
    else:
        raise ValueError("Unknown action safety mode: {!r}.".format(action_safety_mode))
    print(
        "{} at {}: shape={}, executable rows=0..{}{}".format(
            description, location, actions.shape, open_loop_horizon - 1, handling
        )
    )
    print("Columns: joint_1 joint_2 joint_3 joint_4 joint_5 joint_6 joint_7 gripper_raw")
    print(
        np.array2string(
            actions,
            precision=6,
            suppress_small=False,
            threshold=actions.size,
            max_line_width=200,
            formatter={"float_kind": lambda value: "{:.6f}".format(value)},
        )
    )


def _joint_safety_limits(joint_position_margin_rad, joint_rate_scale):
    margin = float(joint_position_margin_rad)
    rate_scale = float(joint_rate_scale)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("--joint-position-margin-rad must be finite and non-negative.")
    if not np.isfinite(rate_scale) or rate_scale <= 0.0 or rate_scale > 1.0:
        raise ValueError("--joint-rate-scale must be finite and in (0,1].")
    lower = PANDA_JOINT_MIN_RAD + margin
    upper = PANDA_JOINT_MAX_RAD - margin
    if np.any(lower >= upper):
        raise ValueError(
            "--joint-position-margin-rad={} leaves no valid range for at least one Panda joint.".format(
                margin
            )
        )
    allowed_delta = rate_scale * PANDA_JOINT_VELOCITY_MAX_RAD_S / float(DROID_CONTROL_FREQUENCY_HZ)
    return (lower, upper, allowed_delta)


def validate_action_prefix(
    actions,
    current_joint_position,
    num_actions,
    joint_position_margin_rad=0.05,
    joint_rate_scale=0.8,
):
    actions = np.asarray(actions, dtype=np.float32)
    current = np.asarray(current_joint_position, dtype=np.float32).reshape(-1)
    num_actions = int(num_actions)
    rate_scale = float(joint_rate_scale)
    if actions.ndim != 2 or actions.shape[1] != 8:
        raise ValueError("actions must have shape [T,8], got {}.".format(actions.shape))
    if current.shape != (7,) or not np.isfinite(current).all():
        raise ValueError("current_joint_position must contain 7 finite values.")
    if num_actions <= 0 or num_actions > actions.shape[0]:
        raise ValueError(
            "num_actions must be in [1,{}], got {}.".format(actions.shape[0], num_actions)
        )
    (lower, upper, allowed_delta) = _joint_safety_limits(
        joint_position_margin_rad, joint_rate_scale
    )
    joint_targets = actions[:num_actions, :7]
    invalid_position = np.argwhere(
        np.logical_or(joint_targets < lower[None, :], joint_targets > upper[None, :])
    )
    if invalid_position.size:
        (action_index, joint_index) = invalid_position[0]
        raise UnsafeActionError(
            "Unsafe Panda joint target at action {} joint {}: target={:.6f} allowed=[{:.6f},{:.6f}].".format(
                int(action_index),
                int(joint_index),
                float(joint_targets[action_index, joint_index]),
                float(lower[joint_index]),
                float(upper[joint_index]),
            )
        )
    previous_and_targets = np.concatenate([current.reshape(1, 7), joint_targets], axis=0)
    deltas = np.abs(np.diff(previous_and_targets, axis=0))
    invalid_rate = np.argwhere(deltas > allowed_delta[None, :])
    if invalid_rate.size:
        (action_index, joint_index) = invalid_rate[0]
        raise UnsafeActionError(
            "Unsafe Panda joint step at action {} joint {}: delta={:.6f} rad exceeds {:.6f} rad at {} Hz (rate scale {}).".format(
                int(action_index),
                int(joint_index),
                float(deltas[action_index, joint_index]),
                float(allowed_delta[joint_index]),
                DROID_CONTROL_FREQUENCY_HZ,
                rate_scale,
            )
        )


class ActionChunkBuffer(object):
    def __init__(self, open_loop_horizon):
        self.open_loop_horizon = int(open_loop_horizon)
        if self.open_loop_horizon <= 0:
            raise ValueError("open_loop_horizon must be positive.")
        self.reset()

    def reset(self):
        self._actions = None
        self._index = 0

    def needs_replan(self):
        return self._actions is None or self._index >= self.open_loop_horizon

    def set_chunk(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 8:
            raise ValueError("Action chunk must have shape [T,8].")
        if actions.shape[0] < self.open_loop_horizon:
            raise ValueError(
                "Action chunk length {} is smaller than open-loop horizon {}.".format(
                    actions.shape[0], self.open_loop_horizon
                )
            )
        self._actions = actions
        self._index = 0

    def pop_next(self):
        if self.needs_replan():
            raise RuntimeError("A new policy action chunk is required before execution.")
        action = self._actions[self._index].copy()
        self._index += 1
        return action


class IssuedActionHistoryBuffer(object):
    def __init__(self, horizon, action_dim=8):
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        if self.horizon <= 0 or self.action_dim <= 0:
            raise ValueError("Issued-action history horizon and action_dim must be positive.")
        self.reset()

    def reset(self):
        self._commands = []

    def record_issued_action(self, action):
        command = np.asarray(action, dtype=np.float32).reshape(-1)
        if command.shape != (self.action_dim,) or not np.isfinite(command).all():
            raise ValueError(
                "Issued action must contain {} finite values, got {}.".format(
                    self.action_dim, command.shape
                )
            )
        if self.action_dim == 8 and command[7] not in (0.0, 1.0):
            raise ValueError(
                "DROID issued-action history requires a binary gripper command, got {}.".format(
                    float(command[7])
                )
            )
        self._commands.append(command.copy())
        self._commands = self._commands[-self.horizon :]

    def add_to_observation(self, observation):
        history = np.zeros((self.horizon, self.action_dim), dtype=np.float32)
        is_pad = np.ones((self.horizon,), dtype=np.bool_)
        if self._commands:
            count = len(self._commands)
            history[-count:] = np.stack(self._commands, axis=0)
            is_pad[-count:] = False
        request = dict(observation)
        request[ISSUED_ACTION_HISTORY_KEY] = history
        request[ISSUED_ACTION_HISTORY_IS_PAD_KEY] = is_pad
        return request


def _resize_nearest(image, output_height, output_width):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Preview image must have shape [H,W,3].")
    row_indices = np.linspace(0, image.shape[0] - 1, output_height).astype(np.int64)
    col_indices = np.linspace(0, image.shape[1] - 1, output_width).astype(np.int64)
    return np.ascontiguousarray(image[row_indices][:, col_indices])


def compose_camera_preview(views):
    images = [_resize_nearest(views[name], 180, 320) for name in ("left", "wrist", "right")]
    return np.concatenate(images, axis=1)


def _import_imageio():
    try:
        import imageio.v2 as imageio
    except (ImportError, AttributeError):
        try:
            import imageio
        except ImportError as exc:
            raise RuntimeError(
                "imageio is required to save the camera preview and rollout video."
            ) from exc
    return imageio


def _preview_for_imageio(preview):
    frame = np.asarray(preview, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Preview image must have shape [H,W,3].")
    return np.ascontiguousarray(frame)


def validate_server_inference_seconds(value):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, numbers.Real)
        or (not np.isfinite(value))
        or (value < 0)
    ):
        raise ValueError(
            "Server-duration recording requires a finite, non-negative numeric duration; got {!r}. Update the policy server or explicitly use --no-server-timing.".format(
                value
            )
        )
    return float(value)


def server_inference_seconds_from_response(result, enabled=True):
    if not enabled:
        return None
    timing = result.get("server_timing") if isinstance(result, Mapping) else None
    if not isinstance(timing, Mapping) or "infer_ms" not in timing:
        raise ValueError(
            "Server-duration recording requires response['server_timing']['infer_ms']. Update the policy server or explicitly use --no-server-timing."
        )
    return validate_server_inference_seconds(timing["infer_ms"]) / 1000.0


def format_inference_timing(inference_seconds, server_inference_seconds):
    server = (
        "disabled"
        if server_inference_seconds is None
        else "{:.3f}s".format(server_inference_seconds)
    )
    return "client {:.3f}s, server {}".format(inference_seconds, server)


@contextlib.contextmanager
def record_client_time(timing, field):
    """Accumulate computation time without spanning camera reads, RPC, or logging."""
    if timing is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        timing[field] = timing.get(field, 0.0) + (time.perf_counter() - start)


def validate_client_duration(value, field):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, numbers.Real)
        or (not np.isfinite(value))
        or (value < 0)
    ):
        raise ValueError(
            "Client timing requires a finite, non-negative duration for {}; got {!r}.".format(
                field, value
            )
        )
    return float(value)


class RolloutLogger(object):
    def __init__(self, output_dir, metadata, write_video):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.abspath(os.path.join(output_dir, timestamp))
        if os.path.exists(self.run_dir):
            raise FileExistsError("Output directory already exists: {}".format(self.run_dir))
        os.makedirs(self.run_dir)
        self.metadata = dict(metadata)
        self.metadata["created_at"] = datetime.datetime.now().isoformat()
        self.predicted_chunks = []
        self.executed_actions = []
        self.observed_joint_positions = []
        self.observed_gripper_positions = []
        self.inference_seconds = []
        self.server_inference_seconds = []
        self.server_timing_enabled = self.metadata.setdefault("server_timing_enabled", True)
        self.client_timing_enabled = self.metadata.setdefault("client_timing_enabled", False)
        self.client_preprocess_seconds = []
        self.client_postprocess_seconds = []
        self.action_postprocess_seconds = []
        self.executed_action_chunk_indices = []
        if self.client_timing_enabled:
            self.metadata["client_timing_scope"] = {
                "client_preprocess_seconds": "query observation extraction, image crop/resize and request preparation; excludes camera acquisition and preprocessing parity validation",
                "client_postprocess_seconds": "response conversion, validation, and configured chunk safety checks",
                "action_postprocess_seconds": "per executed action: gripper conversion and configured clipping; excludes env.step",
            }
            self.metadata["policy_compute_seconds_formula"] = (
                "client_preprocess_seconds + server_inference_seconds + client_postprocess_seconds"
            )
            self.metadata["policy_compute_seconds_available"] = self.server_timing_enabled
            self.metadata["policy_compute_timing_scope"] = (
                "accepted inference attempt to validated CPU action chunk; excludes transport, camera acquisition, preview/logging, safety-rejected attempts, and per-action execution processing"
            )
        self.metadata["server_timing_scope"] = "policy_call"
        if not self.server_timing_enabled:
            print("Server-duration recording disabled by --no-server-timing.")
        self.safety_rejections = []
        self._video_writer = None
        if write_video:
            imageio = _import_imageio()
            video_path = os.path.join(self.run_dir, "camera_views.mp4")
            try:
                self._video_writer = imageio.get_writer(
                    video_path, fps=DROID_CONTROL_FREQUENCY_HZ, codec="libx264"
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to create rollout video writer at {}: {}".format(video_path, exc)
                )

    def save_preview(self, preview):
        preview_path = os.path.join(self.run_dir, "camera_preview.png")
        _import_imageio().imwrite(preview_path, _preview_for_imageio(preview))
        return preview_path

    def append_observation(self, request, preview):
        self.observed_joint_positions.append(
            np.asarray(request["observation/joint_position"], dtype=np.float32).copy()
        )
        self.observed_gripper_positions.append(
            np.asarray(request["observation/gripper_position"], dtype=np.float32).copy()
        )
        if self._video_writer is not None:
            self._video_writer.append_data(_preview_for_imageio(preview))

    def append_prediction(
        self, actions, inference_seconds, server_inference_seconds=None, client_timing=None
    ):
        actions = np.asarray(actions, dtype=np.float32).copy()
        inference_seconds = float(inference_seconds)
        if self.server_timing_enabled:
            server_inference_seconds = validate_server_inference_seconds(server_inference_seconds)
        if self.client_timing_enabled:
            if not isinstance(client_timing, Mapping):
                raise ValueError(
                    "Client timing is enabled: each prediction requires preprocessing and postprocessing durations."
                )
            preprocess = validate_client_duration(
                client_timing.get("client_preprocess_seconds"), "client_preprocess_seconds"
            )
            postprocess = validate_client_duration(
                client_timing.get("client_postprocess_seconds"), "client_postprocess_seconds"
            )
        self.predicted_chunks.append(actions)
        self.inference_seconds.append(inference_seconds)
        if self.server_timing_enabled:
            self.server_inference_seconds.append(server_inference_seconds)
        if self.client_timing_enabled:
            self.client_preprocess_seconds.append(preprocess)
            self.client_postprocess_seconds.append(postprocess)

    def record_safety_rejection(self, timestep, candidate_number, error):
        self.safety_rejections.append(
            {
                "timestep": None if timestep is None else int(timestep),
                "candidate_number": int(candidate_number),
                "error": str(error),
            }
        )

    def append_action(self, action, postprocess_seconds=None):
        action = np.asarray(action, dtype=np.float32).copy()
        if self.client_timing_enabled:
            postprocess_seconds = validate_client_duration(
                postprocess_seconds, "action_postprocess_seconds"
            )
            if not self.predicted_chunks:
                raise ValueError("Cannot record an executed action without a predicted chunk.")
        self.executed_actions.append(action)
        if self.client_timing_enabled:
            self.action_postprocess_seconds.append(postprocess_seconds)
            self.executed_action_chunk_indices.append(len(self.predicted_chunks) - 1)

    def close(self, status, error=None):
        if self._video_writer is not None:
            self._video_writer.close()
            self._video_writer = None
        self.metadata["status"] = str(status)
        self.metadata["error"] = None if error is None else str(error)
        self.metadata["num_observations"] = len(self.observed_joint_positions)
        self.metadata["num_executed_actions"] = len(self.executed_actions)
        self.metadata["num_safety_rejections"] = len(self.safety_rejections)
        self.metadata["safety_rejections"] = list(self.safety_rejections)
        with open(os.path.join(self.run_dir, "metadata.json"), "w") as stream:
            json.dump(self.metadata, stream, indent=2, sort_keys=True)
            stream.write("\n")
        expected_horizon = int(self.metadata["expected_action_horizon"])
        predicted = np.asarray(self.predicted_chunks, dtype=np.float32).reshape(
            -1, expected_horizon, 8
        )
        executed = np.asarray(self.executed_actions, dtype=np.float32).reshape(-1, 8)
        joint_positions = np.asarray(self.observed_joint_positions, dtype=np.float32).reshape(-1, 7)
        gripper_positions = np.asarray(self.observed_gripper_positions, dtype=np.float32).reshape(
            -1, 1
        )
        timing_arrays = {}
        if self.server_timing_enabled:
            timing_arrays["server_inference_seconds"] = np.asarray(
                self.server_inference_seconds, dtype=np.float64
            )
        if self.client_timing_enabled:
            timing_arrays["client_preprocess_seconds"] = np.asarray(
                self.client_preprocess_seconds, dtype=np.float64
            )
            timing_arrays["client_postprocess_seconds"] = np.asarray(
                self.client_postprocess_seconds, dtype=np.float64
            )
            timing_arrays["action_postprocess_seconds"] = np.asarray(
                self.action_postprocess_seconds, dtype=np.float64
            )
            timing_arrays["executed_action_chunk_indices"] = np.asarray(
                self.executed_action_chunk_indices, dtype=np.int64
            )
            if self.server_timing_enabled:
                timing_arrays["policy_compute_seconds"] = (
                    timing_arrays["client_preprocess_seconds"]
                    + timing_arrays["server_inference_seconds"]
                    + timing_arrays["client_postprocess_seconds"]
                )
        np.savez_compressed(
            os.path.join(self.run_dir, "rollout_data.npz"),
            predicted_action_chunks=predicted,
            executed_actions=executed,
            observed_joint_positions=joint_positions,
            observed_gripper_positions=gripper_positions,
            inference_seconds=np.asarray(self.inference_seconds, dtype=np.float32),
            **timing_arrays,
        )


def _bounded_seed(value):
    try:
        seed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Expected an integer seed, got {!r}.".format(value))
    if seed < 0 or seed > MAX_ACTION_NOISE_SEED:
        raise argparse.ArgumentTypeError(
            "Seed must be in [0,{}], got {}.".format(MAX_ACTION_NOISE_SEED, seed)
        )
    return seed


def infer_action_chunk(
    client,
    env,
    args,
    instruction,
    initial_request,
    logger,
    server_info,
    issued_action_history=None,
    timestep=None,
    client_timing=None,
):
    request = initial_request
    if args.action_safety_mode == DEFAULT_ACTION_SAFETY_MODE:
        total_candidates = int(args.max_safety_regenerations) + 1
    elif args.action_safety_mode in ("clip", "none"):
        total_candidates = 1
    else:
        raise ValueError("Unknown action safety mode: {!r}.".format(args.action_safety_mode))
    fixed_policy_seed = (
        not bool(server_info.get("random_action_noise", False))
        and server_info.get("policy_seed") is not None
    )
    for candidate_index in range(total_candidates):
        with record_client_time(client_timing, "client_preprocess_seconds"):
            inference_request = request
            if issued_action_history is not None:
                inference_request = issued_action_history.add_to_observation(request)
        inference_request = prepare_inference_request(client, inference_request, client_timing)
        inference_start = time.perf_counter()
        with delay_keyboard_interrupt():
            result = client.infer(inference_request, instruction)
        inference_seconds = time.perf_counter() - inference_start
        server_seconds = server_inference_seconds_from_response(result, args.server_timing)
        with record_client_time(client_timing, "client_postprocess_seconds"):
            actions = coerce_action_chunk(result, args.expected_action_horizon)
        if args.action_safety_mode in ("clip", "none"):
            return (actions, inference_seconds, server_seconds)
        try:
            with record_client_time(client_timing, "client_postprocess_seconds"):
                validate_action_prefix(
                    actions,
                    request["observation/joint_position"],
                    args.open_loop_horizon,
                    joint_position_margin_rad=args.joint_position_margin_rad,
                    joint_rate_scale=args.joint_rate_scale,
                )
        except UnsafeActionError as exc:
            candidate_number = candidate_index + 1
            logger.record_safety_rejection(timestep, candidate_number, exc)
            location = (
                "observe-only inference"
                if timestep is None
                else "rollout timestep {}".format(int(timestep))
            )
            print(
                "Safety rejected candidate {}/{} at {}: {}".format(
                    candidate_number, total_candidates, location, exc
                )
            )
            remaining_regenerations = total_candidates - candidate_number
            if remaining_regenerations == 0:
                if args.max_safety_regenerations == 0:
                    raise
                raise UnsafeActionError(
                    "Safety regeneration exhausted after {} candidates at {}. Last failure: {}".format(
                        total_candidates, location, exc
                    )
                ) from exc
            if fixed_policy_seed:
                raise RuntimeError(
                    "Cannot regenerate an unsafe action because the policy server uses fixed policy_seed={!r}; repeated inference cannot resample action noise. First failure: {}".format(
                        server_info.get("policy_seed"), exc
                    )
                ) from exc
            print(
                "Capturing a fresh observation and regenerating; {} attempt(s) remain.".format(
                    remaining_regenerations
                )
            )
            (request, views) = extract_droid_observation(
                args, env.get_observation(), client_timing=client_timing
            )
            logger.append_observation(request, compose_camera_preview(views))
            continue
        return (actions, inference_seconds, server_seconds)
    raise AssertionError("Action inference loop ended without a result.")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-server-timing",
        dest="server_timing",
        action="store_false",
        default=True,
        help="Explicitly disable required server-duration recording explicitly.",
    )
    parser.add_argument("--remote-host", default="127.0.0.1")
    parser.add_argument("--remote-port", type=int, default=6100)
    parser.add_argument(
        "--image-preprocessing",
        choices=("client", "server"),
        default="client",
        help="Crop/resize on CPU before transmission (requires PyTorch), or explicitly use the original server path.",
    )
    parser.add_argument(
        "--image-transport-dtype",
        choices=("fp32", "bf16", "uint8"),
        default=None,
        help="Client image transport: fp32 RGB, bf16 after normalization (default), or lossy uint8 RGB.",
    )
    parser.add_argument("--left-camera-id", required=True)
    parser.add_argument("--right-camera-id", required=True)
    parser.add_argument("--wrist-camera-id", required=True)
    parser.add_argument("--expected-checkpoint", required=True)
    parser.add_argument(
        "--expected-policy-runtime",
        choices=SUPPORTED_POLICY_RUNTIMES,
        default=NATIVE_POLICY_RUNTIME,
    )
    parser.add_argument(
        "--expected-checkpoint-type",
        choices=SUPPORTED_CHECKPOINT_TYPES,
        default=None,
        help="Fail if the policy server reports a different Stage 2 checkpoint type.",
    )
    parser.add_argument("--expected-action-noise-seed", type=_bounded_seed, default=None)
    parser.add_argument("--episode-seed", type=_bounded_seed, default=None)
    parser.add_argument("--expected-action-horizon", type=int, default=32)
    parser.add_argument("--open-loop-horizon", type=int, default=8)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--max-timesteps", type=int, default=600)
    parser.add_argument("--rpc-timeout-sec", type=float, default=600.0)
    parser.add_argument(
        "--camera-rgb-copy",
        choices=("contiguous", "channels"),
        default="contiguous",
        help="Select the exact BGR/BGRA-to-RGB copy implementation.",
    )
    parser.add_argument("--output-dir", default="./outputs/evaluation/droid_real")
    parser.add_argument("--observe-only", action="store_true")
    parser.add_argument(
        "--print-predicted-actions",
        action="store_true",
        help="Print every raw policy action chunk and how its executable prefix is handled. Use with --observe-only to inspect one prediction without robot motion.",
    )
    parser.add_argument("--reset-robot", action="store_true")
    parser.add_argument("--joint-position-margin-rad", type=float, default=0.05)
    parser.add_argument("--joint-rate-scale", type=float, default=0.8)
    parser.add_argument(
        "--action-safety-mode",
        choices=ACTION_SAFETY_MODES,
        default=DEFAULT_ACTION_SAFETY_MODE,
        help="reject-and-regenerate validates joint-position and step-rate bounds; clip clamps joint targets to both bounds relative to measured joints at execution; none executes joint targets without motion safety checks or clipping. All modes validate action shape/finite values and binarize the gripper.",
    )
    parser.add_argument(
        "--gripper-close-threshold",
        type=float,
        default=DEFAULT_DROID_GRIPPER_CLOSE_THRESHOLD,
        help="Normalized raw gripper threshold: values strictly above it close the gripper; all other values open it.",
    )
    parser.add_argument(
        "--max-safety-regenerations",
        type=int,
        default=DEFAULT_MAX_SAFETY_REGENERATIONS,
        help="Maximum new action chunks to generate after a joint-safety rejection. Only applies to reject-and-regenerate. The default allows four total candidates; use 0 to fail immediately.",
    )
    args = parser.parse_args(argv)
    resolve_image_transport_args(parser, args)
    if args.remote_port <= 0 or args.remote_port > 65535:
        parser.error("--remote-port must be in [1,65535].")
    if args.expected_action_horizon <= 0:
        parser.error("--expected-action-horizon must be positive.")
    if args.open_loop_horizon <= 0:
        parser.error("--open-loop-horizon must be positive.")
    if args.open_loop_horizon > args.expected_action_horizon:
        parser.error("--open-loop-horizon cannot exceed --expected-action-horizon.")
    if args.max_timesteps <= 0:
        parser.error("--max-timesteps must be positive.")
    if args.rpc_timeout_sec <= 0:
        parser.error("--rpc-timeout-sec must be positive.")
    try:
        _joint_safety_limits(args.joint_position_margin_rad, args.joint_rate_scale)
    except ValueError as exc:
        parser.error(str(exc))
    if (
        not np.isfinite(args.gripper_close_threshold)
        or args.gripper_close_threshold <= 0.0
        or args.gripper_close_threshold >= 1.0
    ):
        parser.error("--gripper-close-threshold must be finite and in (0,1).")
    if args.max_safety_regenerations < 0:
        parser.error("--max-safety-regenerations must be non-negative.")
    if (args.expected_action_noise_seed is None) != (args.episode_seed is None):
        parser.error("--expected-action-noise-seed and --episode-seed must be supplied together.")
    return args


def canonicalize_droid_gripper_command(
    value, scale, close_threshold=DEFAULT_DROID_GRIPPER_CLOSE_THRESHOLD
):
    try:
        command = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("DROID gripper command must be a finite scalar.") from exc
    if command.size != 1 or not np.isfinite(command).all():
        raise ValueError("DROID gripper command must be a finite scalar, got {!r}.".format(value))
    try:
        command_scale = float(scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("DROID gripper command scale must be finite and positive.") from exc
    if not np.isfinite(command_scale) or command_scale <= 0.0:
        raise ValueError(
            "DROID gripper command scale must be finite and positive, got {!r}.".format(scale)
        )
    try:
        threshold_value = float(close_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("DROID gripper close threshold must be finite and in (0,1).") from exc
    if not np.isfinite(threshold_value) or threshold_value <= 0.0 or threshold_value >= 1.0:
        raise ValueError(
            "DROID gripper close threshold must be finite and in (0,1), got {!r}.".format(
                close_threshold
            )
        )
    threshold = np.float32(threshold_value)
    if threshold <= 0.0 or threshold >= 1.0:
        raise ValueError(
            "DROID gripper close threshold must remain in (0,1) as float32, got {!r}.".format(
                close_threshold
            )
        )
    command_value = np.float32(float(command.reshape(-1)[0]) / command_scale)
    return np.float32(1.0 if command_value > threshold else 0.0)


def _binarize_gripper(action, close_threshold=DEFAULT_DROID_GRIPPER_CLOSE_THRESHOLD):
    action = np.asarray(action, dtype=np.float32).copy()
    action[7] = canonicalize_droid_gripper_command(
        action[7], DROID_GRIPPER_NORMALIZED_COMMAND_SCALE, close_threshold=close_threshold
    )
    return action


def prepare_action_for_execution(
    action,
    current_joint_position=None,
    close_threshold=DEFAULT_DROID_GRIPPER_CLOSE_THRESHOLD,
    action_safety_mode=DEFAULT_ACTION_SAFETY_MODE,
    joint_position_margin_rad=0.05,
    joint_rate_scale=0.8,
):
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (8,) or not np.isfinite(action).all():
        raise ValueError("A robot action must contain 8 finite values.")
    action = _binarize_gripper(action, close_threshold=close_threshold)
    if action_safety_mode in (DEFAULT_ACTION_SAFETY_MODE, "none"):
        return action
    if action_safety_mode != "clip":
        raise ValueError("Unknown action safety mode: {!r}.".format(action_safety_mode))
    current = np.asarray(current_joint_position, dtype=np.float32).reshape(-1)
    if current.shape != (7,) or not np.isfinite(current).all():
        raise ValueError("clip mode requires current_joint_position with 7 finite values.")
    (position_lower, position_upper, allowed_delta) = _joint_safety_limits(
        joint_position_margin_rad, joint_rate_scale
    )
    step_lower = current - allowed_delta
    step_upper = current + allowed_delta
    lower = np.maximum(position_lower, step_lower)
    upper = np.minimum(position_upper, step_upper)
    invalid = np.flatnonzero(lower > upper)
    if invalid.size:
        joint = int(invalid[0])
        raise UnsafeActionError(
            "Cannot clip Panda joint {}: measured={:.6f} rad, position bounds=[{:.6f},{:.6f}], step-rate bounds=[{:.6f},{:.6f}] do not overlap.".format(
                joint,
                float(current[joint]),
                float(position_lower[joint]),
                float(position_upper[joint]),
                float(step_lower[joint]),
                float(step_upper[joint]),
            )
        )
    action[:7] = np.clip(action[:7], lower, upper)
    return action


def main():
    args = _parse_args()
    _require_loopback_host(args.remote_host)
    camera_ids = [args.left_camera_id, args.right_camera_id, args.wrist_camera_id]
    if len(set(camera_ids)) != 3:
        raise ValueError("Left, right, and wrist camera IDs must be distinct.")
    instruction = args.instruction
    if instruction is None:
        instruction = input("Enter the robot instruction: ").strip()
    if not instruction:
        raise ValueError("The robot instruction must be non-empty.")
    print(
        "Connecting to DreamFree policy at {}:{} through the SSH tunnel...".format(
            args.remote_host, args.remote_port
        )
    )
    client = PolicyRpcClient(args.remote_host, args.remote_port, timeout_sec=args.rpc_timeout_sec)
    logger = None
    run_error = None
    run_status = "failed"
    try:
        server_info = client.info()
        validate_server_info(
            server_info,
            expected_checkpoint=args.expected_checkpoint,
            expected_action_horizon=args.expected_action_horizon,
            open_loop_horizon=args.open_loop_horizon,
            expected_policy_runtime=args.expected_policy_runtime,
            expected_checkpoint_type=args.expected_checkpoint_type,
            expected_action_noise_seed=args.expected_action_noise_seed,
        )
        print("Policy server contract validated.")
        image_preprocessing_metadata = configure_image_preprocessing(
            client, args.image_preprocessing, server_info, args.image_transport_dtype
        )
        print("Image preprocessing: {}".format(image_preprocessing_metadata))
        reset_response = client.reset(args.episode_seed)
        validate_reset_response(reset_response, args.episode_seed)
        print("Policy reset contract validated.")
        issued_action_history = None
        if args.expected_policy_runtime in ACTION_HISTORY_POLICY_RUNTIMES:
            issued_action_history = IssuedActionHistoryBuffer(server_info["action_history_horizon"])
        try:
            from droid.robot_env import RobotEnv
        except ImportError as exc:
            raise RuntimeError(
                "Could not import droid.robot_env. Run this file in the DROID conda environment."
            ) from exc
        env = RobotEnv(
            action_space="joint_position", gripper_action_space="position", do_reset=False
        )
        print("Created DROID RobotEnv without moving the robot.")
        if args.reset_robot:
            print("Resetting the Panda to the DROID reset pose.")
            env.reset()
        mode_descriptions = {
            DEFAULT_ACTION_SAFETY_MODE: "Action prefixes must pass Panda joint-position and step-rate checks.",
            "clip": "Joint targets are clipped to Panda position and step-rate bounds relative to measured joints at execution; safety regeneration is disabled.",
            "none": "Joint targets are executed unchanged; motion safety checks, clipping, and safety regeneration are disabled.",
        }
        print(
            "Action safety mode: {}. {}".format(
                args.action_safety_mode, mode_descriptions[args.action_safety_mode]
            )
        )
        first_timing = {}
        (first_request, first_views) = extract_droid_observation(
            args, env.get_observation(), client_timing=first_timing
        )
        first_preview = compose_camera_preview(first_views)
        metadata = {
            "client_timing_enabled": True,
            "server_timing_enabled": args.server_timing,
            "instruction": instruction,
            "observe_only": bool(args.observe_only),
            "print_predicted_actions": bool(args.print_predicted_actions),
            "server_info": dict(server_info),
            "image_preprocessing": image_preprocessing_metadata,
            "camera_rgb_copy": getattr(args, "camera_rgb_copy", "contiguous"),
            "camera_rgb_shapes": {name: list(image.shape) for (name, image) in first_views.items()},
            "camera_rgb_dtypes": {name: str(image.dtype) for (name, image) in first_views.items()},
            "expected_policy_runtime": args.expected_policy_runtime,
            "expected_checkpoint_type": args.expected_checkpoint_type,
            "expected_action_noise_seed": args.expected_action_noise_seed,
            "episode_seed": args.episode_seed,
            "reset_response": dict(reset_response),
            "remote_host": args.remote_host,
            "remote_port": args.remote_port,
            "camera_ids": {
                "left": args.left_camera_id,
                "right": args.right_camera_id,
                "wrist": args.wrist_camera_id,
            },
            "expected_action_horizon": args.expected_action_horizon,
            "open_loop_horizon": args.open_loop_horizon,
            "max_timesteps": args.max_timesteps,
            "action_safety_mode": args.action_safety_mode,
            "joint_position_margin_rad": args.joint_position_margin_rad,
            "joint_rate_scale": args.joint_rate_scale,
            "gripper_close_threshold": args.gripper_close_threshold,
            "max_safety_regenerations": args.max_safety_regenerations,
        }
        logger = RolloutLogger(args.output_dir, metadata, write_video=not args.observe_only)
        preview_path = logger.save_preview(first_preview)
        print("Saved the three-camera preview to {}.".format(preview_path))
        if args.observe_only:
            logger.append_observation(first_request, first_preview)
            (actions, inference_seconds, server_seconds) = infer_action_chunk(
                client,
                env,
                args,
                instruction,
                first_request,
                logger,
                server_info,
                issued_action_history=issued_action_history,
                timestep=None,
                client_timing=first_timing,
            )
            if args.print_predicted_actions:
                print_predicted_action_chunk(
                    actions,
                    args.open_loop_horizon,
                    timestep=None,
                    action_safety_mode=args.action_safety_mode,
                )
            logger.append_prediction(
                actions, inference_seconds, server_seconds, client_timing=first_timing
            )
            if args.action_safety_mode == DEFAULT_ACTION_SAFETY_MODE:
                run_status = "observe_only_validated"
                outcome = "Observe-only validation passed"
            else:
                run_status = "observe_only_{}_ready".format(args.action_safety_mode)
                outcome = "Observe-only inference completed in {} mode; raw predictions were saved without motion safety validation".format(
                    args.action_safety_mode
                )
            print("{}; no env.step call was made. Results: {}".format(outcome, logger.run_dir))
            return
        print(
            "Starting rollout without an interactive confirmation. The rollout will send absolute Panda joint-position targets at {} Hz and close the gripper for raw targets strictly above {}.".format(
                DROID_CONTROL_FREQUENCY_HZ, args.gripper_close_threshold
            )
        )
        action_buffer = ActionChunkBuffer(args.open_loop_horizon)
        pending_observation = (first_request, first_views, first_timing)
        run_status = "completed"
        print("Running rollout. Press Ctrl+C to stop early.")
        for timestep in range(args.max_timesteps):
            step_start = time.time()
            try:
                if pending_observation is None:
                    client_timing = {}
                    (request, views) = extract_droid_observation(
                        args, env.get_observation(), client_timing=client_timing
                    )
                else:
                    (request, views, client_timing) = pending_observation
                    pending_observation = None
                preview = compose_camera_preview(views)
                logger.append_observation(request, preview)
                if action_buffer.needs_replan():
                    (actions, inference_seconds, server_seconds) = infer_action_chunk(
                        client,
                        env,
                        args,
                        instruction,
                        request,
                        logger,
                        server_info,
                        issued_action_history=issued_action_history,
                        timestep=timestep,
                        client_timing=client_timing,
                    )
                    if args.print_predicted_actions:
                        print_predicted_action_chunk(
                            actions,
                            args.open_loop_horizon,
                            timestep=timestep,
                            action_safety_mode=args.action_safety_mode,
                        )
                    action_buffer.set_chunk(actions)
                    logger.append_prediction(
                        actions, inference_seconds, server_seconds, client_timing=client_timing
                    )
                    handling = (
                        "received and safety-validated a"
                        if args.action_safety_mode == DEFAULT_ACTION_SAFETY_MODE
                        else "received a raw"
                    )
                    print(
                        "timestep {}: {} {}-action chunk ({})".format(
                            timestep,
                            handling,
                            args.expected_action_horizon,
                            format_inference_timing(inference_seconds, server_seconds),
                        )
                    )
                predicted_action = action_buffer.pop_next()
                action_timing = {}
                with record_client_time(action_timing, "action_postprocess_seconds"):
                    executed_action = prepare_action_for_execution(
                        predicted_action,
                        current_joint_position=request["observation/joint_position"],
                        close_threshold=args.gripper_close_threshold,
                        action_safety_mode=args.action_safety_mode,
                        joint_position_margin_rad=args.joint_position_margin_rad,
                        joint_rate_scale=args.joint_rate_scale,
                    )
                env.step(executed_action)
                if issued_action_history is not None:
                    issued_action_history.record_issued_action(executed_action)
                logger.append_action(
                    executed_action, postprocess_seconds=action_timing["action_postprocess_seconds"]
                )
                elapsed = time.time() - step_start
                remaining = 1.0 / DROID_CONTROL_FREQUENCY_HZ - elapsed
                if remaining > 0:
                    time.sleep(remaining)
            except KeyboardInterrupt:
                run_status = "interrupted"
                print("Stopping rollout after timestep {}.".format(timestep))
                break
        print("Rollout results: {}".format(logger.run_dir))
    except Exception as exc:
        run_status = "failed"
        run_error = exc
        raise
    finally:
        if logger is not None:
            logger.close(run_status, error=run_error)
        client.close()


if __name__ == "__main__":
    main()
