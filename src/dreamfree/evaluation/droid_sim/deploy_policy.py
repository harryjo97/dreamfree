"""DROID sim deployment wrapper for DreamFree Stage 2 student-native checkpoints."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

import dreamfree.models.policy_loading as dreamfree_deploy
from dreamfree.data.lerobot.processors.robot import DreamFreeProcessor
from dreamfree.data.lerobot.transforms.action_state_merger import ConcatLeftAlign
from dreamfree.data.lerobot.utils.normalizer import load_dataset_stats_from_json
from dreamfree.evaluation import checkpoints, options
from dreamfree.evaluation.action_protocol import extract_supplied_action_history
from dreamfree.evaluation.droid_real.image_preprocessing import (
    PREPROCESSING_KEY,
    TRANSPORT_FORMATS,
    processed_frames,
)
from dreamfree.evaluation.droid_real.image_preprocessing import (
    center_crop_resize_frame as _center_crop_resize_frame,
)
from dreamfree.evaluation.droid_real.image_preprocessing import (
    rgb_to_chw_frame as _rgb_to_chw_frame,
)
from dreamfree.evaluation.runtime import PolicyRuntime
from dreamfree.models import action_history
from dreamfree.models.compile import compile_dit_blocks
from dreamfree.models.conditioning import TeacherStateActionConditioner
from dreamfree.models.wan22.action_dit import ActionDiT
from dreamfree.models.wan22.helpers.distributed_load import (
    load_safetensors_rank0 as load_safetensors,
)

logger = logging.getLogger(__name__)
DREAMFREE_STAGE2_CHECKPOINT_TYPE = checkpoints.DREAMFREE_STAGE2_CHECKPOINT_TYPE
DREAMFREE_STAGE2_CHECKPOINT_VERSION = checkpoints.DREAMFREE_STAGE2_CHECKPOINT_VERSION
DREAMFREE_POLICY_RUNTIME = checkpoints.DREAMFREE_POLICY_RUNTIME
DREAMFREE_STAGE2_REQUIRED_FILES = checkpoints.DREAMFREE_STAGE2_REQUIRED_FILES
SUPPORTED_CHECKPOINT_TYPES = {DREAMFREE_STAGE2_CHECKPOINT_TYPE, action_history.CHECKPOINT_TYPE}
DROID_DATASET_ROOT = Path("./assets")
DROID_DATASET_ALIASES = {"droid", "droid_lerobot"}
DROID_GRID_HEIGHT = 352
DROID_GRID_WIDTH = 640
DROID_PER_CAMERA_HEIGHT = DROID_GRID_HEIGHT // 2
DROID_PER_CAMERA_WIDTH = DROID_GRID_WIDTH // 2
DROID_ACTION_DIM = 8
DROID_PROPRIO_DIM = 8
DROID_CONCAT_MULTI_CAMERA = "droid_grid"
DROID_JOINT_POSITION_KEY = "joint_position"
DROID_GRIPPER_POSITION_KEY = "gripper_position"
DROID_RELATIVE_STATS_FILE = "relative_stats_dreamzero.json"
DROID_EVAL_VIEW_KEYS = (
    "observation/exterior_image_1_left",
    "observation/exterior_image_2_left",
    "observation/wrist_image_left",
)
MAX_ACTION_NOISE_SEED = (1 << 63) - 1


def _is_none_like(value: Any) -> bool:
    return dreamfree_deploy._is_none_like(value)


def _parse_bool(value: Any) -> bool:
    return dreamfree_deploy._parse_bool(value)


def _require_dict(value: Any, *, label: str) -> dict[str, Any]:
    return checkpoints.require_dict(value, label=label)


def _check_equal(mismatches: list[str], name: str, actual: Any, expected: Any) -> None:
    checkpoints.check_equal(mismatches, name, actual, expected)


def _json_load(path: Path) -> dict[str, Any]:
    return checkpoints.read_native_checkpoint_metadata(path)


def _resolve_checkpoint_path(checkpoint_path: str) -> Path:
    from dreamfree.models.release import resolve_policy_checkpoint

    return resolve_policy_checkpoint(checkpoint_path)


def _normalize_droid_dataset_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def _inspect_checkpoint_path(checkpoint_path: Path) -> dict[str, Any]:
    if checkpoint_path.is_file():
        raise ValueError(
            f"DROID sim eval requires a DreamFree Stage 2 checkpoint directory, got file: {checkpoint_path}"
        )
    if not checkpoint_path.is_dir():
        raise ValueError(f"Unsupported checkpoint path: {checkpoint_path}")
    from dreamfree.models.release import read_checkpoint_metadata

    metadata = read_checkpoint_metadata(checkpoint_path, stage="stage2")
    checkpoint_type = str(metadata.get("checkpoint_type", "")).strip()
    if checkpoint_type not in SUPPORTED_CHECKPOINT_TYPES:
        raise ValueError(
            f"Unsupported checkpoint directory type {checkpoint_type!r} at {checkpoint_path}. DROID sim eval supports {sorted(SUPPORTED_CHECKPOINT_TYPES)}."
        )
    return {
        "kind": "dreamfree_stage2_dir",
        "checkpoint_path": checkpoint_path,
        "metadata": metadata,
        "checkpoint_type": checkpoint_type,
    }


def _as_positive_int(mismatches: list[str], name: str, value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        mismatches.append(f"{name} must be a positive integer, got {value!r}")
        return -1
    if parsed <= 0:
        mismatches.append(f"{name} must be positive, got {value!r}")
    return parsed


def _validate_dreamfree_stage2_for_eval(checkpoint_path: Path, metadata: dict[str, Any]) -> None:
    from dreamfree.models.release import required_policy_files

    action_history.validate_checkpoint(metadata)
    checkpoints.require_existing_files(
        checkpoint_path, required_policy_files(metadata), label="DreamFree Stage 2"
    )
    mismatches: list[str] = []
    _check_equal(
        mismatches,
        "checkpoint_type",
        metadata.get("checkpoint_type"),
        action_history.CHECKPOINT_TYPE
        if metadata.get("action_history")
        else DREAMFREE_STAGE2_CHECKPOINT_TYPE,
    )
    _check_equal(
        mismatches,
        "checkpoint_version",
        metadata.get("checkpoint_version"),
        action_history.CHECKPOINT_VERSION
        if metadata.get("action_history")
        else DREAMFREE_STAGE2_CHECKPOINT_VERSION,
    )
    if _is_none_like(metadata.get("source_stage1_checkpoint")):
        mismatches.append("source_stage1_checkpoint is missing")
    if int(metadata.get("source_stage1_global_step", -1)) < 0:
        mismatches.append(
            f"source_stage1_global_step must be non-negative, got {metadata.get('source_stage1_global_step')!r}"
        )
    family = checkpoints.metadata_video_family(metadata, label="DreamFree Stage 2")
    expected_hidden_dim = checkpoints.expected_hidden_dim_for_family(family)
    dataset = _require_dict(metadata.get("dataset"), label="DreamFree Stage 2 metadata.dataset")
    student = _require_dict(metadata.get("student"), label="DreamFree Stage 2 metadata.student")
    readout = _require_dict(metadata.get("readout"), label="DreamFree Stage 2 metadata.readout")
    action = _require_dict(metadata.get("action"), label="DreamFree Stage 2 metadata.action")
    scheduler = _require_dict(
        action.get("scheduler"), label="DreamFree Stage 2 metadata.action.scheduler"
    )
    conditioner = _require_dict(
        metadata.get("conditioner"), label="DreamFree Stage 2 metadata.conditioner"
    )
    trainable = _require_dict(
        metadata.get("trainable_modules"), label="DreamFree Stage 2 metadata.trainable_modules"
    )
    dataset_name = _normalize_droid_dataset_name(dataset.get("name"))
    if dataset_name not in DROID_DATASET_ALIASES:
        mismatches.append(
            f"dataset.name: actual={dataset.get('name')!r} expected one of {sorted(DROID_DATASET_ALIASES)!r}"
        )
    _check_equal(mismatches, "dataset.context_len", int(dataset.get("context_len", -1)), 128)
    _check_equal(
        mismatches,
        "dataset.video_size",
        dataset.get("video_size"),
        [DROID_GRID_HEIGHT, DROID_GRID_WIDTH],
    )
    _check_equal(
        mismatches, "dataset.action_dim", int(dataset.get("action_dim", -1)), DROID_ACTION_DIM
    )
    _check_equal(
        mismatches, "dataset.proprio_dim", int(dataset.get("proprio_dim", -1)), DROID_PROPRIO_DIM
    )
    _check_equal(
        mismatches,
        "dataset.camera_layout",
        dataset.get("camera_layout"),
        DROID_CONCAT_MULTI_CAMERA,
    )
    try:
        metadata_action_horizon = int(action.get("action_horizon", -1))
    except (TypeError, ValueError):
        metadata_action_horizon = -1
    if metadata_action_horizon <= 0:
        mismatches.append(
            f"action.action_horizon must be a positive integer, got {action.get('action_horizon')!r}"
        )
    if dataset.get("action_horizon") is not None:
        _check_equal(
            mismatches,
            "dataset.action_horizon",
            int(dataset.get("action_horizon", -1)),
            metadata_action_horizon,
        )
    _check_equal(mismatches, "student.train_mode", student.get("train_mode"), "frozen")
    _check_equal(mismatches, "student.hook_point", student.get("hook_point"), "post_block")
    _check_equal(
        mismatches,
        "student.video_attention_mask_mode",
        student.get("video_attention_mask_mode"),
        "first_frame_causal",
    )
    _check_equal(
        mismatches, "student.token_norm", student.get("token_norm"), "fixed_tokenwise_layer_norm"
    )
    _check_equal(
        mismatches, "student.hidden_dim", int(student.get("hidden_dim", -1)), expected_hidden_dim
    )
    readout_input_layer = int(readout.get("input_layer_j", student.get("layer_j", -1)))
    if readout_input_layer < 0:
        mismatches.append(f"readout.input_layer_j must be non-negative, got {readout_input_layer}.")
    readout_type = str(readout.get("type", checkpoints.READOUT_TYPE_LATENT_QUERY))
    if readout_type not in checkpoints.SUPPORTED_READOUT_TYPES:
        mismatches.append(
            f"readout.type must be one of {sorted(checkpoints.SUPPORTED_READOUT_TYPES)}, got {readout_type!r}."
        )
    _check_equal(
        mismatches, "readout.input_dim", int(readout.get("input_dim", -1)), expected_hidden_dim
    )
    _check_equal(
        mismatches,
        "readout.output_dim",
        int(readout.get("output_dim", -1)),
        int(action.get("hidden_dim", -1)),
    )
    _check_equal(
        mismatches,
        "action.output_parameterization",
        action.get("output_parameterization"),
        "velocity",
    )
    _check_equal(
        mismatches,
        "action.loss_parameterization",
        action.get("loss_parameterization"),
        "flow_matching",
    )
    _check_equal(
        mismatches, "action.action_dim", int(action.get("action_dim", -1)), DROID_ACTION_DIM
    )
    _check_equal(
        mismatches, "action.proprio_dim", int(action.get("proprio_dim", -1)), DROID_PROPRIO_DIM
    )
    _check_equal(
        mismatches, "action.scheduler.family", scheduler.get("family"), "continuous_flow_matching"
    )
    _check_equal(
        mismatches, "conditioner.mode", conditioner.get("mode"), "latent_query_cross_attention"
    )
    _check_equal(
        mismatches,
        "conditioner.token_policy",
        conditioner.get("token_policy"),
        "all_readout_tokens",
    )
    _check_equal(
        mismatches,
        "conditioner.concat_order",
        conditioner.get("concat_order"),
        "text_then_student_readout",
    )
    _check_equal(
        mismatches, "conditioner.context_norm", conditioner.get("context_norm"), "layernorm_affine"
    )
    _check_equal(
        mismatches,
        "conditioner.final_kv_contract",
        conditioner.get("final_kv_contract"),
        "text_plus_student_readout_only",
    )
    _check_equal(mismatches, "trainable_modules.student", bool(trainable.get("student")), False)
    _check_equal(mismatches, "trainable_modules.readout", bool(trainable.get("readout")), True)
    _check_equal(
        mismatches, "trainable_modules.action_dit", bool(trainable.get("action_dit")), True
    )
    _check_equal(
        mismatches, "trainable_modules.conditioner", bool(trainable.get("conditioner")), True
    )
    for label, value in (
        ("readout.num_queries", readout.get("num_queries")),
        ("readout.num_layers", readout.get("num_layers")),
        ("readout.num_heads", readout.get("num_heads")),
        ("action.scheduler.num_train_timesteps", scheduler.get("num_train_timesteps")),
    ):
        _as_positive_int(mismatches, label, value)
    if float(readout.get("mlp_ratio", -1.0)) <= 0.0:
        mismatches.append(f"readout.mlp_ratio must be positive, got {readout.get('mlp_ratio')!r}")
    if "train_shift" not in scheduler:
        mismatches.append("action.scheduler.train_shift is missing")
    if mismatches:
        raise ValueError(
            "DreamFree Stage 2 checkpoint is incompatible with DROID sim eval: "
            + "; ".join(mismatches)
        )


def droid_shape_meta() -> dict[str, Any]:
    return {
        "images": [
            {"key": "exterior_image_1_left", "raw_shape": [3, 180, 320], "shape": [3, 180, 320]},
            {"key": "exterior_image_2_left", "raw_shape": [3, 180, 320], "shape": [3, 180, 320]},
            {"key": "wrist_image_left", "raw_shape": [3, 180, 320], "shape": [3, 180, 320]},
        ],
        "action": [
            {
                "key": DROID_JOINT_POSITION_KEY,
                "raw_shape": 7,
                "shape": 7,
                "source_key": "action",
                "source_slice": (14, 21),
                "relative_to_state_key": DROID_JOINT_POSITION_KEY,
                "state_source_slice": (7, 14),
            },
            {
                "key": DROID_GRIPPER_POSITION_KEY,
                "raw_shape": 1,
                "shape": 1,
                "source_key": "action",
                "source_slice": (12, 13),
            },
        ],
        "state": [
            {
                "key": DROID_JOINT_POSITION_KEY,
                "raw_shape": 7,
                "shape": 7,
                "source_key": "observation.state",
                "source_slice": (7, 14),
            },
            {
                "key": DROID_GRIPPER_POSITION_KEY,
                "raw_shape": 1,
                "shape": 1,
                "source_key": "observation.state",
                "source_slice": (6, 7),
            },
        ],
    }


def _load_droid_json(dataset_root: Path, name: str) -> dict[str, Any]:
    path = dataset_root / "meta" / name
    if not path.exists() and name == DROID_RELATIVE_STATS_FILE:
        path = dataset_root / name
    if not path.exists():
        raise FileNotFoundError(f"Missing DROID metadata file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"DROID metadata file must contain a JSON object: {path}")
    return payload


def _droid_modality_range(dataset_root: Path, modality: str, key: str) -> tuple[int, int]:
    payload = _load_droid_json(dataset_root, "modality.json")
    modality_payload = payload.get(modality)
    if not isinstance(modality_payload, dict) or key not in modality_payload:
        raise KeyError(
            f"DROID modality metadata missing {modality}.{key}: {dataset_root / 'meta' / 'modality.json'}"
        )
    item = modality_payload[key]
    if not isinstance(item, dict) or "start" not in item or "end" not in item:
        raise ValueError(
            f"DROID modality metadata for {modality}.{key} must define start/end, got {item!r}"
        )
    (start, end) = (int(item["start"]), int(item["end"]))
    if start < 0 or end <= start:
        raise ValueError(
            f"Invalid DROID modality range for {modality}.{key}: start={start}, end={end}"
        )
    return (start, end)


def _slice_stats(
    source: dict[str, Any], *, start: int, end: int, feature_key: str
) -> dict[str, list[float]]:
    sliced: dict[str, list[float]] = {}
    for stat_name in ("min", "max", "mean", "std", "q01", "q99"):
        if stat_name not in source:
            raise KeyError(f"DROID metadata stats for {feature_key!r} are missing {stat_name!r}.")
        values = np.asarray(source[stat_name], dtype=np.float32)
        if values.ndim != 1 or int(values.shape[0]) < end:
            raise ValueError(
                f"DROID metadata stats for {feature_key!r}.{stat_name} must be 1D with at least {end} values, got shape={values.shape}."
            )
        sliced[stat_name] = values[start:end].tolist()
    return sliced


def _convert_feature_stats(source: dict[str, Any], *, feature_key: str) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for stat_name in ("min", "max", "mean", "std", "q01", "q99"):
        if stat_name not in source:
            raise KeyError(f"DROID DreamZero stats for {feature_key!r} are missing {stat_name!r}.")
        tensor = torch.as_tensor(source[stat_name], dtype=torch.float32)
        converted[f"global_{stat_name}"] = tensor
        converted[f"stepwise_{stat_name}"] = tensor.unsqueeze(0)
    return converted


def load_droid_dataset_stats(
    dataset_root: str | Path = DROID_DATASET_ROOT,
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    root = Path(dataset_root).expanduser().resolve()
    stats = _load_droid_json(root, "stats.json")
    relative_stats = _load_droid_json(root, DROID_RELATIVE_STATS_FILE)
    joint_relative_stats = relative_stats.get(DROID_JOINT_POSITION_KEY)
    if not isinstance(joint_relative_stats, dict):
        raise KeyError(f"DROID relative stats must contain {DROID_JOINT_POSITION_KEY!r}: {root}")
    if "action" not in stats or "observation.state" not in stats:
        raise KeyError(
            f"DROID stats must contain 'action' and 'observation.state': {root / 'meta' / 'stats.json'}"
        )
    (state_joint_start, state_joint_end) = _droid_modality_range(
        root, "state", DROID_JOINT_POSITION_KEY
    )
    (state_gripper_start, state_gripper_end) = _droid_modality_range(
        root, "state", DROID_GRIPPER_POSITION_KEY
    )
    (action_gripper_start, action_gripper_end) = _droid_modality_range(
        root, "action", DROID_GRIPPER_POSITION_KEY
    )
    return {
        "action": {
            DROID_JOINT_POSITION_KEY: _convert_feature_stats(
                joint_relative_stats, feature_key="action.joint_position"
            ),
            DROID_GRIPPER_POSITION_KEY: _convert_feature_stats(
                _slice_stats(
                    stats["action"],
                    start=action_gripper_start,
                    end=action_gripper_end,
                    feature_key="action.gripper_position",
                ),
                feature_key="action.gripper_position",
            ),
        },
        "state": {
            DROID_JOINT_POSITION_KEY: _convert_feature_stats(
                _slice_stats(
                    stats["observation.state"],
                    start=state_joint_start,
                    end=state_joint_end,
                    feature_key="state.joint_position",
                ),
                feature_key="state.joint_position",
            ),
            DROID_GRIPPER_POSITION_KEY: _convert_feature_stats(
                _slice_stats(
                    stats["observation.state"],
                    start=state_gripper_start,
                    end=state_gripper_end,
                    feature_key="state.gripper_position",
                ),
                feature_key="state.gripper_position",
            ),
        },
    }


def build_droid_processor(*, dataset_stats: dict[str, Any]) -> DreamFreeProcessor:
    processor = DreamFreeProcessor(
        shape_meta=droid_shape_meta(),
        num_obs_steps=33,
        num_output_cameras=3,
        action_output_dim=DROID_ACTION_DIM,
        proprio_output_dim=DROID_PROPRIO_DIM,
        action_state_transforms=None,
        use_stepwise_action_norm=False,
        norm_default_mode="q01/q99",
        norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(),
        train_transforms=[],
        val_transforms=[],
    ).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    return processor


def _lookup_rgb(obs_data: Mapping[str, Any], key: str) -> Any:
    aliases = {
        "observation/exterior_image_1_left": (
            "observation/exterior_image_1_left",
            "observation/exterior_image_0_left",
            "exterior_image_1_left",
            "external_cam",
        ),
        "observation/exterior_image_2_left": (
            "observation/exterior_image_2_left",
            "observation/exterior_image_1_left_secondary",
            "exterior_image_2_left",
            "external_cam_2",
        ),
        "observation/wrist_image_left": (
            "observation/wrist_image_left",
            "wrist_image_left",
            "wrist_cam",
        ),
    }
    candidates = aliases.get(key, (key,))
    for candidate in candidates:
        if candidate not in obs_data:
            continue
        value = obs_data[candidate]
        if isinstance(value, Mapping):
            if "rgb" not in value:
                raise KeyError(f"DROID obs key {candidate!r} is a mapping but has no 'rgb' entry.")
            return value["rgb"]
        return value
    raise KeyError(f"DROID obs is missing RGB view {key!r}; tried {candidates}.")


def build_droid_image_tensor(
    obs_data: Mapping[str, Any],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build one normalized DROID grid image tensor matching Stage 2 training."""
    if PREPROCESSING_KEY in obs_data:
        processed = processed_frames(obs_data)
        if (
            obs_data[PREPROCESSING_KEY] == TRANSPORT_FORMATS["bf16"]
            and dtype is not None
            and (dtype != torch.bfloat16)
        ):
            raise ValueError("BF16 image transport requires a bfloat16 model input.")
    else:
        frames = [_rgb_to_chw_frame(_lookup_rgb(obs_data, key)) for key in DROID_EVAL_VIEW_KEYS]
        processed = [
            _center_crop_resize_frame(frame, size=(DROID_PER_CAMERA_HEIGHT, DROID_PER_CAMERA_WIDTH))
            for frame in frames
        ]
    left = processed[0]
    right = processed[1]
    wrist = processed[2]
    wrist_wide = wrist.repeat_interleave(2, dim=-1)
    bottom = torch.cat([left, right], dim=-1)
    image = torch.cat([wrist_wide, bottom], dim=-2)
    if tuple(image.shape) != (1, 3, DROID_GRID_HEIGHT, DROID_GRID_WIDTH):
        raise ValueError(f"DROID image tensor must be [1, 3, 352, 640], got {tuple(image.shape)}.")
    if obs_data.get(PREPROCESSING_KEY) != TRANSPORT_FORMATS["bf16"]:
        image = image.clamp(0.0, 1.0) * 2.0 - 1.0
    if device is not None or dtype is not None:
        image = image.to(device=device, dtype=dtype)
    return image.contiguous()


def _as_1d_float_array(value: Any, *, name: str, dim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if int(array.shape[0]) != int(dim):
        raise ValueError(f"{name} must have {dim} dims, got shape={array.shape}.")
    return array


def extract_droid_state(obs_data: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    joint_candidates = ("observation/joint_position", "joint_position", "arm_joint_pos")
    gripper_candidates = ("observation/gripper_position", "gripper_position", "gripper_pos")
    joint = next((obs_data[key] for key in joint_candidates if key in obs_data), None)
    gripper = next((obs_data[key] for key in gripper_candidates if key in obs_data), None)
    if joint is None:
        raise KeyError(f"DROID obs is missing joint position; tried {joint_candidates}.")
    if gripper is None:
        raise KeyError(f"DROID obs is missing gripper position; tried {gripper_candidates}.")
    return (
        _as_1d_float_array(joint, name="observation/joint_position", dim=7),
        _as_1d_float_array(gripper, name="observation/gripper_position", dim=1),
    )


def normalize_droid_state_with_processor(
    processor: DreamFreeProcessor, joint_position: np.ndarray, gripper_position: np.ndarray
) -> torch.Tensor:
    state_batch = {
        "state": {
            DROID_JOINT_POSITION_KEY: torch.as_tensor(joint_position, dtype=torch.float32).reshape(
                1, 7
            ),
            DROID_GRIPPER_POSITION_KEY: torch.as_tensor(
                gripper_position, dtype=torch.float32
            ).reshape(1, 1),
        }
    }
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    state_batch = processor.action_state_merger.forward(state_batch)
    return state_batch["state"]


def denormalize_droid_action_with_processor(
    processor: DreamFreeProcessor, action: torch.Tensor
) -> dict[str, np.ndarray]:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}.")
    if int(action.shape[-1]) != DROID_ACTION_DIM:
        raise ValueError(
            f"Expected DROID action dim {DROID_ACTION_DIM}, got {int(action.shape[-1])}."
        )
    action = action.to(dtype=torch.float32, device="cpu")
    action_dict = {
        DROID_JOINT_POSITION_KEY: action[..., :7],
        DROID_GRIPPER_POSITION_KEY: action[..., 7:8],
    }
    out: dict[str, np.ndarray] = {}
    for key, tensor in action_dict.items():
        normalizer = processor.normalizer.normalizers["action"][key]
        out[key] = normalizer.backward(tensor).numpy()
    return out


def droid_relative_actions_to_sim_actions(
    relative_joint_actions: np.ndarray,
    gripper_actions: np.ndarray,
    current_joint_position: np.ndarray,
    *,
    binarize_gripper: bool = True,
) -> np.ndarray:
    rel = np.asarray(relative_joint_actions, dtype=np.float32)
    grip = np.asarray(gripper_actions, dtype=np.float32)
    joint = np.asarray(current_joint_position, dtype=np.float32).reshape(7)
    if rel.ndim == 3 and int(rel.shape[0]) == 1:
        rel = rel[0]
    if grip.ndim == 3 and int(grip.shape[0]) == 1:
        grip = grip[0]
    if rel.ndim != 2 or int(rel.shape[1]) != 7:
        raise ValueError(f"relative_joint_actions must have shape [T,7], got {rel.shape}.")
    if grip.ndim == 1:
        grip = grip.reshape(-1, 1)
    if grip.ndim != 2 or int(grip.shape[1]) != 1 or int(grip.shape[0]) != int(rel.shape[0]):
        raise ValueError(f"gripper_actions must have shape [T,1], got {grip.shape}.")
    if binarize_gripper:
        grip = (grip > 0.5).astype(np.float32)
    absolute_joint = rel + joint.reshape(1, 7)
    return np.concatenate([absolute_joint, grip], axis=-1).astype(np.float32, copy=False)


class DroidPolicy(PolicyRuntime):
    def __init__(
        self,
        *,
        droid_dataset_root: Path,
        compile_dit: bool,
        compile_dit_block_counts: Mapping[str, int],
        action_noise_seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        if action_noise_seed is not None:
            if kwargs.get("seed") is not None:
                raise ValueError(
                    "action_noise_seed and the legacy policy seed are mutually exclusive."
                )
            if (
                isinstance(action_noise_seed, bool)
                or not isinstance(action_noise_seed, int)
                or (not 0 <= action_noise_seed <= MAX_ACTION_NOISE_SEED)
            ):
                raise ValueError(
                    f"action_noise_seed must be an integer in [0, {MAX_ACTION_NOISE_SEED}], got {action_noise_seed!r}."
                )
        super().__init__(**kwargs)
        self.action_history_horizon = self.stage2_metadata.get("action_history", {}).get(
            "horizon", 0
        )
        self._active_previous_action = self._active_previous_action_is_pad = None
        self.droid_dataset_root = Path(droid_dataset_root).expanduser().resolve()
        self.compile_dit = bool(compile_dit)
        self.compile_dit_block_counts = dict(compile_dit_block_counts)
        self.action_noise_seed = None if action_noise_seed is None else int(action_noise_seed)
        self.action_noise_episode_seed: int | None = None
        self.action_noise_derived_seed: int | None = None
        self._action_noise_generator: torch.Generator | None = None
        logger.info(
            "Initialized DreamFree Stage 2 DROID sim policy | stage1=%s | stage2=%s | dataset_root=%s | horizon=%d | compile_dit=%s",
            self.stage1_checkpoint_path,
            self.stage2_checkpoint_path,
            self.droid_dataset_root,
            self.action_horizon,
            self.compile_dit,
        )

    def reset_action_noise_for_episode(self, episode_seed: int) -> None:
        if self.action_noise_seed is None:
            raise RuntimeError(
                "Cannot reset the action-noise RNG because action_noise_seed is not configured."
            )
        if isinstance(episode_seed, bool) or not isinstance(episode_seed, int) or episode_seed < 0:
            raise ValueError(f"episode_seed must be a non-negative integer, got {episode_seed!r}.")
        self.action_noise_episode_seed = int(episode_seed)
        # Preserve AWAM's seed namespace so identical episode seeds reproduce its noise.
        digest = hashlib.sha256(
            f"awam-droid-action-noise-v1:{self.action_noise_seed}:{self.action_noise_episode_seed}".encode(
                "utf-8"
            )
        ).digest()
        self.action_noise_derived_seed = int.from_bytes(digest[:8], "big") & MAX_ACTION_NOISE_SEED
        if self._action_noise_generator is None:
            self._action_noise_generator = torch.Generator(device=self.noise_device)
        self._action_noise_generator.manual_seed(self.action_noise_derived_seed)

    def _sample_randn(self, shape: tuple[int, ...]) -> torch.Tensor:
        if self.action_noise_seed is None:
            return super()._sample_randn(shape)
        if self._action_noise_generator is None or self.action_noise_derived_seed is None:
            raise RuntimeError(
                "Random action noise was requested before the per-episode RNG was initialized. Call reset_action_noise_for_episode() at episode reset."
            )
        return torch.randn(
            shape,
            generator=self._action_noise_generator,
            device=self.noise_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.model_dtype)

    def _build_observation_image(self, observation):
        return build_droid_image_tensor(observation, device=self.device, dtype=self.model_dtype)

    def _infer_relative_action_chunk(
        self, observation: Dict[str, Any], instruction: str
    ) -> np.ndarray:
        if self.action_history_horizon:
            history, pad = extract_supplied_action_history(
                observation, horizon=self.action_history_horizon, action_dim=8
            )
            if not np.isin(history[~pad, 7], (0, 1)).all():
                raise ValueError("DROID action history requires binary issued gripper commands.")
            current_joint, _ = extract_droid_state(observation)
            raw = {
                DROID_JOINT_POSITION_KEY: torch.from_numpy(
                    history[:, :7] - current_joint.reshape(1, 7)
                ),
                DROID_GRIPPER_POSITION_KEY: torch.from_numpy(history[:, 7:8]),
            }
            normalized = {
                key: self.processor.normalizer.normalizers["action"][key].forward(value)
                for key, value in raw.items()
            }
            previous = self.processor.action_state_merger._concat(
                normalized, self.processor.action_state_merger.action_meta
            )
            previous[torch.from_numpy(pad)] = 0
            self._active_previous_action = previous[None].to(
                device=self.device, dtype=self.model_dtype
            )
            self._active_previous_action_is_pad = torch.from_numpy(pad)[None].to(self.device)
        try:
            return self._infer_relative_action_chunk_with_history(observation, instruction)
        finally:
            self._active_previous_action = self._active_previous_action_is_pad = None

    def _infer_relative_action_chunk_with_history(
        self, observation: Dict[str, Any], instruction: str
    ) -> np.ndarray:
        if self.action_history_horizon and not hasattr(self, "_prepare_action_query"):
            raise ValueError(
                "DROID action-history inference requires the optimized policy builder."
            )
        image_tensor = self._build_observation_image(observation)
        (joint_position, gripper_position) = extract_droid_state(observation)
        proprio = normalize_droid_state_with_processor(
            self.processor, joint_position, gripper_position
        ).to(device=self.device, dtype=self.model_dtype)
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        (context, context_mask, student_context_emb, student_context_mask, proprio_query_emb) = (
            self._build_student_readout_conditioning(
                image_tensor=image_tensor, instruction=instruction, proprio=proprio
            )
        )
        normalized_action = self._infer_dreamfree_action_tokens(
            context=context,
            context_mask=context_mask,
            student_context_emb=student_context_emb,
            student_context_mask=student_context_mask,
            proprio_query_emb=proprio_query_emb,
        )
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0
        denorm = denormalize_droid_action_with_processor(self.processor, normalized_action)
        relative_joint = np.asarray(denorm[DROID_JOINT_POSITION_KEY], dtype=np.float32)
        gripper = np.asarray(denorm[DROID_GRIPPER_POSITION_KEY], dtype=np.float32)
        if relative_joint.ndim == 3 and int(relative_joint.shape[0]) == 1:
            relative_joint = relative_joint[0]
        if gripper.ndim == 3 and int(gripper.shape[0]) == 1:
            gripper = gripper[0]
        if relative_joint.ndim != 2 or int(relative_joint.shape[1]) != 7:
            raise ValueError(
                f"Denormalized DROID relative joint actions must have shape [T,7], got {relative_joint.shape}."
            )
        if gripper.ndim == 1:
            gripper = gripper.reshape(-1, 1)
        if gripper.ndim != 2 or tuple(gripper.shape) != (int(relative_joint.shape[0]), 1):
            raise ValueError(
                f"Denormalized DROID gripper actions must have shape [T,1], got {gripper.shape}."
            )
        return np.concatenate([relative_joint, gripper], axis=-1).astype(np.float32, copy=False)

    def predict_relative_action_chunk(
        self, observation: Dict[str, Any], instruction: str
    ) -> np.ndarray:
        """Predict one unbinarized action chunk in the DROID Stage 2 training space."""
        return self._infer_relative_action_chunk(observation=observation, instruction=instruction)

    def _infer_action_chunk(
        self, observation: Dict[str, Any], instruction: str, *, binarize_gripper: bool = True
    ) -> np.ndarray:
        relative_action = self._infer_relative_action_chunk(
            observation=observation, instruction=instruction
        )
        (joint_position, _) = extract_droid_state(observation)
        return droid_relative_actions_to_sim_actions(
            relative_action[:, :7],
            relative_action[:, 7:8],
            joint_position,
            binarize_gripper=binarize_gripper,
        )

    def predict_action_chunk(
        self, observation: Dict[str, Any], instruction: str, *, binarize_gripper: bool = True
    ) -> np.ndarray:
        return self._infer_action_chunk(
            observation=observation, instruction=instruction, binarize_gripper=binarize_gripper
        )


def build_policy_from_args(
    usr_args: Dict[str, Any],
    *,
    required_checkpoint_type: str | None = None,
    policy_runtime_override: str | None = None,
    stage1_core_loader: Any | None = None,
    checkpoint_inspector: Any = None,
    checkpoint_validator: Any = None,
    policy_cls: Any = None,
    conditioner_adapter: Callable[[torch.nn.Module], torch.nn.Module] | None = None,
) -> DroidPolicy:
    checkpoint_path_raw = usr_args.get("checkpoint_path")
    if _is_none_like(checkpoint_path_raw):
        raise ValueError(
            "`checkpoint_path` is required and must be a DreamFree Stage 2 checkpoint directory."
        )
    checkpoint_path = _resolve_checkpoint_path(str(checkpoint_path_raw))
    resolved_checkpoint_inspector = checkpoint_inspector or _inspect_checkpoint_path
    checkpoint_info = resolved_checkpoint_inspector(checkpoint_path)
    metadata = checkpoint_info["metadata"]
    checkpoint_type = str(checkpoint_info["checkpoint_type"])
    if required_checkpoint_type is not None and checkpoint_type != required_checkpoint_type:
        raise ValueError(
            f"This DROID evaluation mode requires checkpoint_type={required_checkpoint_type!r}, got {checkpoint_type!r}."
        )
    if checkpoint_validator is not None:
        checkpoint_validator(checkpoint_path, metadata)
    else:
        _validate_dreamfree_stage2_for_eval(checkpoint_path, metadata)
    if not _is_none_like(usr_args.get("stage1_checkpoint")):
        raise ValueError(
            "DROID sim eval loads Stage 1 from DreamFree Stage 2 metadata; do not pass `stage1_checkpoint`."
        )
    requested_runtime = usr_args.get("policy_runtime")
    if policy_runtime_override is None:
        runtime = str(requested_runtime or DREAMFREE_POLICY_RUNTIME).strip().lower()
        if runtime != DREAMFREE_POLICY_RUNTIME:
            raise ValueError(
                f"DROID sim eval only supports policy_runtime={DREAMFREE_POLICY_RUNTIME!r}, got {runtime!r}."
            )
    else:
        runtime = str(policy_runtime_override).strip().lower()
        if (
            not _is_none_like(requested_runtime)
            and str(requested_runtime).strip().lower() != runtime
        ):
            raise ValueError(
                f"This DROID evaluation mode requires policy_runtime={runtime!r}, got {requested_runtime!r}."
            )
    device = str(usr_args.get("device") or "cuda")
    if device.startswith("cuda") and (not torch.cuda.is_available()):
        raise ValueError(f"DROID sim eval requested device={device!r}, but CUDA is unavailable.")
    noise_device = str(usr_args.get("noise_device") or "cpu")
    if noise_device.startswith("cuda") and (not torch.cuda.is_available()):
        raise ValueError("DROID sim eval requested CUDA noise_device, but CUDA is unavailable.")
    model_dtype = dreamfree_deploy._mixed_precision_to_model_dtype(
        str(usr_args.get("mixed_precision") or "bf16")
    )
    load_text_encoder = dreamfree_deploy._parse_bool(usr_args.get("load_text_encoder", True))
    text_embedding_cache_dir = usr_args.get("text_embedding_cache_dir")
    if not load_text_encoder and _is_none_like(text_embedding_cache_dir):
        raise ValueError("load_text_encoder=false requires text_embedding_cache_dir.")
    droid_dataset_root = (
        Path(str(usr_args.get("droid_dataset_root") or DROID_DATASET_ROOT)).expanduser().resolve()
    )
    dataset_stats_path = usr_args.get("dataset_stats_path")
    if _is_none_like(dataset_stats_path) and "release" in metadata:
        dataset_stats_path = metadata["dataset"]["dataset_stats_path"]
    if _is_none_like(dataset_stats_path):
        dataset_stats = load_droid_dataset_stats(droid_dataset_root)
        dataset_stats_label = droid_dataset_root / "meta"
    else:
        dataset_stats_label = Path(str(dataset_stats_path)).expanduser().resolve()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_label))
    processor = build_droid_processor(dataset_stats=dataset_stats)
    stage1_checkpoint_path = checkpoints.require_checkpoint_dir(
        metadata.get("source_stage1_checkpoint"), label="DreamFree Stage 2 source Stage 1"
    )
    from dreamfree.models.release import is_release_checkpoint, read_checkpoint_metadata

    if is_release_checkpoint(stage1_checkpoint_path):
        stage1_metadata = read_checkpoint_metadata(stage1_checkpoint_path, stage="stage1")
    else:
        stage1_metadata = _json_load(stage1_checkpoint_path / "trainer_state.json")
    checkpoints.validate_stage1_stage2_lineage(
        stage1_checkpoint_path=stage1_checkpoint_path,
        stage1_metadata=stage1_metadata,
        stage2_metadata=metadata,
    )
    stage2_cfg = checkpoints.stage2_config_from_metadata(metadata)
    if stage1_core_loader is not None:
        video_core = stage1_core_loader(
            stage1_checkpoint_path=stage1_checkpoint_path,
            stage1_metadata=stage1_metadata,
            stage2_metadata=metadata,
            device=device,
            dtype=model_dtype,
            load_text_encoder=load_text_encoder,
        )
    else:
        video_core = dreamfree_deploy._load_stage1_student_core_with_text(
            stage1_checkpoint_path=stage1_checkpoint_path,
            metadata=stage1_metadata,
            device=device,
            dtype=model_dtype,
            load_text_encoder=load_text_encoder,
        )
    readout = checkpoints.load_readout(
        metadata=metadata,
        checkpoint_path=checkpoint_path,
        device=torch.device(video_core.device),
        dtype=video_core.torch_dtype,
    )
    action_model = ActionDiT.from_pretrained(
        action_dit_config={
            "action_dim": int(stage2_cfg.action_dim),
            "hidden_dim": int(stage2_cfg.action_hidden_dim),
            "ffn_dim": int(stage2_cfg.action_ffn_dim),
            "text_dim": int(stage2_cfg.action_text_dim),
            "freq_dim": int(stage2_cfg.action_freq_dim),
            "eps": float(stage2_cfg.action_eps),
            "num_heads": int(stage2_cfg.action_num_heads),
            "attn_head_dim": int(stage2_cfg.action_attn_head_dim),
            "num_layers": int(stage2_cfg.action_num_layers),
            "use_gradient_checkpointing": bool(
                getattr(stage2_cfg, "use_gradient_checkpointing", False)
            ),
        },
        action_dit_pretrained_path=str(getattr(stage2_cfg, "action_dit_pretrained_path", "") or ""),
        skip_dit_load_from_pretrain=True,
        device=str(video_core.device),
        torch_dtype=video_core.torch_dtype,
    ).eval()
    conditioner_meta = _require_dict(
        metadata.get("conditioner"), label="DreamFree Stage 2 metadata.conditioner"
    )
    conditioner = (
        TeacherStateActionConditioner(
            teacher_state_input_dim=int(conditioner_meta["student_readout_input_dim"]),
            action_context_dim=int(conditioner_meta["action_context_dim"]),
            proprio_dim=int(stage2_cfg.proprio_dim),
            proprio_obs_dropout=float(stage2_cfg.proprio_obs_dropout),
        )
        .to(device=video_core.device, dtype=video_core.torch_dtype)
        .eval()
    )
    if conditioner_adapter is not None:
        conditioner = conditioner_adapter(conditioner)
        if not isinstance(conditioner, torch.nn.Module):
            raise TypeError(
                f"DROID conditioner_adapter must return a torch.nn.Module, got {type(conditioner).__name__}."
            )
        conditioner = conditioner.to(device=video_core.device, dtype=video_core.torch_dtype).eval()
    action_model.load_state_dict(
        load_safetensors(
            str(
                checkpoints.require_existing_file(
                    checkpoint_path / "action.safetensors",
                    label="DreamFree Stage 2 action checkpoint",
                )
            ),
            device="cpu",
        ),
        strict=True,
    )
    conditioner.load_teacher_state_dict(
        load_safetensors(
            str(
                checkpoints.require_existing_file(
                    checkpoint_path / "conditioner.safetensors",
                    label="DreamFree Stage 2 conditioner checkpoint",
                )
            ),
            device="cpu",
        ),
        strict=True,
    )
    conditioner.load_proprio_state_dict(
        load_safetensors(
            str(
                checkpoints.require_existing_file(
                    checkpoint_path / "proprio_encoder.safetensors",
                    label="DreamFree Stage 2 proprio encoder checkpoint",
                )
            ),
            device="cpu",
        ),
        strict=True,
    )
    metadata_action_horizon = int(metadata["action"]["action_horizon"])
    action_horizon_override = dreamfree_deploy._parse_optional_int(usr_args.get("action_horizon"))
    if (
        action_horizon_override is not None
        and int(action_horizon_override) != metadata_action_horizon
    ):
        raise ValueError(
            f"DROID sim eval does not support overriding action_horizon to a value different from checkpoint metadata. override={action_horizon_override} metadata={metadata_action_horizon}"
        )
    replan_steps = dreamfree_deploy._parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = metadata_action_horizon
    num_inference_steps = dreamfree_deploy._parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = 10
    action_sampler = (
        str(usr_args.get("action_sampler") or options.ACTION_SAMPLER_EULER).strip().lower()
    )
    action_infer_shift = options.parse_optional_float(usr_args.get("action_infer_shift"))
    action_unipc_solver_order = dreamfree_deploy._parse_optional_int(
        usr_args.get("action_unipc_solver_order")
    )
    if action_unipc_solver_order is None:
        action_unipc_solver_order = 2
    action_unipc_solver_type = (
        str(usr_args.get("action_unipc_solver_type") or "bh2").strip().lower()
    )
    action_unipc_lower_order_final = dreamfree_deploy._parse_bool(
        usr_args.get("action_unipc_lower_order_final", True)
    )
    action_unipc_disable_corrector = options.parse_nonnegative_int_list(
        usr_args.get("action_unipc_disable_corrector"), label="action_unipc_disable_corrector"
    )
    readout_active_num_queries = dreamfree_deploy._parse_optional_int(
        usr_args.get("readout_active_num_queries")
    )
    seed = dreamfree_deploy._parse_optional_int(usr_args.get("seed"))
    action_noise_seed = dreamfree_deploy._parse_optional_int(usr_args.get("action_noise_seed"))
    tiled = dreamfree_deploy._parse_bool(usr_args.get("tiled", False))
    timing_enabled = dreamfree_deploy._parse_bool(usr_args.get("timing_enabled", False))
    compile_dit = dreamfree_deploy._parse_bool(usr_args.get("compile_dit", False))
    compile_dit_block_counts: dict[str, int] = {}
    if compile_dit:
        compile_dit_block_counts["video_core_dit_blocks"] = compile_dit_blocks(
            video_core.dit, label="DROID deploy Stage 1 student"
        )
        compile_dit_block_counts["action_model_dit_blocks"] = compile_dit_blocks(
            action_model, label="DROID deploy ActionDiT"
        )
    resolved_policy_cls = policy_cls or DroidPolicy
    policy = resolved_policy_cls(
        processor=processor,
        video_core=video_core,
        readout=readout,
        action_model=action_model,
        conditioner=conditioner,
        stage1_metadata=stage1_metadata,
        stage2_metadata=metadata,
        stage1_checkpoint_path=stage1_checkpoint_path,
        stage2_checkpoint_path=checkpoint_path,
        dataset_stats_path=Path(dataset_stats_label),
        action_horizon=metadata_action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        readout_active_num_queries=readout_active_num_queries,
        seed=seed,
        action_noise_seed=action_noise_seed,
        noise_device=noise_device,
        tiled=tiled,
        timing_enabled=timing_enabled,
        compile_dit=compile_dit,
        compile_dit_block_counts=compile_dit_block_counts,
        text_embedding_cache_dir=text_embedding_cache_dir,
        action_sampler=action_sampler,
        action_infer_shift=action_infer_shift,
        action_unipc_solver_order=action_unipc_solver_order,
        action_unipc_solver_type=action_unipc_solver_type,
        action_unipc_lower_order_final=action_unipc_lower_order_final,
        action_unipc_disable_corrector=action_unipc_disable_corrector,
        droid_dataset_root=droid_dataset_root,
    )
    policy.policy_runtime = runtime
    return policy
