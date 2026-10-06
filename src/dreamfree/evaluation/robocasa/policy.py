"""RoboCasa deployment wrapper for DreamFree Stage 2 student-native checkpoints."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

import dreamfree.evaluation.robotwin.policy as robotwin_dreamfree
import dreamfree.models.policy_loading as dreamfree_deploy
from dreamfree.data.robocasa_visual import (
    ROBOCASA_DIRECT_352_DATASET_NAMES,
    ROBOCASA_DIRECT_448_DATASET_NAMES,
    ROBOCASA_DREAMZERO_352_DIRECT_BILINEAR_V1,
    ROBOCASA_DREAMZERO_448_DIRECT_BILINEAR_V1,
    ROBOCASA_RAW_CAMERA_SIZE,
    compose_robocasa_dreamzero_uint8,
    normalize_uint8_video,
    required_robocasa_visual_preprocess_id,
)
from dreamfree.evaluation.robocasa.robocasa_utils import (
    extract_robocasa_state,
    prepare_robocasa_observation,
)

logger = logging.getLogger(__name__)
DREAMFREE_STAGE2_CHECKPOINT_TYPE = robotwin_dreamfree.DREAMFREE_STAGE2_CHECKPOINT_TYPE
DREAMFREE_STAGE2_CHECKPOINT_VERSION = robotwin_dreamfree.DREAMFREE_STAGE2_CHECKPOINT_VERSION
DREAMFREE_STAGE2_REQUIRED_FILES = robotwin_dreamfree.DREAMFREE_STAGE2_REQUIRED_FILES
ROBOCASA_EVAL_VIEW_KEYS = ("agentview_left", "agentview_right", "wrist")
ROBOCASA_DATASET_ALIASES = set(ROBOCASA_DIRECT_352_DATASET_NAMES) | set(
    ROBOCASA_DIRECT_448_DATASET_NAMES
)
ROBOCASA_DREAMZERO_VIDEO_SIZES: dict[str, tuple[int, int]] = {
    "robocasa_dreamzero_352": (352, 352),
    "robocasa_dreamzero_448": (448, 448),
}


def _is_none_like(value: Any) -> bool:
    return dreamfree_deploy._is_none_like(value)


def _require_dict(value: Any, *, label: str) -> dict[str, Any]:
    return robotwin_dreamfree._require_dict(value, label=label)


def _check_equal(mismatches: list[str], name: str, actual: Any, expected: Any) -> None:
    robotwin_dreamfree._check_equal(mismatches, name, actual, expected)


def _normalize_robocasa_dataset_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def _supported_robocasa_geometry_text() -> str:
    return ", ".join(
        f"{concat_multi_camera}=>{list(video_size)}"
        for (concat_multi_camera, video_size) in sorted(ROBOCASA_DREAMZERO_VIDEO_SIZES.items())
    )


def _normalize_robocasa_image_size(image_size: Any, *, label: str) -> tuple[int, int]:
    try:
        parsed = tuple(int(v) for v in image_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must be a 2-item integer image size, got {image_size!r}."
        ) from exc
    if len(parsed) != 2:
        raise ValueError(
            f"{label} must have exactly 2 entries [height, width], got {image_size!r}."
        )
    (height, width) = parsed
    if height <= 0 or width <= 0:
        raise ValueError(f"{label} must be positive, got {list(parsed)!r}.")
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(
            f"{label} must be even for RoboCasa DreamZero 2x2 layout, got {list(parsed)!r}."
        )
    if parsed not in set(ROBOCASA_DREAMZERO_VIDEO_SIZES.values()):
        raise ValueError(
            f"{label}={list(parsed)!r} is unsupported for RoboCasa eval. Supported geometries: {_supported_robocasa_geometry_text()}."
        )
    return (height, width)


def _resolve_robocasa_image_size_from_dataset(
    dataset: Mapping[str, Any], *, label: str
) -> tuple[int, int]:
    concat_multi_camera = str(dataset.get("concat_multi_camera", "")).strip()
    expected_size = ROBOCASA_DREAMZERO_VIDEO_SIZES.get(concat_multi_camera)
    if expected_size is None:
        raise ValueError(
            f"{label}.concat_multi_camera={dataset.get('concat_multi_camera')!r} is unsupported. Supported geometries: {_supported_robocasa_geometry_text()}."
        )
    raw_video_size = dataset.get("video_size")
    video_size = _normalize_robocasa_image_size(raw_video_size, label=f"{label}.video_size")
    if video_size != expected_size:
        raise ValueError(
            f"{label}.video_size={list(video_size)!r} is inconsistent with concat_multi_camera={concat_multi_camera!r}; expected {list(expected_size)!r}."
        )
    return video_size


def _append_robocasa_geometry_mismatches(
    mismatches: list[str], dataset: Mapping[str, Any], *, label: str
) -> None:
    try:
        _resolve_robocasa_image_size_from_dataset(dataset, label=label)
    except ValueError as exc:
        mismatches.append(str(exc))


def _robocasa_image_size_from_metadata(
    metadata: Mapping[str, Any], *, label: str
) -> tuple[int, int]:
    dataset = _require_dict(metadata.get("dataset"), label=f"{label}.dataset")
    return _resolve_robocasa_image_size_from_dataset(dataset, label=f"{label}.dataset")


def _robocasa_visual_preprocess_id_from_metadata(metadata: Mapping[str, Any]) -> str | None:
    dataset = _require_dict(metadata.get("dataset"), label="RoboCasa metadata.dataset")
    video_size = _resolve_robocasa_image_size_from_dataset(
        dataset, label="RoboCasa metadata.dataset"
    )
    family = robotwin_dreamfree._metadata_video_family(metadata, label="RoboCasa metadata")
    return required_robocasa_visual_preprocess_id(
        dataset_name=dataset.get("name"),
        concat_multi_camera=dataset.get("concat_multi_camera"),
        height=video_size[0],
        width=video_size[1],
        video_model_family=family,
    )


def _append_robocasa_visual_contract_mismatches(
    mismatches: list[str], metadata: Mapping[str, Any]
) -> None:
    expected = _robocasa_visual_preprocess_id_from_metadata(metadata)
    if expected is None:
        return
    _check_equal(mismatches, "visual_preprocess_id", metadata.get("visual_preprocess_id"), expected)
    _check_equal(
        mismatches,
        "source_stage1_visual_preprocess_id",
        metadata.get("source_stage1_visual_preprocess_id"),
        expected,
    )
    _check_equal(
        mismatches,
        "source_stage0_visual_preprocess_id",
        metadata.get("source_stage0_visual_preprocess_id"),
        expected,
    )


def _validate_robocasa_visual_lineage(
    *, stage1_metadata: Mapping[str, Any], stage2_metadata: Mapping[str, Any]
) -> None:
    expected = _robocasa_visual_preprocess_id_from_metadata(stage2_metadata)
    if expected is None:
        return None
    mismatches: list[str] = []
    _check_equal(
        mismatches,
        "Stage 1 visual_preprocess_id",
        stage1_metadata.get("visual_preprocess_id"),
        expected,
    )
    _check_equal(
        mismatches,
        "Stage 1 source_stage0_visual_preprocess_id",
        stage1_metadata.get("source_stage0_visual_preprocess_id"),
        expected,
    )
    if mismatches:
        raise ValueError(
            "RoboCasa direct-bilinear eval requires a newly trained Stage 0 -> Stage 1 -> Stage 2 visual lineage: "
            + "; ".join(mismatches)
        )


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
    robotwin_dreamfree._require_existing_files(
        checkpoint_path, DREAMFREE_STAGE2_REQUIRED_FILES, label="DreamFree Stage 2"
    )
    mismatches: list[str] = []
    _check_equal(
        mismatches,
        "checkpoint_type",
        metadata.get("checkpoint_type"),
        DREAMFREE_STAGE2_CHECKPOINT_TYPE,
    )
    _check_equal(
        mismatches,
        "checkpoint_version",
        metadata.get("checkpoint_version"),
        DREAMFREE_STAGE2_CHECKPOINT_VERSION,
    )
    if _is_none_like(metadata.get("source_stage1_checkpoint")):
        mismatches.append("source_stage1_checkpoint is missing")
    if int(metadata.get("source_stage1_global_step", -1)) < 0:
        mismatches.append(
            f"source_stage1_global_step must be non-negative, got {metadata.get('source_stage1_global_step')!r}"
        )
    family = robotwin_dreamfree._metadata_video_family(metadata, label="DreamFree Stage 2")
    expected_hidden_dim = robotwin_dreamfree._expected_hidden_dim_for_family(family)
    dataset = _require_dict(metadata.get("dataset"), label="DreamFree Stage 2 metadata.dataset")
    _append_robocasa_visual_contract_mismatches(mismatches, metadata)
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
    dataset_name = _normalize_robocasa_dataset_name(dataset.get("name"))
    if dataset_name not in ROBOCASA_DATASET_ALIASES:
        mismatches.append(
            f"dataset.name: actual={dataset.get('name')!r} expected one of {sorted(ROBOCASA_DATASET_ALIASES)!r}"
        )
    _check_equal(mismatches, "dataset.context_len", int(dataset.get("context_len", -1)), 128)
    _append_robocasa_geometry_mismatches(mismatches, dataset, label="dataset")
    _check_equal(mismatches, "dataset.action_dim", int(dataset.get("action_dim", -1)), 7)
    _check_equal(mismatches, "dataset.proprio_dim", int(dataset.get("proprio_dim", -1)), 9)
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
    _check_equal(mismatches, "readout.type", readout.get("type"), "latent_query_cross_attention")
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
    _check_equal(mismatches, "action.action_dim", int(action.get("action_dim", -1)), 7)
    _check_equal(mismatches, "action.proprio_dim", int(action.get("proprio_dim", -1)), 9)
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
        ("action.action_horizon", action.get("action_horizon")),
        ("action.scheduler.num_train_timesteps", scheduler.get("num_train_timesteps")),
    ):
        _as_positive_int(mismatches, label, value)
    if float(readout.get("mlp_ratio", -1.0)) <= 0.0:
        mismatches.append(f"readout.mlp_ratio must be positive, got {readout.get('mlp_ratio')!r}")
    if "train_shift" not in scheduler:
        mismatches.append("action.scheduler.train_shift is missing")
    if mismatches:
        raise ValueError(
            "DreamFree Stage 2 checkpoint is incompatible with RoboCasa DreamFree eval: "
            + "; ".join(mismatches)
        )


def _lookup_rgb(obs_data: Mapping[str, Any], key: str) -> Any:
    candidates = (key, f"{key}_camera", f"observation.images.{key}")
    for candidate in candidates:
        if candidate not in obs_data:
            continue
        value = obs_data[candidate]
        if isinstance(value, Mapping):
            if "rgb" not in value:
                raise KeyError(
                    f"RoboCasa obs key {candidate!r} is a mapping but has no 'rgb' entry."
                )
            return value["rgb"]
        return value
    raise KeyError(f"RoboCasa obs is missing RGB view {key!r}; tried {candidates}.")


def _rgb_to_chw_frame(rgb: Any) -> torch.Tensor:
    if isinstance(rgb, torch.Tensor):
        tensor = rgb.detach()
    else:
        tensor = torch.as_tensor(np.asarray(rgb))
    if tensor.ndim != 3:
        raise ValueError(f"RoboCasa RGB frame must be 3D HWC or CHW, got {tuple(tensor.shape)}.")
    if int(tensor.shape[-1]) == 3:
        tensor = tensor.permute(2, 0, 1)
    elif int(tensor.shape[0]) != 3:
        raise ValueError(f"RoboCasa RGB frame must have 3 channels, got {tuple(tensor.shape)}.")
    tensor = tensor.contiguous()
    if tensor.dtype == torch.uint8:
        tensor = tensor.to(torch.float32) / 255.0
    elif torch.is_floating_point(tensor):
        tensor = tensor.to(torch.float32)
        if float(tensor.max().item()) > 1.5:
            tensor = tensor / 255.0
    else:
        raise TypeError(f"RoboCasa RGB frame must be uint8 or floating, got {tensor.dtype}.")
    return tensor.unsqueeze(0)


def _rgb_to_chw_uint8_frame(rgb: Any) -> torch.Tensor:
    if isinstance(rgb, torch.Tensor):
        tensor = rgb.detach()
    else:
        tensor = torch.as_tensor(np.asarray(rgb))
    if tensor.ndim != 3:
        raise ValueError(
            f"RoboCasa direct-bilinear RGB frame must be 3D HWC or CHW, got {tuple(tensor.shape)}."
        )
    if int(tensor.shape[-1]) == 3:
        tensor = tensor.permute(2, 0, 1)
    elif int(tensor.shape[0]) != 3:
        raise ValueError(
            f"RoboCasa direct-bilinear RGB frame must have 3 channels, got {tuple(tensor.shape)}."
        )
    if tensor.dtype != torch.uint8:
        raise TypeError(
            f"RoboCasa direct-bilinear RGB frame must be uint8 before resize, got {tensor.dtype}."
        )
    if tuple(int(v) for v in tensor.shape[-2:]) != (224, 224):
        raise ValueError(
            f"RoboCasa direct-bilinear RGB frame must be raw 224x224, got {tuple(int(v) for v in tensor.shape[-2:])}."
        )
    return tensor.contiguous().unsqueeze(0)


def _preprocess_robocasa_camera_frame(
    frame: torch.Tensor, *, camera_size: tuple[int, int]
) -> torch.Tensor:
    if frame.ndim != 4 or int(frame.shape[1]) != 3:
        raise ValueError(f"Expected RoboCasa camera frame [1, 3, H, W], got {tuple(frame.shape)}.")
    return F.interpolate(
        frame, size=camera_size, mode="bilinear", align_corners=False, antialias=True
    )


def build_robocasa_image_tensor(
    obs_data: Mapping[str, Any],
    *,
    image_size: tuple[int, int] = (352, 352),
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    visual_preprocess_id: str | None = ROBOCASA_DREAMZERO_352_DIRECT_BILINEAR_V1,
) -> torch.Tensor:
    """Build one normalized RoboCasa DreamZero-layout image tensor."""
    (expected_height, expected_width) = _normalize_robocasa_image_size(
        image_size, label="image_size"
    )
    if visual_preprocess_id is not None:
        expected_ids = {
            (352, 352): ROBOCASA_DREAMZERO_352_DIRECT_BILINEAR_V1,
            (448, 448): ROBOCASA_DREAMZERO_448_DIRECT_BILINEAR_V1,
        }
        expected_id = expected_ids.get((expected_height, expected_width))
        if expected_id is None or visual_preprocess_id != expected_id:
            raise ValueError(f"Unsupported RoboCasa visual_preprocess_id={visual_preprocess_id!r}.")
        frames = [
            _rgb_to_chw_uint8_frame(_lookup_rgb(obs_data, key)) for key in ROBOCASA_EVAL_VIEW_KEYS
        ]
        image = normalize_uint8_video(
            compose_robocasa_dreamzero_uint8(
                torch.stack(frames, dim=0),
                output_size=(expected_height, expected_width),
                expected_raw_size=ROBOCASA_RAW_CAMERA_SIZE,
            )
        )
        if device is not None or dtype is not None:
            image = image.to(device=device, dtype=dtype)
        return image.contiguous()
    camera_size = (expected_height // 2, expected_width // 2)
    frames = [_rgb_to_chw_frame(_lookup_rgb(obs_data, key)) for key in ROBOCASA_EVAL_VIEW_KEYS]
    processed = [
        _preprocess_robocasa_camera_frame(frame, camera_size=camera_size) for frame in frames
    ]
    video = torch.stack(processed, dim=0)
    left = video[0]
    right = video[1]
    wrist = video[2]
    wrist_wide = wrist.repeat_interleave(2, dim=-1)
    bottom = torch.cat([left, right], dim=-1)
    image = torch.cat([wrist_wide, bottom], dim=-2)
    expected_shape = (1, 3, expected_height, expected_width)
    if tuple(image.shape) != expected_shape:
        raise ValueError(
            f"RoboCasa image tensor must be {list(expected_shape)}, got {tuple(image.shape)}."
        )
    image = image.clamp(0.0, 1.0) * 2.0 - 1.0
    if device is not None or dtype is not None:
        image = image.to(device=device, dtype=dtype)
    return image.contiguous()


class RoboCasaPolicy(robotwin_dreamfree.RobotWinPolicy):
    def __init__(
        self,
        *,
        flip_images: bool,
        robocasa_image_size: tuple[int, int],
        robocasa_visual_preprocess_id: str | None,
        compile_dit: bool,
        compile_dit_block_counts: Mapping[str, int],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.flip_images = bool(flip_images)
        self.robocasa_image_size = _normalize_robocasa_image_size(
            robocasa_image_size, label="robocasa_image_size"
        )
        self.robocasa_visual_preprocess_id = robocasa_visual_preprocess_id
        if self.robocasa_visual_preprocess_id is not None and (not self.flip_images):
            raise ValueError(
                "RoboCasa direct-bilinear evaluation requires flip_images=true to match cached training pixels."
            )
        self.compile_dit = bool(compile_dit)
        self.compile_dit_block_counts = dict(compile_dit_block_counts)
        logger.info(
            "Initialized DreamFree Stage 2 RoboCasa policy | stage1=%s | stage2=%s | stats=%s | image=%sx%s | horizon=%d | replan=%d | compile_dit=%s",
            self.stage1_checkpoint_path,
            self.stage2_checkpoint_path,
            self.dataset_stats_path,
            self.robocasa_image_size[0],
            self.robocasa_image_size[1],
            self.action_horizon,
            self.replan_steps,
            self.compile_dit,
        )

    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        prepared_obs = prepare_robocasa_observation(observation, flip_images=self.flip_images)
        image_tensor = build_robocasa_image_tensor(
            prepared_obs,
            image_size=self.robocasa_image_size,
            device=self.device,
            dtype=self.model_dtype,
            visual_preprocess_id=self.robocasa_visual_preprocess_id,
        )
        state_vector = extract_robocasa_state(observation)
        proprio = dreamfree_deploy._normalize_state_with_processor(self.processor, state_vector).to(
            device=self.device, dtype=self.model_dtype
        )
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        (context, context_mask, student_context_emb, student_context_mask, proprio_query_emb) = (
            self._build_student_readout_conditioning(
                image_tensor=image_tensor, instruction=instruction, proprio=proprio
            )
        )
        action_tensor = self._infer_dreamfree_action_tokens(
            context=context,
            context_mask=context_mask,
            student_context_emb=student_context_emb,
            student_context_mask=student_context_mask,
            proprio_query_emb=proprio_query_emb,
        )
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0
        action = dreamfree_deploy._denormalize_action_with_processor(self.processor, action_tensor)[
            0
        ]
        return action.astype(np.float32, copy=False)

    def predict_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        return self._infer_action_chunk(observation=observation, instruction=instruction)
