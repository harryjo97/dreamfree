"""LIBERO deployment wrapper for DreamFree Stage 2 student-native checkpoints."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image

import dreamfree.evaluation.libero._eval_common as eval_common
import dreamfree.evaluation.robotwin.policy as robotwin_dreamfree
import dreamfree.models.policy_loading as dreamfree_deploy

logger = logging.getLogger(__name__)
DREAMFREE_STAGE2_CHECKPOINT_TYPE = robotwin_dreamfree.DREAMFREE_STAGE2_CHECKPOINT_TYPE
DREAMFREE_STAGE2_CHECKPOINT_VERSION = robotwin_dreamfree.DREAMFREE_STAGE2_CHECKPOINT_VERSION
DREAMFREE_STAGE2_REQUIRED_FILES = robotwin_dreamfree.DREAMFREE_STAGE2_REQUIRED_FILES
LIBERO_EVAL_COMPATIBLE_DATASET_NAMES = {"libero"}


def _is_none_like(value: Any) -> bool:
    return dreamfree_deploy._is_none_like(value)


def _require_dict(value: Any, *, label: str) -> dict[str, Any]:
    return robotwin_dreamfree._require_dict(value, label=label)


def _check_equal(mismatches: list[str], name: str, actual: Any, expected: Any) -> None:
    robotwin_dreamfree._check_equal(mismatches, name, actual, expected)


def _check_libero_eval_dataset_name(mismatches: list[str], actual: Any) -> None:
    if actual in LIBERO_EVAL_COMPATIBLE_DATASET_NAMES:
        return
    expected = ", ".join(sorted(LIBERO_EVAL_COMPATIBLE_DATASET_NAMES))
    mismatches.append(f"dataset.name: actual={actual!r} expected one of [{expected}]")


def _as_positive_int(mismatches: list[str], name: str, value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        mismatches.append(f"{name} must be a positive integer, got {value!r}")
        return -1
    if parsed <= 0:
        mismatches.append(f"{name} must be positive, got {value!r}")
    return parsed


def _validate_dreamfree_stage2_trainable_scope_for_eval(
    mismatches: list[str],
    *,
    metadata: dict[str, Any],
    readout: dict[str, Any],
    trainable: dict[str, Any],
) -> None:
    for name, expected in {
        "student": False,
        "readout": True,
        "action_dit": True,
        "conditioner": True,
    }.items():
        _check_equal(mismatches, f"trainable_modules.{name}", bool(trainable.get(name)), expected)
    if metadata.get("training_variant") is not None:
        mismatches.append("Only standard stage 2 training checkpoints are supported.")


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
    _check_libero_eval_dataset_name(mismatches, dataset.get("name"))
    _check_equal(mismatches, "dataset.context_len", int(dataset.get("context_len", -1)), 128)
    _check_equal(mismatches, "dataset.video_size", dataset.get("video_size"), [224, 448])
    _check_equal(mismatches, "dataset.action_dim", int(dataset.get("action_dim", -1)), 7)
    _check_equal(mismatches, "dataset.proprio_dim", int(dataset.get("proprio_dim", -1)), 8)
    _check_equal(
        mismatches, "dataset.concat_multi_camera", dataset.get("concat_multi_camera"), "horizontal"
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
    _check_equal(mismatches, "action.proprio_dim", int(action.get("proprio_dim", -1)), 8)
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
    _validate_dreamfree_stage2_trainable_scope_for_eval(
        mismatches, metadata=metadata, readout=readout, trainable=trainable
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
            "DreamFree Stage 2 checkpoint is incompatible with LIBERO DreamFree eval: "
            + "; ".join(mismatches)
        )


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    (src_w, src_h) = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize(
        (round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR
    )
    (rw, rh) = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _get_libero_image(
    obs: dict[str, Any], *, image_preprocess_source: str
) -> dict[str, np.ndarray]:
    eval_common._normalize_image_preprocess_source(image_preprocess_source)
    image_slice = (slice(None, None, -1), slice(None, None, -1))
    return {
        "image": np.ascontiguousarray(obs["agentview_image"][image_slice]),
        "wrist_image": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][image_slice]),
    }


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(float(quat[3])) / den


def _invert_gripper_action(action: np.ndarray) -> np.ndarray:
    action[..., -1] = action[..., -1] * -1.0
    return action


def _is_numeric_sequence(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    if not value:
        return True
    first = value[0]
    if isinstance(first, (int, float)):
        return all(isinstance(item, (int, float)) for item in value)
    if isinstance(first, list):
        return all(_is_numeric_sequence(item) for item in value)
    return False


def _dataset_stats_to_tensors(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _dataset_stats_to_tensors(item) for (key, item) in value.items()}
    if _is_numeric_sequence(value):
        return torch.as_tensor(value, dtype=torch.float32)
    if isinstance(value, list):
        return [_dataset_stats_to_tensors(item) for item in value]
    return value


def _load_dataset_stats_for_libero_eval(dataset_stats_path: Path) -> dict[str, Any]:
    stats = eval_common._load_dataset_stats_json_for_eval(dataset_stats_path)
    return _dataset_stats_to_tensors(stats)


def _gripper_postprocess_name(action_data_source: str) -> str:
    eval_common._normalize_action_data_source(action_data_source)
    return "libero_01_to_signed_then_invert"


def _postprocess_libero_gripper(
    action: np.ndarray, *, action_data_source: str, binarize_gripper: bool
) -> np.ndarray:
    source = eval_common._normalize_action_data_source(action_data_source)
    if source == eval_common.LIBERO_ACTION_DATA_SOURCE:
        action[..., -1] = action[..., -1] * 2.0 - 1.0
        action = _invert_gripper_action(action)
    else:
        raise AssertionError(f"Unexpected LIBERO action_data_source: {source}")
    if binarize_gripper:
        action[..., -1] = np.sign(action[..., -1])
    return action


def _extract_libero_state(obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            _quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32).copy()),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    ).astype(np.float32)


class LiberoPolicy(robotwin_dreamfree.RobotWinPolicy):
    def __init__(
        self,
        *,
        input_h: int,
        input_w: int,
        concat_multi_camera: str,
        binarize_gripper: bool,
        action_data_source: str,
        image_preprocess_source: str,
        compile_dit: bool,
        compile_dit_block_counts: Mapping[str, int],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.input_h = int(input_h)
        self.input_w = int(input_w)
        self.concat_multi_camera = str(concat_multi_camera)
        self.binarize_gripper = bool(binarize_gripper)
        self.action_data_source = eval_common._normalize_action_data_source(action_data_source)
        self.image_preprocess_source = eval_common._normalize_image_preprocess_source(
            image_preprocess_source
        )
        self.gripper_postprocess = _gripper_postprocess_name(self.action_data_source)
        self.compile_dit = bool(compile_dit)
        self.compile_dit_block_counts = dict(compile_dit_block_counts)
        if self.input_h <= 0 or self.input_w <= 0:
            raise ValueError(
                f"LIBERO DreamFree input size must be positive, got {self.input_h}x{self.input_w}."
            )
        logger.info(
            "Initialized DreamFree Stage 2 LIBERO policy | stage1=%s | stage2=%s | stats=%s | action_source=%s | image_source=%s | gripper=%s | horizon=%d | replan=%d | compile_dit=%s",
            self.stage1_checkpoint_path,
            self.stage2_checkpoint_path,
            self.dataset_stats_path,
            self.action_data_source,
            self.image_preprocess_source,
            self.gripper_postprocess,
            self.action_horizon,
            self.replan_steps,
            self.compile_dit,
        )

    def _build_libero_image_tensor(self, obs: dict[str, Any]) -> torch.Tensor:
        imgs = _get_libero_image(obs, image_preprocess_source=self.image_preprocess_source)
        image_meta = self.processor.shape_meta["images"]
        num_cameras = int(self.processor.num_output_cameras)
        if len(image_meta) < num_cameras:
            raise ValueError(
                f"shape_meta.images has {len(image_meta)} entries, but num_output_cameras={num_cameras}."
            )

        def _meta_to_hw(meta: dict[str, Any], camera_idx: int) -> tuple[int, int]:
            shape = meta["shape"]
            if len(shape) != 3:
                raise ValueError(
                    f"shape_meta.images[{camera_idx}].shape must be [C,H,W], got {shape}"
                )
            return (int(shape[1]), int(shape[2]))

        if num_cameras == 1:
            (primary_h, primary_w) = _meta_to_hw(image_meta[0], camera_idx=0)
            rgb = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
        elif num_cameras == 2:
            (primary_h, primary_w) = _meta_to_hw(image_meta[0], camera_idx=0)
            (wrist_h, wrist_w) = _meta_to_hw(image_meta[1], camera_idx=1)
            primary = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
            wrist = _center_crop_resize(imgs["wrist_image"], width=wrist_w, height=wrist_h)
            if self.concat_multi_camera == "horizontal":
                rgb = np.concatenate([primary, wrist], axis=1)
            elif self.concat_multi_camera == "vertical":
                rgb = np.concatenate([primary, wrist], axis=0)
            else:
                raise ValueError(f"Invalid LIBERO concat_multi_camera: {self.concat_multi_camera}")
        else:
            raise ValueError(
                f"LIBERO DreamFree eval supports 1 or 2 output cameras, got {num_cameras}."
            )
        (actual_h, actual_w) = (int(rgb.shape[0]), int(rgb.shape[1]))
        if (actual_h, actual_w) != (self.input_h, self.input_w):
            raise ValueError(
                f"LIBERO DreamFree input image size mismatch after camera resize + concat: got (H,W)=({actual_h},{actual_w}), expected ({self.input_h},{self.input_w}); shape_meta.images={[meta['shape'] for meta in image_meta]}, concat_multi_camera={self.concat_multi_camera}."
            )
        image_tensor = (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self.device, dtype=self.model_dtype)
        )
        return image_tensor * (2.0 / 255.0) - 1.0

    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        image_tensor = self._build_libero_image_tensor(observation)
        state_vector = _extract_libero_state(observation)
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
        action = _postprocess_libero_gripper(
            action,
            action_data_source=self.action_data_source,
            binarize_gripper=self.binarize_gripper,
        )
        return action.astype(np.float32, copy=False)

    def predict_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        return self._infer_action_chunk(observation=observation, instruction=instruction)
