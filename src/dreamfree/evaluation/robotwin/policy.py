from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

import dreamfree.models.policy_loading as dreamfree_deploy
from dreamfree.data.text_cache import DEFAULT_PROMPT
from dreamfree.models.student_readout import (
    StudentLatentQueryReadout,
    readout_input_projection_mlp_ratio_from_metadata,
    readout_input_projection_type_from_metadata,
)
from dreamfree.models.wan22.future_state import (
    _lora_enabled,
    fixed_tokenwise_layer_norm,
    temporary_video_attention_mask_mode,
)
from dreamfree.models.wan22.helpers.distributed_load import (
    load_safetensors_rank0 as load_safetensors,
)
from dreamfree.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from dreamfree.models.wan22.schedulers.scheduler_unipc import FlowUniPCMultistepScheduler

logger = logging.getLogger(__name__)
DREAMFREE_STAGE2_CHECKPOINT_TYPE = "dreamfree_stage2"
DREAMFREE_STAGE2_CHECKPOINT_VERSION = 2
DREAMFREE_POLICY_RUNTIME = "dreamfree_stage2_student_native"
READOUT_TYPE_LATENT_QUERY = "latent_query_cross_attention"
SUPPORTED_READOUT_TYPES = {READOUT_TYPE_LATENT_QUERY}
ACTION_SAMPLER_EULER = "euler"
ACTION_SAMPLER_UNIPC = "unipc"
SUPPORTED_ACTION_SAMPLERS = {ACTION_SAMPLER_EULER, ACTION_SAMPLER_UNIPC}
STAGE1_SUPPORTED_CHECKPOINT_TYPES = dreamfree_deploy.STAGE1_SUPPORTED_CHECKPOINT_TYPES
DREAMFREE_STAGE2_REQUIRED_FILES = (
    "trainer_state.json",
    "action.safetensors",
    "readout.safetensors",
    "conditioner.safetensors",
    "proprio_encoder.safetensors",
)


def _json_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload)}.")
    return payload


def _is_none_like(value: Any) -> bool:
    return dreamfree_deploy._is_none_like(value)


def _require_dict(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object, got {type(value).__name__}.")
    return value


def _check_equal(mismatches: list[str], name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        mismatches.append(f"{name}: actual={actual!r} expected={expected!r}")


def _parse_int_list(value: Any) -> list[int]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
        if not text:
            return []
        items = [item.strip() for item in text.split(",")]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return []
    return [int(item) for item in items]


def _normalize_action_sampler(value: Any) -> str:
    sampler = str(value or ACTION_SAMPLER_EULER).strip().lower()
    if sampler not in SUPPORTED_ACTION_SAMPLERS:
        raise ValueError(
            f"Unsupported action_sampler={value!r}. Expected one of {sorted(SUPPORTED_ACTION_SAMPLERS)}."
        )
    return sampler


def _parse_optional_float(value: Any) -> float | None:
    if _is_none_like(value):
        return None
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError(f"Optional float setting must be positive when set, got {value!r}.")
    return parsed


def _parse_nonnegative_int_list(value: Any, *, label: str) -> list[int]:
    parsed = _parse_int_list(value)
    bad = [item for item in parsed if item < 0]
    if bad:
        raise ValueError(f"{label} must contain non-negative step indices, got {parsed}.")
    return parsed


def _readout_type_from_metadata(metadata: dict[str, Any]) -> str:
    readout = _require_dict(metadata.get("readout"), label="DreamFree Stage 2 metadata.readout")
    return str(readout.get("type", READOUT_TYPE_LATENT_QUERY))


def _condition_video_num_frames_from_stage2_metadata(metadata: dict[str, Any]) -> int:
    dataset = _require_dict(metadata.get("dataset"), label="DreamFree Stage 2 metadata.dataset")
    num_frames = int(dataset.get("num_frames", -1))
    action_video_freq_ratio = int(dataset.get("action_video_freq_ratio", -1))
    if num_frames <= 1 or action_video_freq_ratio <= 0:
        raise ValueError(
            f"DreamFree Stage 2 metadata.dataset must define positive num_frames and action_video_freq_ratio, got num_frames={num_frames} action_video_freq_ratio={action_video_freq_ratio}."
        )
    if (num_frames - 1) % action_video_freq_ratio != 0:
        raise ValueError(
            f"DreamFree Stage 2 metadata.dataset num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {action_video_freq_ratio}."
        )
    condition_video_num_frames = (num_frames - 1) // action_video_freq_ratio + 1
    if condition_video_num_frames % 4 != 1:
        raise ValueError(
            f"DreamFree Stage 2 deployment condition video frame count must satisfy T % 4 == 1, got {condition_video_num_frames} from num_frames={num_frames} action_video_freq_ratio={action_video_freq_ratio}."
        )
    return condition_video_num_frames


def _resolve_checkpoint_path(checkpoint_path: str) -> Path:
    resolved = Path(os.path.expanduser(os.path.expandvars(str(checkpoint_path)))).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {resolved}")
    return resolved


def _require_checkpoint_dir(path_value: Any, *, label: str) -> Path:
    if _is_none_like(path_value):
        raise ValueError(f"{label} checkpoint path is required.")
    path = _resolve_checkpoint_path(str(path_value))
    if path.is_file():
        raise ValueError(f"{label} checkpoint must be a directory, got file: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"{label} checkpoint directory not found: {path}")
    return path


def _require_existing_files(checkpoint_path: Path, names: tuple[str, ...], *, label: str) -> None:
    missing = [name for name in names if not (checkpoint_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {label} files {missing}: {checkpoint_path}")


def _require_existing_file(path: Path, *, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Expected {label} to be a file: {path}")
    return path


def _path_identity(path_value: Any) -> Path | None:
    if _is_none_like(path_value):
        return None
    return Path(str(path_value)).expanduser().resolve()


def _build_dreamfree_stage2_cfg_from_metadata(metadata: dict[str, Any]) -> SimpleNamespace:
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise ValueError("DreamFree Stage 2 checkpoint metadata is missing dict `config`.")
    return SimpleNamespace(**config)


def _metadata_video_family(metadata: dict[str, Any], *, label: str) -> str:
    return dreamfree_deploy._metadata_video_family(metadata, label=label)


def _expected_hidden_dim_for_family(family: str) -> int:
    return dreamfree_deploy._expected_hidden_dim_for_family(family)


def _expected_model_id_for_family(family: str) -> str:
    return dreamfree_deploy._expected_model_id_for_family(family)


def _readout_input_layer_j_from_metadata(metadata: dict[str, Any]) -> int:
    student = _require_dict(metadata.get("student"), label="DreamFree Stage 2 metadata.student")
    readout = _require_dict(metadata.get("readout"), label="DreamFree Stage 2 metadata.readout")
    return int(readout.get("input_layer_j", student.get("layer_j", -1)))


def _validate_stage1_for_dreamfree(checkpoint_path: Path, metadata: dict[str, Any]) -> None:
    _require_existing_files(
        checkpoint_path,
        dreamfree_deploy.stage1_required_student_files_for_metadata(metadata),
        label="Stage 1",
    )
    mismatches: list[str] = []
    checkpoint_type = str(metadata.get("checkpoint_type", "")).strip()
    if checkpoint_type not in STAGE1_SUPPORTED_CHECKPOINT_TYPES:
        mismatches.append(
            f"checkpoint_type: actual={checkpoint_type!r} expected one of {sorted(STAGE1_SUPPORTED_CHECKPOINT_TYPES)!r}"
        )
    metadata = dreamfree_deploy._stage1_student_view_metadata(metadata)
    family = _metadata_video_family(metadata, label="Stage 1")
    expected_hidden_dim = _expected_hidden_dim_for_family(family)
    student = _require_dict(metadata.get("student"), label="Stage 1 metadata.student")
    shapes = _require_dict(metadata.get("shapes"), label="Stage 1 metadata.shapes")
    config = _require_dict(metadata.get("config"), label="Stage 1 metadata.config")
    model_contract = metadata.get("model_contract")
    if not isinstance(model_contract, dict):
        model_contract = {}
    _check_equal(
        mismatches,
        "student.video_attention_mask_mode",
        student.get("video_attention_mask_mode"),
        "first_frame_causal",
    )
    _check_equal(mismatches, "student.hook_point", student.get("hook_point"), "post_block")
    _check_equal(
        mismatches, "config.tokenizer_max_len", int(config.get("tokenizer_max_len", -1)), 128
    )
    _check_equal(mismatches, "config.text_dim", int(config.get("text_dim", -1)), 4096)
    _check_equal(
        mismatches,
        "config.video_hidden_dim",
        int(config.get("video_hidden_dim", -1)),
        expected_hidden_dim,
    )
    _check_equal(
        mismatches, "config.model_id", config.get("model_id"), _expected_model_id_for_family(family)
    )
    for key in ("tokens_per_frame", "D_student", "latent_shape"):
        if key not in shapes:
            mismatches.append(f"shapes.{key} is missing")
    if "tokens_per_frame" in shapes and int(shapes["tokens_per_frame"]) <= 0:
        mismatches.append(
            f"shapes.tokens_per_frame must be positive, got {shapes['tokens_per_frame']!r}."
        )
    if "D_student" in shapes:
        _check_equal(mismatches, "shapes.D_student", int(shapes["D_student"]), expected_hidden_dim)
    if "latent_shape" in shapes and len(tuple(shapes["latent_shape"])) != 4:
        mismatches.append(f"shapes.latent_shape must be [C,T,H,W], got {shapes['latent_shape']!r}.")
    if family == dreamfree_deploy.WAN22_A14B_FAMILY:
        _check_equal(
            mismatches,
            "model_contract.video_latent_dim",
            int(model_contract.get("video_latent_dim", config.get("video_latent_dim", -1))),
            16,
        )
        _check_equal(
            mismatches,
            "model_contract.video_dit_input_dim",
            int(model_contract.get("video_dit_input_dim", config.get("video_dit_input_dim", -1))),
            36,
        )
        _check_equal(
            mismatches,
            "model_contract.video_dit_output_dim",
            int(model_contract.get("video_dit_output_dim", config.get("video_dit_output_dim", -1))),
            16,
        )
        _check_equal(
            mismatches,
            "student.i2v_condition_mode",
            student.get("i2v_condition_mode"),
            dreamfree_deploy.WAN22_A14B_I2V_MODE,
        )
    global_step = int(metadata.get("global_step", -1))
    if global_step < 0:
        mismatches.append(f"global_step must be non-negative, got {global_step}.")
    if mismatches:
        raise ValueError(
            "Stage 1 checkpoint is incompatible with DreamFree eval: " + "; ".join(mismatches)
        )


def _validate_dreamfree_source_stage1_matches_stage2(
    *,
    stage1_checkpoint_path: Path,
    stage1_metadata: dict[str, Any],
    stage2_metadata: dict[str, Any],
) -> None:
    _validate_stage1_for_dreamfree(stage1_checkpoint_path, stage1_metadata)
    stage1_metadata = dreamfree_deploy._stage1_student_view_metadata(stage1_metadata)
    stage1_student = _require_dict(stage1_metadata.get("student"), label="Stage 1 metadata.student")
    stage1_shapes = _require_dict(stage1_metadata.get("shapes"), label="Stage 1 metadata.shapes")
    stage2_student = _require_dict(
        stage2_metadata.get("student"), label="DreamFree Stage 2 metadata.student"
    )
    source_student = _require_dict(
        stage2_metadata.get("source_stage1_student_metadata"),
        label="DreamFree Stage 2 metadata.source_stage1_student_metadata",
    )
    stage1_family = _metadata_video_family(stage1_metadata, label="Stage 1")
    stage2_family = _metadata_video_family(stage2_metadata, label="DreamFree Stage 2")
    mismatches: list[str] = []
    _check_equal(
        mismatches,
        "source_stage1_checkpoint",
        stage1_checkpoint_path.resolve(),
        _path_identity(stage2_metadata.get("source_stage1_checkpoint")),
    )
    _check_equal(
        mismatches,
        "source_stage1_global_step",
        int(stage1_metadata.get("global_step", -1)),
        int(stage2_metadata.get("source_stage1_global_step", -1)),
    )
    _check_equal(
        mismatches,
        "Stage 1 video_model_family vs DreamFree Stage 2 video_model_family",
        stage1_family,
        stage2_family,
    )
    _check_equal(
        mismatches,
        "Stage 1 layer_j vs DreamFree student.layer_j",
        int(stage1_student.get("layer_j", -1)),
        int(stage2_student.get("layer_j", -1)),
    )
    _check_equal(
        mismatches,
        "Stage 1 hook_point vs DreamFree student.hook_point",
        stage1_student.get("hook_point"),
        stage2_student.get("hook_point"),
    )
    _check_equal(
        mismatches,
        "Stage 1 tokens_per_frame vs DreamFree student.tokens_per_frame",
        int(stage1_shapes.get("tokens_per_frame", -1)),
        int(stage2_student.get("tokens_per_frame", -1)),
    )
    _check_equal(
        mismatches,
        "Stage 1 D_student vs DreamFree student.hidden_dim",
        int(stage1_shapes.get("D_student", -1)),
        int(stage2_student.get("hidden_dim", -1)),
    )
    _check_equal(
        mismatches,
        "Stage 1 i2v_condition_mode vs DreamFree source",
        str(stage1_student.get("i2v_condition_mode", "none")),
        str(source_student.get("i2v_condition_mode", "none")),
    )
    _check_equal(
        mismatches,
        "Stage 0 checkpoint identity",
        _path_identity(stage1_metadata.get("stage0_lora_checkpoint")),
        _path_identity(stage2_metadata.get("stage0_lora_checkpoint")),
    )
    if mismatches:
        raise ValueError(
            "Stage 1 source checkpoint cannot be composed with DreamFree Stage 2 eval: "
            + "; ".join(mismatches)
        )


class RobotWinPolicy(dreamfree_deploy.RobotWinPolicyBase):
    def __init__(
        self,
        *,
        processor: Any,
        video_core: Any,
        readout: StudentLatentQueryReadout,
        action_model: Any,
        conditioner: Any,
        stage1_metadata: dict[str, Any],
        stage2_metadata: dict[str, Any],
        stage1_checkpoint_path: Path,
        stage2_checkpoint_path: Path,
        dataset_stats_path: Path,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        readout_active_num_queries: int | None = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        timing_enabled: bool = False,
        text_embedding_cache_dir: Any = None,
        action_sampler: str = ACTION_SAMPLER_EULER,
        action_infer_shift: float | None = None,
        action_unipc_solver_order: int = 2,
        action_unipc_solver_type: str = "bh2",
        action_unipc_lower_order_final: bool = True,
        action_unipc_disable_corrector: Sequence[int] | None = None,
        compile_dit: bool = False,
        compile_dit_block_counts: dict[str, int] | None = None,
    ) -> None:
        super().__init__(
            processor=processor,
            action_horizon=action_horizon,
            replan_steps=replan_steps,
            timing_enabled=timing_enabled,
        )
        if int(num_inference_steps) <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}.")
        if int(action_unipc_solver_order) <= 0:
            raise ValueError(
                f"`action_unipc_solver_order` must be positive, got {action_unipc_solver_order}."
            )
        if str(action_unipc_solver_type) not in {"bh1", "bh2"}:
            raise ValueError(
                f"`action_unipc_solver_type` must be 'bh1' or 'bh2', got {action_unipc_solver_type!r}."
            )
        action_infer_shift = None if action_infer_shift is None else float(action_infer_shift)
        if action_infer_shift is not None and action_infer_shift <= 0.0:
            raise ValueError(
                f"`action_infer_shift` must be positive when set, got {action_infer_shift}."
            )
        disable_corrector = (
            []
            if action_unipc_disable_corrector is None
            else [int(idx) for idx in action_unipc_disable_corrector]
        )
        if any(idx < 0 for idx in disable_corrector):
            raise ValueError(
                f"`action_unipc_disable_corrector` must contain non-negative step indices, got {disable_corrector}."
            )
        self.video_core = video_core.eval()
        self.readout = readout.eval()
        self.action_model = action_model.eval()
        self.conditioner = conditioner.eval()
        self.stage1_metadata = stage1_metadata
        self.stage2_metadata = stage2_metadata
        self.stage1_checkpoint_path = stage1_checkpoint_path
        self.stage2_checkpoint_path = stage2_checkpoint_path
        self.dataset_stats_path = dataset_stats_path
        self.device = torch.device(self.video_core.device)
        self.model_dtype = self.video_core.torch_dtype
        self.num_inference_steps = int(num_inference_steps)
        self.seed = seed
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.action_sampler = _normalize_action_sampler(action_sampler)
        self.action_infer_shift = action_infer_shift
        self.action_unipc_solver_order = int(action_unipc_solver_order)
        self.action_unipc_solver_type = str(action_unipc_solver_type)
        self.action_unipc_lower_order_final = bool(action_unipc_lower_order_final)
        self.action_unipc_disable_corrector = disable_corrector
        self.compile_dit = bool(compile_dit)
        self.compile_dit_block_counts = dict(compile_dit_block_counts or {})
        self._prompt_context_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.text_embedding_cache_dirs = dreamfree_deploy._parse_text_embedding_cache_dirs(
            text_embedding_cache_dir
        )
        stage2_student = _require_dict(
            stage2_metadata.get("student"), label="DreamFree Stage 2 metadata.student"
        )
        stage2_readout = _require_dict(
            stage2_metadata.get("readout"), label="DreamFree Stage 2 metadata.readout"
        )
        stage2_action = _require_dict(
            stage2_metadata.get("action"), label="DreamFree Stage 2 metadata.action"
        )
        scheduler_meta = _require_dict(
            stage2_action.get("scheduler"), label="DreamFree Stage 2 metadata.action.scheduler"
        )
        if isinstance(stage1_metadata.get("config"), dict):
            stage1_config = _require_dict(
                stage1_metadata.get("config"), label="Stage 1 metadata.config"
            )
        else:
            stage1_config = _require_dict(
                stage2_metadata.get("config"), label="DreamFree Stage 2 metadata.config"
            )
        self.text_context_len = int(stage1_config.get("tokenizer_max_len", 128))
        self.text_context_dim = int(stage1_config.get("text_dim", 4096))
        family_metadata = (
            stage1_metadata if isinstance(stage1_metadata.get("config"), dict) else stage2_metadata
        )
        stage1_family = dreamfree_deploy._metadata_video_family(family_metadata, label="Stage 1")
        self.text_context_model_id = str(
            stage1_config.get("model_id")
            or dreamfree_deploy._expected_model_id_for_family(stage1_family)
        )
        self.video_model_family = stage1_family
        self.condition_video_num_frames = 1
        if stage1_family == dreamfree_deploy.WAN22_A14B_FAMILY:
            self.condition_video_num_frames = _condition_video_num_frames_from_stage2_metadata(
                stage2_metadata
            )
            stage1_shapes = _require_dict(
                stage1_metadata.get("shapes"), label="Stage 1 metadata.shapes"
            )
            stage1_latent_shape = tuple(int(v) for v in stage1_shapes.get("latent_shape", ()))
            if len(stage1_latent_shape) != 4:
                raise ValueError(
                    f"Stage 1 metadata.shapes.latent_shape must be [C,T,H,W], got {stage1_latent_shape}."
                )
            vae_temporal_stride = int(getattr(self.video_core.vae, "temporal_downsample_factor", 4))
            expected_condition_latent_t = (
                int(self.condition_video_num_frames) - 1
            ) // vae_temporal_stride + 1
            if expected_condition_latent_t != int(stage1_latent_shape[1]):
                raise ValueError(
                    f"Deployment 14B/Stage 1 latent-time contract mismatch: condition_video_num_frames={self.condition_video_num_frames} vae_temporal_stride={vae_temporal_stride} gives latent_t={expected_condition_latent_t}, but Stage 1 latent_shape={stage1_latent_shape}."
                )
        self.student_lora_adapter_name = getattr(self.video_core, "student_lora_adapter_name", None)
        self.student_layer_idx = _readout_input_layer_j_from_metadata(stage2_metadata)
        self.student_hook_point = str(stage2_student["hook_point"])
        self.student_i2v_condition_mode = str(stage2_student.get("i2v_condition_mode", "none"))
        self.expected_student_tokens_per_frame = int(stage2_student["tokens_per_frame"])
        self.expected_student_hidden_dim = int(stage2_student["hidden_dim"])
        self.readout_type = _readout_type_from_metadata(stage2_metadata)
        self.max_readout_tokens = int(stage2_readout["num_queries"])
        if readout_active_num_queries is not None:
            raise ValueError(
                f"EVALUATION.readout_active_num_queries is supported only for nested readout checkpoints; checkpoint readout.type={self.readout_type!r}."
            )
        self.expected_readout_tokens = self.max_readout_tokens
        self.expected_readout_hidden_dim = int(stage2_readout["output_dim"])
        self.action_output_parameterization = str(stage2_action.get("output_parameterization", ""))
        self.action_loss_parameterization = str(stage2_action.get("loss_parameterization", ""))
        self.action_scheduler_family = str(scheduler_meta.get("family", ""))
        self.action_num_train_timesteps = int(scheduler_meta.get("num_train_timesteps", 1000))
        self.action_train_shift = float(scheduler_meta["train_shift"])
        self.proprio_state_timestep_sigma = float(
            stage2_action.get("proprio_conditioning", {}).get("state_timestep_sigma", 0.001)
        )
        self.action_dim = int(stage2_action.get("action_dim", self.action_model.action_dim))
        if not self.text_embedding_cache_dirs and (
            self.video_core.text_encoder is None or self.video_core.tokenizer is None
        ):
            raise ValueError(
                "DreamFree Stage 2 eval requires a loaded Wan text encoder and tokenizer."
            )
        if self.action_output_parameterization != "velocity":
            raise ValueError(
                f"DreamFree Stage 2 eval requires action.output_parameterization='velocity', got {self.action_output_parameterization!r}."
            )
        if self.action_loss_parameterization != "flow_matching":
            raise ValueError(
                f"DreamFree Stage 2 eval requires action.loss_parameterization='flow_matching', got {self.action_loss_parameterization!r}."
            )
        if self.action_scheduler_family != "continuous_flow_matching":
            raise ValueError(
                f"DreamFree Stage 2 eval requires action.scheduler.family='continuous_flow_matching', got {self.action_scheduler_family!r}."
            )
        self.action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=self.action_num_train_timesteps, shift=self.action_train_shift
        )
        logger.info(
            "Initialized DreamFree Stage 2 RobotWin policy | stage1=%s | stage2=%s | stats=%s | horizon=%d | replan=%d | action_sampler=%s",
            stage1_checkpoint_path,
            stage2_checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
            self.action_sampler,
        )

    def _sample_randn(self, shape: tuple[int, ...]) -> torch.Tensor:
        generator = (
            None
            if self.seed is None
            else torch.Generator(device=self.rand_device).manual_seed(self.seed)
        )
        return torch.randn(
            shape, generator=generator, device=self.rand_device, dtype=torch.float32
        ).to(device=self.device, dtype=self.model_dtype)

    def _build_prompt_context(self, instruction: str) -> tuple[str, torch.Tensor, torch.Tensor]:
        prompt = DEFAULT_PROMPT.format(task=instruction)
        cached = self._prompt_context_cache.get(prompt)
        if cached is not None:
            (context, context_mask) = cached
            return (prompt, context, context_mask)
        with torch.no_grad():
            if self.text_embedding_cache_dirs:
                (context, context_mask) = (
                    dreamfree_deploy.load_cached_text_context_for_action_model(
                        prompt=prompt,
                        cache_dirs=self.text_embedding_cache_dirs,
                        context_len=self.text_context_len,
                        text_dim=self.text_context_dim,
                        model_id=self.text_context_model_id,
                        device=self.device,
                        dtype=self.model_dtype,
                    )
                )
            else:
                (context, context_mask) = self.video_core.encode_prompt(prompt)
                context = context.to(device=self.device, dtype=self.model_dtype)
                context_mask = context_mask.to(device=self.device, dtype=torch.bool)
                (context, context_mask) = dreamfree_deploy._prepare_text_context_for_action_model(
                    context, context_mask
                )
        context = context.detach()
        context_mask = context_mask.detach()
        self._prompt_context_cache[prompt] = (context, context_mask)
        return (prompt, context, context_mask)

    def _student_inference_kwargs(self):
        return {}

    def _extract_student_tokens(
        self,
        *,
        first_frame_latents: torch.Tensor,
        i2v_condition_latents: torch.Tensor | None = None,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        if first_frame_latents.ndim != 5 or int(first_frame_latents.shape[2]) != 1:
            raise ValueError(
                f"DreamFree Stage 2 student extraction requires first-frame latents [B,C,1,H,W], got {tuple(first_frame_latents.shape)}."
            )
        timestep = torch.zeros(
            (first_frame_latents.shape[0],),
            device=first_frame_latents.device,
            dtype=self.model_dtype,
        )
        with (
            torch.no_grad(),
            _lora_enabled(self.video_core.dit, getattr(self, "student_lora_adapter_name", None)),
            temporary_video_attention_mask_mode(self.video_core.dit, "first_frame_causal"),
        ):
            active_mode = str(getattr(self.video_core.dit, "video_attention_mask_mode", ""))
            if active_mode != "first_frame_causal":
                raise RuntimeError(
                    f"DreamFree Stage 2 student extraction requires first_frame_causal mask mode, got {active_mode!r}."
                )
            condition_first_frame = None
            if (
                str(
                    getattr(self.video_core, "video_model_family", dreamfree_deploy.WAN22_5B_FAMILY)
                )
                == dreamfree_deploy.WAN22_A14B_FAMILY
            ):
                if (
                    not isinstance(i2v_condition_latents, torch.Tensor)
                    or i2v_condition_latents.ndim != 5
                ):
                    raise ValueError(
                        "14B deployment current-only student extraction requires `i2v_condition_latents` from official pseudo-video VAE encoding."
                    )
                condition_first_frame = i2v_condition_latents[:, :, 0:1]
                if tuple(condition_first_frame.shape) != tuple(first_frame_latents.shape):
                    raise ValueError(
                        f"14B deployment I2V condition first-frame shape must match first_frame_latents: condition={tuple(condition_first_frame.shape)} latents={tuple(first_frame_latents.shape)}."
                    )
            model_latents = dreamfree_deploy._i2v_model_input_for_core(
                core=self.video_core,
                latents=first_frame_latents,
                i2v_condition_latents=condition_first_frame,
            )
            out = self.video_core.dit.forward_hidden_tokens(
                x=model_latents,
                timestep=timestep,
                context=context,
                context_mask=None,
                action=None,
                fuse_vae_embedding_in_latents=bool(
                    getattr(self.video_core.dit, "fuse_vae_embedding_in_latents", False)
                ),
                layer_idx=self.student_layer_idx,
                hook_point=self.student_hook_point,
                **self._student_inference_kwargs(),
            )
        tokens = out["tokens"]
        meta = out["meta"]
        expected_shape = (
            int(first_frame_latents.shape[0]),
            self.expected_student_tokens_per_frame,
            self.expected_student_hidden_dim,
        )
        if int(meta.get("tokens_per_frame", -1)) != self.expected_student_tokens_per_frame:
            raise ValueError(
                f"DreamFree Stage 2 student tokens_per_frame mismatch: actual={meta.get('tokens_per_frame')} expected={self.expected_student_tokens_per_frame}."
            )
        if int(meta.get("sequence_length", -1)) != self.expected_student_tokens_per_frame:
            raise ValueError(
                f"DreamFree Stage 2 current-only student path must return exactly one frame of tokens, got sequence_length={meta.get('sequence_length')}."
            )
        if tuple(tokens.shape) != expected_shape:
            raise ValueError(
                f"DreamFree Stage 2 student token shape mismatch: actual={tuple(tokens.shape)} expected={expected_shape}."
            )
        return tokens.detach()

    def _encode_observation_latents(
        self, image_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return dreamfree_deploy._encode_first_frame_latents_for_deploy(
            core=self.video_core,
            input_image=image_tensor,
            condition_video_num_frames=getattr(self, "condition_video_num_frames", 1),
            tiled=self.tiled,
        )

    def _conditioner_inference_kwargs(self):
        return {}

    def _build_student_readout_conditioning(
        self, *, image_tensor: torch.Tensor, instruction: str, proprio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (_, context, context_mask) = self._build_prompt_context(instruction)
        (first_frame_latents, i2v_condition_latents) = self._encode_observation_latents(
            image_tensor
        )
        extract_kwargs = {
            "first_frame_latents": first_frame_latents,
            "context": context,
            "context_mask": context_mask,
        }
        if i2v_condition_latents is not None:
            extract_kwargs["i2v_condition_latents"] = i2v_condition_latents
        student_tokens = self._extract_student_tokens(**extract_kwargs)
        student_tokens_cond = fixed_tokenwise_layer_norm(student_tokens.float()).to(
            dtype=self.model_dtype
        )
        with torch.no_grad():
            (readout_tokens, readout_mask) = self.readout(student_tokens_cond)
            conditioned = self.conditioner(
                readout_tokens,
                context,
                context_mask,
                proprio.unsqueeze(1),
                **self._conditioner_inference_kwargs(),
            )
        expected_readout_shape = (
            int(image_tensor.shape[0]),
            self.expected_readout_tokens,
            self.expected_readout_hidden_dim,
        )
        if tuple(readout_tokens.shape) != expected_readout_shape:
            raise ValueError(
                f"DreamFree Stage 2 readout token shape mismatch: actual={tuple(readout_tokens.shape)} expected={expected_readout_shape}."
            )
        if tuple(readout_mask.shape) != (int(image_tensor.shape[0]), self.expected_readout_tokens):
            raise ValueError(
                f"DreamFree Stage 2 readout mask shape mismatch: tokens={tuple(readout_tokens.shape)} mask={tuple(readout_mask.shape)}."
            )
        return (
            context,
            context_mask,
            conditioned["teacher_context_emb"],
            conditioned["teacher_context_mask"],
            conditioned["proprio_query_emb"],
        )

    def _forward_action_model_with_conditioning(
        self,
        *,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        student_context_emb: torch.Tensor,
        student_context_mask: torch.Tensor,
        proprio_query_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        prefix_len = int(proprio_query_emb.shape[1])
        if prefix_len <= 0:
            raise ValueError("DreamFree Stage 2 query prefix must contribute at least one token.")
        if tuple(student_context_mask.shape) != (
            student_context_emb.shape[0],
            student_context_emb.shape[1],
        ):
            raise ValueError(
                f"DreamFree Stage 2 student context mask shape mismatch: context={tuple(student_context_emb.shape)} mask={tuple(student_context_mask.shape)}."
            )
        pred_sequence = self.action_model.forward_with_query_prefix_and_extra_context_emb(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            query_prefix_emb=proprio_query_emb,
            query_prefix_timestep_sigma=self.proprio_state_timestep_sigma,
            extra_context_emb=student_context_emb,
            extra_context_mask=student_context_mask,
            num_train_timesteps=float(self.action_num_train_timesteps),
        )
        return (pred_sequence, prefix_len)

    def _predict_action_velocity(
        self,
        *,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        student_context_emb: torch.Tensor,
        student_context_mask: torch.Tensor,
        proprio_query_emb: torch.Tensor,
    ) -> torch.Tensor:
        (pred_sequence, prefix_len) = self._forward_action_model_with_conditioning(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            student_context_emb=student_context_emb,
            student_context_mask=student_context_mask,
            proprio_query_emb=proprio_query_emb,
        )
        pred_velocity = pred_sequence[:, prefix_len:, :]
        if tuple(pred_velocity.shape) != tuple(action_tokens.shape):
            raise ValueError(
                f"Predicted DreamFree Stage 2 velocity shape mismatch: pred={tuple(pred_velocity.shape)} target={tuple(action_tokens.shape)}."
            )
        return pred_velocity

    def _infer_dreamfree_action_tokens_euler(
        self,
        *,
        action_tokens: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        student_context_emb: torch.Tensor,
        student_context_mask: torch.Tensor,
        proprio_query_emb: torch.Tensor,
    ) -> torch.Tensor:
        (infer_timesteps, infer_deltas) = self.action_scheduler.build_inference_schedule(
            num_inference_steps=self.num_inference_steps,
            device=self.device,
            dtype=self.model_dtype,
            shift_override=self.action_infer_shift,
        )
        with torch.no_grad():
            for step_t, step_delta in zip(infer_timesteps, infer_deltas):
                timestep = step_t.unsqueeze(0).to(device=self.device, dtype=self.model_dtype)
                pred_velocity = self._predict_action_velocity(
                    action_tokens=action_tokens,
                    timestep=timestep,
                    context=context,
                    context_mask=context_mask,
                    student_context_emb=student_context_emb,
                    student_context_mask=student_context_mask,
                    proprio_query_emb=proprio_query_emb,
                )
                action_tokens = self.action_scheduler.step(pred_velocity, step_delta, action_tokens)
        return action_tokens

    def _infer_dreamfree_action_tokens_unipc(
        self,
        *,
        action_tokens: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        student_context_emb: torch.Tensor,
        student_context_mask: torch.Tensor,
        proprio_query_emb: torch.Tensor,
    ) -> torch.Tensor:
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.action_num_train_timesteps,
            solver_order=self.action_unipc_solver_order,
            prediction_type="flow_prediction",
            shift=1.0,
            use_dynamic_shifting=False,
            predict_x0=True,
            solver_type=self.action_unipc_solver_type,
            lower_order_final=self.action_unipc_lower_order_final,
            disable_corrector=self.action_unipc_disable_corrector,
            final_sigmas_type="zero",
        )
        infer_shift = (
            self.action_train_shift if self.action_infer_shift is None else self.action_infer_shift
        )
        sample_scheduler.set_timesteps(
            self.num_inference_steps, device=self.device, shift=infer_shift
        )
        with torch.no_grad():
            for step_t in sample_scheduler.timesteps:
                timestep = step_t.reshape(1).to(device=self.device, dtype=self.model_dtype)
                pred_velocity = self._predict_action_velocity(
                    action_tokens=action_tokens,
                    timestep=timestep,
                    context=context,
                    context_mask=context_mask,
                    student_context_emb=student_context_emb,
                    student_context_mask=student_context_mask,
                    proprio_query_emb=proprio_query_emb,
                )
                action_tokens = sample_scheduler.step(
                    model_output=pred_velocity,
                    timestep=step_t,
                    sample=action_tokens,
                    return_dict=False,
                )[0]
        return action_tokens

    def _infer_dreamfree_action_tokens(
        self,
        *,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        student_context_emb: torch.Tensor,
        student_context_mask: torch.Tensor,
        proprio_query_emb: torch.Tensor,
    ) -> torch.Tensor:
        action_tokens = self._sample_randn((1, self.action_horizon, self.action_dim))
        if self.action_sampler == ACTION_SAMPLER_UNIPC:
            return self._infer_dreamfree_action_tokens_unipc(
                action_tokens=action_tokens,
                context=context,
                context_mask=context_mask,
                student_context_emb=student_context_emb,
                student_context_mask=student_context_mask,
                proprio_query_emb=proprio_query_emb,
            )
        if self.action_sampler != ACTION_SAMPLER_EULER:
            raise ValueError(f"Unsupported action_sampler={self.action_sampler!r}.")
        return self._infer_dreamfree_action_tokens_euler(
            action_tokens=action_tokens,
            context=context,
            context_mask=context_mask,
            student_context_emb=student_context_emb,
            student_context_mask=student_context_mask,
            proprio_query_emb=proprio_query_emb,
        )

    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        image_tensor = dreamfree_deploy._build_robotwin_image_tensor(
            observation, device=self.device, dtype=self.model_dtype
        )
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
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
        return dreamfree_deploy._denormalize_action_with_processor(self.processor, action_tensor)[0]

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        action_chunk = self._infer_action_chunk(observation=observation, instruction=instruction)
        self._enqueue_action_chunk(action_chunk)


def _build_readout_for_eval(
    *, metadata: dict[str, Any], checkpoint_path: Path, device: torch.device, dtype: torch.dtype
) -> torch.nn.Module:
    readout_meta = _require_dict(
        metadata.get("readout"), label="DreamFree Stage 2 metadata.readout"
    )
    readout_type = str(readout_meta.get("type", READOUT_TYPE_LATENT_QUERY))
    common_kwargs = {
        "student_dim": int(readout_meta["input_dim"]),
        "readout_dim": int(readout_meta["readout_dim"]),
        "output_dim": int(readout_meta["output_dim"]),
        "num_layers": int(readout_meta["num_layers"]),
        "num_heads": int(readout_meta["num_heads"]),
        "mlp_ratio": float(readout_meta["mlp_ratio"]),
        "dropout": float(readout_meta["dropout"]),
        "input_projection_type": readout_input_projection_type_from_metadata(readout_meta),
        "input_projection_mlp_ratio": readout_input_projection_mlp_ratio_from_metadata(
            readout_meta
        ),
    }
    if readout_type == READOUT_TYPE_LATENT_QUERY:
        readout = StudentLatentQueryReadout(
            **common_kwargs, num_queries=int(readout_meta["num_queries"])
        )
    else:
        raise ValueError(f"Unsupported DreamFree Stage 2 readout.type={readout_type!r}.")
    readout = readout.to(device=device, dtype=dtype)
    readout.load_state_dict(
        load_safetensors(str(checkpoint_path / "readout.safetensors"), device="cpu"), strict=True
    )
    readout.eval()
    return readout
