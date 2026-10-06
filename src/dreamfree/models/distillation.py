from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from dreamfree.data.stage1_cache import _validate_stage1_input_latents_contract
from dreamfree.models.artifacts import _optional_path_identity, _stage0_lora_checkpoint_identity
from dreamfree.models.compile import compile_dit_blocks
from dreamfree.models.wan22.future_state import (
    FirstFrameToTeacherStateProjector,
    FutureStateTeacher,
    StudentFirstFramePredictor,
    fixed_tokenwise_layer_norm,
)
from dreamfree.models.wan22.i2v_condition import WAN22_A14B_I2V_CONDITION_LATENTS_KEY
from dreamfree.models.wan22.stage0_loader import (
    Stage0MergedBackbone,
    stage0_train_sigma_config_from_checkpoint,
    stage0_train_sigma_config_from_metadata,
)
from dreamfree.training.stage1.config import Stage1Config, Stage1StudentSigmaConfig
from dreamfree.training.stage1.constants import (
    STUDENT_SIGMA_POLICIES,
    STUDENT_SIGMA_POLICY_UNIFORM_RANGE,
    STUDENT_SIGMA_POLICY_WAN_SHIFT_FULL,
    STUDENT_SIGMA_POLICY_WAN_SHIFT_RANGE,
    STUDENT_SIGMA_SOURCE_CONFIG_OVERRIDE,
    STUDENT_SIGMA_SOURCE_STAGE0_CHECKPOINT,
    STUDENT_SIGMA_SOURCE_UNUSED,
    TEACHER_TARGET_TOKEN_SCOPE_ALL,
    TEACHER_TARGET_TOKEN_SCOPES,
    ScalarMetric,
)


def _normalize_stage1_student_sigma_policy(value: Any) -> str | None:
    text = _optional_path_identity(None if value is None else str(value))
    if text is None:
        return None
    normalized = text.strip().lower().replace("-", "_")
    if normalized == "uniform":
        normalized = STUDENT_SIGMA_POLICY_UNIFORM_RANGE
    if normalized not in STUDENT_SIGMA_POLICIES:
        raise ValueError(
            f"Unsupported student_sigma_policy={value!r}. Expected one of: {sorted(STUDENT_SIGMA_POLICIES)}."
        )
    return normalized


def _validate_stage1_student_sigma_config(
    cfg: Stage1Config, sigma_config: Stage1StudentSigmaConfig
) -> None:
    policy = _normalize_stage1_student_sigma_policy(sigma_config.sigma_policy)
    if policy != sigma_config.sigma_policy:
        raise ValueError(
            f"Internal student sigma policy is not normalized: {sigma_config.sigma_policy!r}."
        )
    if policy == STUDENT_SIGMA_POLICY_WAN_SHIFT_FULL:
        return
    if sigma_config.sigma_min is None or sigma_config.sigma_max is None:
        raise ValueError(
            f"student_sigma_policy={policy!r} requires student sigma bounds, got student_sigma_min={sigma_config.sigma_min}, student_sigma_max={sigma_config.sigma_max}."
        )
    sigma_min = float(sigma_config.sigma_min)
    sigma_max = float(sigma_config.sigma_max)
    if not math.isfinite(sigma_min) or not math.isfinite(sigma_max):
        raise ValueError(
            f"Stage 1 student sigma bounds must be finite, got student_sigma_min={sigma_config.sigma_min}, student_sigma_max={sigma_config.sigma_max}."
        )
    if not 0.0 <= sigma_min <= sigma_max <= 1.0:
        raise ValueError(
            f"Stage 1 student sigma bounds must satisfy 0 <= student_sigma_min <= student_sigma_max <= 1, got {sigma_min}, {sigma_max}."
        )
    if policy == STUDENT_SIGMA_POLICY_WAN_SHIFT_RANGE and (not sigma_max < 1.0):
        raise ValueError(
            f"student_sigma_policy='wan_shift_range' requires student_sigma_max < 1, got student_sigma_min={sigma_min}, student_sigma_max={sigma_max}."
        )


def resolve_stage1_student_sigma_config(
    cfg: Stage1Config,
    *,
    stage0_trainer_state: dict[str, Any] | None = None,
    stage0_metadata_path: Path | None = None,
    require_stage0_metadata: bool = True,
) -> Stage1StudentSigmaConfig:
    min_set = cfg.student_sigma_min is not None
    max_set = cfg.student_sigma_max is not None
    if min_set != max_set:
        raise ValueError(
            f"student_sigma_min and student_sigma_max must be set together, got student_sigma_min={cfg.student_sigma_min}, student_sigma_max={cfg.student_sigma_max}."
        )
    requested_policy = _normalize_stage1_student_sigma_policy(cfg.student_sigma_policy)
    if min_set:
        sigma_config = Stage1StudentSigmaConfig(
            sigma_policy=requested_policy or STUDENT_SIGMA_POLICY_WAN_SHIFT_RANGE,
            sigma_min=float(cfg.student_sigma_min),
            sigma_max=float(cfg.student_sigma_max),
            source=STUDENT_SIGMA_SOURCE_CONFIG_OVERRIDE,
        )
        _validate_stage1_student_sigma_config(cfg, sigma_config)
        return sigma_config
    if requested_policy is not None:
        if requested_policy != STUDENT_SIGMA_POLICY_WAN_SHIFT_FULL:
            raise ValueError(
                f"student_sigma_policy={requested_policy!r} requires explicit student_sigma_min/student_sigma_max."
            )
        sigma_config = Stage1StudentSigmaConfig(
            sigma_policy=STUDENT_SIGMA_POLICY_WAN_SHIFT_FULL,
            sigma_min=None,
            sigma_max=None,
            source=STUDENT_SIGMA_SOURCE_CONFIG_OVERRIDE,
        )
        _validate_stage1_student_sigma_config(cfg, sigma_config)
        return sigma_config
    checkpoint = _stage0_lora_checkpoint_identity(cfg.stage0_lora_checkpoint)
    if checkpoint is None:
        return Stage1StudentSigmaConfig(
            sigma_policy=STUDENT_SIGMA_POLICY_WAN_SHIFT_FULL,
            sigma_min=None,
            sigma_max=None,
            source=STUDENT_SIGMA_SOURCE_UNUSED,
        )
    if stage0_trainer_state is not None:
        stage0_sigma_config = stage0_train_sigma_config_from_metadata(
            stage0_trainer_state,
            metadata_path=stage0_metadata_path or Path(checkpoint) / "trainer_state.json",
        )
    else:
        metadata_path = Path(checkpoint) / "trainer_state.json"
        if not metadata_path.exists() and (not require_stage0_metadata):
            return Stage1StudentSigmaConfig(
                sigma_policy=STUDENT_SIGMA_POLICY_WAN_SHIFT_FULL,
                sigma_min=None,
                sigma_max=None,
                source=STUDENT_SIGMA_SOURCE_UNUSED,
            )
        stage0_sigma_config = stage0_train_sigma_config_from_checkpoint(
            checkpoint, stage_label="Stage 1 student"
        )
    policy = _normalize_stage1_student_sigma_policy(stage0_sigma_config.train_sigma_policy)
    if policy is None:
        raise ValueError(
            f"Stage 0 checkpoint did not resolve to a train sigma policy: {checkpoint}"
        )
    sigma_config = Stage1StudentSigmaConfig(
        sigma_policy=policy,
        sigma_min=None
        if policy == STUDENT_SIGMA_POLICY_WAN_SHIFT_FULL
        else float(stage0_sigma_config.sigma_min),
        sigma_max=None
        if policy == STUDENT_SIGMA_POLICY_WAN_SHIFT_FULL
        else float(stage0_sigma_config.sigma_max),
        source=STUDENT_SIGMA_SOURCE_STAGE0_CHECKPOINT,
    )
    _validate_stage1_student_sigma_config(cfg, sigma_config)
    return sigma_config


def sample_stage1_student_timestep(
    train_scheduler: Any,
    sigma_config: Stage1StudentSigmaConfig,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    policy = sigma_config.sigma_policy
    if policy == STUDENT_SIGMA_POLICY_WAN_SHIFT_FULL:
        return train_scheduler.sample_training_t(batch_size=batch_size, device=device, dtype=dtype)
    if policy == STUDENT_SIGMA_POLICY_WAN_SHIFT_RANGE:
        if sigma_config.sigma_min is None or sigma_config.sigma_max is None:
            raise ValueError("wan_shift_range student sigma sampling requires sigma_min/sigma_max.")
        return train_scheduler.sample_training_t_bounded(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            sigma_min=float(sigma_config.sigma_min),
            sigma_max=float(sigma_config.sigma_max),
        )
    if policy == STUDENT_SIGMA_POLICY_UNIFORM_RANGE:
        if sigma_config.sigma_min is None or sigma_config.sigma_max is None:
            raise ValueError("uniform_range student sigma sampling requires sigma_min/sigma_max.")
        sigma_min = float(sigma_config.sigma_min)
        sigma_max = float(sigma_config.sigma_max)
        if sigma_min == sigma_max:
            sigma = torch.full((batch_size,), sigma_min, device=device, dtype=torch.float32)
        else:
            sigma = torch.empty((batch_size,), device=device, dtype=torch.float32).uniform_(
                sigma_min, sigma_max
            )
        timestep = sigma * float(train_scheduler.num_train_timesteps)
        return timestep.to(dtype=dtype)
    raise ValueError(f"Unsupported Stage 1 student sigma policy: {policy!r}.")


def build_stage1_cached_inputs(
    core: Any, sample: dict[str, Any], tiled: bool = False
) -> dict[str, Any]:
    """Validate Stage 1 batch inputs and return frozen VAE latents.

    Args:
        core: Wan video core that owns the VAE, DiT, scheduler, device, and dtype.
        sample: Collated dataset batch with cached `context` and
            `context_mask`, plus exactly one of `input_latents` or `video`.
            Cached `input_latents` must have shape `[B, C, T, H, W]`; live
            `video` must have shape `[B, 3, T, H, W]`.
        tiled: Whether to use tiled VAE encoding.

    Returns:
        Dictionary containing device/dtype-normalized `context`
        `[B, 128, 4096]`, boolean `context_mask` `[B, 128]`,
        optional `dit_context_mask` for DiT cross-attention,
        `input_latents` `[B, C, T_latent, H_latent, W_latent]`,
        optional `i2v_condition_latents` `[B, 16, T_latent, H_latent, W_latent]`,
        optional `first_frame_latents` `[B, C, 1, H_latent, W_latent]`,
        and optional metadata.

    Raises:
        TypeError: If tensor inputs have the wrong type.
        ValueError: If required cache fields are missing or tensor shapes are
            incompatible with the Stage 1 setup.
    """
    if "context" not in sample or "context_mask" not in sample:
        raise ValueError(
            "Stage 1 training requires cached `context` and `context_mask`; online prompt encoding is disabled."
        )
    has_latents = "input_latents" in sample
    has_video = "video" in sample
    is_14b = str(getattr(core, "video_model_family", "wan22_ti2v_5b")) == "wan22_i2v_a14b"
    if has_latents and has_video:
        raise ValueError(
            "Stage 1 batches must provide either `input_latents` or `video`, not both."
        )
    if not has_latents and (not has_video):
        raise ValueError(
            "Stage 1 training requires `input_latents` from the video latent cache or live `video`."
        )
    if has_latents:
        latents = sample["input_latents"]
        if not isinstance(latents, torch.Tensor):
            raise TypeError(f"sample['input_latents'] must be a torch.Tensor, got {type(latents)}.")
        if latents.ndim != 5:
            raise ValueError(
                f"sample['input_latents'] must have shape [B,C,T,H,W], got {tuple(latents.shape)}."
            )
        batch_size = int(latents.shape[0])
    else:
        video = sample["video"]
        if not isinstance(video, torch.Tensor):
            raise TypeError(f"sample['video'] must be a torch.Tensor, got {type(video)}.")
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(
                f"sample['video'] must have shape [B,3,T,H,W], got {tuple(video.shape)}."
            )
        (batch_size, _, num_frames, height, width) = video.shape
        if num_frames % 4 != 1:
            raise ValueError(f"Video frame count must satisfy T % 4 == 1, got {num_frames}.")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}."
            )
    context = sample["context"]
    context_mask = sample["context_mask"]
    if (
        not isinstance(context, torch.Tensor)
        or context.ndim != 3
        or tuple(context.shape[1:]) != (128, 4096)
    ):
        raise ValueError(
            f"sample['context'] must have shape [B,128,4096], got {tuple(context.shape)}."
        )
    if (
        not isinstance(context_mask, torch.Tensor)
        or context_mask.ndim != 2
        or tuple(context_mask.shape[1:]) != (128,)
    ):
        raise ValueError(
            f"sample['context_mask'] must have shape [B,128], got {tuple(context_mask.shape)}."
        )
    if context.shape[0] != batch_size or context_mask.shape[0] != batch_size:
        raise ValueError(
            f"Cached context batch must match video batch={batch_size}, got context={context.shape[0]} mask={context_mask.shape[0]}."
        )
    context_mask_bool = context_mask.to(dtype=torch.bool)
    dit_context_mask = (
        None
        if context_mask_bool.device.type == "cpu" and bool(context_mask_bool.all().item())
        else context_mask_bool.to(device=core.device, dtype=torch.bool)
    )
    if has_latents:
        input_latents = latents.to(
            device=core.device, dtype=core.torch_dtype, non_blocking=True
        ).detach()
        if is_14b:
            i2v_condition_raw = sample.get(WAN22_A14B_I2V_CONDITION_LATENTS_KEY)
            if not isinstance(i2v_condition_raw, torch.Tensor):
                raise ValueError(
                    f"14B Stage 1 cached-latent batches require `{WAN22_A14B_I2V_CONDITION_LATENTS_KEY}` from official pseudo-video VAE encoding."
                )
            if tuple(i2v_condition_raw.shape) != tuple(latents.shape):
                raise ValueError(
                    f"`{WAN22_A14B_I2V_CONDITION_LATENTS_KEY}` must match input_latents shape, got {tuple(i2v_condition_raw.shape)} vs {tuple(latents.shape)}."
                )
            i2v_condition_latents = i2v_condition_raw.to(
                device=core.device, dtype=core.torch_dtype, non_blocking=True
            ).detach()
        else:
            i2v_condition_latents = None
    else:
        input_video = video.to(device=core.device, dtype=core.torch_dtype, non_blocking=True)
        with torch.no_grad():
            input_latents = core._encode_video_latents(input_video, tiled=tiled).detach()
            i2v_condition_latents = (
                core._encode_14b_i2v_condition_latents(input_video, tiled=tiled).detach()
                if is_14b
                else None
            )
    first_frame_latents = None
    fuse_flag = False
    if getattr(core.dit, "fuse_vae_embedding_in_latents", False):
        first_frame_latents = input_latents[:, :, 0:1]
        fuse_flag = True
    inputs = {
        "context": context.to(device=core.device, dtype=core.torch_dtype, non_blocking=True),
        "context_mask": context_mask_bool.to(
            device=core.device, dtype=torch.bool, non_blocking=True
        ),
        "dit_context_mask": dit_context_mask,
        "input_latents": input_latents,
        "i2v_condition_latents": i2v_condition_latents,
        "first_frame_latents": first_frame_latents,
        "fuse_vae_embedding_in_latents": fuse_flag,
    }
    for key in ("prompt", "task", "sample_id", "episode_index", "start_frame"):
        if key in sample:
            inputs[key] = sample[key]
    return inputs


def _normalize_teacher_target_token_scope(token_scope: Any) -> str:
    scope = str(token_scope).strip().lower()
    if scope not in TEACHER_TARGET_TOKEN_SCOPES:
        raise ValueError(
            f"teacher_target_token_scope must be one of {sorted(TEACHER_TARGET_TOKEN_SCOPES)}, got {token_scope!r}."
        )
    return scope


def _teacher_target_token_slice_bounds(
    *, meta: dict[str, Any], token_scope: str, sequence_length: int | None = None
) -> tuple[int, int]:
    scope = _normalize_teacher_target_token_scope(token_scope)
    tokens_per_frame = int(meta.get("tokens_per_frame", 0))
    seq_len = int(
        sequence_length if sequence_length is not None else meta.get("sequence_length", 0)
    )
    if tokens_per_frame <= 0:
        raise ValueError(
            f"Teacher target token selection requires positive tokens_per_frame, got {tokens_per_frame}."
        )
    if seq_len <= 0:
        raise ValueError(
            f"Teacher target token selection requires positive sequence_length, got {seq_len}."
        )
    if tokens_per_frame > seq_len:
        raise ValueError(
            f"Teacher target token selection requires tokens_per_frame <= sequence_length, got tokens_per_frame={tokens_per_frame}, sequence_length={seq_len}."
        )
    if seq_len % tokens_per_frame != 0:
        raise ValueError(
            f"Teacher target token selection requires sequence_length divisible by tokens_per_frame, got sequence_length={seq_len}, tokens_per_frame={tokens_per_frame}."
        )
    if scope == TEACHER_TARGET_TOKEN_SCOPE_ALL:
        return (0, seq_len)
    if seq_len <= tokens_per_frame:
        raise ValueError(
            f"teacher_target_token_scope='future' requires at least one future frame, got sequence_length={seq_len}, tokens_per_frame={tokens_per_frame}."
        )
    return (tokens_per_frame, seq_len)


def _select_teacher_target_tokens(
    tokens: torch.Tensor, *, meta: dict[str, Any], token_scope: str
) -> tuple[torch.Tensor, dict[str, int | str]]:
    if tokens.ndim != 3:
        raise ValueError(
            f"Teacher target token selection expects tokens [B,S,D], got {tuple(tokens.shape)}."
        )
    scope = _normalize_teacher_target_token_scope(token_scope)
    (start, end) = _teacher_target_token_slice_bounds(
        meta=meta, token_scope=scope, sequence_length=int(tokens.shape[1])
    )
    selected = tokens[:, start:end, :]
    if int(selected.shape[1]) <= 0:
        raise ValueError(
            f"teacher_target_token_scope={scope!r} selected no tokens from input shape {tuple(tokens.shape)}."
        )
    return (
        selected,
        {
            "teacher_target_token_scope": scope,
            "teacher_full_sequence_length": int(tokens.shape[1]),
            "teacher_target_start_token": int(start),
            "teacher_target_end_token": int(end),
            "teacher_target_sequence_length": int(selected.shape[1]),
        },
    )


def teacher_state_prediction_loss(
    *,
    student_prediction: torch.Tensor,
    teacher_target: torch.Tensor,
    tokens_per_frame: int,
    projector_input_tokens: int,
) -> tuple[torch.Tensor, dict[str, ScalarMetric], dict[str, int]]:
    """Compute DreamFree Stage 1 teacher-state loss with REPA-style cosine distance."""
    if student_prediction.ndim != 3:
        raise ValueError(
            f"`student_prediction` must be 3D [B,S,D], got {tuple(student_prediction.shape)}."
        )
    if teacher_target.ndim != 3:
        raise ValueError(f"`teacher_target` must be 3D [B,S,D], got {tuple(teacher_target.shape)}.")
    if tuple(student_prediction.shape) != tuple(teacher_target.shape):
        raise ValueError(
            f"Projector/teacher token shape mismatch: student={tuple(student_prediction.shape)} teacher={tuple(teacher_target.shape)}"
        )
    tokens_per_frame = int(tokens_per_frame)
    projector_input_tokens = int(projector_input_tokens)
    if tokens_per_frame <= 0:
        raise ValueError(f"`tokens_per_frame` must be positive, got {tokens_per_frame}.")
    if projector_input_tokens != tokens_per_frame:
        raise ValueError(
            f"Projector input tokens must equal tokens_per_frame={tokens_per_frame}, got {projector_input_tokens}."
        )
    pred_raw = student_prediction.float()
    target_raw = teacher_target.detach().float()
    pred = fixed_tokenwise_layer_norm(pred_raw)
    target = fixed_tokenwise_layer_norm(target_raw)
    teacher_sequence_length = int(target_raw.shape[1])
    projector_output_tokens = int(pred_raw.shape[1])
    with torch.no_grad():
        raw_cosine = F.cosine_similarity(
            pred_raw.detach().flatten(0, 1), target_raw.flatten(0, 1), dim=-1
        ).mean()
        raw_loss = torch.ones_like(raw_cosine) - raw_cosine
        teacher_target_norm = target_raw.norm(dim=-1).mean().float()
        student_prediction_norm = pred_raw.detach().norm(dim=-1).mean().float()
    del target_raw
    normalized_cosine = F.cosine_similarity(pred.flatten(0, 1), target.flatten(0, 1), dim=-1).mean()
    loss = torch.ones_like(normalized_cosine) - normalized_cosine
    metrics = {
        "loss_teacher_state": loss.detach().float(),
        "loss_teacher_state_raw": raw_loss,
        "teacher_target_norm": teacher_target_norm,
        "student_prediction_norm": student_prediction_norm,
        "teacher_student_cosine": raw_cosine,
        "teacher_student_cosine_normalized": normalized_cosine.detach().float(),
    }
    shape_meta = {
        "tokens_per_frame": tokens_per_frame,
        "teacher_sequence_length": teacher_sequence_length,
        "projector_input_tokens": projector_input_tokens,
        "projector_output_tokens": projector_output_tokens,
    }
    return (loss, metrics, shape_meta)


class DistillationModel(torch.nn.Module):
    """Wrap the Wan core, teacher/student paths, and projector loss."""

    def __init__(
        self,
        backbone: Stage0MergedBackbone,
        projector: FirstFrameToTeacherStateProjector,
        cfg: Stage1Config,
    ):
        """Attach Stage 1 teacher, student, and projector modules.

        Args:
            backbone: Stage0-merged Wan backbone that owns the core DiT, VAE,
                schedulers, and Stage 1 LoRA metadata.
            projector: Trainable decoder that maps first-frame student tokens
                to the teacher token layout.
            cfg: Stage 1 configuration selecting hook layers, attention modes,
                noise policy, and loss weights.
        """
        super().__init__()
        self.core = backbone.core
        self.projector = projector
        self.cfg = cfg
        self.stage0_checkpoint_dir = str(backbone.checkpoint_dir)
        self.stage0_trainer_state = backbone.trainer_state
        self.stage0_lora_config = backbone.stage0_lora_config
        self.stage1_lora_config = backbone.stage1_lora_config
        self.student_init_mode = str(backbone.student_init_mode)
        self.stage0_merged_into_student_base = bool(backbone.stage0_merged_into_student_base)
        self.teacher_lora_adapter_name = backbone.teacher_lora_adapter_name
        self.student_lora_adapter_name = backbone.student_lora_adapter_name
        self.student_sigma_config = resolve_stage1_student_sigma_config(
            cfg,
            stage0_trainer_state=backbone.trainer_state,
            stage0_metadata_path=Path(backbone.checkpoint_dir) / "trainer_state.json",
        )
        self.teacher = FutureStateTeacher(
            self.core,
            layer_idx=int(cfg.teacher_layer_k),
            hook_point=str(cfg.teacher_hook_point),
            target_type=str(cfg.teacher_target_type),
            sigma_policy=str(cfg.teacher_sigma_policy),
            sigma_min=float(cfg.teacher_sigma_min),
            sigma_max=float(cfg.teacher_sigma_max),
            video_attention_mask_mode=str(cfg.teacher_video_attention_mask_mode),
            i2v_condition_mode=str(cfg.teacher_i2v_condition_mode),
            frame0_mode=str(cfg.teacher_frame0_mode),
            lora_adapter_name=self.teacher_lora_adapter_name,
        )
        self.student = StudentFirstFramePredictor(
            self.core,
            layer_idx=int(cfg.student_layer_j),
            hook_point=str(cfg.student_hook_point),
            input_mode=str(cfg.student_input_mode),
            video_attention_mask_mode=str(cfg.student_video_attention_mask_mode),
            future_token_attention=str(cfg.future_token_attention),
            text_condition_scope=str(cfg.student_text_scope),
            i2v_condition_mode=str(cfg.student_i2v_condition_mode),
            lora_adapter_name=self.student_lora_adapter_name,
        )
        self.last_shape_metadata: dict[str, Any] = {}
        self.set_stage1_train_mode()

    def set_stage1_train_mode(self) -> None:
        """Restore the intended Stage 1 train/eval modes after evaluation.

        The DiT module stays in train mode for the student path and Stage 1
        LoRA, while the VAE and optional text encoder stay in eval mode because
        their outputs are frozen for this stage.
        """
        self.train()
        self.core.train()
        self.core.dit.train()
        if self.core.vae is not None:
            self.core.vae.eval()
        elif not bool(self.cfg.use_cached_latents):
            raise RuntimeError(
                "Stage 1 live-video training requires core.vae, but the model was loaded without a VAE."
            )
        if self.core.text_encoder is not None:
            self.core.text_encoder.eval()
        self.projector.train()

    def training_loss(self, sample: dict[str, Any]) -> tuple[torch.Tensor, dict[str, ScalarMetric]]:
        """Compute total Stage 1 training loss for one collated batch.

        Args:
            sample: Collated dataset batch with `video` `[B, 3, T, H, W]`,
                cached `context` `[B, 128, 4096]`, cached `context_mask`
                `[B, 128]`, and optional metadata fields.

        Returns:
            Tuple of scalar total loss and metrics for logging.

        Raises:
            ValueError: If teacher and student token layouts are inconsistent.
        """
        if bool(self.cfg.use_cached_latents):
            if "video" in sample:
                raise ValueError(
                    "Stage 1 cached-latent training received live `video`; cached batches must be latent-only."
                )
            if "input_latents" not in sample:
                raise ValueError(
                    "Stage 1 cached-latent training requires `input_latents` in every batch. Precompute a full video latent cache or set use_cached_latents=false."
                )
        inputs = build_stage1_cached_inputs(self.core, sample)
        z_clean = inputs["input_latents"]
        if bool(self.cfg.use_cached_latents):
            _validate_stage1_input_latents_contract(z_clean, self.cfg)
        context = inputs["context"]
        dit_context_mask = inputs["dit_context_mask"]
        batch_size = int(z_clean.shape[0])
        eps_fm = torch.randn_like(z_clean)
        timestep_fm = sample_stage1_student_timestep(
            self.core.train_scheduler,
            self.student_sigma_config,
            batch_size=batch_size,
            device=z_clean.device,
            dtype=z_clean.dtype,
        )
        target = self.core.train_scheduler.training_target(z_clean, eps_fm, timestep_fm)
        student = self.student(
            z_clean=z_clean,
            context=context,
            context_mask=dit_context_mask,
            eps_fm=eps_fm,
            timestep_fm=timestep_fm,
            first_frame_latents=inputs["first_frame_latents"],
            fuse_vae_embedding_in_latents=bool(inputs["fuse_vae_embedding_in_latents"]),
            i2v_condition_latents=inputs["i2v_condition_latents"],
        )
        pred = student["pred"]
        if inputs["first_frame_latents"] is not None:
            pred = pred[:, :, 1:]
            target = target[:, :, 1:]
        loss_per_sample = F.mse_loss(pred.float(), target.float(), reduction="none").mean(
            dim=(1, 2, 3, 4)
        )
        if stage1_student_uses_wan_shift_weight(self.student_sigma_config):
            sample_weight = self.core.train_scheduler.training_weight(timestep_fm).to(
                loss_per_sample.device, dtype=loss_per_sample.dtype
            )
            loss_video = (loss_per_sample * sample_weight).mean()
        else:
            loss_video = loss_per_sample.mean()
        teacher = self.teacher(
            z_clean=z_clean,
            context=context,
            context_mask=dit_context_mask,
            i2v_condition_latents=inputs["i2v_condition_latents"],
        )
        y_tgt_full = teacher["tokens"]
        z_world = student["z_world"]
        teacher_meta = teacher["meta"]
        student_meta = student["meta"]
        if int(teacher_meta["tokens_per_frame"]) != int(student_meta["tokens_per_frame"]):
            raise ValueError(
                f"Teacher/student tokens_per_frame mismatch: teacher={teacher_meta}, student={student_meta}."
            )
        if int(z_world.shape[1]) != int(teacher_meta["tokens_per_frame"]):
            raise ValueError(
                f"Projector input must be first-frame tokens only: z_world={tuple(z_world.shape)} teacher_meta={teacher_meta}."
            )
        hat_y_full = self.projector(z_world, target_meta=teacher_meta)
        (y_tgt, target_scope_meta) = _select_teacher_target_tokens(
            y_tgt_full, meta=teacher_meta, token_scope=str(self.cfg.teacher_target_token_scope)
        )
        (hat_y, pred_scope_meta) = _select_teacher_target_tokens(
            hat_y_full, meta=teacher_meta, token_scope=str(self.cfg.teacher_target_token_scope)
        )
        if target_scope_meta != pred_scope_meta:
            raise ValueError(
                f"Teacher target token scope metadata mismatch: target={target_scope_meta} prediction={pred_scope_meta}."
            )
        (loss_teacher_state, teacher_state_metrics, shape_meta) = teacher_state_prediction_loss(
            student_prediction=hat_y,
            teacher_target=y_tgt,
            tokens_per_frame=int(teacher_meta["tokens_per_frame"]),
            projector_input_tokens=int(z_world.shape[1]),
        )
        loss_total = loss_video + float(self.cfg.lambda_teacher_state) * loss_teacher_state
        sigma = teacher["sigma"].detach().float()
        metrics = {
            "loss_video_fm": loss_video.detach().float(),
            "video_fm_timestep_mean": timestep_fm.detach().float().mean(),
        }
        metrics.update(teacher_state_metrics)
        metrics.update(
            {
                "loss_total": loss_total.detach().float(),
                "teacher_sigma_mean": sigma.mean(),
                "teacher_sigma_min": sigma.min(),
                "teacher_sigma_max": sigma.max(),
                "tokens_per_frame": float(shape_meta["tokens_per_frame"]),
                "teacher_sequence_length": float(shape_meta["teacher_sequence_length"]),
                "teacher_full_sequence_length": float(
                    target_scope_meta["teacher_full_sequence_length"]
                ),
                "teacher_target_start_token": float(
                    target_scope_meta["teacher_target_start_token"]
                ),
                "teacher_target_end_token": float(target_scope_meta["teacher_target_end_token"]),
                "projector_input_tokens": float(shape_meta["projector_input_tokens"]),
                "projector_output_tokens": float(shape_meta["projector_output_tokens"]),
            }
        )
        self.last_shape_metadata = {
            "latent_shape": [int(v) for v in z_clean.shape[1:]],
            "teacher_grid_size": [int(v) for v in teacher_meta["grid_size"]],
            "student_grid_size": [int(v) for v in student_meta["grid_size"]],
            "tokens_per_frame": int(teacher_meta["tokens_per_frame"]),
            "teacher_sequence_length": int(shape_meta["teacher_sequence_length"]),
            "teacher_target_token_scope": str(target_scope_meta["teacher_target_token_scope"]),
            "teacher_full_sequence_length": int(target_scope_meta["teacher_full_sequence_length"]),
            "teacher_target_start_token": int(target_scope_meta["teacher_target_start_token"]),
            "teacher_target_end_token": int(target_scope_meta["teacher_target_end_token"]),
            "teacher_target_sequence_length": int(
                target_scope_meta["teacher_target_sequence_length"]
            ),
            "projector_input_tokens": int(shape_meta["projector_input_tokens"]),
            "projector_output_tokens": int(shape_meta["projector_output_tokens"]),
            "D_student": int(z_world.shape[-1]),
            "D_teacher": int(y_tgt.shape[-1]),
            "teacher_state_loss": str(self.cfg.teacher_state_loss),
        }
        return (loss_total, metrics)

    def forward(self, sample: dict[str, Any]) -> tuple[torch.Tensor, dict[str, ScalarMetric]]:
        """Delegate module calls to `training_loss`.

        Args:
            sample: Collated Stage 1 training batch.

        Returns:
            Tuple of scalar total loss and metric dictionary.
        """
        return self.training_loss(sample)


def _compile_stage1_dit_blocks(wrapper: DistillationModel) -> int:
    """Compile shared teacher/student DiT blocks without replacing modules."""
    return compile_dit_blocks(wrapper.core.dit, label="Stage 1")


def stage1_student_uses_wan_shift_weight(sigma_config: Stage1StudentSigmaConfig) -> bool:
    return sigma_config.sigma_policy == STUDENT_SIGMA_POLICY_WAN_SHIFT_FULL
