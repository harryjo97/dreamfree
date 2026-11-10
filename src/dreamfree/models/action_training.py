from __future__ import annotations

import contextlib
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamfree.data.action_batch import _TimingRecorder, prepare_action_batch
from dreamfree.data.action_cache import _validate_stage2_input_latents_contract
from dreamfree.models import action_history
from dreamfree.models.compile import compile_dit_blocks
from dreamfree.models.conditioning import TeacherStateActionConditioner
from dreamfree.models.student import (
    FrozenStage1Student,
    Stage1StudentSource,
    extract_current_only_student_tokens,
)
from dreamfree.models.student_readout import (
    StudentLatentQueryReadout,
    normalize_readout_input_projection_mlp_ratio,
    normalize_readout_input_projection_type,
)
from dreamfree.models.wan22.action_dit import ActionDiT
from dreamfree.models.wan22.future_state import fixed_tokenwise_layer_norm
from dreamfree.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from dreamfree.training.stage2.config import Stage2Config
from dreamfree.training.stage2.constants import READOUT_TYPE_LATENT_QUERY


def _readout_dim(cfg: Stage2Config) -> int:
    value = cfg.readout_output_dim
    return int(cfg.action_hidden_dim if value is None else value)


def _readout_eval_num_queries(cfg: Stage2Config) -> int:
    value = cfg.readout_eval_num_queries
    return int(cfg.readout_num_queries if value is None else value)


def _stage2_readout_layer_idx(cfg: Stage2Config, source: Stage1StudentSource) -> int:
    value = cfg.student_readout_layer
    return int(value) if value is not None else int(source.student_layer_idx)


def _frozen_student_readout_layer_idx(student: FrozenStage1Student) -> int:
    value = getattr(student, "readout_layer_idx", None)
    return int(value) if value is not None else int(student.student_layer_idx)


class StudentReadoutActionConditioner(nn.Module):
    """Small wrapper around the Stage 2 conditioner with student-facing names."""

    def __init__(
        self,
        *,
        student_readout_dim: int,
        action_context_dim: int,
        proprio_dim: int,
        proprio_obs_dropout: float,
    ) -> None:
        super().__init__()
        self.inner = TeacherStateActionConditioner(
            teacher_state_input_dim=int(student_readout_dim),
            action_context_dim=int(action_context_dim),
            proprio_dim=int(proprio_dim),
            proprio_obs_dropout=float(proprio_obs_dropout),
        )

    def forward(
        self,
        student_readout_tokens: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        conditioned = self.inner(student_readout_tokens, context, context_mask, proprio)
        return {
            "student_context_emb": conditioned["teacher_context_emb"],
            "student_context_mask": conditioned["teacher_context_mask"],
            "proprio_query_emb": conditioned["proprio_query_emb"],
        }

    def teacher_state_state_dict(self) -> dict[str, torch.Tensor]:
        return self.inner.teacher_state_state_dict()

    def proprio_state_dict(self) -> dict[str, torch.Tensor]:
        return self.inner.proprio_state_dict()

    def load_teacher_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> None:
        self.inner.load_teacher_state_dict(state_dict, strict=strict)

    def load_proprio_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> None:
        self.inner.load_proprio_state_dict(state_dict, strict=strict)


class ActionTrainingModel(nn.Module):
    """Frozen visual tokens -> trainable readout/conditioning -> action flow loss."""

    def __init__(
        self,
        *,
        student: FrozenStage1Student,
        readout: nn.Module,
        action_model: ActionDiT,
        conditioner: StudentReadoutActionConditioner,
        cfg: Stage2Config,
    ) -> None:
        super().__init__()
        self.student = student
        self.core = student.core
        self.readout = readout
        self.action_model = action_model
        self.conditioner = conditioner
        self.cfg = cfg
        action_history.validate_config(cfg)
        self.action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(cfg.num_train_timesteps), shift=float(cfg.action_train_shift)
        )
        self.source_stage1_checkpoint = str(student.checkpoint_dir)
        self.source_stage1_metadata = student.metadata
        self.last_shape_metadata: dict[str, Any] = {}
        self.profile_sections_enabled = False
        self.core.requires_grad_(False)
        self.core.eval()
        self.readout.requires_grad_(True)
        self.action_model.requires_grad_(True)
        self.conditioner.requires_grad_(True)
        self.conditioner.inner.proprio_mask_token.requires_grad_(
            float(cfg.proprio_obs_dropout) > 0.0
        )
        self._assert_frozen_student()
        self.set_stage2_train_mode()

    def _profile_section(self, name: str):
        """Label a step phase in the profiler trace, or do nothing when not profiling.

        The `diag_*` timing sections use `torch.cuda.synchronize`, which distorts the very
        step time being attributed. These labels are free of that: the profiler already
        records async kernel times, so the annotation only groups them.
        """
        if not self.profile_sections_enabled:
            return contextlib.nullcontext()
        from torch.profiler import record_function

        return record_function(name)

    def _sample_active_readout_num_queries(self) -> int | None:
        return None

    def _assert_frozen_student(self) -> None:
        offenders = [name for (name, param) in self.core.named_parameters() if param.requires_grad]
        if offenders:
            preview = offenders[:10]
            suffix = "..." if len(offenders) > 10 else ""
            raise ValueError(
                f"Stage 1 student/core parameters must be frozen for DreamFree Stage 2, got {preview}{suffix}"
            )

    def set_stage2_train_mode(self) -> None:
        self.train()
        self.core.eval()
        self.core.dit.eval()
        if self.core.vae is not None:
            self.core.vae.eval()
        if self.core.text_encoder is not None:
            self.core.text_encoder.eval()
        self.readout.train()
        self.action_model.train()
        self.conditioner.train()

    def _build_action_query_prefix(
        self, *, proprio_query_emb: torch.Tensor, inputs: dict[str, Any], action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build the fixed ActionDiT query prefix and optional full query mask."""
        if self.cfg.action_history_horizon:
            prefix, mask, fractions = action_history.build_query_prefix(
                self.action_model.action_encoder,
                proprio_query_emb,
                inputs.get("previous_action"),
                inputs.get("previous_action_is_pad"),
                horizon=self.cfg.action_history_horizon,
                action_dim=self.cfg.action_dim,
                future_tokens=action.shape[1],
                dropout=self.cfg.action_history_dropout,
                training=self.training,
            )
            self._last_history_fractions = fractions
            return prefix, mask
        if "previous_action" in inputs or "previous_action_is_pad" in inputs:
            raise ValueError("History was supplied but action_history_horizon is zero.")
        return (proprio_query_emb, None)

    def training_loss(self, sample: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute action flow matching without updating the visual encoder.

        Args:
            sample: Collated live/cached batch accepted by prepare_action_batch:
                normalized action (B, T_action, A), aligned proprio
                (B, T_action, P), text, and visual inputs. action_is_pad=True
                excludes a target from the per-sample loss. Optional issued
                action history uses normalized targets and its own padding mask.

        Returns:
            Scalar loss and logging metrics. Gradients reach the readout,
            conditioner/proprio encoder, and action DiT, but not the student/VAE.
            Loss averages valid timesteps per sample before scheduler weighting.
        """
        if bool(self.cfg.use_cached_latents):
            if "video" in sample:
                raise ValueError(
                    "DreamFree Stage 2 cache-only training received live `video`. Regenerate the Stage 2 latent cache with full coverage."
                )
            if "input_latents" not in sample:
                raise ValueError(
                    "DreamFree Stage 2 cache-only training requires `input_latents` in every batch. Regenerate the Stage 2 latent cache with full coverage."
                )
        timing_recorder = _TimingRecorder(
            enabled=bool(getattr(self.cfg, "diagnostic_timing", False)), device=self.core.device
        )
        with (
            self._profile_section("stage2_input_prep"),
            timing_recorder.record("diag_input_prep_seconds"),
        ):
            inputs = prepare_action_batch(self.core, sample, timing_recorder=timing_recorder)
        z_clean = inputs["input_latents"]
        _validate_stage2_input_latents_contract(z_clean, self.cfg, stage_label="DreamFree Stage 2")
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        dit_context_mask = inputs.get("dit_context_mask", context_mask)
        action = inputs["action"]
        proprio = inputs["proprio"]
        action_is_pad = inputs["action_is_pad"]
        first_frame_latents = z_clean[:, :, 0:1].detach()  # (B, C, 1, h, w)
        with (
            self._profile_section("stage2_student_forward"),
            timing_recorder.record("diag_student_forward_seconds"),
        ):
            (student_tokens, student_meta) = extract_current_only_student_tokens(
                self.student,
                first_frame_latents=first_frame_latents,
                i2v_condition_latents=inputs.get("i2v_condition_latents"),
                context=context,
                context_mask=dit_context_mask,
            )
        if student_tokens.requires_grad:
            raise ValueError(
                "Frozen Stage 1 student tokens must be detached before DreamFree Stage 2 action loss."
            )
        student_tokens_cond = fixed_tokenwise_layer_norm(student_tokens.float()).to(
            dtype=self.core.torch_dtype
        )
        with (
            self._profile_section("stage2_readout"),
            timing_recorder.record("diag_readout_seconds"),
        ):
            active_readout_num_queries = self._sample_active_readout_num_queries()
            (readout_tokens, readout_mask) = self.readout(student_tokens_cond)
        if int(readout_tokens.shape[1]) <= 0:
            raise ValueError("Student readout produced zero tokens.")
        expected_num_queries = (
            int(active_readout_num_queries)
            if active_readout_num_queries is not None
            else int(self.cfg.readout_num_queries)
        )
        expected_readout_shape = (
            int(action.shape[0]),
            expected_num_queries,
            int(self.cfg.action_hidden_dim),
        )
        if tuple(readout_tokens.shape) != expected_readout_shape:
            raise ValueError(
                f"Student readout output shape mismatch: actual={tuple(readout_tokens.shape)} expected={expected_readout_shape}."
            )
        if tuple(readout_mask.shape) != (int(action.shape[0]), int(readout_tokens.shape[1])):
            raise ValueError(
                f"Student readout mask shape mismatch: tokens={tuple(readout_tokens.shape)} mask={tuple(readout_mask.shape)}."
            )
        with (
            self._profile_section("stage2_conditioner"),
            timing_recorder.record("diag_conditioner_seconds"),
        ):
            conditioned = self.conditioner(readout_tokens, context, context_mask, proprio)
        student_context_emb = conditioned["student_context_emb"]
        student_context_mask = conditioned["student_context_mask"]
        proprio_query_emb = conditioned["proprio_query_emb"]
        if student_context_emb.shape[1] <= 0:
            raise ValueError("Student-readout conditioning produced zero context tokens.")
        if tuple(student_context_mask.shape) != (action.shape[0], student_context_emb.shape[1]):
            raise ValueError(
                f"Student context mask shape mismatch: context={tuple(student_context_emb.shape)} mask={tuple(student_context_mask.shape)}."
            )
        (action_query_prefix_emb, action_query_token_mask) = self._build_action_query_prefix(
            proprio_query_emb=proprio_query_emb, inputs=inputs, action=action
        )
        batch_size = int(action.shape[0])
        noise_action = torch.randn_like(action)
        timestep_action = self.action_scheduler.sample_training_t(
            batch_size=batch_size, device=action.device, dtype=action.dtype
        )
        noisy_action = self.action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.action_scheduler.training_target(action, noise_action, timestep_action)
        with (
            self._profile_section("stage2_action_model"),
            timing_recorder.record("diag_action_model_seconds"),
        ):
            action_model_kwargs = {
                "action_tokens": noisy_action,
                "timestep": timestep_action,
                "context": context,
                "context_mask": dit_context_mask,
                "query_prefix_emb": action_query_prefix_emb,
                "query_prefix_timestep_sigma": float(self.cfg.proprio_state_timestep_sigma),
                "extra_context_emb": student_context_emb,
                "extra_context_mask": None,
                "num_train_timesteps": float(self.action_scheduler.num_train_timesteps),
            }
            if action_query_token_mask is not None:
                action_model_kwargs["query_token_mask"] = action_query_token_mask
            pred_sequence = self.action_model.forward_with_query_prefix_and_extra_context_emb(
                **action_model_kwargs
            )
        prefix_len = int(action_query_prefix_emb.shape[1])
        if prefix_len <= 0:
            raise ValueError("DreamFree Stage 2 query prefix must contribute at least one token.")
        # Prefix tokens condition the prediction; only future actions are targets.
        pred_action = pred_sequence[:, prefix_len:, :]  # (B, T_action, A)
        if tuple(pred_action.shape) != tuple(target_action.shape):
            raise ValueError(
                f"Predicted action velocity shape mismatch: pred={tuple(pred_action.shape)} target={tuple(target_action.shape)}."
            )
        with (
            self._profile_section("stage2_loss_compute"),
            timing_recorder.record("diag_loss_compute_seconds"),
        ):
            action_loss_token = F.mse_loss(
                pred_action.float(), target_action.float(), reduction="none"
            ).mean(dim=2)
            if action_is_pad is not None:
                valid = (~action_is_pad).to(
                    device=action_loss_token.device, dtype=action_loss_token.dtype
                )
                valid_sum = valid.sum(dim=1)
                if bool((valid_sum <= 0).any().item()):
                    raise ValueError("DreamFree Stage 2 received an all-padded action sample.")
                loss_action_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
                action_valid_fraction = float(valid.mean().item())
            else:
                loss_action_per_sample = action_loss_token.mean(dim=1)
                action_valid_fraction = 1.0
            action_weight = self.action_scheduler.training_weight(timestep_action).to(
                loss_action_per_sample.device, dtype=loss_action_per_sample.dtype
            )
            loss_action = (loss_action_per_sample * action_weight).mean()
            loss_total = loss_action
        self.last_shape_metadata = {
            "latent_shape": [int(v) for v in z_clean.shape[1:]],
            "student_grid_size": [int(v) for v in student_meta.get("grid_size", [])],
            "student_tokens_per_frame": int(self.student.expected_student_tokens_per_frame),
            "student_hidden_dim": int(student_tokens.shape[-1]),
            "readout_tokens": int(readout_tokens.shape[1]),
            "readout_hidden_dim": int(readout_tokens.shape[-1]),
            "student_context_length": int(student_context_emb.shape[1]),
            "action_horizon": int(action.shape[1]),
            "action_dim": int(action.shape[2]),
            "proprio_horizon": int(proprio.shape[1]),
            "proprio_dim": int(proprio.shape[2]),
            "query_prefix_tokens": int(prefix_len),
        }
        metrics = {
            "loss_action": float(loss_action.detach().item()),
            "loss_total": float(loss_total.detach().item()),
            "action_timestep_mean": float(timestep_action.detach().float().mean().item()),
            "action_training_weight_mean": float(action_weight.detach().float().mean().item()),
            "student_state_norm": float(
                student_tokens_cond.detach().float().pow(2).mean().sqrt().item()
            ),
            "readout_state_norm": float(
                readout_tokens.detach().float().pow(2).mean().sqrt().item()
            ),
            "student_tokens_per_frame": float(self.student.expected_student_tokens_per_frame),
            "readout_tokens": float(readout_tokens.shape[1]),
            "readout_active_queries": float(expected_num_queries),
            "action_valid_fraction": action_valid_fraction,
            "query_prefix_tokens": float(prefix_len),
        }
        if self.cfg.action_history_horizon:
            valid, effective, dropped = self._last_history_fractions.detach().cpu().tolist()
            metrics.update(
                action_history_valid_fraction=valid,
                action_history_effective_valid_fraction=effective,
                action_history_dropout_fraction=dropped,
                action_history_horizon=float(self.cfg.action_history_horizon),
                action_history_dropout=float(self.cfg.action_history_dropout),
            )
            self.last_shape_metadata.update(
                action_history_horizon=self.cfg.action_history_horizon,
                query_token_order=action_history.TOKEN_ORDER,
            )
        if timing_recorder.enabled:
            metrics.update(
                {key: float(value) for (key, value) in sorted(timing_recorder.values.items())}
            )
        return (loss_total, metrics)

    def forward(self, sample: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        return self.training_loss(sample)


def _compile_stage2_student_core_dit_blocks(core: Any, *, label: str) -> int:
    """Compile the Stage 1/Wan student core DiT used by Stage 2."""
    return compile_dit_blocks(core.dit, label=label)


def _compile_frozen_stage1_student_dit_blocks(student: FrozenStage1Student, *, label: str) -> int:
    """Compile a frozen Stage 1 student DiT used for token extraction."""
    return _compile_stage2_student_core_dit_blocks(student.core, label=label)


def _compile_action_dit_blocks(action_model: ActionDiT, *, label: str) -> int:
    """Compile the trainable ActionDiT blocks used by Stage 2 action denoising."""
    return compile_dit_blocks(action_model, label=label)


def _compile_dreamfree_stage2_dit_blocks(
    wrapper: ActionTrainingModel, *, label_prefix: str = "DreamFree Stage 2"
) -> dict[str, int]:
    """Compile both Stage 2 transformer block stacks."""
    return {
        "student_dit_blocks": _compile_frozen_stage1_student_dit_blocks(
            wrapper.student, label=f"{label_prefix} frozen Stage 1 student"
        ),
        "action_dit_blocks": _compile_action_dit_blocks(
            wrapper.action_model, label=f"{label_prefix} ActionDiT"
        ),
    }


def _build_action_model(
    cfg: Stage2Config, *, device: torch.device, dtype: torch.dtype
) -> ActionDiT:
    return ActionDiT.from_pretrained(
        action_dit_config={
            "action_dim": int(cfg.action_dim),
            "hidden_dim": int(cfg.action_hidden_dim),
            "ffn_dim": int(cfg.action_ffn_dim),
            "text_dim": int(cfg.action_text_dim),
            "freq_dim": int(cfg.action_freq_dim),
            "eps": float(cfg.action_eps),
            "num_heads": int(cfg.action_num_heads),
            "attn_head_dim": int(cfg.action_attn_head_dim),
            "num_layers": int(cfg.action_num_layers),
            "use_gradient_checkpointing": bool(cfg.use_gradient_checkpointing),
        },
        action_dit_pretrained_path=str(cfg.action_dit_pretrained_path),
        skip_dit_load_from_pretrain=bool(cfg.skip_dit_load_from_pretrain),
        device=str(device),
        torch_dtype=dtype,
        pretrained_load_broadcast=bool(cfg.pretrained_load_broadcast),
    )


def _build_student_readout(cfg: Stage2Config, *, student: FrozenStage1Student) -> nn.Module:
    common_kwargs = {
        "student_dim": int(student.expected_student_hidden_dim),
        "readout_dim": int(_readout_dim(cfg)),
        "output_dim": int(cfg.action_hidden_dim),
        "num_layers": int(cfg.readout_num_layers),
        "num_heads": int(cfg.readout_num_heads),
        "mlp_ratio": float(cfg.readout_mlp_ratio),
        "dropout": float(cfg.readout_dropout),
        "input_projection_type": normalize_readout_input_projection_type(
            cfg.readout_input_projection_type
        ),
        "input_projection_mlp_ratio": normalize_readout_input_projection_mlp_ratio(
            cfg.readout_input_projection_mlp_ratio
        ),
    }
    if str(cfg.readout_type) != READOUT_TYPE_LATENT_QUERY:
        raise ValueError(f"Unsupported readout_type={cfg.readout_type!r}.")
    return StudentLatentQueryReadout(**common_kwargs, num_queries=int(cfg.readout_num_queries))
