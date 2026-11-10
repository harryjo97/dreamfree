"""Shared frozen-encoder, readout, and action-denoising policy runtime.

Benchmark subclasses provide observation conversion and action postprocessing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

import dreamfree.models.policy_loading as dreamfree_deploy
from dreamfree.data.text_cache import DEFAULT_PROMPT
from dreamfree.evaluation.checkpoints import (
    condition_video_num_frames,
    readout_layer_from_metadata,
    readout_type_from_metadata,
    require_dict,
)
from dreamfree.evaluation.options import (
    ACTION_SAMPLER_EULER,
    ACTION_SAMPLER_UNIPC,
    normalize_action_sampler,
)
from dreamfree.models.student_readout import (
    StudentLatentQueryReadout,
)
from dreamfree.models.wan22.future_state import (
    _lora_enabled,
    fixed_tokenwise_layer_norm,
    temporary_video_attention_mask_mode,
)
from dreamfree.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from dreamfree.models.wan22.schedulers.scheduler_unipc import FlowUniPCMultistepScheduler

logger = logging.getLogger(__name__)


class PolicyRuntime(dreamfree_deploy.ActionChunkPolicy):
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
        noise_device: str = "cpu",
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
        self.noise_device = str(noise_device)
        self.tiled = bool(tiled)
        self.action_sampler = normalize_action_sampler(action_sampler)
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
        stage2_student = require_dict(
            stage2_metadata.get("student"), label="DreamFree Stage 2 metadata.student"
        )
        stage2_readout = require_dict(
            stage2_metadata.get("readout"), label="DreamFree Stage 2 metadata.readout"
        )
        stage2_action = require_dict(
            stage2_metadata.get("action"), label="DreamFree Stage 2 metadata.action"
        )
        scheduler_meta = require_dict(
            stage2_action.get("scheduler"), label="DreamFree Stage 2 metadata.action.scheduler"
        )
        if isinstance(stage1_metadata.get("config"), dict):
            stage1_config = require_dict(
                stage1_metadata.get("config"), label="Stage 1 metadata.config"
            )
        else:
            stage1_config = require_dict(
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
            self.condition_video_num_frames = condition_video_num_frames(stage2_metadata)
            stage1_shapes = require_dict(
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
        self.student_layer_idx = readout_layer_from_metadata(stage2_metadata)
        self.student_hook_point = str(stage2_student["hook_point"])
        self.student_i2v_condition_mode = str(stage2_student.get("i2v_condition_mode", "none"))
        self.expected_student_tokens_per_frame = int(stage2_student["tokens_per_frame"])
        self.expected_student_hidden_dim = int(stage2_student["hidden_dim"])
        self.readout_type = readout_type_from_metadata(stage2_metadata)
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
            "Initialized DreamFree Stage 2 policy | stage1=%s | stage2=%s | stats=%s | horizon=%d | replan=%d | action_sampler=%s",
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
            else torch.Generator(device=self.noise_device).manual_seed(self.seed)
        )
        return torch.randn(
            shape, generator=generator, device=self.noise_device, dtype=torch.float32
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
        """Convert a benchmark observation and instruction into an action chunk."""
        raise NotImplementedError("Benchmark policies must implement _infer_action_chunk.")

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        action_chunk = self._infer_action_chunk(observation=observation, instruction=instruction)
        self._enqueue_action_chunk(action_chunk)
