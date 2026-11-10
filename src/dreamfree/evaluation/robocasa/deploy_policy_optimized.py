"""RoboCasa-specific integration of fixed-shape BF16 inference optimizations."""

from __future__ import annotations

import copy
import random
import time

import numpy as np
import torch

import dreamfree.evaluation.robocasa.inference_options as inference_options
import dreamfree.evaluation.robocasa.policy as base
from dreamfree.data.text_cache import DEFAULT_PROMPT
from dreamfree.evaluation.robocasa.inference_options import FEATURES
from dreamfree.evaluation.robocasa.robocasa_utils import ENV_TO_DREAMFREE_IMAGE_KEYS
from dreamfree.models.action_cuda_graph import ActionEulerGraph
from dreamfree.models.compile import compile_dit_blocks
from dreamfree.models.encoder_cuda_graph import EncoderGraph
from dreamfree.models.first_frame_vae import encode_first_frame
from dreamfree.models.inference_optimization import (
    fuse_attention_projections,
    fused_sdpa_only,
    merge_active_lora_for_inference,
)

COMPILE_MODE = "max-autotune-no-cudagraphs"
COMPILE_MODE_5B = "default"
PROMOTED_FEATURES: frozenset[str] = frozenset()
PROMOTED_5B_FEATURES: frozenset[str] = frozenset()


def resolve_options(metadata: dict, args: dict) -> dict:
    return inference_options.resolve_options(
        metadata,
        args,
        promoted_features=PROMOTED_FEATURES,
        promoted_5b_features=PROMOTED_5B_FEATURES,
    )


def prepare_policy_for_evaluation(policy) -> None:
    """Run after distributed loading exits, on the same thread used for inference."""
    contract = getattr(policy, "optimization_contract", {})
    if contract.get("enabled") and (not contract.get("warmup_complete")):
        policy.initialize_optimized_inference()
        policy.warmup_optimized_inference()


class OptimizedRoboCasaInferenceMixin:
    """Own graph buffers and prepared conditioning in one persistent worker."""

    _active_action_conditioning = None
    _active_action_query_prefix_emb = None
    _active_action_query_token_mask = None

    def _action_graph_signature(self):
        return (
            self.action_sampler,
            self.num_inference_steps,
            self.action_infer_shift,
            self.action_scheduler.shift,
            self.action_scheduler.num_train_timesteps,
            self.action_num_train_timesteps,
            self.proprio_state_timestep_sigma,
            self.action_horizon,
            self.action_dim,
            getattr(self, "action_history_horizon", 0),
            self.model_dtype,
            self.device,
            id(self.action_model),
            len(self.action_model.blocks),
            getattr(self, "action_preserve_context_mask", False),
        )

    def _infer_dreamfree_action_tokens_euler(self, *, action_tokens, **kwargs):
        if not getattr(self, "action_cuda_graph_enabled", False):
            return super()._infer_dreamfree_action_tokens_euler(
                action_tokens=action_tokens, **kwargs
            )
        if self._action_graph_signature() != self._action_cuda_graph_signature:
            raise ValueError(
                "Action CUDA graph model or sampling schedule changed; rebuild the policy."
            )
        prepared = self._active_action_conditioning
        prefix = self._active_action_query_prefix_emb
        mask = self._active_action_query_token_mask
        if prepared is None or prefix is None:
            raise RuntimeError("Action CUDA graph requires prepared conditioning and query prefix.")
        padded = mask is not None
        if padded not in self._action_cuda_graphs:
            if not getattr(self, "_action_graph_capture_allowed", False):
                raise RuntimeError("Action CUDA graph was not captured during policy warmup.")
            if not self._action_cuda_graph_freqs.is_inference():
                raise RuntimeError(
                    "Action CUDA graph rotary cache must be created in inference mode to preserve compiled numerics."
                )

            def run(noise, conditioning, query_prefix, query_mask):
                old_freqs = self.action_model.freqs
                self.action_model.freqs = self._action_cuda_graph_freqs
                self._active_action_conditioning = conditioning
                self._active_action_query_prefix_emb = query_prefix
                self._active_action_query_token_mask = query_mask
                try:
                    return super(
                        OptimizedRoboCasaInferenceMixin, self
                    )._infer_dreamfree_action_tokens_euler(action_tokens=noise, **kwargs)
                finally:
                    self.action_model.freqs = old_freqs
                    self._active_action_conditioning = prepared
                    self._active_action_query_prefix_emb = prefix
                    self._active_action_query_token_mask = mask

            self._action_cuda_graphs[padded] = ActionEulerGraph(
                run, action_tokens, prepared, prefix, mask
            )
        return self._action_cuda_graphs[padded].replay(action_tokens, prepared, prefix, mask)

    def _infer_dreamfree_action_tokens(
        self,
        *,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        student_context_emb: torch.Tensor,
        student_context_mask: torch.Tensor,
        proprio_query_emb: torch.Tensor,
    ) -> torch.Tensor:
        if self._active_action_conditioning is not None:
            raise RuntimeError("ActionDiT prepared conditioning leaked across action chunks.")
        if (
            getattr(self, "action_cuda_graph_enabled", False)
            and self._action_graph_signature() != self._action_cuda_graph_signature
        ):
            raise ValueError(
                "Action CUDA graph model or sampling schedule changed; rebuild the policy."
            )
        with torch.profiler.record_function("robocasa/action_kv_prepare"):
            prepared = self.action_model.prepare_cross_attention_conditioning(
                context=context,
                context_mask=context_mask,
                extra_context_emb=student_context_emb,
                extra_context_mask=student_context_mask,
            )
            (query_prefix_emb, query_token_mask) = self._prepare_action_query(
                proprio_query_emb=proprio_query_emb, action_seq_len=int(self.action_horizon)
            )
        self._active_action_conditioning = prepared
        self._active_action_query_prefix_emb = query_prefix_emb
        self._active_action_query_token_mask = query_token_mask
        try:
            with torch.profiler.record_function("robocasa/action_denoise"):
                return super()._infer_dreamfree_action_tokens(
                    context=context,
                    context_mask=context_mask,
                    student_context_emb=student_context_emb,
                    student_context_mask=student_context_mask,
                    proprio_query_emb=proprio_query_emb,
                )
        finally:
            self._active_action_conditioning = None
            self._active_action_query_prefix_emb = None
            self._active_action_query_token_mask = None

    def _prepare_action_query(
        self, *, proprio_query_emb: torch.Tensor, action_seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        previous_action = getattr(self, "_active_previous_action", None)
        previous_action_is_pad = getattr(self, "_active_previous_action_is_pad", None)
        if previous_action is None and previous_action_is_pad is None:
            return (proprio_query_emb, None)
        if previous_action is None or previous_action_is_pad is None:
            raise RuntimeError(
                "Action-history tensors must either both be active or both be absent."
            )
        history_emb = self.action_model.action_encoder(previous_action)
        query_prefix_emb = torch.cat((proprio_query_emb, history_emb), dim=1)
        batch_size = int(proprio_query_emb.shape[0])
        query_token_mask = torch.cat(
            (
                torch.ones(
                    (batch_size, int(proprio_query_emb.shape[1])),
                    device=proprio_query_emb.device,
                    dtype=torch.bool,
                ),
                ~previous_action_is_pad,
                torch.ones(
                    (batch_size, int(action_seq_len)),
                    device=proprio_query_emb.device,
                    dtype=torch.bool,
                ),
            ),
            dim=1,
        )
        if bool(query_token_mask.all().item()):
            query_token_mask = None
        return (query_prefix_emb, query_token_mask)

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
        del context, context_mask, student_context_emb, student_context_mask
        prepared = self._active_action_conditioning
        if prepared is None:
            raise RuntimeError(
                "ActionDiT prepared conditioning is required during optimized denoising."
            )
        query_prefix_emb = self._active_action_query_prefix_emb
        query_token_mask = self._active_action_query_token_mask
        if query_prefix_emb is None:
            raise RuntimeError(
                "Prepared ActionDiT query prefix is required during optimized denoising."
            )
        pred_sequence = self.action_model.forward_with_query_prefix_and_prepared_context(
            action_tokens=action_tokens,
            timestep=timestep,
            prepared_conditioning=prepared,
            query_prefix_emb=query_prefix_emb,
            query_prefix_timestep_sigma=self.proprio_state_timestep_sigma,
            query_token_mask=query_token_mask,
            query_token_mask_validated=True,
            num_train_timesteps=float(self.action_num_train_timesteps),
            preserve_context_mask=getattr(self, "action_preserve_context_mask", False),
        )
        return (pred_sequence, int(query_prefix_emb.shape[1]))

    def predict_action_chunk(self, observation, instruction):
        if not self.optimization_contract["warmup_complete"] and (
            not getattr(self, "_warming_up", False)
        ):
            raise RuntimeError("Optimized RoboCasa policy must finish warmup before evaluation.")
        with torch.inference_mode(), fused_sdpa_only():
            return super().predict_action_chunk(observation, instruction)

    def _visual_signature(self):
        return (
            id(self.video_core.vae.model),
            id(self.video_core.dit),
            self.device,
            self.model_dtype,
            self.video_core.video_model_family,
            self.tiled,
            self.condition_video_num_frames,
            self.student_layer_idx,
            self.student_hook_point,
            self.expected_student_tokens_per_frame,
            self.expected_student_hidden_dim,
            self.text_context_len,
            self.student_lora_adapter_name,
            tuple(self.robocasa_image_size),
            len(self.video_core.dit.blocks),
        )

    def _encode_observation_latents(self, image_tensor):
        if not self.vae_first_frame_only_enabled:
            return super()._encode_observation_latents(image_tensor)
        if self._visual_signature() != self._visual_optimization_signature:
            raise ValueError("First-frame VAE configuration changed; rebuild the policy.")
        (height, width) = self.robocasa_image_size
        if (
            tuple(image_tensor.shape) != (1, 3, height, width)
            or image_tensor.device != self._visual_device
            or image_tensor.dtype != self.model_dtype
        ):
            raise ValueError(
                "First-frame VAE image shape/device/dtype differs from the policy contract."
            )
        video = image_tensor.new_zeros(1, 3, self.condition_video_num_frames, height, width)
        video[:, :, :1] = image_tensor.unsqueeze(2)
        latent = encode_first_frame(self.video_core.vae.model, video, self.video_core.vae.scale)[
            :, :, :1
        ].detach()
        return (latent, latent)

    def _extract_student_tokens(
        self, *, first_frame_latents, i2v_condition_latents=None, context, context_mask
    ):
        if not self.encoder_cuda_graph_enabled:
            return super()._extract_student_tokens(
                first_frame_latents=first_frame_latents,
                i2v_condition_latents=i2v_condition_latents,
                context=context,
                context_mask=context_mask,
            )
        if self._visual_signature() != self._visual_optimization_signature:
            raise ValueError("Stage 1 CUDA graph configuration changed; rebuild the policy.")
        if context_mask is None:
            raise ValueError("Stage 1 CUDA graph requires a prompt mask.")
        if self.video_core.video_model_family == "wan22_ti2v_5b":
            if i2v_condition_latents is not None:
                raise ValueError("5B Stage 1 CUDA graph requires absent I2V conditioning.")
            condition = None
        else:
            if i2v_condition_latents is None:
                raise ValueError("14B Stage 1 CUDA graph requires I2V conditioning.")
            condition = i2v_condition_latents[:, :, :1]
        if self._encoder_cuda_graph is None:
            if not getattr(self, "_encoder_graph_capture_allowed", False):
                raise RuntimeError("Stage 1 CUDA graph was not captured during policy warmup.")
            if any(not value.is_inference() for value in self._encoder_graph_freqs):
                raise RuntimeError("Stage 1 rotary tensors must be created in inference mode.")

            def run(latents, image_condition, prompt, mask):
                old_freqs = self.video_core.dit.freqs
                self.video_core.dit.freqs = self._encoder_graph_freqs
                try:
                    return super(OptimizedRoboCasaInferenceMixin, self)._extract_student_tokens(
                        first_frame_latents=latents,
                        i2v_condition_latents=image_condition,
                        context=prompt,
                        context_mask=mask,
                    )
                finally:
                    self.video_core.dit.freqs = old_freqs

            self._encoder_cuda_graph = EncoderGraph(
                run, first_frame_latents, condition, context, context_mask
            )
        return self._encoder_cuda_graph.replay(
            first_frame_latents, condition, context, context_mask
        )

    def initialize_optimized_inference(self):
        if getattr(self, "_optimization_initialized", False):
            raise RuntimeError("RoboCasa inference optimization was already initialized.")
        if self.compile_dit:
            raise RuntimeError("Base policy compiled before RoboCasa optimizations were installed.")
        from dreamfree.models.wan22.wan_video_vae import VideoVAE38_, VideoVAE_

        family = self.video_core.video_model_family
        if family not in ("wan22_i2v_a14b", "wan22_ti2v_5b"):
            raise ValueError(f"Unsupported optimized RoboCasa video family: {family}.")
        is_5b = family == "wan22_ti2v_5b"
        (frames, vae_type) = (1, VideoVAE38_) if is_5b else (9, VideoVAE_)
        if self.action_sampler != "euler" or self.condition_video_num_frames != frames:
            raise ValueError(
                f"Optimized RoboCasa {family} requires Euler sampling and {frames}-frame conditioning."
            )
        if type(self.video_core.vae.model) is not vae_type:
            raise ValueError(
                f"Optimized RoboCasa {family} requires the {vae_type.__name__} encoder."
            )
        if is_5b and self.optimization_contract["features"]["vae_first_frame_only"]["enabled"]:
            raise ValueError("5B already encodes one frame; the 14B VAE shortcut must be disabled.")
        merged = merge_active_lora_for_inference(
            self.video_core.dit, adapter_name=self.student_lora_adapter_name
        )
        self.student_lora_adapter_name = self.video_core.student_lora_adapter_name = None
        compile_mode = COMPILE_MODE_5B if is_5b else COMPILE_MODE
        (projection_counts, compile_counts) = ({}, {})
        for name, model in (
            ("video_core_dit_blocks", self.video_core.dit),
            ("action_model_dit_blocks", self.action_model),
        ):
            if is_5b:
                projection_counts[name] = dict(self_qkv=0, cross_kv=0)
            else:
                projection_counts[name] = fuse_attention_projections(model)
                expected = {"self_qkv": len(model.blocks), "cross_kv": len(model.blocks)}
                if projection_counts[name] != expected:
                    raise RuntimeError(
                        f"RoboCasa projection fusion count mismatch for {name}: {projection_counts[name]} != {expected}"
                    )
            compile_counts[name] = compile_dit_blocks(
                model, label=f"optimized RoboCasa {name}", mode=compile_mode
            )
        self.action_preserve_context_mask = is_5b
        if is_5b:
            for block in self.action_model.blocks:
                block.cross_attn.project_context = torch.compile(
                    block.cross_attn.project_context, dynamic=False, mode=compile_mode
                )
        self.compile_dit = True
        self.compile_dit_block_counts = compile_counts
        for name in FEATURES:
            setattr(
                self, name + "_enabled", self.optimization_contract["features"][name]["enabled"]
            )
        self._action_cuda_graphs = {}
        self._encoder_cuda_graph = None
        self._action_cuda_graph_signature = self._action_graph_signature()
        self._visual_optimization_signature = self._visual_signature()
        self._visual_device = torch.device(
            "cuda",
            self.device.index if self.device.index is not None else torch.cuda.current_device(),
        )
        with torch.inference_mode():
            self._action_cuda_graph_freqs = self.action_model.freqs.to(self.device)
            self._encoder_graph_freqs = tuple(
                value.to(self.device) for value in self.video_core.dit.freqs
            )
        self.optimization_contract.update(
            precision="bfloat16",
            compile_dit=True,
            compile_mode=compile_mode,
            compile_block_counts=compile_counts,
            projection_counts=projection_counts,
            merged_lora_count=merged,
            attention_policy="sdpa_fused_no_math",
            compile_auxiliary="none",
            action_cross_kv_cache=True,
            action_cuda_graph_capture_count=0,
            encoder_cuda_graph_capture_count=0,
        )
        if is_5b:
            self.optimization_contract.update(
                projection_policy="unfused_reference",
                action_kv_compile=True,
                action_preserve_context_mask=True,
            )
        self._optimization_initialized = True

    def warmup_optimized_inference(self):
        if not getattr(self, "_optimization_initialized", False):
            raise RuntimeError("Initialize RoboCasa optimizations before warmup.")
        active_names = (
            "_active_action_conditioning",
            "_active_action_query_prefix_emb",
            "_active_action_query_token_mask",
            "_active_previous_action",
            "_active_previous_action_is_pad",
        )
        if any(getattr(self, name, None) is not None for name in active_names):
            raise RuntimeError("Cannot warm up while action conditioning is active.")
        if self.optimization_contract["warmup_complete"]:
            return self.optimization_contract["warmup_seconds"]
        observation = {
            key: np.zeros((224, 224, 3), dtype=np.uint8) for (key, _) in ENV_TO_DREAMFREE_IMAGE_KEYS
        }
        observation.update(
            robot0_gripper_qpos=np.zeros(2, dtype=np.float32),
            robot0_eef_pos=np.zeros(3, dtype=np.float32),
            robot0_eef_quat=np.array([0, 0, 0, 1], dtype=np.float32),
        )
        instruction = "RoboCasa inference warmup"
        old_cache = self._prompt_context_cache
        old_timing = dict(self._timing_rollout)
        (old_python_rng, old_numpy_rng) = (random.getstate(), np.random.get_state())
        old_history = copy.deepcopy(getattr(self, "_issued_action_history", None))
        horizon = int(getattr(self, "action_history_horizon", 0))
        counts = (horizon, horizon // 2, 0) if horizon else (0,)
        started = time.perf_counter()
        self._warming_up = self._action_graph_capture_allowed = (
            self._encoder_graph_capture_allowed
        ) = True
        try:
            with torch.random.fork_rng(devices=[self._visual_device.index]), torch.inference_mode():
                prompt = DEFAULT_PROMPT.format(task=instruction)
                self._prompt_context_cache = {
                    prompt: (
                        torch.zeros(
                            1,
                            self.text_context_len,
                            self.text_context_dim,
                            device=self.device,
                            dtype=self.model_dtype,
                        ),
                        torch.ones(1, self.text_context_len, device=self.device, dtype=torch.bool),
                    )
                }
                for _ in range(2):
                    for pad_count in counts:
                        if horizon:
                            self._issued_action_history.clear()
                            for _ in range(horizon - pad_count):
                                self.record_issued_action(
                                    np.zeros(self.action_dim, dtype=np.float32)
                                )
                        actions = self.predict_action_chunk(observation, instruction)
                        if (
                            actions.shape != (self.action_horizon, self.action_dim)
                            or not np.isfinite(actions).all()
                        ):
                            raise RuntimeError(
                                "Optimized RoboCasa warmup returned invalid actions."
                            )
                torch.cuda.synchronize(self.device)
        finally:
            self._warming_up = self._action_graph_capture_allowed = (
                self._encoder_graph_capture_allowed
            ) = False
            self._prompt_context_cache = old_cache
            self._timing_rollout.clear()
            self._timing_rollout.update(old_timing)
            random.setstate(old_python_rng)
            np.random.set_state(old_numpy_rng)
            if old_history is not None:
                self._issued_action_history.clear()
                self._issued_action_history.extend(old_history)
            for name in active_names:
                setattr(self, name, None)
            self.video_core.vae.model.clear_cache()
        expected = {False, True} if horizon else {False}
        if self.action_cuda_graph_enabled and set(self._action_cuda_graphs) != expected:
            raise RuntimeError(
                f"RoboCasa action graph warmup did not capture required variants: {expected}."
            )
        if self.encoder_cuda_graph_enabled and self._encoder_cuda_graph is None:
            raise RuntimeError("RoboCasa Stage 1 graph warmup did not capture feature extraction.")
        elapsed = time.perf_counter() - started
        self.optimization_contract.update(
            warmup_complete=True,
            warmup_seconds=elapsed,
            warmup_scope="full_action_chunk",
            warmup_history_padding=list(counts),
            action_cuda_graph_capture_count=len(self._action_cuda_graphs),
            encoder_cuda_graph_capture_count=int(self._encoder_cuda_graph is not None),
            peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
            peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(self.device),
        )
        return elapsed


class OptimizedNativeRoboCasaPolicy(OptimizedRoboCasaInferenceMixin, base.RoboCasaPolicy):
    pass
