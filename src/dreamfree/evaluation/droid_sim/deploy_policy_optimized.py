"""Optimized BF16 inference builders for DROID deployment and simulation."""

from __future__ import annotations

import time
from contextlib import ExitStack
from functools import partial
from typing import Any, Dict

import numpy as np
import torch

import dreamfree.evaluation.droid_sim.deploy_policy as droid_base
from dreamfree.evaluation.action_protocol import (
    ISSUED_ACTION_HISTORY_IS_PAD_KEY,
    ISSUED_ACTION_HISTORY_KEY,
)
from dreamfree.models.action_cuda_graph import ActionEulerGraph
from dreamfree.models.compile import compile_dit_blocks
from dreamfree.models.encoder_cuda_graph import EncoderGraph
from dreamfree.models.first_frame_vae import encode_first_frame
from dreamfree.models.inference_optimization import (
    fuse_attention_projections,
    fused_sdpa_only,
    merge_active_lora_for_inference,
)

OPTIMIZED_COMPILE_MODE = "max-autotune-no-cudagraphs"
OPTIMIZED_NATIVE_POLICY_RUNTIME = "dreamfree_stage2_student_native_optimized"
WAN22_A14B_GRAPH_POLICY_RUNTIMES = (OPTIMIZED_NATIVE_POLICY_RUNTIME,)
AUXILIARY_COMPILE_CHOICES = ("none",)
ACTION_CUDA_GRAPH_PROMOTED = True
VAE_FIRST_FRAME_PROMOTED_AUXILIARY_MODES = frozenset({"none"})
STAGE1_GRAPH_PROMOTED_AUXILIARY_MODES = frozenset()


def resolve_visual_optimization(
    value: str, policy: Any, runtime: str, auxiliary: str, *, feature: str
):
    if feature not in ("vae_first_frame_only", "encoder_cuda_graph"):
        raise ValueError(f"Unknown visual optimization: {feature!r}.")
    if value not in ("auto", "true", "false"):
        raise ValueError(f"{feature} must be auto, true, or false, got {value!r}.")
    if feature == "vae_first_frame_only" and auxiliary != "none":
        if value == "true":
            raise ValueError(
                "vae_first_frame_only requires compile_auxiliary=none: the compiled prefix failed numerical validation."
            )
        return (
            False,
            "explicitly_disabled"
            if value == "false"
            else "compiled_prefix_failed_numerical_validation",
        )
    supported = (
        runtime in WAN22_A14B_GRAPH_POLICY_RUNTIMES
        and getattr(policy.video_core, "video_model_family", None) == "wan22_i2v_a14b"
        and (policy.device.type == "cuda")
        and (policy.model_dtype == torch.bfloat16)
        and (not getattr(policy, "tiled", True))
        and (getattr(policy, "condition_video_num_frames", None) == 9)
        and (getattr(policy, "student_layer_idx", None) == 31)
        and (getattr(policy, "student_hook_point", None) == "post_block")
        and (getattr(policy, "expected_student_tokens_per_frame", None) == 880)
        and (getattr(policy, "text_context_len", None) == 128)
        and (getattr(policy, "action_horizon", None) == 32)
        and (policy.action_sampler == "euler")
        and (policy.num_inference_steps in (16, 24))
        and (auxiliary in ("none", "encoders"))
    )
    if value == "true" and (not supported):
        raise ValueError(
            f"{feature} requires optimized 14B standard, CUDA BF16, untiled 352x640 input, nine-frame conditioning, layer 31, context 128, horizon 32, 16- or 24-step Euler, and compile_auxiliary=none."
        )
    if value == "false":
        return (False, "explicitly_disabled")
    if not supported:
        return (False, "configuration_outside_validated_scope")
    if value == "auto" and runtime == OPTIMIZED_NATIVE_POLICY_RUNTIME:
        return (False, "not_promoted_by_benchmark")
    promoted = (
        VAE_FIRST_FRAME_PROMOTED_AUXILIARY_MODES
        if feature == "vae_first_frame_only"
        else STAGE1_GRAPH_PROMOTED_AUXILIARY_MODES
    )
    if value == "auto" and auxiliary not in promoted:
        return (False, "not_promoted_by_benchmark")
    return (True, "explicitly_enabled" if value == "true" else "validated_default")


def resolve_action_cuda_graph(value: str, policy: Any, runtime: str) -> tuple[bool, str]:
    if value not in ("auto", "true", "false"):
        raise ValueError(f"action_cuda_graph must be auto, true, or false, got {value!r}.")
    supported = (
        runtime in WAN22_A14B_GRAPH_POLICY_RUNTIMES
        and getattr(policy.video_core, "video_model_family", None) == "wan22_i2v_a14b"
        and (policy.device.type == "cuda")
        and (policy.model_dtype == torch.bfloat16)
        and (policy.action_sampler == "euler")
        and (policy.num_inference_steps in (16, 24))
    )
    if value == "true" and (not supported):
        raise ValueError(
            "Action CUDA graph requires optimized 14B standard, CUDA BF16, and 16- or 24-step Euler."
        )
    if value == "false":
        return (False, "explicitly_disabled")
    if not supported:
        return (False, "configuration_outside_validated_scope")
    if value == "auto" and runtime == OPTIMIZED_NATIVE_POLICY_RUNTIME:
        return (False, "not_promoted_by_benchmark")
    if value == "auto" and (not ACTION_CUDA_GRAPH_PROMOTED):
        return (False, "awaiting_benchmark_promotion")
    return (True, "explicitly_enabled" if value == "true" else "validated_default")


def resolve_auxiliary_compile(value: str | None, runtime: str) -> str:
    if value not in (None, "none"):
        raise ValueError("Standard inference requires compile_auxiliary=none.")
    return "none"


class OptimizedDroidInferenceMixin:
    """Use fused SDPA and one ActionDiT cross-attention K/V cache per chunk."""

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
            getattr(self, "action_history_horizon", 0),
            self.action_dim,
            self.model_dtype,
            self.device,
            id(self.action_model),
            len(self.action_model.blocks),
        )

    def _run_action_euler(self, **kwargs):
        return super()._infer_dreamfree_action_tokens_euler(**kwargs)

    def _prepare_action_conditioning(self, **kwargs):
        return self.action_model.prepare_cross_attention_conditioning(**kwargs)

    def _action_inference_kwargs(self):
        return {}

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
                    return self._run_action_euler(action_tokens=noise, **kwargs)
                finally:
                    self.action_model.freqs = old_freqs
                    self._active_action_conditioning = prepared
                    self._active_action_query_prefix_emb = prefix
                    self._active_action_query_token_mask = mask

            options = (
                {"borrow_conditioning": True}
                if getattr(self, "_borrow_action_conditioning", False)
                else {}
            )
            self._action_cuda_graphs[padded] = ActionEulerGraph(
                run, action_tokens, prepared, prefix, mask, **options
            )
        return self._action_cuda_graphs[padded].replay(action_tokens, prepared, prefix, mask)

    def _infer_relative_action_chunk(
        self, observation: Dict[str, Any], instruction: str
    ) -> np.ndarray:
        with ExitStack() as stack:
            stack.enter_context(torch.inference_mode())
            stack.enter_context(fused_sdpa_only())
            with torch.profiler.record_function("droid/optimized_total_inference"):
                return super()._infer_relative_action_chunk(observation, instruction)

    def _visual_signature(self):
        return (
            id(self.video_core.vae.model),
            id(self.video_core.dit),
            self.device,
            self.model_dtype,
            self.tiled,
            self.condition_video_num_frames,
            self.student_layer_idx,
            self.student_hook_point,
            self.expected_student_tokens_per_frame,
            self.expected_student_hidden_dim,
            self.text_context_len,
            self.student_lora_adapter_name,
            len(self.video_core.dit.blocks),
        )

    def _encode_observation_latents(self, image_tensor):
        if not getattr(self, "vae_first_frame_only_enabled", False):
            return super()._encode_observation_latents(image_tensor)
        if self._visual_signature() != self._visual_optimization_signature:
            raise ValueError("First-frame VAE configuration changed; rebuild the policy.")
        if (
            tuple(image_tensor.shape) != (1, 3, 352, 640)
            or image_tensor.device != self._visual_device
            or image_tensor.dtype != self.model_dtype
        ):
            raise ValueError("First-frame VAE requires [1,3,352,640] on the policy device in BF16.")
        video = image_tensor.new_zeros(1, 3, self.condition_video_num_frames, 352, 640)
        video[:, :, :1] = image_tensor.unsqueeze(2)
        latent = self._encode_first_frame_vae(video, self.video_core.vae.scale)[:, :, :1].detach()
        return (latent, latent)

    def _extract_student_tokens(
        self, *, first_frame_latents, i2v_condition_latents=None, context, context_mask
    ):
        with torch.profiler.record_function("droid/stage1_dit"):
            if not getattr(self, "encoder_cuda_graph_enabled", False):
                return super()._extract_student_tokens(
                    first_frame_latents=first_frame_latents,
                    i2v_condition_latents=i2v_condition_latents,
                    context=context,
                    context_mask=context_mask,
                )
            if self._visual_signature() != self._visual_optimization_signature:
                raise ValueError("Stage 1 CUDA graph configuration changed; rebuild the policy.")
            if i2v_condition_latents is None or context_mask is None:
                raise ValueError("Stage 1 CUDA graph requires I2V conditioning and a prompt mask.")
            condition = i2v_condition_latents[:, :, :1]
            if self._encoder_cuda_graph is None:
                if not getattr(self, "_encoder_graph_capture_allowed", False):
                    raise RuntimeError("Stage 1 CUDA graph was not captured during policy warmup.")
                if any((not value.is_inference() for value in self._encoder_graph_freqs)):
                    raise RuntimeError("Stage 1 rotary tensors must be created in inference mode.")

                def run(latents, image_condition, prompt, mask):
                    dit = self.video_core.dit
                    original_freqs = dit.freqs
                    dit.freqs = self._encoder_graph_freqs
                    try:
                        return super(OptimizedDroidInferenceMixin, self)._extract_student_tokens(
                            first_frame_latents=latents,
                            i2v_condition_latents=image_condition,
                            context=prompt,
                            context_mask=mask,
                        )
                    finally:
                        dit.freqs = original_freqs

                self._encoder_cuda_graph = EncoderGraph(
                    run,
                    first_frame_latents,
                    condition,
                    context,
                    context_mask,
                    borrow_latents=getattr(self, "_borrow_visual_buffers", False),
                    borrow_output=getattr(self, "_borrow_visual_buffers", False),
                )
            return self._encoder_cuda_graph.replay(
                first_frame_latents, condition, context, context_mask
            )

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
        with torch.profiler.record_function("droid/action_kv_prepare"):
            prepared = self._prepare_action_conditioning(
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
            with torch.profiler.record_function("droid/action_denoise"):
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
        if not getattr(self, "action_history_horizon", 0):
            return (proprio_query_emb, None)
        from dreamfree.models.action_history import build_query_prefix

        prefix, mask, _ = build_query_prefix(
            self.action_model.action_encoder,
            proprio_query_emb,
            self._active_previous_action,
            self._active_previous_action_is_pad,
            horizon=self.action_history_horizon,
            action_dim=8,
            future_tokens=action_seq_len,
        )
        if bool(mask.all().item()):
            mask = None
        return prefix, mask

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
            **self._action_inference_kwargs(),
        )
        return (pred_sequence, int(query_prefix_emb.shape[1]))

    def warmup_optimized_inference(self, *, cached_instruction: str | None = None) -> float:
        """Warm up without consuming action RNG; optionally use a cached benchmark instruction."""
        if cached_instruction is not None:
            if not isinstance(cached_instruction, str) or not cached_instruction.strip():
                raise ValueError("Cached warmup requires a non-empty instruction.")
            if not self.text_embedding_cache_dirs:
                raise ValueError(
                    "Cached warmup requires text_embedding_cache_dirs; online encoding is disabled."
                )
            if not getattr(self, "_full_policy_warmup", False):
                raise ValueError("Cached warmup requires an optimized 14B standard runtime.")
        if getattr(self, "_full_policy_warmup", False):
            return self._warmup_full_inference(cached_instruction=cached_instruction)
        if getattr(self, "action_history_horizon", 0):
            self._active_previous_action = torch.zeros(
                (1, self.action_history_horizon, 8), device=self.device, dtype=self.model_dtype
            )
            self._active_previous_action_is_pad = torch.ones(
                (1, self.action_history_horizon), device=self.device, dtype=torch.bool
            )
        image = np.zeros((180, 320, 3), dtype=np.uint8)
        observation = {
            "observation/exterior_image_1_left": image,
            "observation/exterior_image_2_left": image,
            "observation/wrist_image_left": image,
            "observation/joint_position": np.zeros(7, dtype=np.float32),
            "observation/gripper_position": np.zeros(1, dtype=np.float32),
        }
        image_tensor = droid_base.build_droid_image_tensor(
            observation, device=self.device, dtype=self.model_dtype
        )
        proprio = droid_base.normalize_droid_state_with_processor(
            self.processor,
            observation["observation/joint_position"],
            observation["observation/gripper_position"],
        ).to(device=self.device, dtype=self.model_dtype)
        start = time.perf_counter()
        with torch.inference_mode(), fused_sdpa_only():
            cached_dirs = self.text_embedding_cache_dirs
            self.text_embedding_cache_dirs = []
            try:
                (context, context_mask, student_context, student_mask, proprio_query) = (
                    self._build_student_readout_conditioning(
                        image_tensor=image_tensor,
                        instruction="optimized inference warmup",
                        proprio=proprio,
                    )
                )
            finally:
                self.text_embedding_cache_dirs = cached_dirs
            self._active_action_conditioning = (
                self.action_model.prepare_cross_attention_conditioning(
                    context=context,
                    context_mask=context_mask,
                    extra_context_emb=student_context,
                    extra_context_mask=student_mask,
                )
            )
            try:
                (query_prefix, query_mask) = self._prepare_action_query(
                    proprio_query_emb=proprio_query, action_seq_len=int(self.action_horizon)
                )
                self._active_action_query_prefix_emb = query_prefix
                self._active_action_query_token_mask = query_mask
                self._forward_action_model_with_conditioning(
                    action_tokens=torch.zeros(
                        (1, int(self.action_horizon), int(self.action_dim)),
                        device=self.device,
                        dtype=self.model_dtype,
                    ),
                    timestep=torch.zeros(1, device=self.device, dtype=self.model_dtype),
                    context=context,
                    context_mask=context_mask,
                    student_context_emb=student_context,
                    student_context_mask=student_mask,
                    proprio_query_emb=proprio_query,
                )
                torch.cuda.synchronize(self.device)
            finally:
                self._active_action_conditioning = None
                self._active_action_query_prefix_emb = None
                self._active_action_query_token_mask = None
                self._active_previous_action = self._active_previous_action_is_pad = None
        elapsed = time.perf_counter() - start
        self._active_previous_action = self._active_previous_action_is_pad = None
        self.optimization_contract["warmup_complete"] = True
        self.optimization_contract["warmup_seconds"] = float(elapsed)
        return elapsed

    def _warmup_full_inference(self, *, cached_instruction: str | None = None) -> float:
        """Exercise the real inference path while preserving prompt and episode state."""
        active_names = (
            "_active_action_conditioning",
            "_active_action_query_prefix_emb",
            "_active_action_query_token_mask",
            "_active_previous_action",
            "_active_previous_action_is_pad",
        )
        if any((getattr(self, name, None) is not None for name in active_names)):
            raise RuntimeError("Cannot warm up while action-chunk conditioning is active.")
        self.optimization_contract["warmup_complete"] = False
        image = np.zeros((180, 320, 3), dtype=np.uint8)
        horizon = int(getattr(self, "action_history_horizon", 0))
        padding_counts = (horizon, horizon // 2, 0) if horizon else (0,)
        observation = {
            "observation/exterior_image_1_left": image,
            "observation/exterior_image_2_left": image,
            "observation/wrist_image_left": image,
            "observation/joint_position": np.zeros(7, dtype=np.float32),
            "observation/gripper_position": np.zeros(1, dtype=np.float32),
        }
        old_cache = self._prompt_context_cache
        if horizon:
            observation[ISSUED_ACTION_HISTORY_KEY] = np.zeros((horizon, 8), dtype=np.float32)
        old_dirs = self.text_embedding_cache_dirs
        old_timing = dict(self._timing_rollout)
        rng_names = (
            "_action_noise_generator",
            "action_noise_episode_seed",
            "action_noise_derived_seed",
        )
        old_rng = {name: getattr(self, name) for name in rng_names}
        instruction = (
            cached_instruction if cached_instruction is not None else "optimized inference warmup"
        )
        device_index = (
            self.device.index if self.device.index is not None else torch.cuda.current_device()
        )
        started = time.perf_counter()
        self._action_graph_capture_allowed = True
        self._encoder_graph_capture_allowed = True
        try:
            self._prompt_context_cache = {}
            self.text_embedding_cache_dirs = old_dirs if cached_instruction is not None else []
            with torch.random.fork_rng(devices=[device_index]), torch.inference_mode():
                if self.action_noise_seed is not None:
                    self._action_noise_generator = torch.Generator(
                        device=self.noise_device
                    ).manual_seed(0)
                    self.action_noise_episode_seed = 0
                    self.action_noise_derived_seed = 0
                for _ in range(2):
                    for pad_count in padding_counts:
                        if horizon:
                            observation[ISSUED_ACTION_HISTORY_IS_PAD_KEY] = (
                                np.arange(horizon) < pad_count
                            )
                        actions = self.predict_action_chunk(
                            observation, instruction, binarize_gripper=False
                        )
                        if (
                            actions.shape != (int(self.action_horizon), 8)
                            or not np.isfinite(actions).all()
                        ):
                            raise RuntimeError("Full policy warmup produced invalid actions.")
                torch.cuda.synchronize(self.device)
        finally:
            self._action_graph_capture_allowed = False
            self._encoder_graph_capture_allowed = False
            self._prompt_context_cache = old_cache
            self.text_embedding_cache_dirs = old_dirs
            self._timing_rollout.clear()
            self._timing_rollout.update(old_timing)
            for name, value in old_rng.items():
                setattr(self, name, value)
            for name in active_names:
                setattr(self, name, None)
            self.video_core.vae.model.clear_cache()
        if getattr(self, "action_cuda_graph_enabled", False):
            expected_captures = {False, True} if horizon else {False}
            if set(self._action_cuda_graphs) != expected_captures:
                raise RuntimeError(
                    f"Action CUDA graph warmup requires capture variants {expected_captures}, got {set(self._action_cuda_graphs)}."
                )
            self.optimization_contract.update(
                action_cuda_graph_capture_complete=True,
                action_cuda_graph_capture_count=len(expected_captures),
            )
        if getattr(self, "encoder_cuda_graph_enabled", False):
            if self._encoder_cuda_graph is None:
                raise RuntimeError("Stage 1 CUDA graph warmup did not capture feature extraction.")
            self.optimization_contract.update(
                encoder_cuda_graph_capture_complete=True, encoder_cuda_graph_capture_count=1
            )
        elapsed = time.perf_counter() - started
        self.optimization_contract.update(
            warmup_complete=True,
            warmup_seconds=float(elapsed),
            warmup_scope="full_action_chunk",
            warmup_history_padding=list(padding_counts) if horizon else [],
            warmup_repetitions=2,
        )
        return elapsed


class OptimizedNativeDroidPolicy(OptimizedDroidInferenceMixin, droid_base.DroidPolicy):
    pass


def _optimize_policy(
    policy: Any,
    *,
    runtime: str,
    fuse_qkv: bool,
    cudnn_benchmark: bool,
    compile_auxiliary: str = "none",
    action_cuda_graph: str = "auto",
    vae_first_frame_only: str = "auto",
    encoder_cuda_graph: str = "auto",
) -> Any:
    if policy.model_dtype != torch.bfloat16:
        raise ValueError(
            f"Optimized DROID inference requires BF16 model weights, got {policy.model_dtype}."
        )
    if policy.device.type != "cuda":
        raise ValueError(f"Optimized DROID inference requires CUDA, got device={policy.device}.")
    if bool(getattr(policy, "compile_dit", False)):
        raise RuntimeError("Base policy compiled before inference optimizations were installed.")
    (graph_enabled, graph_reason) = resolve_action_cuda_graph(action_cuda_graph, policy, runtime)
    (vae_enabled, vae_reason) = resolve_visual_optimization(
        vae_first_frame_only, policy, runtime, compile_auxiliary, feature="vae_first_frame_only"
    )
    (stage1_enabled, stage1_reason) = resolve_visual_optimization(
        encoder_cuda_graph, policy, runtime, compile_auxiliary, feature="encoder_cuda_graph"
    )
    merged_lora_count = merge_active_lora_for_inference(
        policy.video_core.dit,
        adapter_name=getattr(policy.video_core, "student_lora_adapter_name", None),
    )
    policy.video_core.student_lora_adapter_name = None
    policy.student_lora_adapter_name = None
    projection_counts: dict[str, dict[str, int]] = {}
    if fuse_qkv:
        projection_counts["stage1"] = fuse_attention_projections(policy.video_core.dit)
        projection_counts["action"] = fuse_attention_projections(policy.action_model)
        expected_projection_counts = {
            "stage1": len(policy.video_core.dit.blocks),
            "action": len(policy.action_model.blocks),
        }
        mismatches = {
            label: (counts, expected_projection_counts[label])
            for (label, counts) in projection_counts.items()
            if counts
            != {
                "self_qkv": expected_projection_counts[label],
                "cross_kv": expected_projection_counts[label],
            }
        }
        if mismatches:
            raise RuntimeError(
                f"Optimized attention projection fusion count mismatch: {mismatches}."
            )
    compile_counts = {
        "video_core_dit_blocks": compile_dit_blocks(
            policy.video_core.dit,
            label="optimized DROID Stage 1 student",
            mode=OPTIMIZED_COMPILE_MODE,
        ),
        "action_model_dit_blocks": compile_dit_blocks(
            policy.action_model, label="optimized DROID ActionDiT", mode=OPTIMIZED_COMPILE_MODE
        ),
    }
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    policy.compile_dit = True
    policy.compile_dit_block_counts = compile_counts
    policy.policy_runtime = runtime
    policy.optimization_contract = {
        "version": 2,
        "precision": "bfloat16",
        "attention_policy": "sdpa_fused_no_math",
        "allowed_attention_backends": ["flash", "cudnn", "efficient"],
        "action_cross_attention_mask": "omitted_after_all_true_validation",
        "action_cross_kv_cache": True,
        "action_text_embedding_cache_per_chunk": True,
        "action_query_prefix_cache_per_chunk": True,
        "compile_dit": True,
        "compile_mode": OPTIMIZED_COMPILE_MODE,
        "compile_block_counts": dict(compile_counts),
        "cuda_graphs": False,
        "qkv_fusion": bool(fuse_qkv),
        "qkv_fused_module_counts": projection_counts,
        "stage1_lora_merged_module_count": int(merged_lora_count),
        "cudnn_benchmark": bool(cudnn_benchmark),
        "vae_rewrite": False,
        "warmup_complete": False,
        "warmup_seconds": None,
    }
    policy._full_policy_warmup = (
        runtime in WAN22_A14B_GRAPH_POLICY_RUNTIMES
        and getattr(policy.video_core, "video_model_family", None) == "wan22_i2v_a14b"
    )
    policy.action_cuda_graph_enabled = graph_enabled
    policy._action_cuda_graphs = {}
    policy._action_cuda_graph_signature = (
        policy._action_graph_signature() if graph_enabled else None
    )
    with torch.inference_mode():
        policy._action_cuda_graph_freqs = (
            policy.action_model.freqs.to(policy.device) if graph_enabled else None
        )
    policy.optimization_contract.update(
        version=3 if graph_enabled else 2,
        cuda_graphs=graph_enabled,
        action_cuda_graph_requested=action_cuda_graph,
        action_cuda_graph_reason=graph_reason,
        action_cuda_graph_scope="action_euler_loop" if graph_enabled else None,
        action_cuda_graph_capture_complete=False,
        action_cuda_graph_capture_count=0,
    )
    policy.vae_first_frame_only_enabled = vae_enabled
    policy.encoder_cuda_graph_enabled = stage1_enabled
    if vae_enabled:
        from dreamfree.models.wan22.wan_video_vae import VideoVAE_

        if type(policy.video_core.vae.model) is not VideoVAE_:
            raise ValueError("First-frame VAE optimization requires the 14B VideoVAE_ encoder.")
        policy._encode_first_frame_vae = partial(encode_first_frame, policy.video_core.vae.model)
    compiled_modules = []
    policy._encoder_cuda_graph = None
    policy._visual_device = (
        torch.device(
            "cuda",
            policy.device.index if policy.device.index is not None else torch.cuda.current_device(),
        )
        if vae_enabled or stage1_enabled
        else None
    )
    policy._visual_optimization_signature = (
        policy._visual_signature() if vae_enabled or stage1_enabled else None
    )
    with torch.inference_mode():
        policy._encoder_graph_freqs = (
            tuple((value.to(policy.device) for value in policy.video_core.dit.freqs))
            if stage1_enabled
            else None
        )
    policy.optimization_contract.update(
        action_cuda_graph_enabled=graph_enabled,
        vae_first_frame_only=vae_enabled,
        vae_first_frame_only_requested=vae_first_frame_only,
        vae_first_frame_only_reason=vae_reason,
        encoder_cuda_graph_enabled=stage1_enabled,
        encoder_cuda_graph_requested=encoder_cuda_graph,
        encoder_cuda_graph_reason=stage1_reason,
        encoder_cuda_graph_scope="current_frame_student_tokens" if stage1_enabled else None,
        encoder_cuda_graph_capture_complete=False,
        encoder_cuda_graph_capture_count=0,
    )
    if vae_enabled or stage1_enabled:
        policy.optimization_contract.update(
            version=4, cuda_graphs=graph_enabled or stage1_enabled, vae_rewrite=vae_enabled
        )
    policy.optimization_contract.update(
        compile_auxiliary=compile_auxiliary,
        auxiliary_compiled_modules=compiled_modules,
        auxiliary_compile_settings={
            "mode": OPTIMIZED_COMPILE_MODE,
            "fullgraph": True,
            "dynamic": False,
        }
        if compiled_modules
        else {},
    )
    return policy


def _base_args(usr_args: Dict[str, Any]) -> tuple[dict[str, Any], bool, bool]:
    if str(usr_args.get("mixed_precision") or "bf16").strip().lower() != "bf16":
        raise ValueError("Optimized DROID inference requires mixed_precision='bf16'.")
    if not droid_base._parse_bool(usr_args.get("compile_dit", True)):
        raise ValueError("Optimized DROID inference requires compile_dit=true.")
    args = dict(usr_args)
    args["compile_dit"] = False
    fuse_qkv = droid_base._parse_bool(usr_args.get("fuse_qkv", True))
    cudnn_benchmark = droid_base._parse_bool(usr_args.get("cudnn_benchmark", True))
    return (args, fuse_qkv, cudnn_benchmark)


def build_policy_from_args(usr_args: Dict[str, Any]) -> OptimizedNativeDroidPolicy:
    from dreamfree.config_names import INFERENCE_RENAMES, rename_fields

    usr_args = rename_fields(usr_args, INFERENCE_RENAMES)
    resolve_auxiliary_compile(usr_args.get("compile_auxiliary"), OPTIMIZED_NATIVE_POLICY_RUNTIME)
    (args, fuse_qkv, cudnn_benchmark) = _base_args(usr_args)
    policy = droid_base.build_policy_from_args(
        args,
        policy_runtime_override=OPTIMIZED_NATIVE_POLICY_RUNTIME,
        policy_cls=OptimizedNativeDroidPolicy,
    )
    return _optimize_policy(
        policy,
        runtime=OPTIMIZED_NATIVE_POLICY_RUNTIME,
        fuse_qkv=fuse_qkv,
        cudnn_benchmark=cudnn_benchmark,
        action_cuda_graph=usr_args.get("action_cuda_graph", "auto"),
        vae_first_frame_only=usr_args.get("vae_first_frame_only", "auto"),
        encoder_cuda_graph=usr_args.get("encoder_cuda_graph", "auto"),
    )
