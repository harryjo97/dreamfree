"""Shared optimized action inference for LIBERO and RoboTwin."""

from __future__ import annotations

import torch

from dreamfree.evaluation.droid.deploy_policy_optimized import OptimizedDroidInferenceMixin
from dreamfree.evaluation.libero.policy import LiberoPolicy
from dreamfree.evaluation.robotwin.policy import RobotWinPolicy
from dreamfree.models.compile import compile_dit_blocks
from dreamfree.models.inference_optimization import (
    fuse_attention_projections,
    fused_sdpa_only,
    merge_active_lora_for_inference,
)


class OptimizedActionMixin(OptimizedDroidInferenceMixin):
    """Reuse prepared action conditioning while keeping benchmark preprocessing."""

    def _action_inference_kwargs(self):
        return {"preserve_context_mask": self.action_preserve_context_mask}

    def _infer_action_chunk(self, observation, instruction):
        with torch.inference_mode(), fused_sdpa_only():
            return super()._infer_action_chunk(observation, instruction)


class OptimizedLiberoPolicy(OptimizedActionMixin, LiberoPolicy):
    pass


class OptimizedRobotWinPolicy(OptimizedActionMixin, RobotWinPolicy):
    pass


def optimize_policy(policy):
    if policy.model_dtype != torch.bfloat16 or policy.device.type != "cuda":
        raise ValueError("Optimized inference requires CUDA BF16.")
    merge_active_lora_for_inference(
        policy.video_core.dit, adapter_name=policy.student_lora_adapter_name
    )
    policy.student_lora_adapter_name = policy.video_core.student_lora_adapter_name = None
    is_5b = policy.video_core.video_model_family == "wan22_ti2v_5b"
    policy.action_preserve_context_mask = is_5b
    counts = {}
    for label, model in (("student", policy.video_core.dit), ("action", policy.action_model)):
        if not is_5b:
            fuse_attention_projections(model)
        counts[label] = compile_dit_blocks(
            model, label=label, mode="default" if is_5b else "max-autotune-no-cudagraphs"
        )
    policy.optimization_contract = dict(
        attention_policy="sdpa_fused_no_math",
        action_cross_kv_cache=True,
        compile_dit=True,
        compile_counts=counts,
    )
