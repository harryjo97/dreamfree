"""Metadata-only RoboCasa optimization contract shared by workers and preflight."""

from __future__ import annotations

from typing import Any

from dreamfree.evaluation.robocasa._eval_common import DREAMFREE_STAGE2_CHECKPOINT_TYPE

FEATURES = ("action_cuda_graph", "vae_first_frame_only", "stage1_cuda_graph")
SUPPORTED_TYPES = frozenset((DREAMFREE_STAGE2_CHECKPOINT_TYPE,))
SUPPORTED_5B_TYPES = frozenset((DREAMFREE_STAGE2_CHECKPOINT_TYPE,))


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def parse_mode(value: Any, name: str) -> str:
    if isinstance(value, bool):
        value = str(value).lower()
    if value not in ("auto", "true", "false"):
        raise ValueError(f"{name} must be auto, true, or false, got {value!r}.")
    return value


def resolve_options(
    metadata: dict,
    args: dict,
    *,
    promoted_features: frozenset[str] = frozenset(),
    promoted_5b_features: frozenset[str] = frozenset(),
) -> dict:
    requested = {
        name: parse_mode(args.get(name, "auto"), name)
        for name in ("inference_optimization", *FEATURES)
    }
    master = requested["inference_optimization"]
    explicit_features = [name for name in FEATURES if requested[name] == "true"]
    if master == "false" and explicit_features:
        raise ValueError(
            f"inference_optimization=false conflicts with explicitly enabled {explicit_features}."
        )
    (dataset, student) = (metadata.get("dataset", {}), metadata.get("student", {}))
    (config, action) = (metadata.get("config", {}), metadata.get("action", {}))
    family = metadata.get("model_contract", {}).get(
        "video_model_family", config.get("video_model_family")
    )
    is_5b = family == "wan22_ti2v_5b"
    if is_5b and requested["vae_first_frame_only"] == "true":
        raise ValueError(
            "vae_first_frame_only=true is a 14B shortcut; 5B already encodes one frame. Use vae_first_frame_only=auto or false for 5B."
        )
    family_supported = (
        family == "wan22_i2v_a14b"
        and metadata.get("checkpoint_type") in SUPPORTED_TYPES
        and (student.get("tokens_per_frame") == 484)
        and (student.get("hidden_dim") == 5120)
        or (
            is_5b
            and metadata.get("checkpoint_type") in SUPPORTED_5B_TYPES
            and (student.get("tokens_per_frame") == 121)
            and (student.get("hidden_dim") == 3072)
        )
    )
    supported = (
        family_supported
        and str(args.get("device", "cuda")).startswith("cuda")
        and (args.get("mixed_precision", "bf16") == "bf16")
        and (not _parse_bool(args.get("tiled", False)))
        and (dataset.get("video_size") == [352, 352])
        and (dataset.get("context_len") == 128)
        and (action.get("action_dim") == 7)
        and (action.get("action_horizon") == 32)
        and (student.get("hook_point") == "post_block")
        and (config.get("num_frames") == 33)
        and (config.get("action_video_freq_ratio") == 4)
        and (int(args.get("num_inference_steps") or 10) in (10, 16))
    )
    if not supported and (master == "true" or explicit_features):
        raise ValueError(
            f"checkpoint_type={metadata.get('checkpoint_type')!r}, family={family!r}: RoboCasa inference optimization requires standard 14B or 5B, CUDA BF16, untiled 352x352 input, context 128, post-block tokens (14B: 484x5120; 5B: 121x3072), horizon 32/action dim 7, num_frames=33/action_video_freq_ratio=4, and 10 or 16 Euler steps."
        )
    promoted_features = promoted_5b_features if is_5b else promoted_features
    promoted_scope = (
        supported
        and metadata.get("checkpoint_type") == DREAMFREE_STAGE2_CHECKPOINT_TYPE
        and (
            metadata.get("readout", {}).get("input_layer_j", student.get("layer_j"))
            == (15 if is_5b else 31)
        )
        and (int(args.get("num_inference_steps") or 10) == 10)
        and (metadata.get("readout", {}).get("num_queries") == 64)
        and (args.get("readout_active_num_queries") in (None, 64))
    )
    enabled = (
        supported
        and master != "false"
        and (
            master == "true"
            or bool(explicit_features)
            or (promoted_scope and bool(promoted_features))
        )
    )
    reason = (
        "explicitly_disabled"
        if master == "false"
        else "unsupported_configuration"
        if not supported
        else "explicitly_enabled"
        if master == "true" or explicit_features
        else "validated_default"
        if enabled
        else "awaiting_robocasa_validation"
    )
    resolved = {}
    for name in FEATURES:
        value = requested[name]
        active = enabled and (
            value == "true" or (value == "auto" and promoted_scope and (name in promoted_features))
        )
        resolved[name] = {
            "requested": value,
            "enabled": bool(active),
            "reason": "explicitly_enabled"
            if active and value == "true"
            else "validated_default"
            if active
            else "explicitly_disabled"
            if value == "false"
            else reason
            if not enabled
            else "not_promoted",
        }
    if is_5b:
        resolved["vae_first_frame_only"].update(enabled=False, reason="already_single_frame")
        if (
            requested["stage1_cuda_graph"] == "auto"
            and resolved["stage1_cuda_graph"]["enabled"]
            and ("action_cuda_graph" in promoted_features)
            and (not resolved["action_cuda_graph"]["enabled"])
        ):
            resolved["stage1_cuda_graph"].update(
                enabled=False, reason="requires_promoted_action_graph"
            )
    return dict(
        version=1,
        requested=requested,
        enabled=bool(enabled),
        reason=reason,
        features=resolved,
        warmup_complete=False,
        compile_dit_requested=_parse_bool(args.get("compile_dit", False)),
    )
