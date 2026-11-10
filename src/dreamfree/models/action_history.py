"""Frozen-student Stage 2 history configuration, artifact contract, and query prefix."""

import math
from types import SimpleNamespace

from dreamfree.data import yam

CHECKPOINT_TYPE = "dreamfree_stage2_action_history"
CHECKPOINT_VERSION = 1
TOKEN_ORDER = "proprio_then_previous_action_then_future_action"


def validate_config(cfg):
    horizon = getattr(cfg, "action_history_horizon", 0)
    dropout = getattr(cfg, "action_history_dropout", None)
    if type(horizon) is not int or horizon < 0:
        raise ValueError("action_history_horizon must be a nonnegative integer.")
    if not horizon:
        if dropout is not None:
            raise ValueError("action_history_dropout requires a positive action_history_horizon.")
        return
    if cfg.dataset_name not in (*yam.DATASET_NAMES, "droid"):
        raise ValueError("DreamFree action-history Stage 2 supports DROID and ABC joint/EEF.")
    if dropout is None or not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("Set action_history_dropout explicitly to a finite value in [0,1).")
    if cfg.student_train_mode != "frozen":
        raise ValueError("Action-history Stage 2 requires a frozen student.")
    if cfg.proprio_conditioning_mode != "query_prefix" or cfg.proprio_obs_horizon != 1:
        raise ValueError("Action history requires one current-proprio query-prefix token.")
    if 1 + horizon + cfg.action_horizon > 1024:
        raise ValueError("Proprio + history + future actions exceed the 1024-token RoPE cache.")


def checkpoint_identity(cfg):
    return (
        (CHECKPOINT_TYPE, CHECKPOINT_VERSION)
        if getattr(cfg, "action_history_horizon", 0)
        else ("dreamfree_stage2", 2)
    )


def contract(cfg):
    validate_config(cfg)
    if not getattr(cfg, "action_history_horizon", 0):
        return None
    return dict(
        horizon=cfg.action_history_horizon,
        padding="left_zero_masked",
        token_order=TOKEN_ORDER,
        action_encoder="shared_action_encoder",
        timestep_sigma=float(cfg.proprio_state_timestep_sigma),
        invalid_tokens="masked_as_self_attention_keys",
        loss_applies_to="future_action_suffix_only",
        action_layout=(
            "relative_joint7_absolute_gripper"
            if cfg.dataset_name == "droid"
            else yam.PRESETS[cfg.dataset_name].contract["action_layout"]
        ),
        normalization="dataset_q01_q99",
        deployment_source="accepted_issued_policy_actions",
        condition_dropout_prob=float(cfg.action_history_dropout),
        condition_dropout_granularity="samplewise_full_history",
        condition_dropout_train_only=True,
        **(
            dict(
                droid_joint_anchor="current_planning_state",
                droid_gripper_history="binary_command_gt_0.5",
            )
            if cfg.dataset_name == "droid"
            else {}
        ),
    )


def add_checkpoint_metadata(metadata, cfg):
    history = contract(cfg)
    if history is None:
        return
    metadata.update(
        checkpoint_type=CHECKPOINT_TYPE,
        checkpoint_version=CHECKPOINT_VERSION,
        training_variant="action_history",
        action_history=history,
    )
    metadata["conditioner"]["query_prefix_tokens"] = 1 + cfg.action_history_horizon
    metadata["shapes"].update(
        query_prefix_tokens=1 + cfg.action_history_horizon,
        action_history_horizon=cfg.action_history_horizon,
        query_token_order=TOKEN_ORDER,
    )


def validate_checkpoint(metadata, cfg=None):
    """Check artifact/config agreement and forbid switching the experiment on resume."""
    saved = SimpleNamespace(**metadata["config"])
    validate_config(saved)
    expected_type, expected_version = checkpoint_identity(saved)
    if (metadata.get("checkpoint_type"), metadata.get("checkpoint_version")) != (
        expected_type,
        expected_version,
    ):
        raise ValueError(
            "Stage 2 checkpoint type/version disagrees with its action-history config."
        )
    history = contract(saved)
    if metadata.get("action_history") != history or metadata.get("training_variant") != (
        "action_history" if history else None
    ):
        raise ValueError("Stage 2 action-history contract differs from its configuration.")
    if cfg is not None and contract(cfg) != history:
        raise ValueError("Resume action-history settings differ from the saved checkpoint.")
    if history:
        proprio = metadata.get("action", {}).get("proprio_conditioning", {})
        if (
            proprio.get("mode") != "query_prefix"
            or proprio.get("obs_horizon") != 1
            or proprio.get("state_timestep_sigma") != history["timestep_sigma"]
        ):
            raise ValueError("Action-history checkpoint proprio-prefix settings are inconsistent.")
        prefix = 1 + history["horizon"]
        if (
            metadata.get("conditioner", {}).get("query_prefix_tokens") != prefix
            or metadata.get("shapes", {}).get("query_prefix_tokens") != prefix
            or metadata.get("shapes", {}).get("action_history_horizon") != history["horizon"]
            or metadata.get("shapes", {}).get("query_token_order") != TOKEN_ORDER
        ):
            raise ValueError(
                "Action-history checkpoint query-prefix geometry/order is inconsistent."
            )


def build_query_prefix(
    action_encoder,
    proprio,
    previous,
    is_pad,
    *,
    horizon,
    action_dim,
    future_tokens,
    dropout=0.0,
    training=False,
):
    """Embed clean history once; padded/dropped tokens cannot act as attention keys."""
    import torch

    batch = proprio.shape[0]
    if not isinstance(previous, torch.Tensor) or tuple(previous.shape) != (
        batch,
        horizon,
        action_dim,
    ):
        raise ValueError(f"previous_action must have shape [{batch},{horizon},{action_dim}].")
    if (
        not isinstance(is_pad, torch.Tensor)
        or is_pad.dtype != torch.bool
        or tuple(is_pad.shape) != (batch, horizon)
    ):
        raise ValueError(f"previous_action_is_pad must be boolean [{batch},{horizon}].")
    if not torch.isfinite(previous).all():
        raise ValueError("Previous actions contain nonfinite values.")
    if (previous[is_pad] != 0).any() or (is_pad[:, 1:] & ~is_pad[:, :-1]).any():
        raise ValueError("Previous-action padding must be contiguous left zeros.")
    dropped = (
        torch.rand((batch, 1), device=previous.device) < dropout
        if training and dropout > 0
        else torch.zeros((batch, 1), device=previous.device, dtype=torch.bool)
    )
    effective_pad = is_pad | dropped
    history_emb = action_encoder(previous.masked_fill(effective_pad.unsqueeze(-1), 0.0))
    prefix = torch.cat((proprio, history_emb), dim=1)
    mask = torch.cat(
        (
            torch.ones((batch, proprio.shape[1]), device=proprio.device, dtype=torch.bool),
            ~effective_pad,
            torch.ones((batch, future_tokens), device=proprio.device, dtype=torch.bool),
        ),
        dim=1,
    )
    fractions = torch.stack(
        ((~is_pad).float().mean(), (~effective_pad).float().mean(), dropped.float().mean())
    )
    return prefix, mask, fractions
