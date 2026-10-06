from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from dreamfree.models.wan22.helpers.distributed_load import (
    broadcast_module_state,
    get_distributed_rank,
    get_pretrained_broadcast_device,
    should_broadcast_pretrained_load,
    sync_distributed_load_error,
)
from dreamfree.models.wan22.helpers.gradient import gradient_checkpoint_forward
from dreamfree.models.wan22.wan_video_dit import (
    DiTBlock,
    _gelu_approx_tanh,
    precompute_freqs_cis,
    sinusoidal_embedding_1d,
)
from dreamfree.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PreparedActionConditioning:
    """Per-layer cross-attention K/V reused for one action denoising chunk."""

    block_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    batch_size: int
    context_length: int
    hidden_dim: int
    device: torch.device
    dtype: torch.dtype


class ActionDiT(nn.Module):
    ACTION_BACKBONE_SKIP_PREFIXES = ("action_encoder.", "head.")
    ACTION_BACKBONE_META_KEYS = (
        "hidden_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "freq_dim",
        "eps",
    )

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), _gelu_approx_tanh(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=attn_head_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(hidden_dim, action_dim)
        self.freqs = precompute_freqs_cis(attn_head_dim, end=1024)
        self.use_gradient_checkpointing = use_gradient_checkpointing

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        return {
            key
            for key in keys
            if not any(key.startswith(prefix) for prefix in cls.ACTION_BACKBONE_SKIP_PREFIXES)
        }

    @classmethod
    def from_pretrained(
        cls,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        pretrained_load_broadcast: bool = True,
    ) -> "ActionDiT":
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required for ActionDiT.from_pretrained().")
        if skip_dit_load_from_pretrain:
            logger.info(
                "Skipping ActionDiT pretrained load (`skip_dit_load_from_pretrain=True`); initializing action expert randomly and expecting checkpoint override."
            )
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        if not action_dit_pretrained_path:
            logger.info(
                "No `action_dit_pretrained_path` provided, initializing ActionDiT with random weights."
            )
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        from pathlib import Path

        p = Path(action_dit_pretrained_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[4] / p
        action_dit_pretrained_path = str(p)
        if not os.path.isfile(action_dit_pretrained_path):
            raise FileNotFoundError(
                f"`action_dit_pretrained_path` does not exist: {action_dit_pretrained_path}"
            )
        use_broadcast = should_broadcast_pretrained_load(pretrained_load_broadcast)
        broadcast_device = get_pretrained_broadcast_device() if use_broadcast else "cuda"
        load_device = "cpu" if broadcast_device == "cpu" else device
        action_cfg = dict(action_dit_config)
        action_expert = cls(**action_cfg).to(device=load_device, dtype=torch_dtype)
        rank = get_distributed_rank()
        load_error: BaseException | None = None
        if use_broadcast and rank != 0:
            sync_distributed_load_error(
                load_error, device=load_device, src=0, context="rank0 ActionDiT backbone load"
            )
            broadcast_module_state(action_expert, src=0)
            if broadcast_device == "cpu":
                action_expert = action_expert.to(device=device, dtype=torch_dtype)
            logger.info("Loaded ActionDiT backbone from rank0 %s broadcast.", broadcast_device)
            return action_expert
        try:
            cls._load_pretrained_backbone_into(
                action_expert=action_expert,
                action_cfg=action_cfg,
                action_dit_pretrained_path=action_dit_pretrained_path,
            )
        except BaseException as exc:
            load_error = exc
        if use_broadcast:
            sync_distributed_load_error(
                load_error, device=load_device, src=0, context="rank0 ActionDiT backbone load"
            )
            broadcast_module_state(action_expert, src=0)
            if broadcast_device == "cpu":
                action_expert = action_expert.to(device=device, dtype=torch_dtype)
            logger.info(
                "Loaded ActionDiT backbone from %s via rank0 %s broadcast.",
                action_dit_pretrained_path,
                broadcast_device,
            )
            return action_expert
        if load_error is not None:
            raise load_error
        return action_expert

    @classmethod
    def _load_pretrained_backbone_into(
        cls,
        *,
        action_expert: "ActionDiT",
        action_cfg: dict[str, Any],
        action_dit_pretrained_path: str,
    ) -> None:
        action_state = action_expert.state_dict()
        expected_backbone_keys = cls.backbone_key_set(action_state.keys())
        payload = torch.load(action_dit_pretrained_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid action backbone payload type from {action_dit_pretrained_path}: {type(payload)}"
            )
        policy = payload.get("policy", {})
        if policy:
            logger.info(f"ActionDiT backbone payload policy: {policy}")
        meta = payload.get("meta")
        expected_meta = {
            "hidden_dim": int(action_cfg["hidden_dim"]),
            "ffn_dim": int(action_cfg["ffn_dim"]),
            "num_layers": int(action_cfg["num_layers"]),
            "num_heads": int(action_cfg["num_heads"]),
            "attn_head_dim": int(action_cfg["attn_head_dim"]),
            "text_dim": int(action_cfg["text_dim"]),
            "freq_dim": int(action_cfg["freq_dim"]),
            "eps": float(action_cfg["eps"]),
        }
        for key in cls.ACTION_BACKBONE_META_KEYS:
            if key not in meta:
                raise ValueError(f"`meta.{key}` missing in {action_dit_pretrained_path}")
            expected_value = expected_meta[key]
            got_value = meta[key]
            if key == "eps":
                if abs(float(got_value) - float(expected_value)) > 1e-12:
                    raise ValueError(
                        f"`meta.{key}` mismatch in {action_dit_pretrained_path}: expected {expected_value}, got {got_value}"
                    )
            elif int(got_value) != int(expected_value):
                raise ValueError(
                    f"`meta.{key}` mismatch in {action_dit_pretrained_path}: expected {expected_value}, got {got_value}"
                )
        backbone_state_dict = payload.get("backbone_state_dict")
        if not isinstance(backbone_state_dict, dict):
            raise ValueError(
                f"`backbone_state_dict` must be a dict in {action_dit_pretrained_path}, got {type(backbone_state_dict)}"
            )
        provided_keys = set(backbone_state_dict.keys())
        missing_keys = sorted(expected_backbone_keys - provided_keys)
        unexpected_keys = sorted(provided_keys - expected_backbone_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(
                f"Action backbone key mismatch in preprocessed payload. missing={missing_keys[:10]}{('...' if len(missing_keys) > 10 else '')}, unexpected={unexpected_keys[:10]}{('...' if len(unexpected_keys) > 10 else '')}"
            )
        merged_state = dict(action_state)
        for key in expected_backbone_keys:
            value = backbone_state_dict[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"`backbone_state_dict[{key}]` must be torch.Tensor in {action_dit_pretrained_path}, got {type(value)}"
                )
            target = merged_state[key]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for `{key}` in {action_dit_pretrained_path}: expected {tuple(target.shape)}, got {tuple(value.shape)}"
                )
            merged_state[key] = value.to(device=target.device, dtype=target.dtype)
        action_expert.load_state_dict(merged_state, strict=True)
        logger.info(
            "Loaded ActionDiT backbone from %s (keys=%d; random_kept_prefixes=%s).",
            action_dit_pretrained_path,
            len(expected_backbone_keys),
            list(cls.ACTION_BACKBONE_SKIP_PREFIXES),
        )

    def _build_time_mod(self, timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if timestep.ndim == 1:
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            return (t, self.time_projection(t).unflatten(1, (6, self.hidden_dim)))
        if timestep.ndim != 2:
            raise ValueError(f"`timestep` must be 1D or 2D, got shape {tuple(timestep.shape)}")
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep.reshape(-1)))
        t = t.reshape(timestep.shape[0], timestep.shape[1], self.hidden_dim)
        return (t, self.time_projection(t).unflatten(2, (6, self.hidden_dim)))

    def _run_blocks(
        self,
        *,
        tokens: torch.Tensor,
        context: Optional[torch.Tensor],
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_kv: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None,
    ) -> torch.Tensor:
        x = tokens
        if cross_attn_kv is not None and len(cross_attn_kv) != len(self.blocks):
            raise ValueError(
                f"Prepared ActionDiT K/V layer count mismatch: cache={len(cross_attn_kv)} blocks={len(self.blocks)}."
            )
        for block_idx, block in enumerate(self.blocks):
            if self.use_gradient_checkpointing:
                if cross_attn_kv is not None:
                    raise ValueError(
                        "Prepared ActionDiT K/V is inference-only and cannot use gradient checkpointing."
                    )
                x = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x,
                    context,
                    t_mod,
                    freqs,
                    context_mask=context_mask,
                    self_attn_mask=self_attn_mask,
                )
            else:
                x = block(
                    x,
                    context,
                    t_mod,
                    freqs,
                    context_mask=context_mask,
                    self_attn_mask=self_attn_mask,
                    cross_attn_kv=None if cross_attn_kv is None else cross_attn_kv[block_idx],
                )
        return x

    def _pre_action_tokens(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        prepared_time: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        if action_tokens.ndim != 3:
            raise ValueError(
                f"`action_tokens` must be 3D [B, T, action_dim], got shape {tuple(action_tokens.shape)}"
            )
        if action_tokens.shape[2] != self.action_dim:
            raise ValueError(
                f"`action_tokens` last dim must be {self.action_dim}, got {action_tokens.shape[2]}"
            )
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        batch_size = action_tokens.shape[0]
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, action timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)
        seq_len = action_tokens.shape[1]
        if seq_len > self.freqs.shape[0]:
            raise ValueError(
                f"Action token length {seq_len} exceeds RoPE cache {self.freqs.shape[0]}."
            )
        if prepared_time is None:
            (t, t_mod) = self._build_time_mod(timestep)
        else:
            if self.training or torch.is_grad_enabled():
                raise ValueError("Prepared action timesteps require frozen inference.")
            (t, t_mod) = prepared_time
            for value, shape in (
                (t, (batch_size, self.hidden_dim)),
                (t_mod, (batch_size, 6, self.hidden_dim)),
            ):
                if (
                    tuple(value.shape) != shape
                    or value.device != timestep.device
                    or value.dtype != action_tokens.dtype
                ):
                    raise ValueError("Prepared action timestep shape/device/dtype mismatch.")
        tokens = self.action_encoder(action_tokens)
        freqs = self.freqs[:seq_len].view(seq_len, 1, -1).to(tokens.device)
        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "meta": {"batch_size": batch_size, "seq_len": seq_len},
        }

    def _embed_text_context(
        self, context: torch.Tensor, context_mask: Optional[torch.Tensor], *, batch_size: int
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B,L,D], got shape {tuple(context.shape)}")
        if int(context.shape[0]) != batch_size:
            raise ValueError(
                f"Batch mismatch between action tokens and text context: {batch_size} vs {context.shape[0]}"
            )
        if int(context.shape[-1]) != self.text_dim:
            raise ValueError(f"Text context dim must be {self.text_dim}, got {context.shape[-1]}.")
        if context_mask is not None:
            if context_mask.ndim != 2 or tuple(context_mask.shape) != tuple(context.shape[:2]):
                raise ValueError(
                    f"`context_mask` shape must match `context` [B,L], got {tuple(context_mask.shape)} vs {tuple(context.shape[:2])}."
                )
            context_mask = context_mask.to(device=context.device, dtype=torch.bool)
        return (self.text_embedding(context), context_mask)

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        state = self._pre_action_tokens(action_tokens, timestep)
        batch_size = int(state["meta"]["batch_size"])
        seq_len = int(state["meta"]["seq_len"])
        (context_emb, context_mask) = self._embed_text_context(
            context, context_mask, batch_size=batch_size
        )
        state.update(
            {
                "context": context_emb,
                "context_token_mask": context_mask,
                "context_mask": None
                if context_mask is None
                else context_mask.unsqueeze(1).expand(-1, seq_len, -1),
            }
        )
        return state

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        return self.head(tokens)

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pre_state = self.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        x = pre_state["tokens"]
        context = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_mask = pre_state["context_mask"]
        x = self._run_blocks(
            tokens=x, context=context, t_mod=t_mod, freqs=freqs, context_mask=context_mask
        )
        return self.post_dit(x, pre_state)

    @torch.no_grad()
    def prepare_cross_attention_conditioning(
        self,
        *,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        extra_context_emb: Optional[torch.Tensor] = None,
        extra_context_mask: Optional[torch.Tensor] = None,
        masks_validated: bool = False,
    ) -> PreparedActionConditioning:
        """Prepare constant cross-attention K/V once for one denoising chunk."""
        batch_size = int(context.shape[0]) if context.ndim >= 1 else 0
        (context_emb, context_mask) = self._embed_text_context(
            context, context_mask, batch_size=batch_size
        )
        if (
            context_mask is not None
            and (not masks_validated)
            and (not bool(context_mask.all().item()))
        ):
            raise ValueError(
                "Optimized ActionDiT requires an all-true text context mask before omitting the attention mask."
            )
        cross_context = context_emb
        if extra_context_emb is not None:
            if extra_context_emb.ndim != 3 or tuple(extra_context_emb.shape[:1]) != (batch_size,):
                raise ValueError(
                    f"`extra_context_emb` must have shape [B,S,hidden_dim], got {tuple(extra_context_emb.shape)}."
                )
            if int(extra_context_emb.shape[-1]) != self.hidden_dim:
                raise ValueError(
                    f"`extra_context_emb` hidden dim must be {self.hidden_dim}, got {extra_context_emb.shape[-1]}."
                )
            if extra_context_mask is not None:
                if extra_context_mask.ndim != 2 or tuple(extra_context_mask.shape) != tuple(
                    extra_context_emb.shape[:2]
                ):
                    raise ValueError(
                        f"`extra_context_mask` must match extra context [B,S], got {tuple(extra_context_mask.shape)} vs {tuple(extra_context_emb.shape[:2])}."
                    )
                extra_context_mask = extra_context_mask.to(
                    device=extra_context_emb.device, dtype=torch.bool
                )
                if not masks_validated and (not bool(extra_context_mask.all().item())):
                    raise ValueError(
                        "Optimized ActionDiT requires an all-true extra context mask before omitting the attention mask."
                    )
            cross_context = torch.cat((cross_context, extra_context_emb), dim=1)
        elif extra_context_mask is not None:
            raise ValueError("`extra_context_mask` requires `extra_context_emb`.")
        block_kv = tuple(block.cross_attn.project_context(cross_context) for block in self.blocks)
        return PreparedActionConditioning(
            block_kv=block_kv,
            batch_size=batch_size,
            context_length=int(cross_context.shape[1]),
            hidden_dim=self.hidden_dim,
            device=cross_context.device,
            dtype=cross_context.dtype,
        )

    def _build_query_state(
        self,
        *,
        pre_state: Dict[str, Any],
        timestep: torch.Tensor,
        query_prefix_emb: Optional[torch.Tensor],
        query_prefix_timestep_sigma: Optional[float],
        query_token_mask: Optional[torch.Tensor],
        num_train_timesteps: float,
        validate_query_token_mask: bool = True,
        prefix_time_mod: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
        batch_size = int(pre_state["meta"]["batch_size"])
        action_seq_len = int(pre_state["meta"]["seq_len"])
        tokens = pre_state["tokens"]
        prefix_len = 0
        if query_prefix_emb is not None:
            if query_prefix_emb.ndim != 3 or int(query_prefix_emb.shape[0]) != batch_size:
                raise ValueError(
                    f"`query_prefix_emb` must have shape [B,P,hidden_dim], got {tuple(query_prefix_emb.shape)}."
                )
            if int(query_prefix_emb.shape[-1]) != self.hidden_dim:
                raise ValueError(
                    f"`query_prefix_emb` hidden dim must be {self.hidden_dim}, got {query_prefix_emb.shape[-1]}."
                )
            prefix_len = int(query_prefix_emb.shape[1])
            tokens = torch.cat((query_prefix_emb, tokens), dim=1)
        query_seq_len = int(tokens.shape[1])
        if query_seq_len > self.freqs.shape[0]:
            raise ValueError(
                f"Combined query token length {query_seq_len} exceeds RoPE cache {self.freqs.shape[0]}."
            )
        self_attn_mask = None
        if query_token_mask is not None:
            if query_token_mask.ndim != 2 or tuple(query_token_mask.shape) != (
                batch_size,
                query_seq_len,
            ):
                raise ValueError(
                    f"`query_token_mask` must have shape [B,query_seq_len], got {tuple(query_token_mask.shape)} expected={(batch_size, query_seq_len)}."
                )
            query_token_mask = query_token_mask.to(device=tokens.device, dtype=torch.bool)
            if validate_query_token_mask and bool((~query_token_mask).all(dim=1).any().item()):
                raise ValueError(
                    "`query_token_mask` must leave at least one valid key token per sample."
                )
            self_attn_mask = query_token_mask[:, None, None, :].expand(
                batch_size, 1, query_seq_len, query_seq_len
            )
        if prefix_len > 0:
            if query_prefix_timestep_sigma is None:
                raise ValueError("`query_prefix_timestep_sigma` is required with a query prefix.")
            if float(num_train_timesteps) <= 0:
                raise ValueError(
                    f"`num_train_timesteps` must be positive, got {num_train_timesteps}."
                )
            if prefix_time_mod is None:
                prefix_timestep = torch.full(
                    (batch_size,),
                    float(query_prefix_timestep_sigma) * float(num_train_timesteps),
                    device=timestep.device,
                    dtype=timestep.dtype,
                )
                (_, prefix_t_mod) = self._build_time_mod(prefix_timestep)
            else:
                if (
                    self.training
                    or torch.is_grad_enabled()
                    or tuple(prefix_time_mod.shape) != (batch_size, 6, self.hidden_dim)
                    or (prefix_time_mod.device != timestep.device)
                    or (prefix_time_mod.dtype != tokens.dtype)
                ):
                    raise ValueError(
                        "Prepared prefix timestep requires matching shape/device/dtype and frozen inference."
                    )
                prefix_t_mod = prefix_time_mod
            t_mod = torch.cat(
                (
                    prefix_t_mod.unsqueeze(1).expand(-1, prefix_len, -1, -1),
                    pre_state["t_mod"].unsqueeze(1).expand(-1, action_seq_len, -1, -1),
                ),
                dim=1,
            )
        else:
            t_mod = pre_state["t_mod"]
        freqs = self.freqs[:query_seq_len].view(query_seq_len, 1, -1).to(tokens.device)
        return (tokens, t_mod, freqs, self_attn_mask, prefix_len)

    def forward_with_query_prefix_and_prepared_context(
        self,
        *,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        prepared_conditioning: PreparedActionConditioning,
        query_prefix_emb: Optional[torch.Tensor] = None,
        query_prefix_timestep_sigma: Optional[float] = None,
        query_token_mask: Optional[torch.Tensor] = None,
        query_token_mask_validated: bool = False,
        num_train_timesteps: float = 1000.0,
        preserve_context_mask: bool = False,
        prepared_time: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        prefix_time_mod: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run ActionDiT while reusing per-layer cross-attention K/V."""
        pre_state = self._pre_action_tokens(action_tokens, timestep, prepared_time=prepared_time)
        batch_size = int(pre_state["meta"]["batch_size"])
        if prepared_conditioning.batch_size != batch_size:
            raise ValueError(
                f"Prepared ActionDiT batch mismatch: cache={prepared_conditioning.batch_size} action={batch_size}."
            )
        if prepared_conditioning.hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Prepared ActionDiT hidden dim mismatch: cache={prepared_conditioning.hidden_dim} model={self.hidden_dim}."
            )
        if (
            prepared_conditioning.device != action_tokens.device
            or prepared_conditioning.dtype != action_tokens.dtype
        ):
            raise ValueError(
                f"Prepared ActionDiT device/dtype mismatch: cache=({prepared_conditioning.device},{prepared_conditioning.dtype}) action=({action_tokens.device},{action_tokens.dtype})."
            )
        if len(prepared_conditioning.block_kv) != len(self.blocks):
            raise ValueError(
                f"Prepared ActionDiT K/V layer count mismatch: cache={len(prepared_conditioning.block_kv)} blocks={len(self.blocks)}."
            )
        if prepared_conditioning.context_length <= 0:
            raise ValueError(
                f"Prepared ActionDiT context length must be positive, got {prepared_conditioning.context_length}."
            )
        for block_idx, ((k, v), block) in enumerate(
            zip(prepared_conditioning.block_kv, self.blocks)
        ):
            expected = (
                batch_size,
                prepared_conditioning.context_length,
                block.cross_attn.attn_hidden_dim,
            )
            if tuple(k.shape) != expected or tuple(v.shape) != expected:
                raise ValueError(
                    f"Prepared ActionDiT K/V shape mismatch at layer {block_idx}: k={tuple(k.shape)} v={tuple(v.shape)} expected={expected}."
                )
        (tokens, t_mod, freqs, self_attn_mask, _) = self._build_query_state(
            pre_state=pre_state,
            timestep=timestep,
            query_prefix_emb=query_prefix_emb,
            query_prefix_timestep_sigma=query_prefix_timestep_sigma,
            query_token_mask=query_token_mask,
            num_train_timesteps=num_train_timesteps,
            validate_query_token_mask=not query_token_mask_validated,
            prefix_time_mod=prefix_time_mod,
        )
        context_mask = None
        if preserve_context_mask:
            context_mask = (
                torch.ones(
                    (batch_size, prepared_conditioning.context_length),
                    device=action_tokens.device,
                    dtype=torch.bool,
                )
                .unsqueeze(1)
                .expand(-1, int(tokens.shape[1]), -1)
            )
        x = self._run_blocks(
            tokens=tokens,
            context=None,
            t_mod=t_mod,
            freqs=freqs,
            context_mask=context_mask,
            self_attn_mask=self_attn_mask,
            cross_attn_kv=prepared_conditioning.block_kv,
        )
        return self.post_dit(x, pre_state)

    def forward_with_query_prefix_and_extra_context_emb(
        self,
        *,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        query_prefix_emb: Optional[torch.Tensor] = None,
        query_prefix_timestep_sigma: Optional[float] = None,
        extra_context_emb: Optional[torch.Tensor] = None,
        extra_context_mask: Optional[torch.Tensor] = None,
        query_token_mask: Optional[torch.Tensor] = None,
        num_train_timesteps: float = 1000.0,
    ) -> torch.Tensor:
        pre_state = self.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        batch_size = int(pre_state["meta"]["batch_size"])
        action_seq_len = int(pre_state["meta"]["seq_len"])
        tokens = pre_state["tokens"]
        prefix_len = 0
        if query_prefix_emb is not None:
            if query_prefix_emb.ndim != 3:
                raise ValueError(
                    f"`query_prefix_emb` must be 3D [B, P, hidden_dim], got shape {tuple(query_prefix_emb.shape)}"
                )
            if int(query_prefix_emb.shape[0]) != batch_size:
                raise ValueError(
                    f"`query_prefix_emb` batch mismatch: expected {batch_size}, got {query_prefix_emb.shape[0]}"
                )
            if int(query_prefix_emb.shape[-1]) != self.hidden_dim:
                raise ValueError(
                    f"`query_prefix_emb` hidden dim must be {self.hidden_dim}, got {query_prefix_emb.shape[-1]}"
                )
            prefix_len = int(query_prefix_emb.shape[1])
            tokens = torch.cat((query_prefix_emb, tokens), dim=1)
        query_seq_len = int(tokens.shape[1])
        if query_seq_len > self.freqs.shape[0]:
            raise ValueError(
                f"Combined query token length {query_seq_len} exceeds RoPE cache {self.freqs.shape[0]}."
            )
        self_attn_mask = None
        if query_token_mask is not None:
            if query_token_mask.ndim != 2 or tuple(query_token_mask.shape) != (
                batch_size,
                query_seq_len,
            ):
                raise ValueError(
                    f"`query_token_mask` must have shape [B, query_seq_len], got {tuple(query_token_mask.shape)} expected={(batch_size, query_seq_len)}."
                )
            query_token_mask = query_token_mask.to(device=tokens.device, dtype=torch.bool)
            if bool((~query_token_mask).all(dim=1).any().item()):
                raise ValueError(
                    "`query_token_mask` must leave at least one valid key token per sample."
                )
            self_attn_mask = query_token_mask[:, None, None, :].expand(
                batch_size, 1, query_seq_len, query_seq_len
            )
        if prefix_len > 0:
            if query_prefix_timestep_sigma is None:
                raise ValueError(
                    "`query_prefix_timestep_sigma` is required when `query_prefix_emb` is provided."
                )
            if float(num_train_timesteps) <= 0:
                raise ValueError(
                    f"`num_train_timesteps` must be positive, got {num_train_timesteps}"
                )
            prefix_timestep = float(query_prefix_timestep_sigma) * float(num_train_timesteps)
            prefix_timestep_tensor = torch.full(
                (batch_size,), prefix_timestep, device=timestep.device, dtype=timestep.dtype
            )
            (_, prefix_t_mod) = self._build_time_mod(prefix_timestep_tensor)
            action_t_mod = pre_state["t_mod"]
            t_mod = torch.cat(
                (
                    prefix_t_mod.unsqueeze(1).expand(-1, prefix_len, -1, -1),
                    action_t_mod.unsqueeze(1).expand(-1, action_seq_len, -1, -1),
                ),
                dim=1,
            )
        else:
            t_mod = pre_state["t_mod"]
        cross_context = pre_state["context"]
        cross_context_token_mask = pre_state["context_token_mask"]
        if extra_context_emb is not None:
            if extra_context_emb.ndim != 3:
                raise ValueError(
                    f"`extra_context_emb` must be 3D [B, S, hidden_dim], got shape {tuple(extra_context_emb.shape)}"
                )
            if int(extra_context_emb.shape[0]) != batch_size:
                raise ValueError(
                    f"`extra_context_emb` batch mismatch: expected {batch_size}, got {extra_context_emb.shape[0]}"
                )
            if int(extra_context_emb.shape[-1]) != self.hidden_dim:
                raise ValueError(
                    f"`extra_context_emb` hidden dim must be {self.hidden_dim}, got {extra_context_emb.shape[-1]}"
                )
            if extra_context_mask is not None:
                if extra_context_mask.ndim != 2:
                    raise ValueError(
                        f"`extra_context_mask` must be 2D [B, S], got shape {tuple(extra_context_mask.shape)}"
                    )
                if tuple(extra_context_mask.shape) != (batch_size, extra_context_emb.shape[1]):
                    raise ValueError(
                        f"`extra_context_mask` shape must match `extra_context_emb` [B, S], got {tuple(extra_context_mask.shape)} vs {(batch_size, extra_context_emb.shape[1])}"
                    )
                extra_context_mask = extra_context_mask.to(
                    device=extra_context_emb.device, dtype=torch.bool
                )
            if cross_context_token_mask is None and extra_context_mask is not None:
                cross_context_token_mask = torch.ones(
                    (batch_size, cross_context.shape[1]),
                    dtype=torch.bool,
                    device=cross_context.device,
                )
            elif cross_context_token_mask is not None and extra_context_mask is None:
                extra_context_mask = torch.ones(
                    (batch_size, extra_context_emb.shape[1]),
                    dtype=torch.bool,
                    device=extra_context_emb.device,
                )
            cross_context = torch.cat((cross_context, extra_context_emb), dim=1)
            if cross_context_token_mask is not None and extra_context_mask is not None:
                cross_context_token_mask = torch.cat(
                    (
                        cross_context_token_mask,
                        extra_context_mask.to(device=cross_context_token_mask.device),
                    ),
                    dim=1,
                )
        context_attn_mask = (
            None
            if cross_context_token_mask is None
            else cross_context_token_mask.unsqueeze(1).expand(-1, query_seq_len, -1)
        )
        freqs = self.freqs[:query_seq_len].view(query_seq_len, 1, -1).to(tokens.device)
        block_kwargs = {
            "tokens": tokens,
            "context": cross_context,
            "t_mod": t_mod,
            "freqs": freqs,
            "context_mask": context_attn_mask,
        }
        if self_attn_mask is not None:
            block_kwargs["self_attn_mask"] = self_attn_mask
        x = self._run_blocks(**block_kwargs)
        return self.post_dit(x, pre_state)
