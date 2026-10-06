"""Latent-query readout from frozen Stage 1 student tokens."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

READOUT_INPUT_PROJECTION_LINEAR = "linear"
READOUT_INPUT_PROJECTION_MLP2 = "mlp2"
READOUT_INPUT_PROJECTION_TYPES = frozenset(
    (READOUT_INPUT_PROJECTION_LINEAR, READOUT_INPUT_PROJECTION_MLP2)
)
DEFAULT_READOUT_INPUT_PROJECTION_MLP_RATIO = 2.0


def normalize_readout_input_projection_type(value: object) -> str:
    if value is None:
        return READOUT_INPUT_PROJECTION_LINEAR
    projection_type = str(value).strip().lower()
    if projection_type not in READOUT_INPUT_PROJECTION_TYPES:
        raise ValueError(
            f"readout input projection type must be one of {sorted(READOUT_INPUT_PROJECTION_TYPES)}, got {value!r}."
        )
    return projection_type


def normalize_readout_input_projection_mlp_ratio(value: object) -> float:
    if value is None:
        return float(DEFAULT_READOUT_INPUT_PROJECTION_MLP_RATIO)
    ratio = float(value)
    if ratio <= 0.0:
        raise ValueError(f"readout input projection MLP ratio must be positive, got {value!r}.")
    return ratio


def readout_input_projection_type_from_metadata(readout: dict[str, object]) -> str:
    return normalize_readout_input_projection_type(readout.get("input_projection_type"))


def readout_input_projection_mlp_ratio_from_metadata(readout: dict[str, object]) -> float:
    return normalize_readout_input_projection_mlp_ratio(readout.get("input_projection_mlp_ratio"))


def readout_input_projection_metadata(
    *, input_projection_type: object, input_projection_mlp_ratio: object
) -> dict[str, object]:
    return {
        "input_projection_type": normalize_readout_input_projection_type(input_projection_type),
        "input_projection_mlp_ratio": normalize_readout_input_projection_mlp_ratio(
            input_projection_mlp_ratio
        ),
    }


def _build_input_projection(
    *,
    student_dim: int,
    readout_dim: int,
    input_projection_type: object,
    input_projection_mlp_ratio: object,
) -> nn.Module:
    projection_type = normalize_readout_input_projection_type(input_projection_type)
    mlp_ratio = normalize_readout_input_projection_mlp_ratio(input_projection_mlp_ratio)
    if projection_type == READOUT_INPUT_PROJECTION_LINEAR:
        return nn.Linear(student_dim, readout_dim)
    if projection_type == READOUT_INPUT_PROJECTION_MLP2:
        hidden_dim = int(readout_dim * mlp_ratio)
        if hidden_dim <= 0:
            raise ValueError(
                f"readout input projection MLP hidden dim must be positive, got readout_dim={readout_dim} mlp_ratio={mlp_ratio}."
            )
        return nn.Sequential(
            nn.LayerNorm(student_dim),
            nn.Linear(student_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, readout_dim),
        )
    raise AssertionError(f"Unhandled readout input projection type: {projection_type!r}")


class _ReadoutBlock(nn.Module):
    def __init__(self, *, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"Readout block dim must be positive, got {dim}.")
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(f"num_heads must divide dim, got num_heads={num_heads} dim={dim}.")
        if mlp_ratio <= 0.0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        mlp_hidden = int(dim * mlp_ratio)
        self.query_self_norm = nn.LayerNorm(dim)
        self.query_self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=float(dropout), batch_first=True
        )
        self.query_cross_norm = nn.LayerNorm(dim)
        self.student_cross_norm = nn.LayerNorm(dim)
        self.query_cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=float(dropout), batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self, queries: torch.Tensor, student_tokens: torch.Tensor, student_mask: torch.Tensor | None
    ) -> torch.Tensor:
        q = self.query_self_norm(queries)
        (self_attn, _) = self.query_self_attn(q, q, q, need_weights=False)
        queries = queries + self_attn
        q = self.query_cross_norm(queries)
        kv = self.student_cross_norm(student_tokens)
        key_padding_mask = None if student_mask is None else ~student_mask.bool()
        (cross_attn, _) = self.query_cross_attn(
            q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False
        )
        queries = queries + cross_attn
        queries = queries + self.mlp(self.mlp_norm(queries))
        return queries


class StudentLatentQueryReadout(nn.Module):
    """Map current-only student tokens to action-facing readout tokens."""

    def __init__(
        self,
        *,
        student_dim: int,
        readout_dim: int,
        output_dim: int,
        num_queries: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float = 0.0,
        input_projection_type: str = READOUT_INPUT_PROJECTION_LINEAR,
        input_projection_mlp_ratio: float = DEFAULT_READOUT_INPUT_PROJECTION_MLP_RATIO,
    ) -> None:
        super().__init__()
        self.student_dim = int(student_dim)
        self.readout_dim = int(readout_dim)
        self.output_dim = int(output_dim)
        self.num_queries = int(num_queries)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout = float(dropout)
        self.input_projection_type = normalize_readout_input_projection_type(input_projection_type)
        self.input_projection_mlp_ratio = normalize_readout_input_projection_mlp_ratio(
            input_projection_mlp_ratio
        )
        if self.student_dim <= 0 or self.readout_dim <= 0 or self.output_dim <= 0:
            raise ValueError(
                f"student_dim, readout_dim, and output_dim must be positive, got {self.student_dim}, {self.readout_dim}, {self.output_dim}."
            )
        if self.num_queries <= 0:
            raise ValueError(f"num_queries must be positive, got {self.num_queries}.")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}.")
        if self.num_heads <= 0 or self.readout_dim % self.num_heads != 0:
            raise ValueError(
                f"num_heads must divide readout_dim, got num_heads={self.num_heads} readout_dim={self.readout_dim}."
            )
        if self.mlp_ratio <= 0.0:
            raise ValueError(f"mlp_ratio must be positive, got {self.mlp_ratio}.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}.")
        self.query = nn.Parameter(torch.empty(self.num_queries, self.readout_dim))
        self.input_projection = _build_input_projection(
            student_dim=self.student_dim,
            readout_dim=self.readout_dim,
            input_projection_type=self.input_projection_type,
            input_projection_mlp_ratio=self.input_projection_mlp_ratio,
        )
        self.blocks = nn.ModuleList(
            [
                _ReadoutBlock(
                    dim=self.readout_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(self.readout_dim)
        self.output_projection = (
            nn.Identity()
            if self.readout_dim == self.output_dim
            else nn.Linear(self.readout_dim, self.output_dim)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query, std=1.0 / math.sqrt(max(self.readout_dim, 1)))

    def forward(
        self,
        student_tokens: torch.Tensor,
        student_mask: torch.Tensor | None = None,
        *,
        projected_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(student_tokens, torch.Tensor) or student_tokens.ndim != 3:
            raise ValueError(
                f"`student_tokens` must have shape [B,N,D_student], got {tuple(getattr(student_tokens, 'shape', ()))}."
            )
        (batch_size, num_tokens, hidden_dim) = student_tokens.shape
        if int(num_tokens) <= 0:
            raise ValueError("Student readout received zero student tokens.")
        if int(hidden_dim) != self.student_dim:
            raise ValueError(
                f"Student token hidden dim mismatch: actual={hidden_dim} expected={self.student_dim}."
            )
        attention_student_mask: torch.Tensor | None = None
        if student_mask is not None:
            if not isinstance(student_mask, torch.Tensor) or tuple(student_mask.shape) != (
                batch_size,
                num_tokens,
            ):
                raise ValueError(
                    f"`student_mask` must have shape {(batch_size, num_tokens)}, got {tuple(getattr(student_mask, 'shape', ()))}."
                )
            student_mask = student_mask.to(device=student_tokens.device, dtype=torch.bool)
            valid_counts = student_mask.sum(dim=1)
            if bool((valid_counts <= 0).any().item()):
                raise ValueError("Student readout received a sample with no valid student tokens.")
            if not bool((valid_counts == num_tokens).all().item()):
                attention_student_mask = student_mask
        student_context = self.input_projection(student_tokens)
        if projected_positions is not None:
            if projected_positions.shape != (num_tokens, self.readout_dim):
                raise ValueError("Projected positions must match [spatial tokens, readout width].")
            if not torch.isfinite(projected_positions).all():
                raise ValueError("Projected positions must be finite.")
            student_context = student_context + projected_positions.to(student_context)
        queries = self.query.to(device=student_tokens.device, dtype=student_context.dtype)
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)
        for block in self.blocks:
            queries = block(queries, student_context, attention_student_mask)
        readout_tokens = self.output_projection(self.output_norm(queries))
        readout_mask = torch.ones(
            (batch_size, self.num_queries), dtype=torch.bool, device=readout_tokens.device
        )
        expected_shape = (batch_size, self.num_queries, self.output_dim)
        if tuple(readout_tokens.shape) != expected_shape:
            raise ValueError(
                f"Student readout output shape mismatch: actual={tuple(readout_tokens.shape)} expected={expected_shape}."
            )
        return (readout_tokens, readout_mask)
