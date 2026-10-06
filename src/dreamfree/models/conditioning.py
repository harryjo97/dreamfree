from __future__ import annotations

import torch
from torch import nn


class TeacherStateActionConditioner(nn.Module):
    def __init__(
        self,
        *,
        teacher_state_input_dim: int,
        action_context_dim: int,
        proprio_dim: int,
        proprio_obs_dropout: float,
    ):
        super().__init__()
        self.teacher_state_input_dim = int(teacher_state_input_dim)
        self.action_context_dim = int(action_context_dim)
        self.proprio_dim = int(proprio_dim)
        self.proprio_obs_dropout = float(proprio_obs_dropout)
        self.teacher_projection = nn.Linear(self.teacher_state_input_dim, self.action_context_dim)
        self.teacher_context_norm = nn.LayerNorm(self.action_context_dim, eps=1e-06)
        self.proprio_projection = nn.Linear(self.proprio_dim, self.action_context_dim)
        self.proprio_mask_token = nn.Parameter(torch.zeros(1, 1, self.action_context_dim))
        nn.init.normal_(self.proprio_mask_token, std=1.0 / max(self.action_context_dim, 1) ** 0.5)

    def forward(
        self,
        y_tgt_cond: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor,
        *,
        prepared_teacher_context_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if y_tgt_cond.ndim != 3 or int(y_tgt_cond.shape[-1]) != self.teacher_state_input_dim:
            raise ValueError(
                f"`y_tgt_cond` must have shape [B,S,{self.teacher_state_input_dim}], got {tuple(y_tgt_cond.shape)}."
            )
        if context.ndim != 3 or tuple(context.shape[1:]) != (128, 4096):
            raise ValueError(f"`context` must have shape [B,128,4096], got {tuple(context.shape)}.")
        if context_mask.ndim != 2 or tuple(context_mask.shape[1:]) != (128,):
            raise ValueError(
                f"`context_mask` must have shape [B,128], got {tuple(context_mask.shape)}."
            )
        if proprio.ndim != 3 or int(proprio.shape[-1]) != self.proprio_dim:
            raise ValueError(
                f"`proprio` must have shape [B,T,{self.proprio_dim}], got {tuple(proprio.shape)}."
            )
        batch_size = int(y_tgt_cond.shape[0])
        if (
            int(context.shape[0]) != batch_size
            or int(context_mask.shape[0]) != batch_size
            or int(proprio.shape[0]) != batch_size
        ):
            raise ValueError(
                f"Conditioner batch mismatch: teacher={batch_size}, context={context.shape[0]}, context_mask={context_mask.shape[0]}, proprio={proprio.shape[0]}."
            )
        teacher_context_emb = self.teacher_context_norm(self.teacher_projection(y_tgt_cond))
        if prepared_teacher_context_mask is None:
            teacher_context_mask = torch.ones(
                (batch_size, teacher_context_emb.shape[1]),
                dtype=torch.bool,
                device=teacher_context_emb.device,
            )
        else:
            teacher_context_mask = prepared_teacher_context_mask
            if (
                self.training
                or teacher_context_mask.dtype != torch.bool
                or teacher_context_mask.device != teacher_context_emb.device
                or (tuple(teacher_context_mask.shape) != tuple(teacher_context_emb.shape[:2]))
            ):
                raise ValueError(
                    "Prepared teacher mask requires eval mode and matching boolean shape/device."
                )
        proprio_query_emb = self.proprio_projection(proprio[:, 0:1, :])
        if self.training and self.proprio_obs_dropout > 0.0:
            dropout_mask = torch.bernoulli(
                torch.full(
                    (batch_size, 1, 1),
                    self.proprio_obs_dropout,
                    device=proprio_query_emb.device,
                    dtype=proprio_query_emb.dtype,
                )
            )
            proprio_query_emb = (
                1.0 - dropout_mask
            ) * proprio_query_emb + dropout_mask * self.proprio_mask_token.to(
                device=proprio_query_emb.device, dtype=proprio_query_emb.dtype
            )
        return {
            "teacher_context_emb": teacher_context_emb,
            "teacher_context_mask": teacher_context_mask,
            "proprio_query_emb": proprio_query_emb,
        }

    def teacher_state_state_dict(self) -> dict[str, torch.Tensor]:
        state = self.state_dict()
        return {
            key: value.detach().cpu().contiguous()
            for (key, value) in state.items()
            if key.startswith("teacher_projection.") or key.startswith("teacher_context_norm.")
        }

    def proprio_state_dict(self) -> dict[str, torch.Tensor]:
        state = self.state_dict()
        return {
            key: value.detach().cpu().contiguous()
            for (key, value) in state.items()
            if key.startswith("proprio_projection.") or key == "proprio_mask_token"
        }

    def load_teacher_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> None:
        current = self.state_dict()
        merged = dict(current)
        teacher_keys = {
            key
            for key in current
            if key.startswith("teacher_projection.") or key.startswith("teacher_context_norm.")
        }
        missing = sorted(teacher_keys - set(state_dict))
        unexpected = sorted(set(state_dict) - teacher_keys)
        if strict and (missing or unexpected):
            raise ValueError(
                f"Teacher conditioner state key mismatch: missing={missing}, unexpected={unexpected}"
            )
        merged.update({key: value for (key, value) in state_dict.items() if key in teacher_keys})
        self.load_state_dict(merged, strict=True)

    def load_proprio_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> None:
        current = self.state_dict()
        merged = dict(current)
        proprio_keys = {
            key
            for key in current
            if key.startswith("proprio_projection.") or key == "proprio_mask_token"
        }
        missing = sorted(proprio_keys - set(state_dict))
        unexpected = sorted(set(state_dict) - proprio_keys)
        if strict and (missing or unexpected):
            raise ValueError(
                f"Proprio conditioner state key mismatch: missing={missing}, unexpected={unexpected}"
            )
        merged.update({key: value for (key, value) in state_dict.items() if key in proprio_keys})
        self.load_state_dict(merged, strict=True)
