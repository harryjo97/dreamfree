"""Future-state hidden-token prediction helpers for Stage 1 training."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamfree.models.lora import (
    is_lora_parameter_name,
    lora_active_adapters,
    lora_disabled,
    restore_lora_enabled,
    set_lora_enabled,
)
from dreamfree.models.wan22.i2v_condition import concat_14b_i2v_condition
from dreamfree.models.wan22.wan_video_dit import (
    TEXT_CONDITION_SCOPE_ALL_FRAMES,
    TEXT_CONDITION_SCOPE_FIRST_FRAME,
    TEXT_CONDITION_SCOPES,
)

TEACHER_STATE_LAYERNORM_TYPE = "layernorm"
TEACHER_STATE_LAYERNORM_EPS = 1e-06
TEACHER_STATE_LOSS_AFFINE = False
I2V_CONDITION_NONE = "none"
I2V_CONDITION_FIRST_IMAGE_MASK = "first_image_mask_condition"
FUTURE_TOKEN_ATTENTION_FULL = "full"
FUTURE_TOKEN_ATTENTION_ITSELF = "itself"
FUTURE_TOKEN_ATTENTION_MODES = frozenset(
    (FUTURE_TOKEN_ATTENTION_FULL, FUTURE_TOKEN_ATTENTION_ITSELF)
)
FUTURE_TOKEN_ATTENTION_ITSELF_MASK_MODE = "first_frame_self_frame"
TEACHER_FRAME0_CLEAN_FIRST_FRAME = "clean_first_frame"
TEACHER_FRAME0_NATIVE_NOISED = "native_noised"
TEACHER_FRAME0_MODES = frozenset((TEACHER_FRAME0_CLEAN_FIRST_FRAME, TEACHER_FRAME0_NATIVE_NOISED))


def normalize_teacher_frame0_mode(value: Any) -> str:
    """Normalize teacher frame-0 input handling mode."""
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in TEACHER_FRAME0_MODES:
        raise ValueError(
            f"Unsupported teacher_frame0_mode={value!r}; expected one of {sorted(TEACHER_FRAME0_MODES)}."
        )
    return normalized


@contextmanager
def _lora_enabled(module: nn.Module, adapter_name: str | None = None):
    """Temporarily enable LoRA adapters for a module.

    Args:
        module: Module whose LoRA enable state should be forced on.
        adapter_name: Optional adapter name. When set, only that adapter is active.

    Yields:
        Control to the caller while LoRA adapters are enabled.
    """
    if adapter_name is not None:
        with lora_active_adapters(module, adapter_name):
            yield
        return
    state = set_lora_enabled(module, True)
    try:
        yield
    finally:
        restore_lora_enabled(module, state)


@contextmanager
def temporary_video_attention_mask_mode(module: nn.Module, mode: str):
    """Temporarily switch a Wan DiT video attention mask mode.

    Args:
        module: Wan DiT module exposing `video_attention_mask_mode`.
        mode: Mask mode to use inside the context.

    Yields:
        Control to the caller with `module.video_attention_mask_mode` set to `mode`.

    Raises:
        AttributeError: If `module` does not expose the mask-mode attribute.
    """
    if not hasattr(module, "video_attention_mask_mode"):
        raise AttributeError(
            f"{type(module).__name__} does not expose `video_attention_mask_mode`."
        )
    previous = getattr(module, "video_attention_mask_mode")
    setattr(module, "video_attention_mask_mode", str(mode))
    try:
        yield
    finally:
        setattr(module, "video_attention_mask_mode", previous)


def assert_non_lora_parameters_frozen(module: nn.Module) -> None:
    """Validate that only LoRA adapter weights are trainable.

    Args:
        module: Module whose parameters are checked.

    Raises:
        ValueError: If any trainable parameter is not a LoRA A/B matrix.
    """
    offenders = [
        name
        for (name, param) in module.named_parameters()
        if param.requires_grad and (not is_lora_parameter_name(name))
    ]
    if offenders:
        preview = offenders[:10]
        suffix = "..." if len(offenders) > 10 else ""
        raise ValueError(f"Non-LoRA DiT parameters require gradients: {preview}{suffix}")


def fixed_tokenwise_layer_norm(
    x: torch.Tensor, *, eps: float = TEACHER_STATE_LAYERNORM_EPS
) -> torch.Tensor:
    """Apply non-affine token-wise LayerNorm over the hidden dimension.

    Args:
        x: Hidden-state tensor with shape `(B, S, D)`.
        eps: Numerical stability constant used by LayerNorm.

    Returns:
        Tensor with the same shape as `x`, normalized over the last dimension.

    Raises:
        ValueError: If `x` is not shaped like a hidden-token sequence.
    """
    if x.ndim != 3:
        raise ValueError(f"`x` must be 3D [B,S,D], got shape {tuple(x.shape)}.")
    return F.layer_norm(x, (int(x.shape[-1]),), weight=None, bias=None, eps=float(eps))


class FutureStateTeacher:
    """Extract frozen teacher hidden tokens from moderately noised future latents."""

    def __init__(
        self,
        core: Any,
        *,
        layer_idx: int = 19,
        hook_point: str = "post_block",
        target_type: str = "hidden",
        sigma_policy: str = "uniform",
        sigma_min: float = 0.6,
        sigma_max: float = 0.8,
        video_attention_mask_mode: str = "bidirectional",
        i2v_condition_mode: str = I2V_CONDITION_NONE,
        frame0_mode: str = TEACHER_FRAME0_CLEAN_FIRST_FRAME,
        lora_adapter_name: str | None = None,
    ):
        """Configure frozen teacher target extraction.

        Args:
            core: Training core exposing `dit` and `train_scheduler`.
            layer_idx: Zero-based DiT block index used for hidden-token targets.
            hook_point: Hidden-token hook point. Stage 1 v1 expects `post_block`.
            target_type: Teacher target kind. Stage 1 v1 expects `hidden`.
            sigma_policy: Noise sampling policy. Stage 1 v1 expects `uniform`.
            sigma_min: Lower inclusive bound for per-sample teacher noise.
            sigma_max: Upper inclusive bound for per-sample teacher noise.
            video_attention_mask_mode: Teacher mask mode. Stage 1 v1 expects `bidirectional`.
            i2v_condition_mode: Optional I2V latent condition mode. Wan2.2-I2V-A14B uses
                `first_image_mask_condition`; Wan2.2-TI2V-5B uses `none`.
            frame0_mode: How the teacher treats frame 0 in the main latent stream.
                `clean_first_frame` preserves the legacy 5B contract; `native_noised`
                keeps 14B Stage 0's native noised-frame-0 contract.
            lora_adapter_name: Optional teacher LoRA adapter to activate. When
                omitted, all LoRA adapters are disabled for legacy merged-base mode.

        Raises:
            ValueError: If a Stage 1 v1 configuration invariant is violated.
        """
        if str(target_type).strip().lower() != "hidden":
            raise ValueError(
                f"Stage 1 v1 only supports teacher target_type='hidden', got {target_type!r}."
            )
        if str(hook_point) != "post_block":
            raise ValueError(
                f"Stage 1 v1 only supports teacher hook_point='post_block', got {hook_point!r}."
            )
        if str(sigma_policy).strip().lower() != "uniform":
            raise ValueError(
                f"Stage 1 v1 only supports teacher sigma_policy='uniform', got {sigma_policy!r}."
            )
        if str(video_attention_mask_mode) != "bidirectional":
            raise ValueError(
                f"Stage 1 requires teacher_video_attention_mask_mode='bidirectional', got {video_attention_mask_mode!r}."
            )
        sigma_min = float(sigma_min)
        sigma_max = float(sigma_max)
        if sigma_min < 0.0 or sigma_max > 1.0 or sigma_min > sigma_max:
            raise ValueError(
                f"Teacher sigma bounds must satisfy 0 <= sigma_min <= sigma_max <= 1, got sigma_min={sigma_min}, sigma_max={sigma_max}."
            )
        if str(i2v_condition_mode) not in {I2V_CONDITION_NONE, I2V_CONDITION_FIRST_IMAGE_MASK}:
            raise ValueError(
                f"Unsupported teacher i2v_condition_mode={i2v_condition_mode!r}; expected 'none' or 'first_image_mask_condition'."
            )
        self.core = core
        self.layer_idx = int(layer_idx)
        self.hook_point = str(hook_point)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.video_attention_mask_mode = str(video_attention_mask_mode)
        self.i2v_condition_mode = str(i2v_condition_mode)
        self.frame0_mode = normalize_teacher_frame0_mode(frame0_mode)
        self.lora_adapter_name = None if lora_adapter_name is None else str(lora_adapter_name)

    def sample_sigma(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Sample teacher noise levels for each batch item.

        Args:
            batch_size: Number of samples in the batch.
            device: Device for the returned tensor.
            dtype: Final dtype for the returned tensor.

        Returns:
            Tensor of shape `(B,)` with values in `[sigma_min, sigma_max]`.

        Raises:
            ValueError: If `batch_size` is not positive.
        """
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}.")
        sigma = torch.empty((batch_size,), device=device, dtype=torch.float32).uniform_(
            self.sigma_min, self.sigma_max
        )
        return sigma.to(dtype=dtype)

    def __call__(
        self,
        *,
        z_clean: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
        i2v_condition_latents: torch.Tensor | None = None,
        sample_id: Any | None = None,
        eps: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Extract frozen full-sequence teacher hidden tokens.

        Args:
            z_clean: Clean video latents with shape `(B, C, T, H, W)`.
            context: Conditioning tokens, typically shaped `(B, L, D_ctx)`.
            context_mask: Optional boolean conditioning mask, typically shaped `(B, L)`.
            i2v_condition_latents: Official 14B I2V condition latents with
                shape `(B, 16, T, H, W)`, produced by VAE-encoding
                `[first_image, zeros, zeros, ...]`.
            sample_id: Optional batch identifier accepted for interface compatibility.
            eps: Optional explicit Gaussian noise with exactly the same shape, device,
                and dtype as `z_clean`. When omitted, noise is sampled normally.
            sigma: Optional explicit noise levels with shape `(B,)` on the same device
                and in the same dtype as `z_clean`. Values must be finite and within
                the teacher's configured sigma bounds.

        Returns:
            Dictionary with detached `tokens` of shape `(B, S, D_teacher)`,
            teacher `meta`, sampled `sigma` of shape `(B,)`, and `timestep` of shape `(B,)`.

        Raises:
            ValueError: If latent rank, explicit-noise, explicit-sigma, or
                trainable-parameter invariants are violated.
            RuntimeError: If the teacher mask mode is not active during extraction.
        """
        del sample_id
        if z_clean.ndim != 5:
            raise ValueError(f"`z_clean` must be 5D [B,C,T,H,W], got shape {tuple(z_clean.shape)}.")
        assert_non_lora_parameters_frozen(self.core.dit)
        batch_size = int(z_clean.shape[0])
        first_frame_latents = z_clean[:, :, 0:1].detach()
        if eps is None:
            eps = torch.randn_like(z_clean)
        else:
            if tuple(eps.shape) != tuple(z_clean.shape):
                raise ValueError(
                    f"Explicit `eps` shape must match `z_clean`: {tuple(eps.shape)} != {tuple(z_clean.shape)}."
                )
            if eps.device != z_clean.device:
                raise ValueError(
                    f"Explicit `eps` device must match `z_clean`: {eps.device} != {z_clean.device}."
                )
            if eps.dtype != z_clean.dtype:
                raise ValueError(
                    f"Explicit `eps` dtype must match `z_clean`: {eps.dtype} != {z_clean.dtype}."
                )
            if not bool(torch.isfinite(eps).all().item()):
                raise ValueError("Explicit `eps` must contain only finite values.")
            eps = eps.detach()
        if sigma is None:
            sigma = self.sample_sigma(batch_size, device=z_clean.device, dtype=z_clean.dtype)
        else:
            if tuple(sigma.shape) != (batch_size,):
                raise ValueError(
                    f"Explicit `sigma` must have shape ({batch_size},), got {tuple(sigma.shape)}."
                )
            if sigma.device != z_clean.device:
                raise ValueError(
                    f"Explicit `sigma` device must match `z_clean`: {sigma.device} != {z_clean.device}."
                )
            if sigma.dtype != z_clean.dtype:
                raise ValueError(
                    f"Explicit `sigma` dtype must match `z_clean`: {sigma.dtype} != {z_clean.dtype}."
                )
            if not bool(torch.isfinite(sigma).all().item()):
                raise ValueError("Explicit `sigma` must contain only finite values.")
            sigma_min = float(sigma.min().item())
            sigma_max = float(sigma.max().item())
            configured_min = torch.tensor(self.sigma_min, device=sigma.device, dtype=sigma.dtype)
            configured_max = torch.tensor(self.sigma_max, device=sigma.device, dtype=sigma.dtype)
            if bool((sigma < configured_min).any().item()) or bool(
                (sigma > configured_max).any().item()
            ):
                raise ValueError(
                    f"Explicit `sigma` values must lie within the configured teacher bounds [{self.sigma_min}, {self.sigma_max}], got [{sigma_min}, {sigma_max}]."
                )
            sigma = sigma.detach()
        timestep = sigma * float(self.core.train_scheduler.num_train_timesteps)
        z_tau = self.core.train_scheduler.add_noise(z_clean, eps, timestep)
        first_frame_timestep_zero = False
        if self.frame0_mode == TEACHER_FRAME0_CLEAN_FIRST_FRAME:
            z_tau[:, :, 0:1] = first_frame_latents
            first_frame_timestep_zero = True
        dit_input = z_tau
        if self.i2v_condition_mode == I2V_CONDITION_FIRST_IMAGE_MASK:
            if i2v_condition_latents is None:
                raise ValueError(
                    "Teacher i2v_condition_mode='first_image_mask_condition' requires `i2v_condition_latents` from official pseudo-video VAE encoding."
                )
            dit_input = concat_14b_i2v_condition(
                latents=z_tau, i2v_condition_latents=i2v_condition_latents
            )
        lora_context = (
            lora_disabled(self.core.dit)
            if self.lora_adapter_name is None
            else lora_active_adapters(self.core.dit, self.lora_adapter_name)
        )
        with (
            torch.no_grad(),
            lora_context,
            temporary_video_attention_mask_mode(self.core.dit, self.video_attention_mask_mode),
        ):
            active_mode = str(getattr(self.core.dit, "video_attention_mask_mode", ""))
            if active_mode != "bidirectional":
                raise RuntimeError(
                    f"Teacher target extraction requires bidirectional mask mode, got {active_mode!r}."
                )
            out = self.core.dit.forward_hidden_tokens(
                x=dit_input,
                timestep=timestep,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=bool(
                    getattr(self.core.dit, "fuse_vae_embedding_in_latents", False)
                ),
                first_frame_timestep_zero=first_frame_timestep_zero,
                layer_idx=self.layer_idx,
                hook_point=self.hook_point,
            )
        return {
            "tokens": out["tokens"].detach(),
            "meta": out["meta"],
            "sigma": sigma.detach(),
            "timestep": timestep.detach(),
        }


class StudentFirstFramePredictor:
    """Run the student FM path and expose first-frame hidden tokens only."""

    def __init__(
        self,
        core: Any,
        *,
        layer_idx: int = 19,
        hook_point: str = "post_block",
        input_mode: str = "flow_noised_video_first_frame_clean",
        video_attention_mask_mode: str = "first_frame_causal",
        future_token_attention: str = FUTURE_TOKEN_ATTENTION_FULL,
        text_condition_scope: str = TEXT_CONDITION_SCOPE_ALL_FRAMES,
        i2v_condition_mode: str = I2V_CONDITION_NONE,
        lora_adapter_name: str | None = None,
    ):
        """Configure the student first-frame feature extractor.

        Args:
            core: Training core exposing `dit` and `train_scheduler`.
            layer_idx: Zero-based DiT block index used for student hidden tokens.
            hook_point: Hidden-token hook point. Stage 1 v1 expects `post_block`.
            input_mode: Student latent noising mode. Stage 1 v1 expects
                `flow_noised_video_first_frame_clean` for TI2V-5B or
                `wan22_i2v_clean_first_frame` for Wan2.2-I2V-A14B.
            video_attention_mask_mode: Student mask mode. Stage 1 v1 expects `first_frame_causal`.
            future_token_attention: Student future-token attention policy. `full` preserves
                first-frame causal attention; `itself` lets each future frame attend to the
                first frame plus its own frame.
            text_condition_scope: Which video tokens may cross-attend to text.
                `first_frame` bottlenecks language through first-frame tokens.
            i2v_condition_mode: Optional I2V latent condition mode. Wan2.2-I2V-A14B uses
                `first_image_mask_condition`; Wan2.2-TI2V-5B uses `none`.
            lora_adapter_name: Optional student LoRA adapter to activate.

        Raises:
            ValueError: If a Stage 1 v1 configuration invariant is violated.
        """
        if str(hook_point) != "post_block":
            raise ValueError(
                f"Stage 1 v1 only supports student hook_point='post_block', got {hook_point!r}."
            )
        if str(input_mode) not in {
            "flow_noised_video_first_frame_clean",
            "wan22_i2v_clean_first_frame",
        }:
            raise ValueError(
                f"Stage 1 v1 only supports student_input_mode='flow_noised_video_first_frame_clean' or 'wan22_i2v_clean_first_frame', got {input_mode!r}."
            )
        if str(video_attention_mask_mode) != "first_frame_causal":
            raise ValueError(
                f"Stage 1 requires student_video_attention_mask_mode='first_frame_causal', got {video_attention_mask_mode!r}."
            )
        future_token_attention = str(future_token_attention).strip().lower()
        if future_token_attention not in FUTURE_TOKEN_ATTENTION_MODES:
            raise ValueError(
                f"Unsupported student future_token_attention={future_token_attention!r}; expected one of {sorted(FUTURE_TOKEN_ATTENTION_MODES)}."
            )
        text_condition_scope = str(text_condition_scope).strip().lower()
        if text_condition_scope not in TEXT_CONDITION_SCOPES:
            raise ValueError(
                f"Unsupported student text_condition_scope={text_condition_scope!r}; expected one of {sorted(TEXT_CONDITION_SCOPES)}."
            )
        if (
            text_condition_scope == TEXT_CONDITION_SCOPE_FIRST_FRAME
            and str(video_attention_mask_mode) != "first_frame_causal"
        ):
            raise ValueError(
                "student text_condition_scope='first_frame' requires first_frame_causal video attention."
            )
        if str(i2v_condition_mode) not in {I2V_CONDITION_NONE, I2V_CONDITION_FIRST_IMAGE_MASK}:
            raise ValueError(
                f"Unsupported student i2v_condition_mode={i2v_condition_mode!r}; expected 'none' or 'first_image_mask_condition'."
            )
        if (
            str(input_mode) == "wan22_i2v_clean_first_frame"
            and str(i2v_condition_mode) != I2V_CONDITION_FIRST_IMAGE_MASK
        ):
            raise ValueError(
                "wan22_i2v_clean_first_frame requires i2v_condition_mode='first_image_mask_condition'."
            )
        self.core = core
        self.layer_idx = int(layer_idx)
        self.hook_point = str(hook_point)
        self.input_mode = str(input_mode)
        self.video_attention_mask_mode = str(video_attention_mask_mode)
        self.future_token_attention = future_token_attention
        self.active_video_attention_mask_mode = (
            self.video_attention_mask_mode
            if future_token_attention == FUTURE_TOKEN_ATTENTION_FULL
            else FUTURE_TOKEN_ATTENTION_ITSELF_MASK_MODE
        )
        self.text_condition_scope = text_condition_scope
        self.i2v_condition_mode = str(i2v_condition_mode)
        self.lora_adapter_name = None if lora_adapter_name is None else str(lora_adapter_name)

    def __call__(
        self,
        *,
        z_clean: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
        eps_fm: torch.Tensor,
        timestep_fm: torch.Tensor,
        first_frame_latents: torch.Tensor | None,
        fuse_vae_embedding_in_latents: bool,
        i2v_condition_latents: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Run the student FM path and return first-frame world tokens.

        Args:
            z_clean: Clean video latents with shape `(B, C, T, H, W)`.
            context: Conditioning tokens, typically shaped `(B, L, D_ctx)`.
            context_mask: Optional boolean conditioning mask, typically shaped `(B, L)`.
            eps_fm: Flow-matching noise with the same shape as `z_clean`.
            timestep_fm: Per-sample flow-matching timesteps with shape `(B,)`.
            first_frame_latents: Optional clean first-frame latents with shape `(B, C, 1, H, W)`.
            i2v_condition_latents: Official 14B I2V condition latents with
                shape `(B, 16, T, H, W)`, produced by VAE-encoding
                `[first_image, zeros, zeros, ...]`.
            fuse_vae_embedding_in_latents: Whether the DiT should fuse VAE embeddings in latent space.

        Returns:
            Dictionary with DiT `pred`, all hidden `tokens` of shape `(B, S, D_student)`,
            `z_world` first-frame tokens of shape `(B, tokens_per_frame, D_student)`, and `meta`.

        Raises:
            ValueError: If input tensor ranks or shape contracts are violated.
            RuntimeError: If the student mask mode is not active during the forward pass.
        """
        if z_clean.ndim != 5:
            raise ValueError(f"`z_clean` must be 5D [B,C,T,H,W], got shape {tuple(z_clean.shape)}.")
        if tuple(eps_fm.shape) != tuple(z_clean.shape):
            raise ValueError(
                f"`eps_fm` shape must match `z_clean`, got {tuple(eps_fm.shape)} vs {tuple(z_clean.shape)}."
            )
        if timestep_fm.ndim != 1 or timestep_fm.shape[0] != z_clean.shape[0]:
            raise ValueError(f"`timestep_fm` must have shape [B], got {tuple(timestep_fm.shape)}.")
        z_fm = self.core.train_scheduler.add_noise(z_clean, eps_fm, timestep_fm)
        clean_first = z_clean[:, :, 0:1] if first_frame_latents is None else first_frame_latents
        z_fm[:, :, 0:1] = clean_first
        dit_input = z_fm
        if self.i2v_condition_mode == I2V_CONDITION_FIRST_IMAGE_MASK:
            if i2v_condition_latents is None:
                raise ValueError(
                    "Student i2v_condition_mode='first_image_mask_condition' requires `i2v_condition_latents` from official pseudo-video VAE encoding."
                )
            dit_input = concat_14b_i2v_condition(
                latents=z_fm, i2v_condition_latents=i2v_condition_latents
            )
        with (
            _lora_enabled(self.core.dit, self.lora_adapter_name),
            temporary_video_attention_mask_mode(
                self.core.dit, self.active_video_attention_mask_mode
            ),
        ):
            active_mode = str(getattr(self.core.dit, "video_attention_mask_mode", ""))
            if active_mode != self.active_video_attention_mask_mode:
                raise RuntimeError(
                    f"Student FM forward active mask mode mismatch: actual={active_mode!r} expected={self.active_video_attention_mask_mode!r}."
                )
            out = self.core.dit.forward_with_hidden_tokens(
                x=dit_input,
                timestep=timestep_fm,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=bool(fuse_vae_embedding_in_latents),
                text_condition_scope=self.text_condition_scope,
                layer_idx=self.layer_idx,
                hook_point=self.hook_point,
            )
        tokens_per_frame = int(out["meta"]["tokens_per_frame"])
        if tokens_per_frame <= 0 or out["tokens"].shape[1] < tokens_per_frame:
            raise ValueError(f"Invalid student hidden metadata: {out['meta']}.")
        z_world = out["tokens"][:, :tokens_per_frame, :]
        return {
            "pred": out["pred"],
            "tokens": out["tokens"],
            "z_world": z_world,
            "meta": out["meta"],
        }


class _DecoderBlock(nn.Module):
    """Transformer decoder block for the future-state projector."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        """Create self-attention, cross-attention, and MLP layers.

        Args:
            dim: Projector hidden dimension.
            num_heads: Number of attention heads.
            mlp_ratio: Expansion ratio for the feed-forward hidden layer.
        """
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dim)
        hidden_dim = int(dim * float(mlp_ratio))
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Update decoded tokens using first-frame memory.

        Args:
            query: Decoded sequence tokens with shape `(B, S, projector_dim)`.
            memory: First-frame memory tokens with shape `(B, tokens_per_frame, projector_dim)`.

        Returns:
            Tensor with the same shape as `query`.
        """
        self_tokens = self.self_norm(query)
        query = query + self.self_attn(self_tokens, self_tokens, self_tokens, need_weights=False)[0]
        memory_tokens = self.memory_norm(memory)
        query = (
            query
            + self.cross_attn(
                self.cross_norm(query), memory_tokens, memory_tokens, need_weights=False
            )[0]
        )
        return query + self.mlp(self.mlp_norm(query))


class FirstFrameToTeacherStateProjector(nn.Module):
    """Decode first-frame student tokens into the full teacher hidden sequence."""

    def __init__(
        self,
        d_student: int = 3072,
        d_teacher: int = 3072,
        projector_dim: int = 1024,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        max_frames: int = 16,
    ):
        """Build the decoder that expands first-frame tokens to teacher tokens.

        Args:
            d_student: Hidden dimension of student tokens.
            d_teacher: Hidden dimension of teacher targets.
            projector_dim: Internal decoder hidden dimension.
            num_layers: Number of decoder blocks.
            num_heads: Number of attention heads per decoder block.
            mlp_ratio: Expansion ratio for each decoder block MLP.
            max_frames: Maximum number of latent frames the projector can decode.

        Raises:
            ValueError: If projector dimensions or frame limits are invalid.
        """
        super().__init__()
        self.d_student = int(d_student)
        self.d_teacher = int(d_teacher)
        self.projector_dim = int(projector_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.max_frames = int(max_frames)
        if self.projector_dim <= 0:
            raise ValueError(f"`projector_dim` must be positive, got {self.projector_dim}.")
        if self.num_layers <= 0:
            raise ValueError(f"`num_layers` must be positive, got {self.num_layers}.")
        if self.num_heads <= 0 or self.projector_dim % self.num_heads != 0:
            raise ValueError(
                f"`projector_num_heads` must divide projector_dim, got dim={self.projector_dim}, heads={self.num_heads}."
            )
        if self.max_frames <= 0:
            raise ValueError(f"`max_frames` must be positive, got {self.max_frames}.")
        self.input_norm = nn.LayerNorm(self.d_student)
        self.memory_proj = nn.Linear(self.d_student, self.projector_dim)
        self.frame_embedding = nn.Parameter(torch.zeros(self.max_frames, self.projector_dim))
        self.blocks = nn.ModuleList(
            [
                _DecoderBlock(self.projector_dim, self.num_heads, self.mlp_ratio)
                for _ in range(self.num_layers)
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(self.projector_dim), nn.Linear(self.projector_dim, self.d_teacher)
        )

    def forward(self, z_world: torch.Tensor, *, target_meta: dict[str, Any]) -> torch.Tensor:
        """Project first-frame student tokens into the teacher hidden-token layout.

        Args:
            z_world: First-frame student tokens with shape `(B, tokens_per_frame, D_student)`.
            target_meta: Teacher metadata containing `tokens_per_frame` and `sequence_length`.

        Returns:
            Predicted teacher tokens with shape `(B, teacher_sequence_length, D_teacher)`.

        Raises:
            ValueError: If input shape or teacher metadata is inconsistent with the projector.
        """
        if z_world.ndim != 3:
            raise ValueError(f"`z_world` must be 3D [B,N,D], got shape {tuple(z_world.shape)}.")
        if z_world.shape[-1] != self.d_student:
            raise ValueError(
                f"`z_world` hidden dim must be {self.d_student}, got {z_world.shape[-1]}."
            )
        tokens_per_frame = int(target_meta["tokens_per_frame"])
        teacher_sequence_length = int(target_meta["sequence_length"])
        if int(z_world.shape[1]) != tokens_per_frame:
            raise ValueError(
                f"Projector input tokens must equal tokens_per_frame={tokens_per_frame}, got {z_world.shape[1]}."
            )
        if teacher_sequence_length % tokens_per_frame != 0:
            raise ValueError(
                f"Teacher sequence length must be divisible by tokens_per_frame, got sequence_length={teacher_sequence_length}, tokens_per_frame={tokens_per_frame}."
            )
        num_frames = teacher_sequence_length // tokens_per_frame
        if num_frames > self.max_frames:
            raise ValueError(
                f"Teacher latent frames={num_frames} exceeds projector max_frames={self.max_frames}."
            )
        memory = self.memory_proj(self.input_norm(z_world))
        query = memory.unsqueeze(1).expand(-1, num_frames, -1, -1).contiguous()
        frame_emb = self.frame_embedding[:num_frames].to(device=query.device, dtype=query.dtype)
        query = query + frame_emb.view(1, num_frames, 1, self.projector_dim)
        query = query.reshape(z_world.shape[0], teacher_sequence_length, self.projector_dim)
        for block in self.blocks:
            query = block(query, memory)
        return self.output(query)
