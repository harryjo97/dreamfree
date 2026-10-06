from __future__ import annotations

import math
import os
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from dreamfree.data.contracts import (
    _normalize_latent_frame0_mode,
    _stage0_latent_cache_expected_contract,
)
from dreamfree.data.lerobot.cached_latent_dataset import validate_stage2_latent_tensor_contract
from dreamfree.models.compile import compile_dit_blocks
from dreamfree.models.lora import (
    inject_lora_adapters,
    normalize_lora_config,
    set_lora_requires_grad,
)
from dreamfree.models.wan22.i2v_condition import (
    WAN22_A14B_I2V_CONDITION_LATENTS_KEY,
    concat_14b_i2v_condition,
)
from dreamfree.training.stage0.config import Stage0Config
from dreamfree.training.stage0.constants import (
    STAGE0_EVAL_SIGMA_BIN_WIDTH,
    STAGE0_VIDEO_ATTENTION_MASK_MODE,
    SUPPORTED_STAGE0_SIGMA_POLICIES,
    logger,
)


def _stage0_train_sigma_bounds(cfg: Any) -> tuple[float, float]:
    sigma_min = float(getattr(cfg, "train_sigma_min", 0.4))
    sigma_max = float(getattr(cfg, "train_sigma_max", 0.875))
    if not math.isfinite(sigma_min) or not math.isfinite(sigma_max):
        raise ValueError(
            f"Stage 0 train sigma bounds must be finite, got train_sigma_min={sigma_min}, train_sigma_max={sigma_max}."
        )
    if not 0.0 <= sigma_min <= sigma_max <= 1.0:
        raise ValueError(
            f"Stage 0 train sigma bounds must satisfy 0 <= train_sigma_min <= train_sigma_max <= 1, got {sigma_min}, {sigma_max}."
        )
    return (sigma_min, sigma_max)


def _validate_stage0_train_sigma_bounds_for_policy(
    policy: str, sigma_min: float, sigma_max: float
) -> None:
    if policy != "wan_shift_range":
        return
    if not 0.0 <= sigma_min < sigma_max < 1.0:
        raise ValueError(
            f"train_sigma_policy='wan_shift_range' requires 0 <= train_sigma_min < train_sigma_max < 1, got {sigma_min}, {sigma_max}."
        )


def _normalize_stage0_sigma_policy(value: Any) -> str:
    if value is None:
        return "wan_shift_full"
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized == "uniform":
        normalized = "uniform_range"
    if normalized not in SUPPORTED_STAGE0_SIGMA_POLICIES:
        raise ValueError(
            f"Unsupported train_sigma_policy={value!r}. Expected one of: {sorted(SUPPORTED_STAGE0_SIGMA_POLICIES)}."
        )
    return normalized


def _stage0_effective_train_sigma_bounds(cfg: Any) -> tuple[float, float]:
    policy = _normalize_stage0_sigma_policy(getattr(cfg, "train_sigma_policy", "wan_shift_full"))
    if policy == "wan_shift_full":
        return (0.0, 1.0)
    return _stage0_train_sigma_bounds(cfg)


def _stage0_sigma_metadata(cfg: Any) -> dict[str, Any]:
    policy = _normalize_stage0_sigma_policy(getattr(cfg, "train_sigma_policy", "wan_shift_full"))
    (sigma_min, sigma_max) = _stage0_effective_train_sigma_bounds(cfg)
    metadata = {
        "train_sigma_policy": policy,
        "sigma_policy": policy if policy != "uniform_range" else "uniform",
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "loss_weight_policy": "wan_shift" if policy == "wan_shift_full" else "flat",
        "stratified_sampling": policy == "uniform_range"
        and bool(getattr(cfg, "train_sigma_stratified_sampling", True))
        and (sigma_min != sigma_max),
        "num_train_timesteps": int(getattr(cfg, "num_train_timesteps")),
        "dit_role": str(getattr(cfg, "dit_role")),
        "video_attention_mask_mode": STAGE0_VIDEO_ATTENTION_MASK_MODE,
    }
    if policy in {"wan_shift_full", "wan_shift_range"}:
        metadata["train_shift"] = float(getattr(cfg, "train_shift"))
    return metadata


def _validate_stage0_dit_config(dit_config: dict[str, Any]) -> dict[str, Any]:
    mask_mode = str(dit_config.get("video_attention_mask_mode", ""))
    if mask_mode != STAGE0_VIDEO_ATTENTION_MASK_MODE:
        raise ValueError(
            f"Stage 0 must use pretrained bidirectional video attention; got video_attention_mask_mode={mask_mode!r}."
        )
    return dit_config


class VideoAdaptationModel(torch.nn.Module):
    """Wan2.2 core wrapper with cached text contexts and video-only loss."""

    def __init__(
        self,
        core: Any,
        *,
        latent_frame0_mode: str = "official",
        train_sigma_policy: str = "wan_shift_full",
        train_sigma_min: float = 0.4,
        train_sigma_max: float = 0.875,
        train_sigma_stratified_sampling: bool = True,
        sigma_strata_per_window: int = 1,
        use_text_embedding_cache: bool = False,
        use_cached_latents: bool = False,
        expected_latent_contract: dict[str, Any] | None = None,
    ):
        """Wrap a loaded Wan core for Stage 0 optimization.

        Args:
            core: ``Wan22Core`` instance with DiT, VAE, text encoder, and scheduler.
            latent_frame0_mode: 14B-only control for whether the main
                noised latent keeps official I2V frame-0 noise or restores a
                clean first frame.
            train_sigma_policy: Training-time sigma sampler. ``uniform_range``
                preserves the existing bounded uniform sampler; ``wan_shift_range``
                uses Wan's shifted sampler inside the configured sigma range;
                ``wan_shift_full`` uses Wan's shifted full-range scheduler sampler.
            train_sigma_min: Lower bound for effective flow-matching sigma.
            train_sigma_max: Upper bound for effective flow-matching sigma.
            train_sigma_stratified_sampling: Whether to stratify sigma samples
                across each gradient accumulation window.
            sigma_strata_per_window: Number of strata consumed per local
                accumulation window.
            use_text_embedding_cache: Whether batches must carry precomputed
                `context/context_mask` tensors instead of prompt strings.
            use_cached_latents: Whether batches must carry precomputed VAE
                latents in `input_latents` instead of decoded video.
            expected_latent_contract: Cache contract used to validate cached
                latent tensors before training.
        """
        super().__init__()
        if sigma_strata_per_window <= 0:
            raise ValueError(
                f"sigma_strata_per_window must be positive, got {sigma_strata_per_window}."
            )
        if not math.isfinite(float(train_sigma_min)) or not math.isfinite(float(train_sigma_max)):
            raise ValueError(
                f"train sigma bounds must be finite, got train_sigma_min={train_sigma_min}, train_sigma_max={train_sigma_max}."
            )
        normalized_train_sigma_policy = _normalize_stage0_sigma_policy(train_sigma_policy)
        if not 0.0 <= float(train_sigma_min) <= float(train_sigma_max) <= 1.0:
            raise ValueError(
                f"train sigma bounds must satisfy 0 <= min <= max <= 1, got train_sigma_min={train_sigma_min}, train_sigma_max={train_sigma_max}."
            )
        _validate_stage0_train_sigma_bounds_for_policy(
            normalized_train_sigma_policy, float(train_sigma_min), float(train_sigma_max)
        )
        self.core = core
        self.latent_frame0_mode = _normalize_latent_frame0_mode(latent_frame0_mode)
        self.train_sigma_policy = normalized_train_sigma_policy
        self.train_sigma_min = float(train_sigma_min)
        self.train_sigma_max = float(train_sigma_max)
        self.train_sigma_stratified_sampling = bool(train_sigma_stratified_sampling)
        self.sigma_strata_per_window = int(sigma_strata_per_window)
        self.use_text_embedding_cache = bool(use_text_embedding_cache)
        self.use_cached_latents = bool(use_cached_latents)
        self.expected_latent_contract = dict(expected_latent_contract or {})
        self._sigma_window: torch.Tensor | None = None
        self._sigma_window_position = 0
        self._context_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def set_stage0_train_mode(self) -> None:
        """Put only the denoising transformer on the training path."""
        self.core.train()
        self.core.dit.train()
        if self.core.vae is not None:
            self.core.vae.eval()
        if self.core.text_encoder is not None:
            self.core.text_encoder.eval()

    def _cached_context(self, prompt: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompts once and reuse text contexts across repeated tasks.

        Args:
            prompt: Prompt strings for the current batch.

        Returns:
            Pair ``(context, mask)`` with shapes ``[B, L, D]`` and ``[B, L]``.
        """
        prompt_list = [str(item) for item in prompt]
        missing = []
        seen = set()
        for item in prompt_list:
            if item not in self._context_cache and item not in seen:
                missing.append(item)
                seen.add(item)
        if missing:
            with torch.no_grad():
                (context, mask) = self.core.encode_prompt(missing)
            for i, item in enumerate(missing):
                self._context_cache[item] = (
                    context[i].detach().to(device="cpu").contiguous(),
                    mask[i].detach().to(device="cpu", dtype=torch.bool).contiguous(),
                )
            del context
            del mask
        context = torch.stack([self._context_cache[item][0] for item in prompt_list], dim=0)
        mask = torch.stack([self._context_cache[item][1] for item in prompt_list], dim=0)
        return (
            context.to(device=self.core.device, dtype=self.core.torch_dtype),
            mask.to(device=self.core.device, dtype=torch.bool),
        )

    def _cached_context_from_sample(
        self, sample: dict[str, Any], batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move cached text context tensors from a dataloader batch to the model device."""
        if "context" not in sample or "context_mask" not in sample:
            raise KeyError(
                "use_text_embedding_cache=true requires batch keys `context` and `context_mask`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        if not isinstance(context, torch.Tensor) or not isinstance(context_mask, torch.Tensor):
            raise TypeError(
                "Cached Stage 0 `context` and `context_mask` must be torch.Tensor values."
            )
        if context.ndim == 2 and batch_size == 1:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1 and batch_size == 1:
            context_mask = context_mask.unsqueeze(0)
        if context.ndim != 3:
            raise ValueError(
                f"Cached Stage 0 `context` must be 3D [B,L,D], got {tuple(context.shape)}."
            )
        if context_mask.ndim != 2:
            raise ValueError(
                f"Cached Stage 0 `context_mask` must be 2D [B,L], got {tuple(context_mask.shape)}."
            )
        if int(context.shape[0]) != int(batch_size):
            raise ValueError(
                f"Cached context batch mismatch: context batch={context.shape[0]} vs video batch={batch_size}."
            )
        if tuple(context_mask.shape) != (int(batch_size), int(context.shape[1])):
            raise ValueError(
                f"Cached context mask shape mismatch: context={tuple(context.shape)} mask={tuple(context_mask.shape)}."
            )
        return (
            context.to(device=self.core.device, dtype=self.core.torch_dtype),
            context_mask.to(device=self.core.device, dtype=torch.bool),
        )

    def _new_sigma_window(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Build one shuffled stratified sigma window on the current device."""
        sigma_min = self.train_sigma_min
        sigma_max = self.train_sigma_max
        if sigma_min == sigma_max:
            return torch.full(
                (self.sigma_strata_per_window,), sigma_min, device=device, dtype=dtype
            )
        edges = torch.linspace(
            sigma_min,
            sigma_max,
            self.sigma_strata_per_window + 1,
            device=device,
            dtype=torch.float32,
        )
        offsets = torch.rand((self.sigma_strata_per_window,), device=device, dtype=torch.float32)
        sigma = edges[:-1] + offsets * (edges[1:] - edges[:-1])
        order = torch.randperm(self.sigma_strata_per_window, device=device)
        return sigma[order].to(dtype=dtype)

    def _sample_training_sigma(
        self, *, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Sample effective flow-matching sigmas for Stage 0 training."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        sigma_min = self.train_sigma_min
        sigma_max = self.train_sigma_max
        if sigma_min == sigma_max:
            return torch.full((batch_size,), sigma_min, device=device, dtype=dtype)
        if not self.train_sigma_stratified_sampling:
            sigma = torch.empty((batch_size,), device=device, dtype=torch.float32).uniform_(
                sigma_min, sigma_max
            )
            return sigma.to(dtype=dtype)
        chunks: list[torch.Tensor] = []
        remaining = batch_size
        while remaining > 0:
            if (
                self._sigma_window is None
                or self._sigma_window_position >= int(self._sigma_window.shape[0])
                or self._sigma_window.device != device
            ):
                self._sigma_window = self._new_sigma_window(device=device, dtype=dtype)
                self._sigma_window_position = 0
            available = int(self._sigma_window.shape[0]) - self._sigma_window_position
            take = min(remaining, available)
            start = self._sigma_window_position
            chunks.append(self._sigma_window[start : start + take])
            self._sigma_window_position += take
            remaining -= take
        return torch.cat(chunks, dim=0).to(device=device, dtype=dtype)

    def _training_timestep(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        sigma: float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return effective sigmas and Wan timesteps for training/eval loss."""
        if sigma is None:
            if self.train_sigma_policy == "wan_shift_full":
                timestep = self.core.train_scheduler.sample_training_t(
                    batch_size=batch_size, device=device, dtype=dtype
                ).to(device=device, dtype=dtype)
                sigma_tensor = (
                    timestep.float() / float(self.core.train_scheduler.num_train_timesteps)
                ).to(device=device, dtype=dtype)
                return (sigma_tensor, timestep)
            if self.train_sigma_policy == "wan_shift_range":
                timestep = self.core.train_scheduler.sample_training_t_bounded(
                    batch_size=batch_size,
                    device=device,
                    dtype=dtype,
                    sigma_min=self.train_sigma_min,
                    sigma_max=self.train_sigma_max,
                ).to(device=device, dtype=dtype)
                sigma_tensor = (
                    timestep.float() / float(self.core.train_scheduler.num_train_timesteps)
                ).to(device=device, dtype=dtype)
                return (sigma_tensor, timestep)
            sigma_tensor = self._sample_training_sigma(
                batch_size=batch_size, device=device, dtype=dtype
            )
        elif isinstance(sigma, torch.Tensor):
            sigma_tensor = sigma.to(device=device, dtype=dtype)
            if sigma_tensor.ndim == 0:
                sigma_tensor = sigma_tensor.reshape(1).expand(batch_size)
            else:
                sigma_tensor = sigma_tensor.reshape(-1)
            if int(sigma_tensor.shape[0]) != batch_size:
                raise ValueError(
                    f"sigma must have shape [B] or be scalar, got {tuple(sigma.shape)} for B={batch_size}."
                )
            sigma_check = sigma_tensor.detach().float()
            if not torch.isfinite(sigma_check).all().item():
                raise ValueError("sigma tensor contains non-finite values.")
            if bool((sigma_check < 0.0).any().item()) or bool((sigma_check > 1.0).any().item()):
                raise ValueError("sigma tensor values must be in [0, 1].")
        else:
            sigma_value = float(sigma)
            if not math.isfinite(sigma_value) or not 0.0 <= sigma_value <= 1.0:
                raise ValueError(f"sigma must be finite and in [0, 1], got {sigma_value}.")
            sigma_tensor = torch.full((batch_size,), sigma_value, device=device, dtype=dtype)
        timestep = sigma_tensor * float(self.core.train_scheduler.num_train_timesteps)
        return (sigma_tensor, timestep)

    @staticmethod
    def _normalize_prompt_field(prompt: Any, batch_size: int) -> list[str]:
        """Validate and normalize the dataloader-collated prompt field.

        Args:
            prompt: A single prompt string or sequence of prompt strings.
            batch_size: Expected number of prompts.

        Returns:
            Prompt list with length ``batch_size``.
        """
        if isinstance(prompt, str):
            prompts = [prompt]
        elif isinstance(prompt, (list, tuple)):
            prompts = [str(item) for item in prompt]
        else:
            raise TypeError(f"sample['prompt'] must be str/list[str], got {type(prompt)}")
        if len(prompts) != batch_size:
            raise ValueError(
                f"Prompt batch mismatch: len(prompt)={len(prompts)} vs batch={batch_size}"
            )
        return prompts

    def training_loss(
        self, sample: dict[str, Any], *, sigma: float | torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the Stage 0 flow-matching loss for one batch.

        Args:
            sample: Batch dict from ``VideoDataset`` with live
                ``video`` or from ``CachedStage0LatentDataset`` with
                precomputed ``input_latents``.
            sigma: Optional fixed effective noise scale for validation. ``None``
                uses the configured training sampler.

        Returns:
            Scalar loss tensor and detached logging metrics.
        """
        has_latents = "input_latents" in sample
        has_video = "video" in sample
        family = str(getattr(self.core, "video_model_family", "wan22_ti2v_5b"))
        is_14b = family == "wan22_i2v_a14b"
        i2v_condition_latents = None
        if self.use_cached_latents:
            if has_video:
                raise ValueError(
                    "Stage 0 cached-latent training received live `video`; cached batches must be latent-only."
                )
            if not has_latents:
                raise ValueError(
                    "Stage 0 cached-latent training requires `input_latents` in every batch. Precompute a full video latent cache or set use_cached_latents=false."
                )
            input_latents_raw = sample["input_latents"]
            if not isinstance(input_latents_raw, torch.Tensor):
                raise TypeError(
                    f"sample['input_latents'] must be a torch.Tensor, got {type(input_latents_raw)}."
                )
            if input_latents_raw.ndim != 5:
                raise ValueError(
                    f"sample['input_latents'] must have shape [B,C,T,H,W], got {tuple(input_latents_raw.shape)}."
                )
            if not self.expected_latent_contract:
                raise ValueError(
                    "Stage 0 cached-latent training requires an expected latent cache contract."
                )
            validate_stage2_latent_tensor_contract(
                input_latents_raw, expected=self.expected_latent_contract, stage_label="Stage 0"
            )
            input_latents = input_latents_raw.to(
                device=self.core.device, dtype=self.core.torch_dtype, non_blocking=True
            ).detach()
            if is_14b:
                i2v_condition_raw = sample.get(WAN22_A14B_I2V_CONDITION_LATENTS_KEY)
                if not isinstance(i2v_condition_raw, torch.Tensor):
                    raise ValueError(
                        f"14B Stage 0 cached-latent training requires `{WAN22_A14B_I2V_CONDITION_LATENTS_KEY}` in every batch. Regenerate the latent cache with official Wan2.2 I2V conditioning."
                    )
                if tuple(i2v_condition_raw.shape) != tuple(input_latents_raw.shape):
                    raise ValueError(
                        f"`{WAN22_A14B_I2V_CONDITION_LATENTS_KEY}` must match input_latents shape, got {tuple(i2v_condition_raw.shape)} vs {tuple(input_latents_raw.shape)}."
                    )
                i2v_condition_latents = i2v_condition_raw.to(
                    device=self.core.device, dtype=self.core.torch_dtype, non_blocking=True
                ).detach()
            batch_size = int(input_latents.shape[0])
        else:
            if has_latents:
                raise ValueError(
                    "Stage 0 live-video training received `input_latents`; set use_cached_latents=true."
                )
            if not has_video:
                raise KeyError("Stage 0 live-video training requires batch key `video`.")
            video = sample["video"]
            if not isinstance(video, torch.Tensor):
                raise TypeError(f"sample['video'] must be a torch.Tensor, got {type(video)}")
            if video.ndim != 5 or video.shape[1] != 3:
                raise ValueError(
                    f"sample['video'] must have shape [B,3,T,H,W], got {tuple(video.shape)}"
                )
            if video.shape[2] % 4 != 1:
                raise ValueError(f"Video frame count must satisfy T % 4 == 1, got {video.shape[2]}")
            batch_size = int(video.shape[0])
            input_video = video.to(device=self.core.device, dtype=self.core.torch_dtype)
            with torch.no_grad():
                input_latents = self.core._encode_video_latents(input_video, tiled=False).detach()
                if is_14b:
                    i2v_condition_latents = self.core._encode_14b_i2v_condition_latents(
                        input_video, tiled=False
                    ).detach()
        if self.use_text_embedding_cache:
            (context, context_mask) = self._cached_context_from_sample(sample, int(batch_size))
        else:
            prompts = self._normalize_prompt_field(sample["prompt"], int(batch_size))
            (context, context_mask) = self._cached_context(prompts)
        noise = torch.randn_like(input_latents)
        (sigma_tensor, timestep) = self._training_timestep(
            batch_size=batch_size, device=self.core.device, dtype=input_latents.dtype, sigma=sigma
        )
        latents = self.core.train_scheduler.add_noise(input_latents, noise, timestep)
        target = self.core.train_scheduler.training_target(input_latents, noise, timestep)
        first_frame_latents = None
        fuse_flag = False
        drop_first_frame_loss = False
        first_frame_timestep_zero = True
        if getattr(self.core.dit, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True
            overwrite_frame0 = not is_14b or self.latent_frame0_mode == "clean_overwrite"
            if overwrite_frame0:
                latents[:, :, 0:1] = first_frame_latents
                drop_first_frame_loss = True
            first_frame_timestep_zero = overwrite_frame0
        model_latents = latents
        if is_14b:
            if i2v_condition_latents is None:
                raise ValueError(
                    "14B Stage 0 requires official I2V condition latents from VAE([first_image, zeros, zeros, ...])."
                )
            model_latents = concat_14b_i2v_condition(
                latents=latents, i2v_condition_latents=i2v_condition_latents
            )
        pred = self.core._model_fn(
            latents=model_latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
            first_frame_timestep_zero=first_frame_timestep_zero,
        )
        if drop_first_frame_loss:
            pred = pred[:, :, 1:]
            target = target[:, :, 1:]
        loss_per_sample = F.mse_loss(pred.float(), target.float(), reduction="none").mean(
            dim=(1, 2, 3, 4)
        )
        sample_weight = torch.ones_like(loss_per_sample)
        if self.train_sigma_policy == "wan_shift_full" and sigma is None:
            sample_weight = self.core.train_scheduler.training_weight(timestep).to(
                loss_per_sample.device, dtype=loss_per_sample.dtype
            )
            if sample_weight.ndim == 0:
                sample_weight = sample_weight.reshape(1).expand_as(loss_per_sample)
            else:
                sample_weight = sample_weight.reshape(-1)
            if int(sample_weight.shape[0]) != int(loss_per_sample.shape[0]):
                raise ValueError(
                    f"Wan shift training weight must return one value per sample, got {tuple(sample_weight.shape)} for batch={int(loss_per_sample.shape[0])}."
                )
        loss = (loss_per_sample * sample_weight).mean()
        sigma_float = sigma_tensor.detach().float()
        return (
            loss,
            {
                "loss_video": float(loss.detach().float().item()),
                "sigma_mean": float(sigma_float.mean().item()),
                "sigma_min": float(sigma_float.min().item()),
                "sigma_max": float(sigma_float.max().item()),
                "loss_weight_mean": float(sample_weight.detach().float().mean().item()),
            },
        )

    def forward(self, sample: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        """Dispatch module calls to ``training_loss``.

        Args:
            sample: Batch dict accepted by ``training_loss``.

        Returns:
            Scalar loss tensor and logging metrics.
        """
        return self.training_loss(sample)


def _default_dit_config(cfg: Stage0Config) -> dict[str, Any]:
    """Return the Wan2.2 DiT config used for video-only Stage 0.

    Args:
        cfg: Stage 0 configuration.

    Returns:
        DiT constructor configuration for ``Wan22Core``.
    """
    family = str(getattr(cfg, "video_model_family", "wan22_ti2v_5b"))
    if family == "wan22_i2v_a14b":
        return {
            "has_image_input": False,
            "patch_size": [1, 2, 2],
            "in_dim": 36,
            "hidden_dim": 5120,
            "ffn_dim": 13824,
            "freq_dim": 256,
            "text_dim": 4096,
            "out_dim": 16,
            "num_heads": 40,
            "attn_head_dim": 128,
            "num_layers": 40,
            "eps": 1e-06,
            "seperated_timestep": True,
            "require_clip_embedding": False,
            "require_vae_embedding": False,
            "fuse_vae_embedding_in_latents": True,
            "use_gradient_checkpointing": bool(cfg.use_gradient_checkpointing),
            "video_attention_mask_mode": STAGE0_VIDEO_ATTENTION_MASK_MODE,
            "action_conditioned": False,
            "action_dim": 1,
            "action_group_causal_mask_mode": "group_diagonal",
        }
    if family != "wan22_ti2v_5b":
        raise ValueError(f"Unsupported video_model_family={family!r}.")
    return {
        "has_image_input": False,
        "patch_size": [1, 2, 2],
        "in_dim": 48,
        "hidden_dim": 3072,
        "ffn_dim": 14336,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 48,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "eps": 1e-06,
        "seperated_timestep": True,
        "require_clip_embedding": False,
        "require_vae_embedding": False,
        "fuse_vae_embedding_in_latents": True,
        "use_gradient_checkpointing": bool(cfg.use_gradient_checkpointing),
        "video_attention_mask_mode": STAGE0_VIDEO_ATTENTION_MASK_MODE,
        "action_conditioned": False,
        "action_dim": 1,
        "action_group_causal_mask_mode": "group_diagonal",
    }


def build_stage0_model(cfg: Stage0Config, device: str, dtype: torch.dtype) -> VideoAdaptationModel:
    """Load Wan2.2 and attach trainable LoRA adapters to the DiT.

    Args:
        cfg: Stage 0 model and LoRA configuration.
        device: Device string used to load the Wan core.
        dtype: Torch dtype for model weights and compute.

    Returns:
        Wrapped Wan model with only LoRA parameters requiring gradients.
    """
    from dreamfree.models.wan22.wan22 import Wan22Core

    os.environ.setdefault("DREAMFREE_MODEL_BASE_PATH", str(cfg.model_base_path))
    dit_config = _validate_stage0_dit_config(_default_dit_config(cfg))
    core = Wan22Core.from_wan22_pretrained(
        device=device,
        torch_dtype=dtype,
        model_id=str(cfg.model_id),
        tokenizer_model_id=str(cfg.tokenizer_model_id),
        tokenizer_max_len=int(cfg.tokenizer_max_len),
        load_text_encoder=not bool(cfg.use_text_embedding_cache),
        load_vae=not bool(cfg.use_cached_latents) or int(cfg.eval_num_inference_steps) > 0,
        redirect_common_files=bool(cfg.redirect_common_files),
        pretrained_load_broadcast=bool(cfg.pretrained_load_broadcast),
        dit_config=dit_config,
        video_model_family=str(cfg.video_model_family),
        dit_role=str(cfg.dit_role),
        train_shift=float(cfg.train_shift),
        infer_shift=float(cfg.infer_shift),
        num_train_timesteps=int(cfg.num_train_timesteps),
    )
    wrapper = VideoAdaptationModel(
        core,
        latent_frame0_mode=_normalize_latent_frame0_mode(cfg.latent_frame0_mode),
        train_sigma_policy=_normalize_stage0_sigma_policy(cfg.train_sigma_policy),
        train_sigma_min=float(cfg.train_sigma_min),
        train_sigma_max=float(cfg.train_sigma_max),
        train_sigma_stratified_sampling=bool(cfg.train_sigma_stratified_sampling),
        sigma_strata_per_window=max(
            1, int(cfg.per_device_batch_size) * int(cfg.gradient_accumulation_steps)
        ),
        use_text_embedding_cache=bool(cfg.use_text_embedding_cache),
        use_cached_latents=bool(cfg.use_cached_latents),
        expected_latent_contract=_stage0_latent_cache_expected_contract(cfg)
        if bool(cfg.use_cached_latents)
        else None,
    )
    wrapper.requires_grad_(False)
    lora_config = normalize_lora_config(
        {
            "enabled": True,
            "rank": int(cfg.lora_rank),
            "alpha": float(cfg.lora_alpha),
            "dropout": float(cfg.lora_dropout),
            "target_modules": str(cfg.lora_target_modules),
        }
    )
    injected = inject_lora_adapters(wrapper.core.dit, lora_config)
    set_lora_requires_grad(wrapper.core.dit, True)
    wrapper.lora_config = lora_config
    wrapper.lora_modules = injected
    wrapper.set_stage0_train_mode()
    trainable = sum(param.numel() for param in wrapper.parameters() if param.requires_grad)
    logger.info(
        "Stage0 LoRA configured: rank=%d alpha=%.1f target=%s trainable_params=%.2fM targets=%d",
        cfg.lora_rank,
        cfg.lora_alpha,
        cfg.lora_target_modules,
        trainable / 1000000.0,
        len(injected),
    )
    return wrapper


def _compile_stage0_dit_blocks(model: VideoAdaptationModel) -> int:
    """Compile Stage 0 Wan video DiT blocks without replacing modules."""
    return compile_dit_blocks(model.core.dit, label="Stage 0")


def _stage0_eval_sigma_bins(cfg: Stage0Config) -> list[tuple[float, float]]:
    (sigma_min, sigma_max) = _stage0_effective_train_sigma_bounds(cfg)
    if sigma_min == sigma_max:
        return [(sigma_min, sigma_max)]
    width = STAGE0_EVAL_SIGMA_BIN_WIDTH
    count = max(1, int(math.ceil((sigma_max - sigma_min) / width - 1e-12)))
    bins: list[tuple[float, float]] = []
    for idx in range(count):
        lo = sigma_min + idx * width
        hi = sigma_min + (idx + 1) * width
        if idx == count - 1 or hi > sigma_max:
            hi = sigma_max
        bins.append((float(lo), float(hi)))
    return bins


def _stage0_sigma_bin_metric_suffix(lo: float, hi: float) -> str:
    return f"sigma_{lo:.2f}_{hi:.2f}".replace(".", "p")
