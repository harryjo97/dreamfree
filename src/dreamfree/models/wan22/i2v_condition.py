"""Wan image-to-video conditioning helpers."""

from __future__ import annotations

import torch

WAN22_A14B_I2V_LATENT_CHANNELS = 16
WAN22_A14B_I2V_MASK_CHANNELS = 4
WAN22_A14B_I2V_CONDITION_LATENTS_KEY = "i2v_condition_latents"
WAN22_A14B_I2V_CONDITION_SOURCE_OFFICIAL = "official_pseudo_video_vae"


def build_14b_i2v_condition(*, i2v_condition_latents: torch.Tensor) -> torch.Tensor:
    """Build the 20-channel Wan2.2-I2V-A14B first-image condition tensor.

    Args:
        i2v_condition_latents: Official I2V condition latents shaped
            ``[B,16,T,H,W]``. These must be produced by VAE-encoding a
            pixel-space pseudo-video ``[first_image, zeros, zeros, ...]``.

    Returns:
        Condition tensor shaped ``[B,20,T,H,W]`` containing four mask channels
        followed by the 16 VAE-encoded official I2V image-condition channels.

    Raises:
        ValueError: If latent rank or channel count is wrong.
    """
    if i2v_condition_latents.ndim != 5:
        raise ValueError(
            f"`i2v_condition_latents` must be 5D [B,16,T,H,W], got shape {tuple(i2v_condition_latents.shape)}."
        )
    (batch_size, channels, num_latent_frames, height, width) = i2v_condition_latents.shape
    if int(channels) != WAN22_A14B_I2V_LATENT_CHANNELS:
        raise ValueError(
            f"Wan2.2-I2V-A14B conditioning requires 16-channel official condition latents, got {channels} channels."
        )
    if int(num_latent_frames) <= 0:
        raise ValueError(
            f"`i2v_condition_latents` must contain at least one latent frame, got {num_latent_frames}."
        )
    mask = torch.zeros(
        (batch_size, WAN22_A14B_I2V_MASK_CHANNELS, num_latent_frames, height, width),
        device=i2v_condition_latents.device,
        dtype=i2v_condition_latents.dtype,
    )
    mask[:, :, 0:1] = 1.0
    return torch.cat([mask, i2v_condition_latents], dim=1)


def concat_14b_i2v_condition(
    *, latents: torch.Tensor, i2v_condition_latents: torch.Tensor
) -> torch.Tensor:
    """Concatenate 16-channel latents with the 20-channel 14B I2V condition."""
    if latents.ndim != 5:
        raise ValueError(f"`latents` must be 5D [B,16,T,H,W], got shape {tuple(latents.shape)}.")
    if int(latents.shape[1]) != WAN22_A14B_I2V_LATENT_CHANNELS:
        raise ValueError(f"`latents` must have 16 channels for 14B I2V, got {latents.shape[1]}.")
    if not isinstance(i2v_condition_latents, torch.Tensor):
        raise TypeError(
            f"`i2v_condition_latents` must be a torch.Tensor, got {type(i2v_condition_latents)}."
        )
    if i2v_condition_latents.ndim != 5:
        raise ValueError(
            f"`i2v_condition_latents` must be 5D [B,16,T,H,W], got shape {tuple(i2v_condition_latents.shape)}."
        )
    latent_layout = (latents.shape[0], latents.shape[2], latents.shape[3], latents.shape[4])
    condition_layout = (
        i2v_condition_latents.shape[0],
        i2v_condition_latents.shape[2],
        i2v_condition_latents.shape[3],
        i2v_condition_latents.shape[4],
    )
    if latent_layout != condition_layout:
        raise ValueError(
            f"`latents` and `i2v_condition_latents` must share batch/time/spatial shape, got {tuple(latents.shape)} vs {tuple(i2v_condition_latents.shape)}."
        )
    condition = build_14b_i2v_condition(
        i2v_condition_latents=i2v_condition_latents.to(device=latents.device, dtype=latents.dtype)
    )
    return torch.cat([latents, condition], dim=1)
