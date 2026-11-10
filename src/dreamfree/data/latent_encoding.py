"""Load the VAE and encode full-episode windows and official I2V conditioning."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from dreamfree.data.latent_config import (
    VAE_COMPILE_MODES,
    PrecomputeConfig,
    _normalizer_mode,
    _normalizer_stats_src,
    _optional_path,
    _precompute_cache_contract,
)
from dreamfree.models.wan22.helpers.loader import (
    _load_model_from_state_dict,
    load_wan22_i2v_a14b_vae,
    load_wan22_ti2v_5b_vae,
)
from dreamfree.models.wan22.helpers.state_dict_converters import wan_video_vae_state_dict_converter
from dreamfree.models.wan22.wan_video_vae import WanVideoVAE, WanVideoVAE38
from dreamfree.utils.logging_config import get_logger

logger = get_logger("dreamfree.data.latent_preparation")


@dataclass
class PrecomputeVAEModel:
    vae: Any
    device: torch.device
    torch_dtype: torch.dtype
    vae_path: str
    stats_src: str
    norm_mode: str
    cache_contract: dict[str, Any]
    compile_enabled: bool = False

    def encode_video_latents(
        self,
        video_tensor: torch.Tensor,
        *,
        tiled: bool = False,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
    ) -> torch.Tensor:
        return self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )


def _encode_windows(
    encoder: PrecomputeVAEModel,
    video_processed: torch.Tensor,
    *,
    num_frames: int,
    action_video_freq_ratio: int,
    batch_size: int,
) -> torch.Tensor:
    """Build one sparse latent window per episode frame and VAE-encode it."""
    T_ep = int(video_processed.shape[0])
    if T_ep <= 0:
        raise ValueError("Latent precompute requires episodes with at least one frame.")
    return _encode_window_range(
        encoder,
        video_processed,
        num_frames=num_frames,
        action_video_freq_ratio=action_video_freq_ratio,
        batch_size=batch_size,
        start_index=0,
        end_index=T_ep,
    )


def _encode_14b_i2v_condition_windows(
    encoder: PrecomputeVAEModel,
    video_processed: torch.Tensor,
    *,
    num_frames: int,
    action_video_freq_ratio: int,
    batch_size: int,
) -> torch.Tensor:
    """Build official 14B I2V pseudo-video windows and VAE-encode them."""
    T_ep = int(video_processed.shape[0])
    if T_ep <= 0:
        raise ValueError("14B I2V condition precompute requires episodes with at least one frame.")
    return _encode_window_range(
        encoder,
        video_processed,
        num_frames=num_frames,
        action_video_freq_ratio=action_video_freq_ratio,
        batch_size=batch_size,
        start_index=0,
        end_index=T_ep,
        a14b_i2v_condition=True,
    )


def _encode_window_range(
    encoder: PrecomputeVAEModel,
    video_processed: torch.Tensor,
    *,
    num_frames: int,
    action_video_freq_ratio: int,
    batch_size: int,
    start_index: int,
    end_index: int,
    a14b_i2v_condition: bool = False,
) -> torch.Tensor:
    """Build sparse latent windows for start indices in ``[start_index, end_index)``."""
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"VAE encode batch_size must be positive, got {batch_size}.")
    T_ep = int(video_processed.shape[0])
    if T_ep <= 0:
        raise ValueError("Latent precompute requires episodes with at least one frame.")
    start_index = int(start_index)
    end_index = int(end_index)
    if start_index < 0 or end_index < start_index or end_index > T_ep:
        raise ValueError(
            f"Invalid encode window range [{start_index}, {end_index}) for episode length {T_ep}."
        )
    subsample = torch.arange(0, num_frames, action_video_freq_ratio, dtype=torch.long)
    assert subsample.numel() > 1
    n_samples = end_index - start_index
    if n_samples == 0:
        probe_idx = subsample.clamp(max=T_ep - 1)
        probe = video_processed[probe_idx].unsqueeze(0).permute(0, 2, 1, 3, 4)
        if a14b_i2v_condition:
            condition_probe = torch.zeros_like(probe)
            condition_probe[:, :, 0:1] = probe[:, :, 0:1]
            probe = condition_probe
        probe = probe.to(device=encoder.device, dtype=encoder.torch_dtype)
        if bool(encoder.compile_enabled) and int(probe.shape[0]) < batch_size:
            pad_count = batch_size - int(probe.shape[0])
            probe = torch.cat([probe, probe[-1:].expand(pad_count, *probe.shape[1:])], dim=0)
        with torch.no_grad():
            probe_latents = encoder.encode_video_latents(probe)
        return probe_latents.new_zeros((0, *probe_latents.shape[1:]), dtype=torch.float16).cpu()
    all_latents: list[torch.Tensor] = []
    max_frame_index = T_ep - 1
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        starts = torch.arange(start_index + batch_start, start_index + batch_end, dtype=torch.long)
        window_idx = (starts.unsqueeze(1) + subsample.unsqueeze(0)).clamp(max=max_frame_index)
        windows = video_processed[window_idx]
        windows = windows.permute(0, 2, 1, 3, 4).contiguous()
        if a14b_i2v_condition:
            condition_windows = torch.zeros_like(windows)
            condition_windows[:, :, 0:1] = windows[:, :, 0:1]
            windows = condition_windows
        windows = windows.to(device=encoder.device, dtype=encoder.torch_dtype, non_blocking=True)
        real_batch_size = int(windows.shape[0])
        if real_batch_size <= 0:
            raise ValueError("VAE encode produced an empty window batch before padding.")
        if bool(encoder.compile_enabled) and real_batch_size < batch_size:
            pad_count = batch_size - real_batch_size
            windows = torch.cat(
                [windows, windows[-1:].expand(pad_count, *windows.shape[1:])], dim=0
            )
        with torch.no_grad():
            latents = encoder.encode_video_latents(windows).detach()
        latents = latents[:real_batch_size]
        all_latents.append(latents.to(torch.float16).cpu())
    return torch.cat(all_latents, dim=0)


def _load_explicit_vae_checkpoint(
    cfg: PrecomputeConfig, *, device: str, dtype: torch.dtype, cache_contract: dict[str, Any]
) -> Any:
    vae_path = Path(str(cfg.vae_checkpoint)).expanduser().resolve()
    family = str(cache_contract["video_model_family"])
    z_dim = int(cache_contract["vae_z_dim"])
    if family == "wan22_i2v_a14b":
        return _load_model_from_state_dict(
            path=str(vae_path),
            model_class=WanVideoVAE,
            model_name="wan21_video_vae_for_i2v_14b",
            torch_dtype=dtype,
            device=device,
            model_kwargs={"z_dim": z_dim},
            state_dict_converter=wan_video_vae_state_dict_converter,
            strict=True,
        )
    if family == "wan22_ti2v_5b":
        return _load_model_from_state_dict(
            path=str(vae_path),
            model_class=WanVideoVAE38,
            model_name="wan22_ti2v_5b_vae",
            torch_dtype=dtype,
            device=device,
            model_kwargs={"z_dim": z_dim},
            state_dict_converter=wan_video_vae_state_dict_converter,
            strict=True,
        )
    raise ValueError(
        f"Unsupported video_model_family={family!r} for explicit VAE checkpoint loading."
    )


def _load_precompute_vae(
    cfg: PrecomputeConfig, *, device: str, dtype: torch.dtype
) -> PrecomputeVAEModel:
    os.environ.setdefault("DREAMFREE_MODEL_BASE_PATH", str(getattr(cfg, "model_base_path")))
    cache_contract = _precompute_cache_contract(cfg)
    explicit_vae_checkpoint = _optional_path(cfg.vae_checkpoint)
    if explicit_vae_checkpoint is not None:
        vae_path = Path(explicit_vae_checkpoint).expanduser().resolve()
        logger.info("Loading explicit VAE checkpoint from %s", vae_path)
        vae = _load_explicit_vae_checkpoint(
            cfg, device=device, dtype=dtype, cache_contract=cache_contract
        )
        vae.requires_grad_(False)
        vae.eval()
        return PrecomputeVAEModel(
            vae=vae,
            device=torch.device(device),
            torch_dtype=dtype,
            vae_path=str(vae_path),
            stats_src=_normalizer_stats_src(cfg),
            norm_mode=_normalizer_mode(cfg),
            cache_contract=cache_contract,
        )
    family = str(cache_contract["video_model_family"])
    if family == "wan22_i2v_a14b":
        loaded = load_wan22_i2v_a14b_vae(
            device=device,
            torch_dtype=dtype,
            model_id=str(cache_contract["model_id"]),
            redirect_common_files=bool(getattr(cfg, "redirect_common_files")),
            pretrained_load_broadcast=bool(getattr(cfg, "pretrained_load_broadcast", True)),
        )
    elif family == "wan22_ti2v_5b":
        loaded = load_wan22_ti2v_5b_vae(
            device=device,
            torch_dtype=dtype,
            model_id=str(cache_contract["model_id"]),
            redirect_common_files=bool(getattr(cfg, "redirect_common_files")),
            pretrained_load_broadcast=bool(getattr(cfg, "pretrained_load_broadcast", True)),
        )
    else:
        raise ValueError(f"Unsupported video_model_family={family!r} for latent precompute.")
    loaded.vae.requires_grad_(False)
    loaded.vae.eval()
    return PrecomputeVAEModel(
        vae=loaded.vae,
        device=torch.device(device),
        torch_dtype=dtype,
        vae_path=str(loaded.vae_path),
        stats_src=_normalizer_stats_src(cfg),
        norm_mode=_normalizer_mode(cfg),
        cache_contract=cache_contract,
    )


def _compile_precompute_vae_encode(encoder: PrecomputeVAEModel, *, mode: str) -> None:
    mode = str(mode).strip()
    if mode not in VAE_COMPILE_MODES:
        raise ValueError(
            f"Unsupported compile_vae_mode={mode!r}; expected one of {VAE_COMPILE_MODES}."
        )
    vae_model = getattr(getattr(encoder, "vae", None), "model", None)
    encode = getattr(vae_model, "encode", None)
    if vae_model is None or not callable(encode):
        raise TypeError(
            f"compile_vae requires encoder.vae.model.encode to be callable; got vae_model={type(vae_model).__name__}, encode={type(encode).__name__}."
        )
    try:
        import torch._dynamo

        torch._dynamo.config.cache_size_limit = max(int(torch._dynamo.config.cache_size_limit), 64)
        torch._dynamo.config.suppress_errors = False
        compiled_encode = torch.compile(encode, dynamic=False, mode=mode)
        setattr(vae_model, "encode", compiled_encode)
        encoder.compile_enabled = True
    except Exception as exc:
        raise RuntimeError(
            f"Failed to enable torch.compile for VAE encode with mode={mode!r}."
        ) from exc
    logger.info(
        "Enabled torch.compile for %s.encode with mode=%s dynamic=False.",
        type(vae_model).__name__,
        mode,
    )
