"""Compute the consumed causal VAE prefix with the original projection layout."""

import torch


def encode_first_frame(model, video, scale):
    """Return the reference projection layout; only frame zero is a valid encoding."""
    model.clear_cache()
    model._enc_conv_idx = [0]
    (out, model._enc_feat_map, model._enc_conv_idx) = model.encoder(
        video[:, :, :1], feat_cache=model._enc_feat_map, feat_idx=model._enc_conv_idx
    )
    latent_frames = 1 + (video.shape[2] - 1) // 4
    first = out
    for _ in range(1, latent_frames):
        out = torch.cat((out, torch.zeros_like(first)), dim=2)
    (mu, _) = model.conv1(out).chunk(2, dim=1)
    if isinstance(scale[0], torch.Tensor):
        scale = [s.to(dtype=mu.dtype, device=mu.device) for s in scale]
        mu = (mu - scale[0].view(1, model.z_dim, 1, 1, 1)) * scale[1].view(1, model.z_dim, 1, 1, 1)
    else:
        scale = scale.to(dtype=mu.dtype, device=mu.device)
        mu = (mu - scale[0]) * scale[1]
    return mu
