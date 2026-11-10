"""Prepare live-video or cached-latent batches for the Stage 2 model."""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any

import torch

from dreamfree.models.wan22.i2v_condition import WAN22_A14B_I2V_CONDITION_LATENTS_KEY


def _normalize_torch_device(device: Any) -> torch.device | None:
    if device is None:
        return None
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _maybe_cuda_synchronize(device: Any) -> None:
    torch_device = _normalize_torch_device(device)
    if torch_device is None or torch_device.type != "cuda":
        return
    torch.cuda.synchronize(torch_device)


class _TimingRecorder:
    def __init__(self, *, enabled: bool, device: Any):
        self.enabled = bool(enabled)
        self.device = _normalize_torch_device(device)
        self.values: dict[str, float] = {}

    def record(self, key: str):
        if not self.enabled:
            return nullcontext()
        return _TimedSection(self, key)

    def add(self, key: str, value: float) -> None:
        if not self.enabled:
            return
        self.values[key] = self.values.get(key, 0.0) + float(value)


class _TimedSection:
    def __init__(self, recorder: _TimingRecorder, key: str):
        self.recorder = recorder
        self.key = key
        self.start: float | None = None

    def __enter__(self) -> None:
        _maybe_cuda_synchronize(self.recorder.device)
        self.start = time.perf_counter()
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        _maybe_cuda_synchronize(self.recorder.device)
        end = time.perf_counter()
        if self.start is not None:
            self.recorder.add(self.key, end - self.start)
        return False


def prepare_action_batch(
    core: Any,
    sample: dict[str, Any],
    tiled: bool = False,
    *,
    timing_recorder: _TimingRecorder | None = None,
) -> dict[str, Any]:
    """Validate supervision and assemble live/cached latents in batch order.

    Args:
        core: Wan core providing device, torch_dtype, model family, and VAE
            encoding methods. Cached-only batches do not invoke the VAE.
        sample: Collated tensors: normalized action (B, T_action, A), aligned
            normalized proprio (B, T_action, P), context (B, 128, 4096), and
            context_mask (B, 128), where True means an attended position.
            Supply video (B_video, 3, T_video, H, W), input_latents
            (B_cached, C, L, h, w), or disjoint subsets of both with
            video_batch_indices/input_latents_batch_indices mapping to B rows.
            Cached 14B input also requires official i2v_condition_latents of
            the same shape. Optional action_is_pad (B, T_action) and
            previous_action_is_pad (B, T_history) use True for padding.
            previous_action (B, T_history, A) uses normalized zeros at padding.
        tiled: Whether live VAE encoding uses tiling.
        timing_recorder: Optional recorder for device transfer and VAE time.

    Returns:
        Model-device/dtype tensors with detached input_latents (B, C, L, h, w),
        optional matching I2V latents, supervision, masks, and prompt/index
        metadata. dit_context_mask may be None when all positions are valid.
        This function neither normalizes targets nor writes/repairs caches.

    Raises:
        ValueError: If required fields, shapes, padding, I2V conditioning, or
            modality row coverage violate the batch contract.
    """
    # Check target and mask contracts before transferring data or invoking the VAE.
    required_common = ("context", "context_mask", "action", "proprio")
    has_latents = "input_latents" in sample
    has_video = "video" in sample
    required = required_common
    if not has_latents and (not has_video):
        required = required_common + ("video",)
    missing = [key for key in required if key not in sample]
    if missing:
        raise ValueError(f"Stage 2 training requires keys {required}; missing={missing}")
    action = sample["action"]
    proprio = sample["proprio"]
    previous_action = sample.get("previous_action")
    previous_action_is_pad = sample.get("previous_action_is_pad")
    context = sample["context"]
    context_mask = sample["context_mask"]
    if not isinstance(action, torch.Tensor) or action.ndim != 3:
        raise ValueError(
            f"`sample['action']` must have shape [B,T,A], got {tuple(getattr(action, 'shape', ()))}."
        )
    if not isinstance(proprio, torch.Tensor) or proprio.ndim != 3:
        raise ValueError(
            f"`sample['proprio']` must have shape [B,T,P], got {tuple(getattr(proprio, 'shape', ()))}."
        )
    if (
        not isinstance(context, torch.Tensor)
        or context.ndim != 3
        or tuple(context.shape[1:]) != (128, 4096)
    ):
        raise ValueError(
            f"`sample['context']` must have shape [B,128,4096], got {tuple(getattr(context, 'shape', ()))}."
        )
    if (
        not isinstance(context_mask, torch.Tensor)
        or context_mask.ndim != 2
        or tuple(context_mask.shape[1:]) != (128,)
    ):
        raise ValueError(
            f"`sample['context_mask']` must have shape [B,128], got {tuple(getattr(context_mask, 'shape', ()))}."
        )
    context_mask_bool = context_mask.to(dtype=torch.bool)
    batch_size = int(action.shape[0])
    if int(context.shape[0]) != batch_size or int(context_mask.shape[0]) != batch_size:
        raise ValueError(
            f"Stage 2 cached context batch mismatch: action={batch_size}, context={context.shape[0]}, mask={context_mask.shape[0]}."
        )
    if int(proprio.shape[0]) != batch_size:
        raise ValueError(
            f"Stage 2 batch mismatch: action={batch_size}, proprio={proprio.shape[0]}."
        )
    if (previous_action is None) != (previous_action_is_pad is None):
        raise ValueError(
            "Stage 2 action history requires both `previous_action` and `previous_action_is_pad`, or neither."
        )
    if previous_action is not None:
        if not isinstance(previous_action, torch.Tensor) or previous_action.ndim != 3:
            raise ValueError(
                f"`sample['previous_action']` must have shape [B,H,A], got {tuple(getattr(previous_action, 'shape', ()))}."
            )
        expected_history_shape = (batch_size, int(previous_action.shape[1]))
        if int(previous_action.shape[1]) <= 0 or int(previous_action.shape[2]) != int(
            action.shape[2]
        ):
            raise ValueError(
                f"Stage 2 previous-action shape must use a positive history horizon and match action_dim: previous={tuple(previous_action.shape)} action={tuple(action.shape)}."
            )
        if (
            not isinstance(previous_action_is_pad, torch.Tensor)
            or tuple(previous_action_is_pad.shape) != expected_history_shape
        ):
            raise ValueError(
                f"`sample['previous_action_is_pad']` must have shape {expected_history_shape}, got {tuple(getattr(previous_action_is_pad, 'shape', ()))}."
            )
        history_pad_mask = previous_action_is_pad.to(
            device=previous_action.device, dtype=torch.bool
        )
        padded_history_nonzero = previous_action.ne(0).any(dim=-1) & history_pad_mask
        if bool(padded_history_nonzero.any().item()):
            raise ValueError("Padded previous-action values must be normalized zeros.")
    if int(action.shape[1]) <= 0 or int(proprio.shape[1]) <= 0:
        raise ValueError(
            f"Stage 2 action/proprio horizon must be positive, got action={action.shape[1]} proprio={proprio.shape[1]}."
        )
    if int(action.shape[1]) != int(proprio.shape[1]):
        raise ValueError(
            f"Stage 2 requires proprio to align with action horizon, got action={action.shape[1]} proprio={proprio.shape[1]}."
        )
    video_model_family = str(getattr(core, "video_model_family", "wan22_ti2v_5b"))
    is_14b = video_model_family == "wan22_i2v_a14b"
    action_is_pad = sample.get("action_is_pad")
    if action_is_pad is not None:
        if not isinstance(action_is_pad, torch.Tensor) or tuple(action_is_pad.shape) != (
            batch_size,
            action.shape[1],
        ):
            raise ValueError(
                f"`sample['action_is_pad']` must have shape {(batch_size, action.shape[1])}, got {tuple(getattr(action_is_pad, 'shape', ()))}."
            )
        valid_sum = (~action_is_pad.bool()).sum(dim=1)
        if bool((valid_sum <= 0).any().item()):
            sample_id = sample.get("sample_id")
            prompt = sample.get("prompt")
            raise ValueError(
                f"Stage 2 received an all-padded action sample. sample_id={sample_id!r} prompt={prompt!r}"
            )

    def _resolve_batch_indices(
        key: str, *, raw_indices: Any, subset_batch_size: int
    ) -> torch.Tensor:
        if raw_indices is None:
            if subset_batch_size != batch_size:
                raise ValueError(
                    f"`sample[{key!r}]` is required when modality subset batch size {subset_batch_size} does not match full batch size {batch_size}."
                )
            return torch.arange(batch_size, dtype=torch.long)
        if not isinstance(raw_indices, torch.Tensor) or raw_indices.ndim != 1:
            raise ValueError(
                f"`sample[{key!r}]` must be 1D long tensor of length {subset_batch_size}, got {tuple(getattr(raw_indices, 'shape', ()))}."
            )
        if int(raw_indices.shape[0]) != subset_batch_size:
            raise ValueError(
                f"`sample[{key!r}]` length must match modality batch size {subset_batch_size}, got {raw_indices.shape[0]}."
            )
        return raw_indices.to(dtype=torch.long, device="cpu")

    # A mixed batch must assign every target row to exactly one visual source.
    assigned_rows = torch.zeros(batch_size, dtype=torch.bool)
    latent_batch_indices_cpu: torch.Tensor | None = None
    video_batch_indices_cpu: torch.Tensor | None = None
    input_latents_raw = sample.get("input_latents")
    i2v_condition_raw = sample.get(WAN22_A14B_I2V_CONDITION_LATENTS_KEY)
    video_raw = sample.get("video")
    if has_latents:
        if not isinstance(input_latents_raw, torch.Tensor) or input_latents_raw.ndim != 5:
            raise ValueError(
                f"`sample['input_latents']` must be 5D [B,C,T,H,W]; got {tuple(getattr(input_latents_raw, 'shape', ()))}."
            )
        if is_14b:
            if not isinstance(i2v_condition_raw, torch.Tensor) or i2v_condition_raw.ndim != 5:
                raise ValueError(
                    f"Stage 2 Wan2.2 14B cached-latent training requires `sample[{WAN22_A14B_I2V_CONDITION_LATENTS_KEY!r}]` with shape [B,C,T,H,W]. Regenerate the latent cache with official Wan2.2 I2V pseudo-video conditioning."
                )
            if tuple(i2v_condition_raw.shape) != tuple(input_latents_raw.shape):
                raise ValueError(
                    f"Stage 2 14B cached I2V condition shape must match input latents: condition={tuple(i2v_condition_raw.shape)} latents={tuple(input_latents_raw.shape)}."
                )
        latent_batch_indices_cpu = _resolve_batch_indices(
            "input_latents_batch_indices",
            raw_indices=sample.get("input_latents_batch_indices"),
            subset_batch_size=int(input_latents_raw.shape[0]),
        )
        if bool(
            ((latent_batch_indices_cpu < 0) | (latent_batch_indices_cpu >= batch_size)).any().item()
        ):
            raise ValueError(
                f"`sample['input_latents_batch_indices']` must stay within [0, {batch_size}), got {latent_batch_indices_cpu.tolist()}."
            )
        if bool(assigned_rows[latent_batch_indices_cpu].any().item()):
            raise ValueError(
                f"`sample['input_latents_batch_indices']` contains duplicate rows: {latent_batch_indices_cpu.tolist()}."
            )
        assigned_rows[latent_batch_indices_cpu] = True
    if has_video:
        if (
            not isinstance(video_raw, torch.Tensor)
            or video_raw.ndim != 5
            or int(video_raw.shape[1]) != 3
        ):
            raise ValueError(
                f"`sample['video']` must have shape [B,3,T,H,W], got {tuple(getattr(video_raw, 'shape', ()))}."
            )
        if video_raw.shape[2] % 4 != 1:
            raise ValueError(
                f"Video frame count must satisfy T % 4 == 1, got {video_raw.shape[2]}."
            )
        if video_raw.shape[3] % 16 != 0 or video_raw.shape[4] % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={video_raw.shape[3]}, W={video_raw.shape[4]}."
            )
        video_batch_indices_cpu = _resolve_batch_indices(
            "video_batch_indices",
            raw_indices=sample.get("video_batch_indices"),
            subset_batch_size=int(video_raw.shape[0]),
        )
        if bool(
            ((video_batch_indices_cpu < 0) | (video_batch_indices_cpu >= batch_size)).any().item()
        ):
            raise ValueError(
                f"`sample['video_batch_indices']` must stay within [0, {batch_size}), got {video_batch_indices_cpu.tolist()}."
            )
        if bool(assigned_rows[video_batch_indices_cpu].any().item()):
            raise ValueError(
                f"Stage 2 mixed cached/live batch assigned the same batch rows to both modalities: {video_batch_indices_cpu.tolist()}."
            )
        assigned_rows[video_batch_indices_cpu] = True
    if not bool(assigned_rows.all().item()):
        missing_rows = (~assigned_rows).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(
            f"Stage 2 mixed cached/live batch left rows without modality assignment: {missing_rows}."
        )
    image_is_pad = sample.get("image_is_pad")
    with (
        timing_recorder.record("diag_input_transfer_seconds")
        if timing_recorder is not None
        else nullcontext()
    ):
        context = context.to(device=core.device, dtype=core.torch_dtype, non_blocking=True)
        context_mask = context_mask_bool.to(device=core.device, dtype=torch.bool, non_blocking=True)
        dit_context_mask = (
            None
            if context_mask_bool.device.type == "cpu" and bool(context_mask_bool.all().item())
            else context_mask_bool.to(device=core.device, dtype=torch.bool, non_blocking=True)
        )
        action = action.to(device=core.device, dtype=core.torch_dtype, non_blocking=True)
        proprio = proprio.to(device=core.device, dtype=core.torch_dtype, non_blocking=True)
        previous_action = (
            None
            if previous_action is None
            else previous_action.to(device=core.device, dtype=core.torch_dtype, non_blocking=True)
        )
        previous_action_is_pad = (
            None
            if previous_action_is_pad is None
            else previous_action_is_pad.to(device=core.device, dtype=torch.bool, non_blocking=True)
        )
        action_is_pad = (
            None
            if action_is_pad is None
            else action_is_pad.to(device=core.device, dtype=torch.bool, non_blocking=True)
        )
        if image_is_pad is not None:
            if (
                not isinstance(image_is_pad, torch.Tensor)
                or image_is_pad.ndim != 2
                or int(image_is_pad.shape[0]) != batch_size
            ):
                raise ValueError(
                    f"`sample['image_is_pad']` must have shape [B,T], got {tuple(getattr(image_is_pad, 'shape', ()))}."
                )
            image_is_pad = image_is_pad.to(device=core.device, dtype=torch.bool, non_blocking=True)
        input_latents_cached = (
            None
            if not has_latents
            else input_latents_raw.to(
                device=core.device, dtype=core.torch_dtype, non_blocking=True
            ).detach()
        )
        i2v_condition_cached = (
            None
            if not (is_14b and has_latents)
            else i2v_condition_raw.to(
                device=core.device, dtype=core.torch_dtype, non_blocking=True
            ).detach()
        )
        input_video = (
            None
            if not has_video
            else video_raw.to(device=core.device, dtype=core.torch_dtype, non_blocking=True)
        )
    # Encode only the live subset; cached latents retain their supplied values.
    input_latents_video = None
    i2v_condition_video = None
    if has_video:
        with (
            timing_recorder.record("diag_live_vae_encode_seconds")
            if timing_recorder is not None
            else nullcontext()
        ):
            with torch.no_grad():
                input_latents_video = core._encode_video_latents(input_video, tiled=tiled).detach()
                if is_14b:
                    i2v_condition_video = core._encode_14b_i2v_condition_latents(
                        input_video, tiled=tiled
                    ).detach()
    reference_latents = (
        input_latents_cached if input_latents_cached is not None else input_latents_video
    )
    if reference_latents is None:
        raise AssertionError("Stage 2 batch must provide cached latents, live video, or both.")
    if input_latents_cached is not None and input_latents_video is not None:
        if tuple(input_latents_cached.shape[1:]) != tuple(input_latents_video.shape[1:]):
            raise ValueError(
                f"Stage 2 cached/live latent shapes do not match: cached={tuple(input_latents_cached.shape)} live={tuple(input_latents_video.shape)}."
            )
    reference_i2v_condition = (
        i2v_condition_cached if i2v_condition_cached is not None else i2v_condition_video
    )
    if is_14b:
        if reference_i2v_condition is None:
            raise ValueError(
                f"Stage 2 Wan2.2 14B requires official I2V condition latents. Provide live video or regenerate cached latents with `{WAN22_A14B_I2V_CONDITION_LATENTS_KEY}`."
            )
        if tuple(reference_i2v_condition.shape[1:]) != tuple(reference_latents.shape[1:]):
            raise ValueError(
                f"Stage 2 14B I2V condition shape must match input latents after the batch axis: condition={tuple(reference_i2v_condition.shape)} latents={tuple(reference_latents.shape)}."
            )
        if i2v_condition_cached is not None and input_latents_cached is not None:
            if tuple(i2v_condition_cached.shape) != tuple(input_latents_cached.shape):
                raise ValueError(
                    f"Stage 2 cached 14B I2V condition shape mismatch: condition={tuple(i2v_condition_cached.shape)} latents={tuple(input_latents_cached.shape)}."
                )
        if i2v_condition_video is not None and input_latents_video is not None:
            if tuple(i2v_condition_video.shape) != tuple(input_latents_video.shape):
                raise ValueError(
                    f"Stage 2 live 14B I2V condition shape mismatch: condition={tuple(i2v_condition_video.shape)} latents={tuple(input_latents_video.shape)}."
                )
    latent_t = int(reference_latents.shape[2])
    expected_video_t = (latent_t - 1) * 4 + 1
    if image_is_pad is not None and tuple(image_is_pad.shape) != (batch_size, expected_video_t):
        raise ValueError(
            f"`sample['image_is_pad']` must have shape {(batch_size, expected_video_t)} for latent_t={latent_t}, got {tuple(getattr(image_is_pad, 'shape', ()))}."
        )
    # Restore target-row order after processing the two visual subsets separately.
    input_latents = torch.empty(
        (batch_size, *reference_latents.shape[1:]), device=core.device, dtype=core.torch_dtype
    )
    if input_latents_cached is not None:
        assert latent_batch_indices_cpu is not None
        input_latents.index_copy_(
            0,
            latent_batch_indices_cpu.to(device=core.device, non_blocking=True),
            input_latents_cached,
        )
    if input_latents_video is not None:
        assert video_batch_indices_cpu is not None
        input_latents.index_copy_(
            0,
            video_batch_indices_cpu.to(device=core.device, non_blocking=True),
            input_latents_video,
        )
    i2v_condition_latents = None
    if is_14b:
        assert reference_i2v_condition is not None
        i2v_condition_latents = torch.empty_like(input_latents)
        if i2v_condition_cached is not None:
            assert latent_batch_indices_cpu is not None
            i2v_condition_latents.index_copy_(
                0,
                latent_batch_indices_cpu.to(device=core.device, non_blocking=True),
                i2v_condition_cached,
            )
        if i2v_condition_video is not None:
            assert video_batch_indices_cpu is not None
            i2v_condition_latents.index_copy_(
                0,
                video_batch_indices_cpu.to(device=core.device, non_blocking=True),
                i2v_condition_video,
            )
    inputs = {
        "context": context,
        "context_mask": context_mask,
        "dit_context_mask": dit_context_mask,
        "input_latents": input_latents,
        "action": action,
        "proprio": proprio,
        "action_is_pad": action_is_pad,
        "image_is_pad": image_is_pad,
    }
    if previous_action is not None:
        inputs["previous_action"] = previous_action
        inputs["previous_action_is_pad"] = previous_action_is_pad
    if i2v_condition_latents is not None:
        inputs[WAN22_A14B_I2V_CONDITION_LATENTS_KEY] = i2v_condition_latents
    for key in ("prompt", "idx"):
        if key in sample:
            inputs[key] = sample[key]
    return inputs


# Compatibility name for callers predating the live/cached batch API rename.
build_stage2_cached_inputs = prepare_action_batch
