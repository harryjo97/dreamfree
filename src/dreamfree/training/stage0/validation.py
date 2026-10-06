from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from accelerate import Accelerator
from PIL import Image, ImageDraw

from dreamfree.data.contracts import _normalize_latent_frame0_mode
from dreamfree.models.adaptation import (
    VideoAdaptationModel,
    _stage0_eval_sigma_bins,
    _stage0_sigma_bin_metric_suffix,
    _stage0_sigma_metadata,
)
from dreamfree.training.runtime import _json_dump
from dreamfree.training.stage0.config import Stage0Config
from dreamfree.utils.video_io import save_mp4


def _stage0_eval_artifact_paths(output_dir: str | Path, global_step: int) -> dict[str, Path]:
    """Return all qualitative Stage 0 eval artifact paths for one step."""
    eval_dir = Path(output_dir) / "eval" / f"step_{int(global_step):06d}"
    return {
        "dir": eval_dir,
        "comparison": eval_dir / "comparison.mp4",
        "generated": eval_dir / "generated.mp4",
        "eval_gt": eval_dir / "eval_gt.mp4",
        "first_image": eval_dir / "first_image.png",
        "prompt": eval_dir / "prompt.txt",
        "metadata": eval_dir / "metadata.json",
    }


def _stage0_normalized_video_to_pil_frames(video: torch.Tensor) -> list[Image.Image]:
    """Convert a Stage 0 video tensor from ``[-1, 1]`` to RGB PIL frames."""
    if not isinstance(video, torch.Tensor):
        raise TypeError(f"`video` must be a torch.Tensor, got {type(video)}")
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError(f"`video` must have shape [3,T,H,W], got {tuple(video.shape)}")
    if video.shape[1] < 1:
        raise ValueError("`video` must contain at least one frame.")
    video_u8 = ((video.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)
    frames = []
    for t in range(video_u8.shape[1]):
        frame = video_u8[:, t].permute(1, 2, 0).contiguous().numpy()
        frames.append(Image.fromarray(frame, mode="RGB"))
    return frames


def _stage0_label_frame(frame: Image.Image, label: str) -> Image.Image:
    """Add a compact label strip above one RGB frame."""
    rgb = frame.convert("RGB")
    strip_height = 28
    canvas = Image.new("RGB", (rgb.width, rgb.height + strip_height), (18, 18, 18))
    canvas.paste(rgb, (0, strip_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), str(label), fill=(255, 255, 255))
    return canvas


def _stage0_make_comparison_frames(
    generated_frames: Sequence[Image.Image], eval_frames: Sequence[Image.Image]
) -> list[Image.Image]:
    """Build side-by-side generated/eval frames for direct visual comparison."""
    generated = [frame.convert("RGB") for frame in generated_frames]
    eval_gt = [frame.convert("RGB") for frame in eval_frames]
    if not generated:
        raise ValueError("Generated video is empty.")
    if len(generated) != len(eval_gt):
        raise ValueError(
            f"Generated/eval frame count mismatch: generated={len(generated)} eval_gt={len(eval_gt)}"
        )
    comparison = []
    for idx, (gen_frame, gt_frame) in enumerate(zip(generated, eval_gt, strict=False)):
        if gen_frame.size != gt_frame.size:
            raise ValueError(
                f"Generated/eval frame size mismatch at frame {idx}: generated={gen_frame.size} eval_gt={gt_frame.size}"
            )
        left = _stage0_label_frame(gen_frame, "generated")
        right = _stage0_label_frame(gt_frame, "eval_gt")
        canvas = Image.new("RGB", (left.width + right.width, left.height), (0, 0, 0))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width, 0))
        comparison.append(canvas)
    return comparison


def _stage0_first_text(field: Any, name: str, *, required: bool = True) -> str | None:
    """Extract the first string from a dataloader-collated text field."""
    if field is None:
        if required:
            raise KeyError(f"Missing required eval field: {name}")
        return None
    if isinstance(field, str):
        return field
    if isinstance(field, (list, tuple)):
        if not field:
            if required:
                raise ValueError(f"Eval field {name} is empty.")
            return None
        return str(field[0])
    return str(field)


def _stage0_first_batch_int(sample: dict[str, Any], key: str) -> int:
    """Extract the first integer from a dataloader-collated scalar metadata field."""
    if key not in sample:
        raise KeyError(f"Missing required eval metadata field: {key}")
    value = sample[key]
    if isinstance(value, torch.Tensor):
        if value.numel() < 1:
            raise ValueError(f"Eval metadata field {key} is empty.")
        return int(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"Eval metadata field {key} is empty.")
        return int(value[0])
    return int(value)


def _stage0_write_eval_sidecars(
    *, paths: dict[str, Path], first_image: Image.Image, prompt: str, metadata: dict[str, Any]
) -> dict[str, Any]:
    """Write first image, prompt text, and metadata next to eval videos."""
    paths["dir"].mkdir(parents=True, exist_ok=True)
    first_image.convert("RGB").save(paths["first_image"])
    paths["prompt"].write_text(str(prompt).rstrip() + "\n", encoding="utf-8")
    metadata_with_paths = dict(metadata)
    metadata_with_paths["artifacts"] = {
        key: str(path) for (key, path) in paths.items() if key != "dir"
    }
    _json_dump(paths["metadata"], metadata_with_paths)
    return metadata_with_paths


@torch.no_grad()
def _save_stage0_eval_rollout_artifacts(
    *, model: VideoAdaptationModel, sample: dict[str, Any], cfg: Stage0Config, global_step: int
) -> dict[str, str]:
    """Generate and save one qualitative Stage 0 eval rollout."""
    if int(cfg.eval_num_inference_steps) <= 0:
        raise ValueError(
            f"Stage 0 eval rollout requires eval_num_inference_steps > 0. Got {cfg.eval_num_inference_steps}."
        )
    if int(cfg.eval_fps) <= 0:
        raise ValueError(f"Stage 0 eval rollout requires eval_fps > 0. Got {cfg.eval_fps}.")
    video = sample["video"]
    if not isinstance(video, torch.Tensor):
        raise TypeError(f"sample['video'] must be a torch.Tensor, got {type(video)}")
    if video.ndim != 5 or video.shape[0] < 1 or video.shape[1] != 3:
        raise ValueError(f"sample['video'] must have shape [B,3,T,H,W], got {tuple(video.shape)}")
    prompts = VideoAdaptationModel._normalize_prompt_field(sample["prompt"], int(video.shape[0]))
    prompt = prompts[0]
    raw_task = _stage0_first_text(sample.get("task"), "task", required=False)
    video0 = video[0].detach()
    gt_frames = _stage0_normalized_video_to_pil_frames(video0)
    input_image = video0[:, 0].unsqueeze(0)
    num_frames = int(video0.shape[1])
    infer_context = None
    infer_context_mask = None
    if bool(getattr(cfg, "use_text_embedding_cache", True)):
        if float(cfg.eval_text_cfg_scale) != 1.0:
            raise ValueError("Stage 0 cached-context eval requires eval_text_cfg_scale=1.0.")
        if "context" not in sample or "context_mask" not in sample:
            raise KeyError(
                "Stage 0 cached-context eval requires batch keys `context` and `context_mask`."
            )
        infer_context = sample["context"]
        infer_context_mask = sample["context_mask"]
        if not isinstance(infer_context, torch.Tensor) or not isinstance(
            infer_context_mask, torch.Tensor
        ):
            raise TypeError(
                "Stage 0 eval `context` and `context_mask` must be torch.Tensor values."
            )
        if infer_context.ndim == 2:
            infer_context = infer_context.unsqueeze(0)
        if infer_context_mask.ndim == 1:
            infer_context_mask = infer_context_mask.unsqueeze(0)
        infer_context = infer_context[0:1]
        infer_context_mask = infer_context_mask[0:1]
    infer_out = model.core.infer(
        prompt=prompt,
        input_image=input_image,
        num_frames=num_frames,
        action=None,
        negative_prompt=None,
        text_cfg_scale=float(cfg.eval_text_cfg_scale),
        action_cfg_scale=1.0,
        num_inference_steps=int(cfg.eval_num_inference_steps),
        seed=int(cfg.eval_seed),
        tiled=bool(cfg.eval_tiled),
        context=infer_context,
        context_mask=infer_context_mask,
        latent_frame0_mode=_normalize_latent_frame0_mode(cfg.latent_frame0_mode),
    )
    if "video" not in infer_out:
        raise KeyError("Stage 0 eval infer output is missing `video`.")
    generated_frames = [frame.convert("RGB") for frame in infer_out["video"]]
    comparison_frames = _stage0_make_comparison_frames(generated_frames, gt_frames)
    paths = _stage0_eval_artifact_paths(cfg.output_dir, global_step)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    save_mp4(generated_frames, str(paths["generated"]), fps=int(cfg.eval_fps))
    save_mp4(gt_frames, str(paths["eval_gt"]), fps=int(cfg.eval_fps))
    save_mp4(comparison_frames, str(paths["comparison"]), fps=int(cfg.eval_fps))
    metadata = {
        "global_step": int(global_step),
        "episode_index": _stage0_first_batch_int(sample, "episode_index"),
        "start_frame": _stage0_first_batch_int(sample, "start_frame"),
        "raw_task": raw_task,
        "prompt": prompt,
        "seed": int(cfg.eval_seed),
        "num_frames": num_frames,
        "height": int(video0.shape[-2]),
        "width": int(video0.shape[-1]),
        "fps": int(cfg.eval_fps),
        "num_inference_steps": int(cfg.eval_num_inference_steps),
        "text_cfg_scale": float(cfg.eval_text_cfg_scale),
        "rollout_metric_role": "diagnostic_only",
        "sigma_eval_bins": [
            {"min": float(lo), "max": float(hi)} for (lo, hi) in _stage0_eval_sigma_bins(cfg)
        ],
        **_stage0_sigma_metadata(cfg),
    }
    _stage0_write_eval_sidecars(
        paths=paths, first_image=gt_frames[0], prompt=prompt, metadata=metadata
    )
    return {
        "comparison_video_path": str(paths["comparison"]),
        "generated_video_path": str(paths["generated"]),
        "eval_gt_video_path": str(paths["eval_gt"]),
        "first_image_path": str(paths["first_image"]),
        "prompt_path": str(paths["prompt"]),
        "metadata_path": str(paths["metadata"]),
    }


@torch.no_grad()
def evaluate_one_batch(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    val_iter,
    visual_eval_iter=None,
    cfg: Stage0Config,
    global_step: int,
    save_rollout_artifacts: bool = True,
) -> dict[str, Any] | None:
    """Run validation on one prepared batch iterator item.

    Args:
        accelerator: Active Accelerate wrapper.
        model: Possibly distributed Stage 0 model.
        val_iter: Iterator over the prepared validation dataloader.
        visual_eval_iter: Optional iterator over live-video validation batches
            used only for qualitative rollout artifacts.
        cfg: Stage 0 configuration controlling qualitative rollout generation.
        global_step: Current optimizer step used in artifact paths.

    Returns:
        Gathered validation metrics, or ``None`` if the iterator is exhausted.
    """
    try:
        sample = next(val_iter)
    except StopIteration:
        return None
    visual_sample = None
    needs_rollout_artifact = bool(save_rollout_artifacts) and int(cfg.eval_num_inference_steps) > 0
    if needs_rollout_artifact:
        if visual_eval_iter is None:
            visual_sample = sample
        else:
            try:
                visual_sample = next(visual_eval_iter)
            except StopIteration:
                return None
        if "video" not in visual_sample:
            raise ValueError(
                "Stage 0 qualitative eval requires a live-video validation batch. Cached-latent validation batches cannot generate visual rollout artifacts."
            )
    unwrapped = accelerator.unwrap_model(model)
    was_training = unwrapped.training
    unwrapped.eval()
    bins = _stage0_eval_sigma_bins(cfg)
    bin_losses: list[float] = []
    result: dict[str, Any] = {}
    for sigma_lo, sigma_hi in bins:
        sigma_mid = (sigma_lo + sigma_hi) * 0.5
        (loss, _) = unwrapped.training_loss(sample, sigma=sigma_mid)
        loss_tensor = loss.detach().float().reshape(1)
        mean_loss = accelerator.gather_for_metrics(loss_tensor).mean().item()
        bin_losses.append(float(mean_loss))
        suffix = _stage0_sigma_bin_metric_suffix(sigma_lo, sigma_hi)
        result[f"val_loss_{suffix}"] = float(mean_loss)
    artifact_paths = {}
    if accelerator.is_main_process and needs_rollout_artifact:
        artifact_paths = _save_stage0_eval_rollout_artifacts(
            model=unwrapped, sample=visual_sample, cfg=cfg, global_step=global_step
        )
    if was_training:
        unwrapped.set_stage0_train_mode()
    mean_bin_loss = float(sum(bin_losses) / max(len(bin_losses), 1))
    result["val_loss"] = mean_bin_loss
    result["val_loss_video"] = mean_bin_loss
    result.update(artifact_paths)
    return result


@torch.no_grad()
def evaluate_stage0_batches(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    val_iter,
    visual_eval_iter,
    val_loader,
    visual_eval_loader,
    cfg: Stage0Config,
    global_step: int,
) -> tuple[dict[str, Any] | None, Any, Any]:
    """Evaluate one or more Stage 0 validation batches and average numeric metrics."""
    max_batches = int(cfg.eval_batches)
    if max_batches <= 0:
        raise ValueError(f"eval_batches must be positive, got {cfg.eval_batches}.")
    metric_sums: dict[str, float] = {}
    first_artifacts: dict[str, Any] = {}
    completed_batches = 0
    rollout_saved = False
    for _ in range(max_batches):
        metrics = evaluate_one_batch(
            accelerator=accelerator,
            model=model,
            val_iter=val_iter,
            visual_eval_iter=visual_eval_iter,
            cfg=cfg,
            global_step=global_step,
            save_rollout_artifacts=not rollout_saved,
        )
        if metrics is None:
            val_iter = iter(val_loader)
            visual_eval_iter = iter(visual_eval_loader) if visual_eval_loader is not None else None
            metrics = evaluate_one_batch(
                accelerator=accelerator,
                model=model,
                val_iter=val_iter,
                visual_eval_iter=visual_eval_iter,
                cfg=cfg,
                global_step=global_step,
                save_rollout_artifacts=not rollout_saved,
            )
        if metrics is None:
            break
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and (not isinstance(value, bool)):
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
            elif not rollout_saved:
                first_artifacts[key] = value
        completed_batches += 1
        rollout_saved = True
    if completed_batches <= 0:
        return (None, val_iter, visual_eval_iter)
    averaged = {
        key: float(value / completed_batches) for (key, value) in sorted(metric_sums.items())
    }
    averaged["val_batches"] = float(completed_batches)
    averaged.update(first_artifacts)
    return (averaged, val_iter, visual_eval_iter)
