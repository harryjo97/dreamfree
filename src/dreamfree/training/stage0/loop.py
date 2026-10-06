from __future__ import annotations

import logging
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator

from dreamfree.data.contracts import _normalize_latent_frame0_mode
from dreamfree.data.sampling import build_dataloader
from dreamfree.data.video import (
    DatasetRoots,
    VideoDataset,
    _normalize_stage0_dataset_format,
    build_stage0_dataset,
    validate_stage0_training_config,
)
from dreamfree.models.adaptation import (
    _compile_stage0_dit_blocks,
    _normalize_stage0_sigma_policy,
    _stage0_effective_train_sigma_bounds,
    _stage0_sigma_metadata,
    build_stage0_model,
)
from dreamfree.training.accumulation import build_fixed_step_gradient_accumulation_plugin
from dreamfree.training.runtime import (
    _format_eta,
    _init_wandb,
    _json_dump,
    _json_load,
    _mixed_precision_to_dtype,
    _normalize_mixed_precision,
    _resolve_train_device,
    _resolve_training_budget,
    _wandb_log,
    assert_dataset_lengths_consistent,
    build_scheduler,
    collect_trainable_params,
    optimizer_parameters,
)
from dreamfree.training.stage0.checkpoint import (
    load_lora_checkpoint_if_requested,
    load_optimizer_state_if_requested,
    save_lora_checkpoint,
)
from dreamfree.training.stage0.config import Stage0Config
from dreamfree.training.stage0.constants import STAGE0_CACHED_LATENTS_DATASET_FORMAT, logger
from dreamfree.training.stage0.export import export_merged_dit_safetensors
from dreamfree.training.stage0.validation import evaluate_stage0_batches
from dreamfree.utils.fs import ensure_dir
from dreamfree.utils.logging_config import setup_logging


def run_stage0_training(cfg: Stage0Config) -> None:
    """Run dataset preparation, LoRA training, checkpointing, and export.

    Args:
        cfg: Complete Stage 0 configuration.
    """
    validate_stage0_training_config(cfg)
    dataset_format = _normalize_stage0_dataset_format(
        getattr(cfg, "dataset_format", "lerobot_root")
    )
    cfg.latent_frame0_mode = _normalize_latent_frame0_mode(cfg.latent_frame0_mode)
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    accelerator = Accelerator(
        gradient_accumulation_plugin=build_fixed_step_gradient_accumulation_plugin(
            int(cfg.gradient_accumulation_steps)
        ),
        mixed_precision=mixed_precision,
        step_scheduler_with_optimizer=False,
    )
    setup_logging(log_level=logging.INFO, is_main_process=accelerator.is_main_process)
    ensure_dir(cfg.output_dir)
    if cfg.export_final:
        ensure_dir(cfg.final_export_dir)
    logger.info(
        "Stage0 accelerator: distributed_type=%s world_size=%d process_index=%d mixed_precision=%s",
        accelerator.distributed_type,
        accelerator.num_processes,
        accelerator.process_index,
        mixed_precision,
    )
    torch.manual_seed(int(cfg.seed) + accelerator.process_index)
    np.random.seed(int(cfg.seed) + accelerator.process_index)
    random.seed(int(cfg.seed) + accelerator.process_index)
    task_roots: list[Path] = []
    if dataset_format == STAGE0_CACHED_LATENTS_DATASET_FORMAT:
        if accelerator.is_main_process:
            _json_dump(
                Path(cfg.output_dir) / "task_roots.json",
                {
                    "dataset_format": dataset_format,
                    "task_roots": [],
                    "stage0_latent_cache_dir": str(cfg.stage0_latent_cache_dir),
                },
            )
        accelerator.wait_for_everyone()
        logger.info(
            "Using cached-latent-only Stage 0 dataset from %s.", cfg.stage0_latent_cache_dir
        )
    else:
        preparer = DatasetRoots(cfg)
        if accelerator.is_main_process:
            prepared_roots = preparer.prepare()
            _json_dump(
                Path(cfg.output_dir) / "task_roots.json",
                {"task_roots": [str(path) for path in prepared_roots]},
            )
        accelerator.wait_for_everyone()
        task_root_payload = _json_load(Path(cfg.output_dir) / "task_roots.json")
        task_roots = [Path(path) for path in task_root_payload["task_roots"]]
        logger.info("Using %d extracted task roots.", len(task_roots))
    if cfg.prepare_only:
        if accelerator.is_main_process:
            _json_dump(Path(cfg.output_dir) / "stage0_config.json", asdict(cfg))
        logger.info("prepare_only=true; exiting after dataset preparation.")
        return
    train_dataset = build_stage0_dataset(task_roots, cfg=cfg, split="train")
    val_dataset = None
    if float(cfg.val_fraction) > 0 and int(cfg.val_epoch_size) > 0:
        val_dataset = build_stage0_dataset(task_roots, cfg=cfg, split="val")
    if val_dataset is None and int(cfg.eval_every) > 0 and (int(cfg.eval_num_inference_steps) > 0):
        raise ValueError(
            "Stage 0 qualitative eval requires a validation dataset. Set val_fraction>0 and val_epoch_size>0, or disable video eval with eval_num_inference_steps=0."
        )
    visual_eval_dataset = None
    if (
        val_dataset is not None
        and bool(cfg.use_cached_latents)
        and (dataset_format != STAGE0_CACHED_LATENTS_DATASET_FORMAT)
        and (int(cfg.eval_num_inference_steps) > 0)
    ):
        visual_eval_dataset = VideoDataset(task_roots, cfg=cfg, split="val")
    assert_dataset_lengths_consistent(accelerator, train_dataset, "train_dataset")
    if val_dataset is not None:
        assert_dataset_lengths_consistent(accelerator, val_dataset, "val_dataset")
    if visual_eval_dataset is not None:
        assert_dataset_lengths_consistent(accelerator, visual_eval_dataset, "visual_eval_dataset")
    train_window_count = getattr(train_dataset, "total_windows", None)
    if train_window_count is None:
        raise ValueError(
            "Stage 0 training dataset is missing `total_windows`; cannot resolve training budget."
        )
    budget = _resolve_training_budget(
        num_epochs=cfg.num_epochs,
        max_steps=int(cfg.max_steps),
        train_samples=int(train_window_count),
        world_size=accelerator.num_processes,
        per_device_batch_size=int(cfg.per_device_batch_size),
        gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        stage_label="Stage 0",
    )
    cfg.max_steps = int(budget.max_steps)
    logger.info(
        "Stage0 training budget: source=%s num_epochs=%s train_samples=%d effective_global_batch=%d max_steps=%d train_epoch_size=%d",
        budget.source,
        cfg.num_epochs,
        budget.train_samples,
        budget.effective_global_batch,
        cfg.max_steps,
        int(cfg.train_epoch_size),
    )
    sigma_policy = _normalize_stage0_sigma_policy(cfg.train_sigma_policy)
    (effective_sigma_min, effective_sigma_max) = _stage0_effective_train_sigma_bounds(cfg)
    sigma_metadata = _stage0_sigma_metadata(cfg)
    logger.info(
        "Stage0 sigma policy: policy=%s sigma_min=%.3f sigma_max=%.3f train_shift=%.3f loss_weight=%s stratified=%s strata_per_rank_window=%d",
        sigma_policy,
        effective_sigma_min,
        effective_sigma_max,
        float(cfg.train_shift),
        sigma_metadata["loss_weight_policy"],
        sigma_metadata["stratified_sampling"],
        max(1, int(cfg.per_device_batch_size) * int(cfg.gradient_accumulation_steps)),
    )
    if accelerator.is_main_process:
        _json_dump(Path(cfg.output_dir) / "stage0_config.json", asdict(cfg))
    train_loader = build_dataloader(train_dataset, cfg, shuffle=True)
    val_loader = (
        build_dataloader(val_dataset, cfg, shuffle=False) if val_dataset is not None else None
    )
    visual_eval_loader = (
        build_dataloader(visual_eval_dataset, cfg, shuffle=False)
        if visual_eval_dataset is not None
        else None
    )
    model_dtype = _mixed_precision_to_dtype(mixed_precision)
    model = build_stage0_model(cfg, device=_resolve_train_device(), dtype=model_dtype)
    global_step = load_lora_checkpoint_if_requested(model=model, cfg=cfg)
    if bool(cfg.compile_dit):
        _compile_stage0_dit_blocks(model)
    trainable_params = collect_trainable_params(model)
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
        betas=(0.9, 0.95),
    )
    scheduler = build_scheduler(optimizer, cfg)
    (model, optimizer, train_loader, scheduler) = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    trainable_params = optimizer_parameters(optimizer)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    if visual_eval_loader is not None:
        visual_eval_loader = accelerator.prepare(visual_eval_loader)
    load_optimizer_state_if_requested(optimizer=optimizer, scheduler=scheduler, cfg=cfg)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.set_stage0_train_mode()
    optimizer.zero_grad(set_to_none=True)
    wandb_run = _init_wandb(accelerator, cfg)
    if int(cfg.max_steps) <= 0:
        logger.info("max_steps<=0; built datasets/model and exiting without training.")
        return
    train_iter = iter(train_loader)
    val_iter = iter(val_loader) if val_loader is not None else None
    visual_eval_iter = iter(visual_eval_loader) if visual_eval_loader is not None else None
    start_step = int(global_step)
    run_start_time = time.perf_counter()
    perf_names = ("data_wait", "forward", "backward", "optimizer", "step_wall")
    perf_sums = {name: 0.0 for name in perf_names}
    perf_logged_steps = 0
    logger.info("Starting stage-0 training at step=%d max_steps=%d", global_step, cfg.max_steps)
    while global_step < int(cfg.max_steps):
        step_wall_start = time.perf_counter()
        data_start = time.perf_counter()
        try:
            sample = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            sample = next(train_iter)
        data_wait_seconds = time.perf_counter() - data_start
        with accelerator.accumulate(model):
            forward_start = time.perf_counter()
            (loss, loss_dict) = model(sample)
            forward_seconds = time.perf_counter() - forward_start
            backward_start = time.perf_counter()
            accelerator.backward(loss)
            backward_seconds = time.perf_counter() - backward_start
            if accelerator.sync_gradients:
                optimizer_start = time.perf_counter()
                grad_norm = accelerator.clip_grad_norm_(trainable_params, float(cfg.max_grad_norm))
                optimizer.step()
                if not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_seconds = time.perf_counter() - optimizer_start
                step_wall_seconds = time.perf_counter() - step_wall_start
                global_step += 1
                gathered_loss = accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                gathered_metrics = {}
                for key, value in loss_dict.items():
                    tensor = torch.tensor(
                        float(value), device=loss.device, dtype=torch.float32
                    ).reshape(1)
                    gathered_metrics[key] = accelerator.gather(tensor).mean().item()
                current_lr = float(optimizer.param_groups[0]["lr"])
                local_perf = torch.tensor(
                    [
                        data_wait_seconds,
                        forward_seconds,
                        backward_seconds,
                        optimizer_seconds,
                        step_wall_seconds,
                    ],
                    device=loss.device,
                    dtype=torch.float32,
                ).reshape(1, -1)
                gathered_perf = accelerator.gather(local_perf).reshape(-1, len(perf_names))
                max_rank_perf = gathered_perf.max(dim=0).values.detach().cpu().tolist()
                for name, value in zip(perf_names, max_rank_perf, strict=True):
                    perf_sums[name] += float(value)
                perf_logged_steps += 1
                if (
                    int(cfg.log_every) > 0
                    and global_step % int(cfg.log_every) == 0
                    and accelerator.is_main_process
                ):
                    (eta, steps_per_sec) = _format_eta(
                        run_start_time, start_step, global_step, int(cfg.max_steps)
                    )
                    perf_denominator = max(perf_logged_steps, 1)
                    perf_avgs = {name: perf_sums[name] / perf_denominator for name in perf_names}
                    metric_text = " ".join(
                        f"{key}={value:.4f}" for (key, value) in sorted(gathered_metrics.items())
                    )
                    logger.info(
                        "[train] step=%d/%d loss=%.4f %s lr=%.2e grad_norm=%.4f speed=%.3f step/s eta=%s perf_data=%.3fs perf_fwd=%.3fs perf_bwd=%.3fs perf_opt=%.3fs perf_step=%.3fs",
                        global_step,
                        cfg.max_steps,
                        gathered_loss,
                        metric_text,
                        current_lr,
                        float(grad_norm),
                        steps_per_sec,
                        eta,
                        perf_avgs["data_wait"],
                        perf_avgs["forward"],
                        perf_avgs["backward"],
                        perf_avgs["optimizer"],
                        perf_avgs["step_wall"],
                    )
                    payload = {
                        "train/loss": float(gathered_loss),
                        "train/lr": current_lr,
                        "train/grad_norm": float(grad_norm),
                        "performance/steps_per_sec": steps_per_sec,
                        "performance/data_wait_seconds": perf_avgs["data_wait"],
                        "performance/forward_seconds": perf_avgs["forward"],
                        "performance/backward_seconds": perf_avgs["backward"],
                        "performance/optimizer_seconds": perf_avgs["optimizer"],
                        "performance/step_wall_seconds": perf_avgs["step_wall"],
                    }
                    payload.update(
                        {f"train/{key}": float(value) for (key, value) in gathered_metrics.items()}
                    )
                    _wandb_log(wandb_run, payload, global_step)
                    perf_sums = {name: 0.0 for name in perf_names}
                    perf_logged_steps = 0
                if (
                    val_iter is not None
                    and int(cfg.eval_every) > 0
                    and (global_step % int(cfg.eval_every) == 0)
                ):
                    (metrics, val_iter, visual_eval_iter) = evaluate_stage0_batches(
                        accelerator=accelerator,
                        model=model,
                        val_iter=val_iter,
                        visual_eval_iter=visual_eval_iter,
                        val_loader=val_loader,
                        visual_eval_loader=visual_eval_loader,
                        cfg=cfg,
                        global_step=global_step,
                    )
                    accelerator.wait_for_everyone()
                    if metrics is not None and accelerator.is_main_process:
                        comparison_path = metrics.get("comparison_video_path")
                        bin_text = " ".join(
                            f"{key}={value:.4f}"
                            for (key, value) in sorted(metrics.items())
                            if key.startswith("val_loss_sigma_") and isinstance(value, (int, float))
                        )
                        logger.info(
                            "[eval] step=%d val_loss=%.4f %s comparison_video=%s",
                            global_step,
                            metrics["val_loss"],
                            bin_text,
                            comparison_path,
                        )
                        wandb_metrics = {
                            f"eval/{k}": v
                            for (k, v) in metrics.items()
                            if isinstance(v, (int, float))
                        }
                        _wandb_log(wandb_run, wandb_metrics, global_step)
                if int(cfg.save_every) > 0 and global_step % int(cfg.save_every) == 0:
                    save_lora_checkpoint(
                        accelerator=accelerator,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        cfg=cfg,
                        global_step=global_step,
                    )
                if int(cfg.export_every) > 0 and global_step % int(cfg.export_every) == 0:
                    export_merged_dit_safetensors(
                        accelerator=accelerator,
                        model=model,
                        export_dir=Path(cfg.output_dir) / "exports" / f"step_{global_step:06d}",
                        cfg=cfg,
                        global_step=global_step,
                    )
    save_lora_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        global_step=global_step,
    )
    if cfg.export_final:
        export_merged_dit_safetensors(
            accelerator=accelerator,
            model=model,
            export_dir=cfg.final_export_dir,
            cfg=cfg,
            global_step=global_step,
        )
    if wandb_run is not None:
        wandb_run.finish()
    logger.info("Stage 0 training finished at step=%d", global_step)
