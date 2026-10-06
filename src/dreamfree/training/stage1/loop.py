from __future__ import annotations

import logging
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
from torch.utils.data import Dataset

from dreamfree.data.distillation import _build_stage1_dataset, _resolve_stage1_task_roots
from dreamfree.data.sampling import build_dataloader
from dreamfree.models.distillation import DistillationModel, _compile_stage1_dit_blocks
from dreamfree.models.wan22.future_state import FirstFrameToTeacherStateProjector
from dreamfree.models.wan22.stage0_loader import load_stage0_merged_backbone
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
from dreamfree.training.stage1.checkpoint import (
    _load_stage1_model_weights,
    _load_stage1_optimizer_state_if_requested,
    _read_resume_metadata,
    save_stage1_checkpoint,
)
from dreamfree.training.stage1.config import Stage1Config
from dreamfree.training.stage1.constants import logger
from dreamfree.training.stage1.validation import (
    _gather_scalar_metrics,
    evaluate_stage1_batches,
    validate_stage1_launch_config,
)
from dreamfree.utils.fs import ensure_dir
from dreamfree.utils.logging_config import setup_logging


def run_stage1_training(cfg: Stage1Config) -> None:
    """Run dataset preparation, model setup, training, eval, and checkpointing.

    Args:
        cfg: Complete Stage 1 configuration, usually produced by `parse_args`.

    Raises:
        ValueError: If config, dataset, resume checkpoint, or tensor layouts are
            incompatible with the strict Stage 1 setup.
    """
    from dreamfree.models.artifacts import resolve_stage0_expectations

    resolve_stage0_expectations(cfg)
    validate_stage1_launch_config(cfg)
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    accelerator = Accelerator(
        gradient_accumulation_plugin=build_fixed_step_gradient_accumulation_plugin(
            int(cfg.gradient_accumulation_steps)
        ),
        mixed_precision=mixed_precision,
        step_scheduler_with_optimizer=False,
        dataloader_config=DataLoaderConfiguration(non_blocking=True),
    )
    setup_logging(log_level=logging.INFO, is_main_process=accelerator.is_main_process)
    ensure_dir(cfg.output_dir)
    logger.info(
        "Stage1 accelerator: distributed_type=%s world_size=%d process_index=%d mixed_precision=%s",
        accelerator.distributed_type,
        accelerator.num_processes,
        accelerator.process_index,
        mixed_precision,
    )
    torch.manual_seed(int(cfg.seed) + accelerator.process_index)
    np.random.seed(int(cfg.seed) + accelerator.process_index)
    random.seed(int(cfg.seed) + accelerator.process_index)
    if accelerator.is_main_process:
        task_roots = _resolve_stage1_task_roots(cfg)
        _json_dump(
            Path(cfg.output_dir) / "task_roots.json",
            {"task_roots": [str(path) for path in task_roots]},
        )
    accelerator.wait_for_everyone()
    task_roots = [
        Path(path) for path in _json_load(Path(cfg.output_dir) / "task_roots.json")["task_roots"]
    ]
    logger.info("Using %d extracted task roots.", len(task_roots))
    if cfg.prepare_only:
        if accelerator.is_main_process:
            _json_dump(Path(cfg.output_dir) / "stage1_config.json", asdict(cfg))
        logger.info("prepare_only=true; exiting after dataset preparation.")
        return
    train_dataset = _build_stage1_dataset(task_roots, cfg, split="train")
    val_dataset: Dataset | None = None
    if float(cfg.val_fraction) > 0.0 and int(cfg.val_epoch_size) > 0:
        val_dataset = _build_stage1_dataset(task_roots, cfg, split="val")
    assert_dataset_lengths_consistent(accelerator, train_dataset, "stage1_train_dataset")
    if val_dataset is not None:
        assert_dataset_lengths_consistent(accelerator, val_dataset, "stage1_val_dataset")
    train_window_count = getattr(train_dataset, "total_windows", None)
    if train_window_count is None:
        raise ValueError(
            "Stage 1 training dataset is missing `total_windows`; cannot resolve training budget."
        )
    budget = _resolve_training_budget(
        num_epochs=cfg.num_epochs,
        max_steps=int(cfg.max_steps),
        train_samples=int(train_window_count),
        world_size=accelerator.num_processes,
        per_device_batch_size=int(cfg.per_device_batch_size),
        gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        stage_label="Stage 1",
    )
    cfg.max_steps = int(budget.max_steps)
    logger.info(
        "Stage1 training budget: source=%s num_epochs=%s train_samples=%d effective_global_batch=%d max_steps=%d train_epoch_size=%d",
        budget.source,
        cfg.num_epochs,
        budget.train_samples,
        budget.effective_global_batch,
        cfg.max_steps,
        int(cfg.train_epoch_size),
    )
    if accelerator.is_main_process:
        _json_dump(Path(cfg.output_dir) / "stage1_config.json", asdict(cfg))
    train_loader = build_dataloader(train_dataset, cfg, shuffle=True)
    val_loader = (
        build_dataloader(val_dataset, cfg, shuffle=False) if val_dataset is not None else None
    )
    resume_payload = _read_resume_metadata(cfg)
    global_step = int(resume_payload[1]["global_step"]) if resume_payload is not None else 0
    model_dtype = _mixed_precision_to_dtype(mixed_precision)
    backbone = load_stage0_merged_backbone(cfg, device=_resolve_train_device(), dtype=model_dtype)
    projector = FirstFrameToTeacherStateProjector(
        d_student=int(cfg.video_hidden_dim),
        d_teacher=int(cfg.video_hidden_dim),
        projector_dim=int(cfg.projector_dim),
        num_layers=int(cfg.projector_num_layers),
        num_heads=int(cfg.projector_num_heads),
        mlp_ratio=float(cfg.projector_mlp_ratio),
        max_frames=(int(cfg.clip_frames) - 1) // 4 + 1,
    ).to(device=backbone.core.device, dtype=backbone.core.torch_dtype)
    wrapper = DistillationModel(backbone, projector, cfg)
    if resume_payload is not None:
        _load_stage1_model_weights(wrapper, resume_payload[0])
        logger.info(
            "Loaded Stage 1 model weights from %s at global_step=%d", resume_payload[0], global_step
        )
    if bool(cfg.compile_dit):
        _compile_stage1_dit_blocks(wrapper)
    trainable_params = collect_trainable_params(wrapper)
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
        betas=(0.9, 0.95),
    )
    scheduler = build_scheduler(optimizer, cfg)
    (wrapper, optimizer, train_loader, scheduler) = accelerator.prepare(
        wrapper, optimizer, train_loader, scheduler
    )
    trainable_params = optimizer_parameters(optimizer)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    _load_stage1_optimizer_state_if_requested(optimizer=optimizer, scheduler=scheduler, cfg=cfg)
    accelerator.unwrap_model(wrapper).set_stage1_train_mode()
    optimizer.zero_grad(set_to_none=True)
    wandb_run = _init_wandb(accelerator, cfg)
    if int(cfg.max_steps) <= 0:
        logger.info("max_steps<=0; built datasets/model and exiting without training.")
        return
    train_iter = iter(train_loader)
    val_iter = iter(val_loader) if val_loader is not None else None
    start_step = int(global_step)
    run_start_time = time.perf_counter()
    perf_names = ("batch_next", "forward", "backward", "optimizer", "step_wall")
    perf_sums = {name: 0.0 for name in perf_names}
    perf_logged_steps = 0
    logger.info("Starting Stage 1 training at step=%d max_steps=%d", global_step, cfg.max_steps)
    while global_step < int(cfg.max_steps):
        step_wall_start = time.perf_counter()
        batch_next_start = time.perf_counter()
        try:
            sample = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            sample = next(train_iter)
        batch_next_seconds = time.perf_counter() - batch_next_start
        with accelerator.accumulate(wrapper):
            forward_start = time.perf_counter()
            (loss, loss_dict) = wrapper(sample)
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
                local_perf_values = (
                    batch_next_seconds,
                    forward_seconds,
                    backward_seconds,
                    optimizer_seconds,
                    step_wall_seconds,
                )
                for name, value in zip(perf_names, local_perf_values, strict=True):
                    perf_sums[name] += float(value)
                perf_logged_steps += 1
                is_log_step = int(cfg.log_every) > 0 and global_step % int(cfg.log_every) == 0
                if is_log_step:
                    gathered_loss = (
                        accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    gathered_metrics = _gather_scalar_metrics(
                        loss_dict, device=loss.device, gather_fn=accelerator.gather
                    )
                    current_lr = float(optimizer.param_groups[0]["lr"])
                    perf_denominator = max(perf_logged_steps, 1)
                    local_perf = torch.tensor(
                        [perf_sums[name] / perf_denominator for name in perf_names],
                        device=loss.device,
                        dtype=torch.float32,
                    ).reshape(1, -1)
                    gathered_perf = accelerator.gather(local_perf).reshape(-1, len(perf_names))
                    max_rank_perf = gathered_perf.max(dim=0).values.detach().cpu().tolist()
                    perf_avgs = {
                        name: float(value)
                        for (name, value) in zip(perf_names, max_rank_perf, strict=True)
                    }
                    if accelerator.is_main_process:
                        (eta, steps_per_sec) = _format_eta(
                            run_start_time, start_step, global_step, int(cfg.max_steps)
                        )
                        metric_text = " ".join(
                            f"{key}={value:.4f}"
                            for (key, value) in sorted(gathered_metrics.items())
                        )
                        logger.info(
                            "[train] step=%d/%d loss=%.4f %s lr=%.2e grad_norm=%.4f speed=%.3f step/s eta=%s perf_batch_next=%.3fs perf_fwd=%.3fs perf_bwd=%.3fs perf_opt=%.3fs perf_step=%.3fs",
                            global_step,
                            cfg.max_steps,
                            gathered_loss,
                            metric_text,
                            current_lr,
                            float(grad_norm),
                            steps_per_sec,
                            eta,
                            perf_avgs["batch_next"],
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
                            "performance/batch_next_seconds": perf_avgs["batch_next"],
                            "performance/forward_seconds": perf_avgs["forward"],
                            "performance/backward_seconds": perf_avgs["backward"],
                            "performance/optimizer_seconds": perf_avgs["optimizer"],
                            "performance/step_wall_seconds": perf_avgs["step_wall"],
                        }
                        payload.update(
                            {
                                f"train/{key}": float(value)
                                for (key, value) in gathered_metrics.items()
                            }
                        )
                        _wandb_log(wandb_run, payload, global_step)
                    perf_sums = {name: 0.0 for name in perf_names}
                    perf_logged_steps = 0
                if (
                    val_iter is not None
                    and int(cfg.eval_every) > 0
                    and (global_step % int(cfg.eval_every) == 0)
                ):
                    (metrics, val_iter) = evaluate_stage1_batches(
                        accelerator=accelerator,
                        model=wrapper,
                        val_iter=val_iter,
                        val_loader=val_loader,
                        cfg=cfg,
                    )
                    accelerator.wait_for_everyone()
                    if metrics is not None and accelerator.is_main_process:
                        logger.info(
                            "[eval] step=%d val_loss=%.4f batches=%d",
                            global_step,
                            metrics["val_loss"],
                            int(metrics["val_batches"]),
                        )
                        _wandb_log(
                            wandb_run, {f"eval/{k}": v for (k, v) in metrics.items()}, global_step
                        )
                if int(cfg.save_every) > 0 and global_step % int(cfg.save_every) == 0:
                    save_stage1_checkpoint(
                        accelerator=accelerator,
                        model=wrapper,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        cfg=cfg,
                        global_step=global_step,
                    )
    save_stage1_checkpoint(
        accelerator=accelerator,
        model=wrapper,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        global_step=global_step,
    )
    if wandb_run is not None:
        wandb_run.finish()
    logger.info("Stage 1 training finished at step=%d", global_step)
