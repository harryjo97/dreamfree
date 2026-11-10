from __future__ import annotations

import contextlib
import logging
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import Dataset

from dreamfree.data.action_batch import _maybe_cuda_synchronize
from dreamfree.data.prepare_text import ensure_training_text_cache
from dreamfree.data.stage2 import _build_stage2_dataloader, _build_stage2_dataset
from dreamfree.models.action_training import (
    ActionTrainingModel,
    StudentReadoutActionConditioner,
    _build_action_model,
    _build_student_readout,
    _compile_dreamfree_stage2_dit_blocks,
)
from dreamfree.models.student import load_frozen_stage1_student, validate_stage1_student_source
from dreamfree.training.accumulation import build_fixed_step_gradient_accumulation_plugin
from dreamfree.training.profiling import _build_dreamfree_stage2_profiler
from dreamfree.training.runtime import (
    _format_eta,
    _init_wandb,
    _json_dump,
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
from dreamfree.training.stage2.checkpoint import (
    _load_dreamfree_stage2_optimizer_state_if_requested,
    _read_resume_metadata,
    prepare_release_initialization,
    save_dreamfree_stage2_checkpoint,
    validate_release_initialization,
)
from dreamfree.training.stage2.config import Stage2Config
from dreamfree.training.stage2.constants import logger
from dreamfree.training.stage2.validation import (
    _apply_selected_droid_action_horizon_default,
    _apply_stage1_checkpoint_model_contract,
    _assert_stage2_readout_trainable,
    _load_dreamfree_stage2_model_weights,
    evaluate_dreamfree_stage2_batches,
    validate_dreamfree_stage2_config,
)
from dreamfree.utils.fs import ensure_dir
from dreamfree.utils.logging_config import setup_logging


def run_dreamfree_stage2_training(
    cfg: Stage2Config, *, explicit_fields: set[str] | None = None
) -> None:
    initialization = prepare_release_initialization(cfg)
    if cfg.action_horizon is None:
        _apply_selected_droid_action_horizon_default(cfg, explicit_fields=explicit_fields)
    _apply_stage1_checkpoint_model_contract(cfg, explicit_fields=explicit_fields)
    validate_dreamfree_stage2_config(cfg)
    source = validate_stage1_student_source(cfg)
    if initialization is not None:
        validate_release_initialization(cfg, initialization[1], source=source)
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    accelerator = Accelerator(
        gradient_accumulation_plugin=build_fixed_step_gradient_accumulation_plugin(
            int(cfg.gradient_accumulation_steps)
        ),
        mixed_precision=mixed_precision,
        step_scheduler_with_optimizer=False,
    )
    setup_logging(log_level=logging.INFO, is_main_process=accelerator.is_main_process)
    logger.info(
        "DreamFree Stage2 accelerator: distributed_type=%s world_size=%d process_index=%d mixed_precision=%s",
        accelerator.distributed_type,
        accelerator.num_processes,
        accelerator.process_index,
        mixed_precision,
    )
    logger.info("DreamFree Stage2 source Stage 1 checkpoint: %s", source.checkpoint_dir)
    logger.info(
        "DreamFree Stage2 diagnostic timing: enabled=%s per_device_batch_size=%d num_workers=%d",
        bool(cfg.diagnostic_timing),
        int(cfg.per_device_batch_size),
        int(cfg.num_workers),
    )
    ensure_training_text_cache(cfg, accelerator)
    torch.manual_seed(int(cfg.seed) + accelerator.process_index)
    np.random.seed(int(cfg.seed) + accelerator.process_index)
    random.seed(int(cfg.seed) + accelerator.process_index)
    eval_enabled = int(cfg.eval_every) > 0 and float(cfg.val_fraction) > 0.0
    train_val_fraction = float(cfg.val_fraction) if eval_enabled else 0.0
    train_dataset = _build_stage2_dataset(cfg, split="train", val_fraction=train_val_fraction)
    val_dataset: Dataset | None = None
    if eval_enabled:
        val_dataset = _build_stage2_dataset(cfg, split="val", val_fraction=float(cfg.val_fraction))
    assert_dataset_lengths_consistent(accelerator, train_dataset, "dreamfree_stage2_train_dataset")
    if val_dataset is not None:
        assert_dataset_lengths_consistent(accelerator, val_dataset, "dreamfree_stage2_val_dataset")
    budget = _resolve_training_budget(
        num_epochs=cfg.num_epochs,
        max_steps=int(cfg.max_steps),
        train_samples=len(train_dataset),
        world_size=accelerator.num_processes,
        per_device_batch_size=int(cfg.per_device_batch_size),
        gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        stage_label="DreamFree Stage 2",
    )
    cfg.max_steps = int(budget.max_steps)
    logger.info(
        "DreamFree Stage2 training budget: source=%s num_epochs=%s train_samples=%d effective_global_batch=%d max_steps=%d",
        budget.source,
        cfg.num_epochs,
        budget.train_samples,
        budget.effective_global_batch,
        cfg.max_steps,
    )
    ensure_dir(cfg.output_dir)
    if accelerator.is_main_process:
        _json_dump(Path(cfg.output_dir) / str("config.json"), asdict(cfg))
    train_loader = _build_stage2_dataloader(train_dataset, cfg, shuffle=True)
    val_loader = (
        _build_stage2_dataloader(val_dataset, cfg, shuffle=False)
        if val_dataset is not None
        else None
    )
    resume_payload = _read_resume_metadata(cfg, source=source)
    global_step = int(resume_payload[1]["global_step"]) if resume_payload is not None else 0
    model_dtype = _mixed_precision_to_dtype(mixed_precision)
    student = load_frozen_stage1_student(
        cfg,
        source=source,
        device=_resolve_train_device(),
        dtype=model_dtype,
        merge_student_lora=bool(cfg.merge_student_lora),
    )
    readout = _build_student_readout(cfg, student=student).to(
        device=student.core.device, dtype=student.core.torch_dtype
    )
    action_model = _build_action_model(
        cfg, device=student.core.device, dtype=student.core.torch_dtype
    )
    conditioner = StudentReadoutActionConditioner(
        student_readout_dim=int(cfg.action_hidden_dim),
        action_context_dim=int(cfg.action_hidden_dim),
        proprio_dim=int(cfg.proprio_dim),
        proprio_obs_dropout=float(cfg.proprio_obs_dropout),
    ).to(device=student.core.device, dtype=student.core.torch_dtype)
    wrapper = ActionTrainingModel(
        student=student,
        readout=readout,
        action_model=action_model,
        conditioner=conditioner,
        cfg=cfg,
    )
    if resume_payload is not None:
        _load_dreamfree_stage2_model_weights(wrapper, resume_payload[0])
        logger.info(
            "Loaded DreamFree Stage 2 model weights from %s at global_step=%d",
            resume_payload[0],
            global_step,
        )
    elif initialization is not None:
        _load_dreamfree_stage2_model_weights(wrapper, initialization[0])
        logger.info(
            "Initialized Stage 2 from release %s with a fresh optimizer at step 0",
            initialization[0],
        )
    wrapper.set_stage2_train_mode()
    _assert_stage2_readout_trainable(wrapper)
    if bool(cfg.compile_dit):
        _compile_dreamfree_stage2_dit_blocks(wrapper)
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
    _load_dreamfree_stage2_optimizer_state_if_requested(
        optimizer=optimizer, scheduler=scheduler, cfg=cfg
    )
    accelerator.unwrap_model(wrapper).set_stage2_train_mode()
    optimizer.zero_grad(set_to_none=True)
    wandb_run = _init_wandb(accelerator, cfg)
    if int(cfg.max_steps) <= 0:
        logger.info(
            "max_steps<=0; built DreamFree Stage 2 datasets/model and exiting without training."
        )
        accelerator.end_training()
        return
    train_iter = iter(train_loader)
    start_step = int(global_step)
    run_start_time = time.perf_counter()
    perf_names = ("data_wait", "forward", "backward", "optimizer", "step_wall")
    perf_sums = {name: 0.0 for name in perf_names}
    perf_logged_steps = 0
    diagnostic_timing_enabled = bool(cfg.diagnostic_timing)
    profiler = _build_dreamfree_stage2_profiler(cfg, accelerator)
    if profiler is not None:
        accelerator.unwrap_model(wrapper).profile_sections_enabled = True
        profiler.start()
    logger.info(
        "Starting DreamFree Stage 2 training at step=%d max_steps=%d", global_step, cfg.max_steps
    )
    while global_step < int(cfg.max_steps):
        step_wall_start = time.perf_counter()
        data_start = time.perf_counter()
        try:
            sample = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            sample = next(train_iter)
        data_wait_seconds = time.perf_counter() - data_start
        with accelerator.accumulate(wrapper):
            forward_start = time.perf_counter()
            (loss, loss_dict) = wrapper(sample)
            forward_seconds = time.perf_counter() - forward_start
            backward_start = time.perf_counter()
            if diagnostic_timing_enabled:
                _maybe_cuda_synchronize(loss.device)
            accelerator.backward(loss)
            if diagnostic_timing_enabled:
                _maybe_cuda_synchronize(loss.device)
            backward_seconds = time.perf_counter() - backward_start
            if accelerator.sync_gradients:
                optimizer_start = time.perf_counter()
                if diagnostic_timing_enabled:
                    _maybe_cuda_synchronize(loss.device)
                grad_norm = accelerator.clip_grad_norm_(trainable_params, float(cfg.max_grad_norm))
                optimizer.step()
                if not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if diagnostic_timing_enabled:
                    _maybe_cuda_synchronize(loss.device)
                optimizer_seconds = time.perf_counter() - optimizer_start
                step_wall_seconds = time.perf_counter() - step_wall_start
                global_step += 1
                gathered_loss = accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                step_metrics = dict(loss_dict)
                if diagnostic_timing_enabled:
                    step_metrics["diag_backward_seconds"] = float(backward_seconds)
                    step_metrics["diag_optimizer_step_seconds"] = float(optimizer_seconds)
                gathered_metrics = {}
                for key, value in step_metrics.items():
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
                        (f"{key}={value:.4f}" for (key, value) in sorted(gathered_metrics.items()))
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
                    val_loader is not None
                    and int(cfg.eval_every) > 0
                    and (global_step % int(cfg.eval_every) == 0)
                ):
                    metrics = evaluate_dreamfree_stage2_batches(
                        accelerator=accelerator,
                        model=wrapper,
                        val_loader=val_loader,
                        max_batches=int(cfg.eval_batches),
                    )
                    accelerator.wait_for_everyone()
                    if metrics is not None and accelerator.is_main_process:
                        logger.info(
                            "[eval] step=%d val_loss=%.4f val_batches=%d",
                            global_step,
                            metrics["val_loss"],
                            int(metrics["val_batches"]),
                        )
                        _wandb_log(
                            wandb_run,
                            {f"eval/{key}": value for (key, value) in metrics.items()},
                            global_step,
                        )
                if int(cfg.save_every) > 0 and global_step % int(cfg.save_every) == 0:
                    save_dreamfree_stage2_checkpoint(
                        accelerator=accelerator,
                        model=wrapper,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        cfg=cfg,
                        global_step=global_step,
                        checkpoint_metadata_builder=None,
                        stage_label="DreamFree Stage 2",
                    )
                if profiler is not None:
                    profiler.step()
    if profiler is not None:
        with contextlib.suppress(AssertionError, RuntimeError):
            profiler.stop()
        accelerator.unwrap_model(wrapper).profile_sections_enabled = False
    save_dreamfree_stage2_checkpoint(
        accelerator=accelerator,
        model=wrapper,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        global_step=global_step,
        checkpoint_metadata_builder=None,
        stage_label="DreamFree Stage 2",
    )
    if wandb_run is not None:
        wandb_run.finish()
    logger.info("DreamFree Stage 2 training finished at step=%d", global_step)
    accelerator.end_training()
