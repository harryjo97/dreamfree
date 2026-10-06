from __future__ import annotations

from pathlib import Path
from typing import Any

from accelerate import Accelerator

from dreamfree.training.stage2.config import Stage2Config
from dreamfree.training.stage2.constants import logger
from dreamfree.utils.fs import ensure_dir


def _dreamfree_stage2_profiler_trace_handler(cfg: Stage2Config) -> Any:
    """Build the `on_trace_ready` hook that reports the profiled window.

    Args:
        cfg: Run config, used for the trace output location.

    Returns:
        A callable taking the finished profiler and emitting both kernel tables
        and a chrome trace.
    """
    trace_dir = Path(cfg.output_dir) / "profile"

    def handler(profiler: Any) -> None:
        ensure_dir(trace_dir)
        averages = profiler.key_averages()
        total_device_us = sum(float(event.self_device_time_total) for event in averages)
        logger.info(
            "DreamFree Stage 2 profiler: %d distinct ops, total self-device time %.1f ms over the window.",
            len(averages),
            total_device_us / 1000.0,
        )
        logger.info(
            "DreamFree Stage 2 profiler: top ops by self-device time\n%s",
            averages.table(
                sort_by="self_device_time_total", row_limit=45, max_name_column_width=90
            ),
        )
        logger.info(
            "DreamFree Stage 2 profiler: top ops by self-device time, split by input shape\n%s",
            profiler.key_averages(group_by_input_shape=True).table(
                sort_by="self_device_time_total", row_limit=30, max_name_column_width=90
            ),
        )
        trace_path = trace_dir / "dreamfree_stage2_trace.json.gz"
        profiler.export_chrome_trace(str(trace_path))
        logger.info("DreamFree Stage 2 profiler: wrote chrome trace to %s", trace_path)

    return handler


def _build_dreamfree_stage2_profiler(cfg: Stage2Config, accelerator: Accelerator) -> Any | None:
    """Build an optional profiler covering a short window of optimizer steps.

    Only the main process profiles: one rank is enough to attribute kernel time,
    and tracing every rank multiplies the overhead and the trace size.

    Args:
        cfg: Run config carrying `profile_steps` and `profile_start_step`.
        accelerator: Prepared accelerator, used to select the profiling rank.

    Returns:
        A `torch.profiler.profile` awaiting `start()`, or None when disabled or
        when this process is not the main one.

    Raises:
        ValueError: If `profile_start_step` is negative.
    """
    steps = int(cfg.profile_steps)
    if steps <= 0:
        return None
    start_step = int(cfg.profile_start_step)
    if start_step < 0:
        raise ValueError(f"profile_start_step must be >= 0, got {start_step}.")
    if not accelerator.is_main_process:
        return None
    from torch.profiler import ProfilerActivity, profile, schedule

    logger.warning(
        "DreamFree Stage 2 profiler enabled: profiling %d optimizer steps after step %d on rank 0 only. Step timings during the window are inflated by tracing overhead.",
        steps,
        start_step,
    )
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=start_step, warmup=1, active=steps, repeat=1),
        on_trace_ready=_dreamfree_stage2_profiler_trace_handler(cfg),
        record_shapes=True,
        with_stack=False,
        profile_memory=False,
    )
