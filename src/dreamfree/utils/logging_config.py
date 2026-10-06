import logging
import os
from typing import Optional

import torch.distributed as dist
from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    log_level: int = logging.INFO,
    is_main_process: Optional[bool] = None,
    rich_handler_kwargs: Optional[dict] = None,
    formatter_kwargs: Optional[dict] = None,
    preserve_hydra_handlers: bool = True,
) -> None:
    """
    Configure the logging system for the entire codebase with Rich formatting.

    In distributed training, only the main process outputs logs while other processes are silenced.
    This function configures the root logger so all child loggers inherit the same configuration.

    Args:
        log_level: Logging level (default INFO), only applies to main process
        is_main_process: Whether this is the main process. If None, infer automatically.
        rich_handler_kwargs: Additional kwargs to pass to RichHandler
        formatter_kwargs: Additional kwargs to pass to Formatter (fmt and datefmt)
        preserve_hydra_handlers: Keep existing FileHandlers from Hydra (default True)

    Example:
        ```python
        # In a single-machine script
        from dreamfree.utils.logging_config import setup_logging
        setup_logging()

        # In a distributed training script
        from accelerate import PartialState
        from dreamfree.utils.logging_config import setup_logging

        distributed_state = PartialState()
        setup_logging(
            log_level=logging.INFO,
            is_main_process=distributed_state.is_main_process
        )
        ```
    """
    if is_main_process is None:
        is_main_process = _is_main_process()
    root_logger = logging.getLogger()
    if is_main_process:
        existing_file_handlers = []
        if preserve_hydra_handlers:
            existing_file_handlers = [
                h for h in root_logger.handlers if isinstance(h, logging.FileHandler)
            ]
        root_logger.handlers.clear()
        default_rich_kwargs = {
            "console": Console(no_color=True),
            "markup": True,
            "rich_tracebacks": True,
            "show_level": True,
            "show_path": True,
            "show_time": True,
        }
        if rich_handler_kwargs:
            default_rich_kwargs.update(rich_handler_kwargs)
        rich_handler = RichHandler(**default_rich_kwargs)
        default_formatter_kwargs = {"fmt": "| >> %(message)s", "datefmt": "%m/%d [%H:%M:%S]"}
        if formatter_kwargs:
            default_formatter_kwargs.update(formatter_kwargs)
        formatter = logging.Formatter(**default_formatter_kwargs)
        rich_handler.setFormatter(formatter)
        root_logger.addHandler(rich_handler)
        for handler in existing_file_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(log_level)
    else:
        root_logger.setLevel(logging.ERROR)


def _is_main_process() -> bool:
    """
    Best-effort check for main process without any synchronization.
    """
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    for key in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        if key in os.environ:
            return os.environ.get(key, "0") in ("0", "0\n", "")
    return True


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Drop-in replacement for accelerate.logging.get_logger:
    - No implicit barriers.
    - Only the main process emits log records by default.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not _is_main_process():
        logger.propagate = False
        logger.disabled = True
    return logger
