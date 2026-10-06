"""Shared torch.compile helpers for DiT block stacks."""

from __future__ import annotations

from typing import Any

from dreamfree.utils.logging_config import get_logger

logger = get_logger(__name__)


def compile_dit_blocks(dit: Any, *, label: str, mode: str | None = None) -> int:
    """Compile DiT block forwards without replacing modules or parameters."""
    import torch._dynamo

    blocks = getattr(dit, "blocks", None)
    if not isinstance(blocks, torch.nn.ModuleList):
        raise TypeError(
            f"{label} compile_dit expected dit.blocks to be a ModuleList, got {type(blocks).__name__}."
        )
    if bool(getattr(dit, "use_gradient_checkpointing", False)):
        raise ValueError(
            f"compile_dit=true is not supported for {label} when the DiT has use_gradient_checkpointing=true."
        )
    torch._dynamo.config.cache_size_limit = max(int(torch._dynamo.config.cache_size_limit), 64)
    for block in blocks:
        compile_kwargs: dict[str, Any] = {"dynamic": False}
        if mode is not None:
            compile_kwargs["mode"] = mode
        block.forward = torch.compile(block.forward, **compile_kwargs)
    logger.info(
        "Compiled %d %s DiT blocks with torch.compile(dynamic=False, mode=%s).",
        len(blocks),
        label,
        mode or "default",
    )
    return len(blocks)
