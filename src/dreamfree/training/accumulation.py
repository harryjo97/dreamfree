"""Gradient-accumulation policy for step-based DreamFree training loops."""

from accelerate.utils import GradientAccumulationPlugin


def build_fixed_step_gradient_accumulation_plugin(
    num_steps: int, *, sync_each_batch: bool = False
) -> GradientAccumulationPlugin:
    """Build accumulation that synchronizes strictly every ``num_steps`` microbatches.

    Args:
        num_steps: Number of training microbatches per optimizer update.
        sync_each_batch: When true, Accelerate skips its ``no_sync`` context so the
            distributed backend reduces gradients on every microbatch instead of
            only at the accumulation boundary. Under FSDP2 this drops the
            ``unsharded_accumulated_grad`` buffer (one full unsharded gradient copy
            in ``reduce_dtype``) in favour of the sharded ``.grad``, trading extra
            reduce-scatters for a large memory saving. The accumulated gradient is
            mathematically unchanged; only the reduction order differs. The
            optimizer still steps once every ``num_steps`` microbatches.

    Returns:
        An Accelerate plugin independent of prepared dataloader lifecycle state.

    Raises:
        ValueError: If ``num_steps`` is not positive.
    """
    if num_steps <= 0:
        raise ValueError(f"gradient_accumulation_steps must be positive, got {num_steps}.")
    return GradientAccumulationPlugin(
        num_steps=num_steps, sync_with_dataloader=False, sync_each_batch=bool(sync_each_batch)
    )
