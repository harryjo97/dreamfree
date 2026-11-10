"""FSDP2 student sharding with a replicated frozen teacher, adapted from AWAM."""

import torch
from accelerate.utils import DistributedType


def is_fsdp2(accelerator):
    return getattr(accelerator, "distributed_type", None) == DistributedType.FSDP and getattr(
        accelerator, "is_fsdp2", False
    )


def configure(accelerator, model):
    if accelerator.distributed_type != DistributedType.FSDP:
        return
    if not is_fsdp2(accelerator) or model.cfg.student_train_mode != "full":
        raise ValueError("Stage 1 sharding requires FSDP2 with a full student.")
    plugin = accelerator.state.fsdp_plugin
    if "DiTBlock" not in (plugin.transformer_cls_names_to_wrap or []):
        raise ValueError("Stage 1 FSDP2 must wrap DiTBlock transformer blocks.")
    if not isinstance(plugin.reshard_after_forward, bool):
        raise ValueError("Stage 1 FSDP2 requires boolean reshard_after_forward.")
    if any(
        p.requires_grad or p.dtype != torch.bfloat16 for p in model.teacher_core.dit.parameters()
    ):
        raise ValueError("Stage 1 FSDP2 requires a frozen BF16 teacher.")
    plugin.ignored_modules = [model.teacher_core]
    # Prevent Accelerate from upcasting the replicated teacher along with the root.
    model._fsdp2_storage_dtype = torch.float32


def finalize(accelerator, model, optimizer):
    if not is_fsdp2(accelerator):
        return
    from torch.distributed.tensor import DTensor

    raw = accelerator.unwrap_model(model)
    raw.core.dit.to(dtype=torch.float32)
    raw.projector.to(dtype=torch.float32)
    for group in optimizer.param_groups:
        if any(not isinstance(p, DTensor) or p.dtype != torch.float32 for p in group["params"]):
            raise ValueError("FSDP2 failed to shard the trainable student/projector in FP32.")
    if any(
        isinstance(p, DTensor) or p.requires_grad or p.dtype != torch.bfloat16
        for p in raw.teacher_core.dit.parameters()
    ):
        raise ValueError("FSDP2 changed the replicated frozen BF16 teacher.")


def accumulation_policy(accelerator, model, cfg):
    if not is_fsdp2(accelerator):
        return
    model.set_reshard_after_backward(
        bool(accelerator.sync_gradients) or accelerator.state.fsdp_plugin.reshard_after_forward,
        recurse=True,
    )
    parallelism = getattr(accelerator, "parallelism_config", None)
    if (
        cfg.gradient_sync_each_batch
        and parallelism is not None
        and parallelism.dp_replicate_size > 1
    ):
        model.set_requires_all_reduce(bool(accelerator.sync_gradients), recurse=True)


def options(*, full):
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    return StateDictOptions(full_state_dict=full, cpu_offload=True, ignore_frozen_params=full)


def reshard_for_checkpoint(model):
    from torch.distributed.fsdp import FSDPModule

    # Validation can leave groups unsharded when no backward follows the forward.
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


def portable_states(accelerator, model):
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    state = get_model_state_dict(model, options=options(full=True))
    if not accelerator.is_main_process:
        return None
    raw = accelerator.unwrap_model(model)
    result = []
    for prefix, expected in (
        ("core.dit.", raw._student_state_keys),
        ("projector.", raw._projector_state_keys),
    ):
        selected = {
            k.removeprefix(prefix): v.detach().cpu().contiguous()
            for k, v in state.items()
            if k.startswith(prefix)
        }
        if set(selected) != set(expected):
            raise ValueError(f"FSDP2 portable checkpoint has incomplete {prefix} weights.")
        result.append(selected)
    return tuple(result)


def load_model(accelerator, model, directory):
    from safetensors.torch import load_file
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    state = {}
    raw = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        for filename, prefix, expected in (
            ("student_dit.safetensors", "core.dit.", raw._student_state_keys),
            ("projector.safetensors", "projector.", raw._projector_state_keys),
        ):
            weights = load_file(str(directory / filename), device="cpu")
            if set(weights) != set(expected):
                raise ValueError(f"FSDP2 resume has incomplete {filename} weights.")
            state.update({prefix + k: v for k, v in weights.items()})
    incompatible = set_model_state_dict(
        model,
        state,
        options=StateDictOptions(
            full_state_dict=True, cpu_offload=True, strict=False, broadcast_from_rank0=True
        ),
    )
    missing = [k for k in incompatible.missing_keys if k.startswith(("core.dit.", "projector."))]
    if missing or incompatible.unexpected_keys:
        raise ValueError(
            f"FSDP2 resume weight mismatch: {missing}, {incompatible.unexpected_keys}."
        )


def optimizer_state(model, optimizer, scheduler, directory, *, load=False):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )

    optimizer = getattr(optimizer, "optimizer", optimizer)
    opts = options(full=False)
    state = {
        "optimizer": get_optimizer_state_dict(model, optimizer, options=opts),
        "scheduler": scheduler.state_dict(),
    }
    if load:
        dcp.load(state, checkpoint_id=str(directory))
        set_optimizer_state_dict(model, optimizer, state["optimizer"], options=opts)
        scheduler.load_state_dict(state["scheduler"])
    else:
        dcp.save(state, checkpoint_id=str(directory))
