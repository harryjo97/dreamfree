import os
import random
from typing import Callable, Dict, Optional

import numpy as np
import torch


def _resolve_global_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(
        os.environ.get("RANK", os.environ.get("SLURM_PROCID", os.environ.get("LOCAL_RANK", "0")))
    )


def set_global_seed(seed: int, get_worker_init_fn: bool = False) -> Optional[Callable[[int], None]]:
    """Sets seed for all randomness libraries (mostly random, numpy, torch) and produces a `worker_init_fn`"""
    assert np.iinfo(np.uint32).min < seed < np.iinfo(np.uint32).max, (
        "Seed outside the np.uint32 bounds!"
    )
    os.environ["EXPERIMENT_GLOBAL_SEED"] = str(seed)
    global_rank = _resolve_global_rank()
    process_seed = seed + global_rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)
    return worker_init_function if get_worker_init_fn else None


def worker_init_function(worker_id: int) -> None:
    """
    Borrowed directly from PyTorch-Lightning; inspired by this issue comment in the PyTorch repo:
        > Ref: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562

    Intuition: You can think of the seed sequence spawn function as a "janky" torch.Generator() or jax.PRNGKey that
    you can run iterative splitting on to get new (predictable) randomness.

    :param worker_id: Identifier for the given worker [0, num_workers) for the Dataloader in question.
    """
    process_seed = torch.initial_seed()
    global_rank = _resolve_global_rank()
    base_seed = process_seed - worker_id
    seed_seq = np.random.SeedSequence([base_seed, worker_id, global_rank])
    np.random.seed(seed_seq.generate_state(4))
    (torch_seed_seq, random_seed_seq, tf_seed_seq) = seed_seq.spawn(3)
    torch.manual_seed(torch_seed_seq.generate_state(1, dtype=np.uint64)[0])
    random_seed = (
        random_seed_seq.generate_state(2, dtype=np.uint64).astype(list) * [1 << 64, 1]
    ).sum()
    random.seed(random_seed)


def dict_apply(
    x: Dict[str, torch.Tensor], func: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result
