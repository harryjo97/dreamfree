"""Prepare complete per-episode video latents and normalized action/state tensors."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields

from dreamfree.config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="A stage 2 training configuration.")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--encode-batch-size", type=int, default=1)
    parser.add_argument(
        "--limit-episodes",
        type=int,
        default=0,
        help="Bound a smoke run; partial caches cannot be used for full training.",
    )
    args = parser.parse_args(argv)
    if args.encode_batch_size <= 0 or args.limit_episodes < 0:
        parser.error("Batch size must be positive and episode limit must be nonnegative.")
    from dreamfree.data.latent_preparation import PrecomputeConfig, run_precompute

    training = load_config(2, args.config, args.set)
    if not training.dataset_dirs:
        raise ValueError("dataset_dirs is required.")
    allowed = {field.name for field in fields(PrecomputeConfig)}
    cfg = PrecomputeConfig(
        **{key: value for (key, value) in asdict(training).items() if key in allowed}
    )
    runtime = argparse.Namespace(
        cache_root=None,
        cache_dir=args.cache_dir,
        encode_batch_size=args.encode_batch_size,
        limit_episodes=args.limit_episodes,
        compile_vae=False,
        prefetch_episodes=0,
        skip_existing=1,
        reuse_cache_root=None,
        world_size=1,
        rank=0,
        local_rank=None,
        world_size_explicit=False,
        rank_explicit=False,
        local_rank_explicit=False,
    )
    run_precompute(cfg, runtime)


if __name__ == "__main__":
    main()
