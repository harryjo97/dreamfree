"""Prepare task-text embeddings using the same prompt and tokenizer as training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import torch

from dreamfree.config import load_config
from dreamfree.data.distillation import format_stage1_prompt, stage1_text_cache_path
from dreamfree.data.text_cache import validate_text_cache
from dreamfree.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from dreamfree.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from dreamfree.training.runtime import _atomic_torch_save


def collect_prompts(roots):
    prompts = set()
    for root in roots:
        directory = Path(root) / "meta"
        if (directory / "tasks.jsonl").is_file():
            rows = [
                json.loads(line)
                for line in (directory / "tasks.jsonl").read_text().splitlines()
                if line.strip()
            ]
            tasks = [row["task"] for row in rows]
        elif (directory / "tasks.parquet").is_file():
            table = pq.read_table(directory / "tasks.parquet")
            column = "task" if "task" in table.column_names else "__index_level_0__"
            if column not in table.column_names:
                raise ValueError(f"Task text column is missing from {directory / 'tasks.parquet'}.")
            tasks = table[column].to_pylist()
        else:
            raise FileNotFoundError(f"No task metadata in {directory}.")
        for task in tasks:
            if not isinstance(task, str) or not task.strip():
                raise ValueError(f"Expected nonempty task text in {directory}; got {task!r}.")
            prompts.add(format_stage1_prompt(task))
    if not prompts:
        raise ValueError("No task prompts found.")
    return sorted(prompts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="A stage 2 training configuration.")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--instructions-file", help="Additional evaluation instructions, one per line."
    )
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    cfg = load_config(2, args.config, args.set)
    if not cfg.text_embedding_cache_dir:
        raise ValueError("text_embedding_cache_dir must be configured.")
    roots = [part.strip() for part in cfg.dataset_dirs.split(",") if part.strip()]
    prompts = set(collect_prompts(roots)) if roots else set()
    if args.instructions_file:
        prompts.update(
            format_stage1_prompt(line)
            for line in Path(args.instructions_file).read_text().splitlines()
            if line.strip()
        )
    if not prompts:
        raise ValueError("Set dataset_dirs or provide a nonempty --instructions-file.")
    prompts = sorted(prompts)
    output = Path(cfg.text_embedding_cache_dir)
    output.mkdir(parents=True, exist_ok=True)
    pending = []
    for prompt in prompts:
        path = stage1_text_cache_path(output, prompt, cfg.context_len, cfg.model_id)
        if path.exists() and (not args.overwrite):
            payload = torch.load(path, map_location="cpu", weights_only=True)
            validate_text_cache(payload, path, context_len=cfg.context_len, text_dim=cfg.text_dim)
            if payload.get("model_id") != cfg.model_id:
                raise ValueError(
                    f"Incompatible existing text cache: {path}. Use --overwrite to regenerate it."
                )
        else:
            pending.append(prompt)
    if not pending:
        print(f"All {len(prompts)} text embeddings are present in {output}.")
        return
    os.environ["DREAMFREE_MODEL_BASE_PATH"] = cfg.model_base_path
    role = "low_noise" if cfg.video_model_family == "wan22_i2v_a14b" else "single"
    (_, text_cfg, _, tokenizer_cfg) = _resolve_configs(
        cfg.model_id, cfg.tokenizer_model_id, cfg.redirect_common_files, dit_role=role
    )
    text_cfg.download_if_necessary()
    tokenizer_cfg.download_if_necessary()
    encoder = _load_registered_model(
        text_cfg.path, "wan_video_text_encoder", torch.bfloat16, args.device
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_cfg.path, seq_len=cfg.context_len, clean="whitespace"
    )
    with torch.inference_mode():
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            (ids, mask) = tokenizer(batch, return_mask=True, add_special_tokens=True)
            mask = mask.to(device=args.device, dtype=torch.bool)
            context = encoder(ids.to(args.device), mask)
            for index, prompt in enumerate(batch):
                path = stage1_text_cache_path(output, prompt, cfg.context_len, cfg.model_id)
                _atomic_torch_save(
                    path,
                    {
                        "format": "dreamfree_text_v1",
                        "model_id": cfg.model_id,
                        "context": context[index].cpu().contiguous(),
                        "mask": mask[index].cpu().contiguous(),
                    },
                )
    print(f"Prepared {len(pending)} text embeddings in {output}.")


if __name__ == "__main__":
    main()
