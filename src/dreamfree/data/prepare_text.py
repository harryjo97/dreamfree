"""Prepare task-text embeddings using the same prompt and tokenizer as training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import torch
from accelerate.utils import broadcast_object_list

from dreamfree.config import load_config
from dreamfree.data import yam
from dreamfree.data.distillation import format_stage1_prompt, stage1_text_cache_path
from dreamfree.data.text_cache import validate_text_cache
from dreamfree.data.video import _task_candidates_from_value
from dreamfree.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from dreamfree.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from dreamfree.training.runtime import _atomic_torch_save


def collect_prompts(roots, *, cfg=None):
    if getattr(cfg, "prompt_policy", "task_text") == "generic":
        return [format_stage1_prompt("a robot manipulation task")]
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
        episodes = directory / "episodes.jsonl"
        if episodes.is_file():
            for line in episodes.read_text().splitlines():
                if line.strip():
                    tasks.extend(_task_candidates_from_value(json.loads(line).get("tasks")))
        else:
            for path in sorted((directory / "episodes").rglob("*.parquet")):
                if "tasks" in pq.read_schema(path).names:
                    for value in pq.read_table(path, columns=["tasks"])["tasks"].to_pylist():
                        tasks.extend(_task_candidates_from_value(value))
        if getattr(cfg, "dataset_name", None) in yam.DATASET_NAMES:
            selection = yam.load_selection(root, cfg.dataset_name)
            tasks.extend(episode["instruction"] for episode in selection["episodes"].values())
        for task in tasks:
            if not isinstance(task, str) or not task.strip():
                raise ValueError(f"Expected nonempty task text in {directory}; got {task!r}.")
            prompts.add(format_stage1_prompt(task))
    if not prompts:
        raise ValueError("No task prompts found.")
    return sorted(prompts)


def prepare_text_embeddings(cfg, prompts, *, device, batch_size=8, overwrite=False):
    """Reuse valid prompt caches and encode missing entries into the first cache directory."""
    if batch_size <= 0:
        raise ValueError("Text preparation batch_size must be positive.")
    context_len = int(cfg.tokenizer_max_len)
    if (context_len, cfg.text_dim) != (128, 4096):
        raise ValueError("Text preparation requires context length 128 and text_dim=4096.")
    if not prompts:
        raise ValueError("Set dataset_dirs or provide a nonempty --instructions-file.")
    prompts = sorted(set(prompts))
    directories = [
        Path(part.strip()).expanduser()
        for part in str(cfg.text_embedding_cache_dir).split(",")
        if part.strip()
    ]
    if not directories:
        raise ValueError("text_embedding_cache_dir must name a writable cache directory.")
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    output = directories[0]
    pending = []
    for prompt in prompts:
        candidates = [
            stage1_text_cache_path(directory, prompt, context_len, cfg.model_id)
            for directory in directories
        ]
        path = next((path for path in candidates if path.is_file()), None)
        if path is not None and not overwrite:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            validate_text_cache(payload, path, context_len=context_len, text_dim=cfg.text_dim)
            if payload.get("model_id") != cfg.model_id:
                raise ValueError(
                    f"Incompatible existing text cache: {path}. Use --overwrite to regenerate it."
                )
        else:
            pending.append(prompt)
    if not pending:
        print(f"All {len(prompts)} text embeddings are present in {directories}.", flush=True)
        return 0
    print(f"Building {len(pending)} missing text embeddings in {output}.", flush=True)
    os.environ["DREAMFREE_MODEL_BASE_PATH"] = cfg.model_base_path
    role = "low_noise" if cfg.video_model_family == "wan22_i2v_a14b" else "single"
    (_, text_cfg, _, tokenizer_cfg) = _resolve_configs(
        cfg.model_id, cfg.tokenizer_model_id, cfg.redirect_common_files, dit_role=role
    )
    text_cfg.download_if_necessary()
    tokenizer_cfg.download_if_necessary()
    encoder = _load_registered_model(
        text_cfg.path,
        "wan_video_text_encoder",
        torch.bfloat16,
        device,
        pretrained_load_broadcast=False,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_cfg.path, seq_len=context_len, clean="whitespace"
    )
    with torch.inference_mode():
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            (ids, mask) = tokenizer(batch, return_mask=True, add_special_tokens=True)
            mask = mask.to(device=device, dtype=torch.bool)
            context = encoder(ids.to(device), mask)
            for index, prompt in enumerate(batch):
                path = stage1_text_cache_path(output, prompt, context_len, cfg.model_id)
                payload = {
                    "format": "dreamfree_text_v1",
                    "model_id": cfg.model_id,
                    "context": context[index].cpu().contiguous(),
                    "mask": mask[index].cpu().contiguous(),
                }
                validate_text_cache(payload, path, context_len=context_len, text_dim=cfg.text_dim)
                _atomic_torch_save(path, payload)
    print(f"Prepared {len(pending)} text embeddings in {output}.", flush=True)
    return len(pending)


def ensure_training_text_cache(cfg, accelerator):
    """Prepare text once on rank 0 and propagate preparation failures to every rank."""
    if not getattr(cfg, "use_text_embedding_cache", True):
        return
    error = None
    status = [None]
    if accelerator.is_main_process:
        try:
            raw_roots = cfg.dataset_dirs or getattr(cfg, "dataset_root", "")
            roots = [part.strip() for part in raw_roots.split(",") if part.strip()]
            prompts = collect_prompts(roots, cfg=cfg)
            prepare_text_embeddings(cfg, prompts, device=accelerator.device)
            # Release the temporary text encoder's allocation before loading the backbone.
            if accelerator.device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:
            error = exc
            status[0] = f"{type(exc).__name__}: {exc}"
    if accelerator.num_processes > 1:
        broadcast_object_list(status, from_process=0)
    if status[0] is not None:
        raise RuntimeError(f"Automatic text-cache preparation failed: {status[0]}") from error


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
    roots = [part.strip() for part in cfg.dataset_dirs.split(",") if part.strip()]
    prompts = set(collect_prompts(roots, cfg=cfg)) if roots else set()
    if args.instructions_file:
        prompts.update(
            format_stage1_prompt(line)
            for line in Path(args.instructions_file).read_text().splitlines()
            if line.strip()
        )
    prepare_text_embeddings(
        cfg, prompts, device=args.device, batch_size=args.batch_size, overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
