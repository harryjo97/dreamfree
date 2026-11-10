"""Package the selected AWAM runs for the three DreamFree Hugging Face repos.

Run with AWAM's environment. This writes portable inference bundles supported by
DreamFree's release loader, without optimizer state for resuming training.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch
from huggingface_hub import save_torch_state_dict
from safetensors import safe_open
from safetensors.torch import load_file

RECIPES = {
    "libero": ("libero_5b", "DreamFree-LIBERO"),
    "robocasa": ("robocasa_14b", "DreamFree-RoboCasa"),
    "droid": ("droid_14b", "DreamFree-DROID"),
}
POLICY_FILES = (
    "action.safetensors",
    "readout.safetensors",
    "conditioner.safetensors",
    "proprio_encoder.safetensors",
)
RECIPE_PATH = Path(__file__).resolve().parents[1] / "configs/awam_recipe_sources.json"
MODEL_CARD_ROOT = Path(__file__).resolve().parents[1] / "docs/model_cards"


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def write_json(path, value):
    def convert(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        raise TypeError(f"Unsupported JSON value: {type(obj).__name__}")

    Path(path).write_text(json.dumps(value, indent=2, default=convert) + "\n")


def check_weights(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        if not handle.keys():
            raise ValueError(f"Empty weights: {path}")


def prepare(dataset):
    key, repo_name = RECIPES[dataset]
    if not (MODEL_CARD_ROOT / repo_name / "README.md").is_file():
        raise FileNotFoundError(MODEL_CARD_ROOT / repo_name / "README.md")
    if not (MODEL_CARD_ROOT / "LICENSE.md").is_file():
        raise FileNotFoundError(MODEL_CARD_ROOT / "LICENSE.md")
    recipe = read_json(RECIPE_PATH)[key]
    paths = {stage: Path(recipe[stage]["awam_checkpoint"]) for stage in ("stage1", "stage2")}
    metadata = {stage: read_json(path / "trainer_state.json") for stage, path in paths.items()}
    s1, s2 = metadata["stage1"], metadata["stage2"]
    for stage, state in metadata.items():
        if state["global_step"] != recipe[stage]["selected_step"]:
            raise ValueError(f"{dataset}: {stage} step disagrees with the selected recipe")
    if Path(s2["source_stage1_checkpoint"]).resolve() != paths["stage1"].resolve():
        raise ValueError(f"{dataset}: Stage 2 points to a different Stage 1 checkpoint")
    if s2["source_stage1_global_step"] != s1["global_step"]:
        raise ValueError(f"{dataset}: Stage 2 records a different Stage 1 step")
    if s2["trainable_modules"]["student"]:
        raise ValueError("This exporter requires a frozen Stage 2 student")
    if s1["checkpoint_type"] not in {"fastwam_stage1", "fastwam_stage1_full"}:
        raise ValueError(f"Unsupported Stage 1 type: {s1['checkpoint_type']}")
    full = s1["checkpoint_type"] == "fastwam_stage1_full"
    check_weights(
        paths["stage1"] / ("student_dit.safetensors" if full else "stage1_lora.safetensors")
    )
    if not full:
        from fastwam.models.wan22.stage0_loader import STAGE1_STUDENT_INIT_STAGE0_MERGED

        if s1["student_init_mode"] != STAGE1_STUDENT_INIT_STAGE0_MERGED:
            raise ValueError("Only Stage 0-initialized LoRA students are supported")
        paths["stage0"] = Path(s1["stage0_lora_checkpoint"])
        if paths["stage0"].resolve() != Path(recipe["stage0"]["awam_checkpoint"]).resolve():
            raise ValueError(f"{dataset}: Stage 1 points to a different Stage 0 checkpoint")
        metadata["stage0"] = read_json(paths["stage0"] / "trainer_state.json")
        if metadata["stage0"]["global_step"] != recipe["stage0"]["selected_step"]:
            raise ValueError(f"{dataset}: Stage 0 step disagrees with the selected recipe")
        check_weights(paths["stage0"] / "lora.safetensors")
        base = Path(s1["config"]["model_base_path"]) / s1["config"]["model_id"]
        if s1["config"]["student_dit_role"] != "single":
            base /= s1["config"]["student_dit_role"] + "_model"
        shards = sorted(base.glob("diffusion_pytorch_model*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No pretrained DiT shards in {base}")
        for shard in shards:
            check_weights(shard)
    for name in POLICY_FILES:
        check_weights(paths["stage2"] / name)
    return repo_name, paths, metadata, full


def normalization_stats(stage2):
    # Reuse the exact training conversion, reading metadata rather than videos.
    from vam.shared.stage2_trainer import LeRobotDatasetMetadata, _load_stage2_dataset_stats

    stats_path = stage2["config"].get("pretrained_norm_stats")
    if str(stats_path).strip().lower() in {"none", "null", ""}:
        stats_path = None
    metas = []
    if stats_path is None:
        for directory in stage2["dataset"]["dataset_dirs"]:
            root = Path(directory)
            if not (root / "meta/info.json").is_file():
                raise FileNotFoundError(root / "meta/info.json")
            metas.append(LeRobotDatasetMetadata(repo_id=directory, root=root))
    return _load_stage2_dataset_stats(
        pretrained_norm_stats=stats_path, dataset_name=stage2["dataset"]["name"], metas=metas
    )


def merge_student(paths, metadata):
    from fastwam.models.lora import has_lora_adapters, load_lora_state_dict, merge_lora_adapters
    from fastwam.models.wan22.stage0_loader import load_stage0_merged_backbone

    cfg = SimpleNamespace(**metadata["stage1"]["config"])
    cfg.stage0_lora_checkpoint = str(paths["stage0"])
    cfg.load_vae = False
    cfg.load_text_encoder = False
    cfg.pretrained_load_broadcast = False
    cfg.use_gradient_checkpointing = False
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = cfg.model_base_path
    print("Loading the pretrained DiT and merging Stage 0 on CPU...", flush=True)
    with torch.inference_mode():
        backbone = load_stage0_merged_backbone(cfg, device="cpu", dtype=torch.bfloat16)
        dit = backbone.core.dit
        print("Merging Stage 1 into the Stage 0-adapted DiT...", flush=True)
        load_lora_state_dict(
            dit,
            load_file(str(paths["stage1"] / "stage1_lora.safetensors"), device="cpu"),
            strict=True,
            adapter_name=backbone.student_lora_adapter_name,
        )
        merge_lora_adapters(
            dit,
            adapter_name=backbone.student_lora_adapter_name,
            allow_single_adapter_from_multi=True,
        )
        if has_lora_adapters(dit):
            raise RuntimeError("Student still contains LoRA adapters after merging")
        return dit.state_dict()


def full_student(path):
    # Convert one tensor at a time; do not materialize the full FP32 copy in RAM.
    state = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            if "lora_" in name:
                raise ValueError(f"Full checkpoint unexpectedly contains LoRA weights: {name}")
            tensor = handle.get_tensor(name)
            state[name] = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
    return state


def save_student(state, destination, max_shard_size="5GB"):
    save_torch_state_dict(
        state,
        destination,
        filename_pattern="student_dit{suffix}.safetensors",
        max_shard_size=max_shard_size,
        metadata={"format": "pt"},
    )
    index = destination / "student_dit.safetensors.index.json"
    return index.name if index.exists() else "student_dit.safetensors"


def portable_config(config):
    # Original machine paths remain available in provenance, not runtime config.
    def local_path(value):
        if isinstance(value, str):
            return any(part.startswith("/") for part in value.split(","))
        if isinstance(value, list):
            return any(local_path(item) for item in value)
        return False

    return {key: value for key, value in config.items() if not local_path(value)}


def export(dataset, output_root, prepared, stats):
    from fastwam.models.wan22.stage0_loader import default_stage1_dit_config

    repo_name, paths, metadata, full = prepared
    target = output_root / repo_name
    staging = output_root / ("." + repo_name + ".incomplete")
    if target.exists() or staging.exists():
        raise FileExistsError(
            f"Refusing to overwrite {target} or {staging}; use a fresh --output-root"
        )
    for directory in ("stage1", "stage2", "provenance"):
        (staging / directory).mkdir(parents=True)
    print(f"Exporting {repo_name} to {staging}", flush=True)
    state = (
        full_student(paths["stage1"] / "student_dit.safetensors")
        if full
        else merge_student(paths, metadata)
    )
    weights = save_student(state, staging / "stage1")
    del state
    s1, s2 = metadata["stage1"], metadata["stage2"]
    write_json(
        staging / "stage1/config.json",
        {
            "format": "dreamfree_release_stage1",
            "format_version": 1,
            "weights": weights,
            "weight_dtype": "bfloat16",
            "source_train_mode": "full" if full else "lora",
            "standalone_student": True,
            "requires_stage0": False,
            "global_step": s1["global_step"],
            "architecture": default_stage1_dit_config(
                video_model_family=s1["config"]["video_model_family"]
            ),
            "student": s1["student"],
            "model_contract": s1["model_contract"],
            "config": portable_config(s1["config"]),
        },
    )
    for name in POLICY_FILES:
        shutil.copy2(paths["stage2"] / name, staging / "stage2" / name)
    write_json(staging / "stage2/dataset_stats.json", stats)
    policy = {
        key: s2[key]
        for key in (
            "global_step",
            "model_contract",
            "student",
            "action",
            "readout",
            "conditioner",
            "shapes",
        )
    }
    policy.update(
        {
            "format": "dreamfree_release_stage2",
            "format_version": 1,
            "stage1_checkpoint": "../stage1",
            "dataset_stats": "dataset_stats.json",
            "weights": list(POLICY_FILES),
            "config": portable_config(s2["config"]),
        }
    )
    if "action_history" in s2:
        policy["action_history"] = s2["action_history"]
    write_json(staging / "stage2/config.json", policy)
    for stage, original in metadata.items():
        write_json(staging / "provenance" / f"{stage}_trainer_state.json", original)
    write_json(
        staging / "provenance/sources.json", {stage: str(path) for stage, path in paths.items()}
    )
    shutil.copy2(MODEL_CARD_ROOT / repo_name / "README.md", staging / "README.md")
    shutil.copy2(MODEL_CARD_ROOT / "LICENSE.md", staging / "LICENSE.md")
    write_json(
        staging / "release.json",
        {
            "format": "dreamfree_release",
            "format_version": 1,
            "dataset": dataset,
            "repo_id": f"harryjo97/{repo_name}",
            "stage1": "stage1",
            "stage2": "stage2",
            "files": {
                path.relative_to(staging).as_posix(): path.stat().st_size
                for path in sorted(staging.rglob("*"))
                if path.is_file()
            },
        },
    )
    staging.rename(target)
    print(f"Completed: {target}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*RECIPES, "all"], required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate inputs and normalization without loading model tensors or writing files",
    )
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    if not args.check_only and args.output_root is None:
        parser.error("--output-root is required for export")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    datasets = list(RECIPES) if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        prepared = prepare(dataset)
        stats = normalization_stats(prepared[2]["stage2"])
        print(
            f"Validated {dataset}: selected checkpoints, weight headers, normalization", flush=True
        )
        if not args.check_only:
            export(dataset, args.output_root.expanduser().resolve(), prepared, stats)


if __name__ == "__main__":
    main()
