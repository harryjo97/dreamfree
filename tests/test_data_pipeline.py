"""Exercise public data builders with small local metadata and deterministic pixels."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

from dreamfree.config import load_config
from dreamfree.data.contracts import _stage0_latent_cache_expected_contract
from dreamfree.data.distillation import (
    _build_stage1_dataset,
    format_stage1_prompt,
    stage1_text_cache_path,
)
from dreamfree.data.latent_preparation import EpisodePayload, _save_episode_cache
from dreamfree.data.stage1_cache import _stage1_latent_cache_expected_contract
from dreamfree.data.video import VideoDataset, build_stage0_dataset
from dreamfree.models.wan22.i2v_condition import WAN22_A14B_I2V_CONDITION_LATENTS_KEY

ROOT = Path(__file__).resolve().parents[1]


def write_metadata(root, camera_keys):
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "fps": 30,
                "chunks_size": 1000,
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": {
                    key: {"dtype": "video", "shape": [224, 224, 3]} for key in camera_keys
                },
            }
        )
    )
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "pick up the cup"}) + "\n"
    )
    (meta / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 33, "tasks": ["pick up the cup"]}) + "\n"
    )
    for key in camera_keys:
        path = root / "videos" / "chunk-000" / key / "episode_000000.mp4"
        path.parent.mkdir(parents=True)
        path.touch()


def write_text_cache(directory, cfg):
    directory.mkdir()
    prompt = format_stage1_prompt("pick up the cup")
    path = stage1_text_cache_path(directory, prompt, 128, cfg.model_id)
    mask = torch.zeros(128, dtype=torch.bool)
    mask[:8] = True
    torch.save(
        {
            "format": "dreamfree_text_v1",
            "model_id": cfg.model_id,
            "context": torch.ones(128, 4096),
            "mask": mask,
        },
        path,
    )


@pytest.mark.parametrize("stage", [0, 1])
@pytest.mark.parametrize("dataset", ["robotwin", "libero", "robocasa", "droid"])
def test_live_video_dataset_builds_and_serves_standard_samples(
    tmp_path, monkeypatch, stage, dataset
):
    cfg = load_config(
        stage, ROOT / f"configs/stage{stage}_14b.yaml", [f"dataset={dataset}", "val_fraction=0"]
    )
    root = tmp_path / "dataset"
    write_metadata(root, cfg.camera_keys.split(","))
    write_text_cache(tmp_path / "text", cfg)
    cfg.dataset_dirs = str(root)
    cfg.text_embedding_cache_dir = str(tmp_path / "text")
    # Isolate codec I/O while exercising indexing, camera composition, and text masking.
    monkeypatch.setattr(
        VideoDataset, "_decode_camera", lambda self, *args: torch.full((9, 3, 224, 224), 0.25)
    )
    builder = build_stage0_dataset if stage == 0 else _build_stage1_dataset
    data = builder([root], cfg, split="train")
    sample = data[0]
    assert sample["video"].shape == (3, 9, cfg.height, cfg.width)
    torch.testing.assert_close(sample["video"], torch.full_like(sample["video"], -0.5))
    assert sample["context_mask"].all()
    assert not sample["context"][8:].any()
    assert sample["task"] == "pick up the cup"


@pytest.mark.parametrize("stage", [0, 1])
@pytest.mark.parametrize("model", ["5b", "14b"])
def test_cache_writer_and_training_reader_share_contract(tmp_path, monkeypatch, stage, model):
    cfg = load_config(
        stage,
        ROOT / f"configs/stage{stage}_{model}.yaml",
        ["val_fraction=0", "height=32", "width=32"],
    )
    root = tmp_path / "dataset"
    write_metadata(root, cfg.camera_keys.split(","))
    write_text_cache(tmp_path / "text", cfg)
    cfg.dataset_dirs = str(root)
    cfg.text_embedding_cache_dir = str(tmp_path / "text")
    cfg.use_cached_latents = True
    cache = tmp_path / "latents"
    cache.mkdir()
    if stage == 0:
        cfg.stage0_latent_cache_dir = str(cache)
        contract = _stage0_latent_cache_expected_contract(cfg)
    else:
        cfg.stage2_latent_cache_dir = str(cache)
        contract = _stage1_latent_cache_expected_contract(cfg)
    shape = (
        33,
        contract["latent_channels"],
        contract["latent_t_per_window"],
        contract["latent_h"],
        contract["latent_w"],
    )
    latents = torch.randn(shape, dtype=torch.bfloat16)
    conditioning = latents.clone() if model == "14b" else None
    payload = EpisodePayload(
        episode_index=0,
        num_frames=33,
        video_processed=torch.empty(0),
        action_norm=torch.zeros(33, 14),
        state_norm=torch.zeros(33, 14),
        task_indices=torch.zeros(33, dtype=torch.long),
        concat_layout="robotwin",
        norm_mode="z-score",
    )
    encoder = SimpleNamespace(cache_contract=contract, vae_path="fixture", stats_src="fixture")
    path = cache / "episode_000000.safetensors"
    _save_episode_cache(
        output_path=path,
        latents=latents,
        i2v_condition_latents=conditioning,
        payload=payload,
        encoder=encoder,
        num_frames=33,
        action_video_freq_ratio=4,
    )
    with safe_open(str(path), framework="pt") as handle:
        assert handle.metadata()["cache_version"] == "3"
    monkeypatch.setattr(
        VideoDataset,
        "_decode_camera",
        lambda *args: pytest.fail("Cached video unexpectedly decoded an MP4."),
    )
    builder = build_stage0_dataset if stage == 0 else _build_stage1_dataset
    sample = builder([root], cfg, split="train")[0]
    torch.testing.assert_close(sample["input_latents"], latents[0])
    assert "video" not in sample
    if model == "14b":
        torch.testing.assert_close(sample[WAN22_A14B_I2V_CONDITION_LATENTS_KEY], conditioning[0])
    path.unlink()
    with pytest.raises(FileNotFoundError):
        builder([root], cfg, split="train")


@pytest.mark.parametrize("dataset", ["robotwin", "libero", "robocasa", "droid"])
def test_stage2_live_and_cache_action_streams_match(tmp_path, monkeypatch, dataset):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from dreamfree.config import DATA_DEFAULTS
    from dreamfree.data.lerobot.lerobot import lerobot_dataset
    from dreamfree.data.stage2 import _build_stage2_dataset, _stage2_latent_cache_expected_contract

    cfg = load_config(
        2, ROOT / "configs/stage2_14b.yaml", [f"dataset={dataset}", "val_set_proportion=0"]
    )
    root = tmp_path / "dataset"
    cameras = DATA_DEFAULTS[dataset]["camera_keys"].split(",")
    write_metadata(root, cameras)
    write_text_cache(tmp_path / "text", cfg)
    cfg.dataset_dirs = str(root)
    cfg.text_embedding_cache_dir = str(tmp_path / "text")
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info.update(
        codebase_version="v2.0",
        robot_type="fixture",
        total_episodes=1,
        total_frames=33,
        total_tasks=1,
        total_videos=len(cameras),
        total_chunks=1,
        data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        splits={"train": "0:1"},
    )
    raw = {
        "action": [[0.5] * cfg.action_dim] * 33,
        "observation.state": [[0.25] * cfg.proprio_dim] * 33,
        "timestamp": [i / 30 for i in range(33)],
        "frame_index": list(range(33)),
        "episode_index": [0] * 33,
        "task_index": [0] * 33,
        "index": list(range(33)),
    }
    for name, values in raw.items():
        dim = len(values[0]) if isinstance(values[0], list) else 1
        info["features"][name] = {
            "dtype": "float32" if name in ("action", "observation.state", "timestamp") else "int64",
            "shape": [dim],
            "names": None,
        }
    info_path.write_text(json.dumps(info))
    path = root / "data/chunk-000/episode_000000.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table(raw), path)

    def stats(dim):
        return {
            name: [value] * dim
            for name, value in {
                "min": -1.0,
                "max": 1.0,
                "mean": 0.0,
                "std": 1.0,
                "q01": -1.0,
                "q99": 1.0,
            }.items()
        } | {"count": [33]}

    (root / "meta/stats.json").write_text(
        json.dumps({"action": stats(cfg.action_dim), "observation.state": stats(cfg.proprio_dim)})
    )
    groups = (
        {"joint_position": (0, 7), "gripper_position": (7, 8)}
        if dataset == "droid"
        else {"default": (0, None)}
    )
    if dataset == "droid":
        (root / "meta/modality.json").write_text(
            json.dumps(
                {
                    group: {key: {"start": a, "end": b} for key, (a, b) in groups.items()}
                    for group in ["action", "state"]
                }
            )
        )
    normalization = {}
    for group, dim in [("action", cfg.action_dim), ("state", cfg.proprio_dim)]:
        normalization[group] = {}
        for key, (start, end) in groups.items():
            group_stats = stats((dim if end is None else end) - start)
            normalization[group][key] = {
                f"global_{name}": values for name, values in group_stats.items() if name != "count"
            }
            normalization[group][key].update(
                {
                    f"stepwise_{name}": [values]
                    for name, values in group_stats.items()
                    if name != "count"
                }
            )
    norm_path = tmp_path / "stats.json"
    norm_path.write_text(json.dumps(normalization))
    cfg.pretrained_norm_stats = str(norm_path)
    monkeypatch.setattr(
        lerobot_dataset,
        "decode_video_frames",
        lambda video_path, timestamps, *args, **kwargs: torch.full(
            (
                len(timestamps),
                3,
                180 if dataset == "droid" else 224,
                320 if dataset == "droid" else 224,
            ),
            0.25,
        ),
    )
    live = _build_stage2_dataset(cfg, split="train")
    first, tail = live[0], live[32]
    cache = tmp_path / "cache"
    cache.mkdir()
    cfg.use_cached_latents = True
    cfg.stage2_latent_cache_dir = str(cache)
    contract = _stage2_latent_cache_expected_contract(cfg)
    shape = (
        33,
        contract["latent_channels"],
        contract["latent_t_per_window"],
        contract["latent_h"],
        contract["latent_w"],
    )
    latents = torch.zeros(shape, dtype=torch.bfloat16)
    extra = None
    if dataset == "droid":
        extra = {
            f"droid_{group}_{key}": torch.full((33, b - a), 0.5 if group == "action" else 0.25)
            for group in ["action", "state"]
            for key, (a, b) in groups.items()
        }
    payload = EpisodePayload(
        episode_index=0,
        num_frames=33,
        video_processed=torch.empty(0),
        action_norm=torch.full((33, cfg.action_dim), 0.5),
        state_norm=torch.full((33, cfg.proprio_dim), 0.25),
        task_indices=torch.zeros(33, dtype=torch.long),
        concat_layout=cfg.concat_multi_camera,
        norm_mode="q01/q99"
        if dataset == "droid"
        else "min/max"
        if dataset == "libero"
        else "z-score",
        extra_tensors=extra,
    )
    encoder = SimpleNamespace(cache_contract=contract, vae_path="fixture", stats_src=str(norm_path))
    _save_episode_cache(
        output_path=cache / "episode_000000.safetensors",
        latents=latents,
        i2v_condition_latents=latents.clone(),
        payload=payload,
        encoder=encoder,
        num_frames=33,
        action_video_freq_ratio=4,
    )
    monkeypatch.setattr(
        lerobot_dataset,
        "decode_video_frames",
        lambda *args, **kwargs: pytest.fail("Cached mode decoded video."),
    )
    cached = _build_stage2_dataset(cfg, split="train")
    for idx, expected in [(0, first), (32, tail)]:
        actual = cached[idx]
        for key in ["action", "proprio"]:
            # Padded targets are excluded from the training objective.
            valid = ~expected[f"{key}_is_pad"]
            torch.testing.assert_close(
                actual[key][valid], expected[key][valid], atol=1e-6, rtol=1e-5
            )
        for key in ["action_is_pad", "proprio_is_pad", "context", "context_mask"]:
            torch.testing.assert_close(actual[key], expected[key])
        assert "video" not in actual


def test_text_preparation_writes_readable_cache_and_reuses_it(tmp_path, monkeypatch):
    from dreamfree.data import prepare_text
    from dreamfree.data.text_cache import load_text_context

    instruction_file = tmp_path / "instructions.txt"
    instruction_file.write_text("pick up the cup\n")
    fake_config = SimpleNamespace(path="fixture", download_if_necessary=lambda: None)
    monkeypatch.setattr(
        prepare_text,
        "_resolve_configs",
        lambda *args, **kwargs: (None, fake_config, None, fake_config),
    )

    class Encoder:
        def eval(self):
            return self

        def __call__(self, ids, mask):
            return torch.ones(len(ids), 128, 4096, dtype=torch.bfloat16)

    class Tokenizer:
        def __init__(self, **kwargs):
            pass

        def __call__(self, prompts, **kwargs):
            mask = torch.zeros(len(prompts), 128, dtype=torch.bool)
            mask[:, :8] = True
            return torch.zeros(len(prompts), 128, dtype=torch.long), mask

    monkeypatch.setattr(prepare_text, "_load_registered_model", lambda *args, **kwargs: Encoder())
    monkeypatch.setattr(prepare_text, "HuggingfaceTokenizer", Tokenizer)
    args = [
        "--config",
        str(ROOT / "configs/stage2_14b.yaml"),
        "--set",
        f"text_embedding_cache_dir={tmp_path / 'text'}",
        "--instructions-file",
        str(instruction_file),
        "--device",
        "cpu",
    ]
    prepare_text.main(args)
    context, mask = load_text_context(
        cache_dir=tmp_path / "text",
        prompt=format_stage1_prompt("pick up the cup"),
        context_len=128,
        text_dim=4096,
        model_id="Wan-AI/Wan2.2-I2V-A14B",
    )
    assert mask.all() and context[:8].all() and not context[8:].any()
    monkeypatch.setattr(
        prepare_text,
        "_load_registered_model",
        lambda *args, **kwargs: pytest.fail("Reloaded encoder for an existing cache."),
    )
    prepare_text.main(args)


@pytest.mark.parametrize("stage", [0, 1, 2])
@pytest.mark.parametrize("model", ["5b", "14b"])
@pytest.mark.parametrize("dataset", ["robotwin", "libero", "robocasa", "droid"])
def test_training_validation_accepts_live_data_mode(tmp_path, stage, model, dataset):
    from dreamfree.data.video import validate_stage0_training_config
    from dreamfree.training.stage1.validation import validate_stage1_config
    from dreamfree.training.stage2.validation import validate_dreamfree_stage2_config

    cfg = load_config(stage, ROOT / f"configs/stage{stage}_{model}.yaml", [f"dataset={dataset}"])
    root = tmp_path / "dataset"
    root.mkdir()
    cfg.dataset_dirs = str(root)
    cfg.text_embedding_cache_dir = str(tmp_path / "text")
    write_text_cache(tmp_path / "text", cfg)
    if stage == 1:
        cfg.stage0_lora_checkpoint = str(tmp_path / "stage0")
        # The source-artifact resolver fills this before the training validator runs.
        cfg.stage0_expected_lora_dropout = 0.0
    if stage == 2:
        cfg.stage1_checkpoint = str(tmp_path / "stage1")
        stats = tmp_path / "stats.json"
        stats.write_text("{}")
        cfg.pretrained_norm_stats = str(stats)
    validators = (
        validate_stage0_training_config,
        validate_stage1_config,
        validate_dreamfree_stage2_config,
    )
    validators[stage](cfg)
    assert cfg.use_cached_latents is False
