"""Exercise ABC's actual training builders and reject mismatched experiment contracts."""

import dataclasses
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_abc_eef import MODEL_PATH
from test_abc_eef import converted as abc_converted

from dreamfree.config import load_config
from dreamfree.data import yam
from dreamfree.data.abc_dataset import ABCCachedDataset
from dreamfree.data.action_processing import _build_stage2_processor
from dreamfree.data.distillation import format_stage1_prompt, stage1_text_cache_path
from dreamfree.data.stage2 import _build_stage2_dataset
from dreamfree.data.video import Stage0Episode, VideoDataset

converted = abc_converted

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "stage,space,dim", [(0, "joint", 14), (1, "joint", 14), (2, "joint", 14), (2, "eef", 20)]
)
@pytest.mark.parametrize("cached", [False, True])
def test_explicit_abc_configurations(stage, space, dim, cached):
    overrides = ["data_mode=cached"] if cached else []
    cfg = load_config(stage, ROOT / f"configs/stage{stage}_abc_{space}_14b.json", overrides)
    assert (cfg.height, cfg.width) == (432, 384)
    assert cfg.use_cached_latents is cached
    if stage == 2:
        assert (cfg.action_dim, cfg.proprio_dim, cfg.action_horizon) == (dim, 14, 32)
        assert cfg.student_train_mode == "frozen"


@pytest.mark.parametrize(
    "override",
    [
        "model=5b",
        "height=448",
        "action_dim=7",
        "proprio_dim=20",
        "global_sample_stride=2",
        "student_train_mode=joint",
    ],
)
def test_incompatible_settings_fail(override):
    with pytest.raises(ValueError):
        load_config(2, ROOT / "configs/stage2_abc_joint_14b.json", [override])


@pytest.mark.parametrize("stage", [0, 1])
def test_eef_is_an_action_dataset_only(stage):
    with pytest.raises(ValueError, match="Stage 2"):
        load_config(
            stage, ROOT / f"configs/stage{stage}_abc_joint_14b.json", ["dataset=yam_abc130k_eef"]
        )


def test_stage1_source_must_use_abc_visual_contract():
    cfg = dataclasses.asdict(load_config(1, ROOT / "configs/stage1_abc_joint_14b.json"))
    yam.validate_stage1_source({"config": cfg})
    cfg["dataset_name"] = "robotwin"
    with pytest.raises(ValueError, match="ABC joint-dataset Stage 1"):
        yam.validate_stage1_source({"config": cfg})


def test_v2_video_cache_requires_explicit_abc_preprocessing():
    contract = yam.PRESETS[yam.ABC_NAME].contract
    metadata = {
        **contract,
        "video_value_contract": "uint8_to_float01_before_resize_minus1_to1_v1",
    }
    assert yam.cache_versions({}, metadata) == (3,)
    assert yam.cache_versions({"abc_contract": contract}, metadata) == (2, 3)
    metadata["video_value_contract"] = "uint8_resize"
    with pytest.raises(ValueError, match="video_value_contract"):
        yam.cache_versions({"abc_contract": contract}, metadata)


def write_text(directory, cfg):
    directory.mkdir(exist_ok=True)
    path = stage1_text_cache_path(directory, format_stage1_prompt("move"), 128, cfg.model_id)
    mask = torch.arange(128) < 8
    torch.save(
        dict(
            format="dreamfree_text_v1",
            model_id=cfg.model_id,
            context=torch.ones(128, 4096),
            mask=mask,
        ),
        path,
    )


@pytest.mark.skipif(not MODEL_PATH.is_file(), reason="Set ABC_ROOT for conversion integration.")
@pytest.mark.parametrize("space,dim", [("joint", 14), ("eef", 20)])
def test_fixed_splits_tail_padding_and_real_cache_routing(converted, tmp_path, space, dim):
    joint_root, cache, eef_root, _ = converted
    root = eef_root if space == "eef" else joint_root
    cache = root / "action_cache" if space == "eef" else cache
    cfg = load_config(
        2,
        ROOT / f"configs/stage2_abc_{space}_14b.json",
        [
            "data_mode=cached",
            f"dataset_dirs={root}",
            f"video_latent_cache_dir={cache}",
            f"dataset_stats_path={root / 'dataset_stats.json'}",
            f"text_embedding_cache_dir={tmp_path / 'text'}",
        ],
    )
    write_text(tmp_path / "text", cfg)
    for split, ids in [("train", [0, 1]), ("val", [2])]:
        dataset = _build_stage2_dataset(cfg, split=split)
        assert isinstance(dataset, ABCCachedDataset)
        assert len(dataset) == 36 * len(ids)
        assert [int(record[0].stem.removeprefix("episode_")) for record in dataset.records] == ids
        for index in (0, 35):
            sample = dataset[index]
            assert sample["action"].shape == (32, dim)
            assert sample["proprio"].shape == (32, 14)
            assert sample["input_latents"].shape == (16, 3, 54, 48)
            assert torch.all(sample["input_latents"] == ids[0])
            assert int(sample["action_is_pad"].sum()) == (31 if index == 35 else 0)
            assert torch.equal(sample["action_is_pad"], sample["proprio_is_pad"])
            assert sample["context_mask"].all()
            assert not sample["context"][8:].any()
            assert sample["previous_action"].shape == (cfg.action_history_horizon, dim)
            assert int(sample["previous_action_is_pad"].sum()) == max(
                cfg.action_history_horizon - index, 0
            )
            if index == 35:
                torch.testing.assert_close(
                    sample["action"], sample["action"][:1].expand_as(sample["action"])
                )
    with pytest.raises(ValueError, match="prepared train/val"):
        _build_stage2_dataset(cfg, split="test")
    # An EEF cache must fail when its immutable source changes after conversion.
    if space == "eef":
        original = converted[1] / "episode_000000.safetensors"
        stat = original.stat()
        os.utime(original, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000_000))
        with pytest.raises(ValueError, match="changed"):
            _build_stage2_dataset(cfg, split="train")


@pytest.mark.skipif(not MODEL_PATH.is_file(), reason="Set ABC_ROOT for conversion integration.")
def test_video_stages_keep_prepared_splits_and_provenance(converted):
    root, _, _, _ = converted
    episodes = [Stage0Episode(str(root), i, 36, 30, "move", {}) for i in range(3)]
    reader = object.__new__(VideoDataset)
    reader.cfg = SimpleNamespace(dataset_name=yam.ABC_NAME, val_fraction=0.99)
    assert [e.episode_index for e in reader._split_episodes(episodes, split="train")] == [0, 1]
    assert [e.episode_index for e in reader._split_episodes(episodes, split="val")] == [2]
    episodes[0].length = 35
    with pytest.raises(ValueError, match="frozen selection"):
        reader._split_episodes(episodes, split="train")


@pytest.mark.parametrize("space,dim", [("joint", 14), ("eef", 20)])
def test_processor_quantiles_and_state_dimensions(space, dim):
    from dreamfree.models.policy_loading import (
        _denormalize_action_with_processor,
        _normalize_state_with_processor,
    )

    cfg = load_config(2, ROOT / f"configs/stage2_abc_{space}_14b.json")
    processor = _build_stage2_processor(cfg).eval()
    stats = {
        kind: {"default": {"global_q01": torch.zeros(d), "global_q99": torch.ones(d)}}
        for kind, d in [("action", dim), ("state", 14)]
    }
    processor.set_normalizer_from_stats(stats)
    state = _normalize_state_with_processor(processor, torch.full((14,), 0.25).numpy())
    torch.testing.assert_close(state, torch.full((1, 14), -0.5))
    action = _denormalize_action_with_processor(processor, torch.zeros(1, 32, dim))
    assert action.shape == (1, 32, dim)
    assert (action == 0.5).all()


@pytest.mark.skipif(not MODEL_PATH.is_file(), reason="Set ABC_ROOT for conversion integration.")
@pytest.mark.parametrize("stage", [0, 1])
def test_video_stage_public_builders_read_abc_v3_and_v2_caches(converted, tmp_path, stage):
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from dreamfree.data.distillation import _build_stage1_dataset
    from dreamfree.data.video import build_stage0_dataset

    root, cache, _, _ = converted
    info = json.loads((root / "meta/info.json").read_text())
    info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    for key in yam.CAMERA_KEYS:
        info["features"][key] = {"dtype": "video", "shape": [480, 640, 3]}
    (root / "meta/info.json").write_text(json.dumps(info))
    meta_path = root / "meta/episodes/chunk-000/file-000.parquet"
    rows = pq.read_table(meta_path).to_pylist()
    for row in rows:
        for key in yam.CAMERA_KEYS:
            row[f"videos/{key}/chunk_index"] = 0
            row[f"videos/{key}/file_index"] = row["episode_index"]
            row[f"videos/{key}/from_timestamp"] = 0.0
            path = root / info["video_path"].format(
                video_key=key, chunk_index=0, file_index=row["episode_index"]
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    pq.write_table(pa.Table.from_pylist(rows), meta_path)
    cfg = load_config(
        stage,
        ROOT / f"configs/stage{stage}_abc_joint_14b.json",
        [
            "data_mode=cached",
            f"dataset_dirs={root}",
            f"video_latent_cache_dir={cache}",
            f"text_embedding_cache_dir={tmp_path / 'text'}",
        ],
    )
    write_text(tmp_path / "text", cfg)
    for split, expected in [("train", 2), ("val", 1)]:
        dataset = (
            build_stage0_dataset([root], cfg, split=split)
            if stage == 0
            else _build_stage1_dataset([root], cfg, split=split)
        )
        assert len(dataset.source_dataset.episodes) == expected
        sample = dataset[0]
        assert sample["input_latents"].shape == (16, 3, 54, 48)
        assert sample["context"].shape == (128, 4096)
