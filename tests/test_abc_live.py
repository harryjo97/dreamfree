"""Exercise live ABC v3 loading without simulator assets or pretrained models."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from test_abc_training import write_text

from dreamfree.config import load_config
from dreamfree.data import abc_dataset, yam
from dreamfree.data.abc_dataset import ABCCachedDataset, ABCLiveDataset
from dreamfree.data.action_cache import _stage2_latent_cache_expected_contract
from dreamfree.data.latent_data import EpisodePayload
from dreamfree.data.latent_preparation import _save_episode_cache
from dreamfree.data.stage2 import _build_stage2_dataset
from dreamfree.data.yam_eef import MODEL_SHA256

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["joint", "eef"])
def prepared(request, tmp_path):
    cfg = load_config(2, ROOT / f"configs/stage2_abc_{request.param}_14b.json")
    root = tmp_path / "dataset"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    cfg.dataset_dirs = str(root)
    cfg.dataset_stats_path = str(root / "dataset_stats.json")
    cfg.text_embedding_cache_dir = str(tmp_path / "text")
    write_text(tmp_path / "text", cfg)
    info = dict(
        codebase_version="v3.0",
        fps=30,
        data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )
    (root / "meta/info.json").write_text(json.dumps(info))
    stats = {
        kind: {"default": {"global_q01": [-0.2] * dim, "global_q99": [0.8] * dim}}
        for kind, dim in (("action", cfg.action_dim), ("state", cfg.proprio_dim))
    }
    Path(cfg.dataset_stats_path).write_text(json.dumps(stats))
    episodes, rows, arrays = {}, [], []
    for index in range(3):
        action = np.linspace(-0.5, 1.2, 36 * cfg.action_dim, dtype=np.float32).reshape(
            36, cfg.action_dim
        )
        state = np.linspace(-0.3, 1.1, 36 * 14, dtype=np.float32).reshape(36, 14)
        action += index * 0.03
        arrays.append((torch.from_numpy(action), torch.from_numpy(state)))
        table = pa.table(
            {
                "action": action.tolist(),
                "observation.state": state.tolist(),
                "episode_index": [index] * 36,
                "index": np.arange(index * 36, (index + 1) * 36),
                "timestamp": np.arange(36) / 30,
                "task_index": [index] * 36,
            }
        )
        pq.write_table(table, root / f"data/chunk-000/file-{index:03d}.parquet")
        row = dict(
            episode_index=index,
            length=36,
            dataset_from_index=index * 36,
            dataset_to_index=(index + 1) * 36,
            **{"data/chunk_index": 0, "data/file_index": index},
        )
        for camera, key in enumerate(yam.CAMERA_KEYS):
            row.update(
                {
                    f"videos/{key}/chunk_index": 0,
                    f"videos/{key}/file_index": index,
                    f"videos/{key}/from_timestamp": 1.5 + camera,
                }
            )
            path = root / info["video_path"].format(video_key=key, chunk_index=0, file_index=index)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        rows.append(row)
        episodes[str(index)] = dict(
            length=36, split="val" if index == 2 else "train", instruction="move", task_id="0"
        )
    pq.write_table(pa.Table.from_pylist(rows), root / "meta/episodes/chunk-000/file-000.parquet")
    selection = dict(
        schema_version=1,
        fps=30,
        episodes=episodes,
        contract=yam.PRESETS[cfg.dataset_name].contract,
        normalization_sha256=yam.fingerprint(stats),
    )
    if request.param == "eef":
        kinematics = dict(model_sha256=MODEL_SHA256, frame="per_arm_base", site="tcp_site")
        (root / "kinematics.json").write_text(json.dumps(kinematics))
        selection["eef_conversion"] = dict(kinematics_sha256=yam.fingerprint(kinematics))
    selection["selection_sha256"] = yam.fingerprint(selection)
    (root / "yam_selection.json").write_text(json.dumps(selection))
    return cfg, root, arrays


@pytest.mark.parametrize("history", [0, 8])
def test_live_and_cached_actions_splits_padding_and_history_match(prepared, monkeypatch, history):
    cfg, root, arrays = prepared
    cfg.action_history_horizon = history
    cfg.action_history_dropout = 0.2 if history else None

    def decode(video_path, timestamps, tolerance_s, backend):
        camera = yam.CAMERA_KEYS.index(video_path.parts[-3])
        times = torch.tensor(timestamps, dtype=torch.float64) - (1.5 + camera)
        assert times.min() >= -1e-9 and times.max() <= 35 / 30 + 1e-9
        assert backend == "torchcodec"
        pixels = times.float() / 2 + camera * 0.1
        return pixels[:, None, None, None].expand(-1, 3, 480, 640)

    monkeypatch.setattr(abc_dataset, "decode_video_frames", decode)
    # Live mode must work before a latent cache directory even exists.
    live = _build_stage2_dataset(cfg, split="train")
    assert isinstance(live, ABCLiveDataset) and len(live) == 72
    assert len(_build_stage2_dataset(cfg, split="val")) == 36
    samples = {index: live[index] for index in (0, 3, 8, 35, 36)}
    for index, sample in samples.items():
        frame = index % 36
        expected_times = (torch.arange(0, 33, 4) + frame).clamp(max=35).float() / 30
        assert sample["video"].shape == (3, 9, 432, 384)
        assert "input_latents" not in sample
        for camera, (y, x) in enumerate(((0, 0), (300, 0), (300, 200))):
            torch.testing.assert_close(
                sample["video"][0, :, y, x], expected_times + camera * 0.2 - 1
            )
        assert sample["action_is_pad"].sum() == max(0, frame + 32 - 36)
        if history:
            count = min(history, frame)
            assert (
                sample["previous_action_is_pad"].tolist()
                == [True] * (history - count) + [False] * count
            )
            assert not sample["previous_action"][: history - count].any()

    cache = root / "cache"
    cache.mkdir()
    cfg.use_cached_latents = True
    cfg.video_latent_cache_dir = str(cache)
    contract = _stage2_latent_cache_expected_contract(cfg)
    latents = torch.zeros(36, 16, 3, 54, 48, dtype=torch.float16)
    for index, (action, state) in enumerate(arrays):
        # Independent q01/q99 arithmetic checks that live normalization matches the cache.
        payload = EpisodePayload(
            episode_index=index,
            num_frames=36,
            video_processed=torch.empty(0),
            action_norm=(action * 2 - 0.6).clamp(-1, 1),
            state_norm=(state * 2 - 0.6).clamp(-1, 1),
            task_indices=torch.full((36,), index),
            concat_layout=cfg.camera_layout,
            norm_mode="q01/q99",
            source_metadata=yam.episode_metadata(root, index, cfg.dataset_name),
        )
        _save_episode_cache(
            output_path=cache / f"episode_{index:06d}.safetensors",
            latents=latents,
            i2v_condition_latents=latents.clone(),
            payload=payload,
            encoder=SimpleNamespace(
                cache_contract=contract, vae_path="fixture", stats_src=cfg.dataset_stats_path
            ),
            num_frames=33,
            action_video_freq_ratio=4,
        )
    monkeypatch.setattr(
        abc_dataset,
        "decode_video_frames",
        lambda **kwargs: pytest.fail("Cached mode decoded video"),
    )
    cached = _build_stage2_dataset(cfg, split="train")
    assert isinstance(cached, ABCCachedDataset) and len(cached) == len(live)
    for index, expected in samples.items():
        actual = cached[index]
        assert "video" not in actual
        for key, value in expected.items():
            if key == "video":
                continue
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual[key], value)
            else:
                assert actual[key] == value
    with pytest.raises(ValueError, match="prepared train/val"):
        _build_stage2_dataset(cfg, split="test")
    (cache / "episode_000001.safetensors").unlink()
    with pytest.raises(ValueError, match="Incomplete YAM cache"):
        _build_stage2_dataset(cfg, split="train")


@pytest.mark.parametrize("corruption", ["nonfinite", "timestamps", "length", "missing_video"])
def test_live_rejects_invalid_prepared_data(prepared, corruption):
    cfg, root, _ = prepared
    if corruption in {"nonfinite", "timestamps"}:
        path = root / "data/chunk-000/file-000.parquet"
        values = pq.read_table(path).to_pydict()
        if corruption == "nonfinite":
            values["action"][0][0] = float("inf")
        else:
            values["timestamp"][1] = 0
        pq.write_table(pa.table(values), path)
        dataset = _build_stage2_dataset(cfg, split="train")
        with pytest.raises(ValueError, match="finite|timestamps"):
            dataset[0]
    elif corruption == "length":
        path = root / "meta/episodes/chunk-000/file-000.parquet"
        rows = pq.read_table(path).to_pylist()
        rows[0]["length"] -= 1
        pq.write_table(pa.Table.from_pylist(rows), path)
        with pytest.raises(ValueError, match="frozen selection"):
            _build_stage2_dataset(cfg, split="train")
    else:
        next((root / "videos").rglob("file-000.mp4")).unlink()
        with pytest.raises(FileNotFoundError, match="Missing video"):
            _build_stage2_dataset(cfg, split="train")
