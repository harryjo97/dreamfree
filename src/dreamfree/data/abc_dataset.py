"""Live and cached ABC data with immutable episode splits and absolute actions."""

import json
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path

import torch
from safetensors import safe_open
from torch.utils.data import Dataset

from dreamfree.data import yam
from dreamfree.data.lerobot.cached_latent_dataset import validate_stage2_latent_cache_file
from dreamfree.data.lerobot.lerobot.datasets.video_utils import decode_video_frames
from dreamfree.data.text_cache import DEFAULT_PROMPT, load_text_context
from dreamfree.models.wan22.i2v_condition import WAN22_A14B_I2V_CONDITION_LATENTS_KEY


def validate_episode_cache(path, root, dataset_name, expected):
    """Check both the action overlay and its referenced video cache before use."""
    metadata = yam.validate_cache(path, root, dataset_name)
    source = (
        Path(metadata["latent_source_path"])
        if metadata.get("overlay_cache_kind") == yam.EEF_OVERLAY_KIND
        else path
    )
    if source != path:
        expected = {**expected, "abc_contract": yam.PRESETS[yam.ABC_NAME].contract}
    validate_stage2_latent_cache_file(source, expected=expected)
    return metadata, source


class _ABCDataset(Dataset):
    """Index every 30 Hz frame, padding only targets beyond each episode boundary."""

    def __init__(self, cfg, *, split):
        from dreamfree.data.action_processing import _build_stage2_processor

        yam.validate_config(cfg, stage=2)
        from dreamfree.models.action_history import validate_config

        validate_config(cfg)
        if split not in {"train", "val"}:
            raise ValueError(f"ABC supports the prepared train/val splits, got {split!r}")
        roots = [
            Path(part).expanduser().resolve()
            for part in cfg.dataset_dirs.split(",")
            if part.strip()
        ]
        if len(roots) != 1:
            raise ValueError("ABC training requires exactly one prepared dataset root.")
        self.root = roots[0]
        if not cfg.dataset_stats_path:
            raise ValueError("ABC requires dataset_stats_path from the selected training pool.")
        yam.checkpoint_contract(cfg)
        self.cfg, self.split = cfg, split
        self.is_training_set = split == "train"
        self.processor = _build_stage2_processor(cfg)
        from dreamfree.data.lerobot.utils.normalizer import load_dataset_stats_from_json

        self.processor.set_normalizer_from_stats(
            load_dataset_stats_from_json(cfg.dataset_stats_path)
        )
        self.selection = yam.load_selection(self.root, cfg.dataset_name)

    def _index_records(self, records):
        self.records, self.ends = [], []
        total = 0
        for record in records:
            episode = record[2]
            self.records.append(record)
            total += int(episode["length"])
            self.ends.append(total)
        if not total:
            raise ValueError(f"ABC prepared selection has no {self.split} frames.")
        # Fail before model allocation when any requested prompt cache is missing.
        for instruction in {row[2]["instruction"] for row in self.records}:
            self._text(instruction)

    def __len__(self):
        return self.ends[-1]

    @lru_cache(maxsize=128)
    def _text(self, instruction):
        return load_text_context(
            cache_dir=self.cfg.text_embedding_cache_dir,
            prompt=DEFAULT_PROMPT.format(task=instruction),
            context_len=self.cfg.tokenizer_max_len,
            text_dim=self.cfg.text_dim,
            model_id=self.cfg.model_id,
        )

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        record_index = bisect_right(self.ends, index)
        frame = index - (self.ends[record_index - 1] if record_index else 0)
        path, _, episode = self.records[record_index]
        length = int(episode["length"])
        end = min(frame + 32, length)
        horizon = self.cfg.action_history_horizon
        count = min(horizon, frame)
        action, proprio, past, visual = self._read_window(record_index, frame, end, count)
        history = {}
        if horizon:
            previous = torch.zeros((horizon, self.cfg.action_dim), dtype=action.dtype)
            if count:
                previous[-count:] = past
            if not torch.isfinite(previous).all():
                raise ValueError(f"ABC previous actions contain nonfinite values: {path}.")
            history = dict(
                previous_action=previous,
                previous_action_is_pad=torch.arange(horizon) < horizon - count,
            )
        if end - frame < 32:
            action = torch.cat((action, action[-1:].expand(32 - (end - frame), -1)))
            proprio = torch.cat((proprio, proprio[-1:].expand(32 - (end - frame), -1)))
        for label, value in [
            ("action", action),
            ("proprio", proprio),
            *visual.items(),
        ]:
            if not torch.isfinite(value).all():
                raise ValueError(f"ABC {label} contains nonfinite values: {path}, frame {frame}.")
        context, mask = self._text(episode["instruction"])
        pad = torch.arange(frame, frame + 32) >= length
        return dict(
            action=action,
            proprio=proprio,
            prompt=DEFAULT_PROMPT.format(task=episode["instruction"]),
            context=context.clone(),
            context_mask=mask.clone(),
            image_is_pad=torch.arange(0, 33, 4) + frame >= length,
            action_is_pad=pad,
            proprio_is_pad=pad.clone(),
            **visual,
            **history,
        )


class ABCCachedDataset(_ABCDataset):
    """Read full-episode latent caches and optional EEF action overlays."""

    def __init__(self, cfg, *, split):
        from dreamfree.data.action_cache import _stage2_latent_cache_expected_contract

        if not cfg.use_cached_latents:
            raise ValueError("ABCCachedDataset requires data_mode=cached.")
        super().__init__(cfg, split=split)
        if not cfg.video_latent_cache_dir:
            raise ValueError("ABC cached mode requires video_latent_cache_dir.")
        directories = [
            Path(part).expanduser()
            for part in str(cfg.video_latent_cache_dir).split(",")
            if part.strip()
        ]
        selected = yam.select_cache_paths(directories, self.root, split, cfg.dataset_name)
        expected = _stage2_latent_cache_expected_contract(cfg)
        records = []
        for path in sorted(selected):
            metadata, source = validate_episode_cache(path, self.root, cfg.dataset_name, expected)
            episode = self.selection["episodes"][metadata["episode_index"]]
            records.append((path, source, episode))
        self._index_records(records)

    def _read_window(self, record_index, frame, end, count):
        path, source, _ = self.records[record_index]
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            action = handle.get_slice("action")[frame:end]
            proprio = handle.get_slice("proprio")[frame:end]
            past = handle.get_slice("action")[frame - count : frame] if count else None
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            visual = {
                "input_latents": handle.get_slice("latents")[frame],
                WAN22_A14B_I2V_CONDITION_LATENTS_KEY: handle.get_slice(
                    WAN22_A14B_I2V_CONDITION_LATENTS_KEY
                )[frame],
            }
        return action, proprio, past, visual


class ABCLiveDataset(_ABCDataset):
    """Read prepared v3 actions and sparse video windows for online VAE encoding."""

    def __init__(self, cfg, *, split):
        from dreamfree.data.latent_data import _load_lerobot_episode_rows
        from dreamfree.data.stage2 import _require_sparse_stage2_video_backend
        from dreamfree.data.video import VideoDataset
        from dreamfree.data.yam_prepare import data_path

        if cfg.use_cached_latents:
            raise ValueError("ABCLiveDataset requires data_mode=live.")
        super().__init__(cfg, split=split)
        self.video_backend = _require_sparse_stage2_video_backend()
        info = json.loads((self.root / "meta/info.json").read_text())
        if info["codebase_version"] != "v3.0" or info["fps"] != 30:
            raise ValueError("ABC live training requires prepared LeRobot v3 data at 30 Hz.")
        rows = _load_lerobot_episode_rows(self.root)
        if set(map(str, rows)) != set(self.selection["episodes"]):
            raise ValueError("ABC episode metadata differs from its frozen selection.")
        records, self.video_paths = [], []
        for index, row in rows.items():
            episode = self.selection["episodes"][str(index)]
            if int(row["length"]) != int(episode["length"]):
                raise ValueError(f"ABC episode {index} length differs from its frozen selection.")
            if episode["split"] != split:
                continue
            path = data_path(self.root, info, row)
            if not path.is_file():
                raise FileNotFoundError(f"Missing ABC episode parquet: {path}")
            cameras = []
            for key in yam.CAMERA_KEYS:
                video_path = VideoDataset._resolve_video_path(
                    root=self.root,
                    template=info["video_path"],
                    video_key=key,
                    episode_index=index,
                    chunks_size=1000,
                    version="v3.0",
                    row=row,
                )
                offset = VideoDataset._video_start_time_v30(row, key)
                cameras.append((video_path, offset))
            records.append((path, row, episode))
            self.video_paths.append(cameras)
        self._index_records(records)

    @lru_cache(maxsize=4)
    def _episode_values(self, record_index):
        from dreamfree.data.latent_data import _read_episode_parquet_slice

        path, row, episode = self.records[record_index]
        values = _read_episode_parquet_slice(
            path, episode_index=int(row["episode_index"]), episode_row=row
        )
        length = int(episode["length"])
        for key, dim in (("action", self.cfg.action_dim), ("state", self.cfg.proprio_dim)):
            tensor = values[key]
            if tensor.shape != (length, dim) or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid ABC {key}: expected finite [{length}, {dim}] in {path}.")
        times = values["timestamps"]
        expected_times = torch.arange(length, dtype=torch.float64) / 30
        if times.shape != (length,) or not torch.allclose(times, expected_times, atol=1e-6, rtol=0):
            raise ValueError(f"ABC episode timestamps must follow the prepared 30 Hz grid: {path}")
        normalized = self.processor.normalizer.forward(
            {"action": {"default": values["action"]}, "state": {"default": values["state"]}}
        )
        return normalized["action"]["default"], normalized["state"]["default"], times

    def _read_window(self, record_index, frame, end, count):
        action, proprio, times = self._episode_values(record_index)
        # Tail frames repeat the final observation, matching offline cache construction.
        indices = (torch.arange(0, 33, 4) + frame).clamp(max=len(times) - 1)
        views = [
            decode_video_frames(
                video_path=path,
                timestamps=(times[indices] + offset).tolist(),
                tolerance_s=1 / 30 - 0.0001,
                backend=self.video_backend,
            )
            for path, offset in self.video_paths[record_index]
        ]
        video = yam.pack_views(views, self.cfg.dataset_name).mul(2).sub(1)
        if video.shape != (9, 3, self.cfg.height, self.cfg.width):
            raise ValueError(f"ABC live video has an invalid shape: {tuple(video.shape)}")
        return (
            action[frame:end],
            proprio[frame:end],
            action[frame - count : frame],
            {"video": video.permute(1, 0, 2, 3).contiguous()},
        )
