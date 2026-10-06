#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import contextlib
import logging
import shutil
import traceback
import warnings
from pathlib import Path
from typing import Callable, List, Literal

import datasets
import numpy as np
import packaging.version
import pyarrow.parquet as pq
import torch
import torch.utils
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import REPOCARD_NAME
from huggingface_hub.errors import RevisionNotFoundError

from dreamfree.data.lerobot.constants import HF_LEROBOT_HOME
from dreamfree.data.lerobot.lerobot.datasets.compute_stats import (
    aggregate_stats,
    compute_episode_stats,
)
from dreamfree.data.lerobot.lerobot.datasets.utils import (
    DEFAULT_FEATURES,
    DEFAULT_IMAGE_PATH,
    INFO_PATH,
    TASKS_PATH,
    _validate_feature_names,
    append_jsonlines,
    backward_compatible_episodes_stats,
    check_timestamps_sync,
    create_empty_dataset_info,
    create_lerobot_dataset_card,
    embed_images,
    get_delta_indices,
    get_episode_data_index,
    get_hf_features_from_features,
    hf_transform_to_torch,
    load_annotations,
    load_episodes,
    load_episodes_stats,
    load_info,
    load_stats,
    load_tasks,
    validate_episode_buffer,
    validate_frame,
    write_episode,
    write_episode_stats,
    write_info,
    write_json,
)
from dreamfree.data.lerobot.lerobot.datasets.video_utils import (
    VideoFrame,
    decode_video_frames,
    encode_video_frames,
    get_safe_default_codec,
    get_video_info,
)

CODEBASE_VERSION = "v2.1"


def _stats_with_count_from_total_frames(
    stats: dict[str, dict], *, total_frames: int, root: Path
) -> dict[str, dict]:
    if not stats:
        return stats
    frame_count = int(total_frames)
    if frame_count <= 0:
        missing = [key for (key, value) in stats.items() if "count" not in value]
        if missing:
            raise ValueError(
                f"LeRobot stats for root={root} are missing 'count' for {missing}, and meta/info.json does not provide a positive total_frames value."
            )
        return stats
    normalized: dict[str, dict] = {}
    for key, value in stats.items():
        feature_stats = dict(value)
        if "count" not in feature_stats:
            feature_stats["count"] = np.asarray([frame_count], dtype=np.int64)
        normalized[key] = feature_stats
    return normalized


class LeRobotDatasetMetadata:
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
    ):
        self.repo_id = repo_id
        self.revision = revision if revision else CODEBASE_VERSION
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            self.pull_from_repo(allow_patterns="meta/")
            self.load_metadata()

    def load_metadata(self):
        self.info = load_info(self.root)
        (self.tasks, self.task_to_task_index) = load_tasks(self.root)
        if (self.root / "annotations").exists():
            self.annotations = load_annotations(self.root)
        self.episodes = load_episodes(self.root)
        if self._version < packaging.version.parse("v2.1"):
            self.stats = load_stats(self.root)
            self.episodes_stats = backward_compatible_episodes_stats(self.stats, self.episodes)
        else:
            self.episodes_stats = load_episodes_stats(self.root)
            self.stats = aggregate_stats(list(self.episodes_stats.values()))

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    @property
    def _version(self) -> packaging.version.Version:
        """Codebase version used to create this dataset."""
        return packaging.version.parse(self.info["codebase_version"])

    def get_data_file_path(self, ep_index: int) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.data_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.video_path.format(
            episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_index
        )
        return Path(fpath)

    def get_episode_chunk(self, ep_index: int) -> int:
        return ep_index // self.chunks_size

    @property
    def data_path(self) -> str:
        """Formattable string for the parquet files."""
        return self.info["data_path"]

    @property
    def video_path(self) -> str | None:
        """Formattable string for the video files."""
        return self.info["video_path"]

    @property
    def robot_type(self) -> str | None:
        """Robot type used in recording this dataset."""
        return self.info["robot_type"]

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.info["fps"]

    @property
    def features(self) -> dict[str, dict]:
        """All features contained in the dataset."""
        return self.info["features"]

    @property
    def image_keys(self) -> list[str]:
        """Keys to access visual modalities stored as images."""
        return [key for (key, ft) in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        """Keys to access visual modalities stored as videos."""
        return [key for (key, ft) in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access visual modalities (regardless of their storage method)."""
        return [key for (key, ft) in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def names(self) -> dict[str, list | dict]:
        """Names of the various dimensions of vector modalities."""
        return {key: ft["names"] for (key, ft) in self.features.items()}

    @property
    def shapes(self) -> dict:
        """Shapes for the different features."""
        return {key: tuple(ft["shape"]) for (key, ft) in self.features.items()}

    @property
    def total_episodes(self) -> int:
        """Total number of episodes available."""
        return self.info["total_episodes"]

    @property
    def total_frames(self) -> int:
        """Total number of frames saved in this dataset."""
        return self.info["total_frames"]

    @property
    def total_tasks(self) -> int:
        """Total number of different tasks performed in this dataset."""
        return self.info["total_tasks"]

    @property
    def total_chunks(self) -> int:
        """Total number of chunks (groups of episodes)."""
        return self.info["total_chunks"]

    @property
    def chunks_size(self) -> int:
        """Max number of episodes per chunk."""
        return self.info["chunks_size"]

    def get_task_index(self, task: str) -> int | None:
        """
        Given a task in natural language, returns its task_index if the task already exists in the dataset,
        otherwise return None.
        """
        return self.task_to_task_index.get(task, None)

    def add_task(self, task: str):
        """
        Given a task in natural language, add it to the dictionary of tasks.
        """
        if task in self.task_to_task_index:
            raise ValueError(f"The task '{task}' already exists and can't be added twice.")
        task_index = self.info["total_tasks"]
        self.task_to_task_index[task] = task_index
        self.tasks[task_index] = task
        self.info["total_tasks"] += 1
        task_dict = {"task_index": task_index, "task": task}
        append_jsonlines(task_dict, self.root / TASKS_PATH)

    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
        raw_file_name: str | None = None,
    ) -> None:
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length
        chunk = self.get_episode_chunk(episode_index)
        if chunk >= self.total_chunks:
            self.info["total_chunks"] += 1
        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        self.info["total_videos"] += len(self.video_keys)
        if len(self.video_keys) > 0:
            self.update_video_info()
        write_info(self.info, self.root)
        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        if raw_file_name is not None:
            episode_dict["raw_file_name"] = raw_file_name
        self.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.root)
        self.episodes_stats[episode_index] = episode_stats
        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats else episode_stats
        write_episode_stats(episode_index, episode_stats, self.root)

    def update_video_info(self) -> None:
        """
        Warning: this function writes info from first episode videos, implicitly assuming that all videos have
        been encoded the same way. Also, this means it assumes the first episode exists.
        """
        for key in self.video_keys:
            if not self.features[key].get("info", None):
                video_path = self.root / self.get_video_file_path(ep_index=0, vid_key=key)
                self.info["features"][key]["info"] = get_video_info(video_path)

    def __repr__(self):
        feature_keys = list(self.features)
        return f"{self.__class__.__name__}({{\n    Repository ID: '{self.repo_id}',\n    Total episodes: '{self.total_episodes}',\n    Total frames: '{self.total_frames}',\n    Features: '{feature_keys}',\n}})',\n"

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        robot_type: str | None = None,
        root: str | Path | None = None,
        use_videos: bool = True,
    ) -> "LeRobotDatasetMetadata":
        """Creates metadata for a LeRobotDataset."""
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
        obj.root.mkdir(parents=True, exist_ok=False)
        features = {**features, **DEFAULT_FEATURES}
        _validate_feature_names(features)
        (obj.tasks, obj.task_to_task_index) = ({}, {})
        (obj.episodes_stats, obj.stats, obj.episodes) = ({}, {}, {})
        obj.info = create_empty_dataset_info(
            CODEBASE_VERSION, fps, features, use_videos, robot_type
        )
        if len(obj.video_keys) > 0 and (not use_videos):
            raise ValueError()
        write_json(obj.info, obj.root / INFO_PATH)
        obj.revision = None
        return obj


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 0.0001,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        video_codec: Literal["h264", "hevc", "libsvtav1", "h264_nvenc"] = "libsvtav1",
        is_compute_episode_stats_image: bool = True,
        video_root: str | Path | None = None,
    ):
        """
        2 modes are available for instantiating this class, depending on 2 different use cases:

        1. Your dataset already exists:
            - On your local disk in the 'root' folder. This is typically the case when you recorded your
              dataset locally and you may or may not have pushed it to the hub yet. Instantiating this class
              with 'root' will load your dataset directly from disk. This can happen while you're offline (no
              internet connection).

            - On the Hugging Face Hub at the address https://huggingface.co/datasets/{repo_id} and not on
              your local disk in the 'root' folder. Instantiating this class with this 'repo_id' will download
              the dataset from that address and load it, pending your dataset is compliant with
              codebase_version v2.0. If your dataset has been created before this new format, you will be
              prompted to convert it using our conversion script from v1.6 to v2.0, which you can find at
              lerobot/datasets/v2/convert_dataset_v1_to_v2.py.


        2. Your dataset doesn't already exists (either on local disk or on the Hub): you can create an empty
           LeRobotDataset with the 'create' classmethod. This can be used for recording a dataset or port an
           existing dataset to the LeRobotDataset format.


        In terms of files, LeRobotDataset encapsulates 3 main things:
            - metadata:
                - info contains various information about the dataset like shapes, keys, fps etc.
                - stats stores the dataset statistics of the different modalities for normalization
                - tasks contains the prompts for each task of the dataset, which can be used for
                  task-conditioned training.
            - hf_dataset (from datasets.Dataset), which will read any values from parquet files.
            - videos (optional) from which frames are loaded to be synchronous with data from parquet files.

        A typical LeRobotDataset looks like this from its root path:
        .
        ├── data
        │   ├── chunk-000
        │   │   ├── episode_000000.parquet
        │   │   ├── episode_000001.parquet
        │   │   ├── episode_000002.parquet
        │   │   └── ...
        │   ├── chunk-001
        │   │   ├── episode_001000.parquet
        │   │   ├── episode_001001.parquet
        │   │   ├── episode_001002.parquet
        │   │   └── ...
        │   └── ...
        ├── meta
        │   ├── episodes.jsonl
        │   ├── info.json
        │   ├── stats.json
        │   └── tasks.jsonl
        └── videos
            ├── chunk-000
            │   ├── observation.images.laptop
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            │   ├── observation.images.phone
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            ├── chunk-001
            └── ...

        Note that this file-based structure is designed to be as versatile as possible. The files are split by
        episodes which allows a more granular control over which episodes one wants to use and download. The
        structure of the dataset is entirely described in the info.json file, which can be easily downloaded
        or viewed directly on the hub before downloading any actual data. The type of files used are very
        simple and do not need complex tools to be read, it only uses .parquet, .json and .mp4 files (and .md
        for the README).

        Args:
            repo_id (str): This is the repo id that will be used to fetch the dataset. Locally, the dataset
                will be stored under root/repo_id.
            root (Path | None, optional): Local directory to use for downloading/writing files. You can also
                set the LEROBOT_HOME environment variable to point to a different location. Defaults to
                '~/.cache/huggingface/lerobot'.
            episodes (list[int] | None, optional): If specified, this will only load episodes specified by
                their episode_index in this list. Defaults to None.
            image_transforms (Callable | None, optional): You can pass standard v2 image transforms from
                torchvision.transforms.v2 here which will be applied to visual modalities (whether they come
                from videos or images). Defaults to None.
            delta_timestamps (dict[list[float]] | None, optional): _description_. Defaults to None.
            tolerance_s (float, optional): Tolerance in seconds used to ensure data timestamps are actually in
                sync with the fps value. It is used at the init of the dataset to make sure that each
                timestamps is separated to the next by 1/fps +/- tolerance_s. This also applies to frames
                decoded from video files. It is also used to check that `delta_timestamps` (when provided) are
                multiples of 1/fps. Defaults to 1e-4.
            revision (str, optional): An optional Git revision id which can be a branch name, a tag, or a
                commit hash. Defaults to current codebase version tag.
            sync_cache_first (bool, optional): Flag to sync and refresh local files first. If True and files
                are already present in the local cache, this will be faster. However, files loaded might not
                be in sync with the version on the hub, especially if you specified 'revision'. Defaults to
                False.
            download_videos (bool, optional): Flag to download the videos. Note that when set to True but the
                video files are already present on local disk, they won't be downloaded again. Defaults to
                True.
            video_backend (str | None, optional): Video backend to use for decoding videos. Defaults to torchcodec when available int the platform; otherwise, defaults to 'pyav'.
                You can also use the 'pyav' decoder used by Torchvision, which used to be the default option, or 'video_reader' which is another decoder of Torchvision.
        """
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.video_root = Path(video_root) if video_root else self.root
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.video_codec = video_codec
        self.is_compute_episode_stats_image = is_compute_episode_stats_image
        self.delta_indices = None
        self.during_training = True
        self.uses_external_video_root = self.video_root != self.root
        self.image_writer = None
        self.episode_buffer = None
        self.root.mkdir(exist_ok=True, parents=True)
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)
        try:
            if force_cache_sync:
                raise FileNotFoundError
            assert all(
                (self.root / fpath).is_file()
                for fpath in self.get_episodes_file_paths(
                    include_videos=not self.uses_external_video_root
                )
            )
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = CODEBASE_VERSION
            self.download_episodes(download_videos and (not self.uses_external_video_root))
            self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
        torch.stack(self.hf_dataset["timestamp"]).numpy()
        torch.stack(self.hf_dataset["episode_index"]).numpy()
        {k: t.numpy() for (k, t) in self.episode_data_index.items()}
        if self.delta_timestamps is not None:
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def push_to_hub(
        self,
        branch: str | None = None,
        tags: list | None = None,
        license: str | None = "apache-2.0",
        tag_version: bool = True,
        push_videos: bool = True,
        private: bool = False,
        allow_patterns: list[str] | str | None = None,
        upload_large_folder: bool = False,
        **card_kwargs,
    ) -> None:
        ignore_patterns = ["images/"]
        if not push_videos:
            ignore_patterns.append("videos/")
        hub_api = HfApi()
        hub_api.create_repo(
            repo_id=self.repo_id, private=private, repo_type="dataset", exist_ok=True
        )
        if branch:
            hub_api.create_branch(
                repo_id=self.repo_id,
                branch=branch,
                revision=self.revision,
                repo_type="dataset",
                exist_ok=True,
            )
        upload_kwargs = {
            "repo_id": self.repo_id,
            "folder_path": self.root,
            "repo_type": "dataset",
            "revision": branch,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
        }
        if upload_large_folder:
            hub_api.upload_large_folder(**upload_kwargs)
        else:
            hub_api.upload_folder(**upload_kwargs)
        if not hub_api.file_exists(
            self.repo_id, REPOCARD_NAME, repo_type="dataset", revision=branch
        ):
            card = create_lerobot_dataset_card(
                tags=tags, dataset_info=self.meta.info, license=license, **card_kwargs
            )
            card.push_to_hub(repo_id=self.repo_id, repo_type="dataset", revision=branch)
        if tag_version:
            with contextlib.suppress(RevisionNotFoundError):
                hub_api.delete_tag(self.repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
            hub_api.create_tag(
                self.repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset"
            )

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    def download_episodes(self, download_videos: bool = True) -> None:
        """Downloads the dataset from the given 'repo_id' at the provided version. If 'episodes' is given, this
        will only download those episodes (selected by their episode_index). If 'episodes' is None, the whole
        dataset will be downloaded. Thanks to the behavior of snapshot_download, if the files are already present
        in 'local_dir', they won't be downloaded again.
        """
        files = None
        ignore_patterns = None if download_videos else "videos/"
        if self.episodes is not None:
            files = self.get_episodes_file_paths()
        self.pull_from_repo(allow_patterns=files, ignore_patterns=ignore_patterns)

    def _hf_features_for_parquet_file(self, parquet_file: str | Path | None) -> datasets.Features:
        raw_features = dict(self.features)
        hf_features = get_hf_features_from_features(raw_features)
        if parquet_file is None:
            return hf_features
        parquet_columns = set(pq.read_schema(str(parquet_file)).names)
        expected_columns = set(hf_features)
        if (
            "index" in expected_columns
            and "index" not in parquet_columns
            and ("frame_index" in parquet_columns)
        ):
            index_feature = dict(raw_features.pop("index"))
            raw_features["frame_index"] = {
                "dtype": index_feature["dtype"],
                "shape": index_feature["shape"],
                "names": index_feature.get("names"),
            }
            hf_features = get_hf_features_from_features(raw_features)
            expected_columns = set(hf_features)
        if expected_columns != parquet_columns:
            raise ValueError(
                f"LeRobot parquet schema mismatch for {parquet_file}: missing_columns={sorted(expected_columns - parquet_columns)}, extra_columns={sorted(parquet_columns - expected_columns)}."
            )
        return hf_features

    def get_episodes_file_paths(self, include_videos: bool = True) -> list[Path]:
        episodes = (
            self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        )
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if include_videos and len(self.meta.video_keys) > 0:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files
        return fpaths

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        if self.episodes is None:
            path = str(self.root / "data")
            sample_file = next((self.root / "data").glob("*/*.parquet"), None)
            features = self._hf_features_for_parquet_file(sample_file)
            hf_dataset = load_dataset("parquet", data_dir=path, split="train", features=features)
        else:
            files = [
                str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes
            ]
            sample_file = files[0] if files else None
            features = self._hf_features_for_parquet_file(sample_file)
            hf_dataset = load_dataset("parquet", data_files=files, split="train", features=features)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def create_hf_dataset(self) -> datasets.Dataset:
        features = get_hf_features_from_features(self.features)
        ft_dict = {col: [] for col in features}
        hf_dataset = datasets.Dataset.from_dict(ft_dict, features=features, split="train")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        return len(self.hf_dataset) if self.hf_dataset is not None else self.meta.total_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        return len(self.episodes) if self.episodes is not None else self.meta.total_episodes

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def hf_features(self) -> datasets.Features:
        """Features of the hf_dataset."""
        if self.hf_dataset is not None:
            return self.hf_dataset.features
        else:
            return get_hf_features_from_features(self.features)

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        ep_start = self.episode_data_index["from"][ep_idx]
        ep_end = self.episode_data_index["to"][ep_idx]
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for (key, delta_idx) in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [
                    (idx + delta < ep_start.item()) | (idx + delta >= ep_end.item())
                    for delta in delta_idx
                ]
            )
            for (key, delta_idx) in self.delta_indices.items()
        }
        return (query_indices, padding)

    def _get_query_timestamps(
        self, current_ts: float, query_indices: dict[str, list[int]] | None = None
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = self.hf_dataset.select(query_indices[key])["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]
        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        return {
            key: torch.stack(self.hf_dataset.select(q_idx)[key])
            for (key, q_idx) in query_indices.items()
            if key not in self.meta.video_keys
        }

    def _query_hf_dataset_fast(self, query_indices: dict[str, list[int]]) -> dict:
        result = {}
        processed_indices = set()
        index_to_selected = {}
        for key, q_idx in query_indices.items():
            if key not in self.meta.video_keys:
                if "images" in key and (not self.during_training):
                    continue
                q_idx_tuple = tuple(q_idx)
                if q_idx_tuple not in processed_indices:
                    selected_data = self.hf_dataset.select(q_idx)
                    index_to_selected[q_idx_tuple] = selected_data
                    processed_indices.add(q_idx_tuple)
                else:
                    selected_data = index_to_selected[q_idx_tuple]
                result[key] = torch.stack(selected_data[key])
        return result

    def get_episode_data(self, episode_id: int) -> dict:
        ep_start = self.episode_data_index["from"][episode_id].item()
        ep_end = self.episode_data_index["to"][episode_id].item()
        q_idx = list(range(ep_start, ep_end))
        selected_data = self.hf_dataset[q_idx]
        res_keys = self.meta.features.keys() - set(self.meta.video_keys)
        res = {key: torch.stack(selected_data[key]) for key in res_keys}
        return res

    def _query_videos(
        self, query_timestamps: dict[str, list[float]], ep_idx: int
    ) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.video_root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)
        return item

    def _add_padding_keys(self, item: dict, padding: dict[str, list[bool]]) -> dict:
        for key, val in padding.items():
            item[key] = torch.BoolTensor(val)
        return item

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        query_indices = None
        if self.delta_indices is not None:
            current_ep_idx = self.episodes.index(ep_idx) if self.episodes is not None else ep_idx
            (query_indices, padding) = self._get_query_indices(idx, current_ep_idx)
            query_result = self._query_hf_dataset_fast(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val
        if len(self.meta.video_keys) > 0 and self.during_training:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}
        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]
        if "coarse_task_index" in item:
            coarse_task_index = item["coarse_task_index"].item()
            item["coarse_task"] = self.meta.tasks[coarse_task_index]
        if "operating_hand_index" in item:
            operating_hand_index = item["operating_hand_index"].item()
            item["operating_hand"] = self.meta.tasks[operating_hand_index]
        if "subtask_annotation" in item and hasattr(self.meta, "annotations"):
            index = item["subtask_annotation"][0].item()
            item["subtask"] = self.meta.annotations["subtask"][index]
        if "atomic_task_index" in item and item["atomic_task_index"] is not None:
            atomic_task_index = item["atomic_task_index"].item()
            item["subtask"] = self.meta.tasks[int(atomic_task_index)]
        return item

    def __repr__(self):
        feature_keys = list(self.features)
        return f"{self.__class__.__name__}({{\n    Repository ID: '{self.repo_id}',\n    Number of selected episodes: '{self.num_episodes}',\n    Number of selected samples: '{self.num_frames}',\n    Features: '{feature_keys}',\n}})',\n"

    def create_episode_buffer(self, episode_index: int | None = None) -> dict:
        current_ep_idx = self.meta.total_episodes if episode_index is None else episode_index
        ep_buffer = {}
        ep_buffer["size"] = 0
        ep_buffer["task"] = []
        for key in self.features:
            ep_buffer[key] = current_ep_idx if key == "episode_index" else []
        return ep_buffer

    def _get_image_file_path(self, episode_index: int, image_key: str, frame_index: int) -> Path:
        fpath = DEFAULT_IMAGE_PATH.format(
            image_key=image_key, episode_index=episode_index, frame_index=frame_index
        )
        return self.root / fpath

    def add_frame(self, frame: dict, task: List[str], timestamp: float | None = None) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        assert len(task) == 4, "Task frame must be of two elements"
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()
        validate_frame(frame, self.features)
        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()
        frame_index = self.episode_buffer["size"]
        if timestamp is None:
            timestamp = frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(task)
        for key in frame:
            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )
            if self.features[key]["dtype"] in ["image", "video"]:
                img_path = self._get_image_file_path(
                    episode_index=self.episode_buffer["episode_index"],
                    image_key=key,
                    frame_index=frame_index,
                )
                if frame_index == 0:
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                self.episode_buffer[key].append(str(img_path))
            else:
                self.episode_buffer[key].append(frame[key])
        self.episode_buffer["size"] += 1

    def save_episode(
        self, episode_data: dict | None = None, raw_file_name: str | None = None
    ) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        if not episode_data:
            episode_buffer = self.episode_buffer
        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set([item for sublist in tasks for item in sublist]))
        episode_index = episode_buffer["episode_index"]
        episode_buffer["index"] = np.arange(
            self.meta.total_frames, self.meta.total_frames + episode_length
        )
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)
        episode_buffer["coarse_task_index"] = np.array(
            [self.meta.get_task_index(task[0]) for task in tasks]
        )
        episode_buffer["task_index"] = np.array(
            [self.meta.get_task_index(task[1]) for task in tasks]
        )
        episode_buffer["coarse_quality_index"] = np.array(
            [self.meta.get_task_index(task[2]) for task in tasks]
        )
        episode_buffer["quality_index"] = np.array(
            [self.meta.get_task_index(task[3]) for task in tasks]
        )
        for key, ft in self.features.items():
            if key in [
                "index",
                "episode_index",
                "coarse_task_index",
                "task_index",
                "coarse_quality_index",
                "quality_index",
            ] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])
        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)
        ep_stats = compute_episode_stats(
            episode_buffer, self.features, self.is_compute_episode_stats_image
        )
        if len(self.meta.video_keys) > 0:
            video_paths = self.encode_episode_videos(episode_index)
            for key in self.meta.video_keys:
                episode_buffer[key] = video_paths[key]
        self.meta.save_episode(
            episode_index, episode_length, episode_tasks, ep_stats, raw_file_name
        )
        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for (k, t) in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )
        video_files = list(self.root.rglob("*.mp4"))
        assert len(video_files) == self.num_episodes * len(self.meta.video_keys)
        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes
        img_dir = self.root / "images"
        if img_dir.is_dir():
            shutil.rmtree(self.root / "images")
        if not episode_data:
            self.episode_buffer = self.create_episode_buffer()

    def _save_episode_table(self, episode_buffer: dict, episode_index: int) -> None:
        episode_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(
            episode_dict, features=self.hf_features, split="train"
        )
        ep_dataset = embed_images(ep_dataset)
        self.hf_dataset = concatenate_datasets([self.hf_dataset, ep_dataset])
        self.hf_dataset.set_transform(hf_transform_to_torch)
        ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_dataset.to_parquet(ep_data_path)

    def clear_episode_buffer(self) -> None:
        episode_index = self.episode_buffer["episode_index"]
        if self.image_writer is not None:
            for cam_key in self.meta.camera_keys:
                img_dir = self._get_image_file_path(
                    episode_index=episode_index, image_key=cam_key, frame_index=0
                ).parent
                if img_dir.is_dir():
                    shutil.rmtree(img_dir)
        self.episode_buffer = self.create_episode_buffer()

    def stop_image_writer(self) -> None:
        """
        Whenever wrapping this dataset inside a parallelized DataLoader, this needs to be called first to
        remove the image_writer in order for the LeRobotDataset object to be picklable and parallelized.
        """
        if self.image_writer is not None:
            self.image_writer.stop()
            self.image_writer = None

    def _wait_image_writer(self) -> None:
        """Wait for asynchronous image writer to finish."""
        if self.image_writer is not None:
            self.image_writer.wait_until_done()

    def encode_videos(self) -> None:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        for ep_idx in range(self.meta.total_episodes):
            self.encode_episode_videos(ep_idx)

    def encode_episode_videos(self, episode_index: int) -> dict:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        video_paths = {}
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            video_paths[key] = str(video_path)
            if video_path.is_file():
                continue
            img_dir = self._get_image_file_path(
                episode_index=episode_index, image_key=key, frame_index=0
            ).parent
            encode_video_frames(
                img_dir, video_path, self.fps, overwrite=True, vcodec=self.video_codec
            )
        return video_paths

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
        tolerance_s: float = 0.0001,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
        video_codec: Literal["h264", "hevc", "libsvtav1", "h264_nvenc"] = "libsvtav1",
        is_compute_episode_stats_image=True,
    ) -> "LeRobotDataset":
        """Create a LeRobot Dataset from scratch in order to record data."""
        obj = cls.__new__(cls)
        obj.meta = LeRobotDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            root=root,
            use_videos=use_videos,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_writer = None
        obj.episode_buffer = obj.create_episode_buffer()
        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.episode_data_index = None
        obj.video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        obj.video_codec = video_codec
        obj.is_compute_episode_stats_image = is_compute_episode_stats_image
        return obj


class MultiLeRobotDataset(torch.utils.data.Dataset):
    """A dataset consisting of multiple underlying `LeRobotDataset`s.

    The underlying `LeRobotDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotDataset`.
    """

    def __init__(
        self,
        dataset_dirs: list[str],
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
        video_roots: dict[str, str | Path] | None = None,
    ):
        super().__init__()
        self.dataset_dirs = dataset_dirs
        ds_roots = [Path(ds_dir) for ds_dir in dataset_dirs]
        ds_names = [ds_dir for ds_dir in dataset_dirs]
        self.ds_names = ds_names
        self.ds_roots = ds_roots
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(ds_names, 0.0001)
        self._datasets = []
        for ds_root, ds_name in zip(ds_roots, ds_names, strict=True):
            try:
                _dataset = LeRobotDataset(
                    ds_name,
                    root=ds_root,
                    episodes=episodes[ds_name] if episodes else None,
                    image_transforms=image_transforms,
                    delta_timestamps=delta_timestamps,
                    tolerance_s=self.tolerances_s[ds_name],
                    download_videos=download_videos,
                    video_backend=video_backend,
                    video_root=video_roots[ds_name] if video_roots is not None else None,
                )
                self._datasets.append(_dataset)
            except Exception as e:
                logging.error(f"Exception while process ds_root: {ds_root}, ds_name: {ds_name}")
                traceback.print_exc()
                raise e
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. The multi-dataset functionality currently only keeps common keys."
            )
        for ds_name, ds in zip(self.ds_names, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            if extra_keys:
                logging.warning(
                    f"keys {extra_keys} of {ds_name} were disabled as they are not contained in all the other datasets."
                )
            self.disabled_features.update(extra_keys)
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.stats = aggregate_stats(
            [
                _stats_with_count_from_total_frames(
                    dataset.meta.stats,
                    total_frames=dataset.meta.total_frames,
                    root=dataset.meta.root,
                )
                for dataset in self._datasets
            ]
        )

    def set_during_training(self, during_training: bool):
        for dataset in self._datasets:
            dataset.during_training = during_training

    @property
    def repo_id_to_index(self):
        """Return a mapping from dataset repo_id to a dataset index automatically created by this class.

        This index is incorporated as a data key in the dictionary returned by `__getitem__`.
        """
        return {repo_id: i for (i, repo_id) in enumerate(self.ds_names)}

    @property
    def repo_index_to_id(self):
        """Return the inverse mapping if repo_id_to_index."""
        return {v: k for (k, v) in self.repo_id_to_index}

    @property
    def fps(self) -> int:
        """Frames per second used during data collection.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].meta.info["fps"]

    @property
    def video(self) -> bool:
        """Returns True if this dataset loads video frames from mp4 files.

        Returns False if it only loads images from png files.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].meta.info.get("video", False)

    @property
    def features(self) -> datasets.Features:
        features = {}
        for dataset in self._datasets:
            features.update(
                {k: v for (k, v) in dataset.hf_features.items() if k not in self.disabled_features}
            )
        return features

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access image and video stream from cameras."""
        keys = []
        for key, feats in self.features.items():
            if isinstance(feats, (datasets.Image, VideoFrame)):
                keys.append(key)
        return keys

    @property
    def video_frame_keys(self) -> list[str]:
        """Keys to access video frames that requires to be decoded into images.

        Note: It is empty if the dataset contains images only,
        or equal to `self.cameras` if the dataset contains videos only,
        or can even be a subset of `self.cameras` in a case of a mixed image/video dataset.
        """
        video_frame_keys = []
        for key, feats in self.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_frames(self) -> int:
        """Number of samples/frames."""
        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return sum(d.num_episodes for d in self._datasets)

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        return 1 / self.fps - 0.0001

    def get_episode_data(self, episode_idx: int) -> dict:
        for dataset in self._datasets:
            if episode_idx < dataset.num_episodes:
                file = str(
                    dataset.root / dataset.meta.get_data_file_path(dataset.episodes[episode_idx])
                )
                table = pq.read_table(str(file))
                result_dict = {}
                for col_name in table.column_names:
                    col = table[col_name]
                    try:
                        np_arr = col.to_numpy(zero_copy_only=True)
                    except Exception:
                        raw = col.to_numpy()
                        np_arr = np.stack(raw) if raw.dtype == object else raw
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="The given NumPy array is not writable",
                            category=UserWarning,
                        )
                        if np_arr.dtype == "O":
                            result_dict[col_name] = np_arr
                        else:
                            result_dict[col_name] = torch.from_numpy(np_arr)
                return result_dict
            else:
                episode_idx -= dataset.num_episodes
        raise IndexError(f"Episode index {episode_idx} out of bounds.")

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            raise AssertionError(
                "We expect the loop to break out as long as the index is within bounds."
            )
        item = self._datasets[dataset_idx][idx - start_idx]
        item["dataset_index"] = torch.tensor(dataset_idx)
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]
        return item

    def __repr__(self):
        return f"{self.__class__.__name__}(\n  Dataset Names: '{self.ds_names}',\n  Number of Samples: {self.num_frames},\n  Number of Episodes: {self.num_episodes},\n  Type: {('video (.mp4)' if self.video else 'image (.png)')},\n  Recorded Frames per Second: {self.fps},\n  Camera Keys: {self.camera_keys},\n  Video Frame Keys: {(self.video_frame_keys if self.video else 'N/A')},\n  Transformations: {self.image_transforms},\n)"
