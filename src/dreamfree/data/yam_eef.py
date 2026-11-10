"""YAM wrist-frame forward kinematics and immutable Stage 2 action overlays."""

from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from dreamfree.data import yam

MODEL_PATH = (
    Path(os.environ.get("ABC_ROOT", Path(__file__).resolve().parents[3] / "third_party/abc"))
    / "abc_sim/models/assets/i2rt_yam/yam.xml"
)
MODEL_SHA256 = "a797c4c27d1c074eb24b04d9cf13cef9a66589f135ebf35bb22f2be3b86f7d28"
EEF_NAMES = [
    f"{side}_{name}"
    for side in ("left", "right")
    for name in ("x", "y", "z", "R00", "R10", "R20", "R01", "R11", "R21", "gripper")
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _transform(element: ET.Element) -> np.ndarray:
    translation = np.fromstring(element.get("pos", "0 0 0"), sep=" ")
    quat = np.fromstring(element.get("quat", "1 0 0 0"), sep=" ")
    if translation.shape != (3,) or quat.shape != (4,) or np.linalg.norm(quat) == 0:
        raise ValueError(f"Invalid YAM fixed transform: {element.attrib}")
    w, x, y, z = quat / np.linalg.norm(quat)
    result = np.eye(4)
    result[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    result[:3, 3] = translation
    return result


class YamWristFK:
    """Batched FK for the pinned six-hinge YAM model, expressed in each arm's base."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = Path(model_path).resolve()
        if sha256_file(self.model_path) != MODEL_SHA256:
            raise ValueError(f"YAM FK model differs from the validated revision: {self.model_path}")
        root = ET.parse(self.model_path).getroot()
        if root.find("compiler").get("angle") != "radian":
            raise ValueError("YAM FK requires a model in radians.")
        node = root.find("worldbody/body[@name='arm']")
        self.fixed, self.limits = [], []
        for index in range(1, 7):
            node = node.find(f"body[@name='link_{index}']")
            joint = node.find(f"joint[@name='joint{index}']")
            if joint is None or joint.get("type", "hinge") != "hinge":
                raise ValueError(f"YAM FK requires hinge joint{index}.")
            self.fixed.append(_transform(node))
            self.limits.append(np.fromstring(joint.attrib["range"], sep=" "))
        self.tool = _transform(node.find("site[@name='tcp_site']"))
        self.limits = np.asarray(self.limits)
        self.manifest = {
            "model_path": str(self.model_path),
            "model_sha256": MODEL_SHA256,
            "frame": "per_arm_base",
            "site": "tcp_site",
            "reference": "wrist, no fingertip offset",
            "joint_order": [f"joint{i}" for i in range(1, 7)],
            "joint_units": "radians",
            "position_units": "meters",
            "rotation_order": EEF_NAMES[3:9],
            "fixed_transforms": [value.tolist() for value in self.fixed],
            "site_transform": self.tool.tolist(),
            "joint_limits": self.limits.tolist(),
            "joint_clipping": False,
        }

    def poses(self, joints: np.ndarray) -> np.ndarray:
        q = np.asarray(joints, dtype=np.float64)
        if q.ndim < 1 or q.shape[-1] != 6 or not np.isfinite(q).all():
            raise ValueError("YAM FK requires finite joint positions with last dimension 6.")
        transform = np.broadcast_to(np.eye(4), (*q.shape[:-1], 4, 4)).copy()
        rotation = np.broadcast_to(np.eye(4), transform.shape).copy()
        for index, fixed in enumerate(self.fixed):
            cosine, sine = np.cos(q[..., index]), np.sin(q[..., index])
            rotation[..., 0, 0] = rotation[..., 1, 1] = cosine
            rotation[..., 0, 1], rotation[..., 1, 0] = -sine, sine
            transform = transform @ fixed @ rotation
        return transform @ self.tool

    def actions(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action)
        if action.ndim != 2 or action.shape[1] != 14 or not np.isfinite(action).all():
            raise ValueError("YAM EEF conversion requires finite raw [T,14] joint/gripper actions.")
        arms = []
        for start in (0, 7):
            pose = self.poses(action[:, start : start + 6])
            arms.extend(
                (pose[:, :3, 3], pose[:, :3, 0], pose[:, :3, 1], action[:, start + 6 : start + 7])
            )
        return np.concatenate(arms, axis=1).astype(np.float32)


def validate_source_header(path: Path, metadata: dict, source_metadata: dict) -> None:
    if metadata.get("overlay_cache_kind") != yam.EEF_OVERLAY_KIND:
        raise ValueError(f"Missing YAM EEF overlay contract: {path}")
    source = Path(metadata["latent_source_path"])
    stat = source.stat()
    if (str(stat.st_size), str(stat.st_mtime_ns)) != (
        metadata.get("latent_source_size"),
        metadata.get("latent_source_mtime_ns"),
    ):
        raise ValueError(f"YAM EEF source cache changed after conversion: {source}")
    if yam.fingerprint(source_metadata) != metadata.get("latent_source_metadata_sha256"):
        raise ValueError(f"YAM EEF source cache metadata changed: {source}")
    for key in (
        "episode_index",
        "num_frames",
        "num_samples",
        "source_task",
        "source_task_index",
        "source_task_class",
        "yam_split",
        "video_model_family",
        "model_id",
        "vae_family",
        "video_h",
        "video_w",
        "concat_layout",
        "video_value_contract",
        "i2v_condition_source",
    ):
        if metadata.get(key) != source_metadata.get(key):
            raise ValueError(f"YAM EEF overlay/source mismatch for {key}: {path}")
    if source_metadata.get("selection_sha256") != metadata.get("source_selection_sha256"):
        raise ValueError(f"YAM EEF source selection mismatch: {path}")


def validate_overlay_source(path: Path, metadata: dict, selection: dict) -> None:
    from safetensors import safe_open

    if (
        metadata.get("source_selection_sha256")
        != selection["eef_conversion"]["source_selection_sha256"]
    ):
        raise ValueError(f"YAM EEF overlay belongs to another source selection: {path}")
    if metadata.get("kinematics_sha256") != selection["eef_conversion"]["kinematics_sha256"]:
        raise ValueError(f"YAM EEF overlay uses a different kinematic model: {path}")
    source = Path(metadata["latent_source_path"])
    if source.resolve() == path.resolve():
        raise ValueError(f"YAM EEF overlay cannot reference itself: {path}")
    source_meta = yam.validate_cache(
        source, selection["eef_conversion"]["source_root"], yam.ABC_NAME
    )
    validate_source_header(path, metadata, source_meta)
    with safe_open(str(source), framework="np") as handle:
        if handle.get_slice("latents").get_shape()[0] != int(metadata["num_frames"]):
            raise ValueError(f"YAM EEF source latent length mismatch: {source}")
