"""Training and evaluation preserve fractional pixels during direct camera resizing."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from dreamfree.data.latent_data import _apply_robocasa_dreamzero_multicam
from dreamfree.data.robocasa_visual import compose_robocasa_dreamzero
from dreamfree.data.stage2 import _SparseStage2RobotVideoDataset
from dreamfree.data.video import ROBOCASA_CAMERA_KEYS, VideoDataset
from dreamfree.evaluation.robocasa.policy import build_robocasa_image_tensor


@pytest.mark.parametrize("size", [352, 448])
def test_robocasa_resize_preserves_float_pixels_and_camera_layout(size):
    raw = torch.randint(
        0, 256, (3, 1, 3, 224, 224), generator=torch.Generator().manual_seed(7), dtype=torch.uint8
    )
    actual = compose_robocasa_dreamzero(raw, output_size=(size, size), expected_raw_size=(224, 224))
    obs = {
        key: raw[i, 0].permute(1, 2, 0).numpy()
        for i, key in enumerate(("agentview_left", "agentview_right", "wrist"))
    }
    evaluated = build_robocasa_image_tensor(obs, image_size=(size, size))
    torch.testing.assert_close(actual, evaluated, rtol=0, atol=0)
    if size == 352:
        trained = _apply_robocasa_dreamzero_multicam(raw, height=size, width=size)
        torch.testing.assert_close(trained, evaluated, rtol=0, atol=0)
    live_cfg = SimpleNamespace(
        dataset_name="robocasa" if size == 352 else "robocasa_success_448",
        camera_layout=f"robocasa_dreamzero_{size}",
    )
    video_dataset = SimpleNamespace(
        cfg=live_cfg,
        height=size,
        width=size,
        _camera_keys_for_available_tensors=lambda _: ROBOCASA_CAMERA_KEYS,
    )
    live_video = VideoDataset._compose_multicam(
        video_dataset, {key: raw[i].float() / 255 for i, key in enumerate(ROBOCASA_CAMERA_KEYS)}
    )
    torch.testing.assert_close(live_video * 2 - 1, evaluated, rtol=0, atol=0)
    action_dataset = SimpleNamespace(
        camera_layout=f"robocasa_dreamzero_{size}", video_size=(size, size)
    )
    live_action = _SparseStage2RobotVideoDataset._compose_video(action_dataset, raw.float() / 255)
    torch.testing.assert_close(live_action.permute(1, 0, 2, 3), evaluated, rtol=0, atol=0)
    wrist = (
        F.interpolate(
            raw[2].float() / 255,
            size=(size // 2, size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        * 2
        - 1
    )
    torch.testing.assert_close(actual[:, :, : size // 2], wrist, rtol=0, atol=0)
    assert actual.dtype == torch.float32
    assert not torch.equal(wrist[..., ::2], wrist[..., 1::2])
    pixel_values = (actual + 1) * 127.5
    assert (pixel_values - pixel_values.round()).abs().max() > 0.1
