"""Check visual-source equivalence, target ordering, and strict batch boundaries."""

from types import SimpleNamespace

import pytest
import torch

from dreamfree.data.action_batch import prepare_action_batch
from dreamfree.models.distillation import prepare_distillation_batch
from dreamfree.models.wan22.i2v_condition import WAN22_A14B_I2V_CONDITION_LATENTS_KEY


class FixtureCore:
    device = torch.device("cpu")
    torch_dtype = torch.float32
    dit = SimpleNamespace(fuse_vae_embedding_in_latents=False)

    def __init__(self, family):
        self.video_model_family = family
        self.channels = 16 if family == "wan22_i2v_a14b" else 48
        self.encoded_rows = []

    def latents(self, video):
        return video[:, :1, ::4, ::8, ::8].expand(-1, self.channels, -1, -1, -1).clone()

    def _encode_video_latents(self, video, *, tiled):
        self.encoded_rows.append(video.shape[0])
        return self.latents(video)

    def _encode_14b_i2v_condition_latents(self, video, *, tiled):
        return self.latents(video[:, :, :1].expand_as(video))


def sample_batch():
    video = torch.stack([torch.full((3, 5, 16, 16), value) for value in (0.25, 0.75)])
    return {
        "video": video,
        "context": torch.ones(2, 128, 4096),
        "context_mask": torch.ones(2, 128, dtype=torch.bool),
        "action": torch.arange(16, dtype=torch.float32).reshape(2, 4, 2) / 16,
        "proprio": torch.zeros(2, 4, 3),
        "action_is_pad": torch.tensor([[False, False, False, True], [False] * 4]),
        "previous_action": torch.zeros(2, 2, 2),
        "previous_action_is_pad": torch.tensor([[True, False], [False, False]]),
    }


@pytest.mark.parametrize("family", ["wan22_ti2v_5b", "wan22_i2v_a14b"])
@pytest.mark.parametrize("prepare", [prepare_action_batch, prepare_distillation_batch])
def test_live_and_cached_batches_preserve_latents_and_skip_cached_vae(family, prepare):
    core = FixtureCore(family)
    live = sample_batch()
    expected = prepare(core, live)
    assert core.encoded_rows == [2]
    cached = {key: value for key, value in live.items() if key != "video"}
    cached["input_latents"] = expected["input_latents"].clone().requires_grad_()
    if family == "wan22_i2v_a14b":
        cached[WAN22_A14B_I2V_CONDITION_LATENTS_KEY] = (
            expected[WAN22_A14B_I2V_CONDITION_LATENTS_KEY].clone().requires_grad_()
        )
    actual = prepare(core, cached)
    assert core.encoded_rows == [2]
    for key in ("input_latents", "context", "context_mask"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert not actual["input_latents"].requires_grad
    if family == "wan22_i2v_a14b":
        key = WAN22_A14B_I2V_CONDITION_LATENTS_KEY
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        assert not actual[key].requires_grad


@pytest.mark.parametrize("family", ["wan22_ti2v_5b", "wan22_i2v_a14b"])
def test_mixed_action_batch_restores_rows_and_preserves_padding(family):
    core = FixtureCore(family)
    sample = sample_batch()
    expected = prepare_action_batch(core, sample)
    mixed = {
        **sample,
        "video": sample["video"][1:],
        "video_batch_indices": torch.tensor([1]),
        "input_latents": expected["input_latents"][:1],
        "input_latents_batch_indices": torch.tensor([0]),
    }
    if family == "wan22_i2v_a14b":
        key = WAN22_A14B_I2V_CONDITION_LATENTS_KEY
        mixed[key] = expected[key][:1]
    actual = prepare_action_batch(core, mixed)
    assert core.encoded_rows == [2, 1]
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("prepare", [prepare_action_batch, prepare_distillation_batch])
def test_cached_14b_requires_official_conditioning_without_encoding(prepare):
    core = FixtureCore("wan22_i2v_a14b")
    sample = sample_batch()
    sample["input_latents"] = core.latents(sample.pop("video"))
    with pytest.raises(ValueError, match="I2V|i2v"):
        prepare(core, sample)
    assert not core.encoded_rows


@pytest.mark.parametrize("invalid", ["all_padded", "nonzero_padded_history", "overlapping_rows"])
def test_invalid_supervision_and_modality_rows_fail_before_encoding(invalid):
    core = FixtureCore("wan22_ti2v_5b")
    sample = sample_batch()
    if invalid == "all_padded":
        sample["action_is_pad"][0] = True
        message = "all-padded"
    elif invalid == "nonzero_padded_history":
        sample["previous_action"][0, 0] = 1
        message = "normalized zeros"
    else:
        sample["input_latents"] = core.latents(sample["video"][:1])
        sample["input_latents_batch_indices"] = torch.tensor([0])
        sample["video"] = sample["video"][:1]
        sample["video_batch_indices"] = torch.tensor([0])
        message = "same batch rows"
    with pytest.raises(ValueError, match=message):
        prepare_action_batch(core, sample)
    assert not core.encoded_rows
