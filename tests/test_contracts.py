import ast
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from dreamfree.config import load_config
from dreamfree.data.text_cache import validate_text_cache
from dreamfree.evaluation.droid_real import client
from dreamfree.models.distillation import teacher_state_prediction_loss
from dreamfree.models.wan22.future_state import fixed_tokenwise_layer_norm
from dreamfree.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("stage", [0, 1, 2])
@pytest.mark.parametrize("model", ["5b", "14b"])
@pytest.mark.parametrize("dataset", ["libero", "robocasa", "robotwin", "droid"])
def test_configs_cover_selected_models_stages_and_datasets(stage, model, dataset):
    cfg = load_config(stage, ROOT / f"configs/stage{stage}_{model}.yaml", [f"dataset={dataset}"])
    assert cfg.dataset_name == dataset
    assert cfg.video_model_family == ("wan22_i2v_a14b" if model == "14b" else "wan22_ti2v_5b")
    assert cfg.use_cached_latents is False


@pytest.mark.parametrize(
    "override",
    [
        "learning_rate=wrong",
        "max_steps=true",
        "unknown=1",
        "model=a14b",
        "dataset=unsupported",
        "data_mode=cached",
        "use_cached_latents=true",
    ],
)
def test_invalid_configuration_fails(override):
    if override == "data_mode=cached":
        # A mode can be selected before filling the required cache path.
        cfg = load_config(2, ROOT / "configs/stage2_14b.yaml", [override])
        assert cfg.use_cached_latents
    else:
        with pytest.raises((ValueError, TypeError)):
            load_config(2, ROOT / "configs/stage2_14b.yaml", [override])


def test_text_cache_rejects_wrong_format_mask_and_nonfinite_values():
    good = {
        "format": "dreamfree_text_v1",
        "context": torch.zeros(4, 8),
        "mask": torch.ones(4, dtype=torch.bool),
    }
    validate_text_cache(good, "fixture", context_len=4, text_dim=8)
    for bad in (
        {**good, "format": "unknown"},
        {**good, "mask": torch.zeros(4, dtype=torch.bool)},
        {**good, "context": torch.full((4, 8), float("nan"))},
    ):
        with pytest.raises(ValueError):
            validate_text_cache(bad, "fixture", context_len=4, text_dim=8)


def test_distillation_loss_and_gradient_match_normalized_cosine_objective():
    torch.manual_seed(7)
    prediction = torch.randn(2, 6, 12, requires_grad=True)
    target = torch.randn_like(prediction, requires_grad=True)
    loss, _, _ = teacher_state_prediction_loss(
        student_prediction=prediction,
        teacher_target=target,
        tokens_per_frame=2,
        projector_input_tokens=2,
    )
    expected = (
        1
        - F.cosine_similarity(
            fixed_tokenwise_layer_norm(prediction.float()),
            fixed_tokenwise_layer_norm(target.detach().float()),
            dim=-1,
        ).mean()
    )
    torch.testing.assert_close(loss, expected)
    actual_grad = torch.autograd.grad(loss, prediction, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, prediction)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert target.grad is None


def test_flow_matching_euler_recovers_constant_velocity_endpoint():
    scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    clean = torch.randn(2, 3, 4)
    noise = torch.randn_like(clean)
    velocity = noise - clean
    _, deltas = scheduler.build_inference_schedule(10, torch.device("cpu"), torch.float32)
    sample = noise.clone()
    for delta in deltas:
        sample = scheduler.step(velocity, delta, sample)
    torch.testing.assert_close(sample, clean, atol=1e-6, rtol=1e-5)


def test_physical_client_checks_nonfinite_actions_and_joint_limits():
    rest = np.array([0.0, -0.2 * np.pi, 0.0, -0.8 * np.pi, 0.0, 0.6 * np.pi, 0.0], dtype=np.float32)
    actions = np.tile(np.r_[rest, 0.5], (32, 1))
    client.validate_action_prefix(actions, rest, 8)
    actions[0, 0] = 100
    with pytest.raises(client.UnsafeActionError):
        client.validate_action_prefix(actions, rest, 8)
    actions[0, 0] = np.nan
    with pytest.raises((ValueError, client.UnsafeActionError)):
        client.coerce_action_chunk({"actions": actions}, 32)


def test_physical_client_remains_python37_parseable():
    ast.parse(Path(client.__file__).read_text(), feature_version=(3, 7))


def test_stage2_does_not_change_the_configured_model_to_match_a_checkpoint(monkeypatch):
    from dreamfree.training.stage2 import validation

    cfg = load_config(2, ROOT / "configs/stage2_5b.yaml", [])
    cfg.stage1_checkpoint = "source"
    monkeypatch.setattr(
        validation,
        "stage1_student_source_model_contract",
        lambda path: {"model_id": "Wan-AI/Wan2.2-I2V-A14B"},
    )
    with pytest.raises(ValueError, match="configuration conflicts"):
        validation._apply_stage1_checkpoint_model_contract(cfg)
    assert cfg.model_id == "Wan-AI/Wan2.2-TI2V-5B"
