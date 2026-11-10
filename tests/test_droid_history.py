"""Verify DROID history normalization and ownership across the RPC boundary."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_droid_molmospaces import CHECKPOINT, observation, simulation_server

from dreamfree.data.lerobot.cached_latent_dataset import CachedStage2LatentDataset
from dreamfree.evaluation.action_protocol import (
    ISSUED_ACTION_HISTORY_IS_PAD_KEY as PAD,
)
from dreamfree.evaluation.action_protocol import (
    ISSUED_ACTION_HISTORY_KEY as HISTORY,
)
from dreamfree.evaluation.droid_sim import deploy_policy as droid
from dreamfree.evaluation.droid_sim.deploy_policy_optimized import OptimizedDroidInferenceMixin
from dreamfree.evaluation.droid_sim.molmospaces import MolmoSpacesClient


def test_history_reanchors_to_current_state_and_matches_training_cache():
    def stats(dim, low, high):
        return droid._convert_feature_stats(
            {
                "min": [low] * dim,
                "max": [high] * dim,
                "mean": [0] * dim,
                "std": [1] * dim,
                "q01": [low] * dim,
                "q99": [high] * dim,
            },
            feature_key="fixture",
        )

    processor = droid.build_droid_processor(
        dataset_stats={
            kind: {"joint_position": stats(7, -2, 2), "gripper_position": stats(1, 0, 1)}
            for kind in ("action", "state")
        }
    )
    actor = object.__new__(droid.DroidPolicy)
    actor.action_history_horizon = 8
    actor.processor = processor
    actor.device, actor.model_dtype = "cpu", torch.float32
    observed = []

    def capture(obs, instruction):
        observed.append(
            (actor._active_previous_action.clone(), actor._active_previous_action_is_pad.clone())
        )
        return np.zeros((32, 8))

    actor._infer_relative_action_chunk_with_history = capture
    commands = np.zeros((8, 8), np.float32)
    commands[-2:, :7] = [[0.5] * 7, [1.5] * 7]
    commands[-2:, 7] = [0, 1]
    pad = np.array([True] * 6 + [False] * 2)
    request = {
        HISTORY: commands,
        PAD: pad,
        "observation/joint_position": np.full(7, 0.25),
        "observation/gripper_position": [0.5],
    }
    actor._infer_relative_action_chunk(request, "move")
    normalized, mask = observed[-1]
    torch.testing.assert_close(normalized[0, -2:, :7], torch.tensor([[0.125] * 7, [0.625] * 7]))
    torch.testing.assert_close(normalized[0, -2:, 7], torch.tensor([-1.0, 1.0]))
    assert not normalized[0, :6].any()
    assert np.array_equal(mask[0], pad)
    assert actor._active_previous_action is None

    # The cache normalizer receives raw episode tensors and the current state anchor.
    cached = object.__new__(CachedStage2LatentDataset)
    cached.action_history_horizon = 8
    cached._processor = processor
    tensors = {
        "droid_action_joint_position": torch.tensor(commands[-2:, :7]),
        "droid_action_gripper_position": torch.tensor([[0.2], [0.8]]),
        "droid_state_joint_position": torch.full((2, 7), 0.25),
        "droid_state_gripper_position": torch.full((2, 1), 0.5),
    }
    handle = SimpleNamespace(keys=lambda: tensors.keys(), get_tensor=tensors.__getitem__)
    actions, _ = cached._droid_dreamzero_action_state_from_cache(
        handle,
        ep_idx=0,
        ep_len=2,
        action_indices=torch.tensor([0] * 7 + [1] + [1] * 32),
        proprio_indices=torch.tensor([1]),
    )
    torch.testing.assert_close(normalized[0, -2:], actions[6:8])
    # History is binary, but future gripper targets retain the recorded continuous values.
    assert actions[8, 7] == pytest.approx(0.6)

    request[HISTORY][-1, 7] = 0.3
    with pytest.raises(ValueError, match="binary"):
        actor._infer_relative_action_chunk(request, "move")
    request.pop(HISTORY)
    with pytest.raises(ValueError, match="execution-owned"):
        actor._infer_relative_action_chunk(request, "move")


def test_molmospaces_sends_only_returned_actions_and_clears_history():
    info = {"checkpoint_type": "dreamfree_stage2_action_history", "action_history_horizon": 8}
    with simulation_server(info_override=info) as (port, calls):
        client = MolmoSpacesClient(expected_checkpoint=CHECKPOINT, port=port, replan_steps=2)
        try:
            first = client.get_action(observation(), "move")
            second = client.get_action(None, "move")
            client.get_action(observation(), "move")
            assert calls["infer"][0][0][PAD].all()
            history = calls["infer"][1][0][HISTORY]
            np.testing.assert_allclose(history[-2:, :7], [first["arm"], second["arm"]])
            np.testing.assert_array_equal(history[-2:, 7], [0, 1])
            assert calls["infer"][1][0][PAD].sum() == 6
            client.reset()
            client.get_action(observation(), "move")
            assert calls["infer"][-1][0][PAD].all()
        finally:
            client.close()


@pytest.mark.parametrize("horizon", [0, 8])
def test_full_warmup_captures_empty_partial_and_full_history(monkeypatch, horizon):
    actor = OptimizedDroidInferenceMixin()
    actor.device = torch.device("cuda:0")
    actor.action_history_horizon = horizon
    actor.action_horizon = 32
    actor.action_cuda_graph_enabled = True
    actor._action_cuda_graphs = {}
    actor.optimization_contract = {}
    actor._prompt_context_cache = {"existing": "entry"}
    actor.text_embedding_cache_dirs = []
    actor._timing_rollout = {"infer_s": 3}
    actor._action_noise_generator = actor.action_noise_episode_seed = None
    actor.action_noise_derived_seed = actor.action_noise_seed = None
    actor.video_core = SimpleNamespace(
        vae=SimpleNamespace(model=SimpleNamespace(clear_cache=lambda: None))
    )
    actor.action_model = SimpleNamespace(action_encoder=torch.nn.Linear(8, 16))
    monkeypatch.setattr(torch.random, "fork_rng", lambda **kwargs: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    seen_padding = []

    def predict(obs, instruction, **kwargs):
        if horizon:
            seen_padding.append(int(obs[PAD].sum()))
            actor._active_previous_action = torch.tensor(obs[HISTORY])[None]
            actor._active_previous_action_is_pad = torch.tensor(obs[PAD])[None]
        prefix, mask = actor._prepare_action_query(
            proprio_query_emb=torch.zeros(1, 1, 16),
            action_seq_len=32,
        )
        assert prefix.shape[1] == 1 + horizon
        actor._action_cuda_graphs[mask is not None] = object()
        return np.zeros((32, 8))

    actor.predict_action_chunk = predict
    actor._warmup_full_inference()
    assert actor.optimization_contract["warmup_complete"]
    assert actor._prompt_context_cache == {"existing": "entry"}
    assert actor._timing_rollout == {"infer_s": 3}
    assert actor._active_previous_action is None
    assert seen_padding == ([8, 4, 0] * 2 if horizon else [])
    assert set(actor._action_cuda_graphs) == ({False, True} if horizon else {False})


@pytest.mark.parametrize("steps", [16, 24])
def test_graph_options_accept_recorded_awam_schedules(steps):
    from dreamfree.evaluation.droid_sim.deploy_policy_optimized import (
        OPTIMIZED_NATIVE_POLICY_RUNTIME,
        resolve_action_cuda_graph,
        resolve_visual_optimization,
    )

    actor = SimpleNamespace(
        video_core=SimpleNamespace(video_model_family="wan22_i2v_a14b"),
        device=torch.device("cuda"),
        model_dtype=torch.bfloat16,
        tiled=False,
        condition_video_num_frames=9,
        student_layer_idx=31,
        student_hook_point="post_block",
        expected_student_tokens_per_frame=880,
        text_context_len=128,
        action_horizon=32,
        action_sampler="euler",
        num_inference_steps=steps,
    )
    assert resolve_action_cuda_graph("true", actor, OPTIMIZED_NATIVE_POLICY_RUNTIME)[0]
    for feature in ("vae_first_frame_only", "encoder_cuda_graph"):
        assert resolve_visual_optimization(
            "true", actor, OPTIMIZED_NATIVE_POLICY_RUNTIME, "none", feature=feature
        )[0]
    actor.num_inference_steps = 10
    with pytest.raises(ValueError, match="16- or 24-step"):
        resolve_action_cuda_graph("true", actor, OPTIMIZED_NATIVE_POLICY_RUNTIME)


def test_action_noise_preserves_awam_seed_namespace():
    # Recorded with AWAM's original reset_action_noise_for_episode implementation.
    actor = SimpleNamespace(action_noise_seed=42, noise_device="cpu", _action_noise_generator=None)
    droid.DroidPolicy.reset_action_noise_for_episode(actor, 7)
    assert actor.action_noise_derived_seed == 3130822223687568163
