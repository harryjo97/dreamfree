"""Check MolmoSpaces observation/action conversion over the simulation RPC protocol."""

import threading
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from dreamfree.evaluation.droid_real.policy_rpc import PolicyRpcServer
from dreamfree.evaluation.droid_sim.molmospaces import (
    GRIPPER_QPOS_OPEN_SCALE,
    MolmoSpacesClient,
    main,
    rpc_observation,
    validate_results,
)
from dreamfree.evaluation.droid_sim.server import policy_info

CHECKPOINT = "/models/dreamfree/stage2/step_100000"


def observation():
    return {
        "exo_camera_1": np.full((6, 8, 3), 10, dtype=np.uint8),
        "exo_camera_2": np.full((6, 8, 3), 20, dtype=np.uint8),
        "wrist_camera_zed_mini": np.full((6, 8, 3), 30, dtype=np.uint8),
        "qpos": {
            "arm": np.arange(7, dtype=np.float32),
            "gripper": np.array([GRIPPER_QPOS_OPEN_SCALE / 2]),
        },
    }


@contextmanager
def simulation_server(*, info_override=None, action_override=None):
    policy = SimpleNamespace(
        policy_runtime="dreamfree_stage2_student_native_optimized",
        stage2_checkpoint_path=CHECKPOINT,
        stage2_metadata={"checkpoint_type": "dreamfree_stage2"},
        stage1_checkpoint_path="/models/dreamfree/stage1/step_100000",
        action_horizon=4,
        replan_steps=2,
        num_inference_steps=10,
        optimization_contract={"attention_policy": "sdpa_fused_no_math"},
    )
    info = {**policy_info(policy), **(info_override or {})}
    calls = {"infer": [], "reset": []}
    actions = np.zeros((4, 8), dtype=np.float32)
    actions[:, :7] = np.arange(4)[:, None] + 0.25
    actions[:, 7] = [0, 1, 1, 0]
    if action_override is not None:
        actions = action_override

    def infer(obs, instruction):
        calls["infer"].append((obs, instruction))
        return {"actions": actions}

    with PolicyRpcServer(
        ("127.0.0.1", 0),
        infer_fn=infer,
        reset_fn=lambda seed: calls["reset"].append(seed),
        info_fn=lambda: info,
    ) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.server_address[1], calls
        finally:
            server.shutdown()
            thread.join()


def test_camera_order_and_gripper_proprioception_match_droid_contract():
    original = observation()
    request = rpc_observation([original])
    np.testing.assert_array_equal(
        request["observation/exterior_image_1_left"], original["exo_camera_1"]
    )
    np.testing.assert_array_equal(
        request["observation/exterior_image_2_left"], original["exo_camera_2"]
    )
    np.testing.assert_array_equal(
        request["observation/wrist_image_left"], original["wrist_camera_zed_mini"]
    )
    np.testing.assert_array_equal(request["observation/joint_position"], np.arange(7))
    np.testing.assert_allclose(request["observation/gripper_position"], [0.5])


@pytest.mark.parametrize(
    "fault", ["missing_camera", "short_arm", "nan_gripper", "invalid_rgb", "batch"]
)
def test_invalid_observations_fail_before_inference(fault):
    obs = observation()
    if fault == "missing_camera":
        del obs["exo_camera_2"]
    elif fault == "short_arm":
        obs["qpos"]["arm"] = [0] * 6
    elif fault == "nan_gripper":
        obs["qpos"]["gripper"] = [np.nan]
    elif fault == "invalid_rgb":
        obs["exo_camera_1"] = np.full((2, 2, 3), np.nan)
    else:
        obs = [obs, obs]
    with pytest.raises((KeyError, ValueError)):
        rpc_observation(obs)


def test_rpc_action_chunks_keep_absolute_joints_and_reset_between_episodes():
    with simulation_server() as (port, calls):
        client = MolmoSpacesClient(expected_checkpoint=CHECKPOINT, port=port, replan_steps=2)
        try:
            first = client.get_action(observation(), "pick up the cup")
            second = client.get_action(None, "pick up the cup")
            np.testing.assert_allclose(first["arm"], np.full(7, 0.25))
            np.testing.assert_array_equal(first["gripper"], [0])
            np.testing.assert_allclose(second["arm"], np.full(7, 1.25))
            np.testing.assert_array_equal(second["gripper"], [255])
            assert len(calls["infer"]) == 1
            client.get_action(observation(), "pick up the cup")
            client.reset()
            restarted = client.get_action(observation(), "open the drawer")
            np.testing.assert_allclose(restarted["arm"], first["arm"])
            assert len(calls["infer"]) == 3
            assert calls["infer"][-1][1] == "open the drawer"
            assert len(calls["reset"]) == 2
        finally:
            client.close()


@pytest.mark.parametrize(
    "override",
    [
        {"server": "dreamfree_droid_stage2_real_optimized"},
        {"policy_runtime": "vam_stage2_student_native_optimized"},
        {"checkpoint_path": "/models/another/checkpoint"},
        {"action_representation": "relative_joint"},
        {"action_dim": 7},
        {"action_horizon": 1},
    ],
)
def test_mismatched_server_contract_is_rejected(override):
    with simulation_server(info_override=override) as (port, calls):
        with pytest.raises(ValueError):
            MolmoSpacesClient(expected_checkpoint=CHECKPOINT, port=port, replan_steps=2)
        assert calls["infer"] == []


@pytest.mark.parametrize(
    "fault", ["short_chunk", "wrong_dimension", "nonfinite", "continuous_gripper"]
)
def test_malformed_action_chunks_are_not_executed(fault):
    actions = np.zeros((4, 8), dtype=np.float32)
    if fault == "short_chunk":
        actions = actions[:1]
    elif fault == "wrong_dimension":
        actions = actions[:, :7]
    elif fault == "nonfinite":
        actions[0, 0] = np.nan
    else:
        actions[0, 7] = 0.5
    with simulation_server(action_override=actions) as (port, _):
        client = MolmoSpacesClient(expected_checkpoint=CHECKPOINT, port=port, replan_steps=2)
        try:
            with pytest.raises(ValueError):
                client.get_action(observation(), "pick up the cup")
            assert not client.actions
        finally:
            client.close()


@pytest.mark.parametrize(
    "total, records, successes", [(1, [True], 1), (2, [True], 1), (2, [True, False], 2)]
)
def test_incomplete_or_inconsistent_results_cannot_report_success(total, records, successes):
    result = SimpleNamespace(
        total_count=total,
        success_count=successes,
        episode_results=[SimpleNamespace(success=success) for success in records],
    )
    with pytest.raises(RuntimeError):
        validate_results(result, expected_count=2)


def test_completed_unsuccessful_episodes_are_valid_results():
    result = SimpleNamespace(
        total_count=2,
        success_count=1,
        episode_results=[SimpleNamespace(success=True), SimpleNamespace(success=False)],
    )
    validate_results(result, expected_count=2)


def test_cli_rejects_missing_benchmark_without_loading_simulator(tmp_path):
    with pytest.raises(FileNotFoundError, match="benchmark.json"):
        main(
            [
                "--benchmark-dir",
                str(tmp_path / "missing"),
                "--expected-checkpoint",
                CHECKPOINT,
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )


def test_cli_rejects_reusing_nonempty_results(tmp_path):
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    (benchmark / "benchmark.json").write_text("{}")
    output = tmp_path / "output"
    output.mkdir()
    (output / "results.json").write_text("{}")
    with pytest.raises(ValueError, match="must be empty"):
        main(
            [
                "--benchmark-dir",
                str(benchmark),
                "--expected-checkpoint",
                CHECKPOINT,
                "--output-dir",
                str(output),
            ]
        )


def test_simulation_reset_passes_molmospaces_episode_seed():
    import torch

    with simulation_server(info_override={"requires_episode_seed_on_reset": True}) as (port, calls):
        with torch.random.fork_rng():
            torch.manual_seed(1234)
            client = MolmoSpacesClient(expected_checkpoint=CHECKPOINT, port=port, replan_steps=2)
            try:
                assert calls["reset"][-1] == 1234
                torch.manual_seed(5678)
                client.reset()
                assert calls["reset"][-1] == 5678
            finally:
                client.close()
