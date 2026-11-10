"""Exercise standard joint/EEF rollouts and checkpoint admission without robots or models."""

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dreamfree.config import load_config
from dreamfree.data import yam
from dreamfree.evaluation import checkpoints
from dreamfree.evaluation.yam import eval_abc, policy, server
from dreamfree.evaluation.yam_real import common
from dreamfree.evaluation.yam_real.eef_control import EefTargetRejected

ROOT = Path(__file__).resolve().parents[1]


def observation():
    return dict(
        state=np.full(14, 0.3, np.float32),
        prompt="move",
        images={name: np.zeros((3, 480, 640), np.uint8) for name in common.CAMERAS},
        masks=dict.fromkeys(common.CAMERAS, True),
    )


def info(space):
    return dict(
        protocol=common.PROTOCOL,
        **common.action_contract(yam.ABC_EEF_NAME if space == "eef" else yam.ABC_NAME),
        variant="standard",
        action_history_horizon=0,
        action_horizon=32,
        replan_steps=2,
        fps=30,
        camera_size_hw=[480, 640],
        cameras=list(common.CAMERAS),
        text_mode="encoder",
    )


class Sim:
    _control_decimation = 10
    model = SimpleNamespace(opt=SimpleNamespace(timestep=1 / 300))
    prompt = "move"

    def __init__(self):
        self.commands = []
        self.forgot = False

    def forget_arm_state(self):
        self.forgot = True

    def reset(self, **kwargs):
        assert self.forgot
        return observation(), {}

    def evaluate_task(self):
        return {}

    def step(self, command, **kwargs):
        assert command.shape == (14,)
        self.commands.append(command.copy())
        return (
            observation(),
            0,
            len(self.commands) == 2,
            False,
            dict(task_success=len(self.commands) == 2, task_reward=len(self.commands)),
        )

    def get_state(self):
        return self.commands[-1]

    def get_obs(self, **kwargs):
        return observation()


class RPC:
    def __init__(self, space):
        self.space = space
        self.seed = None
        self.requests = []

    def reset(self, episode_seed):
        self.seed = episode_seed
        return dict(episode_seed=episode_seed)

    def infer(self, obs, instruction):
        assert "deployment/issued_action_history" not in obs
        self.requests.append(obs)
        return dict(
            actions=np.full((32, 20 if self.space == "eef" else 14), 0.4, np.float32),
            inference_seconds=0.01,
        )


@pytest.mark.parametrize("space", ["joint", "eef"])
def test_simulation_executes_standard_policy_in_selected_space(space):
    env, rpc = Sim(), RPC(space)
    calls = []

    def command(action, *, previous, measured):
        calls.append(action.copy())
        assert action.shape == (20,)
        return np.full(14, 0.5, np.float32), action, {}

    controller = SimpleNamespace(command=command) if space == "eef" else None
    eval_abc.validate_server(info(space))
    result = eval_abc.run_episode(
        env, rpc, info(space), seed=19, max_steps=4, eef_controller=controller
    )
    assert result["success"] and result["steps"] == 2 and rpc.seed == 19
    assert len(env.commands) == 2 and len(rpc.requests) == 1
    assert len(calls) == (2 if space == "eef" else 0)


def test_invalid_eef_target_never_reaches_simulator():
    def reject(*args, **kwargs):
        raise EefTargetRejected("unreachable wrist target")

    env = Sim()
    with pytest.raises(EefTargetRejected):
        eval_abc.run_episode(
            env,
            RPC("eef"),
            info("eef"),
            seed=1,
            max_steps=4,
            eef_controller=SimpleNamespace(command=reject),
        )
    assert not env.commands


@pytest.mark.parametrize("space", ["joint", "eef"])
def test_handshake_accepts_history_and_rejects_mismatched_contracts(space):
    metadata = info(space)
    common.validate_server(metadata, action_space=space, replan_steps=2)
    with pytest.raises(ValueError):
        common.validate_server(
            metadata, action_space="joint" if space == "eef" else "eef", replan_steps=2
        )
    metadata.update(variant="action_history", action_history_horizon=8)
    common.validate_server(metadata, action_space=space, replan_steps=2)
    with pytest.raises(ValueError, match="variant mismatch"):
        common.validate_server(
            metadata, action_space=space, replan_steps=2, expected_variant="standard"
        )
    metadata["action_history_horizon"] = 0
    with pytest.raises(ValueError):
        common.validate_server(metadata, action_space=space, replan_steps=2)


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    cfg = load_config(
        2,
        ROOT / "configs/stage2_abc_joint_14b.json",
        [f"dataset_dirs={tmp_path}", f"dataset_stats_path={tmp_path / 'dataset_stats.json'}"],
    )
    stats = {"test": "immutable statistics"}
    (tmp_path / "dataset_stats.json").write_text(json.dumps(stats))
    selection = dict(
        schema_version=1,
        fps=30,
        contract=yam.PRESETS[yam.ABC_NAME].contract,
        normalization_sha256=yam.fingerprint(stats),
        episodes={},
    )
    selection["selection_sha256"] = yam.fingerprint(selection)
    (tmp_path / "yam_selection.json").write_text(json.dumps(selection))
    source = tmp_path / "stage1"
    source.mkdir()
    stage1 = load_config(1, ROOT / "configs/stage1_abc_joint_14b.json")
    cfg.action_history_horizon = 0
    cfg.action_history_dropout = None
    (source / "trainer_state.json").write_text(json.dumps(dict(config=dataclasses.asdict(stage1))))
    monkeypatch.setattr(checkpoints, "validate_stage1_stage2_lineage", lambda **kwargs: None)
    for name in checkpoints.DREAMFREE_STAGE2_REQUIRED_FILES:
        (tmp_path / name).write_bytes(b"fixture")
    metadata = dict(
        checkpoint_type="dreamfree_stage2",
        checkpoint_version=2,
        config=dataclasses.asdict(cfg),
        source_stage1_checkpoint=str(source),
        abc_contract=yam.checkpoint_contract(cfg),
        dataset=dict(
            name=yam.ABC_NAME,
            video_size=[432, 384],
            action_dim=14,
            proprio_dim=14,
            action_horizon=32,
            num_frames=33,
            action_video_freq_ratio=4,
            context_len=128,
            camera_layout=cfg.camera_layout,
            dataset_dirs=[str(tmp_path)],
            dataset_stats_path=cfg.dataset_stats_path,
        ),
        student=dict(
            train_mode="frozen",
            hook_point="post_block",
            video_attention_mask_mode="first_frame_causal",
            token_norm="fixed_tokenwise_layer_norm",
            hidden_dim=5120,
            tokens_per_frame=648,
            i2v_condition_mode="first_image_mask_condition",
        ),
        action=dict(
            action_dim=14,
            proprio_dim=14,
            action_horizon=32,
            output_parameterization="velocity",
            loss_parameterization="flow_matching",
            scheduler=dict(family="continuous_flow_matching"),
            proprio_conditioning=dict(
                mode=cfg.proprio_conditioning_mode,
                obs_horizon=cfg.proprio_obs_horizon,
                state_timestep_sigma=cfg.proprio_state_timestep_sigma,
            ),
        ),
        readout=dict(type="latent_query_cross_attention", input_dim=5120),
        conditioner=dict(
            mode="latent_query_cross_attention",
            token_policy="all_readout_tokens",
            concat_order="text_then_student_readout",
            context_norm="layernorm_affine",
            final_kv_contract="text_plus_student_readout_only",
        ),
        trainable_modules=dict(student=False, readout=True, action_dit=True, conditioner=True),
    )
    return tmp_path, metadata


def test_checkpoint_accepts_standard_abc_and_checks_stats_identity(checkpoint):
    path, metadata = checkpoint
    policy._validate_dreamfree_stage2_for_eval(path, metadata)
    (path / "dataset_stats.json").write_text("{}")
    with pytest.raises(ValueError, match="statistics differ"):
        policy._validate_dreamfree_stage2_for_eval(path, metadata)


@pytest.mark.parametrize(
    "corruption", ["awam", "eef_dims", "layout", "trainable", "history", "selection"]
)
def test_checkpoint_rejects_incompatible_artifacts(checkpoint, corruption):
    path, metadata = checkpoint
    if corruption == "awam":
        metadata["checkpoint_type"] = "vam_stage2"
    elif corruption == "eef_dims":
        metadata["action"]["action_dim"] = 20
    elif corruption == "layout":
        metadata["dataset"]["camera_layout"] = "robotwin"
    elif corruption == "trainable":
        metadata["trainable_modules"]["student"] = True
    elif corruption == "history":
        metadata["training_variant"] = "action_history"
    else:
        metadata["abc_contract"]["selection_sha256"] = "other dataset"
    with pytest.raises(ValueError):
        policy._validate_dreamfree_stage2_for_eval(path, metadata)


def test_server_requires_explicit_matching_action_space(checkpoint, monkeypatch):
    path, metadata = checkpoint
    monkeypatch.setattr(
        policy, "checkpoint_context", lambda _: (path, metadata, path / "dataset_stats.json")
    )
    with pytest.raises(ValueError, match="action space"):
        server.preflight(SimpleNamespace(checkpoint_path=path, action_space="eef"))
    assert server.preflight(SimpleNamespace(checkpoint_path=path, action_space="joint"))[0] == path
