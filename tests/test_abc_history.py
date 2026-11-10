"""History conditioning and execution ownership without pretrained weights or robots."""

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors import safe_open
from test_abc_eef import MODEL_PATH
from test_abc_eef import converted as abc_converted
from test_abc_eval import Sim, info, observation
from test_abc_eval import checkpoint as abc_checkpoint
from test_abc_training import write_text

from dreamfree.config import load_config
from dreamfree.data.abc_dataset import ABCCachedDataset
from dreamfree.evaluation import checkpoints
from dreamfree.evaluation.action_protocol import (
    ISSUED_ACTION_HISTORY_IS_PAD_KEY as PAD,
)
from dreamfree.evaluation.action_protocol import (
    ISSUED_ACTION_HISTORY_KEY as HISTORY,
)
from dreamfree.evaluation.action_protocol import (
    IssuedActionHistoryBuffer,
    extract_supplied_action_history,
)
from dreamfree.evaluation.yam import eval_abc, policy
from dreamfree.evaluation.yam_real import client
from dreamfree.evaluation.yam_real.common import EefTargetRejected
from dreamfree.models import action_history, action_training
from dreamfree.models.action_training import ActionTrainingModel, StudentReadoutActionConditioner
from dreamfree.models.wan22.action_dit import ActionDiT
from dreamfree.training.stage2.checkpoint import (
    build_dreamfree_stage2_checkpoint_metadata,
    validate_dreamfree_stage2_checkpoint_metadata,
)

converted = abc_converted
checkpoint = abc_checkpoint
ROOT = Path(__file__).resolve().parents[1]


def config(space="joint", overrides=()):
    return load_config(2, ROOT / f"configs/stage2_abc_{space}_history_14b.json", overrides)


@pytest.mark.parametrize("space,dim", [("joint", 14), ("eef", 20)])
def test_history_presets_keep_student_frozen(space, dim):
    cfg = config(space)
    assert cfg.use_cached_latents is False
    assert (cfg.action_history_horizon, cfg.action_history_dropout) == (8, 0.2)
    assert (cfg.action_dim, cfg.proprio_dim) == (dim, 14)
    assert cfg.student_train_mode == "frozen"


@pytest.mark.parametrize(
    "override",
    [
        "action_history_horizon=-1",
        "action_history_horizon=0",
        "action_history_horizon=992",
        "action_history_dropout=null",
        "action_history_dropout=1",
        "action_history_dropout=-0.1",
        "action_history_dropout=.nan",
        "dataset=robotwin",
        "proprio_obs_horizon=2",
    ],
)
def test_history_configuration_fails_before_data_loading(override):
    with pytest.raises((ValueError, TypeError)):
        config(overrides=[override])


@pytest.mark.skipif(not MODEL_PATH.is_file(), reason="Set ABC_ROOT for conversion integration.")
@pytest.mark.parametrize("space", ["joint", "eef"])
def test_history_reads_only_previous_episode_actions_and_matches_inference(
    converted, tmp_path, space
):
    joint_root, joint_cache, eef_root, _ = converted
    root = eef_root if space == "eef" else joint_root
    cache = eef_root / "action_cache" if space == "eef" else joint_cache
    cfg = config(
        space,
        [
            "data_mode=cached",
            f"dataset_dirs={root}",
            f"video_latent_cache_dir={cache}",
            f"dataset_stats_path={root / 'dataset_stats.json'}",
            f"text_embedding_cache_dir={tmp_path / 'text'}",
        ],
    )
    write_text(tmp_path / "text", cfg)
    dataset = ABCCachedDataset(cfg, split="train")
    with safe_open(str(dataset.records[0][0]), framework="pt", device="cpu") as handle:
        all_actions = handle.get_tensor("action")
    actor = object.__new__(policy.YamPolicy)
    actor.action_history_horizon, actor.action_dim = 8, cfg.action_dim
    actor.device, actor.model_dtype = "cpu", torch.float32
    actor.processor = dataset.processor
    for frame in (0, 3, 8, 10, 35):
        sample = dataset[frame]
        count = min(8, frame)
        expected = torch.zeros(8, cfg.action_dim)
        if count:
            expected[-count:] = all_actions[frame - count : frame]
        torch.testing.assert_close(sample["previous_action"], expected)
        assert sample["previous_action_is_pad"].tolist() == [True] * (8 - count) + [False] * count
        raw = dataset.processor.normalizer.normalizers["action"]["default"].backward(expected)
        raw[sample["previous_action_is_pad"]] = 0
        normalized, pad = actor._history_inputs(
            {HISTORY: raw.numpy(), PAD: sample["previous_action_is_pad"].numpy()}
        )
        torch.testing.assert_close(normalized[0], expected, atol=2e-6, rtol=2e-5)
        assert torch.equal(pad[0], sample["previous_action_is_pad"])
    assert dataset[36]["previous_action_is_pad"].all()
    assert not dataset[36]["previous_action"].any()


def test_dropout_is_samplewise_and_eval_preserves_available_history(monkeypatch):
    encoder = torch.nn.Linear(14, 16)
    previous = torch.ones(2, 8, 14)
    previous[:, :2] = 0
    pad = torch.tensor([[True] * 2 + [False] * 6] * 2)
    proprio = torch.randn(2, 1, 16)
    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.tensor([[0.1], [0.9]]))
    prefix, mask, fractions = action_history.build_query_prefix(
        encoder,
        proprio,
        previous,
        pad,
        horizon=8,
        action_dim=14,
        future_tokens=32,
        dropout=0.2,
        training=True,
    )
    assert mask.shape == (2, 41)
    assert mask[:, 0].all() and mask[:, 9:].all()
    assert not mask[0, 1:9].any() and torch.equal(mask[1, 1:9], ~pad[1])
    torch.testing.assert_close(prefix[0, 1:], encoder(torch.zeros(8, 14)))
    torch.testing.assert_close(fractions, torch.tensor([0.75, 0.375, 0.5]))
    _, eval_mask, fractions = action_history.build_query_prefix(
        encoder,
        proprio,
        previous,
        pad,
        horizon=8,
        action_dim=14,
        future_tokens=32,
        dropout=0.2,
        training=False,
    )
    assert torch.equal(eval_mask[:, 1:9], ~pad)
    assert fractions[-1] == 0


def test_optimized_history_query_matches_training_attention_and_masks_padding():
    torch.manual_seed(11)
    model = ActionDiT(
        hidden_dim=16,
        action_dim=14,
        ffn_dim=32,
        text_dim=12,
        freq_dim=8,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=8,
        num_layers=2,
    ).eval()
    actor = object.__new__(policy.YamPolicy)
    actor.action_model, actor.action_dim, actor.action_history_horizon = model, 14, 8
    previous = torch.randn(2, 8, 14)
    pad = torch.tensor([[True] * 8, [True] * 3 + [False] * 5])
    previous[pad] = 0
    actor._active_previous_action, actor._active_previous_action_is_pad = previous, pad
    proprio = torch.randn(2, 1, 16)
    prefix, mask = actor._prepare_action_query(proprio_query_emb=proprio, action_seq_len=32)
    conditioning = dict(
        context=torch.randn(2, 5, 12),
        context_mask=torch.ones(2, 5, dtype=torch.bool),
        extra_context_emb=torch.randn(2, 3, 16),
        extra_context_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    prepared = model.prepare_cross_attention_conditioning(**conditioning)
    for timestep in (900.0, 100.0):
        kwargs = dict(
            action_tokens=torch.randn(2, 32, 14),
            timestep=torch.full((2,), timestep),
            query_prefix_emb=prefix,
            query_token_mask=mask,
            query_prefix_timestep_sigma=0.001,
        )
        expected = model.forward_with_query_prefix_and_extra_context_emb(**kwargs, **conditioning)
        actual = model.forward_with_query_prefix_and_prepared_context(
            **kwargs, prepared_conditioning=prepared, query_token_mask_validated=True
        )
        torch.testing.assert_close(actual[:, 9:], expected[:, 9:], atol=1e-6, rtol=1e-5)
        poisoned = prefix.clone()
        poisoned[:, 1:][pad] = 1000
        kwargs["query_prefix_emb"] = poisoned
        masked = model.forward_with_query_prefix_and_prepared_context(
            **kwargs, prepared_conditioning=prepared
        )
        torch.testing.assert_close(masked[:, 9:], actual[:, 9:], atol=1e-6, rtol=1e-5)
    actor._active_previous_action = None
    with pytest.raises(ValueError, match="previous_action"):
        actor._prepare_action_query(proprio_query_emb=proprio, action_seq_len=32)


class TinyReadout(torch.nn.Linear):
    def forward(self, tokens):
        value = super().forward(tokens)
        return value, torch.ones(value.shape[:2], dtype=torch.bool)


@pytest.mark.parametrize("space", ["joint", "eef"])
def test_training_loss_uses_future_suffix_and_backpropagates_through_history(monkeypatch, space):
    cfg = config(space)
    cfg.action_hidden_dim, cfg.readout_num_queries = 16, 2
    cfg.action_history_dropout = cfg.proprio_obs_dropout = 0
    core = torch.nn.Module()
    core.dit = torch.nn.Linear(16, 16)
    core.vae = core.text_encoder = None
    core.device, core.torch_dtype = torch.device("cpu"), torch.float32
    core.video_model_family = "wan22_i2v_a14b"
    student = SimpleNamespace(
        core=core, checkpoint_dir=Path("source"), metadata={}, expected_student_tokens_per_frame=2
    )
    monkeypatch.setattr(
        action_training,
        "extract_current_only_student_tokens",
        lambda *a, **k: (core.dit(torch.ones(2, 2, 16)).detach(), {}),
    )
    model = ActionDiT(
        hidden_dim=16,
        action_dim=cfg.action_dim,
        ffn_dim=32,
        text_dim=4096,
        freq_dim=8,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=8,
        num_layers=2,
    )
    wrapper = ActionTrainingModel(
        student=student,
        readout=TinyReadout(16, 16),
        action_model=model,
        conditioner=StudentReadoutActionConditioner(
            student_readout_dim=16, action_context_dim=16, proprio_dim=14, proprio_obs_dropout=0
        ),
        cfg=cfg,
    )
    previous = torch.randn(2, 8, cfg.action_dim)
    previous[:, :3] = 0
    previous.requires_grad_()
    sample = dict(
        input_latents=torch.zeros(2, 16, 3, 54, 48),
        i2v_condition_latents=torch.zeros(2, 16, 3, 54, 48),
        context=torch.randn(2, 128, 4096),
        context_mask=torch.ones(2, 128, dtype=torch.bool),
        action=torch.randn(2, 32, cfg.action_dim),
        proprio=torch.randn(2, 32, 14),
        action_is_pad=torch.tensor([[False] * 30 + [True] * 2] * 2),
        previous_action=previous,
        previous_action_is_pad=torch.tensor([[True] * 3 + [False] * 5] * 2),
    )
    captured = []
    original = model.forward_with_query_prefix_and_extra_context_emb

    def forward(**kwargs):
        result = original(**kwargs)
        result.retain_grad()
        captured.append(result)
        return result

    monkeypatch.setattr(model, "forward_with_query_prefix_and_extra_context_emb", forward)
    loss, metrics = wrapper.training_loss(sample)
    loss.backward()
    assert torch.isfinite(loss) and metrics["query_prefix_tokens"] == 9
    assert not captured[0].grad[:, :9].any()
    assert not captured[0].grad[:, -2:].any()
    assert captured[0].grad[:, 9:-2].abs().sum() > 0
    assert not previous.grad[:, :3].any() and previous.grad[:, 3:].abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in core.parameters())
    assert wrapper.readout.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("space", ["joint", "eef"])
def test_history_checkpoint_admission_and_resume_reject_changed_settings(checkpoint, space):
    path, metadata = checkpoint
    cfg = config(space)
    # The joint fixture owns immutable data; exercise EEF history metadata independently.
    metadata["config"].update(action_history_horizon=8, action_history_dropout=0.2)
    if space == "eef":
        metadata["config"]["dataset_name"] = cfg.dataset_name
    saved_cfg = SimpleNamespace(**metadata["config"])
    metadata.setdefault("shapes", {})
    action_history.add_checkpoint_metadata(metadata, saved_cfg)
    action_history.validate_checkpoint(metadata, saved_cfg)
    if space == "joint":
        policy._validate_dreamfree_stage2_for_eval(path, metadata)
    for key, value in [("action_history_horizon", 16), ("action_history_dropout", 0.1)]:
        changed = copy.copy(saved_cfg)
        setattr(changed, key, value)
        with pytest.raises(ValueError, match="Resume"):
            action_history.validate_checkpoint(metadata, changed)
    for section, field, value in [
        ("action_history", "padding", "right"),
        ("conditioner", "query_prefix_tokens", 1),
        ("shapes", "action_history_horizon", 7),
        ("config", "action_history_horizon", 0),
    ]:
        broken = copy.deepcopy(metadata)
        broken[section][field] = value
        with pytest.raises(ValueError):
            action_history.validate_checkpoint(broken)
    assert metadata["checkpoint_type"] == action_history.CHECKPOINT_TYPE


def test_actual_checkpoint_builder_and_resume_preserve_history_contract(checkpoint):
    path, original = checkpoint
    cfg = config(
        overrides=[
            f"dataset_dirs={path}",
            f"dataset_stats_path={path / 'dataset_stats.json'}",
            f"stage1_checkpoint={path / 'stage1'}",
            "student_readout_layer=2",
        ]
    )
    source = SimpleNamespace(
        checkpoint_dir=path / "stage1",
        global_step=7,
        student_layer_idx=2,
        student_hook_point="post_block",
        student_i2v_condition_mode="first_image_mask_condition",
        expected_student_tokens_per_frame=648,
        expected_student_hidden_dim=5120,
        stage0_lora_config={"enabled": False},
        student_init_mode="shared_stage0_merged_dit_with_stage1_lora",
        stage0_merged_into_student_base=False,
        student_lora_adapter_name="stage1",
    )
    source_metadata = dict(
        student=dict(
            layer_j=2,
            hook_point=source.student_hook_point,
            video_attention_mask_mode="first_frame_causal",
            i2v_condition_mode=source.student_i2v_condition_mode,
        ),
        shapes=dict(tokens_per_frame=648, D_student=5120, latent_shape=[16, 3, 54, 48]),
        teacher_state=dict(loss="normalized_cosine"),
    )
    model = SimpleNamespace(
        student=source,
        source_stage1_metadata=source_metadata,
        source_stage1_checkpoint=str(source.checkpoint_dir),
        last_shape_metadata=dict(action_horizon=32, readout_tokens=64, query_prefix_tokens=9),
    )
    saved = build_dreamfree_stage2_checkpoint_metadata(cfg=cfg, model=model, global_step=5)
    validate_dreamfree_stage2_checkpoint_metadata(cfg, saved, source=source)
    assert saved["checkpoint_type"] == action_history.CHECKPOINT_TYPE
    assert saved["abc_contract"] == original["abc_contract"]
    restored = checkpoints.stage2_config_from_metadata(saved)
    assert (restored.action_history_horizon, restored.action_history_dropout) == (8, 0.2)
    cfg.action_history_dropout = 0.1
    with pytest.raises(ValueError, match="Resume"):
        validate_dreamfree_stage2_checkpoint_metadata(cfg, saved, source=source)


def test_server_warmup_supplies_empty_history(monkeypatch):
    from dreamfree.evaluation.yam import server

    calls = []
    actor = SimpleNamespace(
        action_history_horizon=8,
        action_dim=20,
        predict_action_chunk=lambda obs, _: calls.append(obs),
        optimization_contract={},
        reset=lambda: None,
    )
    monkeypatch.setattr(policy, "build_policy", lambda *a, **k: actor)
    server.initialize_policy(
        SimpleNamespace(
            checkpoint_path="checkpoint",
            replan_steps=8,
            num_inference_steps=16,
            seed=0,
            model_base_path=None,
        )
    )
    history, pad = extract_supplied_action_history(calls[0], horizon=8, action_dim=20)
    assert pad.all() and not history.any()
    assert actor.optimization_contract["warmup_complete"]


class HistoryRPC:
    image_preprocessor = None

    def __init__(self, space):
        self.space, self.requests = space, []

    def reset(self, episode_seed):
        return dict(episode_seed=episode_seed)

    def infer(self, obs, instruction):
        self.requests.append(obs)
        actions = np.full((32, 20 if self.space == "eef" else 14), 0.4, np.float32)
        actions[:, [9, 19] if self.space == "eef" else [6, 13]] = 1.5
        return dict(actions=actions, inference_seconds=0.01)


@pytest.mark.parametrize("space", ["joint", "eef"])
def test_simulation_history_contains_issued_commands_and_resets(space):
    rpc = HistoryRPC(space)
    metadata = dict(info(space), variant="action_history", action_history_horizon=8, replan_steps=1)
    accepted = np.full(20, 0.7, np.float32)
    controller = (
        SimpleNamespace(command=lambda *a, **k: (np.full(14, 0.5, np.float32), accepted, {}))
        if space == "eef"
        else None
    )
    for _ in range(2):
        env = Sim()
        eval_abc.run_episode(env, rpc, metadata, seed=1, max_steps=2, eef_controller=controller)
        first, second = rpc.requests[-2:]
        assert first[PAD].all() and not first[HISTORY].any()
        assert second[PAD].sum() == 7
        np.testing.assert_array_equal(
            second[HISTORY][-1], accepted if controller else env.commands[0]
        )


@pytest.mark.parametrize("mode", ["joint", "ik", "osc"])
def test_physical_history_excludes_rejected_targets_and_recovery_holds(monkeypatch, mode):
    monkeypatch.setattr(client.time, "sleep", lambda _: None)
    space = "joint" if mode == "joint" else "eef"
    rpc = HistoryRPC(space)
    accepted = np.full(20, 0.7, np.float32)
    commands = []

    def step(command):
        commands.append(command.copy())
        return np.full(14, 0.3, np.float32)

    calls = []

    def admit(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise EefTargetRejected("unreachable")
        return np.full(14, 0.3, np.float32), accepted, {}

    env = SimpleNamespace(
        reset=lambda: (observation(), {}),
        step_state=step,
        get_obs=lambda **k: observation(),
        step_eef=admit,
        osc_contract={},
    )
    controller = SimpleNamespace(command=admit, contract={}) if mode == "ik" else None
    args = SimpleNamespace(
        episode_seed=3,
        instruction="move",
        debug=False,
        max_timesteps=3,
        open_loop_horizon=1,
        max_joint_delta=None,
        print_predicted_actions=False,
        eef_controller="osc" if mode == "osc" else "ik",
        eef_rejection_retries=1,
    )
    metadata = dict(info(space), variant="action_history", action_history_horizon=8, replan_steps=1)
    result = client.execute_episode(env, rpc, metadata, args, 0, eef_controller=controller)
    assert result["status"] == "complete"
    counts = [(~req[PAD]).sum() for req in rpc.requests]
    assert counts == ([0, 1, 2] if mode == "joint" else [0, 1, 1, 2])
    for req in rpc.requests[1:]:
        np.testing.assert_array_equal(
            req[HISTORY][-1], commands[0] if mode == "joint" else accepted
        )
    if mode == "ik":
        assert len(commands) == 4  # Three policy commands and one retry hold.


@pytest.mark.parametrize("corruption", ["missing", "shape", "mask", "order", "nonzero", "nan"])
def test_wire_history_is_strict(corruption):
    buffer = IssuedActionHistoryBuffer(horizon=3, action_dim=14)
    buffer.record_issued_action(np.ones(14))
    obs = buffer.add_to_observation({})
    if corruption == "missing":
        del obs[PAD]
    elif corruption == "shape":
        obs[HISTORY] = obs[HISTORY][:, :7]
    elif corruption == "mask":
        obs[PAD] = obs[PAD].astype(int)
    elif corruption == "order":
        obs[PAD] = np.array([False, True, False])
    elif corruption == "nonzero":
        obs[HISTORY][0] = 1
    else:
        obs[HISTORY][-1, 0] = np.nan
    with pytest.raises(ValueError):
        extract_supplied_action_history(obs, horizon=3, action_dim=14)
