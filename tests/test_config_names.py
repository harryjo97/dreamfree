"""Keep shared option names consistent without invalidating saved artifacts."""

import copy
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dreamfree.config import config_type, load_config
from dreamfree.config_names import TRAINING_RENAMES, normalize_checkpoint_metadata
from dreamfree.models.release import read_checkpoint_metadata

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("stage", [0, 1, 2])
def test_shared_training_fields_and_legacy_error(stage):
    fields = {field.name for field in dataclasses.fields(config_type(stage))}
    assert {
        "video_latent_cache_dir",
        "text_embedding_cache_dir",
        "camera_layout",
        "val_fraction",
        "tokenizer_max_len",
    } <= fields
    assert not (fields & TRAINING_RENAMES.keys())
    assert "context_len" not in fields
    if stage < 2:
        assert {"lora_rank", "lora_alpha", "lora_dropout", "lora_target_modules"} <= fields
    with pytest.raises(ValueError, match="concat_multi_camera: use camera_layout"):
        load_config(
            stage, ROOT / f"configs/stage{stage}_5b.yaml", ["concat_multi_camera=horizontal"]
        )


def test_old_checkpoint_is_normalized_without_rewriting(tmp_path):
    raw = {
        "config": {
            "context_len": 128,
            "tokenizer_max_len": 128,
            "concat_multi_camera": "horizontal",
            "stage1_lora_rank": 128,
            "teacher_state_loss": "cosine",
            "lambda_teacher_state": 1.0,
            "val_set_proportion": 0.01,
            "merge_student_lora_for_inference": True,
            "pretrained_norm_stats": "/datasets/stats.json",
        },
        "dataset": {"context_len": 128, "concat_multi_camera": "horizontal"},
        "release": {"config": {"stage1_lora_rank": 128}},
    }
    original = copy.deepcopy(raw)
    state = tmp_path / "trainer_state.json"
    state.write_text(json.dumps(raw))
    meta = read_checkpoint_metadata(tmp_path, stage="stage1")
    cfg = meta["config"]
    assert cfg["tokenizer_max_len"] == 128 and "context_len" not in cfg
    assert cfg["camera_layout"] == "horizontal" and cfg["lora_rank"] == 128
    assert cfg["distillation_loss"] == "cosine" and cfg["distillation_loss_weight"] == 1.0
    assert cfg["val_fraction"] == 0.01 and cfg["merge_student_lora"] is True
    assert cfg["dataset_stats_path"] == "/datasets/stats.json"
    assert meta["dataset"]["context_len"] == 128
    assert meta["release"] == raw["release"]
    assert normalize_checkpoint_metadata(raw) == meta and raw == original
    assert json.loads(state.read_text()) == original


@pytest.mark.parametrize(
    "config",
    [
        {"context_len": 64, "tokenizer_max_len": 128},
        {"stage1_lora_rank": 32, "lora_rank": 128},
        {"pretrained_norm_stats": "/old", "dataset_stats_path": "/new"},
    ],
)
def test_conflicting_checkpoint_fields_fail(config):
    with pytest.raises(ValueError, match="Conflicting values"):
        normalize_checkpoint_metadata({"config": config})


@pytest.mark.parametrize(
    "name,family,height,width,layout,proprio,expected",
    [
        ("libero", "wan22_ti2v_5b", 224, 448, "horizontal", 8, "312e17f870f10b32"),
        ("robocasa", "wan22_i2v_a14b", 352, 352, "robocasa_dreamzero_352", 9, "cb7c2aa3b06f4323"),
    ],
)
def test_latent_cache_names_match_before_option_rename(
    name, family, height, width, layout, proprio, expected
):
    from dreamfree.data.latent_config import PrecomputeConfig, _precompute_cache_tag

    cfg = PrecomputeConfig(
        model_base_path="/models",
        dataset_dirs=f"/datasets/{name}",
        dataset_name=name,
        dataset_stats_path="/datasets/stats.json",
        video_model_family=family,
        height=height,
        width=width,
        camera_layout=layout,
        action_dim=7,
        proprio_dim=proprio,
    )
    assert _precompute_cache_tag(cfg) == expected


@pytest.mark.parametrize(
    "key,value",
    [
        ("compile_dit", False),
        ("inference_optimization", "false"),
        ("tiled", True),
        ("text_cfg_scale", 7.0),
        ("sigma_shift", 3.0),
        ("negative_prompt", "ignore me"),
    ],
)
def test_unsupported_inference_options_fail_before_loading_weights(key, value):
    from dreamfree.evaluation.policy import build_policy

    with pytest.raises(ValueError, match=key):
        build_policy("libero", {key: value, "checkpoint_path": "/nonexistent"})


def test_inference_aliases_and_false_strings():
    from dreamfree.evaluation.policy import validate_options

    options = validate_options(
        {
            "ckpt_setting": "/policy",
            "rand_device": "cpu",
            "stage1_cuda_graph": "true",
            "flip_images": "false",
            "binarize_gripper": "false",
            "load_text_encoder": "false",
        }
    )
    assert options["checkpoint_path"] == "/policy" and options["noise_device"] == "cpu"
    assert options["encoder_cuda_graph"] == "true"
    assert not any(options[key] for key in ("flip_images", "binarize_gripper", "load_text_encoder"))


def test_robocasa_rejects_disabled_image_flip_before_loading_weights():
    from dreamfree.evaluation.policy import build_policy

    with pytest.raises(ValueError, match="flip_images=true"):
        build_policy("robocasa", {"flip_images": "false", "checkpoint_path": "/nonexistent"})


def test_old_cli_overrides_new_config_without_conflicting_with_defaults(monkeypatch):
    from dreamfree.evaluation import cli

    received = []
    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda _: SimpleNamespace(eval_single_process=received.append),
    )
    cli.evaluate_main(
        "libero",
        [
            "--config",
            str(ROOT / "configs/eval_libero.yaml"),
            "--set",
            "ckpt=/policy",
            "--set",
            "EVALUATION.rand_device=cuda",
        ],
    )
    assert received[0].checkpoint_path == "/policy"
    assert received[0].EVALUATION.noise_device == "cuda"


@pytest.mark.parametrize("flag", ["--checkpoint-path", "--ckpt"])
def test_droid_server_checkpoint_flag_compatibility(flag):
    from dreamfree.evaluation.droid_real.server import _parse_args

    args = _parse_args([flag, "/policy", "--rand-device", "cpu", "--stage1-cuda-graph", "true"])
    assert args.checkpoint_path == "/policy" and args.noise_device == "cpu"
    assert args.encoder_cuda_graph == "true"


@pytest.mark.parametrize("flag", ["--checkpoint-path", "--checkpoint", "--eval-ckpt"])
def test_abc_server_checkpoint_flag_compatibility(flag):
    from dreamfree.evaluation.yam.server import parse_args

    assert parse_args([flag, "/policy", "--action-space", "joint"]).checkpoint_path == Path(
        "/policy"
    )
