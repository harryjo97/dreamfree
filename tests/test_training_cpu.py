"""Exercise each recipe's real training loops with tiny CPU backbones.

Weights, VAE encoding, and dataset samples are fixtures. Configuration validation,
losses, LoRA/full training, optimizers, evaluation, checkpoint lineage, and resume
use production code. Separate data-pipeline tests exercise real dataset readers.
"""

import importlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from accelerate.state import AcceleratorState, GradientState
from test_abc_eef import MODEL_PATH
from test_abc_eef import converted as converted
from test_abc_training import write_text
from test_data_pipeline import write_metadata, write_text_cache
from torch.utils.data import Dataset

from dreamfree.config import load_config
from dreamfree.models.wan22.wan22 import Wan22Core
from dreamfree.models.wan22.wan_video_dit import WanVideoDiT

ROOT = Path(__file__).resolve().parents[1]
RECIPES = (
    "libero_5b",
    "robocasa_14b",
    "droid_14b",
    "robotwin_5b",
    "robotwin_14b",
    "abc_joint_14b",
    "abc_eef_14b",
)


class FixtureVAE(torch.nn.Module):
    def __init__(self, family):
        super().__init__()
        self.channels, self.stride = (48, 16) if family == "wan22_ti2v_5b" else (16, 8)

    def encode(self, video, **kwargs):
        frames, height, width = video.shape[-3:]
        # Keep frame 0 identical between the full video and I2V pseudo-video.
        sampled = video[:, :, ::4].mean(1, keepdim=True)
        return F.adaptive_avg_pool3d(
            sampled, ((frames - 1) // 4 + 1, height // self.stride, width // self.stride)
        ).repeat(1, self.channels, 1, 1, 1)


def tiny_core(**kwargs):
    assert str(kwargs["device"]) == "cpu"
    config = dict(kwargs["dit_config"])
    config.update(hidden_dim=16, ffn_dim=32, freq_dim=8, num_heads=2, attn_head_dim=8, num_layers=2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        dit = WanVideoDiT(**config)
    return Wan22Core(
        dit,
        vae=FixtureVAE(kwargs["video_model_family"]) if kwargs.get("load_vae", True) else None,
        text_encoder=None,
        tokenizer=None,
        device="cpu",
        torch_dtype=torch.float32,
        video_model_family=kwargs["video_model_family"],
        dit_role=kwargs["dit_role"],
    )


class FixtureDataset(Dataset):
    total_windows = 4

    def __init__(self, cfg, stage):
        self.sample = dict(
            video=torch.full((3, 9, cfg.height, cfg.width), 0.25),
            context=torch.ones(128, 4096),
            context_mask=torch.ones(128, dtype=torch.bool),
            prompt="move",
            task="move",
            sample_id="cpu-fixture",
        )
        if stage == 2:
            self.sample.update(
                action=torch.full((32, cfg.action_dim), 0.3),
                proprio=torch.full((32, cfg.proprio_dim), 0.2),
                action_is_pad=torch.zeros(32, dtype=torch.bool),
                proprio_is_pad=torch.zeros(32, dtype=torch.bool),
                image_is_pad=torch.zeros(9, dtype=torch.bool),
            )
            if cfg.action_history_horizon:
                self.sample.update(
                    previous_action=torch.zeros(cfg.action_history_horizon, cfg.action_dim),
                    previous_action_is_pad=torch.ones(cfg.action_history_horizon, dtype=torch.bool),
                )

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return self.sample


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()
    monkeypatch.setattr(Wan22Core, "from_wan22_pretrained", staticmethod(tiny_core))
    yield
    assert not torch.cuda.is_initialized()
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()


@pytest.mark.parametrize("recipe", RECIPES)
def test_three_stage_training_validation_checkpoint_and_resume(
    tmp_path, monkeypatch, request, recipe
):
    abc = recipe.startswith("abc")
    if abc and not MODEL_PATH.is_file():
        pytest.skip("Set ABC_ROOT for ABC prepared-data fixtures.")
    abc_data = request.getfixturevalue("converted") if abc else None
    previous = None
    for stage in range(3):
        name = "abc_joint_14b" if abc and stage < 2 else recipe
        filename = name.removeprefix("robotwin_") if name.startswith("robotwin_") else name
        suffix = "yaml" if name.startswith("robotwin_") else "json"
        cfg = load_config(stage, ROOT / f"configs/stage{stage}_{filename}.{suffix}")
        cfg = replace(
            cfg,
            mixed_precision="no",
            num_epochs=None,
            max_steps=2,
            per_device_batch_size=1,
            gradient_accumulation_steps=2,
            num_workers=0,
            dataloader_persistent_workers=False,
            compile_dit=False,
            use_gradient_checkpointing=False,
            eval_every=1,
            eval_batches=1,
            save_every=1,
            log_every=1,
            output_dir=str(tmp_path / f"stage{stage}"),
            text_embedding_cache_dir=str(tmp_path / f"text{stage}"),
        )
        if abc:
            root = abc_data[2] if stage == 2 and "eef" in recipe else abc_data[0]
            write_text(Path(cfg.text_embedding_cache_dir), cfg)
        else:
            root = tmp_path / f"data{stage}"
            from dreamfree.config import DATA_DEFAULTS

            write_metadata(root, DATA_DEFAULTS[cfg.dataset_name]["camera_keys"].split(","))
            write_text_cache(Path(cfg.text_embedding_cache_dir), cfg)
            (root / "dataset_stats.json").write_text("{}")
        cfg.dataset_dirs = str(root)
        loop = importlib.import_module(f"dreamfree.training.stage{stage}.loop")
        builder = ("build_stage0_dataset", "_build_stage1_dataset", "_build_stage2_dataset")[stage]
        monkeypatch.setattr(
            loop, builder, lambda *args, cfg=cfg, stage=stage, **kwargs: FixtureDataset(cfg, stage)
        )
        if stage == 0:
            cfg.lora_rank, cfg.lora_alpha = 2, 2.0
            cfg.export_final, cfg.export_every, cfg.eval_num_inference_steps = False, 0, 0
        else:
            original_hidden = cfg.video_hidden_dim
            cfg.video_hidden_dim = 16
            validation_name = (
                "validate_stage1_launch_config"
                if stage == 1
                else "validate_dreamfree_stage2_config"
            )
            validate = getattr(loop, validation_name)
            # Only the backbone width is reduced; validate the production family width.
            monkeypatch.setattr(
                loop,
                validation_name,
                lambda cfg, validate=validate, hidden=original_hidden: validate(
                    replace(cfg, video_hidden_dim=hidden)
                ),
            )
            if stage == 1:
                cfg.stage0_lora_checkpoint, cfg.stage0_expected_global_step = str(previous), 3
                cfg.stage0_expected_lora_rank, cfg.stage0_expected_lora_alpha = 2, 2.0
                cfg.stage0_eval_reference = None
                cfg.teacher_layer_k = cfg.student_layer_j = 1
                cfg.projector_dim, cfg.projector_num_heads, cfg.projector_num_layers = 16, 2, 1
                cfg.lora_rank, cfg.lora_alpha = 2, 2.0
                if cfg.student_train_mode == "full":
                    cfg.warmup_steps = 0
            else:
                cfg.stage1_checkpoint = str(previous)
                cfg.dataset_stats_path = str(root / "dataset_stats.json")
                cfg.student_readout_layer = 1
                cfg.action_hidden_dim, cfg.action_ffn_dim, cfg.action_freq_dim = 16, 32, 8
                cfg.action_num_heads, cfg.action_attn_head_dim, cfg.action_num_layers = 2, 8, 2
                cfg.readout_num_queries, cfg.readout_num_heads, cfg.readout_num_layers = 2, 2, 1
                cfg.skip_dit_load_from_pretrain = True
        run = getattr(
            loop, "run_dreamfree_stage2_training" if stage == 2 else f"run_stage{stage}_training"
        )
        run(cfg)
        saved = Path(cfg.output_dir) / "checkpoints/step_000002"
        assert json.loads((saved / "trainer_state.json").read_text())["global_step"] == 2
        optim = torch.load(saved / "optim.pt", map_location="cpu", weights_only=False)
        assert optim["optimizer"]["state"]
        assert all(
            torch.isfinite(value).all()
            for state in optim["optimizer"]["state"].values()
            for value in state.values()
            if torch.is_tensor(value)
        )
        cfg.resume, cfg.max_steps = str(saved), 3
        run(cfg)
        previous = Path(cfg.output_dir) / "checkpoints/step_000003"
        assert json.loads((previous / "trainer_state.json").read_text())["global_step"] == 3
