"""Automatic text preparation before dataset workers and backbone allocation."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from test_data_pipeline import write_metadata

from dreamfree.config import load_config
from dreamfree.data import prepare_text
from dreamfree.data.distillation import format_stage1_prompt, stage1_text_cache_path
from dreamfree.data.text_cache import load_text_context

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def encoder(monkeypatch):
    batches = []
    weights = SimpleNamespace(path="fixture", download_if_necessary=lambda: None)
    monkeypatch.setattr(
        prepare_text, "_resolve_configs", lambda *args, **kwargs: (None, weights, None, weights)
    )

    class Encoder:
        def eval(self):
            return self

        def __call__(self, ids, mask):
            return torch.ones(len(ids), 128, 4096, dtype=torch.bfloat16)

    def load(*args, **kwargs):
        assert kwargs["pretrained_load_broadcast"] is False
        return Encoder()

    class Tokenizer:
        def __init__(self, **kwargs):
            assert kwargs["seq_len"] == 128

        def __call__(self, prompts, **kwargs):
            batches.append(list(prompts))
            mask = torch.arange(128).expand(len(prompts), -1) < 8
            return torch.zeros_like(mask, dtype=torch.long), mask

    monkeypatch.setattr(prepare_text, "_load_registered_model", load)
    monkeypatch.setattr(prepare_text, "HuggingfaceTokenizer", Tokenizer)
    return batches


def config(tmp_path, stage, mode="live"):
    cfg = load_config(stage, ROOT / f"configs/stage{stage}_14b.yaml", [f"data_mode={mode}"])
    root = tmp_path / "dataset"
    write_metadata(root, ["observation.images.camera"])
    cfg.dataset_dirs = str(root)
    cfg.text_embedding_cache_dir = str(tmp_path / "text")
    cfg.output_dir = str(tmp_path / "output")
    return cfg


def accelerator():
    return SimpleNamespace(
        is_main_process=True,
        num_processes=1,
        process_index=0,
        device=torch.device("cpu"),
        distributed_type="NO",
        wait_for_everyone=lambda: None,
    )


@pytest.mark.parametrize("stage", [0, 1, 2])
@pytest.mark.parametrize("mode", ["live", "cached"])
def test_missing_entries_are_created_and_valid_entries_are_reused(tmp_path, encoder, stage, mode):
    cfg = config(tmp_path, stage, mode)
    prepare_text.ensure_training_text_cache(cfg, accelerator())
    prompt = format_stage1_prompt("pick up the cup")
    path = stage1_text_cache_path(cfg.text_embedding_cache_dir, prompt, 128, cfg.model_id)
    original = path.read_bytes()
    assert encoder == [[prompt]]
    context, mask = load_text_context(
        cache_dir=cfg.text_embedding_cache_dir,
        prompt=prompt,
        context_len=128,
        text_dim=4096,
        model_id=cfg.model_id,
    )
    assert mask.all() and context[:8].all() and not context[8:].any()
    prepare_text.ensure_training_text_cache(cfg, accelerator())
    assert encoder == [[prompt]]
    episodes = Path(cfg.dataset_dirs) / "meta/episodes.jsonl"
    episodes.write_text(json.dumps({"episode_index": 0, "tasks": ["put down the cup"]}) + "\n")
    prepare_text.ensure_training_text_cache(cfg, accelerator())
    assert encoder == [[prompt], [format_stage1_prompt("put down the cup")]]
    assert path.read_bytes() == original


def test_secondary_directory_is_reused_and_missing_prompts_go_to_first(tmp_path, encoder):
    cfg = config(tmp_path, 2)
    prepare_text.ensure_training_text_cache(cfg, accelerator())
    first = tmp_path / "new-text"
    cfg.text_embedding_cache_dir = f"{first},{cfg.text_embedding_cache_dir}"
    prepare_text.ensure_training_text_cache(cfg, accelerator())
    assert first.is_dir() and not list(first.iterdir()) and len(encoder) == 1
    tasks = Path(cfg.dataset_dirs) / "meta/tasks.jsonl"
    tasks.write_text(json.dumps({"task_index": 1, "task": "new instruction"}) + "\n")
    prepare_text.ensure_training_text_cache(cfg, accelerator())
    assert len(list(first.glob("*.pt"))) == 1 and len(encoder) == 2


def test_invalid_existing_cache_is_not_overwritten(tmp_path, encoder):
    cfg = config(tmp_path, 2)
    prepare_text.ensure_training_text_cache(cfg, accelerator())
    path = next(Path(cfg.text_embedding_cache_dir).glob("*.pt"))
    torch.save({"format": "incompatible"}, path)
    original = path.read_bytes()
    with pytest.raises(RuntimeError, match="Automatic text-cache preparation failed"):
        prepare_text.ensure_training_text_cache(cfg, accelerator())
    assert path.read_bytes() == original and len(encoder) == 1


def test_v3_episode_prompts_and_generic_stage0_prompt(tmp_path):
    meta = tmp_path / "meta"
    (meta / "episodes").mkdir(parents=True)
    pq.write_table(pa.table({"task": ["task label"]}), meta / "tasks.parquet")
    pq.write_table(
        pa.table({"tasks": [["episode-specific instruction"]]}), meta / "episodes/file.parquet"
    )
    assert set(prepare_text.collect_prompts([tmp_path])) == {
        format_stage1_prompt("task label"),
        format_stage1_prompt("episode-specific instruction"),
    }
    cfg = SimpleNamespace(prompt_policy="generic")
    assert prepare_text.collect_prompts([tmp_path], cfg=cfg) == [
        format_stage1_prompt("a robot manipulation task")
    ]


@pytest.mark.parametrize("stage", [0, 1, 2])
def test_trainers_prepare_text_before_constructing_datasets(tmp_path, monkeypatch, encoder, stage):
    cfg = config(tmp_path, stage)
    loop = importlib.import_module(f"dreamfree.training.stage{stage}.loop")
    monkeypatch.setattr(loop, "Accelerator", lambda **kwargs: accelerator())
    if stage == 0:
        cfg.export_final = False
    elif stage == 1:
        from dreamfree.models import artifacts
        from dreamfree.training.stage1.validation import validate_stage1_config

        cfg.stage0_lora_checkpoint = str(tmp_path / "stage0")
        cfg.stage0_expected_lora_dropout = 0.0
        monkeypatch.setattr(artifacts, "resolve_stage0_expectations", lambda cfg: None)
        monkeypatch.setattr(loop, "validate_stage1_launch_config", validate_stage1_config)
    else:
        cfg.stage1_checkpoint = str(tmp_path / "stage1")
        cfg.dataset_stats_path = str(tmp_path / "stats.json")
        Path(cfg.dataset_stats_path).write_text("{}")
        monkeypatch.setattr(loop, "_apply_stage1_checkpoint_model_contract", lambda *a, **k: None)
        monkeypatch.setattr(
            loop,
            "validate_stage1_student_source",
            lambda cfg: SimpleNamespace(checkpoint_dir=cfg.stage1_checkpoint),
        )

    class DatasetReached(Exception):
        pass

    def build(*args, **kwargs):
        assert encoder == [[format_stage1_prompt("pick up the cup")]]
        assert len(list(Path(cfg.text_embedding_cache_dir).glob("*.pt"))) == 1
        raise DatasetReached

    builder = ("build_stage0_dataset", "_build_stage1_dataset", "_build_stage2_dataset")[stage]
    monkeypatch.setattr(loop, builder, build)
    run = getattr(
        loop, "run_dreamfree_stage2_training" if stage == 2 else f"run_stage{stage}_training"
    )
    with pytest.raises(DatasetReached):
        run(cfg)


def test_non_main_rank_waits_for_preparation_and_receives_errors(tmp_path, monkeypatch):
    cfg = config(tmp_path, 2)
    rank = accelerator()
    rank.is_main_process, rank.num_processes = False, 2
    monkeypatch.setattr(
        prepare_text, "collect_prompts", lambda *a, **k: pytest.fail("Worker prepared text")
    )
    messages = []

    def broadcast(status, from_process):
        assert from_process == 0
        messages.append(status.copy())
        status[0] = "OSError: encoder weights unavailable"

    monkeypatch.setattr(prepare_text, "broadcast_object_list", broadcast)
    with pytest.raises(RuntimeError, match="encoder weights unavailable"):
        prepare_text.ensure_training_text_cache(cfg, rank)
    assert messages == [[None]]


@pytest.mark.parametrize("stage", [0, 1, 2])
def test_empty_cache_destination_resolves_without_creating_it(tmp_path, monkeypatch, stage):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(stage, ROOT / f"configs/stage{stage}_14b.yaml")
    assert cfg.text_embedding_cache_dir == "./assets/text/robotwin_14b"
    assert not Path(cfg.text_embedding_cache_dir).exists()
