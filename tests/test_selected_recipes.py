"""Keep dataset defaults consistent with the selected run lineage and variants."""

import json
from pathlib import Path

import pytest

from dreamfree.config import load_config

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / "configs/awam_recipe_sources.json").read_text())


@pytest.mark.parametrize("recipe", SOURCES)
def test_selected_lineage_and_global_batches(recipe):
    stages = SOURCES[recipe]
    configs = {
        stage: load_config(int(stage[-1]), ROOT / "configs" / ref["config"])
        for stage, ref in stages.items()
    }
    for stage, cfg in configs.items():
        ref = stages[stage]
        assert (
            cfg.per_device_batch_size
            * cfg.gradient_accumulation_steps
            * ref["reference_world_size"]
            == ref["global_batch_size"]
        )
        assert cfg.max_steps == ref["original_max_steps"]
        assert cfg.num_epochs == ref["original_num_epochs"]
    stage0, stage1, stage2 = (configs[f"stage{n}"] for n in range(3))
    assert stage1.stage0_expected_global_step == stages["stage0"]["selected_step"]
    assert (
        Path(stage1.stage0_lora_checkpoint).name == f"step_{stages['stage0']['selected_step']:06d}"
    )
    assert Path(stage2.stage1_checkpoint).name == f"step_{stages['stage1']['selected_step']:06d}"
    full = recipe.startswith(("droid", "abc"))
    assert stage1.student_train_mode == ("full" if full else "lora")
    assert stage1.use_shared_teacher_student_dit == (not full)
    assert stage1.teacher_target_token_scope == "future"
    assert stage1.future_token_attention == "itself"
    assert stage1.student_text_scope == "first_frame"
    assert stage2.student_train_mode == "frozen"
    assert stage2.readout_num_queries == 64
    assert stage2.action_history_horizon == (8 if full else 0)
    if "14b" in recipe:
        assert stage0.dit_role == stage1.teacher_dit_role == stage1.student_dit_role == "high_noise"
    else:
        assert stage1.student_layer_j == stage2.student_readout_layer == 25


@pytest.mark.parametrize("space", ["joint", "eef"])
def test_abc_history_aliases_match_selected_defaults(space):
    main = ROOT / f"configs/stage2_abc_{space}_14b.json"
    alias = ROOT / f"configs/stage2_abc_{space}_history_14b.json"
    assert json.loads(main.read_text()) == json.loads(alias.read_text())
