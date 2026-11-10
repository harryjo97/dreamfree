# First runs

Run these commands from the repository root after [environment setup](../README.md#environment-setup). Each workflow uses one CUDA GPU. The examples still load the full 5B model; reducing the batch or step count does not make these CPU tests or establish a minimum GPU memory requirement. See [validation](validation.md) for hardware actually exercised.

## Evaluate a release

Prerequisites: a CUDA GPU, network access for the initial model download (or populated model assets), and LIBERO plus its simulator dependencies and task assets installed in the evaluator's environment. No training dataset or Stage 0 checkpoint is required. See [LIBERO setup](libero.md#evaluation).

```bash
uv run huggingface-cli download harryjo97/DreamFree-LIBERO \
  --local-dir ./assets/checkpoints/DreamFree-LIBERO

uv run python -m dreamfree.evaluation.libero \
  --config configs/eval_libero.yaml \
  --set checkpoint_path=./assets/checkpoints/DreamFree-LIBERO \
  --set EVALUATION.task_suite_name=libero_spatial \
  --set EVALUATION.task_id=0 \
  --set EVALUATION.num_trials=1 \
  --set EVALUATION.output_dir=./outputs/quickstart_eval
```

The first call compiles and warms up the policy. The evaluator reports task success and writes results, the resolved evaluation config, and rollout videos under `outputs/quickstart_eval/`. One trial checks the execution path; benchmark results require the full task suite and reference trial counts.

## Fine-tune a release

Prerequisites: the LIBERO release downloaded above and an existing LIBERO LeRobot dataset with videos and metadata. `LIBERO_DATASET_DIRS` below is the only user-supplied path; it can contain comma-separated roots. Fine-tuning uses bundled policy weights and normalization statistics, starts a fresh optimizer, and leaves the visual encoder frozen.

```bash
export LIBERO_DATASET_DIRS=/absolute/path/to/libero_lerobot

uv run python -m dreamfree.training.stage2 \
  --config configs/stage2_libero_5b.json \
  --set init_checkpoint=./assets/checkpoints/DreamFree-LIBERO \
  --set dataset_dirs="$LIBERO_DATASET_DIRS" \
  --set num_epochs=null --set max_steps=2 \
  --set per_device_batch_size=1 --set gradient_accumulation_steps=1 \
  --set num_workers=0 --set dataloader_persistent_workers=false \
  --set compile_dit=false --set use_gradient_checkpointing=true \
  --set eval_every=0 --set save_every=2 \
  --set output_dir=./outputs/quickstart_finetune
```

Expected output: `outputs/quickstart_finetune/checkpoints/step_000002/`, containing policy weights, `trainer_state.json`, and optimizer state. It is a native checkpoint that refers to the downloaded encoder; keep the release available. This two-step check disables periodic evaluation. Set your intended budget, global batch, and evaluation cadence for a real fine-tuning run. `compile_dit=true` and `use_gradient_checkpointing=true` cannot be combined.

## Train all three stages

Prerequisites: the same local LIBERO dataset, sufficient CUDA memory for the full backbone, and upstream Wan model assets (downloaded by the loader on first use). The release download is not needed for this workflow. Text embeddings are prepared automatically; video stays in `live` mode.

This is a two-step wiring check with deliberately random action-backbone initialization. It does **not** reproduce the selected recipes or produce an evaluated policy. Use a fresh output location if these quickstart directories already contain work.

```bash
export LIBERO_DATASET_DIRS=/absolute/path/to/libero_lerobot

uv run python -m dreamfree.training.stage0 \
  --config configs/stage0_libero_5b.json \
  --set dataset_dirs="$LIBERO_DATASET_DIRS" \
  --set num_epochs=null --set max_steps=2 \
  --set per_device_batch_size=1 --set gradient_accumulation_steps=1 \
  --set num_workers=0 --set dataloader_persistent_workers=false \
  --set compile_dit=false --set use_gradient_checkpointing=true \
  --set eval_every=0 --set save_every=2 \
  --set output_dir=./outputs/quickstart_stage0

uv run python -m dreamfree.training.stage1 \
  --config configs/stage1_libero_5b.json \
  --set dataset_dirs="$LIBERO_DATASET_DIRS" \
  --set stage0_lora_checkpoint=./outputs/quickstart_stage0/checkpoints/step_000002 \
  --set stage0_expected_global_step=2 \
  --set num_epochs=null --set max_steps=2 \
  --set per_device_batch_size=1 --set gradient_accumulation_steps=1 \
  --set num_workers=0 --set dataloader_persistent_workers=false \
  --set compile_dit=false --set use_gradient_checkpointing=true \
  --set eval_every=0 --set save_every=2 \
  --set output_dir=./outputs/quickstart_stage1

uv run python -m dreamfree.training.stage2 \
  --config configs/stage2_libero_5b.json \
  --set dataset_dirs="$LIBERO_DATASET_DIRS" \
  --set stage1_checkpoint=./outputs/quickstart_stage1/checkpoints/step_000002 \
  --set skip_dit_load_from_pretrain=true \
  --set num_epochs=null --set max_steps=2 \
  --set per_device_batch_size=1 --set gradient_accumulation_steps=1 \
  --set num_workers=0 --set dataloader_persistent_workers=false \
  --set compile_dit=false --set use_gradient_checkpointing=true \
  --set eval_every=0 --set save_every=2 \
  --set output_dir=./outputs/quickstart_stage2
```

Each command saves `checkpoints/step_000002/` under its output directory. Keep all three directories: the native checkpoints record their source lineage. To reproduce the selected experiments, follow the [preset guide](../configs/README.md), including its dataset selection, checkpoint steps, training budgets, global batches, and [action-backbone initialization](action_backbone.md). Do not carry these smoke-run overrides into a reference reproduction.
