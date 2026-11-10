# LIBERO

Run commands from the repository root after completing [environment setup](../README.md#environment-setup).

## Data and training

| Setting | Value |
| --- | --- |
| Dataset selector | `libero` |
| Video H × W | 224 × 448 |
| Action / state dimensions | 7 / 8 |
| Camera order | image, wrist image |

Provide existing LeRobot dataset roots through `dataset_dirs` (comma-separated for multiple roots). See [dataset preparation](../README.md#dataset-preparation) for the supported dataset formats.

LIBERO can derive normalization statistics from dataset metadata.

Use `configs/stage{0,1,2}_libero_5b.json` with your `dataset_dirs`. The selected recipe uses the 5B backbone, rank-128 LoRA in Stages 0/1, teacher/student layer 25, and 64 Stage 2 readout queries at layer 25. The source data combines the spatial, object, goal, and LIBERO-10 no-noop suites. See the [preset guide](../configs/README.md) for checkpoint lineage, training budgets, and GPU/global-batch counts, and the [shared training workflow](../README.md#training) for preparation commands.

## Evaluation

Read the [shared runtime requirements](evaluation.md) before running the evaluator.

Install LIBERO and its simulation dependencies in the environment used to run the evaluator, and configure its assets as required by LIBERO. After [downloading DreamFree-LIBERO](../README.md#pretrained-checkpoints), run:

```bash
uv run python -m dreamfree.evaluation.libero \
  --config configs/eval_libero.yaml \
  --set checkpoint_path=./assets/checkpoints/DreamFree-LIBERO/stage2 \
  --set EVALUATION.task_suite_name=libero_spatial \
  --set EVALUATION.task_id=0
```

The release supplies normalization statistics automatically. For a native training checkpoint, pass its path and set `EVALUATION.dataset_stats_path` to the matching statistics file.
