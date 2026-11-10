# RoboTwin data and training

Run commands from the repository root after completing [environment setup](../README.md#environment-setup).

RoboTwin data preparation and training are supported. This repository does not include a RoboTwin evaluator.

## Data and training

| Setting | Value |
| --- | --- |
| Dataset selector | `robotwin` |
| Video H × W | 384 × 320 |
| Action / state dimensions | 14 / 14 |
| Camera order | high, left wrist, right wrist |

Provide existing LeRobot dataset roots through `dataset_dirs` (comma-separated for multiple roots). See [dataset preparation](../README.md#dataset-preparation) for the supported dataset formats.

Normalization statistics are required for RoboTwin action training/preparation; set `dataset_stats_path` to the matching statistics file.

Use the [shared training workflow](../README.md#training) with this dataset selector, your `dataset_dirs`, and an `output_dir` for each dataset/stage run. The generic presets use RoboTwin output paths. See [data and artifacts](data.md) for shared cache and checkpoint formats.
