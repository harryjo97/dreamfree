# DROID

Run commands from the repository root after completing [environment setup](../README.md#environment-setup).

## Data and training

| Setting | Value |
| --- | --- |
| Dataset selector | `droid` |
| Video H × W | 352 × 640 |
| Action / state dimensions | 8 / 8 |
| Camera order | exterior left, exterior right, wrist |

Provide existing LeRobot dataset roots through `dataset_dirs` (comma-separated for multiple roots). See [dataset preparation](../README.md#dataset-preparation) for the supported dataset formats.

DROID can derive normalization statistics from dataset metadata and requires `meta/modality.json` with joint-position and gripper ranges.

DROID preserves the joint-position-delta/gripper convention and uses the recorded modality ranges and quantile normalization statistics. Its video caches include raw joint/gripper tensors so sample-relative action normalization remains exact.

Use `configs/stage{0,1,2}_droid_14b.json` with your `dataset_dirs`. The selected full-DROID recipe uses the 14B high-noise backbone, rank-128 Stage 0 LoRA, full-weight Stage 1 distillation at layer 31, and 64 Stage 2 readout queries at layer 31. Stage 2 includes eight preceding actions with history dropout 0.2. Full Stage 1 uses the two-node FSDP2 launcher. See the [preset guide](../configs/README.md) for checkpoint lineage, training budgets, and GPU/global-batch counts, and the [shared training workflow](../README.md#training) for preparation commands.

## Evaluation

Simulation evaluation lives in `dreamfree.evaluation.droid_sim` and supports Isaac Sim and MolmoSpaces. `dreamfree.evaluation.droid_real` is for real-world DROID experiments. Read the [shared runtime requirements](evaluation.md) before starting either server.

The commands below use the public [DreamFree-DROID release](../README.md#pretrained-checkpoints). Its Stage 1 encoder and normalization statistics load automatically; Stage 0 and the DROID training dataset are not required. For native checkpoints, provide `--dataset-stats-path` or `--droid-dataset-root` along with the recorded source checkpoints.

The server accepts standard and action-history checkpoints. The selected default
uses action history in Isaac Sim, MolmoSpaces, and physical deployment. Clients
send only issued commands, clear history on reset, and send explicit empty
history at episode start. The server reanchors past absolute joint targets to
the current joint state, retains binary gripper commands, applies training
normalization, and masks left padding. Unexecuted chunk tails never enter history.

### Simulation

Both simulation backends use the same model server, with 10 denoising steps and 32-step replanning. The selected AWAM MolmoSpaces comparison used action-noise seeds 42, 43, and 44. The server defaults to 42; set `--action-noise-seed` for each repeat. MolmoSpaces passes its episode torch seed at reset, matching AWAM. Run the server in the DreamFree environment:

```bash
uv run python -m dreamfree.evaluation.droid_sim.server \
  --checkpoint-path ./assets/checkpoints/DreamFree-DROID/stage2 \
  --port 6100
```

#### Isaac Sim

The Isaac Sim evaluator uses Isaac Lab and the external `sim_evals` DROID environments. Install their simulator dependencies and assets separately. Run the client in that environment, with this repository's `src` on `PYTHONPATH`:

```bash
PYTHONPATH=/path/to/dreamfree/src python -m dreamfree.evaluation.droid_sim \
  --scene 1 --episodes 10 --port 6100
```

Scenes 1–4 use the simulator's baseline assets. Results go under `outputs/evaluation/`. The simulator and model server can use different Python environments.

#### MolmoSpaces

MolmoSpaces evaluation uses its MuJoCo environment, Franka DROID JSON benchmarks, and benchmark assets. The integration requires the MolmoSpaces `run_evaluation` and `JsonBenchmarkEvalConfig` APIs. Run the following from your MolmoSpaces checkout in its environment, with DreamFree's `src` on `PYTHONPATH`:

```bash
PYTHONPATH=/path/to/dreamfree/src python -m dreamfree.evaluation.droid_sim.molmospaces \
  --benchmark-dir /path/to/molmospaces/benchmark \
  --expected-checkpoint /absolute/path/on/server/to/stage2/checkpoint \
  --output-dir /path/to/outputs/molmospaces_run \
  --max-episodes 10 --replan-steps 32 --port 6100
```

`--benchmark-dir` must contain `benchmark.json`. `--expected-checkpoint` must match the absolute checkpoint path reported by the simulation server. Use an empty output directory for each run. Select particular benchmark episodes with `--episode-indices 0 3 5` instead of `--max-episodes`, and add `--save-videos` to save rollout videos. `--task-horizon-steps` overrides the benchmark's default episode horizon.

The adapter preserves the existing MolmoSpaces camera mapping and gripper-proprioception scale (`0.824033`, configurable with `--gripper-qpos-open-scale`). Image preprocessing runs on the model server. Actions are absolute seven-joint targets with binary gripper commands mapped to MolmoSpaces' 0–255 range. The policy runs at a 66 ms simulation interval, executes `--replan-steps` actions per prediction, and clears pending actions at each episode reset. Benchmark initialization, success criteria, and result writing use the MolmoSpaces evaluator.

The client checks the DreamFree runtime, checkpoint identity, action representation, and action shapes before using predictions. Evaluation fails if any selected episode lacks a result record or aggregate success counts disagree with the records. Keep the model server on localhost; use an SSH tunnel when simulation and inference run on different machines. Simulator dependencies and assets are not installed by DreamFree's `uv sync`.

### Physical deployment

Use `dreamfree.evaluation.droid_real` for experiments on a physical DROID robot.

The server runs in the DreamFree environment; the client remains compatible with Python 3.7 and the robot's DROID environment. Keep the pickle RPC endpoint on localhost, using an SSH tunnel when the model server is remote.

```bash
uv run python -m dreamfree.evaluation.droid_real.server \
  --checkpoint-path ./assets/checkpoints/DreamFree-DROID/stage2 \
  --replan-steps 24 --num-inference-steps 24 \
  --random-action-noise --action-noise-seed 0 \
  --action-cuda-graph true --encoder-cuda-graph true \
  --vae-first-frame-only true --compile-auxiliary none \
  --port 6100
```

These are the recorded AWAM settings for the released action-history checkpoint. The client executes 24 actions from each 32-action prediction.

Start with an observation/prediction check on the robot machine:

```bash
PYTHONPATH=/path/to/dreamfree/src python -m dreamfree.evaluation.droid_real.client \
  --remote-port 6100 \
  --left-camera-id LEFT --right-camera-id RIGHT --wrist-camera-id WRIST \
  --expected-checkpoint /path/on/server/to/stage2/checkpoint \
  --expected-action-noise-seed 0 --episode-seed 0 \
  --open-loop-horizon 24 \
  --instruction 'pick up the cup' \
  --observe-only --print-predicted-actions
```

The client checks checkpoint/runtime identity, image preprocessing, action shape/finite values, and joint position/rate limits. Removing `--observe-only` enables action execution. The repository's validation does not move a robot.
