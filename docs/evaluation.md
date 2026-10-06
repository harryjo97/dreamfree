# Evaluation

All policy entrypoints load standard stage 2 artifacts and assemble the optimized CUDA BF16 runtime. Keep its source stage 1/stage 0 checkpoints and upstream model weights available. Model geometry and normalization must match the benchmark.

The runtime retains fused SDPA, merged inference LoRA, compiled transformer blocks, and prepared action cross-attention conditioning. Projection fusion and optional CUDA graph/first-frame VAE paths follow the retained per-backbone and per-benchmark contracts. Explicitly requested unsupported optimization settings fail. The first rollout may include compilation/warmup. Optimization does not imply bitwise equality with an unoptimized full policy.

## LIBERO

Install LIBERO and its simulation dependencies in the environment used to run the evaluator, and configure its assets as required by LIBERO. Fill the checkpoint/statistics paths in `configs/eval_libero.yaml`:

```bash
uv run python -m dreamfree.evaluation.libero \
  --config configs/eval_libero.yaml \
  --set ckpt=/path/to/stage2/checkpoint \
  --set EVALUATION.dataset_stats_path=/path/to/libero_stats.json \
  --set EVALUATION.task_suite_name=libero_spatial \
  --set EVALUATION.task_id=0
```

## RoboCasa

Use an environment with the matching RoboCasa/robosuite installation and assets. Set `ROBOCASA_ROOT` and `ROBOCASA_CONTROLLER_CONFIG`, or override their YAML fields with paths.

```bash
uv run python -m dreamfree.evaluation.robocasa \
  --config configs/eval_robocasa.yaml \
  --set ckpt=/path/to/stage2/checkpoint \
  --set EVALUATION.dataset_stats_path=/path/to/robocasa_stats.json \
  --set EVALUATION.task_name=TurnOffMicrowave
```

## RoboTwin

Set `ROBOTWIN_ROOT` to the simulator checkout. The launcher writes a resolved configuration and invokes that checkout's `script/eval_policy.py`, using the `get_model`, `eval`, and `reset_model` callbacks in `dreamfree.evaluation.robotwin`.

```bash
uv run python -m dreamfree.evaluation.robotwin \
  --config configs/eval_robotwin.yaml \
  --set task_name=your_task \
  --set task_config=your_simulator_task_config \
  --set ckpt_setting=/path/to/stage2/checkpoint \
  --set dataset_stats_path=/path/to/robotwin_stats.json
```

Set `python=/path/to/simulator/python` if the engine uses a separate interpreter. That interpreter must have the policy's CUDA/PyTorch dependencies available.

## DROID simulation

Run the model server in the DreamFree environment:

```bash
uv run python -m dreamfree.evaluation.droid.server \
  --ckpt /path/to/stage2/checkpoint \
  --droid-dataset-root /path/to/droid_dataset \
  --port 6100
```

Run the client in the external IsaacLab/`sim_evals` environment, with this repository's `src` on `PYTHONPATH`:

```bash
PYTHONPATH=/path/to/dreamfree/src python -m dreamfree.evaluation.droid \
  --scene 1 --episodes 10 --port 6100
```

Scenes 1–4 use the simulator's baseline assets. Results go under `outputs/evaluation/`. The simulator and model server can use different Python environments.

## Physical DROID

The server runs in the DreamFree environment; the client remains compatible with Python 3.7 and the robot's DROID environment. Keep the pickle RPC endpoint on localhost, using an SSH tunnel when the model server is remote.

```bash
uv run python -m dreamfree.evaluation.droid_real.server \
  --ckpt /path/to/stage2/checkpoint \
  --droid-dataset-root /path/to/droid_dataset \
  --port 6100
```

Start with an observation/prediction check on the robot machine:

```bash
PYTHONPATH=/path/to/dreamfree/src python -m dreamfree.evaluation.droid_real.client \
  --remote-port 6100 \
  --left-camera-id LEFT --right-camera-id RIGHT --wrist-camera-id WRIST \
  --expected-checkpoint /path/on/server/to/stage2/checkpoint \
  --instruction 'pick up the cup' \
  --observe-only --print-predicted-actions
```

The client checks checkpoint/runtime identity, image preprocessing, action shape/finite values, and joint position/rate limits. Removing `--observe-only` enables action execution. The repository's validation does not move a robot.
