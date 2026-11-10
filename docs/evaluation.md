# Evaluation

## Source map

Shared policy assembly lives in [policy.py](../src/dreamfree/evaluation/policy.py).
[checkpoints.py](../src/dreamfree/evaluation/checkpoints.py) owns common encoder
validation, Stage 1/2 lineage checks, saved-config reconstruction, and readout
loading. [options.py](../src/dreamfree/evaluation/options.py) owns generic option
parsing. Benchmark adapters own their geometry checks, observation conversion,
and rollout behavior. RoboCasa's optimization eligibility remains in its own
`inference_options.py`; DROID retains its server-compatible assembly path.

The benchmark policy classes share [runtime.py](../src/dreamfree/evaluation/runtime.py),
which owns encoder/readout conditioning and action denoising. Each benchmark
adapter supplies observation conversion and action postprocessing. Shared
checkpoint helpers are independent of the benchmark adapters.

Evaluation configs use `checkpoint_path`, `dataset_stats_path`, `noise_device`,
and `encoder_cuda_graph` where applicable. Server command-line flags use hyphens,
such as `--checkpoint-path`. Older checkpoint/noise/encoder-graph flag aliases
remain accepted. Set denoising steps with `EVALUATION.num_inference_steps` in
LIBERO/RoboCasa configs or `--num-inference-steps` in server commands.

The LIBERO, RoboCasa, and ABC evaluators require optimized CUDA BF16
inference: `compile_dit=false`, `inference_optimization=false`, and `tiled=true`
are rejected. LIBERO's `binarize_gripper` is configurable and defaults to `true`.
RoboCasa requires `flip_images=true` to match its training pixels; disabling it
raises an error before loading weights.

DreamFree supports the simulation benchmarks and real-world robot deployments below. Each dataset guide provides environment setup and evaluation commands.

## Simulation evaluations

| Benchmark / environment | Evaluation entrypoint | Guide |
| --- | --- | --- |
| LIBERO | `dreamfree.evaluation.libero` | [LIBERO evaluation](libero.md#evaluation) |
| RoboCasa | `dreamfree.evaluation.robocasa` | [RoboCasa evaluation](robocasa.md#evaluation) |
| DROID — Isaac Sim / Isaac Lab | `dreamfree.evaluation.droid_sim` | [Isaac Sim evaluation](droid.md#isaac-sim) |
| DROID — MolmoSpaces (MuJoCo) | `dreamfree.evaluation.droid_sim.molmospaces` | [MolmoSpaces evaluation](droid.md#molmospaces) |
| ABC dual-arm YAM | `dreamfree.evaluation.yam` | [ABC simulation evaluation](abc.md#abc-simulation-evaluation) |

Both DROID simulation backends use `dreamfree.evaluation.droid_sim.server` for model inference. Their simulator dependencies and assets belong to separate environments; see the [DROID simulation guide](droid.md#simulation).

ABC uses `dreamfree.evaluation.yam.server` for inference and a separate simulator environment for scored rollouts. Its guide covers the task suite and joint14/EEF20 checkpoints.

## Real-world evaluations

### DROID

The selected 14B preset uses action history. Isaac Sim, MolmoSpaces, and the real-world client supply issued commands and reset history between episodes. Standard DROID checkpoints remain supported.

Use `dreamfree.evaluation.droid_real.server` for model inference and `dreamfree.evaluation.droid_real.client` on the physical robot. See [DROID physical deployment](droid.md#physical-deployment) for robot-environment setup and observation/prediction checks.

With a **14B video backbone**, the paper reports a mean inference time of **91 ms per action chunk (11 Hz)** on a **single NVIDIA H200**, a **17.8× speedup over DreamZero**. Measurements follow startup warmup and include client preprocessing, policy inference on the server, and client postprocessing; communication and robot execution are excluded. See [Figure 7 and Section 4.4](https://arxiv.org/html/2611.04527v2#S4.SS4) for the results and [Appendix C.2](https://arxiv.org/html/2611.04527v2#A3.SS2) for the measurement setup.

### ABC dual-arm YAM

Use `dreamfree.evaluation.yam_real.client` for physical deployment, with the shared `dreamfree.evaluation.yam.server` or its alias `dreamfree.evaluation.yam_real.server`. Joint14 and EEF20 checkpoints are supported, with bounded IK or explicitly configured OSC for EEF actions. See [ABC physical deployment](abc.md#physical-deployment) for robot setup and controller configuration.

Both ABC simulation and physical deployment accept standard or action-history frozen-student Stage 2 checkpoints, automatically supply execution-owned history when required, and reject action-space mismatches.

## Runtime requirements

Policy entrypoints load DreamFree Stage 2 artifacts and assemble the optimized CUDA BF16 runtime. DROID and ABC also accept action-history checkpoints. Model geometry and normalization must match the benchmark.

The public [DROID](https://huggingface.co/harryjo97/DreamFree-DROID), [RoboCasa](https://huggingface.co/harryjo97/DreamFree-RoboCasa), and [LIBERO](https://huggingface.co/harryjo97/DreamFree-LIBERO) releases load directly after [downloading](../README.md#pretrained-checkpoints). Keep `stage1/`, `stage2/`, and `provenance/` together, and pass either the release root or `stage2/` as the checkpoint. The loader uses the standalone encoder and bundled normalization statistics; neither Stage 0 nor the training dataset is needed. Upstream VAE/text/tokenizer assets download on first use. RoboCasa converts pixels to float32 before direct antialiased bilinear resizing: the wrist fills the upper half, with the left and right views below. Training and evaluation share this implementation.

Native Stage 2 checkpoints require their recorded Stage 1 source. A standalone released source needs no Stage 0; the current loader requires the recorded Stage 0 assets for native LoRA and full-weight sources. See [checkpoint layouts and relocation](data.md#checkpoints). Release bundles also support [Stage 2 training and fine-tuning](../README.md#training-from-released-checkpoints), with a fresh optimizer.

The runtime retains fused SDPA, merged inference LoRA, compiled transformer blocks, and prepared action cross-attention conditioning. Projection fusion and optional CUDA graph/first-frame VAE paths follow the retained per-backbone and per-benchmark contracts. Explicitly requested unsupported optimization settings fail. The first rollout may include compilation/warmup. Full GPU checks matched the tested DROID and RoboCasa actions exactly. LIBERO matches AWAM after LoRA merging, but differs from its historical unmerged runtime; see the [numerical and latency results](validation.md#full-gpu-inference-and-latency).

## Evaluation defaults

The released policies use the settings traced from the selected AWAM experiments:

| Evaluation | Backbone | Denoising steps | Actions per prediction | Seed/settings |
| --- | --- | ---: | ---: | --- |
| LIBERO | 5B | 10 | 10 | Seed 42; 50 trials/task; 30 settling steps; binary gripper |
| RoboCasa | 14B | 10 | 32 | Environment and action seed 195; 50 trials/task; flipped images; deterministic actions |
| DROID simulation / MolmoSpaces | 14B | 10 | 32 | Action-noise seed 42 by default; repeat with 43 and 44 for the recorded comparison |
| DROID real-world | 14B | 24 | 24 | Action-noise seed 0 and episode seed 0 in the documented command |

All three policies predict 32-action chunks. DROID uses eight issued actions as history. The DROID physical command explicitly enables the action and Stage 1 CUDA graphs and first-frame VAE path used by AWAM. LIBERO/RoboCasa can compute text embeddings with the upstream text encoder; their recorded AWAM runs used cached embeddings. To use a prepared cache, set `model.load_text_encoder=false` and `EVALUATION.text_embedding_cache_dir`.

RoboCasa uses direct floating-point resizing consistently across training, cache preparation, and evaluation. Existing video latents must be regenerated to use this preprocessing; use a fresh cache directory. The selected archived AWAM RoboCasa evaluation used a different wrist composition, so its historical success rate is not evidence for the current preprocessing. See [validation](validation.md) for the checks completed on the releases.
