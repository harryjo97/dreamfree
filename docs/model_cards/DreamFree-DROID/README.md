---
license: mit
pipeline_tag: robotics
base_model:
- Wan-AI/Wan2.2-I2V-A14B
tags:
- robotics
- world-model
- action-prediction
- representation-distillation
- diffusion
- wam
- pytorch
---

# DreamFree-DROID: Future Representation Distillation for Imagination-free World Action Models

[![arXiv](https://img.shields.io/badge/arXiv-2611.04527-b31b1b.svg)](https://arxiv.org/abs/2611.04527)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717.svg)](https://github.com/harryjo97/dreamfree)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://harryjo97.github.io/dreamfree/)

**DreamFree-DROID** is a DreamFree checkpoint trained on DROID with a **14B video backbone**.

This repository provides the full Stage 1 visual encoder and the Stage 2 action modules used by the selected DROID policy.

## Model Details

- **Architecture:** WAM with a visual encoder (video DiT), readout, and action DiT.
- **Video backbone:** [Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B).
- **Training data:** DROID.
- **Inputs:** three camera views, a language instruction, robot state, and action history.
- **Outputs:** 32-step action chunks (seven joint-position deltas and a gripper value per step).
- **Encoder format:** BF16 Safetensors.

## Capabilities

- **Imagination-free action prediction:** the policy predicts actions using distilled visual representations without generating future video frames.
- **Action history:** the DROID policy conditions on preceding actions as well as the current observation.
- **DROID evaluation:** the codebase includes physical DROID deployment and simulation integrations for Isaac Sim / Isaac Lab and MolmoSpaces.
- **Reported inference time:** **91 ms per action chunk (11 Hz)** with a 14B video backbone on a single NVIDIA H200 in the paper's DROID real-world setting. See the [paper](https://arxiv.org/abs/2611.04527) for the measurement setup.

## How to Use

Download the checkpoint with `huggingface_hub`:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="harryjo97/DreamFree-DROID",
    local_dir="./checkpoints/DreamFree-DROID",
)
```

The checkpoint is organized as follows:

```text
stage1/
  config.json
  student_dit-*.safetensors
  student_dit.safetensors.index.json
stage2/
  action.safetensors
  readout.safetensors
  conditioner.safetensors
  proprio_encoder.safetensors
  config.json
  dataset_stats.json
provenance/
release.json
LICENSE.md
```

The Stage 1 encoder contains the full trained DiT weights; a separate Stage 0 checkpoint is unnecessary. The VAE, text encoder, and tokenizer remain upstream dependencies. Training metadata and checkpoint lineage are recorded in `provenance/`.

## License

This DreamFree release is licensed under the [MIT License](LICENSE.md). Upstream dependencies retain their respective licenses.

## Citation

```bibtex
@article{jo2026dreamfree,
  title  = {Future Representation Distillation for Imagination-free World Action Models},
  author = {Jaehyeong Jo and Sangwon Jang and Sihyun Yu and
            Seonghyeon Ye and Yekyung Nah and Hojung Jung and
            Kimin Lee and Sung Ju Hwang},
  year   = {2026},
  journal = {arXiv preprint arXiv:2611.04527},
}
```
