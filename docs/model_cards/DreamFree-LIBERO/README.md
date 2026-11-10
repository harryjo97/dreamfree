---
license: mit
pipeline_tag: robotics
base_model:
- Wan-AI/Wan2.2-TI2V-5B
tags:
- robotics
- world-model
- action-prediction
- representation-distillation
- diffusion
- wam
- pytorch
---

# DreamFree-LIBERO: Future Representation Distillation for Imagination-free World Action Models

[![arXiv](https://img.shields.io/badge/arXiv-2611.04527-b31b1b.svg)](https://arxiv.org/abs/2611.04527)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717.svg)](https://github.com/harryjo97/dreamfree)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://harryjo97.github.io/dreamfree/)

**DreamFree-LIBERO** is a DreamFree checkpoint trained on LIBERO with a **5B video backbone**.

This repository provides the full Stage 1 visual encoder and the Stage 2 action modules used by the selected LIBERO policy.

## Model Details

- **Architecture:** WAM with a visual encoder (video DiT), readout, and action DiT.
- **Video backbone:** [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B).
- **Training data:** LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-10.
- **Inputs:** two camera views, a language instruction, and robot state.
- **Outputs:** 32-step chunks of 7D actions.
- **Encoder format:** BF16 Safetensors.

## Capabilities

- **Imagination-free action prediction:** the policy predicts actions using distilled visual representations without generating future video frames.
- **Language-conditioned manipulation:** the policy combines camera observations, an instruction, and proprioception to predict action chunks for LIBERO tasks.
- **LIBERO evaluation:** the selected recipe combines all four listed suites and uses the corresponding action normalization and image preprocessing.

## How to Use

Download the checkpoint with `huggingface_hub`:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="harryjo97/DreamFree-LIBERO",
    local_dir="./checkpoints/DreamFree-LIBERO",
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

The Stage 1 encoder contains the full merged DiT weights; separate Stage 0 and Stage 1 adapters are unnecessary. The VAE, text encoder, and tokenizer remain upstream dependencies. Training metadata and checkpoint lineage are recorded in `provenance/`.

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
