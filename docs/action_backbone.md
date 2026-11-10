# ActionDiT initialization

The video backbone and action backbone are different inputs. Downloading a Wan video checkpoint or a DreamFree release does not provide the preprocessed `.pt` initialization file expected by `action_dit_pretrained_path`.

| Training goal | Configuration | What is initialized |
| --- | --- | --- |
| Fine-tune a released policy | `init_checkpoint=/path/to/release` | All four Stage 2 policy components from the release; frozen bundled encoder; fresh optimizer |
| Train a new policy with the reference initialization | `action_dit_pretrained_path=/absolute/path/to/backbone.pt` | ActionDiT backbone from the preprocessed payload; new action input/output heads and other Stage 2 components |
| Train a new policy from random action weights | `skip_dit_load_from_pretrain=true` | Random ActionDiT and new Stage 2 components; still uses the supplied frozen Stage 1 encoder |

Random initialization is an explicit experimental alternative. A released `stage2/action.safetensors` contains a trained policy's action weights, not the reference pretraining payload. Use `init_checkpoint` to fine-tune that policy.

## Reference artifact and provenance

The selected recipes record `ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt` in [awam_recipe_sources.json](../configs/awam_recipe_sources.json). This is an external, preprocessed AWAM artifact. Its original local path is provenance, not a portable download location. This repository does not bundle or automatically download that initialization artifact.

The AWAM preparation implementation is `scripts/archive/preprocess_action_dit_backbone_vam.py` in the source AWAM checkout. It constructs the action backbone from Wan2.2-TI2V-5B weights, resizes mismatched tensors with sequential linear interpolation, and optionally applies input-width scaling. The payload records the preprocessing policy. Exact reproduction requires the original artifact or the matching source preprocessing configuration; matching tensor shapes alone does not establish equivalent initialization. Obtain that artifact before launching a reference recipe. Release fine-tuning and explicitly random initialization are the self-contained alternatives described above.

## Payload contract

[ActionDiT._load_pretrained_backbone_into](../src/dreamfree/models/wan22/action_dit.py) expects a `torch.save` dictionary with `meta` and `backbone_state_dict`; an optional `policy` dictionary records provenance. `meta` must match the selected Stage 2 architecture:

| Metadata key | Selected recipes |
| --- | ---: |
| `hidden_dim` | 1024 |
| `ffn_dim` | 4096 |
| `num_layers` | 30 |
| `num_heads` | 24 |
| `attn_head_dim` | 128 |
| `text_dim` | 4096 |
| `freq_dim` | 256 |
| `eps` | 0.000001 |

`backbone_state_dict` must contain exactly the keys returned by `ActionDiT.backbone_key_set`: the model state excluding `action_encoder.*` and `head.*`. Every value must be a tensor with the target shape. The action input/output heads remain newly initialized for the dataset's action dimension. Missing/unexpected keys, incompatible dimensions, and missing files are errors.

Use an **absolute path** for this input: the retained action loader resolves relative paths against its installed package location, while ordinary dataset and output paths resolve against the working directory.
