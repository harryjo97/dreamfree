#!/usr/bin/env bash
set -euo pipefail

# Run on each of two eight-GPU nodes with the same shared config/data paths.
: "${MACHINE_RANK:?Set MACHINE_RANK to 0 or 1}"
: "${MAIN_PROCESS_IP:?Set MAIN_PROCESS_IP to the rank-0 node address}"
if [[ "$MACHINE_RANK" != 0 && "$MACHINE_RANK" != 1 ]]; then
    echo "MACHINE_RANK must be 0 or 1." >&2
    exit 1
fi
if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/train_stage1_full.sh CONFIG [--set key=value ...]" >&2
    exit 1
fi
recipe_config=$1
shift
exec uv run accelerate launch \
    --num_processes 16 --num_machines 2 --machine_rank "$MACHINE_RANK" \
    --same_network --rdzv_backend static \
    --main_process_ip "$MAIN_PROCESS_IP" --main_process_port "${MAIN_PROCESS_PORT:-29500}" \
    --mixed_precision bf16 \
    --use_fsdp --fsdp_version 2 \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap DiTBlock \
    --fsdp_reshard_after_forward false \
    --fsdp_offload_params false --fsdp_cpu_ram_efficient_loading false \
    --fsdp_activation_checkpointing false --fsdp_state_dict_type SHARDED_STATE_DICT \
    --use_parallelism_config \
    --parallelism_config_dp_replicate_size 2 --parallelism_config_dp_shard_size 8 \
    -m dreamfree.training.stage1 --config "$recipe_config" "$@"
