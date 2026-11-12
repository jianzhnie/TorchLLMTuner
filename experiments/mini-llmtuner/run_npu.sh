#!/usr/bin/env bash
# Run inside an already provisioned container on host-verified idle devices.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
ARCHITECTURE="${ARCHITECTURE:-qwen3_moe}"
TP_SIZE="${TP_SIZE:-2}"
if [[ "$ARCHITECTURE" != "qwen3_moe" ]]; then
    EP_SIZE="${EP_SIZE:-1}"
else
    EP_SIZE="${EP_SIZE:-2}"
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
FSDP_SIZE="${FSDP_SIZE:-$((NPROC_PER_NODE / TP_SIZE / EP_SIZE))}"
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" \
    -m experiments.mini-llmtuner.train --device npu --architecture "$ARCHITECTURE" \
    --tp-size "${TP_SIZE:-2}" --ep-size "${EP_SIZE:-2}" \
    --dp-shard-size "${FSDP_SIZE:-2}" --steps "${STEPS:-3}" "$@"
