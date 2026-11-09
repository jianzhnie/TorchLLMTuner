#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
: "${MODEL_PATH:?Set MODEL_PATH to a HF Qwen3.5 configuration/tokenizer directory}"
: "${DATASET_PATH:?Set DATASET_PATH to the LIMO JSONL file}"
: "${OUTPUT_PATH:?Set OUTPUT_PATH to a dedicated checkpoint directory}"
exec "${PYTHON:-python}" -m torch.distributed.run \
  --nnodes="${NNODES:-1}" --nproc-per-node="${NPROC_PER_NODE:-8}" \
  --node-rank="${NODE_RANK:-0}" --master-addr="${MASTER_ADDR:-127.0.0.1}" \
  --master-port="${MASTER_PORT:-29507}" \
  -m examples.qwen3_5.train --model "$MODEL_PATH" --data "$DATASET_PATH" \
  --output "$OUTPUT_PATH" "$@"
