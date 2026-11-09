#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
: "${MODEL_PATH:=/home/jianzhnie/llmtuner/hfhub/models/deepseek-ai/DeepSeek-V4-Flash}"
: "${DATASET_PATH:=/home/jianzhnie/llmtuner/hfhub/datasets/GAIR/LIMO/limo.jsonl}"
: "${OUTPUT_PATH:?Set OUTPUT_PATH to a dedicated checkpoint directory}"
exec "${PYTHON:-python}" -m torch.distributed.run \
  --nnodes="${NNODES:-1}" --nproc-per-node="${NPROC_PER_NODE:-8}" \
  --node-rank="${NODE_RANK:-0}" --master-addr="${MASTER_ADDR:-127.0.0.1}" \
  --master-port="${MASTER_PORT:-29532}" \
  -m examples.deepseek_v4.train --model "$MODEL_PATH" --data "$DATASET_PATH" \
  --output "$OUTPUT_PATH" "$@"
