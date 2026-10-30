#!/usr/bin/env bash
set -euo pipefail

# Install the pinned offline runtime into every node used by a multi-node NPU run.
# The wheel directory must be visible inside the target container.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
nodes_file="${NODES_FILE:-$repo_root/available_nodes.txt}"
container_name="${CONTAINER_NAME:-hyper-parallel-env}"
wheel_root="${WHEEL_ROOT:-/home/jianzhnie/llmtuner/wheels}"
transformers_root="${TRANSFORMERS_ROOT:-$wheel_root/tf513:$wheel_root/tf511}"
ssh_timeout="${SSH_CONNECT_TIMEOUT:-8}"
install_jobs="${INSTALL_JOBS:-4}"
log_folder="${INSTALL_LOG_FOLDER:-$repo_root/outputs/environment_install_$(date +%Y%m%d_%H%M%S)}"

if [[ ! "$install_jobs" =~ ^[1-9][0-9]*$ ]]; then
  echo "INSTALL_JOBS must be a positive integer" >&2
  exit 1
fi

if [[ ! -f "$nodes_file" ]]; then
  echo "nodes file not found: $nodes_file" >&2
  exit 1
fi

mapfile -t nodes < <(awk 'NF && $1 !~ /^#/ {print $1}' "$nodes_file")
if (( ${#nodes[@]} == 0 )); then
  echo "no nodes found in $nodes_file" >&2
  exit 1
fi

wheel() {
  printf '%s/%s' "$wheel_root" "$1"
}

declare -a wheels=(
  "local/torch-2.12.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl"
  "local/torch_npu-2.12.0-cp312-cp312-manylinux_2_28_aarch64.whl"
  "local/colorama-0.4.6-py2.py3-none-any.whl"
  "local/grain-0.2.18-cp312-cp312-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl"
  "local/etils-1.14.0-py3-none-any.whl"
  "local/array_record-0.8.4-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.whl"
  "local/absl_py-2.5.1-py3-none-any.whl"
  "local/cloudpickle-3.1.2-py3-none-any.whl"
  "local/portpicker-1.6.0-py3-none-any.whl"
  "local/protobuf-5.29.5-cp38-abi3-manylinux2014_aarch64.whl"
  "local/datasets-3.6.0-py3-none-any.whl"
  "local/pyarrow-26.0.0-cp312-cp312-manylinux_2_28_aarch64.whl"
  "local/pandas-3.0.6-cp312-cp312-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl"
  "local/python_dateutil-2.9.0.post0-py2.py3-none-any.whl"
  "local/xxhash-4.0.1-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl"
  "local/dill-0.3.8-py3-none-any.whl"
  "local/multiprocess-0.70.16-py312-none-any.whl"
  "local/fsspec-2025.3.0-py3-none-any.whl"
  "local/tqdm-4.70.1-py3-none-any.whl"
  "local/requests-2.34.2-py3-none-any.whl"
  "local/httpx-0.28.1-py3-none-any.whl"
  "local/httpcore-1.0.9-py3-none-any.whl"
  "local/anyio-4.15.1-py3-none-any.whl"
  "local/h11-0.16.0-py3-none-any.whl"
)

for relative_path in "${wheels[@]}"; do
  if [[ ! -f "$(wheel "$relative_path")" ]]; then
    echo "missing wheel: $(wheel "$relative_path")" >&2
    exit 1
  fi
done

install_args=()
for relative_path in "${wheels[@]}"; do
  install_args+=("$wheel_root/$relative_path")
done

remote_script=$(cat <<'REMOTE'
set -euo pipefail
container_name="$1"
transformers_root="$2"
shift 2
python_args=("$@")
docker exec "$container_name" python -m pip install \
  --disable-pip-version-check --no-input --no-deps --no-index \
  "${python_args[@]}"
docker exec -e PYTHONPATH="$transformers_root" "$container_name" python -c '
import datasets
import grain
import torch
import torch_npu
import transformers

assert torch.__version__.startswith("2.12."), torch.__version__
assert torch_npu.__version__.startswith("2.12."), torch_npu.__version__
assert transformers.__version__.startswith("5.13."), transformers.__version__
print({
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "transformers": transformers.__version__,
    "datasets": datasets.__version__,
    "npu_available": torch.npu.is_available(),
})
'
REMOTE
)

echo "Installing pinned NPU environment on ${#nodes[@]} node(s)..."
mkdir -p "$log_folder"
pids=()
batch_nodes=()
failed=0
printf 'node\texit_code\tlog\n' >"$log_folder/summary.tsv"

wait_batch() {
  local index result node
  for index in "${!pids[@]}"; do
    node="${batch_nodes[$index]}"
    if wait "${pids[$index]}"; then
      result=0
    else
      result=$?
      failed=1
    fi
    printf '%s\t%s\t%s\n' "$node" "$result" "$log_folder/$node.log" \
      >>"$log_folder/summary.tsv"
    echo "$node: exit $result ($log_folder/$node.log)"
  done
  pids=()
  batch_nodes=()
}

for node in "${nodes[@]}"; do
  echo "==> $node"
  ssh -o BatchMode=yes -o ConnectTimeout="$ssh_timeout" "$node" \
    "bash -s -- $(printf '%q ' "$container_name" "$transformers_root" "${install_args[@]}")" \
    <<<"$remote_script" >"$log_folder/$node.log" 2>&1 &
  pids+=("$!")
  batch_nodes+=("$node")
  if (( ${#pids[@]} >= install_jobs )); then
    wait_batch
  fi
done
wait_batch

echo "Environment installation summary: $log_folder/summary.tsv"
if (( failed )); then
  echo "Some nodes failed; inspect their logs before using them for training." >&2
  exit 1
fi
echo "All nodes passed the pinned environment check."
