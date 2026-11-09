#!/usr/bin/env bash
set -euo pipefail

# Install the pinned offline runtime into every node used by a multi-node NPU run.
# The wheel directory must be visible inside the target container.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
nodes_file="${NODES_FILE:-$repo_root/available_nodes.txt}"
container_name="${CONTAINER_NAME:-hyper-parallel-env}"
wheel_root="${WHEEL_ROOT:-/home/jianzhnie/llmtuner/wheels}"
# tf513 is the only accepted Transformers package; tf511 is retained only for
# shared dependency files that are present in the older extracted snapshot.
transformers_root="${TRANSFORMERS_ROOT:-$wheel_root/tf513:$wheel_root/tf511}"
# Override these when the host path is mounted at a different path on a node or
# inside its container.  By default the shared filesystem uses the same paths.
remote_wheel_root="${REMOTE_WHEEL_ROOT:-$wheel_root}"
remote_transformers_root="${REMOTE_TRANSFORMERS_ROOT:-$transformers_root}"
transformers_version_prefix="${TRANSFORMERS_VERSION_PREFIX:-5.13.}"
torch_version_prefix="${TORCH_VERSION_PREFIX:-2.12.}"
torch_npu_version_prefix="${TORCH_NPU_VERSION_PREFIX:-$torch_version_prefix}"
runtime_python_version="${RUNTIME_PYTHON_VERSION:-3.12}"
# Set an absolute container path to install without changing its default Python.
runtime_venv="${RUNTIME_VENV:-}"
if [[ -n "$runtime_venv" && "$runtime_venv" != /* ]]; then
  echo "RUNTIME_VENV must be an absolute container path" >&2
  exit 1
fi
require_npu="${REQUIRE_NPU:-1}"
install_triton="${INSTALL_TRITON:-0}"
triton_version_prefix="${TRITON_VERSION_PREFIX:-3.2.1}"
preflight_only="${PREFLIGHT_ONLY:-0}"
verify_wheel_hashes="${VERIFY_WHEEL_HASHES:-1}"
ssh_timeout="${SSH_CONNECT_TIMEOUT:-8}"
install_jobs="${INSTALL_JOBS:-4}"
log_folder="${INSTALL_LOG_FOLDER:-$repo_root/outputs/environment_install_$(date +%Y%m%d_%H%M%S)}"

if [[ ! "$install_jobs" =~ ^[1-9][0-9]*$ ]]; then
  echo "INSTALL_JOBS must be a positive integer" >&2
  exit 1
fi

if [[ "$require_npu" != 0 && "$require_npu" != 1 ]]; then
  echo "REQUIRE_NPU must be 0 or 1" >&2
  exit 1
fi

if [[ "$install_triton" != 0 && "$install_triton" != 1 ]]; then
  echo "INSTALL_TRITON must be 0 or 1" >&2
  exit 1
fi

if [[ "$preflight_only" != 0 && "$preflight_only" != 1 ]]; then
  echo "PREFLIGHT_ONLY must be 0 or 1" >&2
  exit 1
fi

if [[ "$verify_wheel_hashes" != 0 && "$verify_wheel_hashes" != 1 ]]; then
  echo "VERIFY_WHEEL_HASHES must be 0 or 1" >&2
  exit 1
fi

if [[ ! "$ssh_timeout" =~ ^[1-9][0-9]*$ ]]; then
  echo "SSH_CONNECT_TIMEOUT must be a positive integer" >&2
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

declare -A seen_nodes=()
for node in "${nodes[@]}"; do
  if [[ -n "${seen_nodes[$node]+present}" ]]; then
    echo "duplicate node in $nodes_file: $node" >&2
    exit 1
  fi
  seen_nodes["$node"]=1
done

primary_transformers_root="${transformers_root%%:*}"
primary_transformers_init="$primary_transformers_root/transformers/__init__.py"
if [[ ! -f "$primary_transformers_init" ]]; then
  echo "Primary Transformers runtime not found: $primary_transformers_root" >&2
  echo "Set TRANSFORMERS_ROOT to a mounted extracted Transformers runtime." >&2
  exit 1
fi
transformers_declared_version="$(
  sed -n 's/^__version__ = "\([^"]*\)".*/\1/p' "$primary_transformers_init" | head -1
)"
if [[ "$transformers_declared_version" != "$transformers_version_prefix"* ]]; then
  echo "Transformers runtime version mismatch: expected $transformers_version_prefix*," >&2
  echo "found ${transformers_declared_version:-unknown} in $primary_transformers_init" >&2
  exit 1
fi

wheel() {
  printf '%s/%s' "$wheel_root" "$1"
}

torch_wheel="${TORCH_WHEEL:-local/torch-2.12.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl}"
torch_npu_wheel="${TORCH_NPU_WHEEL:-local/torch_npu-2.12.0-cp312-cp312-manylinux_2_28_aarch64.whl}"

declare -a wheels=(
  "$torch_wheel"
  "$torch_npu_wheel"
  "local/sympy-1.14.0-py3-none-any.whl"
  "local/mpmath-1.3.0-py3-none-any.whl"
  "local/networkx-3.4.2-py3-none-any.whl"
  "local/typing_extensions-4.16.0-py3-none-any.whl"
  "local/filelock-4.1.1-py3-none-any.whl"
  "local/jinja2-3.1.6-py3-none-any.whl"
  "local/markupsafe-3.0.4-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl"
  "local/fsspec-2025.3.0-py3-none-any.whl"
  "local/numpy-2.2.6-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
  "local/packaging-26.3-py3-none-any.whl"
  "local/pyyaml-6.0.3-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl"
  "local/huggingface_hub-1.5.0-py3-none-any.whl"
  "local/typer-0.27.3-py3-none-any.whl"
  "local/click-8.5.0-py3-none-any.whl"
  "local/rich-15.0.0-py3-none-any.whl"
  "local/annotated_doc-0.0.5-py3-none-any.whl"
  "local/markdown_it_py-4.2.0-py3-none-any.whl"
  "local/mdurl-0.1.2-py3-none-any.whl"
  "local/pygments-2.21.0-py3-none-any.whl"
  "local/shellingham-1.5.4-py2.py3-none-any.whl"
  "transformers/hf_xet-1.6.0-cp38-abi3-manylinux_2_28_aarch64.whl"
  "transformers/regex-2026.9.10-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl"
  "transformers/tokenizers-0.22.2-cp39-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
  "transformers/safetensors-0.8.0-cp310-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
  "local/colorama-0.4.6-py2.py3-none-any.whl"
  "local/grain-0.2.18-cp312-cp312-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl"
  "local/etils-1.14.0-py3-none-any.whl"
  "local/array_record-0.8.4-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.whl"
  "local/absl_py-2.5.1-py3-none-any.whl"
  "local/cloudpickle-3.1.2-py3-none-any.whl"
  "local/portpicker-1.6.0-py3-none-any.whl"
  "local/psutil-7.2.2-cp36-abi3-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl"
  "local/protobuf-5.29.5-cp38-abi3-manylinux2014_aarch64.whl"
  "local/datasets-3.6.0-py3-none-any.whl"
  "local/pyarrow-26.0.0-cp312-cp312-manylinux_2_28_aarch64.whl"
  "local/pandas-3.0.6-cp312-cp312-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl"
  "local/python_dateutil-2.9.0.post0-py2.py3-none-any.whl"
  "local/xxhash-4.0.1-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl"
  "local/dill-0.3.8-py3-none-any.whl"
  "local/multiprocess-0.70.16-py312-none-any.whl"
  "local/tqdm-4.70.1-py3-none-any.whl"
  "local/requests-2.34.2-py3-none-any.whl"
  "local/charset_normalizer-3.5.2-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl"
  "local/idna-3.20-py3-none-any.whl"
  "local/urllib3-2.8.0-py3-none-any.whl"
  "local/certifi-2026.10.23-py3-none-any.whl"
  "local/httpx-0.28.1-py3-none-any.whl"
  "local/httpcore-1.0.9-py3-none-any.whl"
  "local/anyio-4.15.1-py3-none-any.whl"
  "local/h11-0.16.0-py3-none-any.whl"
  "local/spmd_types-0.2.5-py3-none-any.whl"
  "local/pillow-12.3.0-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
  "local/torchvision-0.27.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl"
  "local/einops-0.8.2-py3-none-any.whl"
  "local/six-1.17.0-py2.py3-none-any.whl"
  "local/zipp-4.1.1-py3-none-any.whl"
  # Torch 2.12 requires setuptools<82.
  "local/setuptools-81.0.0-py3-none-any.whl"
)

if (( install_triton )); then
  # --no-deps requires explicitly installing the Ascend compiler dependency.
  wheels+=("local/pybind11-3.0.2-py3-none-any.whl")
  wheels+=("${TRITON_WHEEL:-local/triton_ascend-3.2.1-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl}")
fi

for relative_path in "${wheels[@]}"; do
  if [[ ! -f "$(wheel "$relative_path")" ]]; then
    echo "missing wheel: $(wheel "$relative_path")" >&2
    exit 1
  fi
done

if (( verify_wheel_hashes )) && ! command -v sha256sum >/dev/null 2>&1; then
  echo "sha256sum is required when VERIFY_WHEEL_HASHES=1" >&2
  exit 1
fi

declare -a wheel_specs=()
for relative_path in "${wheels[@]}"; do
  wheel_hash="-"
  if (( verify_wheel_hashes )); then
    wheel_hash="$(sha256sum "$(wheel "$relative_path")" | awk '{print $1}')"
  fi
  wheel_specs+=("$relative_path|$wheel_hash")
done

echo "Local preflight passed: ${#nodes[@]} node(s), ${#wheels[@]} wheel(s),"
echo "Transformers=${transformers_root}, Python=${runtime_python_version},"
echo "Torch=${torch_version_prefix}, Torch-NPU=${torch_npu_version_prefix},"
echo "NPU required=${require_npu}, wheel hashes=${verify_wheel_hashes}."
echo "Triton install=${install_triton}, expected distribution=${triton_version_prefix}*."
if (( preflight_only )); then
  exit 0
fi

remote_script=$(cat <<'REMOTE'
set -euo pipefail
container_name="$1"
remote_wheel_root="$2"
transformers_root="$3"
transformers_version_prefix="$4"
torch_version_prefix="$5"
torch_npu_version_prefix="$6"
runtime_python_version="$7"
require_npu="$8"
verify_wheel_hashes="$9"
shift 9
runtime_venv="$1"
install_triton="$2"
triton_version_prefix="$3"
shift 3
wheel_specs=("$@")

if [[ "$(docker inspect -f '{{.State.Running}}' "$container_name" 2>/dev/null)" != true ]]; then
  echo "container is not running: $container_name" >&2
  exit 1
fi

first_transformers_root="${transformers_root%%:*}"
docker exec "$container_name" test -d "$first_transformers_root/transformers"
python_args=()
for wheel_spec in "${wheel_specs[@]}"; do
  relative_path="${wheel_spec%%|*}"
  expected_hash="${wheel_spec#*|}"
  if [[ "$relative_path" == "$wheel_spec" || -z "$relative_path" ]]; then
    echo "invalid wheel specification: $wheel_spec" >&2
    exit 1
  fi
  wheel_path="$remote_wheel_root/$relative_path"
  docker exec "$container_name" test -f "$wheel_path"
  python_args+=("$wheel_path")
done

if [[ "$verify_wheel_hashes" == 1 ]]; then
  docker exec -i "$container_name" python - "$remote_wheel_root" "${wheel_specs[@]}" <<'PY'
import hashlib
import sys

root, *specs = sys.argv[1:]
for spec in specs:
    relative_path, expected = spec.split("|", 1)
    digest = hashlib.sha256()
    with open(f"{root}/{relative_path}", "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise SystemExit(
            f"wheel hash mismatch for {relative_path}: {actual} != {expected}"
        )
PY
fi

runtime_python=python
if [[ -n "$runtime_venv" ]]; then
  if ! docker exec "$container_name" test -x "$runtime_venv/bin/python"; then
    docker exec "$container_name" python -m venv "$runtime_venv"
  fi
  runtime_python="$runtime_venv/bin/python"
fi

docker exec "$container_name" "$runtime_python" -m pip install \
  --disable-pip-version-check --no-input --no-deps --no-index \
  --no-cache-dir --upgrade --force-reinstall \
  "${python_args[@]}"

if [[ -n "$runtime_venv" ]]; then
  # Keep the extracted offline runtime discoverable in child processes even
  # when a launcher intentionally replaces PYTHONPATH with the repository.
  docker exec -i "$container_name" "$runtime_python" - "$transformers_root" <<'PY'
import site
import sys
from pathlib import Path

roots = sys.argv[1].split(":")
if any(not Path(root).is_absolute() or "\n" in root for root in roots):
    raise SystemExit("Transformers roots must be absolute, single-line paths")
target = Path(site.getsitepackages()[0]) / "llmtuner_offline_transformers.pth"
target.write_text("\n".join(roots) + "\n")
PY
fi

docker exec \
  -e PYTHONPATH="$transformers_root" \
  -e EXPECTED_TRANSFORMERS_PREFIX="$transformers_version_prefix" \
  -e EXPECTED_TORCH_PREFIX="$torch_version_prefix" \
  -e EXPECTED_TORCH_NPU_PREFIX="$torch_npu_version_prefix" \
  -e EXPECTED_PYTHON_VERSION="$runtime_python_version" \
  -e REQUIRE_NPU="$require_npu" \
  -e EXPECTED_TRANSFORMERS_ROOT="$first_transformers_root" \
  -e INSTALL_TRITON="$install_triton" \
  -e EXPECTED_TRITON_PREFIX="$triton_version_prefix" \
  "$container_name" "$runtime_python" -c '
import os
import sys
from importlib import import_module, metadata
from pathlib import Path

import datasets
import grain
import spmd_types
import torch
import torch_npu
import transformers

python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
assert python_version == os.environ["EXPECTED_PYTHON_VERSION"], python_version
assert torch.__version__.startswith(os.environ["EXPECTED_TORCH_PREFIX"]), torch.__version__
assert torch_npu.__version__.startswith(
    os.environ["EXPECTED_TORCH_NPU_PREFIX"]
), torch_npu.__version__
assert transformers.__version__.startswith(
    os.environ["EXPECTED_TRANSFORMERS_PREFIX"]
), transformers.__version__
expected_transformers_root = Path(os.environ["EXPECTED_TRANSFORMERS_ROOT"])
assert Path(transformers.__file__).resolve().is_relative_to(
    expected_transformers_root.resolve()
), transformers.__file__

npu_available = bool(torch.npu.is_available())
npu_count = torch.npu.device_count() if npu_available else 0
if os.environ["REQUIRE_NPU"] == "1":
    assert npu_available and npu_count > 0, (npu_available, npu_count)

triton_info = None
if os.environ["INSTALL_TRITON"] == "1":
    triton = import_module("triton")
    triton_version = metadata.version("triton_ascend")
    assert triton_version.startswith(os.environ["EXPECTED_TRITON_PREFIX"]), triton_version
    triton_info = {
        "distribution_version": triton_version,
        "module_version": triton.__version__,
        "file": triton.__file__,
    }

print({
    "python": python_version,
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "transformers": transformers.__version__,
    "transformers_file": transformers.__file__,
    "datasets": datasets.__version__,
    "grain": grain.__version__,
    "npu_available": npu_available,
    "npu_count": npu_count,
    "triton": triton_info,
})
'
REMOTE
)

echo "Installing pinned NPU environment on ${#nodes[@]} node(s)..."
mkdir -p "$log_folder"
pids=()
batch_nodes=()
batch_logs=()
failed=0
printf 'node\texit_code\tlog\n' >"$log_folder/summary.tsv"

cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

wait_batch() {
  local index result node log_file
  for index in "${!pids[@]}"; do
    node="${batch_nodes[$index]}"
    log_file="${batch_logs[$index]}"
    if wait "${pids[$index]}"; then
      result=0
    else
      result=$?
      failed=1
    fi
    printf '%s\t%s\t%s\n' "$node" "$result" "$log_file" \
      >>"$log_folder/summary.tsv"
    echo "$node: exit $result ($log_file)"
  done
  pids=()
  batch_nodes=()
  batch_logs=()
}

for node in "${nodes[@]}"; do
  echo "==> $node"
  log_name="${node//[^[:alnum:]._-]/_}"
  log_file="$log_folder/$log_name.log"
  ssh -o BatchMode=yes -o ConnectTimeout="$ssh_timeout" "$node" \
    "bash -s -- $(printf '%q ' "$container_name" "$remote_wheel_root" \
      "$remote_transformers_root" \
      "$transformers_version_prefix" "$torch_version_prefix" \
      "$torch_npu_version_prefix" "$runtime_python_version" "$require_npu" \
      "$verify_wheel_hashes" "$runtime_venv" "$install_triton" \
      "$triton_version_prefix" "${wheel_specs[@]}")" \
    <<<"$remote_script" >"$log_file" 2>&1 &
  pids+=("$!")
  batch_nodes+=("$node")
  batch_logs+=("$log_file")
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
