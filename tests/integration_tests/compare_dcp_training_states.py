"""Compare trusted native DCP states after structural and memory admission checks.

The configurable memory estimate is not a hard bound on DCP/pickle decoding.
This verifies saved states; it does not prove that either training run succeeded.
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import MetadataIndex, TensorStorageMetadata

from tests.integration_tests.audit_dcp_checkpoint import validate_checkpoint_storage

REQUIRED_STATES = {
    "model",
    "optimizer",
    "lr_scheduler",
    "train_state",
    "dataloader",
    "rng_state",
}


def assert_state_equal(left, right, path: str) -> None:
    if type(left) is not type(right):
        raise AssertionError(f"{path}: value types differ")
    if isinstance(left, torch.Tensor):
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise AssertionError(f"{path}: nonfinite tensor")
        try:
            torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=False)
        except AssertionError as error:
            raise AssertionError(f"{path}: {error}") from error
    elif isinstance(left, np.ndarray):
        if np.issubdtype(left.dtype, np.inexact) and (
            not np.isfinite(left).all() or not np.isfinite(right).all()
        ):
            raise AssertionError(f"{path}: nonfinite array")
        np.testing.assert_array_equal(left, right, strict=True)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"{path}: keys differ")
        for key in left:
            assert_state_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, tuple | list):
        if len(left) != len(right):
            raise AssertionError(f"{path}: lengths differ")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            assert_state_equal(a, b, f"{path}[{index}]")
    else:
        if isinstance(left, float | complex | np.inexact) and (
            not np.isfinite(left) or not np.isfinite(right)
        ):
            raise AssertionError(f"{path}: nonfinite scalar")
        if left != right:
            raise AssertionError(f"{path}: values differ")


DEFAULT_MEMORY_BUDGET_BYTES = 4 * 1024**3
MEMORY_RESERVE_BYTES = 256 * 1024**2
TENSOR_BATCH_BYTES = 256 * 1024**2


def compare_checkpoints(
    expected: Path,
    actual: Path,
    *,
    memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
) -> dict[str, int]:
    if type(memory_budget_bytes) is not int or memory_budget_bytes <= 0:
        raise ValueError("memory_budget_bytes must be a positive integer")
    readers = [dcp.FileSystemReader(path) for path in (expected, actual)]
    full_metadata = [reader.read_metadata() for reader in readers]
    # Both sides must pass, including unmatched extra indices and absent files,
    # before any payload allocation or load. Common damage cannot pass equality.
    inventories = [
        validate_checkpoint_storage(path, meta)
        for path, meta in zip((expected, actual), full_metadata, strict=True)
    ]
    metadata = [meta.state_dict_metadata for meta in full_metadata]
    if metadata[0].keys() != metadata[1].keys():
        raise AssertionError("Checkpoint state keys differ")
    roots = {key.split(".")[0] for key in metadata[0]}
    if not REQUIRED_STATES <= roots:
        raise AssertionError(f"Missing training states: {REQUIRED_STATES - roots}")
    largest_tensor = max(i["largest_logical_tensor_bytes"] for i in inventories)
    largest_record = max(i["largest_serialized_record_bytes"] for i in inventories)
    largest_chunk = max(i["largest_decoded_chunk_bytes"] for i in inventories)
    estimated_peak = MEMORY_RESERVE_BYTES

    def admit(key: str, estimate: int) -> None:
        nonlocal estimated_peak
        estimated_peak = max(estimated_peak, estimate)
        if estimate > memory_budget_bytes:
            raise ValueError(
                f"Memory preflight rejected {key}: estimated {estimate} bytes > "
                f"budget {memory_budget_bytes} bytes; largest logical tensor="
                f"{largest_tensor}, serialized record={largest_record}, "
                f"decoded chunk={largest_chunk}. DCP can decode an entire "
                "record; windowed reads are not implemented. Increase the "
                "explicit budget only with sufficient Host memory. This "
                "estimate is not a hard memory bound."
            )

    byte_keys = []
    batches = []
    batch = []
    batch_bytes = 0
    tensor_roots = set()
    sizes = {}
    for key, spec in metadata[0].items():
        other = metadata[1][key]
        if type(spec) is not type(other):
            raise AssertionError(f"{key}: storage types differ")
        if not isinstance(spec, TensorStorageMetadata):
            byte_keys.append(key)
            continue
        if spec.size != other.size or spec.properties.dtype != other.properties.dtype:
            raise AssertionError(f"{key}: tensor shape/dtype differs")
        tensor_roots.add(key.split(".")[0])
        size = (
            math.prod(spec.size)
            * torch.empty((), dtype=spec.properties.dtype).element_size()
        )
        sizes[key] = size
        if batch and batch_bytes + size > TENSOR_BATCH_BYTES:
            batches.append(batch)
            batch, batch_bytes = [], 0
        batch.append(key)
        batch_bytes += size
    if batch:
        batches.append(batch)
    if not {"model", "optimizer"} <= tensor_roots:
        raise AssertionError("Missing model or optimizer tensors")
    # Inventory/admit ALL work before the first load. Two retained destination
    # batches, a full decoded saved chunk and serialized record, plus generous
    # comparison workspace (assert_close/isfinite) and metadata/Python reserve.
    for keys in batches:
        admit(
            keys[0],
            MEMORY_RESERVE_BYTES
            + 2 * sum(sizes[k] for k in keys)
            + largest_chunk
            + largest_record
            + 8 * max(sizes[k] for k in keys),
        )
    for key in byte_keys:
        records = sum(m.storage_data[MetadataIndex(key)].length for m in full_metadata)
        # Per-key IO prevents retaining all serialized controls. Pickle expansion
        # is unbounded: this multiplier is an admission estimate, not containment.
        admit(key, MEMORY_RESERVE_BYTES + 4 * records + largest_record)

    def load(reader, state):
        dcp.load(
            state,
            storage_reader=reader,
            no_dist=True,
            planner=DefaultLoadPlanner(
                flatten_state_dict=False, flatten_sharded_tensors=False
            ),
        )

    tensors = 0
    for keys in batches:
        values = []
        for reader in readers:
            state = {
                key: torch.empty(
                    metadata[0][key].size, dtype=metadata[0][key].properties.dtype
                )
                for key in keys
            }
            load(reader, state)
            values.append(state)
        assert_state_equal(*values, "tensors")
        tensors += len(keys)
        del values, state
    for key in byte_keys:
        states = []
        for reader in readers:
            state = {key: None}
            load(reader, state)
            # Compare actual decoded controls, not the original None placeholders.
            value = state[key]
            if key.startswith("rng_state.rank_") and isinstance(value, bytes):
                value = pickle.loads(value)
            states.append(value)
        assert_state_equal(*states, f"serialized_states.{key}")
        del states, state, value
    if tensors == 0:
        raise AssertionError("No model/optimizer tensors compared")
    return {
        "tensors": tensors,
        "serialized_values": len(byte_keys),
        "largest_logical_tensor_bytes": largest_tensor,
        "largest_serialized_record_bytes": largest_record,
        "estimated_peak_bytes": estimated_peak,
        "memory_budget_bytes": memory_budget_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--memory-budget-gib", type=float, default=4.0)
    args = parser.parse_args()
    if not math.isfinite(args.memory_budget_gib) or args.memory_budget_gib <= 0:
        parser.error("--memory-budget-gib must be finite and positive")
    budget_bytes = args.memory_budget_gib * 1024**3
    if not math.isfinite(budget_bytes) or budget_bytes < 1:
        parser.error("--memory-budget-gib must convert to at least one finite byte")
    result = compare_checkpoints(
        args.expected,
        args.actual,
        memory_budget_bytes=int(budget_bytes),
    )
    print(f"Exact checkpoint comparison passed: {result}")


if __name__ == "__main__":
    main()
