"""Compare trusted same-layout DCP snapshots one stored chunk at a time.

This avoids materializing a complete model and optimizer on the host. Different
chunk layouts require a resharding comparator and are explicitly rejected.
Only use with trusted local checkpoints: DCP metadata and states use pickle.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


def assert_state_equal(left, right, *, name):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), name
        assert left.dtype == right.dtype and left.shape == right.shape, name
        assert torch.isfinite(left).all() and torch.isfinite(right).all(), name
        assert torch.equal(left, right), name
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray), name
        assert left.dtype == right.dtype and left.shape == right.shape, name
        assert np.isfinite(left).all() and np.isfinite(right).all(), name
        assert np.array_equal(left, right), name
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), name
        for key in left:
            assert_state_equal(left[key], right[key], name=f"{name}.{key}")
    elif isinstance(left, tuple | list):
        assert type(left) is type(right) and len(left) == len(right), name
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            assert_state_equal(a, b, name=f"{name}[{index}]")
    else:
        if isinstance(left, float):
            assert math.isfinite(left) and math.isfinite(right), name
        assert type(left) is type(right) and left == right, name


def _read(root, storage):
    with (root / storage.relative_path).open("rb") as stream:
        stream.seek(storage.offset)
        data = stream.read(storage.length)
    assert len(data) == storage.length, "truncated checkpoint storage"
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


def _validate_tensor_coverage(state: TensorStorageMetadata, name: str) -> None:
    message = f"global tensor coverage differs: {name}"
    assert all(size >= 0 for size in state.size), message
    volume = 0
    nonempty = []
    for chunk in state.chunks:
        assert len(chunk.offsets) == len(chunk.sizes) == len(state.size), message
        assert all(
            offset >= 0 and size >= 0 and offset + size <= bound
            for offset, size, bound in zip(
                chunk.offsets, chunk.sizes, state.size, strict=True
            )
        ), message
        chunk_volume = math.prod(chunk.sizes)
        volume += chunk_volume
        if not chunk_volume:
            continue
        for other in nonempty:
            assert not all(
                start < other_start + other_size and other_start < start + size
                for start, size, other_start, other_size in zip(
                    chunk.offsets, chunk.sizes, other.offsets, other.sizes, strict=True
                )
            ), message
        nonempty.append(chunk)
    # Disjoint boxes within the global bounds cover the tensor exactly when
    # their total volume equals the global volume, including scalar tensors.
    assert volume == math.prod(state.size), message


def compare(left: Path, right: Path):
    with (left / ".metadata").open("rb") as stream:
        a = pickle.load(stream)
    with (right / ".metadata").open("rb") as stream:
        b = pickle.load(stream)
    assert a.state_dict_metadata.keys() == b.state_dict_metadata.keys()
    # Preserve shape/dtype/chunk contracts independently of storage filenames.
    assert a.state_dict_metadata == b.state_dict_metadata, "checkpoint layout differs"
    assert a.planner_data == b.planner_data, "flattened state structure differs"
    assert a.storage_data.keys() == b.storage_data.keys(), "stored chunks differ"
    for metadata in (a, b):
        expected = set()
        for name, state in metadata.state_dict_metadata.items():
            if isinstance(state, TensorStorageMetadata):
                _validate_tensor_coverage(state, name)
                expected.update((name, tuple(chunk.offsets)) for chunk in state.chunks)
            else:
                expected.add((name, None))
        actual = {
            (index.fqn, None if index.offset is None else tuple(index.offset))
            for index in metadata.storage_data
        }
        assert actual == expected, (
            f"checkpoint storage coverage differs: missing={expected - actual}, "
            f"unexpected={actual - expected}"
        )
    tensor_contracts = {
        (name, tuple(chunk.offsets)): (state.properties.dtype, chunk.sizes)
        for name, state in a.state_dict_metadata.items()
        if isinstance(state, TensorStorageMetadata)
        for chunk in state.chunks
    }
    counts = Counter()
    for index, storage in a.storage_data.items():
        value = _read(left, storage)
        other = _read(right, b.storage_data[index])
        if isinstance(a.state_dict_metadata[index.fqn], TensorStorageMetadata):
            dtype, shape = tensor_contracts[(index.fqn, tuple(index.offset))]
            for payload in (value, other):
                assert (
                    isinstance(payload, torch.Tensor)
                    and payload.dtype == dtype
                    and payload.shape == shape
                ), f"checkpoint payload contract differs: {index.fqn}"
        # RNG streams are serialized bytes containing tensor storages. Compare
        # their decoded state; raw pickle bytes can differ for equal tensors.
        if index.fqn.startswith("rng_state."):
            value, other = pickle.loads(value), pickle.loads(other)
        assert_state_equal(value, other, name=index.fqn)
        counts[index.fqn.split(".", 1)[0]] += 1
    assert counts, "empty checkpoint cannot prove recovery"
    required = {
        "model",
        "optimizer",
        "train_state",
        "rng_state",
        "dataloader",
        "lr_scheduler",
    }
    assert required <= counts.keys(), (
        f"missing recovery state: {required - counts.keys()}"
    )
    return {
        "status": "passed",
        "standard": "finite exact state equality",
        "stored_chunks": sum(counts.values()),
        "components": dict(counts),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.left, args.right)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
