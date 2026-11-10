"""Exercise the restore comparator against real native DCP files."""

import copy
import pickle

import numpy as np
import pytest
import torch
import torch.distributed.checkpoint as dcp

from tests.integration_tests.compare_dcp_training_states import compare_checkpoints


def training_state():
    return {
        "model": {"weight": torch.arange(6.0).reshape(2, 3)},
        "optimizer": {"moment": torch.ones(2, 3), "step": 10},
        "lr_scheduler": {"step": 10},
        "train_state": {"step": 10},
        "dataloader": {"cursor": 20},
        "rng_state": {"rank_0": torch.get_rng_state()},
    }


@pytest.mark.parametrize(
    "changed", [None, "model", "optimizer", "dataloader", "rng_state"]
)
def test_real_checkpoint_comparison_detects_state_changes(tmp_path, changed):
    left = training_state()
    right = copy.deepcopy(left)
    if changed == "model":
        right[changed]["weight"][0, 0] += 1
    elif changed == "optimizer":
        right[changed]["moment"][0, 0] += 1
    elif changed == "dataloader":
        right[changed]["cursor"] += 1
    elif changed == "rng_state":
        right[changed]["rank_0"][0] ^= 1
    expected, actual = tmp_path / "expected", tmp_path / "actual"
    dcp.save(left, checkpoint_id=expected, no_dist=True)
    dcp.save(right, checkpoint_id=actual, no_dist=True)
    if changed:
        with pytest.raises(AssertionError):
            compare_checkpoints(expected, actual)
    else:
        result = compare_checkpoints(expected, actual)
        assert result["tensors"] == 3
        assert result["serialized_values"] > 0


@pytest.mark.parametrize("missing", ["rng_state", "optimizer"])
def test_incomplete_checkpoints_cannot_pass(tmp_path, missing):
    state = training_state()
    del state[missing]
    path = tmp_path / "incomplete"
    dcp.save(state, checkpoint_id=path, no_dist=True)
    with pytest.raises(AssertionError, match="Missing training states"):
        compare_checkpoints(path, path)


def test_empty_model_state_cannot_pass(tmp_path):
    state = training_state()
    state["model"] = {"version": 1}
    path = tmp_path / "empty_model"
    dcp.save(state, checkpoint_id=path, no_dist=True)
    with pytest.raises(AssertionError, match="Missing model or optimizer tensors"):
        compare_checkpoints(path, path)


@pytest.mark.parametrize("changed", [None, "cpu", "numpy"])
def test_pickled_rng_compares_generator_values(tmp_path, changed):
    left = training_state()
    payload = {
        "version": 1,
        "cpu": torch.get_rng_state(),
        "numpy": np.random.RandomState(42).get_state(),
    }
    other = copy.deepcopy(payload)
    if changed == "cpu":
        other["cpu"][0] ^= 1
    elif changed == "numpy":
        other["numpy"][1][0] ^= np.uint32(1)
    right = copy.deepcopy(left)
    left["rng_state"]["rank_0"] = pickle.dumps(payload, protocol=4)
    right["rng_state"]["rank_0"] = pickle.dumps(other, protocol=4)
    expected, actual = tmp_path / "expected", tmp_path / "actual"
    dcp.save(left, checkpoint_id=expected, no_dist=True)
    dcp.save(right, checkpoint_id=actual, no_dist=True)
    if changed:
        with pytest.raises(AssertionError):
            compare_checkpoints(expected, actual)
    else:
        compare_checkpoints(expected, actual)


@pytest.mark.parametrize(
    "value",
    [float("inf"), float("nan"), np.float32("inf"), np.array([float("nan")])],
)
def test_equal_nonfinite_serialized_states_cannot_pass(tmp_path, value):
    state = training_state()
    state["train_state"]["observation"] = value
    path = tmp_path / "nonfinite"
    dcp.save(state, checkpoint_id=path, no_dist=True)
    with pytest.raises(AssertionError, match="nonfinite"):
        compare_checkpoints(path, path)


@pytest.mark.parametrize("side", ["expected", "actual", "both"])
@pytest.mark.parametrize(
    "damage",
    [
        "hole",
        "overlap",
        "bounds",
        "index",
        "extra_index",
        "file",
        "truncated",
        "byte_index",
        "alias",
        "escape",
        "directory",
        "float_offset",
        "bool_length",
    ],
)
def test_corrupt_real_dcp_rejected_before_load(tmp_path, monkeypatch, side, damage):
    from torch.distributed.checkpoint.metadata import MetadataIndex

    paths = [tmp_path / "left", tmp_path / "right"]
    for path in paths:
        dcp.save(training_state(), checkpoint_id=path, no_dist=True)
    damaged_paths = paths if side == "both" else [paths[0 if side == "expected" else 1]]
    for path in damaged_paths:
        metadata_file = path / ".metadata"
        meta = pickle.loads(metadata_file.read_bytes())
        spec = meta.state_dict_metadata["model.weight"]
        if damage == "hole":
            spec.chunks.clear()
        elif damage == "overlap":
            spec.chunks.append(copy.deepcopy(spec.chunks[0]))
        elif damage == "bounds":
            spec.chunks[0].offsets = torch.Size([1, 0])
        elif damage == "index":
            del meta.storage_data[MetadataIndex("model.weight", spec.chunks[0].offsets)]
        elif damage == "extra_index":
            meta.storage_data[MetadataIndex("unexpected.bytes")] = next(
                iter(meta.storage_data.values())
            )
        elif damage == "byte_index":
            del meta.storage_data[MetadataIndex("train_state.step")]
        elif damage == "alias":
            indices = list(meta.storage_data)
            meta.storage_data[indices[1]] = meta.storage_data[indices[0]]
        elif damage in {"escape", "directory", "float_offset", "bool_length"}:
            info = next(iter(meta.storage_data.values()))
            if damage == "escape":
                info.relative_path = "../outside.distcp"
            elif damage == "directory":
                info.relative_path = "."
            elif damage == "float_offset":
                info.offset = 0.5
            else:
                info.length = True
        else:
            info = next(iter(meta.storage_data.values()))
            file = path / info.relative_path
            if damage == "file":
                file.unlink()
            else:
                with file.open("r+b") as stream:
                    stream.truncate(1)
        metadata_file.write_bytes(pickle.dumps(meta))

    def forbidden_load(*args, **kwargs):
        pytest.fail("payload load must not run before structural validation")

    monkeypatch.setattr(dcp, "load", forbidden_load)
    with pytest.raises((ValueError, FileNotFoundError)):
        compare_checkpoints(*paths)


def test_giant_logical_tensor_metadata_rejected_before_allocation(
    tmp_path, monkeypatch
):
    path = tmp_path / "giant"
    dcp.save(training_state(), checkpoint_id=path, no_dist=True)
    file = path / ".metadata"
    meta = pickle.loads(file.read_bytes())
    spec = meta.state_dict_metadata["model.weight"]
    spec.size = torch.Size([2**30, 3])
    spec.chunks[0].sizes = spec.size
    file.write_bytes(pickle.dumps(meta))

    def forbidden_load(*args, **kwargs):
        pytest.fail("oversized metadata must fail before loading")

    monkeypatch.setattr(dcp, "load", forbidden_load)
    with pytest.raises(ValueError, match="Memory preflight rejected.*largest logical"):
        compare_checkpoints(path, path)


def test_byte_record_budget_rejected_before_load(tmp_path, monkeypatch):
    from tests.integration_tests.compare_dcp_training_states import MEMORY_RESERVE_BYTES

    state = training_state()
    state["dataloader"]["blob"] = b"x" * (1024**2)
    path = tmp_path / "large_control"
    dcp.save(state, checkpoint_id=path, no_dist=True)

    def forbidden_load(*args, **kwargs):
        pytest.fail("all preflight checks must precede any load")

    monkeypatch.setattr(dcp, "load", forbidden_load)
    with pytest.raises(ValueError, match="Memory preflight rejected dataloader.blob"):
        compare_checkpoints(
            path, path, memory_budget_bytes=MEMORY_RESERVE_BYTES + 2 * 1024**2
        )


def test_serialized_controls_loaded_individually_with_real_reader(
    tmp_path, monkeypatch
):
    from torch.distributed.checkpoint.metadata import BytesStorageMetadata

    path = tmp_path / "controls"
    dcp.save(training_state(), checkpoint_id=path, no_dist=True)
    meta = dcp.FileSystemReader(path).read_metadata()
    byte_keys = {
        k
        for k, v in meta.state_dict_metadata.items()
        if isinstance(v, BytesStorageMetadata)
    }
    actual_load = dcp.load
    seen = []

    def recording_load(state, *args, **kwargs):
        if set(state) & byte_keys:
            assert len(state) == 1
            seen.append(next(iter(state)))
        return actual_load(state, *args, **kwargs)

    monkeypatch.setattr(dcp, "load", recording_load)
    compare_checkpoints(path, path)
    assert len(seen) == 2 * len(byte_keys)


@pytest.mark.parametrize(
    "root,key", [("lr_scheduler", "step"), ("train_state", "step")]
)
def test_scalar_controls_really_compared(tmp_path, root, key):
    left = training_state()
    right = copy.deepcopy(left)
    right[root][key] += 1
    paths = [tmp_path / "left", tmp_path / "right"]
    for state, path in zip((left, right), paths, strict=True):
        dcp.save(state, checkpoint_id=path, no_dist=True)
    with pytest.raises(AssertionError, match="values differ"):
        compare_checkpoints(*paths)


def test_exact_memory_budget_boundary_with_real_dcp(tmp_path, monkeypatch):
    path = tmp_path / "budget_boundary"
    dcp.save(training_state(), checkpoint_id=path, no_dist=True)
    result = compare_checkpoints(path, path)
    estimate = result["estimated_peak_bytes"]
    assert (
        compare_checkpoints(path, path, memory_budget_bytes=estimate)[
            "estimated_peak_bytes"
        ]
        == estimate
    )

    def forbidden_load(*args, **kwargs):
        pytest.fail("one byte below admission must fail before loading")

    monkeypatch.setattr(dcp, "load", forbidden_load)
    with pytest.raises(ValueError, match="Memory preflight rejected"):
        compare_checkpoints(path, path, memory_budget_bytes=estimate - 1)


@pytest.mark.parametrize("budget", [0, -1, True, 4.0])
def test_invalid_memory_budget_rejected(tmp_path, budget):
    with pytest.raises(ValueError, match="positive integer"):
        compare_checkpoints(tmp_path, tmp_path, memory_budget_bytes=budget)


def test_one_ulp_model_difference_still_fails(tmp_path):
    left = training_state()
    right = copy.deepcopy(left)
    right["model"]["weight"][0, 1] = torch.nextafter(
        right["model"]["weight"][0, 1], torch.tensor(float("inf"))
    )
    paths = [tmp_path / "left", tmp_path / "right"]
    for state, path in zip((left, right), paths, strict=True):
        dcp.save(state, checkpoint_id=path, no_dist=True)
    with pytest.raises(AssertionError):
        compare_checkpoints(*paths)
