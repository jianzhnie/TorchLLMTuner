"""Exercise the streaming comparator against real CPU DCP storage."""

import pickle

import numpy as np
import pytest
import torch
import torch.distributed.checkpoint as dcp

from tests.integration_tests.compare_training_checkpoints import compare


def save(path, *, weight=1.0, rng=7):
    dcp.save(
        {
            "model": {"weight": torch.tensor([weight])},
            "optimizer": {"step": torch.tensor(5)},
            "train_state": {"step": 5},
            "lr_scheduler": {"step": 5},
            "dataloader": {"cursor": 5},
            "rng_state": {
                "rank0": pickle.dumps(
                    {"cpu": torch.tensor([rng]), "numpy": np.arange(4)}, protocol=4
                )
            },
        },
        checkpoint_id=path,
    )


def test_equal_checkpoint_compares_decoded_rng_and_all_components(tmp_path):
    left, right = tmp_path / "a", tmp_path / "b"
    save(left)
    save(right)
    result = compare(left, right)
    assert result["status"] == "passed"
    assert set(result["components"]) == {
        "model",
        "optimizer",
        "train_state",
        "rng_state",
        "dataloader",
        "lr_scheduler",
    }


@pytest.mark.parametrize(
    "change", [{"weight": 2.0}, {"weight": float("nan")}, {"rng": 8}]
)
def test_checkpoint_difference_or_nonfinite_fails(tmp_path, change):
    left, right = tmp_path / "a", tmp_path / "b"
    save(left)
    save(right, **change)
    with pytest.raises(AssertionError):
        compare(left, right)


def test_model_only_snapshots_cannot_pass_training_recovery(tmp_path):
    left, right = tmp_path / "a", tmp_path / "b"
    for path in (left, right):
        dcp.save({"model": {"weight": torch.ones(1)}}, checkpoint_id=path)
    with pytest.raises(AssertionError, match="missing recovery state"):
        compare(left, right)


def test_truncated_checkpoint_storage_fails(tmp_path):
    left, right = tmp_path / "a", tmp_path / "b"
    save(left)
    save(right)
    storage = next(right.glob("*.distcp"))
    storage.write_bytes(storage.read_bytes()[:8])
    with pytest.raises(AssertionError, match="truncated checkpoint storage"):
        compare(left, right)


def test_changed_checkpoint_tensor_layout_fails_before_values(tmp_path):
    left, right = tmp_path / "a", tmp_path / "b"
    save(left)
    save(right)
    with (right / ".metadata").open("rb") as stream:
        metadata = pickle.load(stream)
    metadata.state_dict_metadata["model.weight"].size = torch.Size([2])
    with (right / ".metadata").open("wb") as stream:
        pickle.dump(metadata, stream)
    with pytest.raises(AssertionError, match="checkpoint layout differs"):
        compare(left, right)


@pytest.mark.parametrize("declared", ["optimizer.step", "dataloader.cursor"])
def test_matching_missing_storage_entries_cannot_prove_recovery(tmp_path, declared):
    left, right = tmp_path / "a", tmp_path / "b"
    for path in (left, right):
        save(path)
        with (path / ".metadata").open("rb") as stream:
            metadata = pickle.load(stream)
        # A declared tensor or serialized leaf lacks storage in both snapshots.
        # Existing state still keeps every component count nonempty.
        metadata.state_dict_metadata[declared + "_missing"] = (
            metadata.state_dict_metadata[declared]
        )
        with (path / ".metadata").open("wb") as stream:
            pickle.dump(metadata, stream)
    with pytest.raises(AssertionError, match="storage coverage"):
        compare(left, right)


@pytest.mark.parametrize("contract", ["dtype", "shape"])
def test_equal_payloads_must_match_declared_tensor_contract(tmp_path, contract):
    left, right = tmp_path / "a", tmp_path / "b"
    for path in (left, right):
        save(path)
        with (path / ".metadata").open("rb") as stream:
            metadata = pickle.load(stream)
        state = metadata.state_dict_metadata["model.weight"]
        if contract == "dtype":
            state.properties.dtype = torch.float64
        else:
            state.size = torch.Size([2])
            state.chunks[0].sizes = torch.Size([2])
        with (path / ".metadata").open("wb") as stream:
            pickle.dump(metadata, stream)
    with pytest.raises(AssertionError, match="payload contract"):
        compare(left, right)


@pytest.mark.parametrize("layout", ["gap", "overlap", "out_of_bounds"])
def test_equal_snapshots_must_cover_global_tensor_once(tmp_path, layout):
    left, right = tmp_path / "a", tmp_path / "b"
    for path in (left, right):
        save(path)
        with (path / ".metadata").open("rb") as stream:
            metadata = pickle.load(stream)
        state = metadata.state_dict_metadata["model.weight"]
        if layout == "gap":
            state.size = torch.Size([2])
        elif layout == "overlap":
            state.chunks.append(state.chunks[0])
        else:
            state.size = torch.Size([0])
        with (path / ".metadata").open("wb") as stream:
            pickle.dump(metadata, stream)
    with pytest.raises(AssertionError, match="global tensor coverage"):
        compare(left, right)
