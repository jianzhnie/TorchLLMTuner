"""Capture observations survive append, update and fresh-process-style reopen."""

import pytest
import torch

from tests.integration_tests.alignment_history import (
    AlignmentHistory,
    capture_artifact_path,
    load_alignment_history,
    save_shared_snapshot,
)


def test_disk_history_preserves_order_updates_and_reopen(tmp_path):
    history = AlignmentHistory(tmp_path / "forward-batches", writable=True)
    history.append({"input": torch.arange(3)})
    history.append({"input": torch.arange(4)})
    record = history[-1]
    record["logits"] = torch.tensor([1.0, 2.0])
    history[-1] = record
    reopened = load_alignment_history(tmp_path, "forward-batches")
    assert len(reopened) == 2
    torch.testing.assert_close(reopened[0]["input"], torch.arange(3))
    torch.testing.assert_close(reopened[-1]["logits"], torch.tensor([1.0, 2.0]))
    assert [item["input"].numel() for item in reopened] == [3, 4]
    assert len(reopened[:1]) == 1
    with pytest.raises(IndexError):
        _ = reopened[2]
    with pytest.raises(TypeError, match="read-only"):
        reopened.append({})
    with pytest.raises(TypeError, match="read-only"):
        reopened[0] = {}
    with pytest.raises(FileExistsError):
        AlignmentHistory(tmp_path / "forward-batches", writable=True)


def test_legacy_capture_remains_readable(tmp_path):
    torch.save([{"logits": torch.ones(2)}], tmp_path / "forward-batches.pt")
    history = load_alignment_history(tmp_path, "forward-batches")
    torch.testing.assert_close(history[0]["logits"], torch.ones(2))


def test_capture_artifact_prefers_shared_file_and_falls_back_to_rank_file(tmp_path):
    shared_root = tmp_path / "capture"
    rank_root = shared_root / "rank0"
    shared_root.mkdir()
    rank_root.mkdir()
    rank_file = rank_root / "initial-hf.pt"
    rank_file.touch()
    assert capture_artifact_path(shared_root, rank_root, rank_file.name) == rank_file

    shared_file = shared_root / rank_file.name
    shared_file.touch()
    assert capture_artifact_path(shared_root, rank_root, rank_file.name) == shared_file


def test_shared_snapshot_publishes_validated_state_atomically(tmp_path):
    path = tmp_path / "initial-hf.pt"
    state = {"weight": torch.ones(3)}

    def validate(value):
        assert value is state
        assert not path.exists()

    save_shared_snapshot(state, path, validate=validate)
    torch.testing.assert_close(
        torch.load(path, weights_only=True)["weight"], state["weight"]
    )
    assert not path.with_name(f".{path.name}.tmp").exists()


def test_shared_snapshot_validation_failure_does_not_publish(tmp_path):
    path = tmp_path / "initial-hf.pt"

    def validate(value):
        raise ValueError("incorrect HF mapping")

    with pytest.raises(RuntimeError, match="incorrect HF mapping"):
        save_shared_snapshot({}, path, validate=validate)
    assert not path.exists()


def test_shared_snapshot_nonwriter_receives_writer_failure(monkeypatch, tmp_path):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 1)

    def broadcast(status, src):
        assert src == 0
        status[0] = "OSError: disk full"

    monkeypatch.setattr(dist, "broadcast_object_list", broadcast)
    with pytest.raises(RuntimeError, match="disk full"):
        save_shared_snapshot({}, tmp_path / "initial-hf.pt")
    assert not list(tmp_path.iterdir())
