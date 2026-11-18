"""Metadata adaptation preserves rank payloads without changing backend classes."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from llmtuner.components.checkpointer.storage import checkpointing_metadata as metadata


def test_gather_metadata_preserves_variable_length_rank_payloads(monkeypatch):
    import pickle

    payloads = [{"rank": 0}, {"rank": 1, "long": list(range(100))}]
    encoded = [pickle.dumps(value) for value in payloads]
    maximum = max(len(value) + 8 for value in encoded)
    group = object()
    phases = []

    @contextmanager
    def coordinate(phase, *, group):
        phases.append((phase, group))
        yield

    monkeypatch.setattr(metadata, "synchronize_local_errors", coordinate)
    monkeypatch.setattr(
        metadata, "get_comm_device", lambda selected: torch.device("cpu")
    )
    monkeypatch.setattr(metadata.dist, "get_world_size", lambda selected: 2)

    def reduce_max(tensor, op, group):
        tensor.fill_(maximum)

    def gather(output, local, group):
        assert local.numel() == maximum
        for destination, value in zip(output, encoded, strict=True):
            raw = len(value).to_bytes(8, "little") + value
            raw += b"\0" * (maximum - len(raw))
            destination.copy_(torch.frombuffer(bytearray(raw), dtype=torch.uint8))

    monkeypatch.setattr(metadata.dist, "all_reduce", reduce_max)
    monkeypatch.setattr(metadata.dist, "all_gather", gather)
    assert metadata._gather_pickled_metadata(None, payloads[0], group) == dict(
        enumerate(payloads)
    )
    assert phases == [
        (f"Checkpoint metadata {phase}", group)
        for phase in ("encoding", "buffers", "decoding")
    ]


@pytest.mark.parametrize("initialized,device", [(False, "npu"), (True, "cuda")])
def test_metadata_adapter_leaves_other_environments_untouched(
    monkeypatch, initialized, device
):
    monkeypatch.setattr(metadata.dist, "is_initialized", lambda: initialized)
    monkeypatch.setattr(
        metadata, "get_comm_device", lambda: SimpleNamespace(type=device)
    )
    assert not metadata.adapt_metadata_gather(SimpleNamespace())


@pytest.mark.parametrize("device", ["cpu", "npu"])
def test_metadata_adapter_is_scoped_to_one_manager_instance(monkeypatch, device):
    class DefaultMetadataManager:
        def _all_gather_pickled_data(self, data, process_group=None):
            return "original"

    target = DefaultMetadataManager()
    other = DefaultMetadataManager()
    monkeypatch.setattr(metadata.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        metadata, "get_comm_device", lambda: SimpleNamespace(type=device)
    )
    monkeypatch.setattr(metadata, "DefaultMetadataManager", DefaultMetadataManager)
    assert metadata.adapt_metadata_gather(SimpleNamespace(_metadata_manager=target))
    assert target._all_gather_pickled_data.__func__ is metadata._gather_pickled_metadata
    assert other._all_gather_pickled_data(None) == "original"


@pytest.mark.parametrize("phase", ["encoding", "buffers", "decoding"])
def test_metadata_failure_stays_inside_its_local_error_boundary(monkeypatch, phase):
    import pickle

    original = RuntimeError("metadata local failure")
    active = []
    coordinated = []
    communication = []

    @contextmanager
    def coordinate(label, *, group):
        active.append(label)
        try:
            yield
        except BaseException as error:
            coordinated.append((label, error))
            raise
        finally:
            active.pop()

    def fail(*args, **kwargs):
        assert active == [f"Checkpoint metadata {phase}"]
        raise original

    def reduce_max(tensor, **kwargs):
        assert not active
        communication.append("size")

    def gather(output, local, **kwargs):
        assert not active
        communication.append("payload")
        output[0].copy_(local)

    monkeypatch.setattr(metadata, "synchronize_local_errors", coordinate)
    monkeypatch.setattr(metadata, "get_comm_device", lambda group: torch.device("cpu"))
    monkeypatch.setattr(metadata.dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(metadata.dist, "all_reduce", reduce_max)
    monkeypatch.setattr(metadata.dist, "all_gather", gather)
    if phase == "encoding":
        monkeypatch.setattr(pickle, "dumps", fail)
    elif phase == "buffers":
        monkeypatch.setattr(metadata.torch, "empty_like", fail)
    else:
        monkeypatch.setattr(pickle, "loads", fail)

    with pytest.raises(RuntimeError) as caught:
        metadata._gather_pickled_metadata(None, {"step": 7})
    assert caught.value is original
    assert coordinated == [(f"Checkpoint metadata {phase}", original)]
    assert (
        communication
        == {
            "encoding": [],
            "buffers": ["size"],
            "decoding": ["size", "payload"],
        }[phase]
    )
