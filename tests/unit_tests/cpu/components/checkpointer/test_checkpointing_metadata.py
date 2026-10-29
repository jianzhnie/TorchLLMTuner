"""Metadata adaptation preserves rank payloads without changing backend classes."""

import sys
from types import SimpleNamespace

import pytest
import torch

from llmtuner.components.checkpointer import checkpointing_metadata as metadata


def test_gather_metadata_preserves_variable_length_rank_payloads(monkeypatch):
    import pickle

    payloads = [{"rank": 0}, {"rank": 1, "long": list(range(100))}]
    encoded = [pickle.dumps(value) for value in payloads]
    maximum = max(len(value) + 8 for value in encoded)
    group = object()
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


@pytest.mark.parametrize("initialized,device", [(False, "npu"), (True, "cpu")])
def test_metadata_adapter_leaves_other_environments_untouched(
    monkeypatch, initialized, device
):
    monkeypatch.setattr(metadata.dist, "is_initialized", lambda: initialized)
    monkeypatch.setattr(
        metadata, "get_comm_device", lambda: SimpleNamespace(type=device)
    )
    monkeypatch.setattr(
        metadata, "version", lambda name: pytest.fail("Unexpected import")
    )
    assert not metadata.adapt_npu_metadata_gather(SimpleNamespace())


def test_metadata_adapter_is_scoped_to_one_supported_version_instance(monkeypatch):
    class DefaultMetadataManager:
        def _all_gather_pickled_data(self, data, process_group=None):
            return "original"

    target = DefaultMetadataManager()
    other = DefaultMetadataManager()
    monkeypatch.setattr(metadata.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        metadata, "get_comm_device", lambda: SimpleNamespace(type="npu")
    )
    monkeypatch.setattr(metadata, "version", lambda name: "0.1.0")
    monkeypatch.setitem(
        sys.modules,
        "torch_checkpointing.metadata_manager",
        SimpleNamespace(DefaultMetadataManager=DefaultMetadataManager),
    )
    assert metadata.adapt_npu_metadata_gather(SimpleNamespace(_metadata_manager=target))
    assert target._all_gather_pickled_data.__func__ is metadata._gather_pickled_metadata
    assert other._all_gather_pickled_data(None) == "original"
    monkeypatch.setattr(metadata, "version", lambda name: "0.2.0")
    assert not metadata.adapt_npu_metadata_gather(
        SimpleNamespace(_metadata_manager=other)
    )
    assert other._all_gather_pickled_data(None) == "original"
