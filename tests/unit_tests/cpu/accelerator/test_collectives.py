import pytest
import torch

from llmtuner.accelerator import collectives


def test_mean_rejects_integer_data_before_collective(monkeypatch):
    monkeypatch.setattr(collectives, "get_world_size", lambda group: 2)
    with pytest.raises(TypeError, match="floating or complex"):
        collectives.all_reduce(torch.tensor([1]), op="mean")


def test_invalid_operation_is_rejected_for_single_rank(monkeypatch):
    monkeypatch.setattr(collectives, "get_world_size", lambda group: 1)
    with pytest.raises(ValueError, match="reduce op"):
        collectives.all_reduce(torch.tensor([1]), op="invalid")
