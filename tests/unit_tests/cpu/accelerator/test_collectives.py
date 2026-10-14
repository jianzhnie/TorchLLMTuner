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


def test_partial_to_invariant_reduces_forward_without_reducing_backward(monkeypatch):
    group = object()
    calls = []
    remote_partial = torch.tensor([3.0, 5.0])

    def reduce(data, *, group):
        calls.append(group)
        data.add_(remote_partial)

    monkeypatch.setattr(collectives, "all_reduce", reduce)
    local_partial = torch.tensor([1.0, 2.0], requires_grad=True)
    reduced = collectives.PartialToInvariantAllReduce.apply(local_partial, group)
    torch.testing.assert_close(reduced, torch.tensor([4.0, 7.0]))
    torch.testing.assert_close(local_partial, torch.tensor([1.0, 2.0]))

    reduced.square().sum().backward()

    torch.testing.assert_close(local_partial.grad, torch.tensor([8.0, 14.0]))
    assert calls == [group]
