from types import SimpleNamespace

import pytest
import torch
from torch.distributed.tensor import Partial, Replicate

from llmtuner.accelerator import collectives


@pytest.mark.parametrize("order", [1.0, 2.0, float("inf")])
@pytest.mark.parametrize("empty", [False, True])
def test_offloaded_local_norm_promotes_low_precision(order, empty):
    local = (
        torch.empty(0, dtype=torch.float16)
        if empty
        else torch.full((4,), 40000.0, dtype=torch.float16)
    )
    gradient = SimpleNamespace(placements=(Replicate(),), to_local=lambda: local)
    actual = collectives._offloaded_gradient_norm(gradient, order)
    expected = (
        torch.tensor(0.0) if empty else torch.linalg.vector_norm(local.float(), order)
    )
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_offloaded_partial_gradient_is_rejected_before_reading_storage():
    gradient = SimpleNamespace(placements=(Partial(),))
    with pytest.raises(ValueError, match="Replicate/Shard"):
        collectives._offloaded_gradient_norm(gradient, 2.0)


@pytest.mark.parametrize("order", [0.0, -1.0, float("nan")])
def test_offloaded_invalid_norm_order_is_rejected(order):
    with pytest.raises(ValueError, match="positive norm_type"):
        collectives._offloaded_gradient_norm(SimpleNamespace(), order)


def test_mean_rejects_integer_data_before_collective(monkeypatch):
    monkeypatch.setattr(collectives, "get_world_size", lambda group: 2)
    with pytest.raises(TypeError, match="floating or complex"):
        collectives.all_reduce(torch.tensor([1]), op="mean")


def test_invalid_operation_is_rejected_for_single_rank(monkeypatch):
    monkeypatch.setattr(collectives, "get_world_size", lambda group: 1)
    with pytest.raises(ValueError, match="reduce op"):
        collectives.all_reduce(torch.tensor([1]), op="invalid")


def test_strided_reduction_uses_contiguous_buffer_and_preserves_other_columns(
    monkeypatch,
):
    monkeypatch.setattr(collectives, "get_world_size", lambda group: 2)
    monkeypatch.setattr(collectives, "get_default_group", lambda: None)
    monkeypatch.setattr(collectives, "get_comm_device", lambda group: "cpu")

    def reduce(data, **kwargs):
        assert data.is_contiguous()
        data.add_(torch.tensor([2.0, 3.0]))

    monkeypatch.setattr(collectives.dist, "all_reduce", reduce)
    storage = torch.tensor([[1.0, 17.0], [2.0, 19.0]])
    collectives.all_reduce(storage[:, 0])
    torch.testing.assert_close(storage, torch.tensor([[3.0, 17.0], [5.0, 19.0]]))


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


@pytest.mark.parametrize("fails", [False, True])
def test_local_error_boundary_preserves_single_rank_exception(monkeypatch, fails):
    monkeypatch.setattr(collectives, "get_world_size", lambda group: 1)

    def unexpected(*args, **kwargs):
        pytest.fail("single-rank phases must not perform communication")

    monkeypatch.setattr(collectives, "all_reduce", unexpected)
    failure = ValueError("local construction failed")
    if fails:
        with pytest.raises(ValueError) as caught:
            with collectives.synchronize_local_errors("Construction"):
                raise failure
        assert caught.value is failure
    else:
        with collectives.synchronize_local_errors("Construction"):
            pass


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_local_error_boundary_coordinates_before_reraising(monkeypatch, error_type):
    monkeypatch.setattr(collectives, "get_world_size", lambda group: 2)
    monkeypatch.setattr(collectives, "get_comm_device", lambda group: "cpu")
    statuses = []

    def reduce(status, *, op, group):
        assert op == "min"
        statuses.append(status.item())

    monkeypatch.setattr(collectives, "all_reduce", reduce)
    failure = error_type("local construction interrupted")
    with pytest.raises(error_type) as caught:
        with collectives.synchronize_local_errors("Construction"):
            raise failure
    assert caught.value is failure
    assert statuses == [0]


@pytest.mark.parametrize("foreach", [False, True])
def test_fp16_finite_gradients_use_fp32_norm_before_clipping(foreach):
    parameter = torch.nn.Parameter(torch.zeros(4, dtype=torch.float16))
    parameter.grad = torch.full_like(parameter, 40000)
    reference = torch.nn.Parameter(torch.zeros(4, dtype=torch.float32))
    reference.grad = parameter.grad.float()
    expected_norm = torch.nn.utils.clip_grad_norm_(reference, 1.0, foreach=foreach)
    actual_norm = collectives.clip_grad_norm_(
        parameter, 1.0, error_if_nonfinite=True, foreach=foreach
    )
    assert actual_norm.dtype == torch.float32
    torch.testing.assert_close(actual_norm, expected_norm, rtol=0, atol=0)
    torch.testing.assert_close(parameter.grad, reference.grad.half(), rtol=0, atol=0)


@pytest.mark.parametrize("norm_type", [1.0, 2.0, float("inf")])
def test_mixed_precision_norm_matches_full_precision_reference(norm_type):
    parameters = [
        torch.nn.Parameter(torch.zeros(3, dtype=torch.float16)),
        torch.nn.Parameter(torch.zeros(2, dtype=torch.float64)),
    ]
    parameters[0].grad = torch.tensor([40000.0, -40000.0, 20000.0], dtype=torch.float16)
    parameters[1].grad = torch.tensor([3.0, -4.0], dtype=torch.float64)
    reference = torch.cat([p.grad.double() for p in parameters])
    expected = torch.linalg.vector_norm(reference, norm_type)
    actual = collectives.clip_grad_norm_(parameters, 0, norm_type=norm_type)
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=0)
