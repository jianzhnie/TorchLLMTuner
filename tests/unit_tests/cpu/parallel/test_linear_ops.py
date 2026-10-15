"""Autograd capability selection for TP functional collectives."""

from types import SimpleNamespace

import pytest
import torch.distributed as dist

from llmtuner.parallel.tensor_parallel.linear_ops import functional_collectives


@pytest.mark.parametrize("suffix", ["single_autograd", "tensor_autograd"])
def test_functional_collectives_select_complete_autograd_pair(monkeypatch, suffix):
    gather, scatter = lambda *args: None, lambda *args: None
    backend = SimpleNamespace(
        **{f"all_gather_{suffix}": gather, f"reduce_scatter_{suffix}": scatter}
    )
    monkeypatch.setattr(dist, "functional_collectives", backend, raising=False)
    assert functional_collectives() == (gather, scatter)


@pytest.mark.parametrize("incomplete", [False, True])
def test_functional_collectives_reject_non_autograd_fallback(monkeypatch, incomplete):
    backend = SimpleNamespace(
        all_gather_single=lambda *args: None,
        reduce_scatter_single=lambda *args: None,
    )
    if incomplete:
        backend.all_gather_single_autograd = lambda *args: None
    monkeypatch.setattr(dist, "functional_collectives", backend, raising=False)
    with pytest.raises(ImportError, match="autograd-enabled"):
        functional_collectives()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_reduce_scatter_wgrad_preserves_full_precision_dtype(monkeypatch, dtype):
    import torch

    from llmtuner.parallel.tensor_parallel.linear_ops import LinearReduceScatter

    dtype = getattr(torch, dtype)
    x = torch.arange(4, dtype=dtype).reshape(2, 2)
    weight = torch.arange(6, dtype=dtype).reshape(3, 2)
    grad_y = torch.arange(6, dtype=dtype).reshape(2, 3)
    context = SimpleNamespace(
        saved_tensors=(x, weight), group_name="test", has_bias=False
    )
    monkeypatch.setattr(
        torch.ops.symm_mem,
        "fused_all_gather_matmul",
        lambda dy, weights, dim, group: (dy, [dy @ weights[0]]),
        raising=False,
    )
    grad_x, grad_w, *_ = LinearReduceScatter.backward(context, grad_y)
    torch.testing.assert_close(grad_x, grad_y @ weight)
    torch.testing.assert_close(grad_w, grad_y.T @ x)
    assert grad_w.dtype == dtype
