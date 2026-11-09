"""Autograd capability selection for TP functional collectives."""

import subprocess
import sys
from textwrap import dedent
from types import SimpleNamespace

import pytest
import torch

import llmtuner.parallel.tensor_parallel.linear_ops as linear_ops
from llmtuner.parallel.tensor_parallel.linear_ops import (
    LinearReduceScatter,
    resolve_autograd_collectives,
)


@pytest.fixture(autouse=True)
def isolate_collective_resolution(monkeypatch):
    monkeypatch.setattr(linear_ops, "_autograd_collectives", None)


@pytest.mark.parametrize("suffix", ["single_autograd", "tensor_autograd"])
def test_functional_collectives_select_complete_autograd_pair(monkeypatch, suffix):
    gather, scatter = lambda *args: None, lambda *args: None
    backend = SimpleNamespace(
        **{f"all_gather_{suffix}": gather, f"reduce_scatter_{suffix}": scatter}
    )
    monkeypatch.setattr(linear_ops, "funcol", backend)
    assert resolve_autograd_collectives() == (gather, scatter)


def test_collective_resolution_is_stable_after_backend_probe(monkeypatch):
    gather, scatter = lambda *args: None, lambda *args: None
    backend = SimpleNamespace(
        all_gather_single_autograd=gather,
        reduce_scatter_single_autograd=scatter,
    )
    monkeypatch.setattr(linear_ops, "funcol", backend)
    assert resolve_autograd_collectives() == (gather, scatter)
    # Tensor tracing must not repeat capability imports or inspect the backend.
    del backend.all_gather_single_autograd
    del backend.reduce_scatter_single_autograd
    assert resolve_autograd_collectives() == (gather, scatter)


@pytest.mark.parametrize("incomplete", [False, True])
def test_functional_collectives_reject_non_autograd_fallback(monkeypatch, incomplete):
    backend = SimpleNamespace(
        all_gather_single=lambda *args: None,
        reduce_scatter_single=lambda *args: None,
    )
    if incomplete:
        backend.all_gather_single_autograd = lambda *args: None
    monkeypatch.setattr(linear_ops, "funcol", backend)
    with pytest.raises(ImportError, match="autograd-enabled"):
        resolve_autograd_collectives()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_reduce_scatter_wgrad_preserves_full_precision_dtype(monkeypatch, dtype):
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


def test_missing_symmetric_memory_does_not_disable_autograd_collectives():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            dedent(
                """
                import sys
                import torch

                # Complete normal accelerator/dependency imports before injecting
                # the missing optional leaf capability. Otherwise a partially
                # imported torch_npu may re-register its rendezvous handler.
                import llmtuner.parallel.tensor_parallel.linear_ops
                del sys.modules['llmtuner.parallel.tensor_parallel.linear_ops']
                sys.modules['torch.distributed._symmetric_memory'] = None
                from llmtuner.parallel.tensor_parallel.linear_ops import (
                    resolve_autograd_collectives,
                    ensure_symm_mem_ops,
                )

                assert all(callable(op) for op in resolve_autograd_collectives())
                try:
                    ensure_symm_mem_ops()
                except ImportError:
                    pass
                else:
                    raise AssertionError('missing fused backend was accepted')
                """
            ),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
