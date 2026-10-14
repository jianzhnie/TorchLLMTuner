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
