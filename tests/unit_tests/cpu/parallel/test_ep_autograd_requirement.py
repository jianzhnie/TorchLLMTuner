"""Missing differentiable EP transport must never silently drop gradients."""

import pytest
import torch

from llmtuner.errors import EnvironmentUnsupportedError
from llmtuner.parallel.expert_parallel import token_dispatch


def test_missing_autograd_collective_rejects_training(monkeypatch):
    monkeypatch.setattr(token_dispatch, "_differentiable_all_to_all", None)
    monkeypatch.setattr(
        token_dispatch,
        "all_to_all_single",
        lambda *args, **kwargs: pytest.fail("non-autograd training fallback"),
    )
    with pytest.raises(EnvironmentUnsupportedError, match="preserve expert/router"):
        token_dispatch._grad_carrying_all_to_all(
            torch.ones(2, requires_grad=True), None, None, group=object()
        )


def test_missing_autograd_collective_allows_inference(monkeypatch):
    monkeypatch.setattr(token_dispatch, "_differentiable_all_to_all", None)
    calls = []

    def transport(tensor, outputs, inputs, *, group):
        calls.append((outputs, inputs, group))
        return tensor * 2

    monkeypatch.setattr(token_dispatch, "all_to_all_single", transport)
    group = object()
    with torch.no_grad():
        result = token_dispatch._grad_carrying_all_to_all(
            torch.ones(2, requires_grad=True), [2], [2], group=group
        )
    torch.testing.assert_close(result, torch.full((2,), 2.0))
    assert calls == [([2], [2], group)]
