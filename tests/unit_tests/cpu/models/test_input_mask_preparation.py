"""Only build masks that input preprocessing passes to the selected backend."""

from types import SimpleNamespace

import pytest
import torch

from llmtuner.models.hf.attention_backend import ATTENTION_BACKEND
from llmtuner.models.hf.input_preparation import _shard_for_cp


@pytest.mark.parametrize("backend", ["sdpa", "eager", ATTENTION_BACKEND])
def test_unsharded_preparation_only_builds_flex_masks(backend):
    positions = torch.tensor([0, 1, 0, 1])
    inputs = torch.tensor([1, 2, 3, 4])
    labels = inputs + 1
    calls = []
    sentinel = object()

    def build_mask(*, positions):
        calls.append(positions)
        return sentinel

    model = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(_attn_implementation=backend)),
        get_attention_masks=build_mask,
    )
    result = _shard_for_cp(
        model,
        None,
        packed=True,
        inputs=inputs,
        labels=labels,
        positions=positions,
        padding_mask=None,
    )
    assert result[0] is inputs
    assert result[1] is labels
    assert result[2] is positions
    if backend == ATTENTION_BACKEND:
        assert len(calls) == 1 and calls[0] is positions
        assert result[4] is sentinel
    else:
        assert calls == []
        assert result[4] is None
