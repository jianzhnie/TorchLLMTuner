"""CPU contracts for score transport and explicit baseline path selection."""

from types import SimpleNamespace

import torch
from torch import nn

from llmtuner.parallel.expert_parallel.npu_baseline_experts import (
    NpuBaselineRoutedExperts,
)
from llmtuner.parallel.expert_parallel.token_dispatch import AllToAllTokenDispatcher


class ScoreKernel(nn.Module):
    def forward(self, inputs, counts, *, routing_scores):
        return inputs * routing_scores


def test_score_transport_reorders_and_weights_exactly_once(monkeypatch):
    dispatcher = AllToAllTokenDispatcher(4, 2)
    ids = torch.tensor([[3, 0], [2, 1], [0, 3]])
    order = torch.argsort(ids.flatten(), stable=True)
    permutation = torch.tensor([2, 0, 5, 1, 4, 3])
    counts = torch.tensor([2, 1, 1, 2])

    def dispatch(inputs, scores, expert_ids, local_counts):
        assert torch.equal(expert_ids, ids)
        metadata = SimpleNamespace(
            topk_scores_experts_sorted_N=scores.flatten()[order],
            permuted_indices=permutation,
            input_shape=(6, 3),
            input_splits=[3, 3],
            output_splits=[3, 3],
        )
        return inputs[order // 2][permutation], counts, metadata

    monkeypatch.setattr(dispatcher, "dispatch", dispatch)
    monkeypatch.setattr(dispatcher, "_dispatch_token_exchange", lambda x, *args: x)
    monkeypatch.setattr(dispatcher, "_combine_token_exchange", lambda x, *args: x)
    model = NpuBaselineRoutedExperts(ScoreKernel(), dispatcher)
    inputs = torch.arange(9.0).reshape(3, 3).requires_grad_()
    scores = torch.tensor([[0.1, 0.9], [0.3, 0.7], [0.6, 0.4]], requires_grad=True)
    actual = model(inputs, scores, ids, counts)
    expected = inputs * scores.sum(-1, keepdim=True)
    torch.testing.assert_close(actual, expected)
    observed_grads = torch.autograd.grad(actual.sum(), (inputs, scores))
    reference_grads = torch.autograd.grad(expected.sum(), (inputs, scores))
    for observed, reference in zip(observed_grads, reference_grads, strict=True):
        torch.testing.assert_close(observed, reference)


def test_empty_origin_tokens_preserve_output_shape_and_gradients(monkeypatch):
    dispatcher = AllToAllTokenDispatcher(4, 2)
    indices = torch.empty(0, dtype=torch.long)
    counts = torch.zeros(4, dtype=torch.long)

    def dispatch(inputs, scores, expert_ids, local_counts):
        return (
            inputs,
            counts,
            SimpleNamespace(
                topk_scores_experts_sorted_N=scores.flatten(),
                permuted_indices=indices,
                input_shape=(0, 3),
                input_splits=[0, 0],
                output_splits=[0, 0],
            ),
        )

    monkeypatch.setattr(dispatcher, "dispatch", dispatch)
    monkeypatch.setattr(dispatcher, "_dispatch_token_exchange", lambda x, *args: x)
    monkeypatch.setattr(dispatcher, "_combine_token_exchange", lambda x, *args: x)
    model = NpuBaselineRoutedExperts(ScoreKernel(), dispatcher)
    inputs = torch.empty(0, 3, requires_grad=True)
    scores = torch.empty(0, 2, requires_grad=True)
    output = model(inputs, scores, torch.empty(0, 2, dtype=torch.long), counts)
    assert output.shape == (0, 3)
    output.sum().backward()
    assert inputs.grad.shape == inputs.shape
    assert scores.grad.shape == scores.shape
