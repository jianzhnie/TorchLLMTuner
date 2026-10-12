import torch

from llmtuner.models.common.moe.experts import (
    DominoRoutedExperts,
    GroupedExperts,
    RoutedExperts,
)
from llmtuner.parallel.expert_parallel.token_dispatch import LocalTokenDispatcher


def test_domino_two_slice_execution_matches_unsliced_experts() -> None:
    torch.manual_seed(7)
    grouped = GroupedExperts(8, 12, 4, use_grouped_mm=False)
    for parameter in grouped.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    reference = RoutedExperts(grouped, LocalTokenDispatcher(4, 2))
    domino = DominoRoutedExperts(grouped, LocalTokenDispatcher(4, 2))
    x = torch.randn(7, 8)
    expert_ids = torch.tensor([[0, 1], [2, 3], [1, 2], [3, 0], [0, 2], [1, 3], [2, 0]])
    scores = torch.softmax(torch.randn(7, 2), dim=-1)
    counts = torch.bincount(expert_ids.reshape(-1), minlength=4)

    expected = reference(x, scores, expert_ids, counts)
    actual = domino(x, scores, expert_ids, counts)

    torch.testing.assert_close(actual, expected)
