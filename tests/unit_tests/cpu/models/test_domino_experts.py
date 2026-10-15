import pytest
import torch

from llmtuner.models.common.moe.experts import (
    GroupedExperts,
    RoutedExperts,
)
from llmtuner.parallel.expert_parallel.domino_dispatch import DominoRoutedExperts
from llmtuner.parallel.expert_parallel.token_dispatch import LocalTokenDispatcher


@pytest.mark.parametrize("truncated", ["scores", "ids"])
def test_domino_rejects_truncated_routing_before_first_dispatch(monkeypatch, truncated):
    grouped = GroupedExperts(8, 12, 4, use_grouped_mm=False)
    dispatcher = LocalTokenDispatcher(4, 2)
    domino = DominoRoutedExperts(grouped, dispatcher)
    calls = []
    original = dispatcher.dispatch

    def dispatch(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(dispatcher, "dispatch", dispatch)
    scores = torch.ones(4, 2)
    ids = torch.zeros(4, 2, dtype=torch.int64)
    if truncated == "scores":
        scores = scores[:2]
    else:
        ids = ids[:2]
    with pytest.raises(ValueError, match="Routing scores and expert IDs"):
        domino(torch.ones(4, 8), scores, ids, torch.tensor([8, 0, 0, 0]))
    assert calls == []


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


def test_domino_launches_both_slices_before_consuming_tokens(monkeypatch):
    from llmtuner.parallel.expert_parallel.token_dispatch import AllToAllTokenDispatcher

    grouped = GroupedExperts(8, 12, 4, use_grouped_mm=False)
    for parameter in grouped.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    dispatcher = AllToAllTokenDispatcher(4, 2)
    dispatcher.ep_group = object()
    local = LocalTokenDispatcher(4, 2)
    events = []

    def begin(*args, prepared_counts):
        assert prepared_counts is not None
        events.append("launch")
        return args

    def finish(args):
        events.append("consume")
        return local.dispatch(*args)

    def combine(*args):
        events.append("combine")
        return local.combine(*args)

    def prepare(counts):
        events.append("counts")
        assert counts.shape == (2, 4)
        return [object(), object()]

    monkeypatch.setattr(dispatcher, "prepare_dispatch_counts", prepare)
    monkeypatch.setattr(dispatcher, "begin_dispatch", begin)
    monkeypatch.setattr(dispatcher, "finish_dispatch", finish)
    monkeypatch.setattr(dispatcher, "combine", combine)
    domino = DominoRoutedExperts(grouped, dispatcher)
    inputs = torch.randn(3, 8)
    scores = torch.ones(3, 2) / 2
    ids = torch.tensor([[0, 1], [2, 3], [0, 3]])
    counts = torch.bincount(ids.flatten(), minlength=4)
    actual = domino(inputs, scores, ids, counts)
    expected = RoutedExperts(grouped, local)(inputs, scores, ids, counts)
    torch.testing.assert_close(actual, expected)
    assert events == [
        "counts",
        "launch",
        "launch",
        "consume",
        "combine",
        "consume",
        "combine",
    ]


def test_domino_cuda_schedule_waits_before_cross_stream_consumption(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from llmtuner.parallel.expert_parallel.token_dispatch import (
        AllToAllTokenDispatcher,
        DispatchCounts,
    )

    events = []

    class Stream:
        def __init__(self, name):
            self.name = name
            self.events = 0

        def wait_stream(self, other):
            events.append((self.name, "wait_stream", other.name))

        def record_event(self):
            events.append((self.name, "ready"))
            self.events += 1
            return f"event{self.events}"

        def wait_event(self, event):
            events.append((self.name, "wait_event", event))

    compute = Stream("compute")
    comm = Stream("comm")
    active = [compute]

    @contextmanager
    def stream_context(stream):
        previous = active[0]
        active[0] = stream
        try:
            yield
        finally:
            active[0] = previous

    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: compute)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **_: comm)
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    monkeypatch.setattr(
        torch.Tensor,
        "record_stream",
        lambda self, stream: events.append(("storage", stream.name)),
    )
    grouped = GroupedExperts(8, 12, 4, use_grouped_mm=False)
    for parameter in grouped.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    dispatcher = AllToAllTokenDispatcher(4, 2)
    local = LocalTokenDispatcher(4, 2)

    def begin(*args, prepared_counts):
        events.append((active[0].name, "launch"))
        return args

    def finish(args):
        events.append((active[0].name, "consume"))
        inputs, counts, metadata = local.dispatch(*args)
        return (
            inputs,
            counts,
            SimpleNamespace(
                token_indices_experts_sorted_N=metadata.token_indices_experts_sorted_N,
                topk_scores_experts_sorted_N=metadata.topk_scores_experts_sorted_N,
                permuted_indices=torch.arange(inputs.shape[0]),
            ),
        )

    def combine(*args):
        events.append((active[0].name, "combine"))
        return local.combine(*args)

    monkeypatch.setattr(dispatcher, "begin_dispatch", begin)
    monkeypatch.setattr(dispatcher, "finish_dispatch", finish)
    monkeypatch.setattr(dispatcher, "combine", combine)
    domino = DominoRoutedExperts(grouped, dispatcher)
    inputs = torch.randn(3, 8)
    scores = torch.ones(3, 2) / 2
    ids = torch.tensor([[0, 1], [2, 3], [0, 3]])
    slices = [
        (inputs[s], scores[s], ids[s], torch.bincount(ids[s].flatten(), minlength=4))
        for s in (slice(0, 2), slice(2, None))
    ]
    prepared = [
        DispatchCounts(counts=args[3], input_splits=[4], output_splits=[4])
        for args in slices
    ]
    actual = domino._forward_cuda(slices, prepared, dispatcher)
    expected = RoutedExperts(grouped, local)(
        inputs, scores, ids, torch.bincount(ids.flatten(), minlength=4)
    )
    torch.testing.assert_close(actual, expected)
    assert [event for event in events if event[0] != "storage"] == [
        ("compute", "launch"),
        ("compute", "consume"),
        ("comm", "wait_stream", "compute"),
        ("comm", "launch"),
        ("comm", "consume"),
        ("comm", "ready"),
        ("comm", "wait_stream", "compute"),
        ("comm", "combine"),
        ("compute", "wait_event", "event1"),
        ("comm", "wait_stream", "compute"),
        ("comm", "combine"),
        ("comm", "ready"),
        ("compute", "wait_event", "event2"),
    ]
    assert events.count(("storage", "comm")) == 11
    assert events.count(("storage", "compute")) == 4
