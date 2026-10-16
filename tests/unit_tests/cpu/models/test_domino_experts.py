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
    original_eligibility = domino._can_coordinate_experts

    def eligibility(*args):
        events.append("eligibility")
        return original_eligibility(*args)

    monkeypatch.setattr(domino, "_can_coordinate_experts", eligibility)
    inputs = torch.randn(3, 8)
    scores = torch.ones(3, 2) / 2
    ids = torch.tensor([[0, 1], [2, 3], [0, 3]])
    counts = torch.bincount(ids.flatten(), minlength=4)
    actual = domino(inputs, scores, ids, counts)
    expected = RoutedExperts(grouped, local)(inputs, scores, ids, counts)
    torch.testing.assert_close(actual, expected)
    assert events == [
        "eligibility",
        "counts",
        "launch",
        "launch",
        "consume",
        "combine",
        "consume",
        "combine",
    ]


@pytest.mark.parametrize(
    ("failed_slice", "error_type"),
    [(None, ValueError)]
    + [
        (index, error)
        for index in (0, 1)
        for error in (ValueError, KeyboardInterrupt, SystemExit)
    ],
)
def test_domino_cuda_schedule_waits_before_cross_stream_consumption(
    monkeypatch, failed_slice, error_type
):
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
        return SimpleNamespace(args=args, routed_input=args[0])

    def finish(pending):
        args = pending.args
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

    combined_outputs = []

    def combine(*args):
        events.append((active[0].name, "combine"))
        result = local.combine(*args)
        combined_outputs.append(result)
        return result

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
    if failed_slice is not None:
        import llmtuner.parallel.expert_parallel.domino_dispatch as implementation

        drained = []
        original_run = domino._run_experts
        calls = 0
        expected_error = error_type("injected CUDA-schedule expert failure")

        def run(*args, **kwargs):
            nonlocal calls
            index = calls
            calls += 1
            if index == failed_slice:
                raise expected_error
            return original_run(*args, **kwargs)

        monkeypatch.setattr(domino, "_run_experts", run)
        monkeypatch.setattr(
            implementation, "materialize", lambda tensor: drained.append(tensor)
        )
        with pytest.raises(error_type) as caught:
            domino._forward_cuda(slices, prepared, dispatcher, coordinate_experts=True)
        assert caught.value is expected_error
        assert len(drained) == 2 + failed_slice
        assert drained[0] is slices[0][0]
        assert drained[1] is slices[1][0]
        if failed_slice == 1:
            assert drained[2] is combined_outputs[0]
        assert len(combined_outputs) == failed_slice
        assert ("compute", "wait_stream", "comm") in events
        return
    actual = domino._forward_cuda(slices, prepared, dispatcher, coordinate_experts=True)
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


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize(
    "wait_failure", [None, RuntimeError, KeyboardInterrupt, SystemExit]
)
def test_domino_expert_failure_waits_for_all_launched_dispatches(
    monkeypatch, wait_failure, error_type
):
    from types import SimpleNamespace

    from llmtuner.parallel.expert_parallel.token_dispatch import AllToAllTokenDispatcher

    dispatcher = AllToAllTokenDispatcher(4, 2)
    dispatcher.ep_group = object()
    experts = GroupedExperts(8, 12, 2, use_grouped_mm=False)
    domino = DominoRoutedExperts(experts, dispatcher)
    waited = []

    class Transfer:
        def __init__(self, index):
            self.index = index

        def wait(self):
            waited.append(self.index)
            if wait_failure and self.index == 0:
                raise wait_failure("injected transfer wait failure")
            return torch.ones(1, 8)

    def begin(*args, prepared_counts):
        return SimpleNamespace(routed_input=Transfer(prepared_counts))

    monkeypatch.setattr(dispatcher, "prepare_dispatch_counts", lambda counts: [0, 1])
    monkeypatch.setattr(dispatcher, "begin_dispatch", begin)
    monkeypatch.setattr(
        dispatcher,
        "finish_dispatch",
        lambda pending: (torch.ones(1, 8), torch.tensor([1, 0]), None),
    )
    original_error = error_type("injected expert failure")

    def fail(*args):
        raise original_error

    monkeypatch.setattr(experts, "forward", fail)
    ids = torch.tensor([[0, 1], [2, 3]])
    with pytest.raises(error_type, match="injected expert failure") as caught:
        domino(torch.ones(2, 8), torch.ones(2, 2) / 2, ids, torch.ones(4))
    assert caught.value is original_error
    assert waited == [0, 1]
    if wait_failure:
        assert isinstance(caught.value.__cause__, wait_failure)
        assert "injected transfer wait failure" in str(caught.value.__cause__)


@pytest.mark.parametrize(
    "boundary", ["plain", "pre_hook", "post_hook", "activation", "compiled", "remote"]
)
def test_domino_coordinates_only_known_local_expert_bodies(monkeypatch, boundary):
    from contextlib import contextmanager

    import llmtuner.parallel.expert_parallel.domino_dispatch as implementation
    from llmtuner.parallel.expert_parallel.token_dispatch import AllToAllTokenDispatcher

    experts = GroupedExperts(8, 12, 2, use_grouped_mm=False)
    for parameter in experts.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    if boundary == "pre_hook":
        experts.register_forward_pre_hook(lambda *args: None)
    elif boundary == "post_hook":
        experts.activation_fn.register_forward_hook(lambda *args: None)
    elif boundary == "activation":

        class CustomActivation(torch.nn.Module):
            def forward(self, gate, up):
                return torch.nn.functional.silu(gate) * up

        experts.activation_fn = CustomActivation()
    elif boundary == "compiled":
        experts._compiled_call_impl = lambda *args: GroupedExperts.forward(
            experts, *args
        )
    phases = []
    candidates = []

    def eligibility(status, *, op, group):
        assert op == "min"
        candidates.append(status.item())
        if boundary == "remote":
            status.zero_()

    @contextmanager
    def coordinate(phase, *, group):
        phases.append(phase)
        yield

    monkeypatch.setattr(implementation, "all_reduce", eligibility)
    monkeypatch.setattr(implementation, "synchronize_local_errors", coordinate)
    dispatcher = AllToAllTokenDispatcher(4, 2)
    domino = DominoRoutedExperts(experts, dispatcher)
    inputs = torch.randn(2, 8)
    counts = torch.ones(2, dtype=torch.int64)
    eligible = domino._can_coordinate_experts(inputs, dispatcher)
    actual = domino._run_experts(
        inputs, counts, dispatcher, coordinate_experts=eligible
    )
    torch.testing.assert_close(actual, experts(inputs, counts))
    assert candidates == [int(boundary in ("plain", "remote"))]
    assert phases == (["Domino expert computation"] if boundary == "plain" else [])


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_expert_compute_error_reaches_posthook_before_propagation(
    monkeypatch, error_type
):
    from torch.utils._pytree import tree_map

    from llmtuner.models.common.moe.experts import ExpertErrorCapture

    experts = GroupedExperts(8, 12, 2, use_grouped_mm=False)
    inputs = torch.randn(2, 8, requires_grad=True)
    capture = ExpertErrorCapture()
    assert tree_map(lambda value: value, {"capture": capture})["capture"] is capture
    original_error = error_type("injected local compute error")
    calls = []

    def fail(*args):
        raise original_error

    def posthook(module, args, output):
        calls.append(capture.error)
        assert not output.requires_grad

    monkeypatch.setattr(torch.nn.functional, "linear", fail)
    handle = experts.register_forward_hook(posthook)
    try:
        try:
            placeholder = experts(
                inputs, torch.ones(2, dtype=torch.long), error_capture=capture
            )
        except BaseException as error:
            raise AssertionError("Local error bypassed expert capture") from error
        assert capture.error is original_error
        assert calls == [original_error]
        assert placeholder.data_ptr() == inputs.data_ptr()
        with pytest.raises(error_type) as caught:
            experts(inputs, torch.ones(2, dtype=torch.long))
        assert caught.value is original_error
        assert calls == [original_error]
    finally:
        handle.remove()


def test_expert_error_capture_preserves_successful_outputs_and_gradients():
    from llmtuner.models.common.moe.experts import ExpertErrorCapture

    experts = GroupedExperts(8, 12, 2, use_grouped_mm=False)
    for parameter in experts.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    inputs = torch.randn(3, 8, requires_grad=True)
    counts = torch.tensor([1, 2])
    expected = experts(inputs, counts)
    expected_grads = torch.autograd.grad(
        expected.sum(), (inputs, *experts.parameters())
    )
    retry_input = inputs.detach().clone().requires_grad_()
    capture = ExpertErrorCapture(ValueError("previous call"))
    actual = experts(retry_input, counts, error_capture=capture)
    assert capture.error is None
    actual_grads = torch.autograd.grad(
        actual.sum(), (retry_input, *experts.parameters())
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
