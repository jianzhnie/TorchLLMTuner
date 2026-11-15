from contextlib import contextmanager, nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llmtuner.components.loss import cross_entropy_loss
from llmtuner.models.contracts import DecoderStructure
from llmtuner.models.execution import ModelExecutionCapabilities
from llmtuner.trainer.runtime.engine import TrainingEngine
from llmtuner.trainer.runtime.results import AccumulationResult, StepResult


class ToyCausalModel(nn.Module):
    execution_capabilities = ModelExecutionCapabilities(
        hidden_states=True, hidden_size=8
    )
    decoder_structure = DecoderStructure(
        decoder="", embedding="embedding", blocks="layers", norm="norm", head="lm_head"
    )

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(11, 8)
        self.lm_head = nn.Linear(8, 11, bias=False)
        self.vocab_size = 11

    def forward(self, inputs, *, skip_lm_head=False):
        hidden_states = self.embedding(inputs)
        return hidden_states if skip_lm_head else self.lm_head(hidden_states)


def build_engine(model: nn.Module, *, num_chunks: int = 1) -> TrainingEngine:
    return TrainingEngine(
        model_parts=[model],
        parallel_dims=None,
        device=torch.device("cpu"),
        chunked_loss_num_chunks=num_chunks,
        param_context=nullcontext,
    )


@pytest.mark.parametrize("retain_loss", [False, True])
@pytest.mark.parametrize("nonfinite", [None, float("nan"), float("inf")])
def test_accumulation_preserves_group_losses_and_checks_every_group(
    retain_loss, nonfinite
):
    engine = build_engine(ToyCausalModel())
    values = [torch.tensor(2.0), torch.tensor(3.0), torch.tensor(4.0)]
    if nonfinite is not None:
        values[1].fill_(nonfinite)
    originals = [value.clone() for value in values]
    consumed = []

    def groups():
        for index, value in enumerate(values):
            consumed.append(index)
            yield StepResult(value, 1, torch.tensor(3))

    result = engine.accumulate_results(groups(), retain_loss=retain_loss)
    assert consumed == [0, 1, 2]
    assert result.loss_is_finite.dtype == torch.int32
    assert result.loss_is_finite.item() == (nonfinite is None)
    if retain_loss:
        torch.testing.assert_close(
            result.loss_sum, torch.stack(originals).sum(), equal_nan=True
        )
        assert result.loss_sum is not values[0]
    else:
        assert result.loss_sum is None
    for value, original in zip(values, originals, strict=True):
        torch.testing.assert_close(value, original, equal_nan=True)


@pytest.mark.parametrize("retain_loss", [False, True])
def test_empty_accumulation_does_not_produce_successful_result(retain_loss):
    engine = build_engine(ToyCausalModel())
    with pytest.raises(ValueError, match="at least one group"):
        engine.accumulate_results(iter(()), retain_loss=retain_loss)


@pytest.mark.parametrize(
    "verdict",
    [True, torch.tensor(True), torch.tensor(1.0), torch.ones(2, dtype=torch.int32)],
)
def test_accumulation_rejects_verdict_without_collective_scalar_contract(verdict):
    with pytest.raises(TypeError, match="int32 scalar tensor"):
        AccumulationResult(None, verdict)


@pytest.mark.parametrize("invalid", ["gradient", "device", "shape"])
def test_accumulation_rejects_loss_incompatible_with_reporting(invalid):
    loss = (
        torch.ones((), requires_grad=True)
        if invalid == "gradient"
        else torch.ones((), device="meta")
        if invalid == "device"
        else torch.ones(2)
    )
    with pytest.raises(ValueError):
        AccumulationResult(loss, torch.ones((), dtype=torch.int32))


@pytest.mark.parametrize("part_count", [0, 2])
def test_dense_engine_rejects_invalid_part_count_at_construction(part_count):
    with pytest.raises(ValueError, match="model part"):
        TrainingEngine(
            model_parts=[ToyCausalModel() for _ in range(part_count)],
            parallel_dims=None,
            device=torch.device("cpu"),
            chunked_loss_num_chunks=1,
            param_context=nullcontext,
        )


@pytest.mark.parametrize("pipeline_enabled", [False, True])
def test_engine_rejects_schedule_topology_mismatch(pipeline_enabled):
    with pytest.raises(ValueError, match="topology and schedule"):
        TrainingEngine(
            model_parts=[ToyCausalModel()],
            parallel_dims=SimpleNamespace(pp_enabled=pipeline_enabled),
            pp_schedule=None if pipeline_enabled else object(),
            device=torch.device("cpu"),
            chunked_loss_num_chunks=1,
            param_context=nullcontext,
        )


def test_dense_engine_matches_direct_normalized_backward() -> None:
    torch.manual_seed(7)
    model = ToyCausalModel()
    reference = deepcopy(model)
    inputs = torch.randint(0, 11, (10,))
    labels = torch.randint(0, 11, (10,))
    labels[0] = -100
    global_valid_tokens = torch.tensor(7)

    loss_sum = build_engine(model).forward_backward_dense(
        inputs, labels, extra_kwargs={}, global_valid_tokens=global_valid_tokens
    )
    expected_sum = cross_entropy_loss(reference(inputs), labels)
    (expected_sum / global_valid_tokens).backward()

    torch.testing.assert_close(loss_sum, expected_sum.detach(), rtol=0, atol=0)
    assert not loss_sum.requires_grad
    for parameter, expected in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(parameter.grad, expected.grad, rtol=0, atol=0)


@pytest.mark.parametrize("num_chunks", [1, 2, 4])
def test_chunked_engine_preserves_dense_loss_and_gradients(num_chunks: int) -> None:
    torch.manual_seed(11)
    model = ToyCausalModel()
    reference = deepcopy(model)
    inputs = torch.randint(0, 11, (12,))
    labels = torch.randint(0, 11, (12,))
    global_valid_tokens = torch.tensor(9)

    loss_sum = build_engine(model, num_chunks=num_chunks).forward_backward_dense(
        inputs, labels, extra_kwargs={}, global_valid_tokens=global_valid_tokens
    )
    expected_sum = cross_entropy_loss(reference(inputs), labels)
    (expected_sum / global_valid_tokens).backward()

    torch.testing.assert_close(loss_sum, expected_sum.detach())
    for parameter, expected in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(parameter.grad, expected.grad)


@pytest.mark.parametrize("num_chunks", [1, 2, 4])
def test_batched_token_output_matches_flat_loss_and_gradients(num_chunks):
    model = ToyCausalModel()
    reference = deepcopy(model)
    inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])
    labels = torch.tensor([[2, -100, 4], [5, 6, -100]])
    denominator = torch.tensor(4)
    loss = build_engine(model, num_chunks=num_chunks).forward_backward_dense(
        inputs, labels, extra_kwargs={}, global_valid_tokens=denominator
    )
    expected = cross_entropy_loss(reference(inputs).reshape(-1, 11), labels.reshape(-1))
    (expected / denominator).backward()
    torch.testing.assert_close(loss, expected.detach())
    for actual, wanted in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, wanted.grad)


def test_batched_validation_flattens_token_axes_for_injected_loss(monkeypatch):
    model = ToyCausalModel()
    engine = build_engine(model)
    inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])
    labels = torch.tensor([[2, 3, 4], [5, 6, -100]])
    monkeypatch.setattr(
        TrainingEngine, "preprocess", lambda self, batch: (inputs, labels, {})
    )
    shapes = []

    def loss(logits, targets, **kwargs):
        shapes.append((logits.shape, targets.shape))
        return cross_entropy_loss(logits, targets, **kwargs)

    engine.loss_sum = loss
    with torch.no_grad():
        actual = engine.evaluate(None)
        expected = cross_entropy_loss(model(inputs).reshape(-1, 11), labels.reshape(-1))
    assert shapes == [(torch.Size([6, 11]), torch.Size([6]))]
    torch.testing.assert_close(actual, expected)


def test_model_output_contract_rejects_missing_declared_tensor() -> None:
    with pytest.raises(TypeError, match=r"tensor .logits"):
        ModelExecutionCapabilities().output_tensor(object())


@pytest.mark.parametrize("width", [0, -1, True, 1.5])
def test_hidden_output_contract_rejects_invalid_width(width):
    with pytest.raises(ValueError, match="hidden size must be a positive integer"):
        ModelExecutionCapabilities(hidden_states=True, hidden_size=width)


def test_hidden_width_requires_supported_hidden_output():
    with pytest.raises(ValueError, match="requires hidden output support"):
        ModelExecutionCapabilities(hidden_size=8)


def test_chunked_hidden_width_fails_before_projection_or_backward(monkeypatch):
    model = ToyCausalModel()
    engine = build_engine(model, num_chunks=2)
    monkeypatch.setattr(
        model,
        "forward",
        lambda inputs, **kwargs: torch.ones((*inputs.shape, 9), requires_grad=True),
    )

    def reject_projection(*args, **kwargs):
        pytest.fail("Invalid hidden width reached chunked projection")

    engine.chunked_loss_backward = reject_projection
    with pytest.raises(ValueError, match="declared output width"):
        engine.forward_backward_dense(
            torch.ones(4, dtype=torch.long),
            torch.ones(4, dtype=torch.long),
            extra_kwargs={},
            global_valid_tokens=torch.tensor(4),
        )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_optimizer_step_preserves_scheduler_and_ema_order() -> None:
    events = []
    engine = build_engine(ToyCausalModel())
    engine.optimizer = SimpleNamespace(step=lambda: events.append("optimizer"))
    engine.lr_scheduler = SimpleNamespace(step=lambda: events.append("scheduler"))
    engine.ema = SimpleNamespace(step=lambda step: events.append(("ema", step)))

    engine.optimizer_step(4)

    assert events == ["optimizer", "scheduler", ("ema", 4)]


def test_scheduler_failure_does_not_roll_back_optimizer_or_run_ema() -> None:
    events = []

    def fail_scheduler():
        events.append("scheduler")
        raise ValueError("scheduler failed")

    engine = build_engine(ToyCausalModel())
    engine.optimizer = SimpleNamespace(step=lambda: events.append("optimizer"))
    engine.lr_scheduler = SimpleNamespace(step=fail_scheduler)
    engine.ema = SimpleNamespace(step=lambda step: events.append("ema"))

    with pytest.raises(ValueError, match="scheduler failed"):
        engine.optimizer_step(4)

    assert events == ["optimizer", "scheduler"]


def test_engine_accumulation_weights_unequal_valid_target_counts() -> None:
    """Each group uses the step denominator, never a mean of group means."""
    torch.manual_seed(23)
    model = ToyCausalModel()
    reference = deepcopy(model)
    inputs = [torch.tensor([0, 3, 6]), torch.tensor([8, 2, 9, 1, 4])]
    labels = [torch.tensor([-100, 2, -100]), torch.tensor([1, 3, 7, 0, 6])]
    denominator = torch.tensor(6)
    engine = build_engine(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=False)
    expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1, foreach=False)

    actual = sum(
        engine.forward_backward_dense(
            tokens, targets, extra_kwargs={}, global_valid_tokens=denominator
        )
        for tokens, targets in zip(inputs, labels, strict=True)
    )
    expected = cross_entropy_loss(reference(torch.cat(inputs)), torch.cat(labels))
    (expected / denominator).backward()

    assert actual.item() > 0
    torch.testing.assert_close(actual, expected.detach())
    for parameter, target in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(parameter.grad, target.grad)
    optimizer.step()
    expected_optimizer.step()
    for parameter, target in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(parameter, target)


def test_dense_evaluation_preserves_context_and_injected_loss():
    model = ToyCausalModel()
    inputs = torch.tensor([[1, 3, 5], [2, 4, 6]])
    labels = torch.tensor([[3, 5, 7], [4, 6, 8]])
    batch = {"input": inputs, "labels": labels}
    calls = []

    def prepare(value, *, parallel_dims):
        assert parallel_dims is None
        assert value["input"] is inputs
        calls.append("prepare")
        return value["input"], value["labels"], {}

    @contextmanager
    def context():
        calls.append("enter")
        try:
            yield
        finally:
            calls.append("exit")

    def loss(logits, targets, **kwargs):
        assert calls[-1] == "enter"
        assert not torch.is_grad_enabled()
        assert not kwargs
        calls.append("loss")
        return cross_entropy_loss(logits.flatten(0, 1), targets.flatten()) * 2

    model.prepare_inputs = prepare
    with torch.no_grad():
        expected = cross_entropy_loss(model(inputs).flatten(0, 1), labels.flatten()) * 2
        engine = TrainingEngine(
            model_parts=[model],
            device=torch.device("cpu"),
            parallel_dims=None,
            chunked_loss_num_chunks=1,
            param_context=context,
            loss_sum=loss,
        )
        result = engine.evaluate(batch)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert calls == ["prepare", "enter", "loss", "exit"]
    assert all(parameter.grad is None for parameter in model.parameters())
    assert model.training
