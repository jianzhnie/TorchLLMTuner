from contextlib import contextmanager, nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llmtuner.components.loss import cross_entropy_loss
from llmtuner.trainer.runtime.engine import (
    TrainingEngine,
    evaluate_dense_batch,
    model_output_value,
)


class ToyCausalModel(nn.Module):
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
        max_context_length=16,
        chunked_loss_num_chunks=num_chunks,
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


def test_model_output_value_rejects_missing_declared_tensor() -> None:
    with pytest.raises(TypeError, match=r"tensor .logits"):
        model_output_value(object())


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

    def prepare(value, *, parallel_dims, max_context_length):
        assert parallel_dims is None
        assert max_context_length == 16
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
        result = evaluate_dense_batch(
            model,
            batch,
            device=torch.device("cpu"),
            parallel_dims=None,
            max_context_length=16,
            param_context=context,
            loss_sum=loss,
        )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert calls == ["prepare", "enter", "loss", "exit"]
    assert all(parameter.grad is None for parameter in model.parameters())
    assert model.training
