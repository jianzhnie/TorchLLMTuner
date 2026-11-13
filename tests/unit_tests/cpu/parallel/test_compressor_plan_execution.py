"""Compressor execution consumes explicit plans before installing forwards."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from llmtuner.config import ParallelConfig
from llmtuner.parallel.context_parallel import compressor_attention as compressor
from llmtuner.parallel.plan import ParallelPlan, PlanTarget


def make_plan(model: torch.nn.Module, *, degree: int = 2) -> ParallelPlan:
    return ParallelPlan(
        stage="cp",
        degree=degree,
        implementation="compressor",
        targets=(
            PlanTarget("decoder", "decoder", model.decoder, "stateless"),
            PlanTarget("attention", "attention", model.attention, "sliding_attention"),
        ),
        module_bindings=tuple(model.named_modules(remove_duplicate=False)),
        parameter_bindings=tuple(model.named_parameters(remove_duplicate=False)),
        buffer_bindings=tuple(model.named_buffers(remove_duplicate=False)),
    )


@pytest.mark.parametrize("stale", [False, True])
def test_invalid_plan_rejected_before_forward_installation(stale: bool) -> None:
    model = torch.nn.Module()
    model.decoder = torch.nn.Linear(2, 2)
    model.attention = torch.nn.Linear(2, 2)
    model.attention.layer_type = "sliding_attention"
    plan = make_plan(model, degree=2 if stale else 4)
    if stale:
        model.attention.weight = torch.nn.Parameter(torch.ones(2, 2))
    original_decoder = model.decoder.forward
    original_attention = model.attention.forward
    mesh = SimpleNamespace(size=lambda: 2)
    with pytest.raises(ValueError):
        compressor.apply_compressor_context_parallel(
            model,
            cp_mesh=mesh,
            cfg=ParallelConfig(),
            resolved_plan=plan,
        )
    assert model.decoder.forward == original_decoder
    assert model.attention.forward == original_attention


def test_execution_consumes_plan_without_resolving_again(monkeypatch) -> None:
    model = torch.nn.Module()
    model.decoder = torch.nn.Linear(2, 2)
    model.attention = torch.nn.Linear(2, 2)
    model.attention.layer_type = "sliding_attention"
    plan = make_plan(model)
    bindings = []
    model.set_cp_mesh = lambda mesh, **kwargs: bindings.append((mesh, kwargs))
    group = object()
    mesh = SimpleNamespace(size=lambda: 2, get_group=lambda: group)

    def reject_resolution(*args, **kwargs):
        pytest.fail("Execution must consume the supplied plan")

    def attention_factory(bound_group):
        assert bound_group is group

        def forward(self, value):
            return value + 1

        return forward

    def decoder_forward(self, value):
        return value * 2

    monkeypatch.setattr(compressor, "resolve_compressor_plan", reject_resolution)
    monkeypatch.setattr(compressor, "compressor_attention_math", object())
    monkeypatch.setattr(compressor, "compressor_attention_forward", attention_factory)
    monkeypatch.setattr(compressor, "compressor_decoder_forward", decoder_forward)
    assert (
        compressor.apply_compressor_context_parallel(
            model, cp_mesh=mesh, cfg=ParallelConfig(), resolved_plan=plan
        )
        is model
    )
    assert bindings == [(mesh, {"strategy": "compressor_ulysses"})]
    value = torch.tensor([3.0])
    assert torch.equal(model.attention(value), torch.tensor([4.0]))
    assert torch.equal(model.decoder(value), torch.tensor([6.0]))


def test_duplicate_compressor_targets_fail_before_forward_installation():
    model = torch.nn.Module()
    model.decoder = torch.nn.Linear(2, 2)
    model.attention = torch.nn.Linear(2, 2)
    model.attention.layer_type = "sliding_attention"
    plan = make_plan(model)
    plan = replace(plan, targets=(*plan.targets, plan.targets[1]))
    original = model.attention.forward
    with pytest.raises(ValueError, match="duplicate attention"):
        compressor.apply_compressor_context_parallel(
            model,
            cp_mesh=SimpleNamespace(size=lambda: 2),
            cfg=ParallelConfig(),
            resolved_plan=plan,
        )
    assert model.attention.forward == original
