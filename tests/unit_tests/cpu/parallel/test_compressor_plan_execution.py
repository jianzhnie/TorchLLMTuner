"""Compressor execution consumes explicit plans before installing forwards."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from llmtuner.config import ParallelConfig
from llmtuner.config.parallel_config import ContextParallelConfig
from llmtuner.models.contracts import DecoderStructure
from llmtuner.parallel.context_parallel import compressor_attention as compressor
from llmtuner.parallel.plan import ParallelPlan, PlanTarget


def make_plan(model: torch.nn.Module, *, degree: int = 2) -> ParallelPlan:
    model.decoder.validate_compressor_context_parallel = lambda **kwargs: None
    model.attention.num_heads = degree
    model.attention.attention_dropout = 0.0
    model.attention.validate_compressor_context_parallel = lambda **kwargs: None
    model.decoder.layers = torch.nn.ModuleList(
        [torch.nn.ModuleDict({"self_attn": model.attention})]
    )
    model.decoder_structure = DecoderStructure(
        decoder="decoder",
        embedding="embedding",
        blocks="decoder.layers",
        norm="norm",
        head="head",
        attention="self_attn",
    )
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
    monkeypatch.setattr(compressor, "compressor_attention_forward", attention_factory)
    model.decoder.forward_compressor_context_parallel = decoder_forward.__get__(
        model.decoder
    )
    model.attention.forward_compressor_context_parallel = lambda *args: None
    assert (
        compressor.apply_compressor_context_parallel(
            model,
            cp_mesh=mesh,
            cfg=ParallelConfig(
                context_parallel_ulysses_size=2,
                cp_config=ContextParallelConfig(context_parallel_strategy="ulysses"),
            ),
            resolved_plan=plan,
        )
        is model
    )
    assert bindings == [(mesh, {})]
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


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ("degree", "degree does not match configuration"),
        ("validation", "model-owned execution validation"),
        ("validator_reject", "Indexer contract changed"),
        ("decoder_validation", "model-owned execution validation"),
        ("forward", "model-owned forward interfaces"),
        ("mesh_setter", "model-owned set_cp_mesh"),
    ],
)
def test_changed_configuration_rejected_before_installation(changed, message):
    model = torch.nn.Module()
    model.decoder = torch.nn.Linear(2, 2)
    model.attention = torch.nn.Linear(2, 2)
    model.attention.layer_type = "sliding_attention"
    make_plan(model)
    model.decoder.forward_compressor_context_parallel = lambda *args: None
    model.attention.forward_compressor_context_parallel = lambda *args: None
    model.set_cp_mesh = lambda *args, **kwargs: pytest.fail("Unexpected mesh binding")
    cfg = ParallelConfig(
        context_parallel_ulysses_size=2,
        cp_config=ContextParallelConfig(context_parallel_strategy="ulysses"),
    )
    plan = compressor.resolve_compressor_plan(model, degree=2, cfg=cfg)
    if changed == "degree":
        cfg.context_parallel_ulysses_size = 4
    elif changed == "validation":
        model.attention.validate_compressor_context_parallel = None
    elif changed == "validator_reject":

        def reject_contract(**kwargs):
            raise ValueError("Indexer contract changed")

        model.attention.validate_compressor_context_parallel = reject_contract
    elif changed == "decoder_validation":
        model.decoder.validate_compressor_context_parallel = None
    elif changed == "forward":
        model.attention.forward_compressor_context_parallel = None
    else:
        model.set_cp_mesh = None
    original_decoder = model.decoder.forward
    original_attention = model.attention.forward
    mesh = SimpleNamespace(
        size=lambda: 2,
        get_group=lambda: pytest.fail("Unexpected group acquisition"),
    )
    with pytest.raises((ValueError, TypeError), match=message):
        compressor.apply_compressor_context_parallel(
            model, cp_mesh=mesh, cfg=cfg, resolved_plan=plan
        )
    assert model.decoder.forward == original_decoder
    assert model.attention.forward == original_attention
