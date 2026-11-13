"""TP preflight is local and its bindings cannot silently become stale."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from llmtuner.config.parallel_config import FSDPConfig, ParallelConfig
from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.contracts import DecoderStructure
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel.fully_shard_parallel.plan import FSDPPatternReferences
from llmtuner.parallel.plan import ParallelPlan
from llmtuner.parallel.tensor_parallel.linear import ColwiseLinearNoGather
from llmtuner.parallel.tensor_parallel.tensor_parallel import apply_tp, requires_full_token_history, resolve_tp_plan
from tests.unit_tests.cpu.parallel.test_tp_moe import _MoeModel


@pytest.mark.parametrize("ep_degree", [1, 2])
def test_explicit_qwen_tp_resolution_never_probes_hf_layouts(monkeypatch, ep_degree):
    model = Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            moe_intermediate_size=12,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            num_experts=4,
            num_experts_per_tok=2,
        )
    )
    # A contract on an unrelated child does not grant it decoder ownership.
    model.unowned = nn.Module()
    model.unowned.moe_contract = model.layers[0].mlp.moe_contract
    parameters = tuple(model.parameters())
    rng = torch.get_rng_state().clone()

    def reject(*args, **kwargs):
        raise AssertionError("Explicit models must not use HF layout inspection")

    monkeypatch.setattr(
        "llmtuner.parallel.tensor_parallel.tensor_parallel.looks_like_attention", reject
    )
    plan = resolve_tp_plan(
        model,
        cfg=ParallelConfig(tensor_parallel_size=2, expert_parallel_size=ep_degree),
    )
    assert plan.targets
    moe_targets = [target for target in plan.targets if target.kind == "moe"]
    assert [target.module for target in moe_targets] == (
        [model.layers[0].mlp] if ep_degree == 1 else []
    )
    assert any(
        target.kind == "attention" and target.layout == "gather"
        for target in plan.targets
    )
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("ep_degree", [1, 2])
@pytest.mark.parametrize("contract", [None, "invalid"])
def test_explicit_tp_rejects_missing_sparse_contract(monkeypatch, ep_degree, contract):
    model = Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            moe_intermediate_size=12,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            num_experts=4,
            num_experts_per_tok=2,
        )
    )
    model.layers[0].mlp.moe_contract = contract
    parameters = tuple(model.parameters())
    rng = torch.get_rng_state().clone()

    def reject():
        raise AssertionError("Explicit models must not infer missing sparse math")

    with pytest.raises(
        UnsupportedCombinationError, match="model-owned MoEBlockContract"
    ):
        resolve_tp_plan(
            model,
            cfg=ParallelConfig(tensor_parallel_size=2, expert_parallel_size=ep_degree),
        )
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("shard_recurrence", [False, True])
def test_declared_recurrence_needs_full_tokens_independent_of_class_and_name(
    shard_recurrence,
):
    model = nn.Module()
    model.mixer = nn.Module()
    model.mixer.requires_full_token_history = True
    model.mixer.projection = nn.Linear(8, 12, bias=False)
    model.mlp = nn.Linear(8, 12, bias=False)
    target = "mixer.projection" if shard_recurrence else "mlp"
    if shard_recurrence:
        with pytest.raises(ValueError, match="requires replicated GDN projections"):
            resolve_tp_plan(
                model,
                cfg=ParallelConfig(tensor_parallel_size=2),
                plan={target: "colwise"},
            )
    else:
        plan = resolve_tp_plan(
            model,
            cfg=ParallelConfig(tensor_parallel_size=2),
            plan={target: "colwise"},
        )
        boundary = [item for item in plan.targets if item.kind == "attention"]
        assert len(boundary) == 1
        assert boundary[0].path == "mixer" and boundary[0].module is model.mixer
        assert boundary[0].layout == "replicate_tokens"


@pytest.mark.parametrize("invalid", ["false", 1, 0])
def test_full_history_declaration_rejects_truthy_coercion(invalid):
    module = nn.Module()
    module.requires_full_token_history = invalid
    with pytest.raises(ValueError, match="must be a bool"):
        requires_full_token_history(module)


@pytest.mark.parametrize(
    "before,after",
    [(True, False), (False, True), (None, True), (True, 1), (False, "false")],
)
def test_tp_rejects_changed_token_history_declaration_before_mesh(before, after):
    model = nn.Module()
    model.mixer = nn.Module()
    if before is not None:
        model.mixer.requires_full_token_history = before
    model.mlp = nn.Linear(8, 12, bias=False)
    cfg = ParallelConfig(tensor_parallel_size=2)
    plan = resolve_tp_plan(model, cfg=cfg, plan={"mlp": "colwise"})
    parameters = tuple(model.parameters())
    rng = torch.get_rng_state().clone()
    model.mixer.requires_full_token_history = after
    with pytest.raises(ValueError, match="Stale plan token history requirement: mixer"):
        apply_tp(model, mesh=object(), cfg=cfg, resolved_plan=plan)
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("shard_attention", [False, True])
def test_declared_attention_name_controls_head_geometry_preflight(shard_attention):
    model = nn.Module()
    model.config = SimpleNamespace(num_attention_heads=3, num_key_value_heads=3)
    model.embedding = nn.Embedding(8, 8)
    model.norm = nn.Identity()
    model.head = nn.Linear(8, 8, bias=False)
    block = nn.Module()
    block.attention = nn.Module()
    block.attention.projection = nn.Linear(8, 12, bias=False)
    block.mlp = nn.Linear(8, 12, bias=False)
    model.blocks = nn.ModuleList([block])
    model.decoder_structure = DecoderStructure(
        decoder="",
        embedding="embedding",
        blocks="blocks",
        norm="norm",
        head="head",
        attention="attention",
    )
    target = "blocks.0.attention.projection" if shard_attention else "blocks.0.mlp"
    parameters = tuple(model.parameters())
    rng = torch.get_rng_state().clone()
    if shard_attention:
        with pytest.raises(ValueError, match="num_attention_heads.*divisible"):
            resolve_tp_plan(
                model,
                cfg=ParallelConfig(tensor_parallel_size=2),
                plan={target: "colwise"},
            )
    else:
        plan = resolve_tp_plan(
            model,
            cfg=ParallelConfig(tensor_parallel_size=2),
            plan={target: "colwise"},
        )
        assert plan.targets[0].path == target
        attention_targets = [item for item in plan.targets if item.kind == "attention"]
        assert len(attention_targets) == 1
        assert attention_targets[0].path == "blocks.0.attention"
        assert attention_targets[0].module is block.attention
        assert attention_targets[0].layout == "replicate_tokens"
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(torch.get_rng_state(), rng)


def test_resolving_tp_preserves_model_storage_rng_and_forward():
    model = nn.Sequential(nn.Linear(8, 12, bias=False))
    parameter = model[0].weight
    forward = model[0].forward
    rng = torch.get_rng_state().clone()
    plan = resolve_tp_plan(
        model, cfg=ParallelConfig(tensor_parallel_size=2), plan={"0": "colwise"}
    )
    assert plan.targets[0].module is model[0]
    assert plan.targets[0].spec.kind == "colwise"
    assert plan.stage == "tp"
    assert model[0].weight is parameter
    assert model[0].forward == forward
    assert torch.equal(torch.get_rng_state(), rng)
    plan.validate_binding(model)


def test_invalid_moe_shared_geometry_fails_during_pure_preflight():
    model = _MoeModel()
    block = model.layers[0]["mlp"]
    shared = nn.Module()
    shared.gate_proj = nn.Linear(16, 7, bias=False)
    shared.up_proj = nn.Linear(16, 7, bias=False)
    shared.down_proj = nn.Linear(7, 16, bias=False)
    block.shared_expert = shared
    parameters = tuple(model.parameters())
    with pytest.raises(ValueError, match="not divisible"):
        resolve_tp_plan(model, cfg=ParallelConfig(tensor_parallel_size=2))
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert not getattr(block, "_tp_moe_boundary", False)


@pytest.mark.parametrize(
    "replacement", ["module", "parameter", "alias", "new_module", "forward"]
)
def test_resolved_tp_rejects_replaced_bindings_before_mesh_access(replacement):
    model = nn.Sequential(nn.Linear(8, 12, bias=False))
    cfg = ParallelConfig(tensor_parallel_size=2)
    plan = resolve_tp_plan(model, cfg=cfg, plan={"0": "colwise"})
    if replacement == "forward":
        model[0].forward = lambda inputs: inputs
    elif replacement == "module":
        model[0] = nn.Linear(8, 12, bias=False)
    elif replacement == "parameter":
        model[0].weight = nn.Parameter(model[0].weight.detach().clone())
    elif replacement == "alias":
        model.register_parameter("external_alias", model[0].weight)
    else:
        model.add_module("extra", nn.Identity())
    with pytest.raises(ValueError, match="Stale plan"):
        apply_tp(model, mesh=object(), cfg=cfg, resolved_plan=plan)


def test_tp_preflight_rejects_invalid_geometry_without_replacing_parameters():
    model = nn.Sequential(nn.Linear(8, 11, bias=False))
    parameter = model[0].weight
    with pytest.raises(ValueError, match="not divisible"):
        resolve_tp_plan(
            model, cfg=ParallelConfig(tensor_parallel_size=2), plan={"0": "colwise"}
        )
    assert model[0].weight is parameter


def test_resolved_tp_rejects_ambiguous_or_mismatched_requests():
    model = nn.Sequential(nn.Linear(8, 12, bias=False))
    cfg = ParallelConfig(tensor_parallel_size=2)
    plan = resolve_tp_plan(model, cfg=cfg, plan={"0": "colwise"})
    with pytest.raises(ValueError, match="either plan or resolved_plan"):
        apply_tp(model, object(), cfg, plan={"0": "colwise"}, resolved_plan=plan)
    with pytest.raises(ValueError, match="degree"):
        apply_tp(
            model, object(), ParallelConfig(tensor_parallel_size=4), resolved_plan=plan
        )
    with pytest.raises(ValueError, match="Expected a TP plan"):
        apply_tp(
            model,
            object(),
            cfg,
            resolved_plan=ParallelPlan("ep", (), 2, "alltoall"),
        )


def test_attention_realizer_and_boundary_are_resolved_before_execution():
    attention = nn.Module()
    for name in ("q_proj", "k_proj", "v_proj"):
        attention.add_module(name, nn.Linear(8, 8, bias=False))
    model = nn.Sequential(attention)
    plan = resolve_tp_plan(
        model,
        cfg=ParallelConfig(tensor_parallel_size=2),
        plan={f"0.{name}": "colwise" for name in ("q_proj", "k_proj", "v_proj")},
    )
    boundary = next(target for target in plan.targets if target.kind == "attention")
    assert boundary.module is attention
    assert boundary.layout == "gather"
    assert all(
        target.spec.implementation is ColwiseLinearNoGather
        for target in plan.targets
        if target.kind == "dense"
    )


def test_tp_parameter_mapping_rebinds_fsdp_exclusions():
    model = nn.Sequential(nn.Linear(8, 12, bias=False))
    original = model[0]
    weight = original.weight
    cfg = ParallelConfig(
        tensor_parallel_size=2,
        fsdp_config=FSDPConfig(fsdp_apply_modules={"": {"ignored_params": {weight}}}),
    )
    references = FSDPPatternReferences(model, cfg)
    axis = SimpleNamespace(
        device_type="cpu",
        size=lambda: 2,
        get_group=lambda: None,
        get_local_rank=lambda: 0,
    )
    apply_tp(
        model,
        {"tp": axis},
        cfg,
        plan={"0": "colwise"},
        on_replacement=references.record_replacements,
    )
    assert model[0] is not original
    assert model[0].weight is not weight
    bound = references.bind(model)
    assert bound.fsdp_config.fsdp_apply_modules[""]["ignored_params"] == {
        model[0].weight
    }


def test_resolved_tp_rejects_resized_storage_before_mesh_access():
    model = nn.Sequential(nn.Linear(8, 12, bias=False))
    cfg = ParallelConfig(tensor_parallel_size=2)
    plan = resolve_tp_plan(model, cfg=cfg, plan={"0": "colwise"})
    parameter = model[0].weight
    parameter.data = parameter.data[:10]
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="Stale plan parameter shape: 0.weight"):
        apply_tp(model, mesh=object(), cfg=cfg, resolved_plan=plan)
    assert model[0].weight is parameter
    assert isinstance(model[0], nn.Linear)
    assert torch.equal(torch.get_rng_state(), rng)


def test_resolved_tp_allows_parameter_value_updates():
    model = nn.Sequential(nn.Linear(8, 12, bias=False))
    plan = resolve_tp_plan(
        model, cfg=ParallelConfig(tensor_parallel_size=2), plan={"0": "colwise"}
    )
    with torch.no_grad():
        model[0].weight.add_(1)
    plan.validate_binding(model)


@pytest.mark.parametrize(
    "mutation", ["dtype", "device", "stride", "offset", "requires_grad"]
)
def test_resolved_tp_rejects_changed_parameter_metadata_before_mesh(mutation):
    model = nn.Sequential(nn.Linear(8, 12, bias=False))
    cfg = ParallelConfig(tensor_parallel_size=2)
    plan = resolve_tp_plan(model, cfg=cfg, plan={"0": "colwise"})
    parameter = model[0].weight
    if mutation == "dtype":
        parameter.data = parameter.data.double()
    elif mutation == "device":
        torch.utils.swap_tensors(
            parameter, nn.Parameter(torch.empty_like(parameter, device="meta"))
        )
    elif mutation == "stride":
        parameter.data = parameter.data.t().contiguous().t()
    elif mutation == "offset":
        parameter.data = torch.empty(parameter.numel() + 1)[1:].view_as(parameter)
    else:
        parameter.requires_grad_(False)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="Stale plan parameter metadata: 0.weight"):
        apply_tp(model, mesh=object(), cfg=cfg, resolved_plan=plan)
    assert model[0].weight is parameter
    assert isinstance(model[0], nn.Linear)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("mutation", ["replace", "alias", "remove"])
def test_resolved_tp_rejects_changed_buffer_ownership_before_mesh(mutation):
    model = nn.Sequential(nn.Linear(8, 12, bias=False))
    model.register_buffer("routing_state", torch.arange(4))
    cfg = ParallelConfig(tensor_parallel_size=2)
    plan = resolve_tp_plan(model, cfg=cfg, plan={"0": "colwise"})
    if mutation == "replace":
        model.routing_state = model.routing_state.clone()
    elif mutation == "alias":
        model.register_buffer("external_alias", model.routing_state)
    else:
        del model.routing_state
    with pytest.raises(ValueError, match="Stale plan buffer"):
        apply_tp(model, mesh=object(), cfg=cfg, resolved_plan=plan)


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("owner_path", ["", "0"])
def test_resolved_tp_rejects_changed_buffer_persistence_before_mesh(
    persistent, owner_path
):
    model = nn.Sequential(nn.Linear(8, 12, bias=False))
    owner = model.get_submodule(owner_path)
    owner.register_buffer("routing_state", torch.arange(4), persistent=persistent)
    cfg = ParallelConfig(tensor_parallel_size=2)
    plan = resolve_tp_plan(model, cfg=cfg, plan={"0": "colwise"})
    original = owner.routing_state
    keys = tuple(model.state_dict())
    owner.register_buffer("routing_state", original, persistent=not persistent)
    assert owner.routing_state is original
    assert tuple(model.state_dict()) != keys
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="Stale plan buffer persistence"):
        apply_tp(model, mesh=object(), cfg=cfg, resolved_plan=plan)
    assert isinstance(model[0], nn.Linear)
    assert owner.routing_state is original
    assert torch.equal(torch.get_rng_state(), rng)


def test_resolved_tp_allows_buffer_value_updates_without_rebinding():
    model = nn.Sequential(nn.Linear(8, 12, bias=False))
    model.register_buffer("routing_state", torch.arange(4))
    plan = resolve_tp_plan(
        model, cfg=ParallelConfig(tensor_parallel_size=2), plan={"0": "colwise"}
    )
    model.routing_state.add_(3)
    plan.validate_binding(model)
