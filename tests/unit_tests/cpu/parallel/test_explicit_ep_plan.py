"""Model-owned Qwen3 EP plans remain pure and reject stale execution."""

from dataclasses import replace

import pytest
import torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)

from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.common.aux_loss import AuxLoss
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.models.qwen.qwen3_5_moe.model import Qwen3_5MoeForCausalLM
from llmtuner.parallel.expert_parallel import block_conversion, model_replacement
from llmtuner.parallel.tensor_parallel.tensor_parallel import resolve_tp_plan
from tests.unit_tests.cpu.models.test_owned_family_construction import text_config
from tests.unit_tests.cpu.models.test_qwen3_model_contract import model_config

TOL = 1e-6


@pytest.fixture(autouse=True)
def restore_auxiliary_state():
    counts = dict(AuxLoss._group_counts)
    accumulation = dict(AuxLoss.group_acc)
    denominator = AuxLoss._step_denominator
    try:
        yield
    finally:
        AuxLoss._group_counts.clear()
        AuxLoss._group_counts.update(counts)
        AuxLoss.group_acc.clear()
        AuxLoss.group_acc.update(accumulation)
        AuxLoss._step_denominator = denominator


def _model():
    config = model_config(num_hidden_layers=2, num_experts=8, mlp_only_layers=[])
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    torch.manual_seed(19)
    return Qwen3MoeForCausalLM(config).double().eval()


def _data():
    return torch.tensor([1, 7, 3, 9, 2]), torch.arange(5)


def test_ep_only_preflight_calls_model_capability_contract(monkeypatch):
    model = _model()
    calls = []
    original = model.validate_parallel

    def validate(*, tp_degree, ep_degree):
        calls.append((tp_degree, ep_degree))
        original(tp_degree=tp_degree, ep_degree=ep_degree)

    monkeypatch.setattr(model, "validate_parallel", validate)
    parameters = tuple(model.parameters())
    rng = torch.get_rng_state().clone()
    plan = model_replacement.resolve_ep_plan(model, degree=2)
    assert plan.degree == 2
    assert calls == [(1, 2)]
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(torch.get_rng_state(), rng)


def test_ep_only_preflight_rejects_stale_model_declared_geometry():
    model = _model()
    model.config.hidden_size += 1
    parameters = tuple(model.parameters())
    rng = torch.get_rng_state().clone()
    with pytest.raises(
        ValueError, match="gate/up storage differs from declared geometry"
    ):
        model_replacement.resolve_ep_plan(model, degree=2)
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(torch.get_rng_state(), rng)


def test_ep_execution_revalidates_model_geometry_before_conversion():
    model = _model()
    plan = model_replacement.resolve_ep_plan(model)
    model.config.hidden_size += 1
    blocks = tuple(layer.mlp for layer in model.layers)
    parameters = tuple(model.parameters())
    rng = torch.get_rng_state().clone()
    with pytest.raises(
        ValueError, match="gate/up storage differs from declared geometry"
    ):
        model_replacement.swap_moe_blocks(model, resolved_plan=plan)
    assert all(
        layer.mlp is block for layer, block in zip(model.layers, blocks, strict=True)
    )
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("contract", [None, "invalid"])
def test_explicit_ep_rejects_missing_math_contract_without_legacy_probe(
    monkeypatch, contract
):
    model = _model()
    model.layers[0].mlp.moe_contract = contract
    blocks = tuple(layer.mlp for layer in model.layers)
    parameters = tuple(model.parameters())
    rng = torch.get_rng_state().clone()

    def reject(*args, **kwargs):
        raise AssertionError("Explicit EP must not probe legacy layouts")

    monkeypatch.setattr(block_conversion, "_legacy_inspection", reject)
    monkeypatch.setattr(model_replacement, "_legacy_inspection", reject)
    with pytest.raises(
        UnsupportedCombinationError, match="model-owned MoEBlockContract"
    ):
        model_replacement.resolve_ep_plan(model)
    assert all(
        layer.mlp is block for layer, block in zip(model.layers, blocks, strict=True)
    )
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("persistent", [False, True])
def test_ep_plan_rejects_buffer_persistence_change_before_replacement(persistent):
    model = _model()
    owner = model.layers[0].mlp
    owner.register_buffer("migration_marker", torch.ones(2), persistent=persistent)
    plan = model_replacement.resolve_ep_plan(model)
    marker = owner.migration_marker
    blocks = tuple(layer.mlp for layer in model.layers)
    owner.register_buffer("migration_marker", marker, persistent=not persistent)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="Stale plan buffer persistence"):
        model_replacement.swap_moe_blocks(model, resolved_plan=plan)
    assert all(a.mlp is b for a, b in zip(model.layers, blocks, strict=True))
    assert owner.migration_marker is marker
    assert torch.equal(torch.get_rng_state(), rng)


def test_ep_target_plan_is_pure_and_execution_does_not_rediscover(monkeypatch):
    model = _model()
    parameters = tuple(model.parameters())
    keys = tuple(model.state_dict())
    rng = torch.get_rng_state().clone()
    ids, positions = _data()
    with torch.no_grad():
        expected = model(ids, positions=positions)
    plan = model_replacement.resolve_ep_plan(model)
    assert plan.count("moe") == 2
    assert tuple(model.state_dict()) == keys
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(rng, torch.get_rng_state())

    def reject_discovery(*args, **kwargs):
        raise AssertionError("Execution must consume resolved decoder slots")

    monkeypatch.setattr(model_replacement, "find_moe_candidates", reject_discovery)
    for name in ("_legacy_inspection", "resolve_moe_routing", "resolve_packed_experts"):
        monkeypatch.setattr(block_conversion, name, reject_discovery)
    monkeypatch.setattr(model_replacement, "_legacy_inspection", reject_discovery)
    model_replacement.swap_moe_blocks(
        model, resolved_plan=plan, router_aux_loss_coef=0.0
    )
    with torch.no_grad():
        actual = model(ids, positions=positions)
    torch.testing.assert_close(actual, expected, atol=TOL, rtol=TOL)


@pytest.mark.parametrize(
    "change", ["parameter", "device", "alias", "backend", "degree"]
)
def test_ep_target_plan_rejects_stale_or_incompatible_execution(change):
    model = _model()
    plan = model_replacement.resolve_ep_plan(
        model,
        degree=2 if change == "degree" else 1,
        implementation="torchao" if change == "backend" else "alltoall",
    )
    original_blocks = tuple(layer.mlp for layer in model.layers)
    if change == "parameter":
        experts = model.layers[0].mlp.experts
        experts.down_proj = torch.nn.Parameter(experts.down_proj.detach().clone())
    elif change == "device":
        parameter = next(model.parameters())
        torch.utils.swap_tensors(
            parameter,
            torch.nn.Parameter(torch.empty_like(parameter, device="meta")),
        )
    elif change == "alias":
        model.register_parameter("unexpected_alias", next(model.parameters()))
    with pytest.raises(ValueError, match="Stale plan|does not match"):
        model_replacement.swap_moe_blocks(model, resolved_plan=plan)
    assert all(
        layer.mlp is original
        for layer, original in zip(model.layers, original_blocks, strict=True)
    )


@pytest.mark.parametrize("degree", [0, -1, True, 1.5, 3])
def test_ep_plan_rejects_invalid_degree_without_changing_model(degree):
    model = _model()
    parameters = tuple(model.parameters())
    keys = tuple(model.state_dict())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="degree"):
        model_replacement.resolve_ep_plan(model, degree=degree)
    assert tuple(model.state_dict()) == keys
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(rng, torch.get_rng_state())


def test_ep_packed_plan_keeps_parameter_references_and_rejects_changed_geometry():
    model = _model()
    plan = model_replacement.resolve_ep_plan(model)
    target = plan.targets[0]
    binding = target.spec.packed_experts
    experts = target.module.experts
    assert binding.gate_up is experts.gate_up_proj
    assert binding.down is experts.down_proj
    # No Parameter replacement: identity checks alone cannot detect .data resizing.
    experts.down_proj.data = experts.down_proj.data[:, :, :-1]
    original_blocks = tuple(layer.mlp for layer in model.layers)
    with pytest.raises(ValueError, match="Stale plan parameter shape"):
        model_replacement.swap_moe_blocks(model, resolved_plan=plan)
    assert all(
        layer.mlp is block
        for layer, block in zip(model.layers, original_blocks, strict=True)
    )


def test_ep_packed_binding_rejects_another_block():
    model = _model()
    plan = model_replacement.resolve_ep_plan(model)
    binding = plan.targets[0].spec.packed_experts
    with pytest.raises(ValueError, match="Stale EP packed"):
        binding.projections(model.layers[1].mlp)


@pytest.mark.parametrize(
    "change",
    [
        "top_k",
        "route_norm",
        "route_scale",
        "activation",
        "shared_gate",
        "output",
        "remove",
    ],
)
def test_changed_model_semantics_reject_before_replacement(monkeypatch, change):
    model = _model()
    plan = model_replacement.resolve_ep_plan(model)
    block = model.layers[0].mlp
    original_blocks = tuple(layer.mlp for layer in model.layers)
    if change == "output":
        block.router_logits_in_output = not block.router_logits_in_output
    elif change == "remove":
        del block.moe_contract
    elif change == "shared_gate":
        block.moe_contract = replace(
            block.moe_contract,
            shared_expert="declared_shared",
            shared_expert_gate="declared_gate",
        )
    else:
        value = {
            "top_k": 1,
            "route_norm": not block.moe_contract.route_norm,
            "route_scale": 2.0,
            "activation": "gelu",
        }[change]
        block.moe_contract = replace(block.moe_contract, **{change: value})

    def forbidden(*args, **kwargs):
        raise AssertionError("stale contract reached expert allocation")

    monkeypatch.setattr(model_replacement, "convert_block", forbidden)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="Stale EP.*contract"):
        model_replacement.swap_moe_blocks(model, resolved_plan=plan)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(
        layer.mlp is original
        for layer, original in zip(model.layers, original_blocks, strict=True)
    )


@pytest.mark.parametrize("engine", ["tp", "ep"])
def test_missing_declared_shared_expert_rejects_during_pure_resolution(engine):
    model = _model()
    block = model.layers[0].mlp
    block.moe_contract = replace(block.moe_contract, shared_expert="missing_shared")
    parameters = tuple(model.parameters())
    blocks = tuple(layer.mlp for layer in model.layers)
    keys = tuple(model.state_dict())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="shared expert.*must be a child module"):
        if engine == "ep":
            model_replacement.resolve_ep_plan(model)
        else:
            resolve_tp_plan(model, cfg=ParallelConfig(tensor_parallel_size=2))
    assert tuple(model.state_dict()) == keys
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert all(a is b.mlp for a, b in zip(blocks, model.layers, strict=True))
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("family", ["qwen3", "qwen35"])
def test_non_silu_experts_reject_before_ep_changes_mathematics(degree, family):
    if family == "qwen3":
        config = model_config(hidden_act="gelu", mlp_only_layers=[])
        constructor = Qwen3MoeForCausalLM
    else:
        config = text_config(Qwen3_5MoeTextConfig)
        config.hidden_act = "gelu"
        constructor = Qwen3_5MoeForCausalLM
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    model = constructor(config)
    assert model.layers[0].mlp.moe_contract.activation == "gelu"
    tokens, positions = _data()
    with torch.no_grad():
        logits = model(tokens, positions=positions).logits
    assert logits.isfinite().all() and logits.count_nonzero() > 0
    parameters = tuple(model.parameters())
    keys = tuple(model.state_dict())
    blocks = tuple(layer.mlp for layer in model.layers)
    rng = torch.get_rng_state().clone()
    with pytest.raises(UnsupportedCombinationError, match="require SiLU.*gelu"):
        model_replacement.resolve_ep_plan(model, degree=degree)
    assert tuple(model.state_dict()) == keys
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert all(a is b.mlp for a, b in zip(blocks, model.layers, strict=True))
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("projection", ["gate_up_proj", "down_proj"])
@pytest.mark.parametrize("shape", [(), (4,), (4, 8), (1, 4, 8, 16)])
def test_malformed_packed_rank_rejects_with_explicit_geometry_error(projection, shape):
    model = _model()
    block = model.layers[0].mlp
    setattr(block.experts, projection, torch.nn.Parameter(torch.empty(shape)))
    parameters = tuple(model.parameters())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="requires rank-3 expert parameters"):
        block_conversion.resolve_packed_experts(block, degree=1)
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("degree", [0, -1, True, 1.5, "2", None])
def test_packed_resolver_validates_degree_before_reading_storage(degree):
    class UnreadableBlock(torch.nn.Module):
        def __getattribute__(self, name):
            if name == "_modules":
                raise AssertionError("Invalid degree reached expert storage")
            return super().__getattribute__(name)

    block = UnreadableBlock()
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="EP degree must be a positive integer"):
        block_conversion.resolve_packed_experts(block, degree=degree)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("shape", [(), (8,), (7, 16), (8, 15), (1, 8, 16)])
def test_router_geometry_rejects_in_pure_ep_preflight(shape, monkeypatch):
    model = _model()
    block = model.layers[0].mlp
    block.gate.weight = torch.nn.Parameter(torch.empty(shape))
    parameters = tuple(model.parameters())
    blocks = tuple(layer.mlp for layer in model.layers)
    keys = tuple(model.state_dict())
    rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid router geometry reached expert allocation")

    monkeypatch.setattr(model_replacement, "convert_block", forbidden)
    with pytest.raises(ValueError, match="router weight must have shape"):
        model_replacement.resolve_ep_plan(model)
    assert tuple(model.state_dict()) == keys
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert all(a is b.mlp for a, b in zip(blocks, model.layers, strict=True))
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("missing", ["experts", "gate_up_proj", "down_proj"])
def test_declared_packed_layout_never_falls_back_to_legacy_probe(missing, monkeypatch):
    model = _model()
    block = model.layers[0].mlp
    if missing == "experts":
        del block.experts
    else:
        delattr(block.experts, missing)
    parameters = tuple(model.parameters())
    keys = tuple(model.state_dict())
    rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("Declared layouts must not enter legacy inspection")

    monkeypatch.setattr(block_conversion, "_legacy_inspection", forbidden)
    monkeypatch.setattr(model_replacement, "_legacy_inspection", forbidden)
    with pytest.raises(ValueError, match="Declared fused experts require"):
        model_replacement.resolve_ep_plan(model)
    assert tuple(model.state_dict()) == keys
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(torch.get_rng_state(), rng)


def test_explicit_ep_missing_slot_never_enters_legacy_inspection(monkeypatch):
    model = _model()
    del model.layers[1].moe_block_name
    parameters = tuple(model.parameters())
    keys = tuple(model.state_dict())
    rng = torch.get_rng_state().clone()

    def forbidden():
        raise AssertionError("explicit decoder entered legacy inspection")

    monkeypatch.setattr(model_replacement, "_legacy_inspection", forbidden)
    with pytest.raises(ValueError, match="requires moe_block_name"):
        model_replacement.resolve_ep_plan(model)
    assert tuple(model.state_dict()) == keys
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(rng, torch.get_rng_state())


def test_ep_uses_declared_blocks_instead_of_convenience_property(monkeypatch):
    model = _model()
    monkeypatch.setattr(
        Qwen3MoeForCausalLM, "layers", property(lambda self: torch.nn.ModuleList())
    )
    plan = model_replacement.resolve_ep_plan(model)
    assert len(plan.targets) == 2
    assert plan.targets[0].module is model.model.layers[0].mlp
    assert plan.targets[1].module is model.model.layers[1].mlp


def test_ep_rejects_malformed_structure_without_legacy_fallback(monkeypatch):
    model = _model()
    model.decoder_structure = "model.layers"
    monkeypatch.setattr(
        model_replacement,
        "_legacy_inspection",
        lambda: pytest.fail("malformed structure entered legacy inspection"),
    )
    with pytest.raises(TypeError, match="decoder_structure"):
        model_replacement.resolve_ep_plan(model)


@pytest.mark.parametrize("flag", ["false", 0, None, True])
def test_ep_resolution_rejects_ambiguous_router_output_before_allocation(flag):
    model = _model()
    block = model.layers[0].mlp
    block.router_logits_in_output = flag
    parameters = tuple(model.parameters())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="router.*boolean|router output.*conflicts"):
        model_replacement.resolve_ep_plan(model)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert model.layers[0].mlp is block


def test_ep_resolution_consumes_contract_only_router_output():
    model = _model()
    block = model.layers[0].mlp
    del block.router_logits_in_output
    block.moe_contract = replace(block.moe_contract, router_logits_in_output=True)
    model.config.output_router_logits = True
    with pytest.raises(NotImplementedError, match="returns router logits"):
        model_replacement.resolve_ep_plan(model)


def test_ep_plan_rejects_new_external_buffer_alias_before_conversion(monkeypatch):
    model = _model()
    model.layers[0].mlp.register_buffer("routing_state", torch.arange(4))
    plan = model_replacement.resolve_ep_plan(model)
    model.register_buffer("external_alias", model.layers[0].mlp.routing_state)

    def forbidden(*args, **kwargs):
        raise AssertionError("Stale buffer ownership reached expert conversion")

    monkeypatch.setattr(model_replacement, "convert_block", forbidden)
    with pytest.raises(ValueError, match="Stale plan buffer ownership"):
        model_replacement.swap_moe_blocks(model, resolved_plan=plan)


@pytest.mark.parametrize(
    ("implementation", "error", "message"),
    [
        ("unknown", ValueError, "ep_token_dispatcher"),
        ("deepep", NotImplementedError, "registered gap"),
        ("hybridep", NotImplementedError, "registered gap"),
    ],
)
def test_ep_preflight_rejects_unavailable_dispatcher_before_model_binding(
    monkeypatch, implementation, error, message
):
    model = _model()
    rng = torch.get_rng_state().clone()
    parameters = tuple(model.parameters())

    def forbidden(*args, **kwargs):
        raise AssertionError("Unavailable dispatcher reached model binding")

    monkeypatch.setattr(model_replacement, "find_moe_candidates", forbidden)
    with pytest.raises(error, match=message):
        model_replacement.resolve_ep_plan(model, implementation=implementation)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))


def test_ep_preflight_rejects_pre_down_score_semantics_before_model_binding(
    monkeypatch,
):
    model = _model()
    model.config.experts_implementation = "pre_down"
    rng = torch.get_rng_state().clone()
    parameters = tuple(model.parameters())

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported score semantics reached model binding")

    monkeypatch.setattr(model_replacement, "find_moe_candidates", forbidden)
    with pytest.raises(UnsupportedCombinationError, match="pre_down"):
        model_replacement.resolve_ep_plan(model)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))


def test_ep_execution_rejects_pre_down_enabled_after_plan_resolution(monkeypatch):
    model = _model()
    plan = model_replacement.resolve_ep_plan(model)
    model.config.experts_implementation = "pre_down"
    rng = torch.get_rng_state().clone()
    blocks = tuple(layer.mlp for layer in model.layers)
    parameters = tuple(model.parameters())

    def forbidden(*args, **kwargs):
        raise AssertionError("Changed score semantics reached expert conversion")

    monkeypatch.setattr(model_replacement, "convert_block", forbidden)
    with pytest.raises(UnsupportedCombinationError, match="pre_down"):
        model_replacement.swap_moe_blocks(model, resolved_plan=plan)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert all(a is b.mlp for a, b in zip(blocks, model.layers, strict=True))


def test_ep_execution_rechecks_router_output_config_before_conversion(monkeypatch):
    model = _model()
    block = model.layers[0].mlp
    del block.router_logits_in_output
    block.moe_contract = replace(block.moe_contract, router_logits_in_output=True)
    model.config.output_router_logits = False
    plan = model_replacement.resolve_ep_plan(model)
    model.config.output_router_logits = True
    parameters = tuple(model.parameters())
    blocks = tuple(layer.mlp for layer in model.layers)
    rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("Changed router output reached expert conversion")

    monkeypatch.setattr(model_replacement, "convert_block", forbidden)
    with pytest.raises(UnsupportedCombinationError, match="returns router logits"):
        model_replacement.swap_moe_blocks(model, resolved_plan=plan)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert all(a is b.mlp for a, b in zip(blocks, model.layers, strict=True))


@pytest.mark.parametrize("slot", [None, "self_attn", "missing", 3, "deleted"])
def test_ep_plan_rejects_changed_decoder_slot_declaration(monkeypatch, slot):
    model = _model()
    plan = model_replacement.resolve_ep_plan(model)
    if slot == "deleted":
        del model.layers[0].moe_block_name
    else:
        model.layers[0].moe_block_name = slot
    parameters = tuple(model.parameters())
    blocks = tuple(layer.mlp for layer in model.layers)
    rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("Stale decoder declaration reached expert conversion")

    monkeypatch.setattr(model_replacement, "convert_block", forbidden)
    with pytest.raises(ValueError, match="Stale EP decoder declaration"):
        model_replacement.swap_moe_blocks(model, resolved_plan=plan)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert all(a is b.mlp for a, b in zip(blocks, model.layers, strict=True))


@pytest.mark.parametrize(
    ("projection", "degree", "alias_kind"),
    [
        ("gate_up_proj", 1, "parameter"),
        ("gate_up_proj", 2, "parameter"),
        ("down_proj", 2, "parameter"),
        ("gate_up_proj", 1, "buffer_view"),
        ("down_proj", 2, "parameter_view"),
    ],
)
def test_ep_preflight_rejects_external_packed_expert_aliases(
    projection, degree, alias_kind
):
    model = _model()
    source = getattr(model.layers[0].mlp.experts, projection)
    if alias_kind == "parameter":
        model.register_parameter("external_expert", source)
    elif alias_kind == "buffer_view":
        model.register_buffer("external_expert", source.detach().view(-1))
    else:
        model.register_parameter(
            "external_expert", torch.nn.Parameter(source.detach().view(-1))
        )
    parameters = tuple(model.parameters())
    blocks = tuple(layer.mlp for layer in model.layers)
    rng = torch.get_rng_state().clone()
    with pytest.raises(UnsupportedCombinationError, match="external_expert"):
        model_replacement.resolve_ep_plan(model, degree=degree)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert all(a is b.mlp for a, b in zip(blocks, model.layers, strict=True))


@pytest.mark.parametrize("alias_kind", ["parameter", "buffer_view"])
def test_local_ep_preserves_unchanged_down_projection_alias(alias_kind):
    model = _model()
    source = model.layers[0].mlp.experts.down_proj
    if alias_kind == "parameter":
        model.register_parameter("external_down", source)
    else:
        model.register_buffer("external_down", source.detach().view(-1))
    plan = model_replacement.resolve_ep_plan(model, degree=1)
    model_replacement.swap_moe_blocks(
        model, resolved_plan=plan, router_aux_loss_coef=0.0
    )
    converted = model.layers[0].mlp.routed_experts.inner_experts.w2_EDF
    assert converted is source
    assert (
        model.external_down.untyped_storage()._cdata == source.untyped_storage()._cdata
    )
    with torch.no_grad():
        converted.add_(1)
    torch.testing.assert_close(model.external_down.reshape_as(converted), converted)
