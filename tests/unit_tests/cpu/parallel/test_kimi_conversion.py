"""Kimi EP conversion parity against the native per-expert implementation."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.common.moe.layer import collect_moe_layers
from llmtuner.models.kimi.kimi_k3.configuration import KimiLinearConfig
from llmtuner.models.kimi.kimi_k3.moe import KimiSparseMoE
from llmtuner.parallel.expert_parallel.model_replacement import (
    resolve_ep_plan,
    swap_moe_blocks,
)


def make_model(*, latent=False, shared=0, activation="silu", top_k=2):
    config = KimiLinearConfig(
        hidden_size=8,
        num_experts=4,
        num_experts_per_token=top_k,
        moe_intermediate_size=12,
        num_shared_experts=shared,
        num_expert_group=2,
        topk_group=1,
        routed_expert_hidden_size=4 if latent else None,
        latent_moe_use_norm=latent,
        hidden_act=activation,
    )
    model = nn.Module()
    layer = nn.Module()
    layer.block_sparse_moe = KimiSparseMoE(config)
    model.layers = nn.ModuleList([layer])
    return model


@pytest.mark.parametrize("change", ["tp", "backend", "quantile", "dtype", "grad"])
def test_kimi_plan_rejects_unsupported_conversion_without_allocation(
    monkeypatch, change
):
    model = make_model()
    block = model.layers[0].block_sparse_moe
    if change == "dtype":
        block.experts[1].w2.double()
    if change == "grad":
        block.experts[1].w2.weight.requires_grad_(False)
    parameters = tuple(model.parameters())
    rng = torch.get_rng_state().clone()

    def reject(*args, **kwargs):
        raise AssertionError("EP parsing must not allocate tensors")

    monkeypatch.setattr(torch, "empty", reject)
    with pytest.raises(UnsupportedCombinationError, match="Kimi EP"):
        resolve_ep_plan(
            model,
            tp_enabled=change == "tp",
            implementation="mc2" if change == "backend" else "alltoall",
            quantile_balancing=change == "quantile",
        )
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("change", ["shape", "bias", "count"])
def test_kimi_plan_rejects_lossy_expert_conversion(monkeypatch, change):
    model = make_model()
    block = model.layers[0].block_sparse_moe
    if change == "shape":
        block.experts[1].w2.weight = nn.Parameter(torch.randn(8, 11))
    elif change == "bias":
        block.experts[1].w3.bias = nn.Parameter(torch.ones(12))
    else:
        del block.experts[-1]
    rng = torch.get_rng_state().clone()

    def reject(*args, **kwargs):
        raise AssertionError("Invalid expert geometry must fail before allocation")

    monkeypatch.setattr(torch, "empty", reject)
    with pytest.raises((ValueError, UnsupportedCombinationError), match="Kimi EP"):
        resolve_ep_plan(model)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("change", ["down", "up", "norm", "missing_up", "input"])
def test_kimi_plan_rejects_inconsistent_latent_geometry(monkeypatch, change):
    model = make_model(latent=True)
    block = model.layers[0].block_sparse_moe
    if change in ("down", "up"):
        projection = getattr(block, f"routed_expert_{change}_proj")
        projection.weight = nn.Parameter(projection.weight[:, :-1].clone())
    elif change == "norm":
        block.routed_expert_norm.weight = nn.Parameter(torch.ones(3))
    elif change == "missing_up":
        del block.routed_expert_up_proj
    else:
        del block.routed_expert_down_proj
        del block.routed_expert_up_proj
    parameters = tuple(model.parameters())
    rng = torch.get_rng_state().clone()

    def reject(*args, **kwargs):
        raise AssertionError("Latent geometry must fail before allocation")

    monkeypatch.setattr(torch, "empty", reject)
    with pytest.raises(ValueError, match="Kimi EP"):
        resolve_ep_plan(model)
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("external", [False, True])
def test_kimi_plan_rejects_expert_aliases_before_allocation(monkeypatch, external):
    model = make_model()
    block = model.layers[0].block_sparse_moe
    if external:
        model.register_parameter("expert_alias", block.experts[0].w1.weight)
    else:
        block.experts[1].w1.weight = block.experts[0].w1.weight
    rng = torch.get_rng_state().clone()

    def reject(*args, **kwargs):
        raise AssertionError("Alias preflight must not allocate replacements")

    monkeypatch.setattr(torch, "empty", reject)
    with pytest.raises(UnsupportedCombinationError, match="shared individual expert"):
        resolve_ep_plan(model)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("activation", ["silu", "situ"])
def test_kimi_conversion_preserves_live_activation_after_config_edit(activation):
    model = make_model(activation=activation)
    baseline = deepcopy(model)
    assert swap_moe_blocks(baseline) == 1
    block = model.layers[0].block_sparse_moe
    block.config.hidden_act = "situ" if activation == "silu" else "silu"
    block.config.activation_situ_beta = 7.0
    block.config.activation_situ_linear_beta = 2.0
    values = torch.randn(2, 5, 8)
    expected = baseline.layers[0].block_sparse_moe(values)
    assert swap_moe_blocks(model) == 1
    actual = model.layers[0].block_sparse_moe(values)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_kimi_plan_rejects_nonuniform_live_activation():
    model = make_model(activation="situ")
    model.layers[0].block_sparse_moe.experts[1].act_fn.beta = 7.0
    with pytest.raises(UnsupportedCombinationError, match="uniform expert activations"):
        resolve_ep_plan(model)


@pytest.mark.parametrize(
    "field,value",
    [
        ("routed_scaling_factor", float("nan")),
        ("routed_scaling_factor", float("inf")),
        ("routed_scaling_factor", True),
        ("routed_scaling_factor", "1.0"),
        ("moe_renormalize", 1),
        ("moe_renormalize", "false"),
    ],
)
def test_kimi_invalid_live_routing_fails_before_replacement(monkeypatch, field, value):
    model = make_model()
    block = model.layers[0].block_sparse_moe
    setattr(block.gate, field, value)
    parameters = tuple(model.parameters())
    rng = torch.get_rng_state().clone()

    def reject(*args, **kwargs):
        pytest.fail("Invalid routing must fail before replacement allocation")

    monkeypatch.setattr(torch, "empty", reject)
    with pytest.raises(ValueError, match=field):
        swap_moe_blocks(model)
    assert model.layers[0].block_sparse_moe is block
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(torch.get_rng_state(), rng)


def test_kimi_conversion_consumes_instantiated_router_math():
    torch.manual_seed(19)
    model = make_model()
    baseline = deepcopy(model)
    assert swap_moe_blocks(baseline) == 1
    block = model.layers[0].block_sparse_moe
    # Editing config does not edit the live native gate. Conversion must not
    # reinterpret that edit as a request to change routing mathematics.
    block.config.moe_router_activation_func = "softmax"
    block.config.routed_scaling_factor = 3.0
    block.config.moe_renormalize = not block.gate.moe_renormalize
    block.config.num_expert_group = 1
    block.config.topk_group = 1
    values = torch.randn(2, 5, 8, requires_grad=True)
    expected = baseline.layers[0].block_sparse_moe(values)
    expected_gradient = torch.autograd.grad(expected.square().sum(), values)[0]
    assert swap_moe_blocks(model) == 1
    actual_values = values.detach().clone().requires_grad_()
    actual = model.layers[0].block_sparse_moe(actual_values)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    observed_gradient = torch.autograd.grad(actual.square().sum(), actual_values)[0]
    torch.testing.assert_close(observed_gradient, expected_gradient)
    assert observed_gradient.count_nonzero()


@pytest.mark.parametrize("change", ["scale", "top_k", "activation", "latent_norm"])
def test_kimi_execution_rejects_changed_math_after_resolution(monkeypatch, change):
    model = make_model(activation="situ", latent=True)
    plan = resolve_ep_plan(model)
    block = model.layers[0].block_sparse_moe
    if change == "scale":
        block.gate.routed_scaling_factor = 3.0
    elif change == "top_k":
        block.gate.top_k = 1
    elif change == "activation":
        for expert in block.experts:
            expert.act_fn.beta = 7.0
    else:
        block.routed_expert_norm.variance_epsilon = 0.5

    def reject(*args, **kwargs):
        raise AssertionError("Stale math must fail before conversion allocation")

    monkeypatch.setattr(torch, "empty", reject)
    with pytest.raises(ValueError, match="Stale EP Kimi mathematics"):
        swap_moe_blocks(model, resolved_plan=plan)
    assert model.layers[0].block_sparse_moe is block


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("top_k", 5),
        ("num_expert_group", 3),
        ("topk_group", 0),
        ("moe_router_activation_func", "unknown"),
    ],
)
def test_kimi_plan_validates_live_router_geometry_before_allocation(
    monkeypatch, name, value
):
    model = make_model()
    setattr(model.layers[0].block_sparse_moe.gate, name, value)
    rng = torch.get_rng_state().clone()

    def reject(*args, **kwargs):
        raise AssertionError("Invalid router geometry must fail before allocation")

    monkeypatch.setattr(torch, "empty", reject)
    with pytest.raises(ValueError, match="Kimi"):
        resolve_ep_plan(model)
    assert torch.equal(torch.get_rng_state(), rng)


def test_kimi_revalidates_all_planned_targets_before_first_conversion(monkeypatch):
    model = make_model()
    model.layers.append(deepcopy(model.layers[0]))
    plan = resolve_ep_plan(model)
    originals = tuple(layer.block_sparse_moe for layer in model.layers)
    # Config mutation is not a tensor/module mutation and leaves the live gate
    # snapshot unchanged, but still invalidates the conversion's count contract.
    originals[1].config.num_experts = 8
    rng = torch.get_rng_state().clone()

    def reject(*args, **kwargs):
        raise AssertionError("All targets must be validated before any allocation")

    monkeypatch.setattr(torch, "empty", reject)
    with pytest.raises(ValueError, match="router and expert count must agree"):
        swap_moe_blocks(model, resolved_plan=plan)
    assert all(
        layer.block_sparse_moe is original
        for layer, original in zip(model.layers, originals, strict=True)
    )
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize(
    "alias", ["expert", "external_parameter", "buffer", "same_tensor_buffer"]
)
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_kimi_plan_rejects_distinct_tensor_storage_aliases(monkeypatch, alias, device):
    with torch.device(device):
        model = make_model()
    block = model.layers[0].block_sparse_moe
    weight = block.experts[0].w1.weight
    if alias == "expert":
        block.experts[1].w1.weight = nn.Parameter(weight.detach())
    elif alias == "external_parameter":
        model.register_parameter("storage_alias", nn.Parameter(weight.detach()))
    elif alias == "buffer":
        model.register_buffer("storage_alias", weight.detach())
    else:
        model.register_buffer("storage_alias", weight)
    rng = torch.get_rng_state().clone()

    def reject(*args, **kwargs):
        raise AssertionError("Storage aliases must fail before conversion allocation")

    monkeypatch.setattr(torch, "empty", reject)
    with pytest.raises(UnsupportedCombinationError, match="storage"):
        resolve_ep_plan(model)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("change", ["rank", "width", "bias", "count"])
def test_kimi_plan_validates_router_storage_before_allocation(monkeypatch, change):
    model = make_model()
    gate = model.layers[0].block_sparse_moe.gate
    if change == "rank":
        gate.weight = nn.Parameter(torch.randn(4))
    elif change == "width":
        gate.weight = nn.Parameter(torch.randn(4, 7))
    elif change == "bias":
        gate.e_score_correction_bias = nn.Parameter(torch.zeros(1))
    else:
        gate.num_experts = 8
    rng = torch.get_rng_state().clone()

    def reject(*args, **kwargs):
        raise AssertionError("Invalid router storage must fail before allocation")

    monkeypatch.setattr(torch, "empty", reject)
    with pytest.raises(ValueError, match="Kimi router"):
        resolve_ep_plan(model)
    assert torch.equal(torch.get_rng_state(), rng)


def test_kimi_conversion_preserves_projection_dtypes():
    model = make_model().to(dtype=torch.bfloat16)
    source = model.layers[0].block_sparse_moe
    for expert in source.experts:
        expert.w2.float()
    expected = torch.stack([expert.w2.weight.detach() for expert in source.experts])
    assert swap_moe_blocks(model) == 1
    grouped = model.layers[0].block_sparse_moe.routed_experts.inner_experts
    assert grouped.w1_EFD.dtype == grouped.w3_EFD.dtype == torch.bfloat16
    assert grouped.w2_EDF.dtype == torch.float32
    torch.testing.assert_close(grouped.w2_EDF, expected, rtol=0, atol=0)


def test_bf16_conversion_preserves_fp32_combine():
    torch.manual_seed(19)
    model = make_model(top_k=2).bfloat16()
    source = deepcopy(model.layers[0].block_sparse_moe)
    inputs = torch.randn(1, 31, 8, dtype=torch.bfloat16, requires_grad=True)
    other = inputs.detach().clone().requires_grad_()
    assert swap_moe_blocks(model) == 1
    converted = model.layers[0].block_sparse_moe
    # Width 12 is intentionally below grouped-MM's BF16 alignment contract;
    # this regression isolates combine rather than the grouped kernel.
    converted.routed_experts.inner_experts.use_grouped_mm = False
    assert converted.routed_experts.token_dispatcher.accumulate_in_fp32
    actual, expected = converted(inputs), source(other)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(inputs.grad, other.grad)
    torch.testing.assert_close(
        converted.router.gate.weight.grad, source.gate.weight.grad
    )


def test_kimi_gate_preserves_fp32_routing_under_autocast():
    torch.manual_seed(17)
    gate = make_model().layers[0].block_sparse_moe.gate
    inputs = torch.randn(2, 8, 8, dtype=torch.bfloat16)
    expected_ids, expected_scores = gate(inputs)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual_ids, actual_scores = gate(inputs)
    assert actual_scores.dtype == torch.float32
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual_scores, expected_scores, rtol=0, atol=0)


def test_kimi_conversion_rejects_different_dtypes_within_stacked_projection():
    model = make_model()
    source = model.layers[0].block_sparse_moe
    source.experts[1].w2.to(dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="different dtypes"):
        swap_moe_blocks(model)
    assert model.layers[0].block_sparse_moe is source


@pytest.mark.parametrize("latent,shared", [(False, 0), (False, 1), (True, 1)])
@pytest.mark.parametrize("activation", ["silu", "situ"])
@pytest.mark.parametrize("top_k", [1, 2])
def test_kimi_conversion_preserves_outputs_gradients_and_parameter_owners(
    latent, shared, activation, top_k
):
    torch.manual_seed(3)
    model = make_model(
        latent=latent, shared=shared, activation=activation, top_k=top_k
    ).eval()
    source = model.layers[0].block_sparse_moe
    source.gate.e_score_correction_bias.data.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4]))
    reference = deepcopy(source)
    assert swap_moe_blocks(model) == 1
    converted = model.layers[0].block_sparse_moe
    assert collect_moe_layers(model) == [converted]
    assert converted.router.gate.weight is source.gate.weight
    assert converted.expert_bias_E is source.gate.e_score_correction_bias
    assert converted.shared_experts is source.shared_experts
    assert converted.load_balance_coeff is None
    assert not converted.training
    if latent:
        assert converted.routed_experts.routed_expert_norm is source.routed_expert_norm
    inputs = torch.randn(2, 5, 8, requires_grad=True)
    other = inputs.detach().clone().requires_grad_()
    expected, actual = reference(other), converted(inputs)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(inputs.grad, other.grad, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        converted.router.gate.weight.grad, reference.gate.weight.grad
    )
    assert converted.expert_bias_E.grad is None
    for native, stacked in (("w1", "w1_EFD"), ("w2", "w2_EDF"), ("w3", "w3_EFD")):
        expected_grad = torch.stack(
            [getattr(expert, native).weight.grad for expert in reference.experts]
        )
        torch.testing.assert_close(
            getattr(converted.routed_experts.inner_experts, stacked).grad,
            expected_grad,
            atol=2e-6,
            rtol=2e-6,
        )
    for path in (
        "shared_experts",
        "routed_expert_down_proj",
        "routed_expert_norm",
        "routed_expert_up_proj",
    ):
        original = getattr(reference, path, None)
        target = getattr(
            converted if path == "shared_experts" else converted.routed_experts,
            path,
            None,
        )
        if original is not None:
            for left, right in zip(
                target.parameters(), original.parameters(), strict=True
            ):
                torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("module_alias", [False, True])
def test_kimi_conversion_rejects_expert_aliases_without_modifying_model(module_alias):
    model = make_model()
    source = model.layers[0].block_sparse_moe
    if module_alias:
        source.experts[1] = source.experts[0]
    else:
        source.experts[1].w1.weight = source.experts[0].w1.weight
    with pytest.raises(NotImplementedError, match="shared individual expert"):
        swap_moe_blocks(model)
    assert model.layers[0].block_sparse_moe is source


def test_kimi_native_ep_format_loads_into_a_new_converted_model():
    model, restored = (
        make_model(latent=True, shared=1),
        make_model(latent=True, shared=1),
    )
    swap_moe_blocks(model)
    swap_moe_blocks(restored)
    restored.load_state_dict(model.state_dict(), strict=True)
    inputs = torch.randn(2, 3, 8)
    torch.testing.assert_close(
        model.layers[0].block_sparse_moe(inputs),
        restored.layers[0].block_sparse_moe(inputs),
        atol=0,
        rtol=0,
    )


def test_meta_kimi_conversion_reconstructs_empty_routing_counters():
    from llmtuner.memory.materialization import materialize_meta_model

    with torch.device("meta"):
        model = make_model(latent=True, shared=1)
        swap_moe_blocks(model)
    block = model.layers[0].block_sparse_moe
    assert block.tokens_per_expert_E.is_meta
    materialize_meta_model(
        model,
        torch.device("cpu"),
        runtime_buffers={
            (block, "tokens_per_expert_E"): torch.zeros(
                4, dtype=block.tokens_per_expert_E.dtype
            )
        },
    )
    assert block.tokens_per_expert_E.dtype == torch.float32
    assert torch.equal(block.tokens_per_expert_E, torch.zeros(4))


@pytest.mark.parametrize("whole_module", [False, True])
def test_kimi_fsdp_references_preserve_complete_stacked_exclusions(whole_module):
    from llmtuner.config.parallel_config import FSDPConfig, ParallelConfig
    from llmtuner.parallel.fully_shard_parallel.plan import (
        FSDPPatternReferences,
        _resolve_pattern_units,
    )

    model = make_model()
    source = model.layers[0].block_sparse_moe
    originals = {expert.w1.weight for expert in source.experts}
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_apply_modules={
                "": {} if whole_module else {"ignored_params": originals}
            },
            fsdp_ignored_modules=(
                ["layers.0.block_sparse_moe.experts"] if whole_module else []
            ),
        )
    )
    references = FSDPPatternReferences(model, cfg)
    swap_moe_blocks(model, on_replacement=references.record_replacements)
    grouped = model.layers[0].block_sparse_moe.routed_experts.inner_experts
    rebound = references.bind(model)
    if whole_module:
        _, _, ignored = _resolve_pattern_units(model, rebound)
        assert ignored == set(grouped.parameters())
    else:
        assert rebound.fsdp_config.fsdp_apply_modules[""]["ignored_params"] == {
            grouped.w1_EFD
        }
        assert cfg.fsdp_config.fsdp_apply_modules[""]["ignored_params"] == originals


@pytest.mark.parametrize("module_exclusion", [False, True])
def test_kimi_partial_stacked_exclusion_rejected_before_installation(module_exclusion):
    from llmtuner.config.parallel_config import FSDPConfig, ParallelConfig
    from llmtuner.errors import UnsupportedCombinationError
    from llmtuner.parallel.fully_shard_parallel.plan import FSDPPatternReferences

    model = make_model()
    source = model.layers[0].block_sparse_moe
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_apply_modules={
                "": {}
                if module_exclusion
                else {"ignored_params": {source.experts[0].w1.weight}}
            },
            fsdp_ignored_modules=(
                ["layers.0.block_sparse_moe.experts.0"] if module_exclusion else []
            ),
        )
    )
    references = FSDPPatternReferences(model, cfg)
    with pytest.raises(UnsupportedCombinationError, match="part of a stacked"):
        swap_moe_blocks(model, on_replacement=references.record_replacements)
    assert model.layers[0].block_sparse_moe is source


@pytest.mark.parametrize("explicit_root", [False, True])
@pytest.mark.parametrize("frozen_bias", [False, True])
def test_kimi_gate_module_exclusion_keeps_moved_bias_outside_router(
    explicit_root, frozen_bias
):
    from llmtuner.config.parallel_config import FSDPConfig, ParallelConfig
    from llmtuner.parallel.fully_shard_parallel.plan import (
        FSDPPatternReferences,
        _resolve_pattern_units,
    )

    model = make_model()
    source = model.layers[0].block_sparse_moe
    source.gate.e_score_correction_bias.requires_grad_(not frozen_bias)
    gate_parameters = set(source.gate.parameters())
    units = {"layers.0.block_sparse_moe.experts": {}}
    if explicit_root:
        units[""] = {"reshard_after_forward": False}
    cfg = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_apply_modules=units,
            fsdp_ignored_modules=["layers.0.block_sparse_moe.gate"],
        )
    )
    references = FSDPPatternReferences(model, cfg)
    swap_moe_blocks(model, on_replacement=references.record_replacements)
    converted = model.layers[0].block_sparse_moe
    rebound = references.bind(model)
    _, units, ignored = _resolve_pattern_units(model, rebound)
    assert ignored == {converted.router.gate.weight}
    assert units[""]["ignored_params"] == {converted.expert_bias_E}
    assert converted.expert_bias_E.requires_grad is not frozen_bias
    for options in units.values():
        assert ignored | options["ignored_params"] == gate_parameters
    assert cfg.fsdp_config.fsdp_ignored_modules == ["layers.0.block_sparse_moe.gate"]
    if explicit_root:
        assert cfg.fsdp_config.fsdp_apply_modules[""] == {
            "reshard_after_forward": False
        }
        assert (
            rebound.fsdp_config.fsdp_apply_modules[""]["reshard_after_forward"] is False
        )
    else:
        assert "" not in cfg.fsdp_config.fsdp_apply_modules


def test_kimi_conversion_rejects_zero_situ_beta_before_installation():
    model = make_model(activation="situ")
    source = model.layers[0].block_sparse_moe
    for expert in source.experts:
        expert.act_fn.beta = 0.0
    parameters = dict(model.named_parameters())
    with pytest.raises(ValueError, match="beta must be finite and positive"):
        swap_moe_blocks(model)
    assert model.layers[0].block_sparse_moe is source
    assert all(
        dict(model.named_parameters())[key] is value
        for key, value in parameters.items()
    )


@pytest.mark.parametrize("start,count", [(0, 2), (2, 2), (0, 4)])
def test_model_owned_expert_packing_preserves_slice_and_projection_state(start, count):
    block = make_model().layers[0].block_sparse_moe
    for expert in block.experts:
        expert.w2.double()
        expert.w3.requires_grad_(False)
    source_parameters = tuple(block.parameters())
    packed = block.build_parallel_experts(start=start, count=count)
    for source_name, target_name in (
        ("w1", "w1_EFD"),
        ("w2", "w2_EDF"),
        ("w3", "w3_EFD"),
    ):
        sources = [
            getattr(expert, source_name).weight
            for expert in block.experts[start : start + count]
        ]
        target = getattr(packed, target_name)
        assert target.dtype == sources[0].dtype
        assert target.requires_grad == sources[0].requires_grad
        torch.testing.assert_close(target, torch.stack(sources), atol=0, rtol=0)
    assert all(
        before is after
        for before, after in zip(source_parameters, block.parameters(), strict=True)
    )


@pytest.mark.parametrize(
    "start,count", [(-1, 1), (0, 0), (3, 2), (True, 1), (0, False)]
)
def test_model_owned_expert_packing_rejects_invalid_ownership_without_rng(start, count):
    block = make_model().layers[0].block_sparse_moe
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="expert ownership"):
        block.build_parallel_experts(start=start, count=count)
    assert torch.equal(torch.get_rng_state(), rng)
