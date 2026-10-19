"""Kimi EP conversion parity against the native per-expert implementation."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from llmtuner.models.common.moe.layer import collect_moe_layers
from llmtuner.models.kimi.kimi_linear_config import KimiLinearConfig
from llmtuner.models.kimi.moe import KimiSparseMoE
from llmtuner.parallel.expert_parallel.model_replacement import swap_hf_moe_blocks


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
    assert swap_hf_moe_blocks(model) == 1
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
        swap_hf_moe_blocks(model)
    assert model.layers[0].block_sparse_moe is source


def test_kimi_native_ep_format_loads_into_a_new_converted_model():
    model, restored = (
        make_model(latent=True, shared=1),
        make_model(latent=True, shared=1),
    )
    swap_hf_moe_blocks(model)
    swap_hf_moe_blocks(restored)
    restored.load_state_dict(model.state_dict(), strict=True)
    inputs = torch.randn(2, 3, 8)
    torch.testing.assert_close(
        model.layers[0].block_sparse_moe(inputs),
        restored.layers[0].block_sparse_moe(inputs),
        atol=0,
        rtol=0,
    )


def test_meta_kimi_conversion_reconstructs_empty_routing_counters():
    from llmtuner.models.hf.materialization import materialize_meta_model

    with torch.device("meta"):
        model = make_model(latent=True, shared=1)
        swap_hf_moe_blocks(model)
    block = model.layers[0].block_sparse_moe
    assert block.tokens_per_expert_E.is_meta
    materialize_meta_model(model, torch.device("cpu"))
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
    swap_hf_moe_blocks(model, on_replacement=references.record_replacements)
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
        swap_hf_moe_blocks(model, on_replacement=references.record_replacements)
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
    swap_hf_moe_blocks(model, on_replacement=references.record_replacements)
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
    source.config.activation_situ_beta = 0.0
    parameters = dict(model.named_parameters())
    with pytest.raises(ValueError, match="beta must be finite and positive"):
        swap_hf_moe_blocks(model)
    assert model.layers[0].block_sparse_moe is source
    assert all(
        dict(model.named_parameters())[key] is value
        for key, value in parameters.items()
    )
