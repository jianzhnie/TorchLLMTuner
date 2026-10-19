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
