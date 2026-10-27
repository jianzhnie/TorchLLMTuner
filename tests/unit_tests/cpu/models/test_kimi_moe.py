"""Trainable Kimi local experts against all-expert weighted enumeration."""

from copy import deepcopy

import pytest
import torch

from llmtuner.models.kimi.kimi_linear_config import KimiLinearConfig
from llmtuner.models.kimi.moe import KimiSparseMoE


@pytest.mark.parametrize(
    "shared,latent,grouped", [(0, False, False), (1, False, True), (1, True, True)]
)
def test_kimi_sparse_training_matches_dense_expert_enumeration(shared, latent, grouped):
    torch.manual_seed(4)
    config = KimiLinearConfig(
        hidden_size=8,
        num_experts=4,
        num_experts_per_token=2,
        moe_intermediate_size=12,
        num_shared_experts=shared,
        num_expert_group=2 if grouped else 1,
        topk_group=1,
        routed_expert_hidden_size=4 if latent else None,
        latent_moe_use_norm=latent,
    )
    model = KimiSparseMoE(config)
    reference = deepcopy(model)
    inputs = torch.randn(1, 5, 8, requires_grad=True)
    other = inputs.detach().clone().requires_grad_()
    logits = torch.nn.functional.linear(
        other.reshape(-1, 8).float(), reference.gate.weight.float()
    )
    scores = logits.sigmoid()
    choices = scores + reference.gate.e_score_correction_bias
    if grouped:
        # Independent group enumeration: exactly one two-expert group is retained.
        groups = choices.reshape(-1, 2, 2).sum(-1).argmax(-1)
        retained = torch.arange(4)[None, :] // 2 == groups[:, None]
        choices = choices.masked_fill(~retained, float("-inf"))
    ids = choices.topk(2, dim=-1, sorted=False).indices
    weights = scores.gather(1, ids)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    dense_weights = torch.zeros_like(scores).scatter(1, ids, weights)
    expert_inputs = other.reshape(-1, 8)
    if latent:
        expert_inputs = reference.routed_expert_down_proj(expert_inputs)
    expected = sum(
        expert(expert_inputs) * dense_weights[:, i, None]
        for i, expert in enumerate(reference.experts)
    )
    if latent:
        expected = reference.routed_expert_norm(expected)
        expected = reference.routed_expert_up_proj(expected)
    expected = expected.reshape_as(other)
    if shared:
        expected = expected + reference.shared_experts(other)
    actual = model(inputs)
    torch.testing.assert_close(actual, expected)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(inputs.grad, other.grad)
    for (name, param), (_, ref) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        if ref.grad is None:
            assert param.grad is None
        else:
            torch.testing.assert_close(param.grad, ref.grad, msg=name)


@pytest.mark.parametrize(
    "experts,groups,retained,topk", [(5, 2, 1, 2), (4, 4, 1, 1), (4, 2, 1, 3)]
)
def test_kimi_group_routing_rejects_impossible_selection(
    experts, groups, retained, topk
):
    config = KimiLinearConfig(
        hidden_size=8,
        num_experts=experts,
        num_experts_per_token=topk,
        moe_intermediate_size=12,
        num_expert_group=groups,
        topk_group=retained,
    )
    with pytest.raises(ValueError, match="groups"):
        KimiSparseMoE(config)


@pytest.mark.parametrize("empty", [False, True])
def test_kimi_router_accepts_noncontiguous_and_empty_batches(empty):
    config = KimiLinearConfig(
        hidden_size=8,
        num_experts=4,
        num_experts_per_token=2,
        moe_intermediate_size=12,
        num_expert_group=2,
        topk_group=1,
    )
    model = KimiSparseMoE(config)
    base = torch.randn(3, 2, 8, requires_grad=True)
    inputs = base.transpose(0, 1)
    if empty:
        inputs = inputs[:, :0]
    else:
        assert not inputs.is_contiguous()
    actual = model(inputs)
    expected = model(inputs.contiguous())
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(base.grad).all()
    assert all(p.grad is not None for p in model.experts.parameters())


@pytest.mark.parametrize("width", [0, -1, True, 1.5])
def test_kimi_rejects_invalid_latent_expert_width(width):
    with pytest.raises(ValueError, match="routed_expert_hidden_size"):
        KimiLinearConfig(routed_expert_hidden_size=width)


@pytest.mark.parametrize(
    "field", ["num_experts", "num_experts_per_token", "num_expert_group", "topk_group"]
)
@pytest.mark.parametrize("value", [True, 1.0, 0, -1])
def test_kimi_router_rejects_invalid_integer_configuration(field, value):
    from llmtuner.models.kimi.moe import KimiMoEGate

    settings = dict(
        hidden_size=8,
        num_experts=4,
        num_experts_per_token=1,
        num_expert_group=1,
        topk_group=1,
    )
    settings[field] = value
    config = KimiLinearConfig(**settings)
    with pytest.raises(ValueError, match=field):
        KimiMoEGate(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("moe_layer_freq", 0),
        ("moe_layer_freq", -1),
        ("moe_layer_freq", True),
        ("moe_layer_freq", 1.5),
        ("first_k_dense_replace", -1),
        ("first_k_dense_replace", True),
        ("first_k_dense_replace", 1.5),
    ],
)
def test_kimi_rejects_invalid_moe_layer_selection(field, value):
    with pytest.raises(ValueError, match=field):
        KimiLinearConfig(**{field: value})
