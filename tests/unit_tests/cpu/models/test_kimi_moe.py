"""Trainable Kimi local experts against all-expert weighted enumeration."""

from copy import deepcopy

import pytest
import torch

from llmtuner.models.kimi.kimi_k3.configuration import KimiLinearConfig
from llmtuner.models.kimi.kimi_k3.moe import KimiMoEGate, KimiSparseMoE


@pytest.mark.parametrize(
    "field,value",
    [
        ("routed_scaling_factor", float("nan")),
        ("routed_scaling_factor", float("inf")),
        ("routed_scaling_factor", True),
        ("moe_renormalize", "false"),
    ],
)
def test_invalid_router_config_fails_without_allocating_weights(
    monkeypatch, field, value
):
    config = KimiLinearConfig(hidden_size=8, num_experts=4, num_experts_per_token=2)
    setattr(config, field, value)
    rng = torch.get_rng_state().clone()

    def reject(*args, **kwargs):
        pytest.fail("Router configuration must be checked before weight allocation")

    monkeypatch.setattr(torch, "empty", reject)
    with pytest.raises(ValueError, match=field):
        KimiMoEGate(config)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("top_k", [1, 2])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("score_func", ["sigmoid", "softmax"])
def test_parallel_router_preserves_live_gate_math(top_k, grouped, score_func):
    torch.manual_seed(41)
    block = KimiSparseMoE(
        KimiLinearConfig(
            hidden_size=8,
            num_experts=4,
            num_experts_per_token=top_k,
            moe_intermediate_size=12,
            num_expert_group=2 if grouped else 1,
            topk_group=1,
            moe_router_activation_func=score_func,
            moe_renormalize=True,
            routed_scaling_factor=1.7,
        )
    )
    gate = block.gate
    with torch.no_grad():
        gate.e_score_correction_bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4]))
    # Conversion consumes the live gate, not subsequently edited config fields.
    block.config.routed_scaling_factor = 9.0
    native = gate.build_parallel_router(aux_loss_coefficient=None)
    assert native.gate.weight is gate.weight
    inputs = torch.randn(2, 7, 8, requires_grad=True)
    other = inputs.detach().clone().requires_grad_()
    expected_ids, expected_scores = gate(inputs)
    scores, ids, _ = native(other.reshape(-1, 8), gate.e_score_correction_bias)
    assert torch.equal(ids, expected_ids)
    assert torch.count_nonzero(scores) == scores.numel()
    torch.testing.assert_close(scores, expected_scores, rtol=1e-6, atol=1e-6)
    gradient = torch.randn_like(scores)
    expected_gradients = torch.autograd.grad(
        expected_scores, (inputs, gate.weight), gradient
    )
    gradients = torch.autograd.grad(scores, (other, native.gate.weight), gradient)
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("degree", [1, 2, 4])
def test_expert_parallel_validation_preserves_live_model_and_rng(degree):
    model = KimiSparseMoE(
        KimiLinearConfig(
            hidden_size=8,
            num_experts=4,
            num_experts_per_token=2,
            moe_intermediate_size=12,
        )
    )
    parameters = dict(model.named_parameters())
    values = {name: value.detach().clone() for name, value in parameters.items()}
    forward = model.forward
    rng = torch.get_rng_state().clone()
    model.validate_expert_parallel(
        degree=degree,
        tp_enabled=False,
        quantile_balancing=False,
        token_dispatcher="alltoall",
    )
    assert model.forward == forward
    assert dict(model.named_parameters()).keys() == parameters.keys()
    for name, value in model.named_parameters():
        assert value is parameters[name]
        torch.testing.assert_close(value, values[name], rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)


def test_bf16_moe_combines_experts_before_rounding():
    torch.manual_seed(19)
    model = KimiSparseMoE(
        KimiLinearConfig(
            hidden_size=8,
            num_experts=4,
            num_experts_per_token=3,
            moe_intermediate_size=12,
        )
    ).bfloat16()
    inputs = torch.randn(1, 31, 8, dtype=torch.bfloat16, requires_grad=True)
    ids, scores = model.gate(inputs)
    selected = inputs.new_zeros(31, 3, 8)
    prematurely_rounded = torch.zeros_like(inputs.reshape(-1, 8))
    for index, expert in enumerate(model.experts):
        tokens, slots = (ids == index).nonzero(as_tuple=True)
        values = expert(inputs.reshape(-1, 8).index_select(0, tokens))
        selected[tokens, slots] = values
        prematurely_rounded = prematurely_rounded.index_add(
            0, tokens, (values.float() * scores[tokens, slots, None]).bfloat16()
        )
    expected = (selected.float() * scores.unsqueeze(-1)).sum(1).bfloat16()
    actual = model(inputs).reshape_as(expected)
    assert torch.count_nonzero(expected != prematurely_rounded) > 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradient = torch.randn_like(actual)
    parameters = (inputs, model.gate.weight, model.experts[0].w1.weight)
    actual_grads = torch.autograd.grad(actual, parameters, gradient, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, parameters, gradient)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


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
    from llmtuner.models.kimi.kimi_k3.moe import KimiMoEGate

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
