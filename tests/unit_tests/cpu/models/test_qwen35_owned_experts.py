"""Qwen3.5 packed experts own score placement, layout and eager execution."""

from copy import deepcopy

import pytest
import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeExperts as ReferenceExperts,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM as ReferenceModel,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeMLP as ReferenceMLP,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoePreTrainedModel as ReferencePreTrainedModel,
)

from llmtuner.models.qwen.qwen3_5_moe.model import (
    Qwen3_5MoeExperts,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeMLP,
    Qwen3_5MoeSparseMoeBlock,
)
from tests.unit_tests.cpu.models.test_owned_family_construction import text_config


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("duplicate_routes", [False, True])
def test_owned_experts_match_reference_outputs_all_gradients_and_updates(
    dtype, duplicate_routes, monkeypatch
):
    config = text_config(Qwen3_5MoeForCausalLM.config_class)
    config.num_experts = 4
    config._experts_implementation = "eager"
    torch.manual_seed(83)
    model = Qwen3_5MoeForCausalLM(config).to(dtype)
    owned = model.layers[0].mlp.experts
    assert type(owned) is Qwen3_5MoeExperts
    reference = ReferenceExperts(deepcopy(config)).to(dtype)
    reference.load_state_dict(owned.state_dict(), strict=True)
    inputs = torch.randn(3, config.hidden_size, dtype=dtype, requires_grad=True)
    other = inputs.detach().clone().requires_grad_()
    routes = torch.tensor([[0, 1], [2, 0], [1, 2]])
    if duplicate_routes:
        routes[0] = 0
    scores = torch.rand(3, 2, requires_grad=True)
    other_scores = scores.detach().clone().requires_grad_()
    expected = reference(other, routes, other_scores)

    def forbidden(*args, **kwargs):
        raise AssertionError("Owned experts delegated to reference forward")

    monkeypatch.setattr(ReferenceExperts, "forward", forbidden)
    actual = owned(inputs, routes, scores)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    assert actual.isfinite().all() and actual.count_nonzero() > 0
    for left, right in (
        (actual, expected),
        (inputs.grad, other.grad),
        (scores.grad, other_scores.grad),
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert left.isfinite().all() and left.count_nonzero() > 0
    parameters = dict(reference.named_parameters())
    for name, parameter in owned.named_parameters():
        torch.testing.assert_close(
            parameter.grad, parameters[name].grad, rtol=0, atol=0
        )
        assert parameter.grad[3].count_nonzero() == 0
    before = owned.gate_up_proj.detach().clone()
    # Keep epsilon representable in FP16 CPU optimizer state; this is a
    # model update oracle, not a test of mixed-precision optimizer storage.
    for experts in (owned, reference):
        torch.optim.AdamW(experts.parameters(), lr=1e-3, eps=1e-4, foreach=False).step()
    assert not torch.equal(owned.gate_up_proj, before)
    for name, parameter in owned.named_parameters():
        assert parameter.isfinite().all(), name
        torch.testing.assert_close(parameter, parameters[name], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_shared_expert_owns_math_without_changing_initialization(dtype, monkeypatch):
    config = text_config(Qwen3_5MoeForCausalLM.config_class)
    config._experts_implementation = "eager"
    torch.manual_seed(97)
    owned_model = Qwen3_5MoeForCausalLM(deepcopy(config)).to(dtype)
    owned = owned_model.layers[0].mlp.shared_expert
    rng = torch.get_rng_state().clone()
    torch.manual_seed(97)
    reference_model = ReferenceModel(deepcopy(config)).to(dtype)
    reference = reference_model.model.layers[0].mlp.shared_expert
    assert torch.equal(torch.get_rng_state(), rng)
    assert type(owned) is Qwen3_5MoeMLP
    assert owned.state_dict().keys() == reference.state_dict().keys()
    for name, value in owned.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)

    def forbidden(*args, **kwargs):
        raise AssertionError("Shared expert delegated to reference forward")

    hidden = torch.randn(2, 3, config.hidden_size, dtype=dtype).transpose(0, 1)
    hidden = hidden.requires_grad_()
    other = hidden.detach().clone().requires_grad_()
    expected = reference(other)
    monkeypatch.setattr(ReferenceMLP, "forward", forbidden)
    actual = owned(hidden)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(hidden.grad, other.grad, rtol=0, atol=0)
    assert hidden.grad.isfinite().all() and hidden.grad.count_nonzero() > 0
    for name, parameter in owned.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            dict(reference.named_parameters())[name].grad,
            rtol=0,
            atol=0,
        )


def test_owned_moe_initialization_does_not_delegate_family_rules(monkeypatch):
    config = text_config(Qwen3_5MoeForCausalLM.config_class)
    torch.manual_seed(103)
    reference = ReferenceModel(deepcopy(config))
    expected_rng = torch.get_rng_state().clone()
    original = ReferencePreTrainedModel._init_weights

    def guarded(self, module):
        if isinstance(module, (Qwen3_5MoeExperts, Qwen3_5MoeSparseMoeBlock)):
            raise AssertionError("Owned MoE initialization delegated family rules")
        return original(self, module)

    monkeypatch.setattr(ReferencePreTrainedModel, "_init_weights", guarded)
    torch.manual_seed(103)
    owned = Qwen3_5MoeForCausalLM(deepcopy(config))
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert owned.state_dict().keys() == reference.state_dict().keys()
    for name, value in owned.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("loaded", range(8))
def test_partial_loaded_moe_parameters_keep_values_identity_and_rng(dtype, loaded):
    config = text_config(Qwen3_5MoeForCausalLM.config_class)
    model = Qwen3_5MoeForCausalLM(config).to(dtype)
    block = model.layers[0].mlp
    parameters = (
        block.experts.gate_up_proj,
        block.experts.down_proj,
        block.gate.weight,
    )
    identities = tuple(id(parameter) for parameter in parameters)
    expected = []
    for index, parameter in enumerate(parameters):
        with torch.no_grad():
            parameter.fill_(index + 0.25)
        parameter._is_hf_initialized = bool(loaded & (1 << index))
        expected.append(parameter.detach().clone())
    torch.manual_seed(107)
    for index, tensor in enumerate(expected):
        if not loaded & (1 << index):
            torch.nn.init.normal_(tensor, mean=0.0, std=config.initializer_range)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(107)
    model._init_weights(block.experts)
    model._init_weights(block)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    current = (block.experts.gate_up_proj, block.experts.down_proj, block.gate.weight)
    assert identities == tuple(id(parameter) for parameter in current)
    for index, (parameter, value) in enumerate(zip(current, expected, strict=True)):
        torch.testing.assert_close(parameter, value, rtol=0, atol=0)
        assert parameter.isfinite().all() and parameter.count_nonzero() > 0
        if loaded & (1 << index):
            assert torch.equal(parameter, torch.full_like(parameter, index + 0.25))
