"""Construction-selected policies own instances, never external forward methods."""

from copy import deepcopy
from types import MethodType

import pytest
import torch
from torch import nn
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock as ReferenceBlock,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeTopKRouter as ReferenceRouter,
)

from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.models.qwen.routing import Qwen3MoeFP32TopKRouter
from tests.unit_tests.cpu.models.test_qwen3_model_contract import model_config


@pytest.mark.parametrize("policy", ["router", "experts"])
def test_policy_construction_is_local_and_preserves_external_forward(policy):
    config = model_config()
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    kind = ReferenceRouter if policy == "router" else ReferenceBlock
    external = kind(deepcopy(config))

    def external_forward(self, inputs):
        return inputs

    external.forward = MethodType(external_forward, external)
    with torch.no_grad():
        for parameter in external.parameters():
            parameter.fill_(0.125)
    external_parameters = tuple(external.parameters())
    external_state = {
        key: value.clone() for key, value in external.state_dict().items()
    }
    original_class_forward = kind.forward
    torch.manual_seed(53)
    baseline = Qwen3MoeForCausalLM(deepcopy(config))
    expected_rng = torch.get_rng_state().clone()
    if policy == "router":
        config.router_compute_dtype = "float32"
    else:
        config.experts_implementation = "batched_mm"
    torch.manual_seed(53)
    model = Qwen3MoeForCausalLM(config)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert baseline.state_dict().keys() == model.state_dict().keys()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    assert kind.forward is original_class_forward
    assert external.forward.__func__ is external_forward
    assert all(
        a is b for a, b in zip(external_parameters, external.parameters(), strict=True)
    )
    for name, value in external.state_dict().items():
        torch.testing.assert_close(value, external_state[name], rtol=0, atol=0)
    block = model.layers[0].mlp
    if policy == "router":
        assert isinstance(block.gate, Qwen3MoeFP32TopKRouter)
        assert not isinstance(baseline.layers[0].mlp.gate, Qwen3MoeFP32TopKRouter)
    else:
        assert model.config._experts_implementation == "batched_mm"
        assert baseline.config._experts_implementation == "eager"
    identities = {name: id(value) for name, value in model.named_parameters()}
    keys = tuple(model.state_dict())
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        output = model(torch.tensor([[1, 3, 7, 2]])).logits
        assert output.isfinite().all() and output.count_nonzero() > 0
        output.square().sum().backward()
        assert block.gate.weight.grad.count_nonzero() > 0
        assert block.experts.gate_up_proj.grad.count_nonzero() > 0
    assert identities == {name: id(value) for name, value in model.named_parameters()}
    assert tuple(model.state_dict()) == keys
    assert all("forward" not in layer.mlp.__dict__ for layer in model.layers)


@pytest.mark.parametrize(
    "setting,value",
    [("router_compute_dtype", "bfloat16"), ("experts_implementation", "torchtitan")],
)
def test_invalid_policy_rejected_before_construction(setting, value, monkeypatch):
    config = model_config()
    setattr(config, setting, value)
    rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid policy reached parameter allocation")

    monkeypatch.setattr(nn.Embedding, "__init__", forbidden)
    message = (
        "router_compute_dtype"
        if setting == "router_compute_dtype"
        else "model-owned expert implementation"
    )
    with pytest.raises(ValueError, match=message):
        Qwen3MoeForCausalLM(config)
    assert torch.equal(torch.get_rng_state(), rng)
