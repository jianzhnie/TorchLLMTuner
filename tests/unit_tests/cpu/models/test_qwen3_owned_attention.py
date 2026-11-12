"""Native Qwen3 attention owns its execution without reference forward calls."""

from copy import deepcopy

import pytest
import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention as ReferenceAttention,
)

from llmtuner.models.qwen.qwen3.model import (
    Qwen3MoeAttention,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
)
from tests.unit_tests.cpu.models.test_qwen3_model_contract import model_config


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("bias", [False, True])
def test_native_attention_is_exact_and_preserves_call_order(backend, bias, monkeypatch):
    config = model_config(attention_bias=bias)
    config._attn_implementation = backend
    torch.manual_seed(101)
    baseline = ReferenceAttention(config, 0)
    actual = Qwen3MoeAttention(deepcopy(config), 0)
    actual.load_state_dict(baseline.state_dict())
    hidden = torch.randn(2, 3, 16, requires_grad=True)
    other = hidden.detach().clone().requires_grad_()
    angles = torch.randn(2, 3, config.head_dim)
    embeddings = (angles.cos(), angles.sin())
    mask = torch.zeros(2, 1, 3, 3)
    expected = baseline(hidden, embeddings, mask)[0]

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit attention must not inherit forward execution")

    monkeypatch.setattr(ReferenceAttention, "forward", forbidden)
    order = []
    hooks = [
        actual.get_submodule(name).register_forward_hook(
            lambda module, args, result, name=name: order.append(name)
        )
        for name in ("q_proj", "q_norm", "k_proj", "k_norm", "v_proj", "o_proj")
    ]
    try:
        result = actual(other, embeddings, mask)[0]
    finally:
        for hook in hooks:
            hook.remove()
    assert order == ["q_proj", "q_norm", "k_proj", "k_norm", "v_proj", "o_proj"]
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    probe = torch.randn_like(result)
    result.backward(probe)
    expected.backward(probe)
    torch.testing.assert_close(other.grad, hidden.grad, rtol=0, atol=0)
    for name, parameter in actual.named_parameters():
        target = dict(baseline.named_parameters())[name]
        torch.testing.assert_close(parameter.grad, target.grad, rtol=0, atol=0)
    assert actual.q_proj.weight.grad.count_nonzero() > 0


@pytest.mark.parametrize("bias", [False, True])
def test_attention_constructor_owns_parameters_and_kernel_metadata(bias, monkeypatch):
    config = model_config(attention_bias=bias)
    torch.manual_seed(103)
    reference = ReferenceAttention(config, 0)
    rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("local attention must own parameter construction")

    monkeypatch.setattr(ReferenceAttention, "__init__", forbidden)
    torch.manual_seed(103)
    actual = Qwen3MoeAttention(deepcopy(config), 0)
    assert torch.equal(rng, torch.get_rng_state())
    assert actual._modules.keys() == reference._modules.keys()
    assert actual._hidden_kernels.keys() == reference._hidden_kernels.keys()
    assert actual.state_dict().keys() == reference.state_dict().keys()
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
    for name in ("head_dim", "num_key_value_groups", "scaling", "sliding_window"):
        assert getattr(actual, name) == getattr(reference, name)


@pytest.mark.parametrize("entry", ["attention", "decoder", "lm"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("num_attention_heads", 0),
        ("num_key_value_heads", 0),
        ("num_attention_heads", True),
        ("head_dim", 7),
        ("hidden_size", -1),
        ("num_attention_heads", 3),
    ],
)
def test_bad_geometry_fails_before_allocation(entry, field, value, monkeypatch):
    config = model_config()
    setattr(config, field, value)
    before = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid geometry must fail before parameter allocation")

    monkeypatch.setattr(torch.nn.Linear, "__init__", forbidden)
    monkeypatch.setattr(torch.nn.Embedding, "__init__", forbidden)
    with pytest.raises(ValueError, match="Qwen3"):
        if entry == "attention":
            Qwen3MoeAttention(config, 0)
        else:
            constructor = Qwen3MoeModel if entry == "decoder" else Qwen3MoeForCausalLM
            constructor(config)
    assert torch.equal(before, torch.get_rng_state())


def test_missing_head_dim_keeps_reference_inference():
    config = model_config()
    del config.head_dim
    torch.manual_seed(109)
    reference = ReferenceAttention(config, 0)
    torch.manual_seed(109)
    actual = Qwen3MoeAttention(config, 0)
    assert actual.head_dim == reference.head_dim == 8
    for name, value in actual.state_dict().items():
        assert torch.equal(value, reference.state_dict()[name])
