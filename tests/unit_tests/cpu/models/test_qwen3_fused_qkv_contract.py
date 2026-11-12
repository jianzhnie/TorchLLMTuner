"""Model-owned QKV selection keeps logical checkpoint parameters intact."""

from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn.modules import module as module_hooks
from torch.nn.utils.parametrize import register_parametrization
from torch.utils.checkpoint import DefaultDeviceType, checkpoint

from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM, Qwen3MoeModel
from tests.unit_tests.cpu.models.test_qwen3_model_contract import model_config


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_fused_qkv_preserves_state_and_training(bias, backend):
    config = model_config()
    config.attention_bias = bias
    config._attn_implementation = backend
    config._experts_implementation = "eager"
    torch.manual_seed(29)
    reference = Qwen3MoeForCausalLM(deepcopy(config))
    rng = torch.get_rng_state().clone()
    config.qkv_projection = "fused"
    torch.manual_seed(29)
    actual = Qwen3MoeForCausalLM(config)
    assert torch.equal(rng, torch.get_rng_state())
    assert actual.state_dict().keys() == reference.state_dict().keys()
    identities = {name: id(value) for name, value in actual.named_parameters()}
    for name, value in actual.state_dict().items():
        assert torch.equal(value, reference.state_dict()[name])
    tokens = torch.tensor([[1, 7, 3, 9], [2, 8, 4, 5]])
    expected = reference(tokens, labels=tokens)
    result = actual(tokens, labels=tokens)
    # One GEMM changes accumulation order; declare bounds before comparison.
    torch.testing.assert_close(result.logits, expected.logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(result.loss, expected.loss, rtol=1e-5, atol=1e-6)
    result.loss.backward()
    expected.loss.backward()
    for name, value in actual.named_parameters():
        assert id(value) == identities[name]
        other = dict(reference.named_parameters())[name]
        assert (value.grad is None) == (other.grad is None)
        if value.grad is not None:
            torch.testing.assert_close(value.grad, other.grad, rtol=1e-5, atol=1e-6)
    grad = actual.model.layers[0].self_attn.q_proj.weight.grad
    assert grad.isfinite().all() and grad.count_nonzero() > 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_qkv_low_precision_backward_and_recomputation(dtype):
    config = model_config()
    config.qkv_projection = "fused"
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    torch.manual_seed(37)
    model = Qwen3MoeForCausalLM(config).to(dtype)
    attention = model.model.layers[0].self_attn
    replay = deepcopy(attention)
    hidden = torch.randn(2, 3, config.hidden_size, dtype=dtype, requires_grad=True)
    other = hidden.detach().clone().requires_grad_()
    positions = torch.arange(3).expand(2, -1)
    embeddings = model.model.rotary_emb(hidden, positions)
    mask = torch.zeros(2, 1, 3, 3, dtype=dtype)
    expected = attention(hidden, embeddings, mask)[0]
    previous = DefaultDeviceType.get_device_type()
    DefaultDeviceType.set_device_type("cpu")
    try:
        result = checkpoint(
            lambda inputs: replay(inputs, embeddings, mask)[0],
            other,
            use_reentrant=False,
        )
        upstream = torch.randn_like(result)
        expected.backward(upstream)
        result.backward(upstream)
    finally:
        DefaultDeviceType.set_device_type(previous)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(other.grad, hidden.grad, rtol=0, atol=0)
    for name, value in replay.named_parameters():
        reference = dict(attention.named_parameters())[name]
        torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)
    assert other.grad.isfinite().all() and other.grad.count_nonzero() > 0


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_fused_decoder_prefill_and_cached_decode(backend):
    config = model_config()
    config._attn_implementation = backend
    config._experts_implementation = "eager"
    torch.manual_seed(47)
    reference = Qwen3MoeForCausalLM(deepcopy(config)).eval().model
    config.qkv_projection = "fused"
    actual = Qwen3MoeForCausalLM(config).eval().model
    actual.load_state_dict(reference.state_dict())
    tokens = torch.tensor([[1, 4, 8], [2, 5, 9]])
    with torch.no_grad():
        expected = reference(tokens, use_cache=True)
        result = actual(tokens, use_cache=True)
        torch.testing.assert_close(
            result.last_hidden_state,
            expected.last_hidden_state,
            rtol=1e-5,
            atol=1e-6,
        )
        next_tokens = torch.tensor([[7], [3]])
        expected = reference(
            next_tokens,
            past_key_values=expected.past_key_values,
            use_cache=True,
        )
        result = actual(
            next_tokens,
            past_key_values=result.past_key_values,
            use_cache=True,
        )
    torch.testing.assert_close(
        result.last_hidden_state,
        expected.last_hidden_state,
        rtol=1e-5,
        atol=1e-6,
    )
    assert result.past_key_values.get_seq_length() == 4


@pytest.mark.parametrize("decoder_only", [False, True])
def test_invalid_qkv_policy_rejects_before_allocation(monkeypatch, decoder_only):
    config = model_config()
    config.qkv_projection = "unknown"
    before = torch.get_rng_state().clone()

    def forbidden_embedding(*args, **kwargs):
        raise AssertionError("validation must precede parameter allocation")

    monkeypatch.setattr(torch.nn.Embedding, "__init__", forbidden_embedding)
    constructor = Qwen3MoeModel if decoder_only else Qwen3MoeForCausalLM
    with pytest.raises(ValueError, match="qkv_projection"):
        constructor(config)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("projection", ["q_proj", "k_proj", "v_proj"])
@pytest.mark.parametrize("hook_kind", ["pre", "forward", "backward_pre", "backward"])
def test_fused_qkv_rejects_projection_hooks_before_math(projection, hook_kind):
    config = model_config()
    config.qkv_projection = "fused"
    attention = Qwen3MoeForCausalLM(config).layers[0].self_attn
    target = getattr(attention, projection)
    register = {
        "pre": target.register_forward_pre_hook,
        "forward": target.register_forward_hook,
        "backward_pre": target.register_full_backward_pre_hook,
        "backward": target.register_full_backward_hook,
    }[hook_kind]
    handle = register(lambda *args: None)
    hidden = torch.randn(1, 3, config.hidden_size, requires_grad=True)
    rotary = (torch.ones(1, 3, config.head_dim), torch.zeros(1, 3, config.head_dim))
    try:
        with pytest.raises(ValueError, match="fused QKV.*projection"):
            attention(hidden, rotary, None)
        assert hidden.grad is None
    finally:
        handle.remove()
    # Removing the conflict restores the original fused path, without any
    # permanent installation marker or changed projection parameters.
    output = attention(hidden, rotary, None)[0]
    assert output.isfinite().all() and output.count_nonzero() > 0


@pytest.mark.parametrize("projection", ["q_proj", "k_proj", "v_proj"])
def test_fused_qkv_rejects_custom_projection_forward(projection, monkeypatch):
    config = model_config()
    config.qkv_projection = "fused"
    attention = Qwen3MoeForCausalLM(config).layers[0].self_attn
    target = getattr(attention, projection)
    monkeypatch.setattr(target, "forward", lambda value: value)
    hidden = torch.randn(1, 3, config.hidden_size)
    rotary = (torch.ones(1, 3, config.head_dim), torch.zeros(1, 3, config.head_dim))
    with pytest.raises(ValueError, match="fused QKV.*projection"):
        attention(hidden, rotary, None)


@pytest.mark.parametrize("kind", ["forward_pre", "forward", "backward_pre", "backward"])
def test_fused_qkv_rejects_global_module_hooks(kind):
    config = model_config()
    config.qkv_projection = "fused"
    attention = Qwen3MoeForCausalLM(config).layers[0].self_attn
    hidden = torch.randn(1, 3, config.hidden_size)
    rotary = (torch.ones(1, 3, config.head_dim), torch.zeros(1, 3, config.head_dim))
    name = f"full_{kind}" if kind.startswith("backward") else kind
    register = getattr(module_hooks, f"register_module_{name}_hook")
    previous_backward_kind = module_hooks._global_is_full_backward_hook
    handle = register(lambda *args: None)
    try:
        with pytest.raises(ValueError, match="cannot bypass global module hooks"):
            attention(hidden, rotary, None)
    finally:
        handle.remove()
        module_hooks._global_is_full_backward_hook = previous_backward_kind
    assert attention(hidden, rotary, None)[0].isfinite().all()


@pytest.mark.parametrize("projection", ["q_proj", "k_proj", "v_proj"])
def test_fused_qkv_rejects_individually_compiled_projection(projection):
    config = model_config()
    config.qkv_projection = "fused"
    attention = Qwen3MoeForCausalLM(config).layers[0].self_attn
    getattr(attention, projection).compile(backend="eager")
    hidden = torch.randn(1, 3, config.hidden_size)
    rotary = (torch.ones(1, 3, config.head_dim), torch.zeros(1, 3, config.head_dim))
    with pytest.raises(ValueError, match="cannot bypass modified projection"):
        attention(hidden, rotary, None)


def test_fused_projection_checks_preserve_fullgraph_compilation():
    config = model_config()
    config.qkv_projection = "fused"
    config._attn_implementation = "eager"
    attention = Qwen3MoeForCausalLM(config).layers[0].self_attn
    reference = deepcopy(attention)
    hidden = torch.randn(1, 3, config.hidden_size, requires_grad=True)
    other = hidden.detach().clone().requires_grad_()
    rotary = (torch.ones(1, 3, config.head_dim), torch.zeros(1, 3, config.head_dim))
    expected = reference(hidden, rotary, None)[0]
    actual = torch.compile(attention, backend="aot_eager", fullgraph=True)(
        other, rotary, None
    )[0]
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    actual.backward(upstream)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(other.grad, hidden.grad, rtol=0, atol=0)
    assert other.grad.isfinite().all() and other.grad.count_nonzero() > 0
    for name, parameter in attention.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            dict(reference.named_parameters())[name].grad,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("projection", ["q_proj", "k_proj", "v_proj"])
@pytest.mark.parametrize("method", ["__call__", "_call_impl"])
def test_fused_qkv_rejects_modified_projection_call(projection, method):
    class ModifiedLinear(nn.Linear):
        pass

    def modified_call(self, value):
        return nn.functional.linear(value, self.weight, self.bias) * 2

    setattr(ModifiedLinear, method, modified_call)
    config = model_config()
    config.qkv_projection = "fused"
    attention = Qwen3MoeForCausalLM(config).layers[0].self_attn
    target = getattr(attention, projection)
    replacement = ModifiedLinear(target.in_features, target.out_features, bias=False)
    replacement.weight = target.weight
    setattr(attention, projection, replacement)
    hidden = torch.randn(1, 3, config.hidden_size)
    rotary = (torch.ones(1, 3, config.head_dim), torch.zeros(1, 3, config.head_dim))
    with pytest.raises(ValueError, match="fused QKV.*projection"):
        attention(hidden, rotary, None)


@pytest.mark.parametrize("projection", ["q_proj", "k_proj", "v_proj"])
def test_fused_qkv_preserves_differentiable_weight_parametrization(projection):
    class ScaleWeight(nn.Module):
        def forward(self, weight):
            return weight * 0.75

    config = model_config()
    config.qkv_projection = "fused"
    config._attn_implementation = "eager"
    attention = Qwen3MoeForCausalLM(config).layers[0].self_attn
    register_parametrization(getattr(attention, projection), "weight", ScaleWeight())
    reference = deepcopy(attention)
    reference._llmtuner_fused_qkv = False
    hidden = torch.randn(1, 3, config.hidden_size, requires_grad=True)
    other = hidden.detach().clone().requires_grad_()
    rotary = (torch.ones(1, 3, config.head_dim), torch.zeros(1, 3, config.head_dim))
    actual = attention(hidden, rotary, None)[0]
    expected = reference(other, rotary, None)[0]
    upstream = torch.randn_like(expected)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(hidden.grad, other.grad, rtol=1e-5, atol=1e-6)
    for name, parameter in attention.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            dict(reference.named_parameters())[name].grad,
            rtol=1e-5,
            atol=1e-6,
        )
    gradient = getattr(attention, projection).parametrizations.weight.original.grad
    assert gradient.isfinite().all() and gradient.count_nonzero() > 0


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("target", ["q_proj", "k_proj", "v_proj"])
def test_fused_qkv_rejects_partial_bias_layout_before_projection(bias, target):
    config = model_config()
    config.qkv_projection = "fused"
    config.attention_bias = bias
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    model = Qwen3MoeForCausalLM(config)
    attention = model.layers[0].self_attn
    projection = getattr(attention, target)
    projection.bias = (
        None if bias else nn.Parameter(torch.zeros(projection.out_features))
    )
    identities = {name: id(value) for name, value in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="biases on all three projections"):
        model(torch.tensor([[1, 7, 3, 9]]))
    assert identities == {name: id(value) for name, value in model.named_parameters()}
    assert torch.equal(rng, torch.get_rng_state())
    assert all(value.grad is None for value in model.parameters())


@pytest.mark.parametrize("frozen", ["q_proj", "k_proj", "v_proj"])
def test_fused_qkv_frozen_projection_survives_optimizer_resume(frozen):
    config = model_config()
    config.qkv_projection = "fused"
    config.attention_bias = True
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    torch.manual_seed(83)
    model = Qwen3MoeForCausalLM(config)
    projection = getattr(model.layers[0].self_attn, frozen)
    projection.requires_grad_(False)
    original = {
        name: value.detach().clone() for name, value in projection.named_parameters()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    first = torch.tensor([[1, 7, 3, 9], [2, 8, 4, 5]])
    model(first, labels=first).loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    restored = Qwen3MoeForCausalLM(deepcopy(config))
    restored.load_state_dict(deepcopy(model.state_dict()), strict=True)
    getattr(restored.layers[0].self_attn, frozen).requires_grad_(False)
    restored_optimizer = torch.optim.AdamW(
        restored.parameters(), lr=1e-3, foreach=False
    )
    restored_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    second = torch.tensor([[3, 6, 2, 8], [9, 1, 5, 7]])
    for current, current_optimizer in (
        (model, optimizer),
        (restored, restored_optimizer),
    ):
        current(second, labels=second).loss.backward()
        for name in ("q_proj", "k_proj", "v_proj"):
            weight = getattr(current.layers[0].self_attn, name).weight
            if name == frozen:
                assert weight.grad is None
                assert weight not in current_optimizer.state
            else:
                assert weight.grad.isfinite().all() and weight.grad.count_nonzero() > 0
        current_optimizer.step()
    for name, value in projection.named_parameters():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    for name, value in model.named_parameters():
        other = dict(restored.named_parameters())[name]
        torch.testing.assert_close(value, other, rtol=0, atol=0)
        assert (
            optimizer.state.get(value, {}).keys()
            == restored_optimizer.state.get(other, {}).keys()
        )
        for key, state in optimizer.state.get(value, {}).items():
            torch.testing.assert_close(
                state, restored_optimizer.state[other][key], rtol=0, atol=0
            )
