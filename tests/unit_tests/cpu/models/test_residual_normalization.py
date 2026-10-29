"""Joint residual norm arithmetic, graph, and model ownership contracts."""

from __future__ import annotations

from tests.caps import require_env

require_env("flex_attention", "spmd_types")

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm

from llmtuner.config.llmtuner_config import LLMTunerConfig
from llmtuner.config.model_config import ModelConfig
from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.hf.factory import build_model_config_for
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.models.hf.model_setup import (
    _isolate_norm_input,
    stabilize_residual_norm_gradients,
)
from llmtuner.models.hf.residual_normalization import (
    apply_joint_residual_norm,
    forward_joint_residual_block,
    forward_joint_residual_norm,
    joint_rms_norm,
)


def make_model(backend="joint", projection="separate"):
    config = LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            norm_backend=backend,
            qkv_projection=projection,
            arch_overrides={"head_dim": 8},
        )
    )
    return HFTransformerModel(build_model_config_for(config))


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(4,), (0, 4), (3, 4), (2, 3, 4)])
def test_joint_norm_matches_hf_outputs_and_first_order_gradients(
    input_dtype, weight_dtype, shape
):
    torch.manual_seed(42)
    reference = Qwen3RMSNorm(4, eps=1e-6).to(weight_dtype)
    weight = reference.weight.detach().clone().requires_grad_()
    source = torch.randn(shape, dtype=input_dtype)
    if source.ndim > 1:
        source = source.transpose(0, -1).contiguous().transpose(0, -1)
    source.requires_grad_()
    reference_source = source.detach().clone().requires_grad_()
    actual_norm, actual_residual = joint_rms_norm(source, weight, 1e-6)
    expected_norm = reference(reference_source)
    torch.testing.assert_close(actual_norm, expected_norm, atol=0, rtol=0)
    torch.testing.assert_close(actual_residual, source, atol=0, rtol=0)
    upstream = torch.randn_like(expected_norm)
    residual = torch.randn_like(source)
    torch.autograd.backward((actual_norm, actual_residual), (upstream, residual))
    torch.autograd.backward((expected_norm, reference_source), (upstream, residual))
    torch.testing.assert_close(weight.grad, reference.weight.grad, atol=0, rtol=0)
    if input_dtype == torch.float32:
        torch.testing.assert_close(source.grad, reference_source.grad)
    else:
        torch.testing.assert_close(source.grad, reference_source.grad, atol=0, rtol=0)


def test_float32_joint_norm_enforces_direct_residual_indirect_merge():
    torch.manual_seed(19)
    source = torch.randn(2, 3, 4)
    weight = torch.randn(4)
    upstream = torch.randn_like(source)
    residual = torch.randn_like(source)
    terms = []
    for direct_only in (True, False):
        value = source.clone().requires_grad_()
        numerator = value if direct_only else value.detach()
        variance_source = value.detach() if direct_only else value
        normalized = weight * (
            numerator
            * torch.rsqrt(variance_source.pow(2).mean(-1, keepdim=True) + 1e-6)
        )
        terms.append(torch.autograd.grad(normalized, value, upstream)[0])
    actual = source.clone().requires_grad_()
    outputs = joint_rms_norm(actual, weight, 1e-6)
    torch.autograd.backward(outputs, (upstream, residual))
    expected = (terms[0] + residual) + terms[1]
    torch.testing.assert_close(actual.grad, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_joint_norm_operator_fake_schema_autograd_and_nonaliasing(dtype):
    source = torch.randn(2, 4, 3, dtype=dtype).transpose(1, 2).requires_grad_()
    weight = torch.randn(4, requires_grad=True)
    torch.library.opcheck(joint_rms_norm, (source, weight, 1e-6))
    normalized, residual = joint_rms_norm(source, weight, 1e-6)
    assert normalized.data_ptr() != source.data_ptr()
    assert residual.data_ptr() != source.data_ptr()


@pytest.mark.parametrize("output_index", [0, 1])
def test_joint_norm_handles_an_unused_output(output_index):
    source = torch.randn(2, 4, requires_grad=True)
    weight = torch.ones(4, requires_grad=True)
    outputs = joint_rms_norm(source, weight, 1e-6)
    outputs[output_index].sum().backward()
    assert torch.isfinite(source.grad).all()
    assert torch.isfinite(weight.grad).all()
    if output_index == 1:
        torch.testing.assert_close(source.grad, torch.ones_like(source), atol=0, rtol=0)
        torch.testing.assert_close(
            weight.grad, torch.zeros_like(weight), atol=0, rtol=0
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_joint_norm_fullgraph_aot_preserves_exact_first_order_results(dtype):
    def function(source, weight):
        normalized, residual = joint_rms_norm(source, weight, 1e-6)
        return normalized + residual

    torch.manual_seed(72)
    source = torch.randn(2, 3, 4, dtype=dtype, requires_grad=True)
    other_source = source.detach().clone().requires_grad_()
    weight = torch.randn(4, dtype=dtype, requires_grad=True)
    other_weight = weight.detach().clone().requires_grad_()
    expected = function(source, weight)
    actual = torch.compile(function, backend="aot_eager", fullgraph=True)(
        other_source, other_weight
    )
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    actual.backward(upstream)
    for observed, reference in (
        (actual, expected),
        (source.grad, other_source.grad),
        (weight.grad, other_weight.grad),
    ):
        torch.testing.assert_close(observed, reference, atol=0, rtol=0)


@pytest.mark.parametrize("projection", ["separate", "fused"])
def test_joint_model_preserves_weights_and_global_hf_classes(projection):
    original_block = Qwen3DecoderLayer.forward
    original_norm = Qwen3RMSNorm.forward
    model = make_model("native", projection)
    parameters = {name: id(parameter) for name, parameter in model.named_parameters()}
    state = copy.deepcopy(model.state_dict())
    model.model.config.norm_backend = "joint"
    stabilize_residual_norm_gradients(model.model)
    assert parameters == {
        name: id(parameter) for name, parameter in model.named_parameters()
    }
    assert model.state_dict().keys() == state.keys()
    assert Qwen3DecoderLayer.forward is original_block
    assert Qwen3RMSNorm.forward is original_norm
    for layer in model.layers:
        assert layer.forward.__func__ is forward_joint_residual_block
        assert layer.input_layernorm.forward.__func__ is forward_joint_residual_norm
        assert layer.self_attn.q_norm.forward.__func__ is original_norm
        assert layer.self_attn.k_norm.forward.__func__ is original_norm
        assert _isolate_norm_input in layer.input_layernorm._forward_pre_hooks.values()
    model(torch.arange(8)).square().sum().backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_joint_backend_reapplication_cleans_only_owned_adaptations():
    model = make_model("native")
    handle = model.layers[0].self_attn.q_norm.register_forward_pre_hook(
        lambda *args: None
    )
    try:
        for backend in ("joint", "joint", "aten", "joint", "native"):
            model.model.config.norm_backend = backend
            stabilize_residual_norm_gradients(model.model)
            assert handle.id in model.layers[0].self_attn.q_norm._forward_pre_hooks
            for layer in model.layers:
                function = getattr(layer.forward, "__func__", None)
                assert (function is forward_joint_residual_block) == (
                    backend == "joint"
                )
                hooks = layer.input_layernorm._forward_pre_hooks.values()
                assert sum(hook is _isolate_norm_input for hook in hooks) == (
                    backend != "aten"
                )
            assert torch.isfinite(model(torch.arange(8))).all()
    finally:
        handle.remove()


@pytest.mark.parametrize("kind", ["forward_pre", "forward", "backward_pre", "backward"])
def test_joint_backend_preflight_rejects_norm_hook_contract_changes(kind):
    model = make_model("native")
    model.model.config.norm_backend = "joint"
    norm = model.layers[1].post_attention_layernorm
    register = {
        "forward_pre": norm.register_forward_pre_hook,
        "forward": norm.register_forward_hook,
        "backward_pre": norm.register_full_backward_pre_hook,
        "backward": norm.register_full_backward_hook,
    }[kind]
    handle = register(lambda *args: None)
    try:
        with pytest.raises(UnsupportedCombinationError, match="hook contracts"):
            apply_joint_residual_norm(model.model)
        assert all("forward" not in layer.__dict__ for layer in model.layers)
    finally:
        handle.remove()


def test_joint_backend_rejects_other_families_and_float64_inputs():
    model = nn.Module()
    model.config = SimpleNamespace(model_type="qwen3_moe", norm_backend="joint")
    with pytest.raises(ValueError, match="dense Qwen3"):
        stabilize_residual_norm_gradients(model)
    with pytest.raises(TypeError, match="supports float16"):
        joint_rms_norm(torch.randn(2, 4, dtype=torch.float64), torch.ones(4), 1e-6)


def test_joint_fused_block_fullgraph_aot_matches_eager():
    torch.manual_seed(63)
    reference = make_model("joint", "fused").eval().layers[0]
    actual = copy.deepcopy(reference)
    source = torch.randn(2, 4, 32, requires_grad=True)
    other_source = source.detach().clone().requires_grad_()
    positions = (torch.ones(2, 4, 8), torch.zeros(2, 4, 8))
    expected = reference(source, position_embeddings=positions)
    observed = torch.compile(actual, backend="aot_eager", fullgraph=True)(
        other_source, position_embeddings=positions
    )
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    observed.backward(upstream)
    torch.testing.assert_close(observed, expected, atol=0, rtol=0)
    torch.testing.assert_close(other_source.grad, source.grad, atol=0, rtol=0)
    for name, parameter in actual.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            dict(reference.named_parameters())[name].grad,
            atol=0,
            rtol=0,
        )


def test_joint_norm_rejects_hooks_added_after_installation():
    model = make_model()
    handle = model.layers[0].input_layernorm.register_forward_hook(lambda *args: None)
    try:
        with pytest.raises(UnsupportedCombinationError, match="hook contracts"):
            model(torch.arange(8))
    finally:
        handle.remove()


def test_joint_norm_frozen_weight_and_first_order_only():
    source = torch.randn(2, 4, requires_grad=True)
    normalized, residual = joint_rms_norm(source, torch.ones(4), 1e-6)
    (normalized + residual).sum().backward()
    assert torch.isfinite(source.grad).all()
    source = torch.randn(2, 4, requires_grad=True)
    normalized, residual = joint_rms_norm(source, torch.ones(4), 1e-6)
    gradient = torch.autograd.grad(
        (normalized + residual).sum(), source, create_graph=True
    )[0]
    with pytest.raises(RuntimeError, match="autograd|backward"):
        torch.autograd.grad(gradient.sum(), source)
