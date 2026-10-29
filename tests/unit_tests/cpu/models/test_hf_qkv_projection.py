"""Fused HF projection ownership, arithmetic, and execution contracts."""

from __future__ import annotations

from tests.caps import require_env

require_env("flex_attention", "spmd_types")

import copy

import pytest
import torch
from torch import nn
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from llmtuner.config.llmtuner_config import LLMTunerConfig
from llmtuner.config.model_config import ModelConfig
from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.errors import ConfigError, UnsupportedCombinationError
from llmtuner.models.hf.factory import build_model_config_for, unwrap_text_config
from llmtuner.models.hf.materialization import materialize_meta_model
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.models.hf.qkv_projection import (
    apply_fused_qkv_projection,
    forward_fused_qkv_attention,
    fused_qkv_projection,
)
from llmtuner.parallel.tensor_parallel.tensor_parallel import apply_tp


def make_config(projection: str = "fused") -> LLMTunerConfig:
    return LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            qkv_projection=projection,
            arch_overrides={"head_dim": 8},
        )
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_fused_projection_matches_separate_forward_and_gradients(dtype, bias, kv_heads):
    torch.manual_seed(31)
    config = AutoConfig.for_model(
        "qwen3",
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=kv_heads,
        head_dim=8,
        attention_bias=bias,
    )
    reference = Qwen3Attention(config, layer_idx=0).to(dtype)
    actual = copy.deepcopy(reference)
    hidden = torch.randn(3, 2, 32, dtype=dtype).transpose(0, 1).requires_grad_()
    reference_hidden = hidden.detach().clone().requires_grad_()
    expected = tuple(
        projection(reference_hidden).unflatten(-1, (-1, 8))
        for projection in (reference.q_proj, reference.k_proj, reference.v_proj)
    )
    observed = fused_qkv_projection(actual, hidden)
    upstream = tuple(torch.randn_like(value) for value in expected)
    for value, target in zip(observed, expected, strict=True):
        torch.testing.assert_close(value, target)
    torch.autograd.backward(observed, upstream)
    torch.autograd.backward(expected, upstream)
    mathematical_gradient = torch.zeros_like(hidden, dtype=torch.float64)
    for gradient, projection in zip(
        upstream, (reference.q_proj, reference.k_proj, reference.v_proj), strict=True
    ):
        mathematical_gradient += (
            gradient.double().flatten(-2).matmul(projection.weight.detach().double())
        )
    torch.testing.assert_close(hidden.grad, mathematical_gradient.to(dtype))
    if dtype == torch.float32:
        torch.testing.assert_close(hidden.grad, reference_hidden.grad)
    else:
        assert not torch.equal(hidden.grad, reference_hidden.grad)
    for name in ("q_proj", "k_proj", "v_proj"):
        torch.testing.assert_close(
            getattr(actual, name).weight.grad, getattr(reference, name).weight.grad
        )
        if bias:
            torch.testing.assert_close(
                getattr(actual, name).bias.grad, getattr(reference, name).bias.grad
            )


@pytest.mark.parametrize("projection", ["separate", "fused"])
def test_projection_config_reaches_model_without_global_class_changes(projection):
    original_forward = Qwen3Attention.forward
    config = build_model_config_for(make_config(projection))
    assert config.qkv_projection == projection
    model = HFTransformerModel(config)
    attention = model.layers[0].self_attn
    selected = getattr(attention.forward, "__func__", None)
    assert (selected is forward_fused_qkv_attention) == (projection == "fused")
    assert Qwen3Attention.forward is original_forward
    assert torch.isfinite(model(torch.arange(8))).all()


def test_qwen3_moe_fused_fp32_rotary_and_torch_norm_execute() -> None:
    from llmtuner.models.hf.model_setup import forward_torch_rms_norm

    config = AutoConfig.for_model(
        "qwen3_moe",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
    )
    config.qkv_projection = "fused"
    config.rotary_compute_dtype = "float32"
    config.norm_backend = "torch"
    model = HFTransformerModel(config).to(torch.bfloat16)
    attention = model.layers[0].self_attn

    assert attention.forward.__func__ is forward_fused_qkv_attention
    assert model.layers[0].input_layernorm.forward.__func__ is (forward_torch_rms_norm)
    hidden = torch.randn(2, 3, 32).bfloat16()
    expected_norm = torch.nn.functional.rms_norm(
        hidden, (32,), model.layers[0].input_layernorm.weight, 1e-6
    )
    torch.testing.assert_close(
        model.layers[0].input_layernorm(hidden), expected_norm, atol=0, rtol=0
    )
    assert torch.isfinite(model(torch.arange(8))).all()


def test_qwen3_moe_torchtitan_experts_execute_and_backpropagate() -> None:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

    from llmtuner.models.hf.model_setup import forward_torchtitan_moe

    config = AutoConfig.for_model(
        "qwen3_moe",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
    )
    config.experts_implementation = "torchtitan"
    original_forward = Qwen3MoeSparseMoeBlock.forward
    model = HFTransformerModel(config).to(torch.bfloat16)
    moe = model.layers[0].mlp

    assert moe.forward.__func__ is forward_torchtitan_moe
    assert Qwen3MoeSparseMoeBlock.forward is original_forward
    logits = model(torch.arange(8))
    assert torch.isfinite(logits).all()
    logits.float().square().sum().backward()
    assert torch.isfinite(moe.experts.gate_up_proj.grad).all()
    assert torch.isfinite(moe.experts.down_proj.grad).all()
    assert moe.experts.gate_up_proj.grad.dtype == torch.bfloat16


def test_explicit_flex_backend_reaches_model() -> None:
    from llmtuner.models.hf.attention_backend import ATTENTION_BACKEND

    config = make_config()
    config.model.attention_backend = "flex"
    model = HFTransformerModel(build_model_config_for(config))
    assert model.model.config._attn_implementation == ATTENTION_BACKEND
    with torch.no_grad():
        assert torch.isfinite(model(torch.arange(8))).all()


def test_fused_installation_preserves_parameter_identity_keys_and_is_idempotent():
    model = HFTransformerModel(build_model_config_for(make_config("separate")))
    parameters = {name: id(parameter) for name, parameter in model.named_parameters()}
    state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.model.config.qkv_projection = "fused"
    apply_fused_qkv_projection(model.model)
    apply_fused_qkv_projection(model.model)
    assert parameters == {
        name: id(parameter) for name, parameter in model.named_parameters()
    }
    assert model.state_dict().keys() == state.keys()
    assert {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    } == set(parameters.values())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], atol=0, rtol=0)
    model(torch.arange(8)).sum().backward()
    optimizer.step()
    restored = HFTransformerModel(build_model_config_for(make_config()))
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(
            restored(torch.arange(8)), model(torch.arange(8)), atol=0, rtol=0
        )


@pytest.mark.parametrize(
    "hook_kind", ["forward", "forward_pre", "backward", "backward_pre"]
)
def test_fused_projection_rejects_hooks_added_after_installation(hook_kind):
    model = HFTransformerModel(build_model_config_for(make_config()))
    attention = model.layers[0].self_attn
    register = {
        "forward": attention.q_proj.register_forward_hook,
        "forward_pre": attention.q_proj.register_forward_pre_hook,
        "backward": attention.q_proj.register_full_backward_hook,
        "backward_pre": attention.q_proj.register_full_backward_pre_hook,
    }[hook_kind]
    handle = register(lambda *args: None)
    try:
        with pytest.raises(UnsupportedCombinationError, match="cannot bypass"):
            model(torch.arange(8))
    finally:
        handle.remove()


def test_fused_projection_rejects_custom_projection_forwards_and_partial_biases():
    model = HFTransformerModel(build_model_config_for(make_config()))
    attention = model.layers[0].self_attn
    attention.q_proj.forward = lambda value: value
    with pytest.raises(UnsupportedCombinationError, match="cannot bypass"):
        model(torch.arange(8))
    del attention.q_proj.forward
    attention.q_proj.bias = nn.Parameter(torch.zeros(attention.q_proj.out_features))
    with pytest.raises(UnsupportedCombinationError, match="all three"):
        model(torch.arange(8))


def test_fused_projection_rejects_unadapted_tensor_parallelism_before_mesh_access():
    with pytest.raises(ConfigError, match="tensor parallelism"):
        LLMTunerConfig(
            model=make_config().model, parallel=ParallelConfig(tensor_parallel_size=2)
        )
    model = HFTransformerModel(build_model_config_for(make_config()))
    identities = {name: id(parameter) for name, parameter in model.named_parameters()}
    with pytest.raises(UnsupportedCombinationError, match="tensor parallelism"):
        apply_tp(model, None, ParallelConfig(tensor_parallel_size=2))
    assert identities == {
        name: id(parameter) for name, parameter in model.named_parameters()
    }


def test_fused_projection_rejects_other_model_families():
    config = make_config()
    config.model.model_name_or_path = "llama"
    with pytest.raises(UnsupportedCombinationError, match="Qwen3 or Qwen3-MoE"):
        HFTransformerModel(build_model_config_for(config))


def test_composite_config_keeps_projection_selection():
    parent = AutoConfig.for_model("qwen2_5_vl")
    parent.qkv_projection = "fused"
    assert unwrap_text_config(parent).qkv_projection == "fused"


def test_fused_meta_materialization_and_saved_config_preserve_execution(tmp_path):
    config = AutoConfig.for_model(
        "qwen3",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    config.qkv_projection = "fused"
    reference = HFTransformerModel(config).eval()
    config.save_pretrained(tmp_path)
    restored_config = AutoConfig.from_pretrained(tmp_path)
    assert restored_config.qkv_projection == "fused"
    with torch.device("meta"):
        restored = HFTransformerModel(restored_config).eval()
    materialize_meta_model(restored, torch.device("cpu"))
    restored.load_state_dict(reference.state_dict(), strict=True)
    torch.testing.assert_close(
        restored(torch.arange(8)), reference(torch.arange(8)), atol=0, rtol=0
    )


def test_fused_optimizer_resume_matches_next_update():
    model = HFTransformerModel(build_model_config_for(make_config()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.arange(8)).square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    restored = HFTransformerModel(build_model_config_for(make_config()))
    restored.load_state_dict(model.state_dict(), strict=True)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for current_model, current_optimizer in (
        (model, optimizer),
        (restored, restored_optimizer),
    ):
        current_model(torch.arange(4, 12)).square().sum().backward()
        current_optimizer.step()
    for name, parameter in restored.named_parameters():
        expected = dict(model.named_parameters())[name]
        torch.testing.assert_close(parameter, expected, atol=0, rtol=0)
        for state_name, value in restored_optimizer.state[parameter].items():
            torch.testing.assert_close(
                value, optimizer.state[expected][state_name], atol=0, rtol=0
            )


def test_fused_installation_preflight_prevents_partial_adaptation():
    config = make_config("separate")
    config.model.num_hidden_layers = 2
    model = HFTransformerModel(build_model_config_for(config))
    model.model.config.qkv_projection = "fused"
    handle = model.layers[1].self_attn.q_proj.register_forward_hook(lambda *args: None)
    try:
        with pytest.raises(UnsupportedCombinationError, match="cannot bypass"):
            apply_fused_qkv_projection(model.model)
        assert all("forward" not in layer.self_attn.__dict__ for layer in model.layers)
    finally:
        handle.remove()


def test_fused_projection_respects_frozen_parameters():
    model = HFTransformerModel(build_model_config_for(make_config()))
    attention = model.layers[0].self_attn
    attention.q_proj.weight.requires_grad_(False)
    model(torch.arange(8)).sum().backward()
    assert attention.q_proj.weight.grad is None
    assert torch.isfinite(attention.k_proj.weight.grad).all()
    assert torch.isfinite(attention.v_proj.weight.grad).all()


def test_fused_attention_executes_fullgraph_aot_with_matching_gradients():
    torch.manual_seed(17)
    model = HFTransformerModel(build_model_config_for(make_config())).eval()
    reference = model.layers[0].self_attn
    actual = copy.deepcopy(reference)
    compiled = torch.compile(actual, backend="aot_eager", fullgraph=True)
    hidden = torch.randn(2, 4, 32, requires_grad=True)
    other_hidden = hidden.detach().clone().requires_grad_()
    position_embeddings = (torch.ones(2, 4, 8), torch.zeros(2, 4, 8))
    expected, _ = reference(hidden, position_embeddings, None)
    observed, _ = compiled(other_hidden, position_embeddings, None)
    torch.testing.assert_close(observed, expected, atol=0, rtol=0)
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    observed.backward(upstream)
    torch.testing.assert_close(hidden.grad, other_hidden.grad)
    for name, parameter in actual.named_parameters():
        torch.testing.assert_close(
            parameter.grad, dict(reference.named_parameters())[name].grad
        )


def test_fp32_rotary_rejects_scaled_frequencies():
    from types import SimpleNamespace

    from llmtuner.models.hf.qkv_projection import apply_configured_rotary

    module = SimpleNamespace(
        config=SimpleNamespace(
            rotary_compute_dtype="float32",
            rope_parameters={"rope_type": "linear", "factor": 2.0},
        )
    )
    query = torch.randn(1, 2, 3, 8)
    with pytest.raises(UnsupportedCombinationError, match="without scaling"):
        apply_configured_rotary(module, query, query, None, None, torch.arange(3))
