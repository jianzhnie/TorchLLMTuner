"""CPU ownership/guard contracts, not NPU arithmetic acceptance."""

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet

from llmtuner.config.model_config import ModelConfig
from llmtuner.errors import ConfigError, UnsupportedCombinationError
from llmtuner.models.hf.factory import apply_runtime_config, unwrap_text_config
from llmtuner.models.qwen3_5 import gdn_numerics as adapter


def holder():
    config = AutoConfig.for_model(
        "qwen3_5_moe_text",
        hidden_size=2048,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        hidden_act="silu",
        rms_norm_eps=1e-6,
    )
    config.gdn_numerical_policy = adapter.POLICY
    model = nn.Module()
    model.config = config
    with torch.device("meta"):
        model.gdn = Qwen3_5MoeGatedDeltaNet(config, 0).bfloat16()
    return model


def test_config_factory_default_and_composite_propagation():
    assert ModelConfig().gdn_numerical_policy == "native"
    with pytest.raises(ConfigError, match="gdn_numerical_policy"):
        ModelConfig(gdn_numerical_policy="auto")
    cfg = SimpleNamespace(
        model=ModelConfig(gdn_numerical_policy=adapter.POLICY),
        dataloader=SimpleNamespace(attention_mask_type=lambda: "causal"),
        norm_backend="native",
        qkv_projection="separate",
        rotary_init_device="constructor",
        attention_backend="auto",
    )
    config = PretrainedConfig()
    config.text_config = PretrainedConfig()
    apply_runtime_config(config, cfg)
    assert unwrap_text_config(config).gdn_numerical_policy == adapter.POLICY


def test_native_noop_and_install_identity_idempotency():
    native = nn.Module()
    native.config = SimpleNamespace()
    adapter.configure_gdn_numerics(native)
    model = holder()
    before, keys = dict(model.named_parameters()), set(model.state_dict())
    recurrence = getattr(model.gdn, "chunk_gated_delta_rule", None)
    adapter.configure_gdn_numerics(model)
    adapter.configure_gdn_numerics(model)
    assert set(model.state_dict()) == keys
    assert all(dict(model.named_parameters())[k] is p for k, p in before.items())
    assert len(model.gdn._forward_pre_hooks) == 1
    assert getattr(model.gdn, "chunk_gated_delta_rule", None) is recurrence
    assert model.gdn.in_proj_qkv.forward.__func__ is adapter.forward_split_qkv
    model.config.gdn_numerical_policy = "native"
    with pytest.raises(UnsupportedCombinationError, match="removed"):
        adapter.configure_gdn_numerics(model)


def test_reject_custom_or_wrong_dtype_before_any_mutation():
    model = holder()
    model.gdn.norm.forward = MethodType(lambda self, x, gate: x, model.gdn.norm)
    with pytest.raises(UnsupportedCombinationError, match="unmodified"):
        adapter.configure_gdn_numerics(model)
    assert "forward" not in vars(model.gdn.in_proj_qkv)
    model = holder().float()
    with pytest.raises(UnsupportedCombinationError, match="BF16"):
        adapter.configure_gdn_numerics(model)


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_size", 16),
        ("model_type", "qwen3_moe"),
        ("attn_mask_type", "block_causal"),
        ("qkv_projection", "fused"),
    ],
)
def test_config_geometry_and_single_policy_rejection(field, value):
    model = holder()
    setattr(model.config, field, value)
    with pytest.raises(UnsupportedCombinationError):
        adapter.configure_gdn_numerics(model)


def test_projection_views_and_norm_backward_on_cpu():
    projection = nn.Linear(2048, 8192, bias=False, device="meta", dtype=torch.bfloat16)
    result = adapter.forward_split_qkv(
        projection, torch.empty(1, 64, 2048, device="meta", dtype=torch.bfloat16)
    )
    assert result.shape == (1, 64, 8192)
    norm = nn.Module()
    norm.weight = nn.Parameter(torch.ones(8, dtype=torch.bfloat16))
    norm.variance_epsilon = 1e-6
    x = torch.randn(2, 8, dtype=torch.bfloat16, requires_grad=True)
    gate = torch.randn_like(x, requires_grad=True)
    dy = torch.randn_like(x)
    actual = adapter.forward_fp32_weight_norm(norm, x, gate)
    grads = torch.autograd.grad(actual, (x, gate, norm.weight), dy)
    fp32 = x.float()
    normalized = fp32 * torch.rsqrt(fp32.pow(2).mean(-1, keepdim=True) + 1e-6)
    expected = (norm.weight.float() * normalized).bfloat16()
    expected = (expected * nn.functional.silu(gate.float())).bfloat16()
    oracle = torch.autograd.grad(expected, (x, gate, norm.weight), dy)
    assert torch.equal(actual, expected)
    assert all(torch.equal(a, b) for a, b in zip(grads, oracle, strict=True))


def test_batch_metadata_guards_and_direct_cache_rejection():
    batch = {
        "input": torch.zeros(1, 64, dtype=torch.long),
        "positions": torch.arange(64),
    }
    adapter.validate_gdn_numerical_batch(batch)
    for extra in (
        {"padding_mask": torch.zeros(63, dtype=torch.bool)},
        {"padding_mask": torch.ones(64, dtype=torch.bool)},
        {"padding_mask": torch.zeros(64)},
        {"positions": torch.zeros(64)},
        {"input": torch.zeros(2, 64)},
        {"packed": True},
    ):
        with pytest.raises(UnsupportedCombinationError):
            adapter.validate_gdn_numerical_batch(batch | extra)
    with pytest.raises(UnsupportedCombinationError, match="cache"):
        adapter._validate_forward(
            nn.Module(), (torch.empty(1, 64, 2048),), {"cache_params": object()}
        )


def test_training_guard_independent_recurrence_and_rejections():
    model = holder()
    cfg = SimpleNamespace(
        model=ModelConfig(gdn_numerical_policy=adapter.POLICY),
        parallel=SimpleNamespace(
            tp_size=1,
            cp_size=1,
            ep_size=1,
            pp_size=1,
            fsdp_config=SimpleNamespace(fsdp_cpu_offload=False),
        ),
        training=SimpleNamespace(
            compile=False,
            activation_checkpoint_mode="full",
            activation_swap=False,
            global_batch_size=8,
            max_seq_len=2048,
        ),
        dataloader=SimpleNamespace(
            global_context_rows=False, attention_mask_type=lambda: "causal"
        ),
    )
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        adapter.validate_gdn_numerical_training(
            cfg,
            model.config,
            device=SimpleNamespace(type="npu"),
            parallel_dims=SimpleNamespace(dp_shard=8, dp_replicate=1),
        )
        cfg.parallel.ep_size = 2
        with pytest.raises(UnsupportedCombinationError, match="TP/CP/EP/PP"):
            adapter.validate_gdn_numerical_training(
                cfg,
                model.config,
                device=SimpleNamespace(type="npu"),
                parallel_dims=None,
            )
        cfg.parallel.ep_size = 1
        cfg.training.compile = True
        with pytest.raises(UnsupportedCombinationError, match="eager"):
            adapter.validate_gdn_numerical_training(
                cfg,
                model.config,
                device=SimpleNamespace(type="npu"),
                parallel_dims=None,
            )
    finally:
        torch.set_default_dtype(previous)
