"""Explicit Qwen3.5 FP32 routing preserves storage and checkpoint contracts."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTopKRouter
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter

from llmtuner.config.model_config import ModelConfig
from llmtuner.errors import ConfigError
from llmtuner.models.hf.expert_kernels import (
    forward_fp32_topk_router,
    stabilize_moe_router,
)
from llmtuner.models.hf.factory import apply_runtime_config, unwrap_text_config


def test_config_rejects_unsupported_router_dtype():
    assert ModelConfig().router_compute_dtype is None
    assert ModelConfig(router_compute_dtype="float32").router_compute_dtype == "float32"
    for value in ("bfloat16", "float16", "auto", False):
        with pytest.raises(ConfigError, match="router_compute_dtype"):
            ModelConfig(router_compute_dtype=value)


def test_runtime_field_propagates_to_native_text_config():
    cfg = SimpleNamespace(
        model=ModelConfig(router_compute_dtype="float32"),
        dataloader=SimpleNamespace(attention_mask_type=lambda: "causal"),
        norm_backend="native",
        qkv_projection="separate",
        rotary_init_device="constructor",
        attention_backend="auto",
    )
    composite = PretrainedConfig()
    composite.text_config = PretrainedConfig()
    apply_runtime_config(composite, cfg)
    assert unwrap_text_config(composite).router_compute_dtype == "float32"
    cfg.model.router_compute_dtype = None
    apply_runtime_config(composite, cfg)
    assert unwrap_text_config(composite).router_compute_dtype is None


@pytest.mark.parametrize("family", ["qwen3_5_moe_text", "qwen3_moe"])
@pytest.mark.parametrize("setting", [None, "float32"])
@pytest.mark.parametrize("normalize", [False, True])
def test_real_router_forward_vjp_and_storage(family, setting, normalize):
    config = AutoConfig.for_model(
        family,
        hidden_size=32,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=normalize,
    )
    config.router_compute_dtype = setting
    router_type = Qwen3MoeTopKRouter if family == "qwen3_moe" else Qwen3_5MoeTopKRouter
    router = router_type(config).to(torch.bfloat16)
    generator = torch.Generator().manual_seed(41)
    with torch.no_grad():
        router.weight.copy_(torch.randn((8, 32), generator=generator) * 0.2)
    model = nn.Module()
    model.config = config
    model.router = router
    parameter = router.weight
    before = {key: value.clone() for key, value in model.state_dict().items()}
    original_forward = router.forward.__func__
    stabilize_moe_router(model)
    enabled = family == "qwen3_moe" or setting == "float32"
    assert router.forward.__func__ is (
        forward_fp32_topk_router if enabled else original_forward
    )
    assert router.weight is parameter and parameter.dtype == torch.bfloat16
    assert model.state_dict().keys() == before.keys()
    for key, value in before.items():
        assert torch.equal(value, model.state_dict()[key])
    if not enabled:
        return
    x = torch.randn((7, 32), generator=generator).bfloat16().requires_grad_()
    xr = x.detach().clone().requires_grad_()
    wr = parameter.detach().clone().requires_grad_()
    # An outer autocast must not undo the explicit FP32 gate contract.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits, scores, indices = router(x)
    expected_logits = xr.float() @ wr.float().T
    expected_scores, expected_indices = expected_logits.softmax(-1).topk(
        2, dim=-1, sorted=False
    )
    if family == "qwen3_5_moe_text" or normalize:
        expected_scores = expected_scores / (expected_scores.sum(-1, True) + 1e-20)
    assert logits.dtype == scores.dtype == torch.float32
    assert torch.equal(indices, expected_indices)
    torch.testing.assert_close(logits, expected_logits, atol=0, rtol=0)
    torch.testing.assert_close(scores, expected_scores, atol=0, rtol=0)
    coefficients = torch.randn(scores.shape, generator=generator)
    (scores * coefficients).sum().backward()
    (expected_scores * coefficients).sum().backward()
    for actual, expected in ((x.grad, xr.grad), (parameter.grad, wr.grad)):
        assert actual is not None and expected is not None
        assert bool(actual.isfinite().all()) and bool(actual.count_nonzero())
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_explicit_router_dtype_rejects_unsupported_family():
    model = nn.Module()
    model.config = SimpleNamespace(model_type="llama", router_compute_dtype="float32")
    with pytest.raises(ValueError, match="supports qwen3_moe"):
        stabilize_moe_router(model)
