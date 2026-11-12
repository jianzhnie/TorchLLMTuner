"""Explicit Qwen3.5 FP32 routing preserves storage and checkpoint contracts."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeTopKRouter as HFQwen35TopKRouter,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeTopKRouter as HFQwen3MoeTopKRouter,
)

from llmtuner.config.model_config import ModelConfig
from llmtuner.errors import ConfigError
from llmtuner.models.configuration import apply_runtime_config, unwrap_text_config
from llmtuner.models.qwen.qwen3.model import (
    Qwen3MoeForCausalLM,
    Qwen3MoeSparseMoeBlock,
    Qwen3MoeTopKRouter,
)
from llmtuner.models.qwen.qwen3_5_moe.model import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeSparseMoeBlock,
)
from llmtuner.models.qwen.routing import (
    Qwen3_5MoeFP32TopKRouter,
    Qwen3_5MoeTopKRouter,
    Qwen3MoeFP32TopKRouter,
)
from tests.unit_tests.cpu.models.test_owned_family_construction import text_config
from tests.unit_tests.cpu.models.test_qwen3_model_contract import model_config


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
    block_type = (
        Qwen3MoeSparseMoeBlock if family == "qwen3_moe" else Qwen3_5MoeSparseMoeBlock
    )
    block = block_type(config).to(torch.bfloat16)
    router = block.gate
    native_type = Qwen3MoeTopKRouter if family == "qwen3_moe" else Qwen3_5MoeTopKRouter
    fp32_type = (
        Qwen3MoeFP32TopKRouter if family == "qwen3_moe" else Qwen3_5MoeFP32TopKRouter
    )
    assert type(router) is (fp32_type if setting == "float32" else native_type)
    generator = torch.Generator().manual_seed(41)
    with torch.no_grad():
        # Direct block construction does not run the owning model's post_init.
        # Initialize experts too: uninitialized storage can contain NaNs and
        # cannot establish an exact before/after checkpoint-state invariant.
        for value in block.parameters():
            value.copy_(torch.randn(value.shape, generator=generator) * 0.2)
            assert bool(value.isfinite().all())
    parameter = router.weight
    before = {key: value.clone() for key, value in block.state_dict().items()}
    x = torch.randn((7, 32), generator=generator).bfloat16().requires_grad_()
    xr = x.detach().clone().requires_grad_()
    wr = parameter.detach().clone().requires_grad_()
    # An outer autocast must not undo the explicit FP32 gate contract.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits, scores, indices = router(x)
    if setting is None:
        reference_type = (
            HFQwen3MoeTopKRouter if family == "qwen3_moe" else HFQwen35TopKRouter
        )
        baseline = reference_type(config).to(torch.bfloat16)
        baseline.load_state_dict(router.state_dict())
        with torch.autocast("cpu", dtype=torch.bfloat16):
            expected = baseline(xr)
        torch.testing.assert_close((logits, scores, indices), expected, atol=0, rtol=0)
        coefficients = torch.randn(scores.shape, generator=generator)
        (scores * coefficients).sum().backward()
        (expected[1] * coefficients).sum().backward()
        torch.testing.assert_close(x.grad, xr.grad, atol=0, rtol=0)
        torch.testing.assert_close(parameter.grad, baseline.weight.grad, atol=0, rtol=0)
        assert router.weight is parameter
        for key, value in before.items():
            torch.testing.assert_close(block.state_dict()[key], value, rtol=0, atol=0)
        return
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


@pytest.mark.parametrize("family", ["qwen3_moe", "qwen3_5_moe_text"])
@pytest.mark.parametrize("setting", ["bfloat16", "auto", False, []])
def test_invalid_router_policy_rejected_before_allocation(family, setting):
    config = AutoConfig.for_model(family)
    config.router_compute_dtype = setting
    constructor = (
        Qwen3MoeForCausalLM if family == "qwen3_moe" else Qwen3_5MoeForCausalLM
    )
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="router_compute_dtype"):
        constructor(config)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("family", ["qwen3", "qwen35"])
def test_router_policy_preserves_model_initialization_and_state(family):
    config = (
        model_config()
        if family == "qwen3"
        else text_config(Qwen3_5MoeForCausalLM.config_class)
    )
    constructor = Qwen3MoeForCausalLM if family == "qwen3" else Qwen3_5MoeForCausalLM
    # This CPU router test uses eager experts: the tiny BF16 expert width is
    # not aligned for the reference grouped-MM kernel's 16-byte stride rule.
    config._experts_implementation = "eager"
    config.router_compute_dtype = None
    torch.manual_seed(61)
    baseline = constructor(deepcopy(config))
    rng = torch.get_rng_state().clone()
    config.router_compute_dtype = "float32"
    torch.manual_seed(61)
    actual = constructor(config)
    assert torch.equal(torch.get_rng_state(), rng)
    assert actual.state_dict().keys() == baseline.state_dict().keys()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], value, atol=0, rtol=0)
    native_gate = baseline.layers[0].mlp.gate
    actual_gate = actual.layers[0].mlp.gate
    assert native_gate.weight.dtype == actual_gate.weight.dtype
    actual.to(torch.bfloat16)
    # Exercise the real model call so policy selection cannot pass by only
    # constructing a standalone router that the decoder never invokes.
    calls = []
    handle = actual_gate.register_forward_hook(
        lambda module, args, output: calls.append(output)
    )
    try:
        result = actual(torch.tensor([[1, 3, 7, 2]])).logits
    finally:
        handle.remove()
    assert result.isfinite().all() and result.count_nonzero() > 0
    assert len(calls) == 1 and calls[0][0].dtype == torch.float32
    assert calls[0][1].dtype == torch.float32
    result.square().sum().backward()
    assert (
        actual_gate.weight.grad is not None
        and actual_gate.weight.grad.count_nonzero() > 0
    )
    assert actual_gate.weight.grad.dtype == torch.bfloat16


def test_native_qwen3_router_does_not_call_reference(monkeypatch):
    """Local construction/execution must survive removal of reference methods."""
    config = AutoConfig.for_model(
        "qwen3_moe", hidden_size=16, num_experts=4, num_experts_per_tok=2
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("reference router execution")

    monkeypatch.setattr(HFQwen3MoeTopKRouter, "__init__", forbidden)
    monkeypatch.setattr(HFQwen3MoeTopKRouter, "forward", forbidden)
    router = Qwen3MoeTopKRouter(config)
    generator = torch.Generator().manual_seed(37)
    with torch.no_grad():
        router.weight.copy_(torch.randn(4, 16, generator=generator))
    tokens = torch.randn(2, 3, 16, generator=generator, requires_grad=True)
    logits, scores, indices = router(tokens)
    assert logits.shape == (6, 4)
    assert indices.shape == scores.shape == (6, 2)
    assert torch.all(scores[:, 0] >= scores[:, 1])
    (scores * torch.tensor([1.0, 3.0])).sum().backward()
    assert torch.count_nonzero(tokens.grad) > 0
    assert torch.count_nonzero(router.weight.grad) > 0
