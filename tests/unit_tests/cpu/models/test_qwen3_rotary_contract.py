"""Explicit Qwen3 precision policy preserves rotary backward cast ordering."""

from copy import deepcopy

import pytest
import torch

from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.models.qwen.qwen3.rotary import apply_fp32_rotary
from tests.unit_tests.cpu.models.test_qwen3_model_contract import model_config


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fp32_application_matches_shared_cast_reference(dtype):
    torch.manual_seed(23)
    query = torch.randn(2, 3, 4, 8, dtype=dtype, requires_grad=True)
    key = torch.randn(2, 1, 4, 8, dtype=dtype, requires_grad=True)
    angles = torch.randn(2, 4, 8)
    cosine, sine = angles.cos(), angles.sin()
    actual = apply_fp32_rotary(query, key, cosine, sine)
    expected = []
    for value in (query, key):
        promoted = value.float()
        left, right = promoted.chunk(2, -1)
        expected.append(
            (
                promoted * cosine[:, None]
                + torch.cat((-right, left), -1) * sine[:, None]
            ).to(dtype)
        )
    probes = tuple(torch.randn_like(value) for value in actual)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    observed = torch.autograd.grad(actual, (query, key), probes, retain_graph=True)
    reference = torch.autograd.grad(expected, (query, key), probes)
    for result, target in zip(observed, reference, strict=True):
        torch.testing.assert_close(result, target, rtol=0, atol=0)
        assert result.isfinite().all() and result.count_nonzero() > 0


@pytest.mark.parametrize("projection", ["separate", "fused"])
@pytest.mark.parametrize("cache", [None, "float32", "bfloat16"])
def test_model_precision_executes_without_changing_initialization(projection, cache):
    config = model_config()
    config._experts_implementation = "eager"
    torch.manual_seed(13)
    baseline = Qwen3MoeForCausalLM(deepcopy(config))
    rng = torch.get_rng_state().clone()
    config.rotary_compute_dtype = "float32"
    config.rotary_cache_dtype = cache
    config.qkv_projection = projection
    torch.manual_seed(13)
    actual = Qwen3MoeForCausalLM(config)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in actual.state_dict().items():
        assert torch.equal(value, baseline.state_dict()[name])
    actual = actual.to(torch.bfloat16)
    captures = []
    hook = actual.model.rotary_emb.register_forward_hook(
        lambda module, args, output: captures.append(output)
    )
    try:
        tokens = torch.tensor([[1, 3, 7], [2, 4, 8]])
        output = actual(tokens, labels=tokens)
        output.loss.backward()
    finally:
        hook.remove()
    assert captures
    expected_dtype = torch.bfloat16 if cache == "bfloat16" else torch.float32
    assert captures[0][0].dtype == expected_dtype
    assert output.loss.isfinite()
    gradient = actual.model.layers[0].self_attn.q_proj.weight.grad
    assert gradient.isfinite().all() and gradient.count_nonzero() > 0


def test_fp32_rotary_meta_buffer_contract():
    config = model_config()
    config.rotary_compute_dtype = "float32"
    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(config)
    replacements = model.materialization_buffers(torch.device("cpu"))
    assert replacements
    for (owner, name), value in replacements.items():
        assert owner is model.model.rotary_emb
        assert value.device.type == "cpu" and value.isfinite().all()
        assert value.shape == owner._buffers[name].shape
