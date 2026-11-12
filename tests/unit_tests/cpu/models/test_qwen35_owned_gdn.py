"""Model-owned GDN preserves native initialization, gradients and cache modes."""

import pytest
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as reference_kernels
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedDeltaNet as ReferenceDelta,
)

from llmtuner.models.qwen.qwen3_5_moe.gated_delta import Qwen3_5MoeGatedDeltaNet
from llmtuner.models.qwen.qwen3_5_moe.model import Qwen3_5MoeForCausalLM
from tests.unit_tests.cpu.models.test_owned_family_construction import text_config
from tests.unit_tests.cpu.models.test_qwen35_owned_decoder import models


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_owned_gdn_init_forward_vjp_and_update_match_reference(monkeypatch, dtype):
    config = text_config(Qwen3_5MoeForCausalLM.config_class)
    torch.manual_seed(149)
    expected = ReferenceDelta(config, 0).to(dtype)
    expected_rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        pytest.fail("Owned GDN delegated its construction or forward")

    monkeypatch.setattr(ReferenceDelta, "__init__", forbidden)
    torch.manual_seed(149)
    actual = Qwen3_5MoeGatedDeltaNet(config, 0).to(dtype)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[name], rtol=0, atol=0)
    hidden = torch.randn(5, 2, config.hidden_size, dtype=dtype).transpose(0, 1)
    hidden.requires_grad_()
    other = hidden.detach().clone().requires_grad_()
    reference = expected(other)
    monkeypatch.setattr(ReferenceDelta, "forward", forbidden)
    observed = actual(hidden)
    torch.testing.assert_close(observed, reference, rtol=0, atol=0)
    upstream = torch.randn_like(reference)
    reference.backward(upstream)
    observed.backward(upstream)
    torch.testing.assert_close(hidden.grad, other.grad, rtol=0, atol=0)
    assert hidden.grad.isfinite().all() and hidden.grad.count_nonzero() > 0
    for name, parameter in actual.named_parameters():
        torch.testing.assert_close(
            parameter.grad, dict(expected.named_parameters())[name].grad, rtol=0, atol=0
        )
    for model in (expected, actual):
        torch.optim.AdamW(model.parameters(), lr=1e-3, eps=1e-4, foreach=False).step()
    for name, parameter in actual.named_parameters():
        assert parameter.isfinite().all()
        torch.testing.assert_close(
            parameter, dict(expected.named_parameters())[name], rtol=0, atol=0
        )


def test_owned_gdn_cached_single_and_chunk_continuation_without_delegation(monkeypatch):
    baseline, owned = models()
    baseline.eval()
    owned.eval()
    batches = [torch.tensor([[1, 5, 2]]), torch.tensor([[7]]), torch.tensor([[3, 8]])]
    expected = []
    cache = None
    with torch.no_grad():
        for tokens in batches:
            output = baseline(tokens, use_cache=True, past_key_values=cache)
            cache = output.past_key_values
            expected.append(output.last_hidden_state.clone())

    def forbidden(*args, **kwargs):
        pytest.fail("Owned GDN cache path delegated to reference forward")

    monkeypatch.setattr(ReferenceDelta, "forward", forbidden)
    cache = None
    with torch.no_grad():
        for tokens, target in zip(batches, expected, strict=True):
            output = owned(tokens, use_cache=True, past_key_values=cache)
            cache = output.past_key_values
            torch.testing.assert_close(output.last_hidden_state, target, rtol=0, atol=0)
    assert cache.get_seq_length() == 6


@pytest.mark.parametrize("construction_device", ["cpu", "meta"])
def test_cpu_gdn_ignores_installed_cuda_only_extensions(
    monkeypatch, construction_device
):
    config = text_config(Qwen3_5MoeForCausalLM.config_class)
    config.layer_types = ["linear_attention"] * config.num_hidden_layers
    torch.manual_seed(157)
    baseline = Qwen3_5MoeGatedDeltaNet(config, 0)
    hidden = torch.randn(2, 6, config.hidden_size)
    expected = baseline(hidden)
    chunks = (hidden[:, :3], hidden[:, 3:4], hidden[:, 4:])
    expected_chunks = []
    baseline_cache = DynamicCache(config=config)
    with torch.no_grad():
        for chunk in chunks:
            expected_chunks.append(baseline(chunk, cache_params=baseline_cache).clone())

    def forbidden(*args, **kwargs):
        pytest.fail("CPU/meta GDN selected a CUDA-only extension")

    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    for name in (
        "FusedRMSNormGated",
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
    ):
        monkeypatch.setattr(reference_kernels, name, forbidden)
    with torch.device(construction_device):
        actual = Qwen3_5MoeGatedDeltaNet(config, 0)
    if construction_device == "meta":
        actual.to_empty(device="cpu")
    actual.load_state_dict(baseline.state_dict(), strict=True)
    observed = actual(hidden)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    observed.sum().backward()
    assert actual.in_proj_qkv.weight.grad.isfinite().all()
    assert actual.in_proj_qkv.weight.grad.count_nonzero() > 0
    actual_cache = DynamicCache(config=config)
    with torch.no_grad():
        for chunk, target in zip(chunks, expected_chunks, strict=True):
            result = actual(chunk, cache_params=actual_cache)
            torch.testing.assert_close(result, target, rtol=0, atol=0)
