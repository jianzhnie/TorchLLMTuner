"""Real HF call-chain metadata contracts; abort before any GDN computation."""

from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.qwen3_5.gdn_numerics import _validate_forward
from llmtuner.models.qwen3_5.linear_attention import validate_gdn_inputs


class CapturedGDN(Exception):
    pass


@pytest.mark.parametrize("causal", [None, True])
def test_native_causallm_metadata_reaches_both_guards(causal):
    config = AutoConfig.for_model(
        "qwen3_5_moe_text",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
    )
    if causal is not None:
        config.is_causal = causal
    model = AutoModelForCausalLM.from_config(config).bfloat16()
    captured = {}

    def capture(module, args, kwargs):
        captured.update(kwargs)
        raise CapturedGDN

    handle = model.model.layers[0].linear_attn.register_forward_pre_hook(
        capture, with_kwargs=True
    )
    try:
        with pytest.raises(CapturedGDN):
            model(input_ids=torch.zeros(1, 64, dtype=torch.long), use_cache=False)
    finally:
        handle.remove()
    expected = {
        "hidden_states",
        "cache_params",
        "attention_mask",
        "use_cache",
        "output_router_logits",
    }
    if causal is not None:
        expected.add("is_causal")
    assert set(captured) == expected
    assert captured["output_router_logits"] is False
    assert captured.get("is_causal") is causal
    assert captured["cache_params"] is None
    assert captured["attention_mask"] is None
    # Keep the genuine metadata but use production geometry so CPU device is
    # the only expected failure. This is guard coverage, not NPU acceptance.
    captured["hidden_states"] = torch.empty(1, 64, 2048, dtype=torch.bfloat16)
    for guard in (_validate_forward, validate_gdn_inputs):
        with pytest.raises(UnsupportedCombinationError, match="BF16 NPU"):
            guard(SimpleNamespace(), (), captured)


@pytest.mark.parametrize("guard", [_validate_forward, validate_gdn_inputs])
@pytest.mark.parametrize(
    "extra",
    [
        {"output_router_logits": 0},
        {"output_attentions": None},
        {"output_hidden_states": "false"},
        {"is_causal": False},
        {"is_causal": 1},
        {"unknown_metadata": False},
        {"cache_params": object()},
        {"cu_seqlens": torch.tensor([0, 64])},
    ],
)
def test_metadata_validation_remains_narrow(guard, extra):
    kwargs = {"hidden_states": torch.empty(1, 64, 2048), **extra}
    with pytest.raises(UnsupportedCombinationError):
        guard(SimpleNamespace(), (), kwargs)
