"""Explicit Qwen3 dense/MoE RoPE reconstruction and meta storage contracts."""

from copy import deepcopy

import pytest
import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

from llmtuner.memory.materialization import materialize_meta_model
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from tests.unit_tests.cpu.models.test_qwen3_model_contract import model_config


@pytest.mark.parametrize("dense", [False, True])
def test_target_rotary_policy_rejected_before_buffer_reconstruction(dense):
    config = model_config(mlp_only_layers=[0, 1] if dense else [])
    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(config)
    model.config.rotary_init_device = "target"
    parameters = tuple(model.parameters())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="constructor RoPE"):
        model.materialization_buffers(torch.device("cpu"))
    assert all(p.is_meta for p in parameters)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_meta_reconstruction_matches_constructor_buffers_and_caches(dense, dtype):
    config = model_config(mlp_only_layers=[0, 1] if dense else [])
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        reference = Qwen3MoeRotaryEmbedding(deepcopy(config))
        with torch.device("meta"):
            model = Qwen3MoeForCausalLM(config)
        rotary = model.model.rotary_emb
        model.register_buffer("frequency_alias", rotary.inv_freq, persistent=False)
        parameters = {name: id(p) for name, p in model.named_parameters()}
        keys = tuple(model.state_dict())
        rng = torch.get_rng_state().clone()
        buffers = model.materialization_buffers(torch.device("cpu"))
        assert torch.equal(torch.get_rng_state(), rng)
        materialize_meta_model(model, torch.device("cpu"), runtime_buffers=buffers)
        assert parameters == {name: id(p) for name, p in model.named_parameters()}
        assert tuple(model.state_dict()) == keys
        assert not any("inv_freq" in key or "frequency_alias" in key for key in keys)
        assert model.frequency_alias is rotary.inv_freq
        assert rotary.inv_freq.dtype == torch.float32
        torch.testing.assert_close(rotary.inv_freq, reference.inv_freq, rtol=0, atol=0)
        value = torch.zeros(1, 128, config.head_dim, dtype=dtype)
        positions = torch.arange(128)[None]
        actual_cache = rotary(value, positions)
        expected_cache = reference(value, positions)
        torch.testing.assert_close(actual_cache, expected_cache, rtol=0, atol=0)
        assert actual_cache[0].isfinite().all() and actual_cache[0].count_nonzero() > 0
        model.initialize_parameters()
        torch.testing.assert_close(rotary.inv_freq, reference.inv_freq, rtol=0, atol=0)
        output = model(torch.tensor([[1, 3, 7, 2]])).logits
        assert output.isfinite().all() and output.count_nonzero() > 0
    finally:
        torch.set_default_dtype(previous)
