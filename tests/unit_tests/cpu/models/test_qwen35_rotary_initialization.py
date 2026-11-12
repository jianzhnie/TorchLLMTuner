"""Explicit model CPU initialization lifecycle contracts, not full35B/NPU acceptance.

Small dense/MoE geometries bound CPU memory; real rotary classes use partial
rotary dim64. One-ULP frequency perturbations represent materialized-device
values that CPU reinitialization must not overwrite. No forward math is mocked.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.distributed.checkpoint as dcp
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from llmtuner.memory.materialization import materialize_meta_model
from llmtuner.models.configuration import resolve_model_class
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.models.qwen.qwen3_5_moe.model import Qwen3_5MoeForCausalLM

FAMILIES = [
    pytest.param(Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM, id="moe"),
]


def text_config(config_type, *, dynamic=False):
    rope = {
        "rope_type": "dynamic" if dynamic else "default",
        "rope_theta": 10_000_000.0,
        "partial_rotary_factor": 0.5,
        "mrope_section": [11, 11, 10],
    }
    if dynamic:
        rope["factor"] = 2.0
    return config_type(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        layer_types=["linear_attention"],
        rope_parameters=rope,
    )


def native_initialization_reference(model):
    """Reference traversal through the model-owned leaf initialization rules."""
    model.apply(model._init_weights)


def distinct_frequencies(rotary):
    frequencies = rotary.inv_freq.detach().clone()
    return torch.nextafter(frequencies, torch.full_like(frequencies, float("inf")))


def assert_parameter_contract(left, right):
    actual, expected = dict(left.named_parameters()), dict(right.named_parameters())
    assert actual.keys() == expected.keys()
    for name, parameter in actual.items():
        assert torch.isfinite(parameter).all(), name
        assert torch.equal(parameter, expected[name]), name
        assert parameter.dtype == expected[name].dtype
    assert any(bool(parameter.count_nonzero()) for parameter in actual.values())


@pytest.mark.parametrize("config_type,model_type", FAMILIES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_default_dtype_initialization_matches_parameters_and_rng(
    config_type, model_type, dtype
):
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.random.fork_rng(devices=[]):
            model = model_type(text_config(config_type))
            reference = copy.deepcopy(model)
            torch.manual_seed(919)
            native_initialization_reference(reference)
            expected_rng = torch.get_rng_state().clone()
            torch.manual_seed(919)
            model.initialize_parameters()
            assert_parameter_contract(model, reference)
            assert torch.equal(torch.get_rng_state(), expected_rng)
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize("config_type,model_type", FAMILIES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shared", [False, True], ids=["independent", "shared"])
def test_random_initialization_preserves_materialized_frequencies_and_rng(
    config_type, model_type, dtype, shared
):
    with torch.random.fork_rng(devices=[]):
        model = model_type(text_config(config_type))
        for parameter in model.parameters():
            parameter.data = parameter.data.to(dtype)
        rotary = model.model.rotary_emb
        materialized = distinct_frequencies(rotary)
        rotary.inv_freq.copy_(materialized)
        if shared:
            rotary.original_inv_freq = rotary.inv_freq
        else:
            rotary.original_inv_freq.copy_(materialized)
        model.register_buffer("runtime_alias", rotary.inv_freq, persistent=False)
        model.register_buffer("unknown_buffer", torch.tensor([37.0]), persistent=False)
        aliases = {
            name: id(value)
            for name, value in model.named_buffers(remove_duplicate=False)
        }
        parameter_ids = {name: id(value) for name, value in model.named_parameters()}
        state_keys = tuple(model.state_dict())
        reference = copy.deepcopy(model)
        torch.manual_seed(909)
        native_initialization_reference(reference)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(909)
        model.initialize_parameters()
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert_parameter_contract(model, reference)
        assert parameter_ids == {
            name: id(value) for name, value in model.named_parameters()
        }
        assert aliases == {
            name: id(value)
            for name, value in model.named_buffers(remove_duplicate=False)
        }
        assert tuple(model.state_dict()) == state_keys
        assert torch.equal(rotary.inv_freq, materialized)
        assert torch.equal(rotary.original_inv_freq, materialized)
        assert model.runtime_alias is rotary.inv_freq
        assert (rotary.inv_freq is rotary.original_inv_freq) == shared
        assert torch.equal(model.unknown_buffer, reference.unknown_buffer)
        assert not torch.equal(reference.model.rotary_emb.inv_freq, materialized)
        repeated = copy.deepcopy(model)
        torch.manual_seed(909)
        repeated.initialize_parameters()
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert_parameter_contract(model, repeated)


@pytest.mark.parametrize("config_type,model_type", FAMILIES)
def test_dynamic_rotary_keeps_native_initialization(config_type, model_type):
    with torch.random.fork_rng(devices=[]):
        model = model_type(text_config(config_type, dynamic=True))
        rotary = model.model.rotary_emb
        previous = distinct_frequencies(rotary)
        rotary.inv_freq.copy_(previous)
        rotary.original_inv_freq.copy_(previous)
        reference = copy.deepcopy(model)
        torch.manual_seed(918)
        native_initialization_reference(reference)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(918)
        model.initialize_parameters()
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert_parameter_contract(model, reference)
        assert torch.equal(rotary.inv_freq, reference.model.rotary_emb.inv_freq)
        assert not torch.equal(rotary.inv_freq, previous)


def test_other_native_architecture_keeps_original_rotary_initialization():
    with torch.random.fork_rng(devices=[]):
        model = Qwen3MoeForCausalLM(
            Qwen3MoeConfig(
                vocab_size=32,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=64,
                num_experts=2,
                num_experts_per_tok=1,
                moe_intermediate_size=16,
            )
        )
        rotary = model.model.rotary_emb
        previous = distinct_frequencies(rotary)
        rotary.inv_freq.copy_(previous)
        rotary.original_inv_freq.copy_(previous)
        reference = copy.deepcopy(model)
        torch.manual_seed(917)
        native_initialization_reference(reference)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(917)
        model.initialize_parameters()
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert_parameter_contract(model, reference)
        assert torch.equal(rotary.inv_freq, reference.model.rotary_emb.inv_freq)
        assert not torch.equal(rotary.inv_freq, previous)


@pytest.mark.parametrize("config_type,model_type", FAMILIES)
def test_meta_materialization_and_native_resume_preserve_runtime_aliases(
    config_type, model_type, tmp_path
):
    with torch.random.fork_rng(devices=[]):
        with torch.device("meta"):
            model = model_type(text_config(config_type))
        model.register_buffer(
            "runtime_alias", model.model.rotary_emb.inv_freq, persistent=False
        )
        rng = torch.get_rng_state().clone()
        materialize_meta_model(
            model,
            torch.device("cpu"),
            runtime_buffers=model.materialization_buffers(torch.device("cpu")),
        )
        assert torch.equal(torch.get_rng_state(), rng)
        assert all(not value.is_meta for value in model.parameters())
        assert all(not value.is_meta for value in model.buffers())
        assert model.runtime_alias is model.model.rotary_emb.inv_freq
        frequencies = distinct_frequencies(model.model.rotary_emb)
        model.model.rotary_emb.inv_freq.copy_(frequencies)
        model.model.rotary_emb.original_inv_freq.copy_(frequencies)
        model.initialize_parameters()
        state = model.state_dict()
        assert not any("inv_freq" in name or "runtime_alias" in name for name in state)
        dcp.save({"model": state}, checkpoint_id=tmp_path / "step-50")
        with torch.device("meta"):
            restored = model_type(text_config(config_type))
        materialize_meta_model(
            restored,
            torch.device("cpu"),
            runtime_buffers=restored.materialization_buffers(torch.device("cpu")),
        )
        restored.model.rotary_emb.inv_freq.copy_(frequencies)
        restored.model.rotary_emb.original_inv_freq.copy_(frequencies)
        restored.initialize_parameters()
        destination = {"model": restored.state_dict()}
        dcp.load(destination, checkpoint_id=tmp_path / "step-50")
        restored.load_state_dict(destination["model"], strict=True)
        assert_parameter_contract(model, restored)
        assert torch.equal(restored.model.rotary_emb.inv_freq, frequencies)
        assert torch.equal(restored.model.rotary_emb.original_inv_freq, frequencies)
        assert model.runtime_alias is model.model.rotary_emb.inv_freq


@pytest.mark.parametrize("config_type,model_type", FAMILIES)
def test_native_hf_checkpoint_loading_preserves_parameter_values(
    config_type, model_type, tmp_path
):
    with torch.random.fork_rng(devices=[]):
        model = model_type(text_config(config_type))
        model.initialize_parameters()
        model.save_pretrained(tmp_path / "hf")
        restored = model_type.from_pretrained(tmp_path / "hf", local_files_only=True)
        assert_parameter_contract(model, restored)
        assert torch.isfinite(restored.model.rotary_emb.inv_freq).all()
        assert torch.isfinite(restored.model.rotary_emb.original_inv_freq).all()


def test_unregistered_dense_text_model_has_no_automatic_hf_fallback():
    with pytest.raises(ValueError, match="No local model registered"):
        resolve_model_class(text_config(Qwen3_5TextConfig))
