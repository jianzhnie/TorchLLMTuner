"""Configuration construction stays independent from removed training adapters."""

from copy import deepcopy

import pytest
from transformers import AutoConfig, Qwen3MoeConfig

from llmtuner.config.llmtuner_config import LLMTunerConfig
from llmtuner.config.model_config import ModelConfig
from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models import registry
from llmtuner.models.configuration import (
    apply_runtime_config,
    build_model_config,
    build_model_config_for,
    unwrap_text_config,
)
from llmtuner.models.explicit import build_explicit_model


def test_saved_configuration_uses_registry_without_global_mutation(
    tmp_path, monkeypatch
):
    def reject(*args, **kwargs):
        raise AssertionError("Configuration loading must not use AutoConfig")

    monkeypatch.setattr(AutoConfig, "register", reject)
    monkeypatch.setattr(AutoConfig, "from_pretrained", reject)
    config = build_model_config(
        "qwen3_moe",
        seq_len=32,
        arch_overrides={"hidden_size": 16, "num_hidden_layers": 1},
    )
    config.save_pretrained(tmp_path)
    loaded = build_model_config(str(tmp_path), seq_len=64)
    assert isinstance(loaded, Qwen3MoeConfig)
    assert loaded.hidden_size == 16
    assert loaded.num_hidden_layers == 1
    assert loaded.max_position_embeddings >= 64


def test_unregistered_offline_configuration_is_rejected():
    with pytest.raises(ValueError, match="No local configuration registered"):
        build_model_config("llama", seq_len=32, arch_overrides={})


def test_run_configuration_preserves_overrides_and_sets_input_policy():
    overrides = {"hidden_size": 16, "num_experts": 4, "moe_intermediate_size": 8}
    cfg = LLMTunerConfig(model=ModelConfig(arch_overrides=overrides))
    config = build_model_config_for(cfg)
    assert config.hidden_size == 16
    assert config.num_experts == 4
    assert overrides == cfg.arch_overrides
    assert config.attn_mask_type == cfg.dataloader.attention_mask_type()


def test_text_config_unwrap_preserves_backend_choice_and_external_names():
    outer = Qwen3MoeConfig()
    outer.text_config = Qwen3MoeConfig(architectures=None)
    outer.text_config._attn_implementation = "sdpa"
    outer.router_compute_dtype = "float32"
    inner = unwrap_text_config(outer)
    assert inner is outer.text_config
    assert inner._attn_implementation == "sdpa"
    assert inner.architectures is None
    assert inner.router_compute_dtype == "float32"


def test_registered_qwen_is_available_through_explicit_construction():
    config = build_model_config(
        "qwen3_moe",
        seq_len=16,
        arch_overrides={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "moe_intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "num_experts": 4,
            "num_experts_per_tok": 2,
        },
    )
    model = build_explicit_model(config)
    assert type(model).__module__ == "llmtuner.models.qwen.qwen3.model"
    assert model.decoder_structure.bind(model).blocks is model.model.layers


@pytest.mark.parametrize("backend", ["flex", "eager_flex"])
@pytest.mark.parametrize("composite", [False, True])
def test_rejected_runtime_backend_does_not_partially_mutate_config(backend, composite):
    config = Qwen3MoeConfig()
    if composite:
        config.text_config = Qwen3MoeConfig()
    config.norm_backend = "saved-native-policy"
    config.attn_mask_type = "saved-mask-policy"
    before = deepcopy(config.to_dict())
    cfg = LLMTunerConfig()
    cfg.model.attention_backend = backend
    with pytest.raises(UnsupportedCombinationError, match="attention_backend"):
        apply_runtime_config(config, cfg)
    assert config.to_dict() == before


def test_offline_registered_config_does_not_require_hf_attribute_map(monkeypatch):
    class LocalConfig:
        model_type = "test_local_plain_config"

        def __init__(self, **values):
            self.__dict__.update(values)

    def forbidden_model_loader():
        raise AssertionError("Configuration construction imported model execution")

    monkeypatch.setattr(registry, "_MODELS", dict(registry._MODELS))
    registry.register_model(
        registry.ModelSpec.from_types(
            LocalConfig.model_type,
            config_type=LocalConfig,
            model_class_loader=forbidden_model_loader,
            explicit=True,
        )
    )
    cfg = LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path=LocalConfig.model_type,
            arch_overrides={"hidden_size": 16},
        )
    )
    result = build_model_config_for(cfg)
    assert isinstance(result, LocalConfig)
    assert result.hidden_size == 16
    assert result.max_position_embeddings >= cfg.max_seq_len
    assert result.attn_mask_type == cfg.dataloader.attention_mask_type()
    assert not hasattr(result, "attribute_map")
