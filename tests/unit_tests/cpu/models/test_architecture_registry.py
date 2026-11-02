"""Custom architectures use one registration seam without model imports."""

import pytest
from transformers import PretrainedConfig, PreTrainedModel

from llmtuner.models import architecture
from llmtuner.models.architecture import (
    ModelArchitecture,
    get_model_architecture,
    register_model_architecture,
)
from llmtuner.models.hf.factory import build_model_config, resolve_model_class


class _CustomConfig(PretrainedConfig):
    model_type = "llmtuner_test_custom"


class _CustomModel(PreTrainedModel):
    config_class = _CustomConfig


def test_custom_architecture_builds_config_and_lazily_resolves_model(tmp_path):
    registration = ModelArchitecture.from_types(
        _CustomConfig.model_type,
        config_type=_CustomConfig,
        model_class_loader=lambda: _CustomModel,
    )
    register_model_architecture(registration)
    try:
        config = build_model_config(
            _CustomConfig.model_type,
            seq_len=32,
            arch_overrides={"hidden_size": 16},
        )
        assert isinstance(config, _CustomConfig)
        assert config.hidden_size == 16
        assert config.max_position_embeddings == 32
        assert resolve_model_class(config) is _CustomModel
        assert get_model_architecture(_CustomConfig.model_type) is registration

        config.save_pretrained(tmp_path)
        restored = build_model_config(str(tmp_path), seq_len=64)
        assert isinstance(restored, _CustomConfig)
        assert restored.hidden_size == 16
        assert restored.max_position_embeddings == 64
    finally:
        architecture._ARCHITECTURES.pop(_CustomConfig.model_type)


def test_duplicate_custom_architecture_requires_explicit_replacement():
    registration = ModelArchitecture.from_types(
        _CustomConfig.model_type,
        config_type=_CustomConfig,
        model_class_loader=lambda: _CustomModel,
    )
    register_model_architecture(registration)
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_model_architecture(registration)
        register_model_architecture(registration, exist_ok=True)
    finally:
        architecture._ARCHITECTURES.pop(_CustomConfig.model_type)
