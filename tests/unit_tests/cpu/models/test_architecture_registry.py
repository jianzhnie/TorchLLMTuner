"""Custom architectures use one registration seam without model imports."""

import pytest
from transformers import PretrainedConfig, PreTrainedModel
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from llmtuner.models import architecture
from llmtuner.models.architecture import (
    ModelArchitecture,
    get_model_architecture,
    register_model_architecture,
)
from llmtuner.models.hf.factory import build_model_config, resolve_model_class


@pytest.fixture(autouse=True)
def _isolate_hf_config_registry(monkeypatch):
    monkeypatch.setattr(
        CONFIG_MAPPING, "_extra_content", dict(CONFIG_MAPPING._extra_content)
    )


class _CustomConfig(PretrainedConfig):
    model_type = "llmtuner_test_custom"


class _CustomModel(PreTrainedModel):
    config_class = _CustomConfig


@pytest.mark.parametrize("model_type", [None, 1, True, "", "  ", "\t\n"])
def test_invalid_architecture_name_does_not_change_registry(model_type):
    previous = dict(architecture._ARCHITECTURES)
    registration = ModelArchitecture.from_types(
        model_type,
        config_type=_CustomConfig,
        model_class_loader=lambda: _CustomModel,
    )
    with pytest.raises(ValueError, match="non-empty string"):
        register_model_architecture(registration)
    assert architecture._ARCHITECTURES == previous


@pytest.mark.parametrize(
    "field", ["config_factory", "config_type_loader", "model_class_loader"]
)
def test_invalid_loader_cannot_replace_existing_architecture(field):
    from dataclasses import replace

    previous = architecture.get_model_architecture("kimi_linear")
    invalid = replace(previous, **{field: None})
    with pytest.raises(TypeError, match=field):
        register_model_architecture(invalid, exist_ok=True)
    assert architecture.get_model_architecture("kimi_linear") is previous


def test_registration_does_not_invoke_construction_hooks(monkeypatch):
    def unexpected():
        pytest.fail("Registration must leave config and model loading lazy")

    monkeypatch.setattr(architecture, "_ARCHITECTURES", {})
    registration = ModelArchitecture("lazy_test", unexpected, unexpected, unexpected)
    register_model_architecture(registration)
    assert get_model_architecture("lazy_test") is registration


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


def test_saved_custom_config_without_position_limit_matches_offline_construction(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(architecture, "_ARCHITECTURES", {})
    register_model_architecture(
        ModelArchitecture.from_types(
            _CustomConfig.model_type,
            config_type=_CustomConfig,
            model_class_loader=lambda: _CustomModel,
        )
    )
    _CustomConfig(hidden_size=16).save_pretrained(tmp_path)

    restored = build_model_config(str(tmp_path), seq_len=64)
    offline = build_model_config(
        _CustomConfig.model_type, seq_len=64, arch_overrides={"hidden_size": 16}
    )

    assert type(restored) is type(offline)
    assert restored.hidden_size == offline.hidden_size == 16
    assert restored.max_position_embeddings == offline.max_position_embeddings == 64


@pytest.mark.parametrize("custom", [False, True])
def test_unrelated_optional_architecture_does_not_block_saved_hf_config(
    monkeypatch, tmp_path, custom
):
    from transformers import LlamaConfig

    def unavailable_config():
        raise ImportError("optional custom configuration dependency is absent")

    optional = ModelArchitecture(
        "unavailable_optional_model",
        unavailable_config,
        unavailable_config,
        unavailable_config,
    )
    monkeypatch.setattr(architecture, "_ARCHITECTURES", {})
    register_model_architecture(optional)
    if custom:
        register_model_architecture(
            ModelArchitecture.from_types(
                _CustomConfig.model_type,
                config_type=_CustomConfig,
                model_class_loader=lambda: _CustomModel,
            )
        )
        config_type = _CustomConfig
    else:
        config_type = LlamaConfig
    config_type(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    ).save_pretrained(tmp_path)

    restored = build_model_config(str(tmp_path), seq_len=64)

    assert isinstance(restored, config_type)
    assert restored.hidden_size == 16
    assert restored.max_position_embeddings == 64


def test_saved_config_uses_replaced_architecture_loader(monkeypatch, tmp_path):
    class ReplacementConfig(_CustomConfig):
        pass

    monkeypatch.setattr(architecture, "_ARCHITECTURES", {})
    _CustomConfig(hidden_size=16, max_position_embeddings=32).save_pretrained(tmp_path)
    for config_type in (_CustomConfig, ReplacementConfig):
        register_model_architecture(
            ModelArchitecture.from_types(
                _CustomConfig.model_type,
                config_type=config_type,
                model_class_loader=lambda: _CustomModel,
            ),
            exist_ok=True,
        )
        restored = build_model_config(str(tmp_path), seq_len=64)
        assert type(restored) is config_type
        assert restored.hidden_size == 16
        assert restored.max_position_embeddings == 64


def test_replacement_does_not_construct_stale_hf_config(monkeypatch, tmp_path):
    class StaleConfig(_CustomConfig):
        reject_construction = False

        def __init__(self, **kwargs):
            if self.reject_construction:
                raise ImportError("stale architecture dependency is no longer present")
            super().__init__(**kwargs)

    monkeypatch.setattr(architecture, "_ARCHITECTURES", {})
    _CustomConfig(hidden_size=16, max_position_embeddings=32).save_pretrained(tmp_path)
    register_model_architecture(
        ModelArchitecture.from_types(
            _CustomConfig.model_type,
            config_type=StaleConfig,
            model_class_loader=lambda: _CustomModel,
        )
    )
    assert isinstance(build_model_config(str(tmp_path), seq_len=32), StaleConfig)
    StaleConfig.reject_construction = True
    register_model_architecture(
        ModelArchitecture.from_types(
            _CustomConfig.model_type,
            config_type=_CustomConfig,
            model_class_loader=lambda: _CustomModel,
        ),
        exist_ok=True,
    )

    restored = build_model_config(str(tmp_path), seq_len=64)

    assert type(restored) is _CustomConfig
    assert restored.hidden_size == 16
    assert restored.max_position_embeddings == 64


def test_selected_optional_architecture_preserves_dependency_failure(
    monkeypatch, tmp_path
):
    class OptionalConfig(PretrainedConfig):
        model_type = "llmtuner_test_missing_dependency"

    failure = ImportError("selected architecture dependency is absent")

    def unavailable():
        raise failure

    monkeypatch.setattr(architecture, "_ARCHITECTURES", {})
    register_model_architecture(
        ModelArchitecture(
            OptionalConfig.model_type, unavailable, unavailable, unavailable
        )
    )
    OptionalConfig(max_position_embeddings=32).save_pretrained(tmp_path)
    with pytest.raises(ImportError) as caught:
        build_model_config(str(tmp_path), seq_len=64)
    assert caught.value is failure


def test_kimi_config_remains_available_when_model_dependencies_are_missing(monkeypatch):
    cause = ImportError("missing Kimi model dependency")
    monkeypatch.setattr(architecture, "KIMI_MODEL_IMPORT_ERROR", cause)
    registration = get_model_architecture("kimi_linear")
    config = registration.config_factory(hidden_size=32)
    assert config.hidden_size == 32
    assert isinstance(config, registration.config_type_loader())
    with pytest.raises(ImportError, match="Kimi Linear model dependencies") as failure:
        registration.model_class_loader()
    assert failure.value.__cause__ is cause
