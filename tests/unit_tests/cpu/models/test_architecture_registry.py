"""Custom architectures use one registration seam without model imports."""

import subprocess
import sys

import pytest
from torch import nn
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


def test_registered_plain_module_takes_precedence_over_hf_mapping(monkeypatch):
    class NativeModel(nn.Module):
        pass

    monkeypatch.setattr(architecture, "_ARCHITECTURES", {})
    register_model_architecture(
        ModelArchitecture.from_types(
            _CustomConfig.model_type,
            config_type=_CustomConfig,
            model_class_loader=lambda: NativeModel,
        )
    )
    config = _CustomConfig()
    config.architectures = ["LlamaForCausalLM"]
    assert resolve_model_class(config) is NativeModel


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

    def unavailable(*args):
        raise cause

    registration = get_model_architecture("kimi_linear")
    config = registration.config_factory(hidden_size=32)
    assert config.hidden_size == 32
    assert isinstance(config, registration.config_type_loader())
    monkeypatch.setattr(architecture, "import_module", unavailable)
    with pytest.raises(ImportError, match="Kimi Linear model dependencies") as failure:
        registration.model_class_loader()
    assert failure.value.__cause__ is cause


def test_config_lookup_does_not_import_kimi_decoder_in_fresh_process():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from llmtuner.models.hf.factory import build_model_config; "
            "from llmtuner.models.architecture import get_model_architecture; "
            "build_model_config('qwen3', seq_len=32, arch_overrides={}); "
            "registration = get_model_architecture('kimi_linear'); "
            "config = registration.config_factory(hidden_size=32); "
            "assert isinstance(config, registration.config_type_loader()); "
            "assert 'llmtuner.models.kimi.kimi_k3.model' not in sys.modules; "
            "model = registration.model_class_loader(); "
            "assert model.__name__ == 'KimiLinearForCausalLM'; "
            "assert 'llmtuner.models.kimi.kimi_k3.model' in sys.modules; "
            "assert registration.model_class_loader() is model",
        ],
        check=True,
    )


def test_native_registration_and_update_do_not_require_transformers():
    source = """
import importlib.abc
import sys

class RejectTransformers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'transformers' or fullname.startswith('transformers.'):
            raise ImportError('Transformers is unavailable for this native model')

sys.meta_path.insert(0, RejectTransformers())
from dataclasses import dataclass
import torch
from torch import nn
from llmtuner.models.architecture import (
    ModelArchitecture, register_model_architecture, get_model_architecture,
)

@dataclass
class Config:
    width: int = 3

class NativeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projection = nn.Linear(config.width, 1)

    def forward(self, inputs):
        return self.projection(inputs)

register_model_architecture(ModelArchitecture.from_types(
    'native_test', config_type=Config, model_class_loader=lambda: NativeModel,
    explicit=True,
))
registration = get_model_architecture('native_test')
assert registration.config_type_loader() is Config
from llmtuner.models.explicit import build_explicit_model
config = registration.config_factory()
config.model_type = "native_test"
model = build_explicit_model(config)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
before = model.projection.weight.detach().clone()
model(torch.ones(2, 3)).sum().backward()
optimizer.step()
assert not torch.equal(before, model.projection.weight)
assert not any(name == 'transformers' or name.startswith('transformers.')
               for name in sys.modules)
assert 'llmtuner.models.kimi.kimi_k3.model' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", source], check=True)


def test_explicit_factory_is_opt_in_and_lazy(monkeypatch):
    from llmtuner.models import explicit

    calls = []
    expected = object()

    def model_class(config):
        calls.append(config)
        return expected

    registration = ModelArchitecture.from_types(
        "deepseek_v4",
        explicit=True,
        config_type=PretrainedConfig,
        model_class_loader=lambda: model_class,
    )
    monkeypatch.setitem(architecture._ARCHITECTURES, "deepseek_v4", registration)
    config = PretrainedConfig(model_type="deepseek_v4")
    assert explicit.build_explicit_model(config) is expected
    assert calls == [config]


def test_explicit_and_training_resolution_share_registered_constructor(monkeypatch):
    from llmtuner.models.explicit import build_explicit_model

    class Model(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.projection = nn.Linear(2, 2)

    registration = ModelArchitecture.from_types(
        "deepseek_v4",
        explicit=True,
        config_type=PretrainedConfig,
        model_class_loader=lambda: Model,
    )
    monkeypatch.setitem(architecture._ARCHITECTURES, "deepseek_v4", registration)
    config = PretrainedConfig(model_type="deepseek_v4")
    model = build_explicit_model(config)
    assert type(model) is resolve_model_class(config)
    assert model.config is config
    loss = model.projection(model.projection.weight).square().sum()
    loss.backward()
    assert model.projection.weight.grad is not None
    assert model.projection.weight.grad.abs().sum() > 0


def test_explicit_factory_does_not_enable_other_registered_architectures():
    from llmtuner.models.explicit import build_explicit_model

    assert get_model_architecture("qwen3_moe") is not None
    with pytest.raises(ValueError, match="No explicit model for 'qwen3_moe'"):
        build_explicit_model(PretrainedConfig(model_type="qwen3_moe"))


@pytest.mark.parametrize("model_type", ["kimi_k3", "qwen3_5", "qwen3_5_moe"])
def test_composite_explicit_models_remain_outside_automatic_registry(model_type):
    assert get_model_architecture(model_type) is None


@pytest.mark.parametrize("model_type", ["kimi_k3", "qwen3_5", "qwen3_5_moe"])
def test_composite_registration_is_task_scoped(model_type):
    entry = get_model_architecture(model_type, task="conditional_generation")
    assert entry is not None
    assert entry.explicit
    assert get_model_architecture(model_type) is None


def test_explicit_factory_uses_registered_composite_constructor(monkeypatch):
    from types import SimpleNamespace

    from llmtuner.models.explicit import build_explicit_model

    config = SimpleNamespace(model_type="composite_test")
    expected = object()
    entry = ModelArchitecture.from_types(
        config.model_type,
        config_type=SimpleNamespace,
        model_class_loader=lambda: (
            lambda received: expected if received is config else None
        ),
        task="conditional_generation",
        explicit=True,
    )
    monkeypatch.setitem(architecture._ARCHITECTURES, config.model_type, entry)
    assert get_model_architecture(config.model_type) is None
    assert build_explicit_model(config) is expected
