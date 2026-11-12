"""Custom architectures use one registration seam without model imports."""

import subprocess
import sys

import pytest
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from llmtuner.models import registry as architecture
from llmtuner.models.configuration import build_model_config, resolve_model_class
from llmtuner.models.registry import (
    ModelSpec,
    build_registered_model,
    get_model_spec,
    register_model,
)


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

    monkeypatch.setattr(architecture, "_MODELS", {})
    register_model(
        ModelSpec.from_types(
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
    previous = dict(architecture._MODELS)
    registration = ModelSpec.from_types(
        model_type,
        config_type=_CustomConfig,
        model_class_loader=lambda: _CustomModel,
    )
    with pytest.raises(ValueError, match="non-empty string"):
        register_model(registration)
    assert architecture._MODELS == previous


@pytest.mark.parametrize(
    "field", ["config_factory", "config_type_loader", "model_class_loader"]
)
def test_invalid_loader_cannot_replace_existing_architecture(field):
    from dataclasses import replace

    previous = architecture.get_model_spec("kimi_linear")
    invalid = replace(previous, **{field: None})
    with pytest.raises(TypeError, match=field):
        register_model(invalid, exist_ok=True)
    assert architecture.get_model_spec("kimi_linear") is previous


def test_registration_does_not_invoke_construction_hooks(monkeypatch):
    def unexpected():
        pytest.fail("Registration must leave config and model loading lazy")

    monkeypatch.setattr(architecture, "_MODELS", {})
    registration = ModelSpec("lazy_test", unexpected, unexpected, unexpected)
    register_model(registration)
    assert get_model_spec("lazy_test") is registration


def test_custom_architecture_builds_config_and_lazily_resolves_model(tmp_path):
    registration = ModelSpec.from_types(
        _CustomConfig.model_type,
        config_type=_CustomConfig,
        model_class_loader=lambda: _CustomModel,
    )
    register_model(registration)
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
        assert get_model_spec(_CustomConfig.model_type) is registration

        config.save_pretrained(tmp_path)
        restored = build_model_config(str(tmp_path), seq_len=64)
        assert isinstance(restored, _CustomConfig)
        assert restored.hidden_size == 16
        assert restored.max_position_embeddings == 64
    finally:
        architecture._MODELS.pop(_CustomConfig.model_type)


def test_duplicate_custom_architecture_requires_explicit_replacement():
    registration = ModelSpec.from_types(
        _CustomConfig.model_type,
        config_type=_CustomConfig,
        model_class_loader=lambda: _CustomModel,
    )
    register_model(registration)
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_model(registration)
        register_model(registration, exist_ok=True)
    finally:
        architecture._MODELS.pop(_CustomConfig.model_type)


def test_saved_custom_config_without_position_limit_matches_offline_construction(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(architecture, "_MODELS", {})
    register_model(
        ModelSpec.from_types(
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
def test_saved_config_requires_a_registered_architecture(monkeypatch, tmp_path, custom):
    from transformers import LlamaConfig

    def unavailable_config():
        raise ImportError("optional custom configuration dependency is absent")

    optional = ModelSpec(
        "unavailable_optional_model",
        unavailable_config,
        unavailable_config,
        unavailable_config,
    )
    monkeypatch.setattr(architecture, "_MODELS", {})
    register_model(optional)
    if custom:
        register_model(
            ModelSpec.from_types(
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

    if not custom:
        with pytest.raises(ValueError, match="No local configuration registered"):
            build_model_config(str(tmp_path), seq_len=64)
        return

    restored = build_model_config(str(tmp_path), seq_len=64)
    assert isinstance(restored, config_type)
    assert restored.hidden_size == 16
    assert restored.max_position_embeddings == 64


def test_saved_config_uses_replaced_architecture_loader(monkeypatch, tmp_path):
    class ReplacementConfig(_CustomConfig):
        pass

    monkeypatch.setattr(architecture, "_MODELS", {})
    _CustomConfig(hidden_size=16, max_position_embeddings=32).save_pretrained(tmp_path)
    for config_type in (_CustomConfig, ReplacementConfig):
        register_model(
            ModelSpec.from_types(
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

    monkeypatch.setattr(architecture, "_MODELS", {})
    _CustomConfig(hidden_size=16, max_position_embeddings=32).save_pretrained(tmp_path)
    register_model(
        ModelSpec.from_types(
            _CustomConfig.model_type,
            config_type=StaleConfig,
            model_class_loader=lambda: _CustomModel,
        )
    )
    assert isinstance(build_model_config(str(tmp_path), seq_len=32), StaleConfig)
    StaleConfig.reject_construction = True
    register_model(
        ModelSpec.from_types(
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

    monkeypatch.setattr(architecture, "_MODELS", {})
    register_model(
        ModelSpec(OptionalConfig.model_type, unavailable, unavailable, unavailable)
    )
    OptionalConfig(max_position_embeddings=32).save_pretrained(tmp_path)
    with pytest.raises(ImportError) as caught:
        build_model_config(str(tmp_path), seq_len=64)
    assert caught.value is failure


def test_kimi_config_remains_available_when_model_dependencies_are_missing(monkeypatch):
    cause = ImportError("missing Kimi model dependency")

    def unavailable(*args):
        raise cause

    registration = get_model_spec("kimi_linear")
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
            "from llmtuner.models.configuration import build_model_config; "
            "from llmtuner.models.registry import get_model_spec; "
            "build_model_config('qwen3_moe', seq_len=32, arch_overrides={}); "
            "registration = get_model_spec('kimi_linear'); "
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
from llmtuner.models.registry import (
    ModelSpec, register_model, get_model_spec,
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

register_model(ModelSpec.from_types(
    'native_test', config_type=Config, model_class_loader=lambda: NativeModel,
    explicit=True,
))
registration = get_model_spec('native_test')
assert registration.config_type_loader() is Config
from llmtuner.models.registry import build_registered_model
config = registration.config_factory()
config.model_type = "native_test"
model = build_registered_model(config, explicit_only=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=False)
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
    calls = []
    expected = object()

    def model_class(config):
        calls.append(config)
        return expected

    registration = ModelSpec.from_types(
        "deepseek_v4",
        explicit=True,
        config_type=PretrainedConfig,
        model_class_loader=lambda: model_class,
    )
    monkeypatch.setitem(architecture._MODELS, "deepseek_v4", registration)
    config = PretrainedConfig(model_type="deepseek_v4")
    assert build_registered_model(config, explicit_only=True) is expected
    assert calls == [config]


def test_explicit_and_training_resolution_share_registered_constructor(monkeypatch):
    class Model(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.projection = nn.Linear(2, 2)

    registration = ModelSpec.from_types(
        "deepseek_v4",
        explicit=True,
        config_type=PretrainedConfig,
        model_class_loader=lambda: Model,
    )
    monkeypatch.setitem(architecture._MODELS, "deepseek_v4", registration)
    config = PretrainedConfig(model_type="deepseek_v4")
    model = build_registered_model(config, explicit_only=True)
    assert type(model) is resolve_model_class(config)
    assert model.config is config
    loss = model.projection(model.projection.weight).square().sum()
    loss.backward()
    assert model.projection.weight.grad is not None
    assert model.projection.weight.grad.abs().sum() > 0


def test_explicit_factory_builds_registered_architectures():
    config = build_model_config(
        "qwen3_moe",
        seq_len=32,
        arch_overrides={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_experts": 2,
            "num_experts_per_tok": 1,
        },
    )
    model = build_registered_model(config, explicit_only=True)
    assert type(model).__name__ == "Qwen3MoeForCausalLM"


@pytest.mark.parametrize("model_type", ["kimi_k3", "qwen3_5", "qwen3_5_moe"])
def test_composite_explicit_models_remain_outside_automatic_registry(model_type):
    assert get_model_spec(model_type) is None


@pytest.mark.parametrize("model_type", ["kimi_k3", "qwen3_5", "qwen3_5_moe"])
def test_composite_registration_is_task_scoped(model_type):
    entry = get_model_spec(model_type, task="conditional_generation")
    assert entry is not None
    assert entry.explicit
    assert get_model_spec(model_type) is None


def test_explicit_factory_uses_registered_composite_constructor(monkeypatch):
    from types import SimpleNamespace

    config = SimpleNamespace(model_type="composite_test")
    expected = object()
    entry = ModelSpec.from_types(
        config.model_type,
        config_type=SimpleNamespace,
        model_class_loader=lambda: (
            lambda received: expected if received is config else None
        ),
        task="conditional_generation",
        explicit=True,
    )
    monkeypatch.setitem(architecture._MODELS, config.model_type, entry)
    assert get_model_spec(config.model_type) is None
    assert build_registered_model(config, explicit_only=True) is expected


def test_spec_constructor_rejects_wrong_config_before_loading_model():
    calls = []
    specification = ModelSpec.from_types(
        "expected_model",
        config_type=_CustomConfig,
        model_class_loader=lambda: calls.append("loaded"),
    )
    with pytest.raises(ValueError, match="cannot consume"):
        specification.build_model(_CustomConfig())
    assert calls == []


def test_registered_constructor_enforces_task_without_loading(monkeypatch):
    calls = []
    expected = object()
    specification = ModelSpec.from_types(
        _CustomConfig.model_type,
        config_type=_CustomConfig,
        model_class_loader=lambda: lambda config: calls.append(config) or expected,
        task="conditional_generation",
        explicit=True,
    )
    monkeypatch.setitem(architecture._MODELS, specification.model_type, specification)
    config = specification.build_config(hidden_size=24)
    with pytest.raises(ValueError, match="No model"):
        build_registered_model(config, task="causal_lm")
    assert calls == []
    assert (
        build_registered_model(
            config, task="conditional_generation", explicit_only=True
        )
        is expected
    )
    assert calls == [config]


@pytest.mark.parametrize("returned", [None, object(), _CustomConfig()])
def test_config_factory_must_return_its_registered_architecture(returned):
    def forbidden():
        raise AssertionError("Invalid config loaded model code")

    registration = ModelSpec(
        "another_architecture",
        lambda **kwargs: returned,
        lambda: _CustomConfig,
        forbidden,
    )
    with pytest.raises(ValueError, match="another_architecture.*configuration"):
        registration.build_config()


@pytest.mark.parametrize("returned", [None, object(), _CustomConfig()])
def test_saved_config_loader_cannot_change_registered_architecture(
    monkeypatch, tmp_path, returned
):
    class SavedConfig(PretrainedConfig):
        model_type = "test_saved_config_boundary"

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return returned

    SavedConfig().save_pretrained(tmp_path)
    entry = ModelSpec.from_types(
        SavedConfig.model_type,
        config_type=SavedConfig,
        model_class_loader=lambda: pytest.fail("Config loading imported model code"),
    )
    monkeypatch.setitem(architecture._MODELS, entry.model_type, entry)
    with pytest.raises(ValueError, match="configuration.*model_type"):
        build_model_config(str(tmp_path), seq_len=64)
