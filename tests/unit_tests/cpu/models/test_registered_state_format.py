"""External state formats extend the sole registry without loading models."""

from types import SimpleNamespace

import pytest

from llmtuner.components.checkpointer.model_state import ModelStateDictAdapter
from llmtuner.models import registry
from llmtuner.models.registry import ModelSpec, register_model, resolve_state_format
from llmtuner.models.state import ModelStateFormat


def test_registered_format_maps_state_without_loading_model(monkeypatch):
    class ExternalFormat(ModelStateFormat):
        def __init__(self, config, *, expert_parallel, sharded_expert_parallel):
            self.prefix = config.external_prefix

        def map_key(self, key, *, to_hf):
            return self.prefix + key if to_hf else key.removeprefix(self.prefix)

    def unexpected_load():
        pytest.fail("Checkpoint conversion must not load the model or config class")

    monkeypatch.setattr(registry, "_MODELS", {})
    register_model(
        ModelSpec(
            "external_format_test",
            unexpected_load,
            unexpected_load,
            unexpected_load,
            state_format_loader=lambda: ExternalFormat,
        )
    )
    adapter = ModelStateDictAdapter(
        SimpleNamespace(model_type="external_format_test", external_prefix="ext."),
        None,
    )
    value = object()
    original = {"weight": value}
    exported = adapter.to_hf(original)
    assert exported == {"ext.weight": value}
    assert adapter.from_hf(exported) == original
    assert exported["ext.weight"] is value
    assert original == {"weight": value}
    assert not adapter.coordinates_state_conversion


@pytest.mark.parametrize("loaded", [None, object, lambda: ModelStateFormat()])
def test_invalid_format_class_is_rejected_before_checkpoint_io(monkeypatch, loaded):
    monkeypatch.setattr(registry, "_MODELS", {})
    register_model(
        ModelSpec(
            "invalid_format_test",
            lambda: None,
            lambda: object,
            lambda: object,
            state_format_loader=lambda: loaded,
        )
    )
    with pytest.raises(TypeError, match="ModelStateFormat class"):
        ModelStateDictAdapter(
            SimpleNamespace(model_type="invalid_format_test"), "/missing/checkpoint"
        )


@pytest.mark.parametrize("sharded", [False, True])
def test_model_formats_preserve_distinct_ep_checkpoint_semantics(sharded):
    qwen = resolve_state_format(
        SimpleNamespace(model_type="qwen3_moe", num_experts=4),
        expert_parallel=True,
        sharded_expert_parallel=sharded,
    )
    kimi = resolve_state_format(
        SimpleNamespace(model_type="kimi_linear", num_experts=4),
        expert_parallel=True,
        sharded_expert_parallel=sharded,
    )
    assert qwen.grouped_experts is sharded
    assert not qwen.coordinates_conversion
    assert kimi.grouped_experts
    assert kimi.coordinates_conversion
