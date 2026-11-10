"""Opt-in construction for explicit model subclasses."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from .architecture import get_model_architecture

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

# Preserve the opt-in API's supported types. Adding an architecture to the
# training registry must not implicitly enable it at this construction entry.
_REGISTERED_EXPLICIT_MODELS = frozenset({"deepseek_v4"})

# Composite models remain opt-in: registering them for automatic causal-LM
# resolution would also change configuration and text-model selection.
_EXPLICIT_MODELS = {
    "kimi_k3": (
        ".kimi.kimi_k3.multimodal",
        "KimiK3ForConditionalGeneration",
    ),
    "qwen3_5": (
        ".qwen.qwen3_5.multimodal",
        "Qwen3_5ForConditionalGeneration",
    ),
    "qwen3_5_moe": (
        ".qwen.qwen3_5_moe.multimodal",
        "Qwen3_5MoeForConditionalGeneration",
    ),
}


def build_explicit_model(config: PretrainedConfig) -> PreTrainedModel:
    """Build an explicitly selected model without changing HF auto resolution."""
    if config.model_type in _REGISTERED_EXPLICIT_MODELS:
        architecture = get_model_architecture(config.model_type)
        if architecture is None:
            raise ValueError(
                f"Model architecture {config.model_type!r} is not registered"
            )
        return architecture.model_class_loader()(config)
    try:
        module_name, class_name = _EXPLICIT_MODELS[config.model_type]
    except KeyError as error:
        supported = ", ".join(
            sorted(_EXPLICIT_MODELS.keys() | _REGISTERED_EXPLICIT_MODELS)
        )
        raise ValueError(
            f"No explicit model for {config.model_type!r}; choose one of {supported}"
        ) from error
    module = import_module(module_name, __package__)
    return getattr(module, class_name)(config)


__all__ = ["build_explicit_model"]
