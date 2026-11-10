"""Opt-in construction for explicit model subclasses."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

_EXPLICIT_MODELS = {
    "deepseek_v4": (
        ".deepseek.deepseek_v4.modeling",
        "DeepseekV4ForCausalLM",
    ),
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
    try:
        module_name, class_name = _EXPLICIT_MODELS[config.model_type]
    except KeyError as error:
        supported = ", ".join(sorted(_EXPLICIT_MODELS))
        raise ValueError(
            f"No explicit model for {config.model_type!r}; choose one of {supported}"
        ) from error
    module = import_module(module_name, __package__)
    return getattr(module, class_name)(config)


__all__ = ["build_explicit_model"]
