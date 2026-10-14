"""Hugging Face attention backend registration and capability selection."""

from __future__ import annotations

import torch
from transformers.integrations.flex_attention import flex_attention_forward
from transformers.modeling_utils import AttentionInterface

from ...parallel.model_compile import maybe_regional_inductor

ATTENTION_BACKEND = "flex_torchtitan"


def available_attention_backend() -> str:
    """Return the attention backend supported by the current device."""
    return ATTENTION_BACKEND if torch.cuda.is_available() else "sdpa"


def flex_attention_adapter(module, query, key, value, attention_mask, **kwargs):
    """Adapt the Hugging Face attention interface to llmtuner flex kernels."""
    kernel = getattr(module, "_titan_flex_kernel", None)
    if kernel is None:
        with maybe_regional_inductor(
            {}, enabled=getattr(module, "_regional_inductor_enabled", False)
        ):
            return flex_attention_forward(
                module, query, key, value, attention_mask, **kwargs
            )
    output = kernel(
        query, key, value, module=module, block_mask=attention_mask, **kwargs
    )
    return output, None


def uses_deepseek_sparse_attention(config) -> bool:
    """Return whether the model uses DeepSeek sparse attention."""
    return getattr(config, "index_topk", None) is not None


def register_attention_backend() -> None:
    """Register llmtuner's flex adapter with Hugging Face."""
    AttentionInterface._global_mapping[ATTENTION_BACKEND] = flex_attention_adapter


__all__ = [
    "ATTENTION_BACKEND",
    "available_attention_backend",
    "flex_attention_adapter",
    "register_attention_backend",
    "uses_deepseek_sparse_attention",
]
