"""Hugging Face attention backend registration and capability selection."""

from __future__ import annotations

import torch
from transformers.integrations.flex_attention import flex_attention_forward
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import AttentionInterface

from ...parallel.model_compile import maybe_regional_inductor

ATTENTION_BACKEND = "flex_torchtitan"
# The interface a CP run uses when flex is not available (NPU): sdpa plus the
# layer's CP kernel, which owns the redistribution and the mask.
CP_SDPA_BACKEND = "sdpa_cp"


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


def cp_sdpa_attention_adapter(module, query, key, value, attention_mask, **kwargs):
    """Adapt the HF attention interface to llmtuner's sdpa CP kernel.

    ``apply_cp`` attaches ``_titan_cp_sdpa_kernel`` to every attention module it
    covers, so the kernel is the layer's own CP implementation. Layers without
    one (a non-covered layer, or CP attached elsewhere) fall back to plain sdpa,
    which is what makes this safe to select for the whole model.
    """
    kernel = getattr(module, "_titan_cp_sdpa_kernel", None)
    if kernel is None:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, **kwargs
        )
    return kernel(
        query, key, value, module=module, block_mask=attention_mask, **kwargs
    ), None


def uses_deepseek_sparse_attention(config) -> bool:
    """Return whether the model uses DeepSeek sparse attention."""
    return getattr(config, "index_topk", None) is not None


def register_attention_backend() -> None:
    """Register llmtuner's attention adapters with Hugging Face."""
    AttentionInterface.register(ATTENTION_BACKEND, flex_attention_adapter)
    AttentionInterface.register(CP_SDPA_BACKEND, cp_sdpa_attention_adapter)


__all__ = [
    "ATTENTION_BACKEND",
    "CP_SDPA_BACKEND",
    "available_attention_backend",
    "cp_sdpa_attention_adapter",
    "flex_attention_adapter",
    "register_attention_backend",
    "uses_deepseek_sparse_attention",
]
