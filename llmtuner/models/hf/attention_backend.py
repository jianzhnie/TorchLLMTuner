"""Hugging Face attention backend registration and capability selection."""

from __future__ import annotations

import importlib

import torch
from transformers.integrations.flex_attention import flex_attention_forward
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import AttentionInterface

from ...parallel.model_compile import maybe_regional_inductor

ATTENTION_BACKEND = "flex_torchtitan"
# The interface a CP run uses when flex is not available (NPU): sdpa plus the
# layer's CP kernel, which owns the redistribution and the mask.
CP_SDPA_BACKEND = "sdpa_cp"
EAGER_FLEX_BACKEND = "llmtuner_eager_flex"


def eager_flex_attention_adapter(module, query, key, value, attention_mask, **kwargs):
    """Consume HF dense masks through explicit eager Flex without global patches."""
    from torch.nn.attention.flex_attention import create_block_mask

    from llmtuner.errors import UnsupportedCombinationError
    from llmtuner.models.common.attention.eager_flex import eager_flex_attention

    if kwargs.get("dropout", 0.0) != 0.0:
        raise UnsupportedCombinationError(
            "eager Flex does not implement attention dropout"
        )
    batch, heads, query_length, _ = query.shape
    key_length = key.shape[-2]
    if query_length != key_length:
        raise UnsupportedCombinationError(
            "eager Flex HF adapter requires self-attention without KV cache"
        )
    if attention_mask is None:
        dense = torch.ones(
            query_length, key_length, dtype=torch.bool, device=query.device
        ).tril()[None, None]
    else:
        if attention_mask.ndim != 4:
            raise ValueError("eager Flex HF adapter requires a four-dimensional mask")
        dense = attention_mask[..., :query_length, :key_length]
        if dense.dtype != torch.bool:
            if not torch.all(
                (dense == 0)
                | torch.isneginf(dense)
                | (dense == torch.finfo(dense.dtype).min)
            ):
                raise UnsupportedCombinationError(
                    "eager Flex accepts boolean or zero/blocked masks, "
                    "not attention biases"
                )
            dense = dense == 0
    if (
        dense.shape[-2:] != (query_length, key_length)
        or dense.shape[0] not in (1, batch)
        or dense.shape[1] not in (1, heads)
    ):
        raise ValueError(
            "eager Flex mask must broadcast to Q/K batch, heads and lengths"
        )

    def mask_mod(batch_index, head, query_index, key_index):
        return (
            (query_index < query_length)
            & (key_index < key_length)
            & dense[
                0 if dense.shape[0] == 1 else batch_index,
                0 if dense.shape[1] == 1 else head,
                query_index.clamp(max=query_length - 1),
                key_index.clamp(max=key_length - 1),
            ]
        )

    mask = create_block_mask(
        mask_mod,
        batch,
        None if dense.shape[1] == 1 else heads,
        query_length,
        key_length,
        device=query.device,
        _compile=False,
    )
    output = eager_flex_attention(query, key, value, mask, scale=kwargs.get("scaling"))
    return output.transpose(1, 2).contiguous(), None


FLEX_INDUCTOR_CONFIGS = {
    "wrap_inductor_compiled_regions": True,
    "max_autotune": True,
    "coordinate_descent_tuning": True,
    "triton.cudagraphs": False,
}


def allow_npu_flex_attention() -> None:
    module = importlib.import_module("torch.nn.attention.flex_attention")
    if not hasattr(module, "_validate_device") or hasattr(
        module, "_llmtuner_original_validate_device"
    ):
        return
    original = module._validate_device

    def validate_device(query, key, value):
        if all(tensor.device.type == "npu" for tensor in (query, key, value)):
            return
        original(query, key, value)

    module._llmtuner_original_validate_device = original
    module._validate_device = validate_device


def available_attention_backend() -> str:
    """Return the attention backend supported by the current device."""
    return ATTENTION_BACKEND if torch.cuda.is_available() else "sdpa"


def flex_attention_adapter(module, query, key, value, attention_mask, **kwargs):
    """Adapt the Hugging Face attention interface to llmtuner flex kernels."""
    if kwargs.get("dropout", 0.0) != 0.0:
        from ...errors import UnsupportedCombinationError

        raise UnsupportedCombinationError("Flex attention does not support dropout")
    kernel = getattr(module, "_titan_flex_kernel", None)
    if kernel is None:
        from torch.nn.attention.flex_attention import (
            AuxRequest,
            BlockMask,
            flex_attention,
        )

        explicit_flex = getattr(module.config, "attention_backend", None) == "flex"
        if explicit_flex and query.device.type == "npu":
            allow_npu_flex_attention()
        num_heads = query.shape[1]
        if (
            explicit_flex
            and isinstance(attention_mask, BlockMask)
            and not num_heads & (num_heads - 1)
            and kwargs.get("sliding_window") is None
            and kwargs.get("softcap") is None
            and kwargs.get("s_aux") is None
        ):
            with maybe_regional_inductor(
                FLEX_INDUCTOR_CONFIGS,
                enabled=getattr(module, "_regional_inductor_enabled", False),
            ):
                output, _ = flex_attention(
                    query,
                    key,
                    value,
                    score_mod=None,
                    block_mask=attention_mask,
                    scale=kwargs.get("scaling"),
                    enable_gqa=True,
                    return_aux=AuxRequest(lse=False),
                    kernel_options=kwargs.get("kernel_options", {}),
                )
            return output.transpose(1, 2).contiguous(), None
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
    AttentionInterface.register(EAGER_FLEX_BACKEND, eager_flex_attention_adapter)
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

    ALL_MASK_ATTENTION_FUNCTIONS.register(
        EAGER_FLEX_BACKEND, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]
    )


__all__ = [
    "ATTENTION_BACKEND",
    "CP_SDPA_BACKEND",
    "available_attention_backend",
    "cp_sdpa_attention_adapter",
    "flex_attention_adapter",
    "register_attention_backend",
    "uses_deepseek_sparse_attention",
]
