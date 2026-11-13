"""Build dense and Flex attention masks for the HF training wrapper."""

from __future__ import annotations

import torch
from torch.nn.attention.flex_attention import BlockMask, and_masks

from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.common.attention.masks import (
    build_dense_attention_mask,
    create_attention_mask,
    get_causal_mask_mod,
    get_efficient_causal_mask_mod_for_packed_document,
)


def mask_dtype(model) -> torch.dtype:
    """Derive mask dtype from parameters available on this pipeline stage."""
    for module in (model.tok_embeddings, model.lm_head):
        weight = getattr(module, "weight", None)
        if weight is not None:
            return weight.dtype
    for parameter in model.parameters():
        return parameter.dtype
    raise ValueError(
        "cannot derive a dense attention mask's dtype: the model holds no "
        "embedding, head or parameter on this stage"
    )


def get_attention_masks(model, positions: torch.Tensor) -> BlockMask | torch.Tensor:
    """Build a causal mask, optionally restricting attention to each document."""
    block_causal = (
        getattr(model.model.config, "attn_mask_type", "causal") == "block_causal"
    )
    if model.uses_dsa:
        return build_dense_attention_mask(
            positions,
            dtype=mask_dtype(model),
            block_causal=block_causal,
        )
    if block_causal:
        mask_mod = and_masks(
            get_causal_mask_mod(),
            get_efficient_causal_mask_mod_for_packed_document(positions),
        )
    else:
        mask_mod = get_causal_mask_mod()

    num_tokens = positions.shape[0]
    return create_attention_mask(
        mask_mod,
        B=1,
        H=None,
        Q_LEN=num_tokens,
        KV_LEN=num_tokens,
        device=positions.device,
        BLOCK_SIZE=128,
        separate_full_blocks=True,
    )


def get_cp_attention_masks(model, positions: torch.Tensor) -> BlockMask:
    """Reject the removed generic CP mask path without rebuilding local masks."""
    raise UnsupportedCombinationError(
        "Generic HF context-parallel attention kernels have been removed; "
        "use a model-owned CP backend with its own mask contract"
    )
