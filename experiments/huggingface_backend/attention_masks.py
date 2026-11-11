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
from llmtuner.parallel.context_parallel.input_sharding import (
    shard_attention_mask_for_cp,
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
    """Build the full-sequence mask expected by the active CP strategy."""
    if model.uses_dsa:
        raise UnsupportedCombinationError(
            "Context parallel with a DSA (dense-mask) model is not "
            "implemented: the CP mask path shards a flex BlockMask, and a "
            "dense additive mask has to be sliced by hand. Unlock by "
            "sharding the dense mask Q-axis (load balancer included) and "
            "re-validating numerics on multiple ranks."
        )
    if getattr(model.model.config, "attn_mask_type", "causal") == "block_causal":
        raise ValueError(
            "Context parallel with packed sequences needs a prebuilt mask: "
            "build the full-length BlockMask with get_attention_masks from "
            "the FULL positions and pass it to forward as attention_masks "
            "-- Q-sharded with shard_attention_mask_for_cp under "
            "'kv_allgather', or full-length (unsharded) under 'ulysses', "
            "whose all-to-all reassembles the full sequence on every rank. "
            "The positions this forward receives are already CP-sharded and "
            "cannot describe the full document structure."
        )
    cp_size = model.cp_mesh.size()
    full_len = positions.shape[0] * cp_size
    full_positions = torch.arange(full_len, device=positions.device)
    mask = model.get_attention_masks(positions=full_positions)
    if model._cp_strategy == "ulysses":
        return mask
    return shard_attention_mask_for_cp(
        mask,
        getattr(model, "_cp_mask_mesh", model.cp_mesh),
        model._cp_load_balancer,
    )
