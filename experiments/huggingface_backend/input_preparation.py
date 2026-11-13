"""Normalize trainer batches and shard model inputs for CP and TP."""

from __future__ import annotations

from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.nn.attention.flex_attention import BlockMask, and_masks

from llmtuner.components.loss import next_token_targets
from llmtuner.datasets.contracts import Batch, TrainerBatch
from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.common.attention.masks import (
    build_dense_attention_mask,
    create_attention_mask,
    get_causal_mask_mod,
    get_efficient_causal_mask_mod_for_packed_document,
)
from llmtuner.parallel.context_parallel.input_sharding import (
    shard_batch_for_cp,
    shard_batch_for_tp,
    shard_padding_mask_for_cp,
    shard_padding_mask_for_tp,
)
from llmtuner.parallel.topology import ParallelTopology

from .attention_backend import ATTENTION_BACKEND
from .attention_masks import mask_dtype


def _shard_for_cp(
    model,
    cp_mesh: DeviceMesh | None,
    *,
    packed: bool,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    positions: torch.Tensor | None,
    padding_mask: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    BlockMask | torch.Tensor | None,
]:
    if cp_mesh is None:
        attention_masks = None
        if (
            positions is not None
            and model.model.config._attn_implementation == ATTENTION_BACKEND
        ):
            attention_masks = model.get_attention_masks(positions=positions)
        return inputs, labels, positions, padding_mask, attention_masks

    if model._cp_strategy not in {"kcp", "compressor_ulysses"}:
        raise UnsupportedCombinationError(
            "Generic HF context-parallel attention kernels have been removed; "
            "use a model-owned CP backend"
        )
    if packed:
        raise UnsupportedCombinationError(
            "Model-owned HF CP input preparation does not support packed documents"
        )
    if positions is None:
        positions = torch.arange(inputs.numel(), device=inputs.device)
    attention_masks = None
    inputs, labels, positions = shard_batch_for_cp(
        inputs,
        labels,
        positions,
        cp_mesh,
        load_balancer=model._cp_load_balancer,
    )
    if padding_mask is not None:
        padding_mask = shard_padding_mask_for_cp(
            padding_mask, cp_mesh, model._cp_load_balancer
        )
    return inputs, labels, positions, padding_mask, attention_masks


def _shard_for_tp(
    model,
    tp_mesh: DeviceMesh | None,
    *,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    positions: torch.Tensor | None,
    padding_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if tp_mesh is None:
        return inputs, labels, positions, padding_mask
    if positions is None:
        positions = torch.arange(inputs.numel(), device=inputs.device)
    if padding_mask is not None:
        padding_mask = shard_padding_mask_for_tp(padding_mask, tp_mesh)
    inputs, labels = shard_batch_for_tp(inputs, labels, tp_mesh)
    return inputs, labels, positions, padding_mask


def preprocess_inputs(
    model,
    input_dict: Batch | TrainerBatch,
    *,
    parallel_dims: ParallelTopology | None,
    max_context_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Turn a dataloader batch into ``(inputs, labels, forward_kwargs)``."""
    extra_kwargs: dict[str, Any] = {}
    positions = None
    padding_mask = None

    if isinstance(input_dict, Batch):
        labels = next_token_targets(
            input_dict.labels.reshape(-1), seq_len=input_dict.labels.shape[-1]
        )
        inputs = input_dict.input_ids
    else:
        for field in (
            "pixel_values",
            "pixel_values_videos",
            "grid_thw",
            "grid_thw_videos",
            "mrope_positions",
        ):
            if input_dict.get(field) is not None:
                raise UnsupportedCombinationError(
                    f"HFTransformerModel cannot consume {field}: its text "
                    "decoder does not implement multimodal inputs. Use a "
                    "text dataset or a model adapter that consumes media "
                    "and its positional metadata."
                )
        inputs, labels = input_dict["input"], input_dict["labels"]
        positions = input_dict.get("positions")
        padding_mask = input_dict.get("padding_mask")

    row_length = inputs.shape[-1] if inputs.ndim == 2 and inputs.shape[0] > 1 else None
    inputs, labels = inputs.reshape(-1), labels.reshape(-1)
    if positions is not None:
        positions = positions.reshape(-1)
    if padding_mask is not None:
        padding_mask = padding_mask.reshape(-1).to(torch.bool)

    for name, value in (
        ("labels", labels),
        ("positions", positions),
        ("padding_mask", padding_mask),
    ):
        if value is not None and value.numel() != inputs.numel():
            raise ValueError(
                f"{name} has {value.numel()} tokens, but input has "
                f"{inputs.numel()}; batch token fields must have equal lengths"
            )

    packed = positions is not None and (
        getattr(model.model.config, "attn_mask_type", "causal") == "block_causal"
    )
    cp_mesh = None if parallel_dims is None else parallel_dims.get_optional_mesh("cp")
    row_attention_mask = None
    if row_length is not None and cp_mesh is not None:
        raise UnsupportedCombinationError(
            "Experimental HF CP does not support multiple independent batch rows"
        )
    attention_masks = None
    # Without CP, independent rows build their own mask below. A generic Flex
    # mask here would be discarded before forward.
    if cp_mesh is not None or row_length is None:
        inputs, labels, positions, padding_mask, attention_masks = _shard_for_cp(
            model,
            cp_mesh,
            # The full row-aware mask already contains document boundaries.
            packed=packed and row_attention_mask is None,
            inputs=inputs,
            labels=labels,
            positions=positions,
            padding_mask=padding_mask,
        )
    if row_attention_mask is not None:
        attention_masks = row_attention_mask
    if row_length is not None and cp_mesh is None:
        if packed:
            # Batch rows are independent even when RoPE positions continue from
            # an earlier chunk. Add mask boundaries without changing RoPE.
            mask_positions = positions.clone()
            mask_positions[::row_length] = 0
        else:
            mask_positions = torch.arange(inputs.numel(), device=inputs.device)
            mask_positions = mask_positions.remainder(row_length)
            if positions is None:
                positions = mask_positions
        if model.model.config._attn_implementation != ATTENTION_BACKEND:
            attention_masks = build_dense_attention_mask(
                mask_positions,
                dtype=mask_dtype(model),
                block_causal=True,
            )
        else:
            mask_mod = and_masks(
                get_causal_mask_mod(),
                get_efficient_causal_mask_mod_for_packed_document(mask_positions),
            )
            attention_masks = create_attention_mask(
                mask_mod,
                B=1,
                H=None,
                Q_LEN=inputs.numel(),
                KV_LEN=inputs.numel(),
                device=inputs.device,
                BLOCK_SIZE=128,
                separate_full_blocks=True,
            )
    if attention_masks is not None:
        extra_kwargs["attention_masks"] = attention_masks

    tp_mesh = None if parallel_dims is None else parallel_dims.get_optional_mesh("tp")
    inputs, labels, positions, padding_mask = _shard_for_tp(
        model,
        tp_mesh,
        inputs=inputs,
        labels=labels,
        positions=positions,
        padding_mask=padding_mask,
    )

    if positions is not None:
        extra_kwargs["positions"] = positions
    if padding_mask is not None:
        extra_kwargs["padding_mask"] = padding_mask
    return inputs, labels, extra_kwargs
