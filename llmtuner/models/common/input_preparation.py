"""Prepare text batches for explicit decoder models.

This module owns only the framework-level token layout. A model-specific
``prepare_inputs`` method can add extra fields, but it must return the same
``(flat_inputs, flat_labels, forward_kwargs)`` contract consumed by Trainer.
"""

from __future__ import annotations

from typing import Any

import torch

from ...datasets.contracts import (
    Batch,
    TrainerBatch,
    validate_token_alignment,
    validated_token_fields,
)
from ...errors import UnsupportedCombinationError
from ...parallel.context_parallel.input_sharding import (
    shard_batch_for_cp,
    shard_batch_for_tp,
    shard_padding_mask_for_cp,
)
from ...parallel.topology import ParallelTopology
from .attention.masks import build_dense_attention_mask
from .targets import next_token_targets

__all__ = ["prepare_text_inputs", "restore_text_batch_rows"]


def restore_text_batch_rows(
    input_ids: torch.Tensor | None,
    labels: torch.Tensor | None,
    *,
    batch_shape: tuple[int, int] | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, bool]:
    """Restore declared independent rows before model execution and causal loss.

    The boolean records the caller's flat layout for output restoration. Position
    broadcasting, padding and attention/recurrent semantics remain model-owned.
    """
    flat_input = input_ids is not None and input_ids.ndim == 1
    if batch_shape is None:
        return input_ids.unsqueeze(0) if flat_input else input_ids, labels, flat_input
    if not flat_input:
        raise ValueError("input_batch_shape requires flat Trainer tokens")
    if (
        not isinstance(batch_shape, tuple)
        or len(batch_shape) != 2
        or any(type(size) is not int or size <= 0 for size in batch_shape)
    ):
        raise ValueError("input_batch_shape must contain two positive integers")
    if input_ids.numel() != batch_shape[0] * batch_shape[1]:
        raise ValueError("input_batch_shape does not match input token count")
    if labels is not None:
        if labels.ndim == 1 and labels.numel() == input_ids.numel():
            labels = labels.reshape(batch_shape)
        elif tuple(labels.shape) != batch_shape:
            raise ValueError("labels must match the declared input batch layout")
    return input_ids.reshape(batch_shape), labels, flat_input


def _batch_fields(
    batch: Batch | TrainerBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    tokens, labels = validated_token_fields(batch)
    if isinstance(batch, Batch):
        return (
            tokens.reshape(-1),
            next_token_targets(labels.reshape(-1), seq_len=tokens.shape[-1]),
            None,
            None,
        )
    unsupported = {
        name
        for name in (
            "pixel_values",
            "pixel_values_videos",
            "grid_thw",
            "grid_thw_videos",
            "mrope_positions",
        )
        if batch.get(name) is not None
    }
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(
            f"Text explicit models cannot consume multimodal fields: {names}"
        )
    positions = batch.get("positions")
    padding_mask = batch.get("padding_mask")
    for name, value in (
        ("positions", positions),
        ("padding_mask", padding_mask),
    ):
        if value is None:
            continue
        validate_token_alignment(tokens, value, name=name)
    if positions is not None and positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("positions must contain int32 or int64 token indices")
    inputs = tokens.reshape(-1)
    labels = labels.reshape(-1)
    if positions is not None:
        positions = positions.reshape(-1)
    if padding_mask is not None:
        padding_mask = padding_mask.reshape(-1).to(torch.bool)
    return inputs, labels, positions, padding_mask


def prepare_text_inputs(
    batch: Batch | TrainerBatch,
    *,
    parallel_dims: ParallelTopology | None,
    packed: bool,
    mask_dtype: torch.dtype,
    cp_load_balancer: Any = None,
    preserve_batch_rows: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Normalize and shard a text batch without inspecting model structure.

    The caller declares packed semantics, mask dtype and CP load balancing;
    this helper only moves related token tensors together. A dense block-causal
    mask is constructed from explicit positions for declared packed batches.
    Models opting into row preservation consume input_batch_shape in forward;
    the helper retains layout only, not attention/recurrent reset mathematics.
    """
    inputs, labels, positions, padding_mask = _batch_fields(batch)
    if packed and positions is None:
        # Document boundaries cannot be recovered from token IDs or a local
        # shard. Never turn declared packed attention into plain causal math.
        raise ValueError("Packed text input requires positions for document boundaries")
    tokens = batch.input_ids if isinstance(batch, Batch) else batch["input"]
    multirow = preserve_batch_rows and tokens.ndim == 2 and tokens.shape[0] > 1
    if (
        multirow
        and parallel_dims is not None
        and any(
            parallel_dims.get_optional_mesh(axis) is not None for axis in ("cp", "tp")
        )
    ):
        raise UnsupportedCombinationError(
            "Multirow text input requires row-aware CP/TP sharding"
        )
    extra: dict[str, Any] = {}
    if parallel_dims is not None:
        cp_mesh = parallel_dims.get_optional_mesh("cp")
        if cp_mesh is not None:
            if packed:
                raise UnsupportedCombinationError(
                    "Packed CP input preparation requires a model-specific "
                    "full-sequence mask contract; the generic explicit text "
                    "adapter refuses to build an incorrect local mask."
                )
            if positions is None:
                positions = torch.arange(inputs.numel(), device=inputs.device)
            inputs, labels, positions = shard_batch_for_cp(
                inputs,
                labels,
                positions,
                cp_mesh,
                cp_load_balancer,
            )
            if padding_mask is not None:
                padding_mask = shard_padding_mask_for_cp(
                    padding_mask,
                    cp_mesh,
                    cp_load_balancer,
                )
        tp_mesh = parallel_dims.get_optional_mesh("tp")
        if tp_mesh is not None:
            if positions is None:
                positions = torch.arange(inputs.numel(), device=inputs.device)
            inputs, labels = shard_batch_for_tp(inputs, labels, tp_mesh)
            # TP attention gathers the residual sequence before QKV. Positions
            # and padding describe that gathered sequence, unlike local input
            # IDs and loss targets. CP has already selected its own sequence.
    if positions is not None:
        extra["positions"] = positions
    if padding_mask is not None:
        extra["padding_mask"] = padding_mask
    if multirow:
        rows, width = tokens.shape
        extra["input_batch_shape"] = (rows, width)
        for name in ("positions", "padding_mask"):
            if name in extra:
                extra[name] = extra[name].reshape(rows, width)
    if packed:
        if multirow:
            # Build B independent S*S masks, not a (B*S)*(B*S) temporary.
            # Row-local document numbering has the same equality semantics.
            mask = torch.cat(
                [
                    build_dense_attention_mask(row, dtype=mask_dtype, block_causal=True)
                    for row in positions.reshape(rows, width)
                ],
                dim=0,
            )
        else:
            mask = build_dense_attention_mask(
                positions, dtype=mask_dtype, block_causal=True
            )
        if padding_mask is not None:
            padding_keys = (
                padding_mask.reshape(rows, 1, 1, width)
                if multirow
                else padding_mask[None, :]
            )
            # An explicit attention mask takes precedence in model.forward.
            mask = mask.masked_fill(padding_keys, float("-inf"))
        extra["attention_masks"] = mask
    return inputs, labels, extra
