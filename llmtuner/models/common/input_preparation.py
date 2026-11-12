"""Prepare text batches for explicit decoder models.

This module owns only the framework-level token layout. A model-specific
``prepare_inputs`` method can add extra fields, but it must return the same
``(flat_inputs, flat_labels, forward_kwargs)`` contract consumed by Trainer.
"""

from __future__ import annotations

from typing import Any

import torch

from ...components.loss import next_token_targets
from ...datasets.contracts import Batch, TrainerBatch
from ...errors import UnsupportedCombinationError
from ...parallel.context_parallel.input_sharding import (
    shard_batch_for_cp,
    shard_batch_for_tp,
    shard_padding_mask_for_cp,
)
from ...parallel.topology import ParallelTopology
from .attention.masks import build_dense_attention_mask

__all__ = ["prepare_text_inputs"]


def _batch_fields(
    batch: Batch | TrainerBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    tokens = batch.input_ids if isinstance(batch, Batch) else batch["input"]
    if tokens.ndim not in (1, 2):
        raise ValueError("Text input must have shape [tokens] or [batch, sequence]")
    if tokens.dtype not in (torch.int32, torch.int64):
        raise TypeError("Text input must contain int32 or int64 token indices")
    labels = batch.labels if isinstance(batch, Batch) else batch["labels"]
    if labels.dtype not in (torch.int32, torch.int64):
        raise TypeError("labels must contain int32 or int64 target indices")
    if isinstance(batch, Batch):
        # Synthetic labels are unshifted. Their row width controls which
        # terminal targets become ignored, so equal numel alone is insufficient.
        if labels.shape != tokens.shape:
            raise ValueError("Synthetic labels must have the same shape as input_ids")
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
    inputs = tokens.reshape(-1)
    labels = labels.reshape(-1)
    positions = batch.get("positions")
    padding_mask = batch.get("padding_mask")
    if positions is not None:
        positions = positions.reshape(-1)
    if padding_mask is not None:
        padding_mask = padding_mask.reshape(-1).to(torch.bool)
    return inputs, labels, positions, padding_mask


def prepare_text_inputs(
    model: torch.nn.Module,
    batch: Batch | TrainerBatch,
    *,
    parallel_dims: ParallelTopology | None,
    max_context_length: int | None = None,
    preserve_batch_rows: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Normalize and shard a text batch without inspecting model structure.

    The model supplies the semantic decoder contract; this helper only moves
    related token tensors together. A dense block-causal mask is constructed
    from explicit positions when the configuration says the batch is packed.
    Models opting into row preservation consume input_batch_shape in forward;
    the helper retains layout only, not attention/recurrent reset mathematics.
    """
    del max_context_length
    inputs, labels, positions, padding_mask = _batch_fields(batch)
    for name, value in (
        ("labels", labels),
        ("positions", positions),
        ("padding_mask", padding_mask),
    ):
        if value is not None and value.numel() != inputs.numel():
            raise ValueError(
                f"{name} has {value.numel()} tokens, but input has {inputs.numel()}"
            )
        if value is not None and value.device != inputs.device:
            raise ValueError(
                f"{name} is on {value.device}, but input is on {inputs.device}"
            )
    if positions is not None and positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("positions must contain int32 or int64 token indices")
    config = getattr(model, "config", None)
    if config is None:
        raise TypeError(f"{type(model).__name__} must expose a config")
    packed = getattr(config, "attn_mask_type", "causal") == "block_causal"
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
            f"{type(model).__name__} multirow input requires row-aware CP/TP sharding"
        )
    extra: dict[str, Any] = {}
    if parallel_dims is not None:
        cp_mesh = parallel_dims.get_optional_mesh("cp")
        load_balancer = getattr(model, "_cp_load_balancer", None)
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
                load_balancer,
            )
            if padding_mask is not None:
                padding_mask = shard_padding_mask_for_cp(
                    padding_mask,
                    cp_mesh,
                    load_balancer,
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
        dtype = next(model.parameters()).dtype
        if multirow:
            # Build B independent S*S masks, not a (B*S)*(B*S) temporary.
            # Row-local document numbering has the same equality semantics.
            mask = torch.cat(
                [
                    build_dense_attention_mask(row, dtype=dtype, block_causal=True)
                    for row in positions.reshape(rows, width)
                ],
                dim=0,
            )
        else:
            mask = build_dense_attention_mask(positions, dtype=dtype, block_causal=True)
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
