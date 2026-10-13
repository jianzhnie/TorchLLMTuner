"""Input shape normalization and language-model target construction."""

from __future__ import annotations

import torch

from ...components.loss import next_token_targets


def collapse_batch_dims(
    inputs: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten a ``(B, T)`` batch into the ``(B*T,)`` shape the forward takes.

    The wrapper is a single-sequence entry point -- it adds and removes its own
    batch dim around the decoder call. The synthetic source yields one document
    per row of length ``max_seq_len``, so the concatenation is exactly the
    single causal document the fallback attention path expects; RoPE is driven
    per row because positions restart at each row boundary.

    A packed (Grain) batch already arrives as a flat token stream, so for it
    this is the identity -- which is what makes the two sources one code path
    from here on. Used by both the training path (``preprocess_inputs``) and the
    pipeline path, which chunks rows before collapsing; keeping it in one place
    is what makes the two chunkings agree.
    """
    return inputs.reshape(-1), labels.reshape(-1)


def document_shift(labels: torch.Tensor, *, seq_len: int) -> torch.Tensor:
    """Next-token targets within a row, ``IGNORE_INDEX`` at each row end.

    The synthetic source hands over labels equal to its inputs, so the shift is
    the model's. Rows are independent documents of length ``seq_len``, so the
    shift has to stay *within* a row: the row-final position would predict the
    next document's first token, which the model had no context for. Those
    positions come back ``IGNORE_INDEX`` and are excluded from the loss.

    The packed source arrives already shifted and already masked at its document
    boundaries, so it never calls this.
    """
    return next_token_targets(labels.reshape(-1), seq_len=seq_len)
