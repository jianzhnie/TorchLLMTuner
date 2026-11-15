"""Model input target alignment and the shared masked-label convention."""

from __future__ import annotations

import torch

# PyTorch's default cross-entropy ignore index, shared with packed data labels.
IGNORE_INDEX = -100


def next_token_targets(labels: torch.Tensor, *, seq_len: int) -> torch.Tensor:
    """Shift ``labels`` into next-token targets, one document per row.

    ``labels`` is the flat ``(B * T,)`` token stream and ``seq_len`` the row
    length. The model emits ``logits[t]`` predicting ``labels[t + 1]``, and the
    shift is therefore within a *row*, not across the flattened sequence:
    position ``t`` of a row predicts position ``t + 1`` of that same row. The
    last position of every row would predict the first token of the *next*
    document, which is not a prediction the model was given context for, so it
    is marked ``IGNORE_INDEX`` rather than dropped -- keeping the tensor
    rectangular means the ignored entries are excluded from both the loss and
    its denominator (``(targets != IGNORE_INDEX).sum()``), so nothing is
    silently over- or under-counted.
    """
    if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if labels.numel() % seq_len != 0:
        raise ValueError(
            f"next_token_targets got {labels.numel()} labels, which is not a "
            f"whole number of rows of length {seq_len}"
        )
    targets = torch.full_like(labels, IGNORE_INDEX)
    # Row r of the (B, T) layout occupies [r*T, (r+1)*T); its predictions are the
    # positions [r*T + 1, (r+1)*T) and its targets are [r*T + 1, (r+1)*T).
    targets = targets.reshape(-1, seq_len)
    targets[:, :-1] = labels.reshape(-1, seq_len)[:, 1:]
    targets = targets.reshape(labels.shape)
    return targets
