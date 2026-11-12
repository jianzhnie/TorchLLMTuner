# Copyright 2026 The DeepSeek-AI Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Transformers 5.11 modeling_deepseek_v4.
"""Shared Ca/Cb window layout for CSA and its indexer."""

import torch


def assemble_overlap_windows(
    window_kv: torch.Tensor, window_gate: torch.Tensor, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine prior-window Ca with current Cb; leave initial Ca invalid.

    Inputs are [batch, windows, ratio, 2*head_dim]. The caller owns inserting
    a prior-call Ca into the first window and all cache mutation.
    """
    batch, windows, ratio, _ = window_kv.shape
    values = window_kv.new_zeros((batch, windows, 2 * ratio, head_dim))
    gates = window_gate.new_full((batch, windows, 2 * ratio, head_dim), float("-inf"))
    values[:, :, ratio:] = window_kv[..., head_dim:]
    gates[:, :, ratio:] = window_gate[..., head_dim:]
    if windows > 1:
        values[:, 1:, :ratio] = window_kv[:, :-1, :, :head_dim]
        gates[:, 1:, :ratio] = window_gate[:, :-1, :, :head_dim]
    return values, gates
