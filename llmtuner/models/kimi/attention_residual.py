# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
"""Portable Kimi attention residual aggregation.

Adapted from FSDPTurbo 0af4eca ops/cpu/attn_res.py. Values keep their
original magnitude; RMS normalization is used only for scoring block choices.
"""

import torch


def attention_residual(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    projection_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Mix historical blocks and the current prefix independently per token."""
    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    full_precision = values.float()
    normalized = full_precision * torch.rsqrt(
        full_precision.square().mean(-1, keepdim=True) + eps
    )
    score_weight = norm_weight.float() * projection_weight.squeeze(0).float()
    probabilities = (normalized * score_weight).sum(-1).softmax(-1)
    return (probabilities.unsqueeze(1) @ full_precision).squeeze(1).to(values.dtype)
