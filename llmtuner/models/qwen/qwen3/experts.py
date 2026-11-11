"""Qwen3-MoE grouped expert execution over HF-owned parameters."""

from __future__ import annotations

import torch
import torch.nn as nn

from ....accelerator.npu_ops import moe_token_permute
from ...common.scatter_add import deterministic_scatter_add


def forward_torchtitan_moe(module: nn.Module, hidden_states: torch.Tensor):
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError("experts_implementation='torchtitan' requires BF16 inputs")
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    _, scores, expert_ids = module.gate(hidden_states)
    counts = torch.nn.functional.one_hot(
        expert_ids, num_classes=module.experts.num_experts
    ).sum(dim=(0, 1))
    npu_dispatch = hidden_states.device.type == "npu"
    if npu_dispatch:
        routed = moe_token_permute(hidden_states, expert_ids)
        routed_scores = moe_token_permute(
            scores.reshape(-1, 1), expert_ids.reshape(-1, 1)
        )
    else:
        dispatcher = module._llmtuner_local_dispatcher
        routed, counts, metadata = dispatcher.dispatch(
            hidden_states, scores, expert_ids, counts
        )
        routed_scores = metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
    offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
    gate_weight, up_weight = module.experts.gate_up_proj.chunk(2, dim=1)
    gate = torch._grouped_mm(
        routed.bfloat16(),
        gate_weight.bfloat16().transpose(-2, -1),
        offs=offsets,
    )
    up = torch._grouped_mm(
        routed.bfloat16(), up_weight.bfloat16().transpose(-2, -1), offs=offsets
    )
    hidden = module.experts.act_fn(gate) * up
    hidden = (hidden.float() * routed_scores.float()).to(hidden.dtype)
    routed_output = torch._grouped_mm(
        hidden,
        module.experts.down_proj.bfloat16().transpose(-2, -1),
        offs=offsets,
    ).type_as(routed)
    if npu_dispatch:
        # Match the reference's expert-ordered deterministic accumulation.
        # Unpermute followed by a top-k sum uses a different reduction tree.
        # Integer argsort falls back to AI CPU on NPU. Expert IDs are exactly
        # representable in float32; stable sorting preserves the reduction order.
        sort_keys = expert_ids.reshape(-1)
        if module.experts.num_experts <= 2**24:
            sort_keys = sort_keys.float()
        token_indices = torch.argsort(sort_keys, stable=True) // expert_ids.shape[-1]
    else:
        token_indices = metadata.token_indices_experts_sorted_N
    output = deterministic_scatter_add(
        torch.zeros_like(hidden_states),
        token_indices.reshape(-1, 1).expand(-1, hidden_dim),
        routed_output,
    )
    return output.reshape(batch_size, sequence_length, hidden_dim)
