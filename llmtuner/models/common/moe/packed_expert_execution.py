"""Grouped execution for models that store gate/up experts in one tensor."""

from __future__ import annotations

import torch
from torch import nn

from ....kernel.ops.moe import moe_token_permute
from ..scatter_add import deterministic_scatter_add


def forward_packed_experts(
    module: nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Run packed gate/up and down projections with deterministic accumulation.

    The model declares the packed layout; this function only executes the
    already-bound expert storage. NPU uses the kernel dispatcher, while CPU
    keeps the reference local dispatcher used by unit tests.
    """
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError("packed expert execution requires BF16 inputs")
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat_hidden = hidden_states.reshape(-1, hidden_dim)
    _, scores, expert_ids = module.gate(flat_hidden)
    counts = torch.nn.functional.one_hot(
        expert_ids, num_classes=module.experts.num_experts
    ).sum(dim=(0, 1))
    on_npu = hidden_states.device.type == "npu"
    if on_npu:
        routed = moe_token_permute(flat_hidden, expert_ids)
        routed_scores = moe_token_permute(
            scores.reshape(-1, 1), expert_ids.reshape(-1, 1)
        )
    else:
        dispatcher = module._llmtuner_local_dispatcher
        routed, counts, metadata = dispatcher.dispatch(
            flat_hidden, scores, expert_ids, counts
        )
        routed_scores = metadata.topk_scores_experts_sorted_N.reshape(-1, 1)

    offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
    gate_weight, up_weight = module.experts.gate_up_proj.chunk(2, dim=1)
    gate = torch._grouped_mm(
        routed.bfloat16(), gate_weight.bfloat16().transpose(-2, -1), offs=offsets
    )
    up = torch._grouped_mm(
        routed.bfloat16(), up_weight.bfloat16().transpose(-2, -1), offs=offsets
    )
    expert_hidden = module.experts.act_fn(gate) * up
    expert_hidden = (expert_hidden.float() * routed_scores.float()).to(
        expert_hidden.dtype
    )
    routed_output = torch._grouped_mm(
        expert_hidden,
        module.experts.down_proj.bfloat16().transpose(-2, -1),
        offs=offsets,
    ).type_as(routed)
    if on_npu:
        sort_keys = expert_ids.reshape(-1)
        if module.experts.num_experts <= 2**24:
            sort_keys = sort_keys.float()
        token_indices = torch.argsort(sort_keys, stable=True) // expert_ids.shape[-1]
    else:
        token_indices = metadata.token_indices_experts_sorted_N
    output = deterministic_scatter_add(
        torch.zeros_like(flat_hidden),
        token_indices.reshape(-1, 1).expand(-1, hidden_dim),
        routed_output,
    )
    return output.reshape(batch_size, sequence_length, hidden_dim)
