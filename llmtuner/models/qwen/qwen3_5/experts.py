"""Qwen3.5 pre-down score multiplication over HF-owned expert parameters."""

from __future__ import annotations

import torch
import torch.nn as nn

from ....parallel.expert_parallel.token_dispatch import LocalTokenDispatcher
from ...common.scatter_add import deterministic_scatter_add


def _validate_pre_down_grouped_inputs(module, hidden_states, indices, scores):
    """Validate the eager local BF16 grouped-GEMM contract before dispatch."""
    if torch.compiler.is_compiling():
        raise ValueError(
            "pre_down NPU grouped experts currently require eager execution"
        )
    gate_up, down = module.gate_up_proj, module.down_proj
    tensors = (hidden_states, gate_up, down, indices, scores)
    if any(type(tensor) not in (torch.Tensor, nn.Parameter) for tensor in tensors):
        raise TypeError("pre_down grouped experts require local materialized tensors")
    if any(tensor.device != hidden_states.device for tensor in tensors):
        raise ValueError("pre_down grouped inputs and parameters must share a device")
    if any(t.dtype != torch.bfloat16 for t in (hidden_states, gate_up, down)):
        raise TypeError("pre_down grouped activations and parameters must be BF16")
    if indices.dtype != torch.int64 or scores.dtype not in (
        torch.float32,
        torch.bfloat16,
    ):
        raise TypeError("pre_down grouped requires int64 IDs and FP32/BF16 scores")
    if hidden_states.ndim != 2 or gate_up.ndim != 3 or down.ndim != 3:
        raise ValueError("pre_down grouped expects 2D activations and 3D weights")
    experts, twice_hidden, dim = gate_up.shape
    hidden = twice_hidden // 2
    if (
        experts != module.num_experts
        or twice_hidden % 2
        or dim < 8
        or hidden < 8
        or dim % 8
        or hidden % 8
        or down.shape != (experts, dim, hidden)
        or hidden_states.shape[1] != dim
    ):
        raise ValueError("pre_down grouped requires matching D/F geometry aligned to 8")
    if (
        indices.ndim != 2
        or scores.shape != indices.shape
        or indices.shape[0] != hidden_states.shape[0]
        or not 0 < indices.shape[1] <= experts
    ):
        raise ValueError(
            "pre_down grouped requires matching (tokens, top_k) IDs/scores"
        )
    if any(
        not tensor.is_contiguous() or tensor.data_ptr() % 16
        for tensor in (hidden_states, gate_up, down)
    ):
        raise ValueError(
            "pre_down grouped requires contiguous 16-byte aligned operands"
        )
    if hidden_states.shape[0] == 0:
        raise ValueError(
            "pre_down grouped requires nonempty tokens; empty experts are OK"
        )


def _forward_pre_down_grouped_experts(
    module, hidden_states, top_k_index, top_k_weights
):
    """Local dispatch and separate grouped GEMMs over per-forward HF views.

    This private compute path also permits CPU logical tests with a grouped-MM
    stub; production selects it only for NPU and never falls back to a loop.
    Empty expert segments are passed to the native grouped operator unchanged.
    """
    _validate_pre_down_grouped_inputs(module, hidden_states, top_k_index, top_k_weights)
    counts = torch.nn.functional.one_hot(top_k_index, module.num_experts).sum((0, 1))
    dispatcher = LocalTokenDispatcher(module.num_experts, top_k_index.shape[1])
    routed, counts, metadata = dispatcher.dispatch(
        hidden_states, top_k_weights, top_k_index, counts
    )
    offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
    # Never cache views: FSDP can replace/materialize parameter storage each forward.
    gate_weight, up_weight = module.gate_up_proj.chunk(2, dim=1)
    gate = torch._grouped_mm(routed, gate_weight.transpose(-2, -1), offs=offsets)
    up = torch._grouped_mm(routed, up_weight.transpose(-2, -1), offs=offsets)
    hidden = module.act_fn(gate) * up
    hidden = (
        hidden.float() * metadata.topk_scores_experts_sorted_N.float().reshape(-1, 1)
    ).to(hidden.dtype)
    output = torch._grouped_mm(hidden, module.down_proj.transpose(-2, -1), offs=offsets)
    # LocalTokenDispatcher.combine applies post-down scores; skip that multiply.
    return deterministic_scatter_add(
        torch.zeros_like(hidden_states),
        metadata.token_indices_experts_sorted_N.reshape(-1, 1).expand(
            -1, hidden_states.shape[-1]
        ),
        output,
    )


def forward_pre_down_experts(module, hidden_states, top_k_index, top_k_weights):
    """FP32 score multiplication before the BF16 down GEMM, opt-in only.

    CPU retains the original per-expert loop. NPU uses local token dispatch,
    separate gate/up grouped GEMMs and deterministic scatter combine. Original
    HF parameters, router and shared branch remain owned by HF. This changes
    arithmetic/gradient accumulation, not a bitwise native HF equivalence claim.
    Only isolated first-order BF16 comparisons exist; no full-training claim.
    """
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError("pre_down experts require BF16 activations")
    if hidden_states.device.type == "npu":
        return _forward_pre_down_grouped_experts(
            module, hidden_states, top_k_index, top_k_weights
        )
    if hidden_states.device.type != "cpu":
        raise ValueError("pre_down experts currently support CPU and NPU only")
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(
            top_k_index, num_classes=module.num_experts
        ).permute(2, 1, 0)
        expert_hit = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero()
    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == module.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        gate, up = torch.nn.functional.linear(
            hidden_states[token_idx], module.gate_up_proj[expert_idx]
        ).chunk(2, dim=-1)
        hidden = module.act_fn(gate) * up
        hidden = (
            hidden.float() * top_k_weights[token_idx, top_k_pos, None].float()
        ).to(hidden.dtype)
        output = torch.nn.functional.linear(hidden, module.down_proj[expert_idx])
        final_hidden_states.index_add_(
            0, token_idx, output.to(final_hidden_states.dtype)
        )
    return final_hidden_states


def forward_pre_down_moe(module, hidden_states):
    """Route first, then evaluate complete shared branch over the HF flat view.

    Branch construction order is part of the opt-in BF16 backward arithmetic.
    Reuse every original HF submodule/parameter; no cloned branch inputs.
    """
    batch, length, dim = hidden_states.shape
    flat = hidden_states.view(-1, dim)
    _, scores, ids = module.gate(flat)
    routed = module.experts(flat, ids, scores)
    shared = module.shared_expert(flat)
    shared = torch.sigmoid(module.shared_expert_gate(flat)) * shared
    return (routed + shared).view(batch, length, dim)
