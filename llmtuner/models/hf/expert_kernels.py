"""Alternative MoE forward kernels applied during model construction."""

from __future__ import annotations

from types import MethodType

import torch
import torch.nn as nn
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock,
    Qwen3MoeTopKRouter,
)

from ...parallel.expert_parallel.token_dispatch import LocalTokenDispatcher
from ..common.moe.gate import RouterGateLinearFunction
from ..common.scatter_add import deterministic_scatter_add

try:
    import torch_npu
except ImportError as error:
    torch_npu = None
    TORCH_NPU_IMPORT_ERROR = error


def forward_fp32_topk_router(module: nn.Module, hidden_states: torch.Tensor):
    hidden_states = hidden_states.reshape(-1, module.hidden_dim)
    router_logits = RouterGateLinearFunction.apply(hidden_states, module.weight)
    probabilities = torch.nn.functional.softmax(router_logits, dim=-1)
    scores, indices = torch.topk(probabilities, module.top_k, dim=-1, sorted=False)
    if module.norm_topk_prob:
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    return router_logits, scores, indices


def stabilize_moe_router(model: nn.Module) -> None:
    if getattr(model.config, "model_type", None) != "qwen3_moe":
        return
    for module in model.modules():
        if type(module) is Qwen3MoeTopKRouter:
            module.forward = MethodType(forward_fp32_topk_router, module)


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
        if torch_npu is None:
            raise ImportError(
                "NPU kernels require a working torch-npu installation"
            ) from (TORCH_NPU_IMPORT_ERROR)

        routed, permutation = torch_npu.npu_moe_token_permute(hidden_states, expert_ids)
        routed_scores, _ = torch_npu.npu_moe_token_permute(
            scores.reshape(-1, 1), expert_ids.reshape(-1, 1)
        )
    else:
        dispatcher = module._llmtuner_local_dispatcher
        routed, counts, metadata = dispatcher.dispatch(
            hidden_states, scores, expert_ids, counts
        )
        routed_scores = metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
    offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
    projected = torch._grouped_mm(
        routed.bfloat16(),
        module.experts.gate_up_proj.bfloat16().transpose(-2, -1),
        offs=offsets,
    )
    if projected.device.type == "npu":
        if torch_npu is None:
            raise ImportError(
                "NPU kernels require a working torch-npu installation"
            ) from (TORCH_NPU_IMPORT_ERROR)

        hidden = torch_npu.npu_swiglu(projected, dim=-1)
    else:
        gate_values, up_values = projected.chunk(2, dim=-1)
        hidden = module.experts.act_fn(gate_values) * up_values
    hidden = hidden * routed_scores.to(hidden.dtype)
    routed_output = torch._grouped_mm(
        hidden,
        module.experts.down_proj.bfloat16().transpose(-2, -1),
        offs=offsets,
    ).type_as(routed)
    if npu_dispatch:
        output = torch_npu.npu_moe_token_unpermute(routed_output, permutation, None)
        output = output.view(-1, expert_ids.shape[-1], hidden_dim).sum(dim=1)
        return output.reshape(batch_size, sequence_length, hidden_dim)
    output = deterministic_scatter_add(
        torch.zeros_like(hidden_states),
        metadata.token_indices_experts_sorted_N.reshape(-1, 1).expand(-1, hidden_dim),
        routed_output,
    )
    return output.reshape(batch_size, sequence_length, hidden_dim)


def stabilize_moe_experts(model: nn.Module) -> None:
    if getattr(model.config, "experts_implementation", "native") != "torchtitan":
        return
    if getattr(model.config, "hidden_act", None) != "silu":
        raise ValueError("Qwen3-MoE torchtitan experts require hidden_act='silu'")
    modules = [
        module for module in model.modules() if type(module) is Qwen3MoeSparseMoeBlock
    ]
    if not modules:
        raise ValueError("Qwen3-MoE torchtitan experts require routed MoE blocks")
    for module in modules:
        module._llmtuner_local_dispatcher = LocalTokenDispatcher(
            module.experts.num_experts, module.gate.top_k
        )
        module.forward = MethodType(forward_torchtitan_moe, module)
