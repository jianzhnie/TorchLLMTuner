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
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeExperts,
        Qwen3_5MoeSparseMoeBlock,
        Qwen3_5MoeTopKRouter,
    )
except ImportError:
    Qwen3_5MoeExperts = None
    Qwen3_5MoeSparseMoeBlock = None
    Qwen3_5MoeTopKRouter = None

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
    # Native Qwen3.5 always normalizes and has no norm_topk_prob attribute.
    # Qwen3-MoE keeps its configurable normalization behavior.
    if getattr(module, "norm_topk_prob", True):
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    return router_logits, scores, indices


def stabilize_moe_router(model: nn.Module) -> None:
    family = getattr(model.config, "model_type", None)
    compute_dtype = getattr(model.config, "router_compute_dtype", None)
    if compute_dtype not in (None, "float32"):
        raise ValueError("router_compute_dtype must be float32 or None")
    if family == "qwen3_moe":
        router_type = Qwen3MoeTopKRouter
    elif family == "qwen3_5_moe_text" and compute_dtype == "float32":
        if Qwen3_5MoeTopKRouter is None:
            raise ImportError("FP32 Qwen3.5 router requires native HF Qwen3.5 MoE")
        router_type = Qwen3_5MoeTopKRouter
    elif compute_dtype is not None:
        raise ValueError(
            "router_compute_dtype=float32 supports qwen3_moe/qwen3_5_moe_text only"
        )
    else:
        return
    for module in model.modules():
        if type(module) is router_type:
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


def stabilize_moe_experts(model: nn.Module) -> None:
    if getattr(model.config, "experts_implementation", "native") == "pre_down":
        if (
            model.config.model_type != "qwen3_5_moe_text"
            or model.config.hidden_act != "silu"
        ):
            raise ValueError("pre_down experts require Qwen3.5 MoE with SiLU")
        modules = [m for m in model.modules() if type(m) is Qwen3_5MoeExperts]
        if not modules:
            raise ValueError("pre_down requires native HF Qwen3.5 experts")
        blocks = [m for m in model.modules() if type(m) is Qwen3_5MoeSparseMoeBlock]
        if not blocks or any(
            type(block.experts) is not Qwen3_5MoeExperts for block in blocks
        ):
            raise ValueError("pre_down requires unmodified native HF MoE blocks")
        for module in modules:
            if (
                "forward" in vars(module)
                and getattr(module.forward, "__func__", None)
                is not forward_pre_down_experts
            ):
                raise ValueError(
                    "pre_down cannot replace external custom expert forwards"
                )
        for block in blocks:
            if (
                "forward" in vars(block)
                and getattr(block.forward, "__func__", None) is not forward_pre_down_moe
            ) or (
                "forward" in vars(block.experts)
                and getattr(block.experts.forward, "__func__", None)
                is not forward_pre_down_experts
            ):
                raise ValueError("pre_down cannot replace external custom MoE forwards")
        for module in modules:
            module.forward = MethodType(forward_pre_down_experts, module)
        for block in blocks:
            block.forward = MethodType(forward_pre_down_moe, block)
        return
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
