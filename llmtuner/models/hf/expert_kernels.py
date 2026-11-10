"""Alternative MoE forward kernels applied during model construction."""

from __future__ import annotations

from types import MethodType

import torch
import torch.nn as nn
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock,
    Qwen3MoeTopKRouter,
)

from ...errors import UnsupportedCombinationError
from ...parallel.expert_parallel.token_dispatch import LocalTokenDispatcher
from ..common.moe.gate import RouterGateLinearFunction
from ..common.scatter_add import deterministic_scatter_add
from ..qwen.qwen3_5.experts import (
    _forward_pre_down_grouped_experts as _forward_pre_down_grouped_experts,
)
from ..qwen.qwen3_5.experts import (
    _validate_pre_down_grouped_inputs as _validate_pre_down_grouped_inputs,
)
from ..qwen.qwen3_5.experts import (
    forward_pre_down_experts as forward_pre_down_experts,
)
from ..qwen.qwen3_5.experts import (
    forward_pre_down_moe as forward_pre_down_moe,
)

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
    modules = [module for module in model.modules() if type(module) is router_type]
    for module in modules:
        if "forward" in vars(module) and (
            getattr(module.forward, "__func__", None) is not forward_fp32_topk_router
        ):
            raise UnsupportedCombinationError(
                "FP32 router cannot replace an external custom forward"
            )
    for module in modules:
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

        routed, _ = torch_npu.npu_moe_token_permute(hidden_states, expert_ids)
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
        token_indices = (
            torch.argsort(expert_ids.reshape(-1), stable=True) // expert_ids.shape[-1]
        )
    else:
        token_indices = metadata.token_indices_experts_sorted_N
    output = deterministic_scatter_add(
        torch.zeros_like(hidden_states),
        token_indices.reshape(-1, 1).expand(-1, hidden_dim),
        routed_output,
    )
    return output.reshape(batch_size, sequence_length, hidden_dim)


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
        if "forward" in vars(module) and (
            getattr(module.forward, "__func__", None) is not forward_torchtitan_moe
        ):
            raise UnsupportedCombinationError(
                "torchtitan experts cannot replace an external custom MoE forward"
            )
    for module in modules:
        module._llmtuner_local_dispatcher = LocalTokenDispatcher(
            module.experts.num_experts, module.gate.top_k
        )
        module.forward = MethodType(forward_torchtitan_moe, module)
