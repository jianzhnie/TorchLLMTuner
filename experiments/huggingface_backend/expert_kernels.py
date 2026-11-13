"""Alternative MoE forward kernels applied during model construction."""

from __future__ import annotations

from types import MethodType

import torch
import torch.nn as nn
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeTopKRouter,
)

from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.common.moe.gate import RouterGateLinearFunction

try:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeTopKRouter,
    )
except ImportError:
    Qwen3_5MoeTopKRouter = None


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
