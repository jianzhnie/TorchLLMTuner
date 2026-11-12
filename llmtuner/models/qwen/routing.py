"""Explicit Qwen router precision policies, selected during model construction."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeTopKRouter as ReferenceQwen35Router,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter


def validate_router_compute_dtype(config: PretrainedConfig) -> None:
    """Reject unsupported explicit settings before model allocation."""
    setting = getattr(config, "router_compute_dtype", None)
    if setting not in (None, "float32"):
        raise ValueError("Qwen router_compute_dtype must be float32 or None")


def fp32_topk_route(
    hidden_states: Tensor, weight: Tensor, *, top_k: int, normalize: bool
) -> tuple[Tensor, Tensor, Tensor]:
    """Score and select in FP32 while retaining the parameter storage dtype.

    Match the established explicit FP32 policy: unsorted top-k and stabilized
    normalization. This opt-in policy is separate from each native router's
    ordering, normalization and output dtype. Autocast must not downcast the
    gate matmul; autograd casts gradients back to input and weight storage.
    """
    with torch.autocast(device_type=hidden_states.device.type, enabled=False):
        logits = F.linear(
            hidden_states.reshape(-1, weight.shape[1]).float(), weight.float()
        )
        scores, indices = logits.softmax(-1).topk(top_k, dim=-1, sorted=False)
        if normalize:
            scores = scores / (scores.sum(-1, keepdim=True) + 1e-20)
    return logits, scores, indices


class Qwen3MoeFP32TopKRouter(Qwen3MoeTopKRouter):
    """Qwen3 FP32 routing with its configurable top-k normalization."""

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return fp32_topk_route(
            hidden_states, self.weight, top_k=self.top_k, normalize=self.norm_topk_prob
        )


class Qwen3_5MoeTopKRouter(ReferenceQwen35Router):
    """Own native Qwen3.5 routing, distinct from the opt-in FP32 policy.

    Gate GEMM uses input/weight dtype, softmax uses FP32, top-k is sorted,
    selected scores normalize without epsilon, then cast to the logits dtype.
    """

    def __init__(self, config: PretrainedConfig) -> None:
        nn.Module.__init__(self)
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(hidden_states, self.weight)
        probabilities = F.softmax(logits, dtype=torch.float, dim=-1)
        scores, indices = torch.topk(probabilities, self.top_k, dim=-1)
        scores /= scores.sum(dim=-1, keepdim=True)
        return logits, scores.to(logits.dtype), indices


class Qwen3_5MoeFP32TopKRouter(Qwen3_5MoeTopKRouter):
    """Qwen3.5 FP32 routing always normalizes the selected experts."""

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return fp32_topk_route(
            hidden_states, self.weight, top_k=self.top_k, normalize=True
        )
