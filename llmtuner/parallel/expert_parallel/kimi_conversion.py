"""Convert Kimi's individual experts to the native EP execution layout.

Routing and latent projection order follow FSDPTurbo 0af4eca
models/kimi/modeling_kimi_linear.py. Native checkpoints use the grouped expert
FQNs; the source correction bias remains a Parameter, with no automatic update.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from ...accelerator.distributed import get_rank, get_world_size
from ...errors import UnsupportedCombinationError
from ...models.common.activation import SituGLU, SwiGLU
from ...models.common.moe.auxiliary_loss import MicrobatchWiseLoadBalanceLoss
from ...models.common.moe.experts import GroupedExperts, RoutedExperts
from ...models.common.moe.layer import MoE
from ...models.common.moe.routing import TokenChoiceTopKRouter
from ...models.kimi.moe import KimiSparseMoE
from .checkpointing import mark_experts_ep_sharded
from .token_dispatch import AllToAllTokenDispatcher, LocalTokenDispatcher


class KimiRoutedExperts(RoutedExperts):
    """Route in latent space, then normalize/project the combined result.

    The gate and shared expert still consume the original model-width input.
    Latent normalization belongs after weighted combine, never inside experts.
    """

    def __init__(self, experts, dispatcher, source: KimiSparseMoE) -> None:
        super().__init__(experts, dispatcher)
        for name in (
            "routed_expert_down_proj",
            "routed_expert_norm",
            "routed_expert_up_proj",
        ):
            module = getattr(source, name, None)
            if module is not None:
                setattr(self, name, module)

    def forward(self, x_TD, scores_TK, ids_TK, counts_E):
        if hasattr(self, "routed_expert_down_proj"):
            x_TD = self.routed_expert_down_proj(x_TD)
        output = super().forward(x_TD, scores_TK, ids_TK, counts_E)
        if hasattr(self, "routed_expert_norm"):
            output = self.routed_expert_norm(output)
        if hasattr(self, "routed_expert_up_proj"):
            output = self.routed_expert_up_proj(output)
        return output


def convert_kimi_block(
    block: KimiSparseMoE,
    *,
    ep_group: dist.ProcessGroup | None,
    aux_loss_coeff: float | None,
    quantile_balancing: bool,
    token_dispatcher: str,
    tp_enabled: bool,
) -> MoE:
    """Copy one EP expert slice, preserving replicated parameter ownership."""
    if tp_enabled or quantile_balancing or token_dispatcher != "alltoall":
        raise UnsupportedCombinationError(
            "Kimi EP currently requires alltoall dispatch, TP=1 and "
            "moe_quantile_balancing=False; other layout/backend adapters "
            "are not implemented"
        )
    config = block.config
    if config.hidden_act not in ("silu", "situ"):
        raise UnsupportedCombinationError(
            f"Kimi EP activation {config.hidden_act!r} is not implemented"
        )
    size = 1 if ep_group is None else get_world_size(ep_group)
    rank = 0 if ep_group is None else get_rank(ep_group)
    count = len(block.experts)
    if count % size:
        raise ValueError(f"EP degree {size} must divide Kimi num_experts={count}")
    local_count = count // size
    start = rank * local_count
    activation = (
        SituGLU(
            (
                1.0
                if config.activation_situ_beta is None
                else config.activation_situ_beta
            ),
            config.activation_situ_linear_beta,
        )
        if config.hidden_act == "situ"
        else SwiGLU()
    )
    first = block.experts[0]
    experts = GroupedExperts(
        first.w1.in_features,
        first.w1.out_features,
        local_count,
        activation_fn=activation,
    ).to(device=first.w1.weight.device, dtype=first.w1.weight.dtype)
    for source_name, target_name in (
        ("w1", "w1_EFD"),
        ("w2", "w2_EDF"),
        ("w3", "w3_EFD"),
    ):
        source = [getattr(expert, source_name).weight for expert in block.experts]
        if len({weight.requires_grad for weight in source}) != 1:
            raise UnsupportedCombinationError(
                "Kimi EP cannot stack experts with different requires_grad flags "
                f"for {source_name}"
            )
        target = getattr(experts, target_name)
        target.requires_grad_(source[0].requires_grad)
        with torch.no_grad():
            target.copy_(torch.stack(source[start : start + local_count]))
    grouped = config.num_expert_group > config.topk_group
    router = TokenChoiceTopKRouter(
        num_experts=count,
        dim=block.gate.weight.shape[1],
        top_k=block.gate.top_k,
        score_func=config.moe_router_activation_func,
        route_norm=config.moe_renormalize and block.gate.top_k > 1,
        route_scale=config.routed_scaling_factor,
        num_expert_groups=config.num_expert_group if grouped else None,
        num_limited_groups=config.topk_group if grouped else None,
        aux_loss=(
            MicrobatchWiseLoadBalanceLoss(coeff=aux_loss_coeff)
            if aux_loss_coeff
            else None
        ),
    ).to(device=block.gate.weight.device, dtype=block.gate.weight.dtype)
    # No dtype conversion of reused modules: preserve mixed precision and ties.
    router.gate.weight = block.gate.weight
    dispatcher = (
        LocalTokenDispatcher(count, block.gate.top_k)
        if ep_group is None
        else AllToAllTokenDispatcher(count, block.gate.top_k)
    )
    dispatcher.set_ep_group(ep_group=ep_group)
    if ep_group is not None:
        mark_experts_ep_sharded(experts, ep_group)
    moe = MoE(
        count,
        KimiRoutedExperts(experts, dispatcher, block),
        router,
        load_balance_coeff=None,
        shared_experts=block.shared_experts,
    )
    # Kimi stores this non-differentiable selection state as a Parameter. Keep
    # its identity, dtype and freeze state; don't enable the native sign update.
    moe.expert_bias_E = block.gate.e_score_correction_bias
    moe.training = block.training
    moe.routed_experts.training = block.training
    router.train(block.training)
    experts.train(block.training)
    moe.tokens_per_expert_E = moe.tokens_per_expert_E.to(block.gate.weight.device)
    return moe
