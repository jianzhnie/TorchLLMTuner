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
from ...models.common.moe.auxiliary_loss import MicrobatchWiseLoadBalanceLoss
from ...models.common.moe.experts import GroupedExperts
from ...models.common.moe.layer import MoE
from ...models.common.moe.routing import TokenChoiceTopKRouter
from ...models.kimi.kimi_k3.moe import KimiRoutedExperts, KimiSparseMoE
from .checkpointing import mark_experts_ep_sharded
from .token_dispatch import AllToAllTokenDispatcher, LocalTokenDispatcher


def validate_kimi_conversion(
    block: KimiSparseMoE,
    *,
    degree: int,
    tp_enabled: bool,
    quantile_balancing: bool,
    token_dispatcher: str,
) -> None:
    """Check Kimi EP capabilities without allocation, mutation or collectives."""
    if type(degree) is not int or degree < 1:
        raise ValueError("Kimi EP degree must be a positive integer")
    if tp_enabled or quantile_balancing or token_dispatcher != "alltoall":
        raise UnsupportedCombinationError(
            "Kimi EP currently requires alltoall dispatch, TP=1 and "
            "moe_quantile_balancing=False; other layout/backend adapters "
            "are not implemented"
        )
    block.validate_expert_geometry()
    block.validate_grouped_activation()
    count = len(block.experts)
    if count == 0 or count % degree:
        raise ValueError(f"EP degree {degree} must divide Kimi num_experts={count}")
    for name in ("w1", "w2", "w3"):
        projections = [getattr(expert, name) for expert in block.experts]
        if any(projection.bias is not None for projection in projections):
            raise UnsupportedCombinationError(
                f"Kimi EP does not support expert projection bias for {name}"
            )
        weights = [projection.weight for projection in projections]
        if len({(weight.dtype, weight.device) for weight in weights}) != 1:
            raise UnsupportedCombinationError(
                "Kimi EP cannot stack experts with different dtypes or devices "
                f"for {name}; one grouped parameter requires uniform storage"
            )
        if len({weight.requires_grad for weight in weights}) != 1:
            raise UnsupportedCombinationError(
                "Kimi EP cannot stack experts with different requires_grad flags "
                f"for {name}"
            )


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
    size = 1 if ep_group is None else get_world_size(ep_group)
    validate_kimi_conversion(
        block,
        degree=size,
        tp_enabled=tp_enabled,
        quantile_balancing=quantile_balancing,
        token_dispatcher=token_dispatcher,
    )
    rank = 0 if ep_group is None else get_rank(ep_group)
    count = len(block.experts)
    local_count = count // size
    start = rank * local_count
    first = block.experts[0]
    activation = block.build_grouped_activation()
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
        target = getattr(experts, target_name)
        # Different projections may retain different precision under autocast.
        # Copying every projection into w1's dtype would change optimizer state.
        target.data = target.data.to(device=source[0].device, dtype=source[0].dtype)
        target.requires_grad_(source[0].requires_grad)
        with torch.no_grad():
            target.copy_(torch.stack(source[start : start + local_count]))
    # The instantiated gate owns routing math. Its config may have been edited
    # after construction without changing the gate's actual forward semantics.
    gate = block.gate
    grouped = gate.num_expert_group > gate.topk_group
    router = TokenChoiceTopKRouter(
        num_experts=count,
        dim=block.gate.weight.shape[1],
        top_k=block.gate.top_k,
        score_func=gate.moe_router_activation_func,
        route_norm=gate.moe_renormalize and gate.top_k > 1,
        route_scale=gate.routed_scaling_factor,
        num_expert_groups=gate.num_expert_group if grouped else None,
        num_limited_groups=gate.topk_group if grouped else None,
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
    # Official Kimi combines weighted expert outputs before the BF16 cast.
    dispatcher.accumulate_in_fp32 = True
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
