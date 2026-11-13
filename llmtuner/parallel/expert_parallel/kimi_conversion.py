"""Convert Kimi's individual experts to the native EP execution layout.

Routing and latent projection order follow FSDPTurbo 0af4eca
models/kimi/modeling_kimi_linear.py. Native checkpoints use the grouped expert
FQNs; the source correction bias remains a Parameter, with no automatic update.
"""

from __future__ import annotations

import torch.distributed as dist

from ...accelerator.distributed import get_rank, get_world_size
from ...models.common.moe.layer import MoE
from ...models.kimi.kimi_k3.moe import KimiRoutedExperts, KimiSparseMoE
from .checkpointing import mark_experts_ep_sharded
from .token_dispatch import AllToAllTokenDispatcher, LocalTokenDispatcher


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
    block.validate_expert_parallel(
        degree=size,
        tp_enabled=tp_enabled,
        quantile_balancing=quantile_balancing,
        token_dispatcher=token_dispatcher,
    )
    rank = 0 if ep_group is None else get_rank(ep_group)
    count = len(block.experts)
    local_count = count // size
    start = rank * local_count
    experts = block.build_parallel_experts(start=start, count=local_count)
    router = block.gate.build_parallel_router(aux_loss_coefficient=aux_loss_coeff)
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
