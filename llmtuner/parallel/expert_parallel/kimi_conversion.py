"""Convert Kimi's individual experts to the native EP execution layout.

The model owns routing, latent projection order and replicated state binding.
This adapter selects an expert slice and binds its communication/ownership.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.distributed as dist

from ...accelerator.distributed import get_rank, get_world_size
from ...models.common.moe.layer import MoE
from .checkpointing import mark_experts_ep_sharded
from .token_dispatch import AllToAllTokenDispatcher, LocalTokenDispatcher

if TYPE_CHECKING:
    from ...models.kimi.kimi_k3.moe import KimiSparseMoE


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
    if ep_group is not None:
        mark_experts_ep_sharded(experts, ep_group)
    return block.build_parallel_moe(
        experts=experts, router=router, dispatcher=dispatcher
    )
