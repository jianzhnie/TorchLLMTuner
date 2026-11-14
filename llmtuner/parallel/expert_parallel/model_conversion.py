"""Bind model-owned expert construction to generic EP communication.

The model owns projection packing, routing and replicated state. The executor
selects an expert slice and binds dispatcher and checkpoint ownership.
"""

from __future__ import annotations

import torch.distributed as dist
from torch import nn

from ...accelerator.distributed import get_rank, get_world_size
from ...models.common.moe.layer import MoE
from .checkpointing import mark_experts_ep_sharded
from .token_dispatch import AllToAllTokenDispatcher, LocalTokenDispatcher


def uses_model_expert_conversion(block: nn.Module) -> bool:
    """Validate an opted-in model conversion interface without changing state."""
    if not callable(getattr(block, "build_parallel_moe", None)):
        return False
    layout = getattr(block, "expert_parallel_layout", None)
    if not isinstance(layout, str) or not layout:
        raise TypeError("Model-owned EP conversion requires an explicit expert layout")
    for name in (
        "validate_expert_parallel",
        "validate_expert_parallel_ownership",
        "expert_parallel_dimensions",
        "build_parallel_experts",
        "build_parallel_router",
        "expert_parallel_replacements",
        "expert_parallel_math_signature",
    ):
        if not callable(getattr(block, name, None)):
            raise TypeError(
                f"Model-owned EP conversion requires {type(block).__name__}.{name}()"
            )
    return True


def convert_model_block(
    block: nn.Module,
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
    count, top_k = block.expert_parallel_dimensions()
    if (
        type(count) is not int
        or type(top_k) is not int
        or count < 1
        or not 1 <= top_k <= count
        or count % size
    ):
        raise ValueError(
            "Model-owned EP dimensions require valid top-k and equal slices"
        )
    local_count = count // size
    start = rank * local_count
    experts = block.build_parallel_experts(start=start, count=local_count)
    router = block.build_parallel_router(aux_loss_coefficient=aux_loss_coeff)
    dispatcher = (
        LocalTokenDispatcher(count, top_k)
        if ep_group is None
        else AllToAllTokenDispatcher(count, top_k)
    )
    dispatcher.set_ep_group(ep_group=ep_group)
    if ep_group is not None:
        mark_experts_ep_sharded(experts, ep_group)
    return block.build_parallel_moe(
        experts=experts, router=router, dispatcher=dispatcher
    )
