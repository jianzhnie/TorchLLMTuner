"""Wire EP onto a model: swap every HF MoE block for the llmtuner MoE."""

from __future__ import annotations

import torch.distributed as dist
import torch.nn as nn

from llmtuner.config.parallel_config import ParallelConfig

from ...utils.logging import get_logger
from .. import compatibility
from .domino_dispatch import DominoRoutedExperts
from .model_replacement import MoEReplacementObserver, swap_hf_moe_blocks

logger = get_logger(__name__)

__all__ = ["apply_ep"]


def apply_ep(
    model: nn.Module,
    cfg: ParallelConfig,
    *,
    ep_group: dist.ProcessGroup | None = None,
    error_group: dist.ProcessGroup | None = None,
    on_replacement: MoEReplacementObserver | None = None,
) -> nn.Module:
    """Swap every HF MoE block for the EP-capable llmtuner MoE.

    EP=1 keeps the HF model unless fixed_router requests native local routing.

    error_group, when supplied, spans every rank sharing later stage-local
    collectives. Domino and MC2 use it for failure coordination; standalone callers
    without it coordinate only their EP group.

    on_replacement receives explicit conversion mappings before installation,
    allowing assembly to rebind programmatic FSDP overrides.

    The swap moves weights, so it must happen before FSDP wraps the model.
    With a multi-rank group each rank keeps ``num_experts / ep`` of them and
    tokens cross ranks by all-to-all; the two are orthogonal to the CP
    attention kernel, so CP and EP compose freely.
    """
    if cfg.ep_size == 1:
        if cfg.ep_config.moe_quantile_balancing:
            compatibility.quantile_requires_ep()
        if not cfg.ep_config.fixed_router:
            return model
        # Local fixed routing needs no process group. Never use WORLD merely
        # because this model runs within a larger DP or PP job.
        ep_group = None

    if cfg.ep_size > 1 and (ep_group is None or ep_group.size() != cfg.ep_size):
        raise ValueError(
            f"ep={cfg.ep_size} requires an EP process group of that size, got "
            f"{None if ep_group is None else ep_group.size()}. The group comes "
            "from the sparse mesh's 'ep' axis (parallelize_hf_transformers "
            "resolves it from parallel_dims)."
        )
    swapped = swap_hf_moe_blocks(
        model,
        ep_group=ep_group,
        router_aux_loss_coef=cfg.ep_config.router_aux_loss_coef,
        quantile_balancing=cfg.ep_config.moe_quantile_balancing,
        fixed_router=cfg.ep_config.fixed_router,
        token_dispatcher=cfg.ep_config.ep_token_dispatcher,
        torchao_pad_multiple=cfg.ep_config.ep_torchao_pad_multiple,
        tp_enabled=cfg.tp_size > 1,
        on_replacement=on_replacement,
    )
    # Failures must reach all ranks sharing later FSDP/TP/CP collectives,
    # not only the ranks in one token-exchange group.
    if error_group is not None:
        from .mc2_dispatch import MC2RoutedExperts

        for module in model.modules():
            if isinstance(module, DominoRoutedExperts | MC2RoutedExperts):
                module.error_group = error_group
    logger.info(
        "Applied EP (%s dispatch): swapped %d MoE blocks, degree %d",
        cfg.ep_config.ep_token_dispatcher,
        swapped,
        cfg.ep_size,
    )
    return model
