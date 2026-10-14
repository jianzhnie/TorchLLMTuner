"""Construct an EP-capable MoE from a probed HF block and copy its weights."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from ...accelerator.distributed import get_rank, get_world_size
from ...models.common.aux_loss import AuxLoss
from ...models.common.moe.auxiliary_loss import MicrobatchWiseLoadBalanceLoss
from ...models.common.moe.experts import (
    DominoRoutedExperts,
    GroupedExperts,
    RoutedExperts,
)
from ...models.common.moe.layer import MoE
from ...models.common.moe.routing import (
    QuantileBalancedTopKRouter,
    TokenChoiceTopKRouter,
)
from .. import compatibility
from .checkpointing import mark_experts_ep_sharded
from .hf_inspection import (
    fused_experts_of,
    read_expert_groups,
    read_route_norm,
    read_route_scale,
    resolve_score_func,
    resolve_shared_expert,
    resolve_top_k,
    router_of,
)
from .token_dispatch import (
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
)
from .torchao_dispatch import TorchAOTokenDispatcher


def restore_fp32_state_buffers(module: nn.Module) -> None:
    """Undo the dtype conversion ``Module.to(dtype=...)`` applies to float buffers.

    ``Module.to`` converts parameters *and* every floating-point buffer, with no
    way to ask for one and not the other. The MoE registers two buffers as fp32
    on purpose -- ``expert_bias_E`` (an additive load-balancing correction that
    would erode under a bf16 round per step) and ``tokens_per_expert_E`` (a
    token *count*, which bf16 cannot hold exactly past 256: 1001 becomes 1000).
    Neither has a gradient, so casting them buys nothing and silently costs
    precision. Upstream keeps both fp32 unconditionally.

    Restore only buffers owned by MoE statistics and auxiliary losses. Shared
    experts are reused from the HF block and retain their original dtypes.
    """
    for child in module.modules():
        if isinstance(child, MoE | AuxLoss):
            for buffer in child.buffers(recurse=False):
                if torch.is_floating_point(buffer):
                    buffer.data = buffer.data.to(torch.float32)


def convert_block(
    block: nn.Module,
    *,
    ep_group: dist.ProcessGroup | None,
    aux_loss_coeff: float | None,
    load_balance_coeff: float | None,
    quantile_balancing: bool,
    token_dispatcher: str,
    torchao_pad_multiple: int,
    tp_enabled: bool = False,
) -> MoE:
    """Build the llmtuner MoE for one HF block and move its weights over."""
    ep_size = 1 if ep_group is None else get_world_size(ep_group)
    ep_rank = 0 if ep_group is None else get_rank(ep_group)

    router_gate = router_of(block)
    assert router_gate is not None  # the probe established this
    if getattr(router_gate, "bias", None) is not None:
        compatibility.router_bias(router_gate)
    _, shared = resolve_shared_expert(block)
    if shared is not None and hasattr(block, "shared_expert_gate"):
        compatibility.shared_expert_gate(block)
    if shared is not None and tp_enabled:
        compatibility.shared_expert_tp_ep(block)
    experts = fused_experts_of(block)
    assert experts is not None  # the probe established this

    num_experts = experts.num_experts
    top_k = resolve_top_k(block)
    assert top_k is not None  # the probe established this
    dim = router_gate.weight.shape[1]
    hidden = experts.gate_EFD.shape[1]
    if num_experts % ep_size != 0:
        raise ValueError(
            f"EP degree {ep_size} does not divide num_experts={num_experts}; "
            "each EP rank must hold the same number of experts."
        )
    num_local = num_experts // ep_size
    lo = ep_rank * num_local

    num_expert_groups, num_limited_groups = read_expert_groups(block, router_gate)
    grouped = GroupedExperts(dim, hidden, num_local)
    score_func = resolve_score_func(block, router_gate)
    router_options = {
        "num_experts": num_experts,
        "dim": dim,
        "top_k": top_k,
        "route_norm": read_route_norm(block, router_gate),
        "route_scale": read_route_scale(block, router_gate),
        "aux_loss": (
            MicrobatchWiseLoadBalanceLoss(coeff=aux_loss_coeff)
            if aux_loss_coeff
            else None
        ),
    }
    if quantile_balancing:
        # The quantile scheme is defined over sigmoid scores (the histogram
        # range derives from their [0, 1] bound) and routes freely over all
        # experts, so a softmax family or a group-limited one cannot adopt it.
        if score_func != "sigmoid":
            compatibility.quantile_requires_sigmoid(score_func, block)
        if num_expert_groups is not None and num_expert_groups > 1:
            compatibility.quantile_no_group_limit(block)
        router = QuantileBalancedTopKRouter(
            **router_options,
        )
        # The quantile update owns expert_bias_E; the sign-based update is
        # off (MoE registers the buffer for a quantile router regardless).
        load_balance_coeff = None
    else:
        router = TokenChoiceTopKRouter(
            **router_options,
            score_func=score_func,
            num_expert_groups=num_expert_groups,
            num_limited_groups=num_limited_groups,
        )
    if token_dispatcher == "torchao":
        # Optional-import adapter: the constructor raises ImportError with an
        # install hint when torchao is absent. EP=1 is supported by the
        # dispatcher itself (local padded permute only).
        dispatcher = TorchAOTokenDispatcher(
            num_experts, top_k, pad_multiple=torchao_pad_multiple
        )
    elif ep_group is None:
        dispatcher = LocalTokenDispatcher(num_experts, top_k)
    else:
        dispatcher = AllToAllTokenDispatcher(num_experts, top_k)
    if ep_group is not None:
        dispatcher.set_ep_group(ep_group=ep_group)
        # Flag the stacked expert weights as expert-dim-sharded over the EP
        # group so the checkpointer can all-gather them on save and slice
        # them back on load (see expert_parallel/checkpointing.py).
        mark_experts_ep_sharded(grouped, ep_group)

    routed_experts_cls = (
        DominoRoutedExperts if token_dispatcher == "domino" else RoutedExperts
    )
    moe = MoE(
        num_experts=num_experts,
        routed_experts=routed_experts_cls(grouped, dispatcher),
        router=router,
        # Auxiliary-loss-free load balancing, mirroring the HF block's own
        # e_score_correction_bias. Off for models that carry no such buffer --
        # a frozen zero bias is dead state in the checkpoint.
        load_balance_coeff=load_balance_coeff,
    )

    # Match the block's dtype/device before the copies so they are exact, and
    # keep its mode: a fresh module defaults to training=True, which would
    # flip eval-built models (aux-loss injection, token counting) back on.
    moe.to(dtype=router_gate.weight.dtype, device=router_gate.weight.device)
    restore_fp32_state_buffers(moe)
    moe.train(block.training)
    router.gate.weight.requires_grad_(router_gate.weight.requires_grad)
    grouped.w1_EFD.requires_grad_(experts.gate_EFD.requires_grad)
    grouped.w3_EFD.requires_grad_(experts.up_EFD.requires_grad)
    grouped.w2_EDF.requires_grad_(experts.down_EDF.requires_grad)
    with torch.no_grad():
        router.gate.weight.copy_(router_gate.weight)
        # HF keeps all E experts in two stacked parameters; each rank keeps its
        # own slice of them. ``gate_EFD``/``up_EFD`` are the two halves of
        # ``gate_up_proj`` split along its output dim -- see ``fused_experts_of``.
        grouped.w1_EFD.copy_(experts.gate_EFD[lo : lo + num_local])
        grouped.w3_EFD.copy_(experts.up_EFD[lo : lo + num_local])
        grouped.w2_EDF.copy_(experts.down_EDF[lo : lo + num_local])
        # The load-balancing bias is optimization state, not a learned weight:
        # it is copied so a resumed or swapped run keeps the balance the HF
        # model had reached.
        bias = getattr(router_gate, "e_score_correction_bias", None)
        if bias is not None and moe.expert_bias_E is not None:
            moe.expert_bias_E.copy_(bias.float())
    # Reuse the HF shared expert after initializing the new subtree. Casting
    # or setting the mode recursively must not mutate this original module.
    moe.shared_experts = shared
    return moe
