"""Build EP execution from resolved expert layout and routing declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.distributed as dist
import torch.nn as nn

from ...accelerator.device import device_type
from ...accelerator.distributed import get_rank, get_world_size
from ...accelerator.linear_collectives import all_gather_along, reduce_scatter_along
from ...errors import UnsupportedCombinationError
from ...models.common.aux_loss import AuxLoss
from ...models.common.moe.auxiliary_loss import MicrobatchWiseLoadBalanceLoss
from ...models.common.moe.experts import (
    GroupedExperts,
    RoutedExperts,
)
from ...models.common.moe.layer import MoE
from ...models.common.moe.packed_expert_execution import forward_packed_experts
from ...models.common.moe.routing import (
    QuantileBalancedTopKRouter,
    TokenChoiceTopKRouter,
)
from ...models.contracts import MoEBlockContract
from .. import compatibility
from .checkpointing import mark_experts_ep_sharded
from .domino_dispatch import DominoRoutedExperts
from .mc2_dispatch import MC2GroupedExperts, MC2RoutedExperts
from .npu_baseline_experts import (
    NpuBaselineAllToAllTokenDispatcher,
    NpuBaselineGroupedExperts,
    NpuBaselineRoutedExperts,
)
from .token_dispatch import (
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
    materialize,
)
from .torchao_dispatch import TorchAOTokenDispatcher


class PackedExpertViews(NamedTuple):
    """Views of an explicitly declared fused gate/up expert storage."""

    gate_EFD: torch.Tensor
    up_EFD: torch.Tensor
    down_EDF: torch.Tensor
    num_experts: int


class FeatureParallelMoE(MoE):
    """Keep feature-sharded experts between TP sequence collectives.

    Routing sees the gathered token order on every rank. The staged mask stays
    local so activation checkpoint replay gathers the same microbatch again.
    """

    def forward(
        self, x: torch.Tensor, *, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        mask = self._pending_padding_mask if padding_mask is None else padding_mask
        gathered = all_gather_along(x, -2, self._tp_seq_group)
        if mask is not None:
            mask = mask.reshape(x.shape[:-1])
            # Checkpoint replay may stop before routing consumes this metadata.
            # Drain its gather now so early-stop cannot abandon an async work.
            mask = materialize(all_gather_along(mask, -1, self._tp_seq_group))
        output = super().forward(gathered, padding_mask=mask)
        return reduce_scatter_along(output, -2, self._tp_seq_group)


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


@dataclass(frozen=True, slots=True)
class MoERoutingBinding:
    """Model routing choices resolved before allocating replacement weights."""

    source: nn.Module
    router: nn.Module
    shared: nn.Module | None
    top_k: int
    score_func: str
    route_norm: bool
    route_scale: float
    num_expert_groups: int | None
    num_limited_groups: int | None
    shared_gate: nn.Module | None = None


def resolve_moe_routing(block: nn.Module) -> MoERoutingBinding:
    """Bind routing from the model-owned contract without inspecting forward."""
    contract = getattr(block, "moe_contract", None)
    if contract is not None:
        if not isinstance(contract, MoEBlockContract):
            raise TypeError("moe_contract must be a MoEBlockContract")
        router, _, shared = contract.bind(block)
        return MoERoutingBinding(
            source=block,
            router=router,
            shared=shared,
            shared_gate=(
                block._modules[contract.shared_expert_gate]
                if contract.shared_expert_gate is not None
                else None
            ),
            top_k=contract.top_k,
            score_func=contract.score_func,
            route_norm=contract.route_norm,
            route_scale=contract.route_scale,
            num_expert_groups=contract.num_expert_groups,
            num_limited_groups=contract.num_limited_groups,
        )
    raise UnsupportedCombinationError(
        "MoE conversion requires a model-owned MoEBlockContract"
    )


@dataclass(frozen=True, slots=True)
class PackedExpertBinding:
    """Validated gate/up and down parameters; no copied or stacked storage.

    Resolve after TP, and bind only to the concrete source block. Views are
    created when conversion executes, so current parameter values are copied.
    """

    source: nn.Module
    owner: nn.Module
    gate_up: nn.Parameter
    down: nn.Parameter
    shape: tuple[int, int, int]

    def validate(self, block: nn.Module) -> None:
        """Reject replaced parameters or geometry before allocating conversion state."""
        if (
            block is not self.source
            or block._modules.get("experts") is not self.owner
            or self.owner._parameters.get("gate_up_proj") is not self.gate_up
            or self.owner._parameters.get("down_proj") is not self.down
            or tuple(self.gate_up.shape) != self.shape
            or tuple(self.down.shape)
            != (self.shape[0], self.shape[2], self.shape[1] // 2)
        ):
            raise ValueError("Stale EP packed expert binding")

    def projections(self, block: nn.Module) -> PackedExpertViews:
        """Create only role-specific views of the validated live parameters."""
        self.validate(block)
        hidden = self.shape[1] // 2
        return PackedExpertViews(
            self.gate_up[:, :hidden],
            self.gate_up[:, hidden:],
            self.down,
            self.shape[0],
        )


def resolve_packed_experts(block: nn.Module, *, degree: int) -> PackedExpertBinding:
    """Bind fused parameter storage described by the model contract."""
    if type(degree) is not int or degree < 1:
        raise ValueError("EP degree must be a positive integer")
    contract = getattr(block, "moe_contract", None)
    if contract is None:
        raise UnsupportedCombinationError(
            "EP packed experts require a model-owned MoEBlockContract"
        )
    if not isinstance(contract, MoEBlockContract):
        raise TypeError("moe_contract must be a MoEBlockContract")
    owner = block._modules.get("experts")
    if owner is None:
        raise ValueError("Declared fused experts require an experts child module")
    gate_up = owner._parameters.get("gate_up_proj")
    down = owner._parameters.get("down_proj")
    if not isinstance(gate_up, nn.Parameter) or not isinstance(down, nn.Parameter):
        raise ValueError(
            "Declared fused experts require gate_up_proj and down_proj parameters"
        )
    # Validate rank before indexing geometry, including direct resolver calls.
    if gate_up.ndim != 3 or down.ndim != 3:
        raise ValueError("EP fused gate/up requires rank-3 expert parameters")
    # Explicit contracts own mathematical restrictions. Fused tensors only
    # produce views, never copied parameters.
    if contract.activation != "silu":
        raise UnsupportedCombinationError(
            f"EP grouped experts require SiLU, but the model declares "
            f"{contract.activation!r}; retain the native expert path"
        )
    num_experts = contract.num_experts
    router, _, _ = contract.bind(block)
    weight = router._parameters.get("weight")
    if not isinstance(weight, nn.Parameter) or tuple(weight.shape) != (
        num_experts,
        gate_up.shape[2],
    ):
        raise ValueError(
            "Declared MoE router weight must have shape "
            "[num_experts, hidden_size] matching packed expert storage"
        )
    if len({weight.device, gate_up.device, down.device}) != 1:
        raise UnsupportedCombinationError(
            "Declared MoE router and experts are on different devices"
        )
    if tuple(gate_up.shape) != (
        num_experts,
        gate_up.shape[1],
        gate_up.shape[2],
    ) or tuple(down.shape) != (
        num_experts,
        down.shape[1],
        down.shape[2],
    ):
        raise ValueError("Declared MoE expert count differs from packed storage")
    if gate_up.shape[1] % 2 != 0 or tuple(down.shape[1:]) != (
        gate_up.shape[2],
        gate_up.shape[1] // 2,
    ):
        raise ValueError("Declared fused gate/up expert geometry is invalid")
    if num_experts % degree:
        raise ValueError(
            f"EP degree {degree} does not divide num_experts={num_experts}"
        )
    if gate_up.shape[1] == 0 or gate_up.shape[2] == 0:
        raise ValueError("EP expert dimensions must be positive")
    return PackedExpertBinding(block, owner, gate_up, down, tuple(gate_up.shape))


def validate_moe_conversion(
    block: nn.Module,
    routing: MoERoutingBinding,
    *,
    tp_enabled: bool,
    quantile_balancing: bool,
    token_dispatcher: str,
    has_ep_group: bool,
) -> bool:
    """Check metadata-only conversion restrictions before any weight allocation.

    Return whether the existing torchtitan NPU expert implementation must be
    preserved. No process group queries, parameter mutation or RNG consumption.
    The caller supplies group ownership explicitly; size-one groups retain
    their distinction from local conversion without a group.
    """
    if routing.source is not block:
        raise ValueError("MoE routing binding belongs to another block")
    if getattr(routing.router, "bias", None) is not None:
        compatibility.router_bias(routing.router)
    if routing.shared is not None:
        if any(
            value.device != routing.router.weight.device
            for value in (
                *routing.shared.parameters(),
                *routing.shared.buffers(),
            )
        ):
            raise UnsupportedCombinationError(
                "MoE shared expert and router are on different devices"
            )
        if routing.shared_gate is not None:
            compatibility.shared_expert_gate(block)
        if tp_enabled and not (
            getattr(block, "_tp_moe_boundary", False) and not has_ep_group
        ):
            compatibility.shared_expert_tp_ep(block)
    if quantile_balancing:
        if routing.score_func != "sigmoid":
            compatibility.quantile_requires_sigmoid(routing.score_func, block)
        if routing.num_expert_groups is not None and routing.num_expert_groups > 1:
            compatibility.quantile_no_group_limit(block)
    original_forward = block.__dict__.get("_tp_moe_original_forward", block.forward)
    baseline_npu = (
        device_type == "npu"
        and getattr(original_forward, "__func__", None) is forward_packed_experts
    )
    if baseline_npu:
        if token_dispatcher != "alltoall":
            raise ValueError(
                "torchtitan NPU experts with EP require token_dispatcher='alltoall'"
            )
        if not has_ep_group:
            raise ValueError("torchtitan NPU expert conversion requires an EP group")
    return baseline_npu


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
    routing: MoERoutingBinding | None = None,
    packed_experts: PackedExpertBinding | None = None,
) -> MoE:
    """Build the llmtuner MoE for one HF block and move its weights over."""
    ep_size = 1 if ep_group is None else get_world_size(ep_group)
    ep_rank = 0 if ep_group is None else get_rank(ep_group)

    routing = routing if routing is not None else resolve_moe_routing(block)
    baseline_npu = validate_moe_conversion(
        block,
        routing,
        tp_enabled=tp_enabled,
        quantile_balancing=quantile_balancing,
        token_dispatcher=token_dispatcher,
        has_ep_group=ep_group is not None,
    )
    router_gate = routing.router
    shared = routing.shared
    feature_parallel = bool(getattr(block, "_tp_moe_boundary", False))
    if packed_experts is None:
        packed_experts = resolve_packed_experts(block, degree=ep_size)
    experts = packed_experts.projections(block)

    num_experts = experts.num_experts
    top_k = routing.top_k
    dim = router_gate.weight.shape[1]
    hidden = experts.gate_EFD.shape[1]
    if num_experts % ep_size != 0:
        raise ValueError(
            f"EP degree {ep_size} does not divide num_experts={num_experts}; "
            "each EP rank must hold the same number of experts."
        )
    num_local = num_experts // ep_size
    lo = ep_rank * num_local

    num_expert_groups = routing.num_expert_groups
    num_limited_groups = routing.num_limited_groups
    if baseline_npu:
        grouped = NpuBaselineGroupedExperts(dim, hidden, num_local)
    elif token_dispatcher == "mc2":
        grouped = MC2GroupedExperts(dim, hidden, num_local)
    else:
        grouped = GroupedExperts(dim, hidden, num_local)
    score_func = routing.score_func
    router_options = {
        "num_experts": num_experts,
        "dim": dim,
        "top_k": top_k,
        "route_norm": routing.route_norm,
        "route_scale": routing.route_scale,
        "aux_loss": (
            MicrobatchWiseLoadBalanceLoss(
                coeff=aux_loss_coeff,
                token_replication_size=(
                    block._tp_seq_group.size()
                    if getattr(block, "_tp_moe_boundary", False)
                    else 1
                ),
            )
            if aux_loss_coeff
            else None
        ),
    }
    if quantile_balancing:
        # The quantile scheme is defined over sigmoid scores (the histogram
        # range derives from their [0, 1] bound) and routes freely over all
        # experts, so a softmax family or a group-limited one cannot adopt it.
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
        if baseline_npu:
            dispatcher = NpuBaselineAllToAllTokenDispatcher(num_experts, top_k)
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
    if token_dispatcher == "mc2":
        routed_experts_cls = MC2RoutedExperts
    if baseline_npu and ep_group is not None:
        routed_experts_cls = NpuBaselineRoutedExperts
    moe_cls = FeatureParallelMoE if feature_parallel else MoE
    moe = moe_cls(
        num_experts=num_experts,
        routed_experts=routed_experts_cls(grouped, dispatcher),
        router=router,
        # Auxiliary-loss-free load balancing, mirroring the HF block's own
        # e_score_correction_bias. Off for models that carry no such buffer --
        # a frozen zero bias is dead state in the checkpoint.
        load_balance_coeff=load_balance_coeff,
    )

    # Allocate the large expert subtree in its source dtype directly, avoiding
    # temporary FP32 expert storage when only the router needs FP32.
    moe.to(dtype=experts.gate_EFD.dtype, device=router_gate.weight.device)
    router.to(dtype=router_gate.weight.dtype, device=router_gate.weight.device)
    # Routers may intentionally stay FP32 while expert GEMMs use BF16.
    # Allocate each projection in its source dtype before copying: casting
    # through the router dtype can otherwise change precision and memory use.
    expert_projections = (
        (grouped.w1_EFD, experts.gate_EFD),
        (grouped.w3_EFD, experts.up_EFD),
        (grouped.w2_EDF, experts.down_EDF),
    )
    for target, source in expert_projections:
        target.data = target.data.to(device=source.device, dtype=source.dtype)
    restore_fp32_state_buffers(moe)
    # Preserve eval mode too: new modules otherwise default to training.
    moe.train(block.training)
    router.gate.weight.requires_grad_(router_gate.weight.requires_grad)
    for target, source in expert_projections:
        target.requires_grad_(source.requires_grad)
    with torch.no_grad():
        router.gate.weight.copy_(router_gate.weight)
        # HF keeps all E experts in two stacked parameters; each rank keeps its
        # own slice of them. ``gate_EFD``/``up_EFD`` are the two halves of
        # ``gate_up_proj`` split along its declared output-feature dimension.
        for target, source in expert_projections:
            target.copy_(source[lo : lo + num_local])
        # The load-balancing bias is optimization state, not a learned weight:
        # it is copied so a resumed or swapped run keeps the balance the HF
        # model had reached.
        bias = getattr(router_gate, "e_score_correction_bias", None)
        if bias is not None and moe.expert_bias_E is not None:
            moe.expert_bias_E.copy_(bias.float())
    # Reuse the HF shared expert after initializing the new subtree. Casting
    # or setting the mode recursively must not mutate this original module.
    moe.shared_experts = shared
    if feature_parallel:
        moe._tp_seq_group = block._tp_seq_group
        moe._tp_moe_boundary = True
        moe.tp_sharded_param_names = {
            f"routed_experts.inner_experts.{name}"
            for name in ("w1_EFD", "w3_EFD", "w2_EDF")
        }
        if shared is not None:
            moe.tp_sharded_param_names.update(
                f"shared_experts.{name}.weight"
                for name in ("gate_proj", "up_proj", "down_proj")
            )
    return moe
