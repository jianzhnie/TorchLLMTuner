"""HF MoE block layout probes for the EP swap.

The entry point is ``model_replacement.py`` (``swap_moe_blocks``); the conversion is
``block_conversion.py``. This module reads explicit decoder/output declarations
where available and provides legacy layout detection, shared by the swap and
by ``tensor_parallel.py``
(``is_hf_moe_block``).

Every sparse decoder layer of a transformers 5.x MoE model holds the same block
under ``layer.mlp``: a router, plus two stacked expert parameters. This module
only *detects* that layout and reads its dimensions -- the weight move into the
grouped-GEMM stack lives in ``block_conversion.py``; the orchestration lives
in ``model_replacement.py``.

Both halves of the block vary by family, and the probe is duck-typed against
structures rather than class names, which are not stable across transformers
releases:

* **The router.** ``block.gate`` or ``block.router``, either a plain
  ``nn.Linear`` (DeepSeek-V2) or a bespoke module holding the same ``(E, D)``
  weight plus, for DeepSeek-V3/GLM4, an ``e_score_correction_bias`` buffer.
  Identified by its *weight tensor*, not its type (``router_of``).
* **The expert weights.** Fused layouts use ``gate_up_proj (E, 2F, D)`` and
  ``down_proj (E, D, F)``. ``fused_experts_of`` splits the gate/up weight on
  its output-feature axis (dim 1), yielding two ``(E, F, D)`` views. The HF
  forward instead splits the gate/up GEMM output on its last axis.

Where a family keeps its routing attributes also moved in 5.x: ``top_k``,
``n_group``, ``topk_group``, ``norm_topk_prob``, ``scoring_func`` and
``routed_scaling_factor`` now sit on the *block* for every family, having been
split between block and router in 4.x. Every ``_read_*`` helper therefore takes
both owners and prefers whichever declares the field.

Weight layout: HF stores each projection as ``(out, in)`` and applies it as
``x @ W.T``; ``GroupedExperts`` stores the same orientation per expert
(``w1_EFD``/``w3_EFD`` are ``(F, D)``, ``w2_EDF`` is ``(D, F)``) and applies it
with the same ``F.linear``. Moving a weight is therefore elementwise -- no
transpose, no regrouping -- except for the one split of ``gate_up_proj`` into
its gate and up halves. ``tests/unit_tests/cpu/parallel/test_ep_model_replacement.py``
pins both halves of that claim against the HF block being replaced.

Routing parity with the HF block: Qwen3Moe and Mixtral score with a softmax over
fp32 logits; DeepSeek-V3/GLM4 score with a sigmoid and apply
``routed_scaling_factor``; DeepSeek-V2 scores with a softmax but never
renormalizes, whatever its config declares (see ``ignores_norm_topk_prob``).
``RouterGateLinear`` computes in fp32, so the same score function reproduces it,
and ``TokenChoiceTopKRouter`` takes node-limited routing as
``num_expert_groups``/``num_limited_groups``. The one deliberate addition is the
load-balance machinery HF never gets to run through this wrapper (the wrapper
never asks for router logits): the router carries a
``MicrobatchWiseLoadBalanceLoss`` with the config's ``router_aux_loss_coef``,
whose gradient is injected on backward (see ``models/common/aux_loss.py``).

Two families are deliberately *not* swapped, each because the HF block asks for
something the llmtuner stack does not express -- refusing beats approximating,
since the difference shows up only as different experts being chosen, which no
loss curve reveals:

* ``GPT-OSS``: per-expert bias vectors, a transposed ``(E, D, 2F)`` layout, and
  a hardcoded clamped sigmoid-GLU activation rather than a module
  (see ``fused_experts_of``).
* ``DeepSeek-V2`` with ``topk_method="group_limited_greedy"``: scores a group by
  its single best expert where V3/GLM4 sum the top-2 (see
  ``read_expert_groups``). Its default ``"greedy"`` is supported and exact.

``Qwen2Moe``'s ``shared_expert_gate`` multiplies where ``MoE.shared_experts``
only adds, and is refused at the point the shared expert is found.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import NamedTuple

import torch
import torch.nn as nn

from ...errors import UnsupportedCombinationError
from ...models.common.moe.layer import MOE_LAYER_ATTRS, declared_moe_block
from ...models.kimi.kimi_k3.moe import KimiSparseMoE
from ...utils.logging import get_logger
from .. import compatibility

logger = get_logger(__name__)

try:
    # transformers 5.x spells the gated activation as its own module; it computes
    # the same ``torch.nn.functional.silu``.
    from transformers.activations import SiLUActivation
except ImportError:  # transformers < 5 uses torch.nn.SiLU directly
    SiLUActivation = nn.SiLU

__all__ = [
    "FusedExperts",
    "fused_experts_of",
    "ignores_norm_topk_prob",
    "is_hf_moe_block",
    "moe_block_of",
    "per_expert_projection_names",
    "per_expert_weights_of",
    "read_expert_groups",
    "read_route_norm",
    "read_route_scale",
    "resolve_score_func",
    "resolve_shared_expert",
    "resolve_top_k",
    "router_of",
    "returns_router_logits",
]


def router_of(block: nn.Module) -> nn.Module | None:
    """Resolve a declared router, or inspect a legacy HF block."""
    contract = getattr(block, "moe_contract", None)
    if contract is not None:
        router, _, _ = contract.bind(block)
        return router
    return getattr(block, "gate", None) or getattr(block, "router", None)


class FusedExperts(NamedTuple):
    """One family's expert weights, as the three separate tensors llmtuner wants.

    A transformers 5.x expert keeps all E experts in two parameters rather than
    a list of per-expert MLPs: ``gate_up_proj (E, 2F, D)`` -- gate and up
    interleaved along the output dim, split by ``chunk(2, dim=-1)`` -- and
    ``down_proj (E, D, F)``. ``gate_EFD``/``up_EFD`` are views into the first,
    not copies.
    """

    gate_EFD: torch.Tensor
    up_EFD: torch.Tensor
    down_EDF: torch.Tensor
    num_experts: int


# The three separate projections a pre-5.x expert module carries, per family
# spelling: (gate, up, down). Mixtral/GraniteMoe name them ``w1``/``w3``/``w2``
# -- ``act_fn(w1(x)) * w3(x)`` then ``w2`` -- the rest use the descriptive names.
PER_EXPERT_PROJECTIONS = (
    ("gate_proj", "up_proj", "down_proj"),
    ("w1", "w3", "w2"),
)


def _has_projection_weight(module: object) -> bool:
    """Whether ``module`` is an expert projection: a 2-D ``weight``, by shape.

    Not ``isinstance(nn.Linear)``: by the time this probe runs under TP the
    projections may already be llmtuner's parallel linears, which carry the
    same ``(out, in)`` weight but are not HF's class.
    """
    weight = getattr(module, "weight", None)
    return isinstance(weight, torch.Tensor) and weight.dim() == 2


def _validate_silu_activation(experts: nn.Module) -> None:
    act_fn = getattr(experts, "act_fn", None)
    if act_fn is not None and not isinstance(act_fn, nn.SiLU | SiLUActivation):
        raise NotImplementedError(
            f"Expert activation {type(act_fn).__name__} is not SiLU; the swap "
            "would silently change the experts' nonlinearity. Wire the "
            "activation through GroupedExperts first."
        )


def per_expert_projection_names(experts: nn.Module) -> tuple[str, str, str] | None:
    """The (gate, up, down) attribute spelling of a per-expert module list.

    Returns ``None`` for the fused 5.x layout (two stacked tensors) and for
    anything that is not a recognized per-expert list; callers that find a
    spelling here can rely on ``fused_experts_of`` having accepted the block.
    """
    if not isinstance(experts, nn.ModuleList | nn.Sequential | list | tuple):
        return None
    members = list(experts)
    if not members:
        return None
    first = members[0]
    for names in PER_EXPERT_PROJECTIONS:
        if all(_has_projection_weight(getattr(first, name, None)) for name in names):
            return names
    return None


class PerExpertWeights(NamedTuple):
    """Validated references to legacy expert weights, without stacked storage."""

    gates: tuple[torch.Tensor, ...]
    ups: tuple[torch.Tensor, ...]
    downs: tuple[torch.Tensor, ...]


def per_expert_weights_of(experts: nn.Module) -> PerExpertWeights | None:
    """Validate individual projections without allocating replacement tensors."""
    if not isinstance(experts, nn.ModuleList | nn.Sequential | list | tuple):
        return None
    members = list(experts)
    if not members:
        return None
    first = members[0]
    for gate_name, up_name, down_name in PER_EXPERT_PROJECTIONS:
        # Identified by their ``(out, in)`` weight, not by class: under TP these
        # projections are llmtuner's parallel linears, under EP-swap ordering
        # they are still HF's ``nn.Linear``, and the shapes are what the reader
        # actually needs.
        if not all(
            _has_projection_weight(getattr(first, name, None))
            for name in (gate_name, up_name, down_name)
        ):
            continue
        hidden, dim = getattr(first, gate_name).weight.shape
        down_dim, down_hidden = getattr(first, down_name).weight.shape
        if down_dim != dim or down_hidden != hidden:
            raise ValueError(
                f"unrecognized expert weight shapes: {gate_name} "
                f"{tuple(getattr(first, gate_name).weight.shape)}, {down_name} "
                f"{tuple(getattr(first, down_name).weight.shape)}. Expected "
                f"(F, D) and (D, F)."
            )
        gates: list[torch.Tensor] = []
        ups: list[torch.Tensor] = []
        downs: list[torch.Tensor] = []
        for expert in members:
            _validate_silu_activation(expert)
            for name in (gate_name, up_name, down_name):
                if getattr(getattr(expert, name), "bias", None) is not None:
                    raise NotImplementedError(
                        f"Expert projection {name} has bias; GroupedExperts "
                        "cannot preserve projection bias during conversion."
                    )
            gate = getattr(expert, gate_name).weight
            up = getattr(expert, up_name).weight
            down = getattr(expert, down_name).weight
            if gate.shape != (hidden, dim) or up.shape != (hidden, dim):
                raise ValueError(
                    f"expert {gate_name}/{up_name} shapes differ across the "
                    f"list: got {tuple(gate.shape)} and {tuple(up.shape)}, "
                    f"expected {(hidden, dim)} for all {len(members)} experts."
                )
            if down.shape != (down_dim, down_hidden):
                raise ValueError(
                    f"expert {down_name} shape differs across the list: got "
                    f"{tuple(down.shape)}, expected {(down_dim, down_hidden)}."
                )
            gates.append(gate)
            ups.append(up)
            downs.append(down)
        for name, weights in zip(
            (gate_name, up_name, down_name), (gates, ups, downs), strict=True
        ):
            if len({(weight.dtype, weight.device) for weight in weights}) != 1:
                raise UnsupportedCombinationError(
                    "EP cannot stack experts with different dtypes or devices "
                    f"for {name}; one grouped parameter requires uniform storage."
                )
            if len({id(weight) for weight in weights}) != len(weights):
                raise UnsupportedCombinationError(
                    "EP cannot stack a shared individual expert parameter "
                    f"for {name}; stacked rows cannot preserve expert weight tying."
                )
            if len({weight.requires_grad for weight in weights}) != 1:
                raise UnsupportedCombinationError(
                    "EP cannot stack experts with different requires_grad flags "
                    f"for {name}; grouped parameters require uniform freezing."
                )
        return PerExpertWeights(tuple(gates), tuple(ups), tuple(downs))
    return None


def _fused_experts_of_per_expert_list(experts: nn.Module) -> FusedExperts | None:
    """Stack validated legacy weights only when conversion needs tensor storage."""
    weights = per_expert_weights_of(experts)
    if weights is None:
        return None
    # Preserve each projection's freezing even under a no_grad caller.
    return FusedExperts(
        gate_EFD=torch.stack(weights.gates).requires_grad_(
            weights.gates[0].requires_grad
        ),
        up_EFD=torch.stack(weights.ups).requires_grad_(weights.ups[0].requires_grad),
        down_EDF=torch.stack(weights.downs).requires_grad_(
            weights.downs[0].requires_grad
        ),
        num_experts=len(weights.gates),
    )


def fused_experts_of(block: nn.Module) -> FusedExperts | None:
    """Probe a MoE block for the fused expert tensors, or ``None``.

    An explicit ``expert_layout="fused_gate_up"`` declaration selects this
    layout directly. Its storage is still validated; missing or invalid
    tensors must fail rather than trigger legacy per-expert discovery.

    One probe covers every supported family because transformers 5.x moved all
    of Qwen3Moe, OLMoE, Mixtral, DeepSeek-V2/V3 and GLM4 onto the same two
    parameters under the same forward shape::

        gate, up = linear(x, gate_up_proj[e]).chunk(2, dim=-1)
        out = linear(act(gate) * up, down_proj[e])

    so the split is by role -- the first half is the gate, the second the up
    projection -- for every one of them. That is asserted family by family
    against HF's own output in
    ``tests/unit_tests/cpu/parallel/test_ep_model_replacement.py``.

    The probe reads *shapes*, not class names: the experts' class name is not
    stable across transformers versions (``Qwen3MoeExperts``,
    ``DeepseekV3NaiveMoe``, ``MixtralExperts``), but ``(E, 2F, D)`` is.

    Raises:
        NotImplementedError: for GPT-OSS, whose transposed layout
            (``gate_up_proj`` is ``(E, D, 2F)``) and per-expert bias vectors
            have no counterpart in llmtuner's ``GroupedExperts``. Copying it under
            the shared convention would put the wrong weight in the wrong slot.
    """
    contract = getattr(block, "moe_contract", None)
    layout = getattr(
        block,
        "expert_layout",
        contract.expert_layout if contract is not None else None,
    )
    if layout is not None and layout != "fused_gate_up":
        raise ValueError(f"Unsupported declared expert layout {layout!r}")
    experts = getattr(block, "experts", None)
    if experts is None:
        if layout is not None:
            raise ValueError("fused_gate_up requires an experts child module")
        return None
    gate_up = getattr(experts, "gate_up_proj", None)
    down = getattr(experts, "down_proj", None)
    if not isinstance(gate_up, torch.Tensor) or not isinstance(down, torch.Tensor):
        if layout is not None:
            raise ValueError(
                "fused_gate_up requires gate_up_proj and down_proj tensors"
            )
        # Pre-5.x layout: one module per expert instead of two stacked tensors.
        return _fused_experts_of_per_expert_list(experts)
    if gate_up.dim() != 3 or down.dim() != 3:
        if layout is not None:
            raise ValueError("fused_gate_up requires rank-3 expert tensors")
        return None
    if gate_up.shape[0] != down.shape[0]:
        if layout is not None:
            raise ValueError("fused_gate_up expert counts must match")
        return None
    # GroupedExperts hard-codes SwiGLU; every supported family activates with
    # silu today, but the probe must say so rather than let a future HF family
    # with a different gated activation be swapped in silently.
    _validate_silu_activation(experts)
    if hasattr(experts, "gate_up_proj_bias") or hasattr(experts, "down_proj_bias"):
        compatibility.gpt_oss_layout(experts)
    num_experts, double_hidden, dim = gate_up.shape
    # Both projections must agree on D and on the expert width F. F may differ
    # from the dense MLP width; only the actual expert tensor shapes matter.
    if double_hidden % 2 != 0 or down.shape[1:] != (dim, double_hidden // 2):
        raise ValueError(
            f"unrecognized expert weight shapes: gate_up_proj "
            f"{tuple(gate_up.shape)}, down_proj {tuple(down.shape)}. Expected "
            "(E, 2F, D) and (E, D, F), the layout shared by every supported "
            "family in transformers 5.x."
        )
    hidden = double_hidden // 2
    return FusedExperts(
        gate_EFD=gate_up[:, :hidden],
        up_EFD=gate_up[:, hidden:],
        down_EDF=down,
        num_experts=num_experts,
    )


def has_router_weight(router: nn.Module | None) -> bool:
    """Whether a router carries its own ``(E, D)`` gate weight.

    DeepSeek-V3 and GLM4 routers are bespoke classes rather than ``nn.Linear``,
    so the shape is what identifies them -- the class name is not stable across
    transformers versions.
    """
    weight = getattr(router, "weight", None)
    return isinstance(weight, nn.Parameter) and weight.dim() == 2


def is_hf_moe_block(module: nn.Module) -> bool:
    """Structural probe for the HF sparse-MoE block shape.

    Deliberately duck-typed rather than an ``isinstance`` against one family's
    class: Qwen3Moe, OLMoE, Mixtral, DeepSeek-V2/V3 and GLM4 share this shape
    (a router with an ``(E, D)`` weight, plus fused expert tensors) under
    different class names and different router spellings.

    The router is identified by its *weight tensor*, not by being ``nn.Linear``:
    DeepSeek-V3's and GLM4's routers are plain ``nn.Module``s holding the same
    ``(E, D)`` parameter, while DeepSeek-V2's really is an ``nn.Linear``.

    ``top_k`` is read from the block or the router because the families disagree
    on where it lives -- DeepSeek and Mixtral declare it on the block, Qwen3Moe
    and OLMoE on the router.
    """
    # Unsupported layouts must propagate their error rather than be skipped
    # as ordinary dense blocks.
    contract = getattr(module, "moe_contract", None)
    layout = getattr(
        module,
        "expert_layout",
        contract.expert_layout if contract is not None else None,
    )
    owner = getattr(module, "experts", None)
    if layout is None and isinstance(
        owner, nn.ModuleList | nn.Sequential | list | tuple
    ):
        weights = per_expert_weights_of(owner)
        if weights is None:
            return False
        num_experts = len(weights.gates)
        projection_devices = {weight.device for group in weights for weight in group}
    else:
        experts = fused_experts_of(module)
        if experts is None:
            return False
        num_experts = experts.num_experts
        projection_devices = {
            experts.gate_EFD.device,
            experts.up_EFD.device,
            experts.down_EDF.device,
        }
    if contract is not None and num_experts != contract.num_experts:
        raise ValueError(
            f"Declared {contract.num_experts} experts, found {num_experts}"
        )
    router = router_of(module)
    if not has_router_weight(router) or resolve_top_k(module) is None:
        return False
    if projection_devices != {router.weight.device}:
        raise UnsupportedCombinationError(
            "EP cannot convert a MoE whose router and routed expert "
            "projections are on different devices; move the complete MoE "
            "block to one device before conversion."
        )
    _, shared = resolve_shared_expert(module)
    if shared is not None and any(
        parameter.device != router.weight.device for parameter in shared.parameters()
    ):
        raise UnsupportedCombinationError(
            "EP cannot convert a MoE whose shared expert parameters are on a "
            "different device from its router; move the complete MoE block "
            "to one device before conversion."
        )
    return True


def resolve_shared_expert(block: nn.Module) -> tuple[str, nn.Module | None]:
    """Resolve a declared shared expert, or inspect a legacy HF block."""
    contract = getattr(block, "moe_contract", None)
    if contract is not None:
        _, name, shared = contract.bind(block)
        return name, shared
    for name in ("shared_expert", "shared_experts"):
        shared = getattr(block, name, None)
        if shared is not None:
            return name, shared
    return "shared_expert", None


def resolve_top_k(block: nn.Module) -> int | None:
    """Top-K per token, from the model contract or a legacy HF block."""
    contract = getattr(block, "moe_contract", None)
    if contract is not None:
        return contract.top_k
    for owner in (block, router_of(block)):
        if owner is None:
            continue
        for attr in ("top_k", "num_experts_per_tok"):
            value = getattr(owner, attr, None)
            if isinstance(value, int):
                return value
    return None


def ignores_norm_topk_prob(block: nn.Module, router: nn.Module) -> bool:
    """Whether the block declares ``norm_topk_prob`` without ever applying it.

    Only DeepSeek-V2 does. Its routing ignores the field entirely::

        router_logits = router_logits.softmax(dim=-1, dtype=torch.float32)
        ...
        topk_weight = topk_weight * self.routed_scaling_factor

    that is, the selected scores are scaled but never divided by their own sum.
    A model with ``norm_topk_prob=True`` in its config therefore still behaves
    as ``route_norm=False``.

    The tell is ``topk_method``, not the class name (which transformers renames
    between versions) and not ``norm_topk_prob`` itself -- transformers 5.x
    *removed* that field from DeepSeek-V2 while keeping the behaviour, so a
    guard that read it would find nothing, fall through to the ``softmax ->
    renormalize`` default, and silently route differently from HF.
    """
    if getattr(block, "topk_method", None) is not None:
        return True
    return getattr(router, "topk_method", None) is not None


def read_route_norm(block: nn.Module, router: nn.Module) -> bool:
    """Read declared route normalization or inspect a legacy HF block."""
    contract = getattr(block, "moe_contract", None)
    if contract is not None:
        return contract.route_norm
    if ignores_norm_topk_prob(block, router):
        return False
    for owner in (block, router):
        value = getattr(owner, "norm_topk_prob", None)
        if value is not None:
            return bool(value)
    return resolve_score_func(block, router) != "sigmoid"


def resolve_score_func(block: nn.Module, router: nn.Module) -> str:
    """Read the model-owned score function or inspect a legacy HF router."""
    contract = getattr(block, "moe_contract", None)
    if contract is not None:
        return contract.score_func
    buffers = getattr(router, "_buffers", {})
    if "e_score_correction_bias" in buffers:
        return "sigmoid"
    declared = getattr(block, "scoring_func", None) or getattr(
        router, "scoring_func", None
    )
    if declared:
        return str(declared).lower()
    return "softmax"


def read_int_attr(block: nn.Module, router: nn.Module, name: str) -> int | None:
    """Read an optional integer attribute off the block or the router."""
    for owner in (block, router):
        value = getattr(owner, name, None)
        if value is not None:
            return int(value)
    return None


def read_route_scale(block: nn.Module, router: nn.Module) -> float:
    """Read declared route scaling or inspect a legacy HF block."""
    contract = getattr(block, "moe_contract", None)
    if contract is not None:
        return contract.route_scale
    for owner in (block, router):
        value = getattr(owner, "routed_scaling_factor", None)
        if value is not None:
            return float(value)
    return 1.0


def returns_router_logits(block: nn.Module) -> bool:
    """Whether ``block``'s forward returns ``(hidden_states, router_logits)``.

    Explicit models declare ``router_logits_in_output``. Legacy HF blocks fall
    back to source inspection because transformers has no single convention:
    4.x Mixtral's host unpacks a tuple from ``block_sparse_moe``,
    OLMoE's unpacks one from ``mlp``, while Qwen3Moe's tolerates either and
    DeepSeek's takes a bare tensor from ``moe``. Neither the attribute name nor
    the class name separates them, so the answer is read off the block itself --
    from the AST of its ``forward``, not by running it.

    Preserving the block's own arity is correct by construction: HF's host layer
    is written against HF's own block, so reproducing the return shape keeps the
    host's contract intact whatever the transformers version does.

    Reading the source rather than calling the block matters: a runtime probe
    would have to pick a dtype/device for a throwaway input, and models that
    deliberately mix dtypes (tests do) would make it fail inside an unrelated
    ``Linear``.

    Falls back to ``False`` (bare tensor, what most families and every 5.x block
    return) when the source cannot be read, e.g. an installed-and-stripped
    build; that case is logged rather than assumed silently.
    """
    contract = getattr(block, "moe_contract", None)
    if contract is not None:
        return contract.router_logits_in_output
    declared = getattr(block, "router_logits_in_output", None)
    if declared is not None:
        if not isinstance(declared, bool):
            raise TypeError("router_logits_in_output must be a bool when declared")
        return declared
    try:
        source = textwrap.dedent(inspect.getsource(type(block).forward))
    except (OSError, TypeError):
        logger.warning(
            "Cannot read %s.forward to learn whether it returns router logits; "
            "assuming a bare tensor, which is what the 5.x families do.",
            type(block).__name__,
        )
        return False
    for node in ast.walk(ast.parse(source)):
        # ``return a, b`` and ``return (a, b)`` are both ast.Tuple.
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
            return True
    return False


def moe_block_of(layer: nn.Module) -> tuple[str, nn.Module | None]:
    """The layer's MoE block and the attribute it is held under.

    Explicit decoders declare ``moe_block_name`` (None for dense layers).
    Only legacy decoders use the candidate search below.

    The attribute is returned alongside the block because the swap has to
    *replace* the attribute, so the replacement must land in the same slot.
    Which attribute that is varies: 5.x keeps the block on ``mlp`` for every
    family, while 4.x uses ``block_sparse_moe`` (Mixtral, GraniteMoe) or ``moe``
    (DeepSeek, GLM4). DeepSeek layers additionally carry a *dense* ``mlp``, so
    the probe prefers whichever attribute actually holds a MoE and only falls
    back to the first one present.

    ``MOE_LAYER_ATTRS`` is shared with the expert-bias hook in
    ``models/common/moe/layer.py`` -- that hook rediscovers these blocks after the
    swap, so the two must agree on where a MoE can live.
    """
    declared = declared_moe_block(layer)
    if declared is not None:
        return declared

    fallback: tuple[str, nn.Module | None] = (MOE_LAYER_ATTRS[0], None)
    for name in MOE_LAYER_ATTRS:
        block = getattr(layer, name, None)
        if block is None:
            continue
        # KimiSparseMoE is llmtuner's own block spelling, not the generic HF
        # shape: the fused-experts probe refuses its SituGLU activation, so it
        # must be recognized directly, the same way swap_moe_blocks does.
        if isinstance(block, KimiSparseMoE) or is_hf_moe_block(block):
            return name, block
        if fallback[1] is None:
            fallback = (name, block)
    return fallback


def read_expert_groups(
    block: nn.Module, router: nn.Module
) -> tuple[int | None, int | None]:
    """Read declared group-limited routing or inspect a legacy HF block."""
    contract = getattr(block, "moe_contract", None)
    if contract is not None:
        return contract.num_expert_groups, contract.num_limited_groups
    # ``n_group`` is the DeepSeek-V3/GLM4 spelling, ``num_group`` DeepSeek-V2's.
    # Both are probed: reading only one silently disables grouping for the
    # family that spells it the other way, which is a routing change HF does not
    # make and no error surfaces.
    num_groups = read_int_attr(block, router, "n_group")
    if num_groups is None:
        num_groups = read_int_attr(block, router, "num_group")
    if num_groups is None:
        return None, None
    if num_groups == 0:
        raise ValueError(
            f"{type(block).__name__} declares n_group=0, which cannot "
            "partition the experts; check the model config."
        )

    # ``topk_method`` sits on the block in transformers 5.x (it was on
    # DeepSeek-V2's router in 4.x), so both are read.
    topk_method = getattr(block, "topk_method", None) or getattr(
        router, "topk_method", None
    )
    if topk_method == "greedy":
        return None, None
    if ignores_norm_topk_prob(block, router):
        compatibility.group_limited_greedy()
    return num_groups, read_int_attr(block, router, "topk_group")
