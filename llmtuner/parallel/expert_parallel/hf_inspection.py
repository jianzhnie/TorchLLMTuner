"""HF MoE block layout probes for the EP swap.

The entry point is ``model_replacement.py`` (``swap_hf_moe_blocks``); the conversion is
``block_conversion.py``. This module is the duck-typed detector every HF family is
identified through, shared by the swap and by ``tensor_parallel.py``
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

from ...models.common.moe.layer import MOE_LAYER_ATTRS
from ...models.kimi.moe import KimiSparseMoE
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
    """The block's router module, whatever the family calls it.

    ``gate`` is the common spelling and what every supported family but GPT-OSS
    uses. DeepSeek-V2's is a plain ``nn.Linear``; Qwen3Moe's, Mixtral's,
    DeepSeek-V3's and GLM4's are bespoke classes that differ only in the buffers
    they carry.
    """
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


def _fused_experts_of_per_expert_list(experts: nn.Module) -> FusedExperts | None:
    """Read a pre-5.x per-expert module list as the fused 5.x parameters.

    transformers 5.x collapsed "one module per expert" into two stacked
    parameters; the 4.x line still holds one
    ``Qwen3MoeMLP(gate_proj, up_proj, down_proj)`` -- or
    ``MixtralBlockSparseTop2MLP(w1, w3, w2)`` -- per expert. Stacking those three
    projections into ``(E, 2F, D)`` / ``(E, D, F)`` here keeps the swap, the
    weight move and the tests on the single layout they already share, instead
    of teaching each of them a second one.

    Returns ``None`` for a list that is not one of the recognized spellings, so
    the caller keeps its "no MoE block here" behaviour.
    """
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
        act_fn = getattr(first, "act_fn", None)
        if act_fn is not None and not isinstance(act_fn, nn.SiLU | SiLUActivation):
            raise NotImplementedError(
                f"Expert activation {type(act_fn).__name__} is not SiLU; the "
                "swap would silently change the experts' nonlinearity. Wire the "
                "activation through GroupedExperts first."
            )
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
        # First half gate, second half up: the split ``chunk(2, dim=-1)`` in the
        # 5.x forward, and the same convention ``GroupedExperts`` reads.
        gate_up = torch.cat([torch.stack(gates), torch.stack(ups)], dim=1)
        return FusedExperts(
            gate_EFD=gate_up[:, :hidden],
            up_EFD=gate_up[:, hidden:],
            down_EDF=torch.stack(downs),
            num_experts=len(members),
        )
    return None


def fused_experts_of(block: nn.Module) -> FusedExperts | None:
    """Probe a MoE block for the fused expert tensors, or ``None``.

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
    experts = getattr(block, "experts", None)
    if experts is None:
        return None
    gate_up = getattr(experts, "gate_up_proj", None)
    down = getattr(experts, "down_proj", None)
    if not isinstance(gate_up, torch.Tensor) or not isinstance(down, torch.Tensor):
        # Pre-5.x layout: one module per expert instead of two stacked tensors.
        return _fused_experts_of_per_expert_list(experts)
    if gate_up.dim() != 3 or down.dim() != 3:
        return None
    if gate_up.shape[0] != down.shape[0]:
        return None
    # GroupedExperts hard-codes SwiGLU; every supported family activates with
    # silu today, but the probe must say so rather than let a future HF family
    # with a different gated activation be swapped in silently.
    act_fn = getattr(experts, "act_fn", None)
    if act_fn is not None and not isinstance(act_fn, nn.SiLU | SiLUActivation):
        raise NotImplementedError(
            f"Expert activation {type(act_fn).__name__} is not SiLU; the swap "
            "would silently change the experts' nonlinearity. Wire the "
            "activation through GroupedExperts first."
        )
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
    experts = fused_experts_of(module)
    if experts is None:
        return False
    return has_router_weight(router_of(module)) and resolve_top_k(module) is not None


def resolve_shared_expert(block: nn.Module) -> tuple[str, nn.Module | None]:
    """Select a populated shared-expert attribute and preserve its HF path."""
    for name in ("shared_expert", "shared_experts"):
        shared = getattr(block, name, None)
        if shared is not None:
            return name, shared
    return "shared_expert", None


def resolve_top_k(block: nn.Module) -> int | None:
    """Top-K per token, from the block or its router."""
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
    """Whether the selected K scores are renormalized to sum to 1.

    ``norm_topk_prob`` may live on either the block or the router. When neither
    declares it, sigmoid routing scores are used as-is and anything else is taken
    to be normalized -- matching how torchtitan's probe resolves it.
    """
    if ignores_norm_topk_prob(block, router):
        return False
    for owner in (block, router):
        value = getattr(owner, "norm_topk_prob", None)
        if value is not None:
            return bool(value)
    return resolve_score_func(block, router) != "sigmoid"


def resolve_score_func(block: nn.Module, router: nn.Module) -> str:
    """The router's scoring function.

    ``e_score_correction_bias`` is the reliable DeepSeek-V3/GLM4 marker: a router
    that carries that buffer scores with sigmoid. Otherwise the block's declared
    ``scoring_func`` wins, defaulting to softmax as HF does.
    """
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
    """The multiplier applied to the selected K scores after normalization.

    HF puts ``routed_scaling_factor`` on whichever object owns the rest of the
    routing: the MoE block for Qwen3Moe, the router for DeepSeek-V3/GLM4. Both
    are checked so the same probe serves either layout.
    """
    for owner in (block, router):
        value = getattr(owner, "routed_scaling_factor", None)
        if value is not None:
            return float(value)
    return 1.0


def returns_router_logits(block: nn.Module) -> bool:
    """Whether ``block``'s forward returns ``(hidden_states, router_logits)``.

    transformers has no single convention here, and it varies by *family*, not
    by spelling: 4.x Mixtral's host unpacks a tuple from ``block_sparse_moe``,
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
    fallback: tuple[str, nn.Module | None] = (MOE_LAYER_ATTRS[0], None)
    for name in MOE_LAYER_ATTRS:
        block = getattr(layer, name, None)
        if block is None:
            continue
        # KimiSparseMoE is llmtuner's own block spelling, not the generic HF
        # shape: the fused-experts probe refuses its SituGLU activation, so it
        # must be recognized directly, the same way swap_hf_moe_blocks does.
        if isinstance(block, KimiSparseMoE) or is_hf_moe_block(block):
            return name, block
        if fallback[1] is None:
            fallback = (name, block)
    return fallback


def read_expert_groups(
    block: nn.Module, router: nn.Module
) -> tuple[int | None, int | None]:
    """The node-limited-routing config, as ``(num_expert_groups, num_limited_groups)``.

    Returns ``(None, None)`` when the block does no group-limited routing, which
    is the case for Qwen3Moe, OLMoE and Mixtral.

    The grouping attributes are read only when the model is actually routing by
    group. DeepSeek-V2 carries ``num_group``/``topk_group`` but consults them
    only under ``topk_method="group_limited_greedy"``; its default ``"greedy"``
    routes freely over all experts. Reading the attributes without checking the
    method feeds those tokens into a group restriction HF never applies.

    Raises:
        NotImplementedError: for DeepSeek-V2's ``group_limited_greedy``, whose
            rule differs from the one implemented here: it scores a group by its
            single best expert (``max``) where DeepSeek-V3/GLM4 sum the group's
            top-2. Both are "group-limited", so the distinction is invisible in
            the config and would only show up as different experts being chosen.
    """
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
