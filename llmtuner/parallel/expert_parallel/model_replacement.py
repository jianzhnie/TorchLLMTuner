"""Swap a HF MoE block for llmtuner's EP-capable MoE, weights included.

Kimi's ``block_sparse_moe`` and individual experts use ``kimi_conversion.py``;
the fused-layout details below apply to the transformers model families.

Every sparse decoder layer of a transformers 5.x MoE model holds the same block
under ``layer.mlp``: a router, plus two stacked expert parameters. Expert
parallelism needs those weights in grouped-GEMM form with a token dispatcher in
front -- the stack in ``models/common/`` (``GroupedExperts`` /
``TokenChoiceTopKRouter`` / ``MoE``). This module is the bridge: it detects the
HF block, moves the weights into the llmtuner layout, and replaces ``layer.mlp``.
It also flags the layer for the FSDP MoE branch (``layer.moe_enabled = True``
plus a non-registered ``layer.moe`` alias, as upstream's swap does), without
which the expert weights would be sharded as dense parameters over the dense
DP mesh. The swap is in place and dtype-preserving, so it must run before FSDP
wraps the model and after the weights are loaded.

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

from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn as nn

from llmtuner.config.parallel_config import (
    ExpertParallelConfig,
    validate_router_aux_loss_coefficient,
)
from llmtuner.errors import UnsupportedCombinationError

from ...models.common.aux_loss import AuxLoss
from ...models.common.moe.layer import MoE
from ...models.kimi.moe import KimiSparseMoE
from ...utils.logging import get_logger
from .block_conversion import convert_block
from .hf_inspection import is_hf_moe_block, moe_block_of, router_of
from .kimi_conversion import convert_kimi_block

logger = get_logger(__name__)

__all__ = ["swap_hf_moe_blocks"]

MoEReplacementObserver = Callable[
    [dict[nn.Module, nn.Module], dict[nn.Parameter, tuple[nn.Parameter, ...]]],
    None,
]


def swap_hf_moe_blocks(
    model: nn.Module,
    *,
    ep_group: dist.ProcessGroup | None = None,
    router_aux_loss_coef: float | None = None,
    quantile_balancing: bool = False,
    fixed_router: bool = False,
    token_dispatcher: str = "alltoall",
    torchao_pad_multiple: int = 16,
    tp_enabled: bool = False,
    on_replacement: MoEReplacementObserver | None = None,
) -> int:
    """Replace every HF MoE block in ``model`` with llmtuner's MoE, in place.

    Args:
        model: a ``HFTransformerModel`` (anything exposing ``.layers``).
        ep_group: the EP process group. ``None`` (or a size-1 group is not
            special-cased -- pass ``None``) keeps all experts local and routes
            them with the reordering-only dispatcher; a multi-rank group
            shards the experts across it and routes with all-to-alls.
        router_aux_loss_coef: coefficient for the per-forward load-balance
            loss. ``None`` takes the HF config's ``router_aux_loss_coef``
            (Qwen3Moe has one; DeepSeek-V3's config has no such field and gets
            no loss). A float here overrides the config for every MoE layer.
        quantile_balancing: replace the sign-based load-balancing bias with
            quantile-balanced routing (``QuantileBalancedTopKRouter``): the
            bias is then re-solved from a required-bias histogram once per
            optimizer step instead of nudged by the sign rule, and
            ``load_balance_coeff`` is forced off. Requires sigmoid router
            scores and no group-limited routing.
        fixed_router: use FSDPTurbo fixed assignments and uniform debug weights.
        token_dispatcher: EP dispatch backend (``"alltoall"`` default,
            ``"torchao"`` optional-import adapter, ``"mc2"`` Ascend fusion,
            ``"domino"`` two-slice pipeline, ``"deepep"``/``"hybridep"``
            registered gaps refused here). See
            ``ParallelConfig.ep_token_dispatcher``.
        torchao_pad_multiple: padding multiple for the ``"torchao"`` backend.
        tp_enabled: the model is also tensor-parallelized (tp x ep). Used only
            for fail-fast validation of unverified combinations (currently a
            shared expert): the swap itself is layout-identical either way --
            the swapped block consumes and produces the T/tp sequence shard
            directly, which is the layout upstream's ep+sp MoE uses.
        on_replacement: receive module and fused-parameter mappings after all
            replacements are planned, before any block is installed. Used by
            assembly to preserve programmatic references across layout changes.

    Returns:
        The number of blocks swapped. Mixed sparse/dense models (e.g.
        Qwen3Moe's ``decoder_sparse_step``) swap only the sparse layers.

    Raises:
        UnsupportedCombinationError: if no layer carries a recognizable HF MoE
            block. EP on a dense model is a config mistake; swapping nothing
            would run it replicated.
        NotImplementedError: if a MoE block uses a layout this swap does not
            implement -- GPT-OSS's transposed, bias-bearing experts (see
            ``fused_experts_of``) or DeepSeek-V2's ``group_limited_greedy``
            routing (see ``read_expert_groups``).
    """
    layers = getattr(model, "layers", None)
    if layers is None:
        raise UnsupportedCombinationError(
            f"swap_hf_moe_blocks expects a model with .layers; got "
            f"{type(model).__name__}."
        )
    # Backend gating, before any probing: ParallelConfig.__post_init__ is the
    # primary gate; this is the defensive copy for callers that reach the swap
    # directly.
    ExpertParallelConfig(
        router_aux_loss_coef=router_aux_loss_coef,
        moe_quantile_balancing=quantile_balancing,
        fixed_router=fixed_router,
        ep_token_dispatcher=token_dispatcher,
        ep_torchao_pad_multiple=torchao_pad_multiple,
    )

    hf_config = getattr(getattr(model, "model", None), "config", None)
    aux_loss_coeff = (
        router_aux_loss_coef
        if router_aux_loss_coef is not None
        else getattr(hf_config, "router_aux_loss_coef", None)
    )
    validate_router_aux_loss_coefficient(aux_loss_coeff)
    # The load-balance coefficient belongs to a *native* llmtuner run; an HF
    # checkpoint has no equivalent field. It is therefore derived from the
    # probed bias: a block that carries ``e_score_correction_bias`` is one the
    # model was trained with the auxiliary-loss-free scheme on (DeepSeek-V3,
    # GLM4), and gets the llmtuner default. Everything else stays off, so a
    # frozen zero bias never lands in the checkpoint.
    default_coeff = getattr(hf_config, "load_balance_coeff", 1e-3)

    # Probe every layer before constructing replacements: a later unsupported
    # expert layout must not leave earlier layers converted.
    candidates: list[tuple[nn.Module, str, nn.Module]] = []
    for layer in layers:
        attr, block = moe_block_of(layer)
        # Dense layers of a mixed model (Qwen3Moe's decoder_sparse_step, and the
        # first_k_dense_replace of the DeepSeek/GLM4 families) are left alone.
        if block is None or not (
            isinstance(block, KimiSparseMoE) or is_hf_moe_block(block)
        ):
            continue
        candidates.append((layer, attr, block))

    if not candidates:
        raise UnsupportedCombinationError(
            f"no HF MoE block found on {type(model).__name__} "
            f"({type(getattr(model, 'model', model)).__name__}): no layer's "
            "``mlp`` has the router-gate + fused gate_up_proj/down_proj shape "
            "this swap recognizes. Add the model family's layout to the probe "
            "in parallel/expert_parallel/hf_inspection.py."
        )

    module_paths = list(model.named_modules(remove_duplicate=False))
    replaced_roots = [
        f"{path}.{attr}" if path else attr
        for path, module in module_paths
        for layer, attr, _ in candidates
        if module is layer
    ]
    external_parameters: dict[int, list[str]] = {}
    for path, module in module_paths:
        if any(path == root or path.startswith(f"{root}.") for root in replaced_roots):
            continue
        for name, parameter in module.named_parameters(
            recurse=False, remove_duplicate=False
        ):
            fqn = f"{path}.{name}" if path else name
            external_parameters.setdefault(id(parameter), []).append(fqn)

    replacements: list[tuple[nn.Module, str, MoE]] = []
    converted_blocks: dict[int, MoE] = {}
    converted_parameters: dict[int, tuple[nn.Parameter, ...]] = {}
    original_parameters = (
        {id(parameter): parameter for parameter in model.parameters()}
        if on_replacement is not None
        else {}
    )
    converted_biases: dict[int, torch.Tensor] = {}
    aux_group_counts = dict(AuxLoss._group_counts)
    # Stacking a projection cannot preserve individual expert aliases. Reject
    # them before constructing any replacement, including cross-block aliases.
    kimi_expert_ids = {
        id(parameter)
        for _, _, block in candidates
        if isinstance(block, KimiSparseMoE)
        for parameter in block.experts.parameters()
    }
    owners: dict[int, tuple[int, str]] = {}
    checked_kimi_blocks: set[int] = set()
    for _, _, block in candidates:
        if not isinstance(block, KimiSparseMoE) or id(block) in checked_kimi_blocks:
            continue
        checked_kimi_blocks.add(id(block))
        for name, parameter in block.experts.named_parameters(remove_duplicate=False):
            owner = (id(block), name)
            previous = owners.setdefault(id(parameter), owner)
            if previous != owner:
                raise UnsupportedCombinationError(
                    "Kimi EP cannot stack a shared individual expert parameter "
                    f"at experts.{name}; expert aliases require a layout adapter"
                )
    owners.clear()
    for path, module in module_paths:
        for name, parameter in module.named_parameters(recurse=False):
            if id(parameter) not in kimi_expert_ids:
                continue
            owner = (id(module), name)
            previous = owners.setdefault(id(parameter), owner)
            if previous != owner or id(parameter) in external_parameters:
                raise UnsupportedCombinationError(
                    f"Kimi EP cannot stack a shared individual expert parameter "
                    f"at {path}.{name}; expert aliases require a layout adapter"
                )
    try:
        for layer, attr, block in candidates:
            # Repeated layer slots may reference one physical HF block. Preserve
            # that sharing rather than cloning trainable weights and statistics.
            if id(block) in converted_blocks:
                replacements.append((layer, attr, converted_blocks[id(block)]))
                continue
            if isinstance(block, KimiSparseMoE):
                moe = convert_kimi_block(
                    block,
                    ep_group=ep_group,
                    aux_loss_coeff=aux_loss_coeff,
                    quantile_balancing=quantile_balancing,
                    token_dispatcher=token_dispatcher,
                    tp_enabled=tp_enabled,
                )
                moe.router.fixed_router = fixed_router
                if on_replacement is not None:
                    grouped = moe.routed_experts.inner_experts
                    for expert in block.experts:
                        for source_name, target_name in (
                            ("w1", "w1_EFD"),
                            ("w2", "w2_EDF"),
                            ("w3", "w3_EFD"),
                        ):
                            source = getattr(expert, source_name).weight
                            converted_parameters[id(source)] = (
                                getattr(grouped, target_name),
                            )
                converted_blocks[id(block)] = moe
                replacements.append((layer, attr, moe))
                continue
            router_gate = router_of(block)
            load_balance_coeff = (
                float(default_coeff)
                if getattr(router_gate, "e_score_correction_bias", None) is not None
                else None
            )
            moe = convert_block(
                block,
                ep_group=ep_group,
                aux_loss_coeff=aux_loss_coeff,
                load_balance_coeff=load_balance_coeff,
                quantile_balancing=quantile_balancing,
                token_dispatcher=token_dispatcher,
                torchao_pad_multiple=torchao_pad_multiple,
                tp_enabled=tp_enabled,
            )
            # Distinct blocks may tie only a projection. The fused gate/up
            # source maps to two parameters, which must be reused as a pair.
            # Bind after dtype/device conversion, before installing any block.
            grouped = moe.routed_experts.inner_experts
            for owner, names, source in (
                (moe.router.gate, ("weight",), router_gate.weight),
                (grouped, ("w1_EFD", "w3_EFD"), block.experts.gate_up_proj),
                (grouped, ("w2_EDF",), block.experts.down_proj),
            ):
                shared = converted_parameters.get(id(source))
                if shared is None:
                    targets = tuple(getattr(owner, name) for name in names)
                    # Unchanged layouts can keep the original Parameter, also
                    # preserving consumers outside the converted MoE subtree.
                    # Split gate/up and EP-sharded projections have new layouts.
                    if (
                        len(targets) == 1
                        and isinstance(source, nn.Parameter)
                        and type(source) is type(targets[0])
                        and source.shape == targets[0].shape
                        and source.dtype == targets[0].dtype
                        and source.device == targets[0].device
                    ):
                        setattr(owner, names[0], source)
                        targets = (source,)
                    external = external_parameters.get(id(source), [])
                    if external and not (len(targets) == 1 and targets[0] is source):
                        raise UnsupportedCombinationError(
                            f"MoE conversion changes the layout of a parameter "
                            f"also registered outside converted blocks at {external}; "
                            "split or EP-sharded expert aliases require an adapted "
                            "external consumer. Register aliases to the converted "
                            "expert parameters after EP conversion."
                        )
                    converted_parameters[id(source)] = targets
                else:
                    for name, parameter in zip(names, shared, strict=True):
                        setattr(owner, name, parameter)
            # Shared bias consumers contribute counts or histograms to one
            # policy-specific update in bias_update.py.
            source_bias = getattr(router_gate, "e_score_correction_bias", None)
            if isinstance(source_bias, torch.Tensor) and moe.expert_bias_E is not None:
                if id(source_bias) not in converted_biases:
                    converted_biases[id(source_bias)] = moe.expert_bias_E
                else:
                    moe.expert_bias_E = converted_biases[id(source_bias)]
            # Fixed assignment and constant scores belong to the router,
            # shared by every dispatcher.
            moe.router.fixed_router = fixed_router
            converted_blocks[id(block)] = moe
            replacements.append((layer, attr, moe))
        # TP registers physical HF slices before this swap. Gate/up becomes two
        # independent parameters, so each checkpoint slice now has its own name
        # and no longer uses packed-half gathering. Preserve the registration's
        # owner, which may be the model root rather than the replaced block.
        converted_layouts = []
        for owner_path, owner in module_paths:
            layouts = owner.__dict__.get("_tp_checkpoint_layouts")
            if layouts is None:
                continue
            expert_owner = next(
                (
                    converted_blocks[id(block)].routed_experts.inner_experts
                    for _, _, block in candidates
                    if not isinstance(block, KimiSparseMoE) and owner is block.experts
                ),
                None,
            )
            if expert_owner is not None:
                updated = {}
                for name, layout in layouts.items():
                    names = {
                        "gate_up_proj": ("w1_EFD", "w3_EFD"),
                        "down_proj": ("w2_EDF",),
                    }.get(name, (name,))
                    for target_name in names:
                        updated[target_name] = (
                            (layout[0], False, layout[2])
                            if name in ("gate_up_proj", "down_proj")
                            else layout
                        )
                converted_layouts.append((expert_owner, updated))
                continue
            updated = {}
            for name, layout in layouts.items():
                full_name = f"{owner_path}.{name}" if owner_path else name
                replacement = None
                for root in replaced_roots:
                    if full_name == f"{root}.experts.gate_up_proj":
                        replacement = ("w1_EFD", "w3_EFD")
                        break
                    if full_name == f"{root}.experts.down_proj":
                        replacement = ("w2_EDF",)
                        break
                if replacement is None:
                    for root in replaced_roots:
                        prefix = f"{root}.shared_expert."
                        if full_name.startswith(prefix) and "shared_expert." in name:
                            # The reused shared module keeps its own relative
                            # registrations. Only an ancestor's path needs renaming.
                            parent, suffix = name.rsplit("shared_expert.", 1)
                            name = f"{parent}shared_experts.{suffix}"
                            break
                    updated[name] = layout
                    continue
                prefix = name.rsplit("experts.", 1)[0]
                for parameter_name in replacement:
                    updated[
                        f"{prefix}routed_experts.inner_experts.{parameter_name}"
                    ] = (
                        layout[0],
                        False,
                        layout[2],
                    )
            converted_layouts.append((converted_blocks.get(id(owner), owner), updated))

        if on_replacement is not None:
            module_targets = {}
            for _, _, block in candidates:
                moe = converted_blocks[id(block)]
                module_targets[block] = moe
                module_targets[block.experts] = moe.routed_experts.inner_experts
                if not isinstance(block, KimiSparseMoE):
                    # Kimi's gate also owns a correction-bias Parameter which
                    # moves outside router.gate. Mapping only that subtree
                    # would silently drop part of a module exclusion.
                    module_targets[router_of(block)] = moe.router.gate
            on_replacement(
                module_targets,
                {
                    original_parameters[source]: targets
                    for source, targets in converted_parameters.items()
                },
            )

    except BaseException:
        # Failed replacement planning must not inflate auxiliary-loss metric divisors.
        AuxLoss._group_counts.clear()
        AuxLoss._group_counts.update(aux_group_counts)
        raise

    # Install only after every replacement was constructed successfully.
    # A later conversion error must not leave a mixture of HF and native blocks.
    for layer, attr, moe in replacements:
        # Replace in the slot the block was actually found in. Writing to a
        # different attribute would leave the original block in place and route
        # around the swap entirely.
        setattr(layer, attr, moe)
        # FSDP's MoE branch (fully_shard_parallel/plan.py) keys off these two
        # attributes, as does upstream's swap (moe_replacement.py):
        # ``moe_enabled`` marks the block as sparse, and ``moe`` is where the
        # branch reads the expert weights from. Without them the experts are
        # sharded as dense parameters over the dense DP mesh, mixing ranks of
        # different EP coordinates into one FSDP group. ``object.__setattr__``
        # keeps ``moe`` out of the module registry: it is the same module as
        # the block attribute, and registering it would double every expert
        # weight in the state_dict.
        layer.moe_enabled = True
        object.__setattr__(layer, "moe", moe)

    for owner, layouts in converted_layouts:
        owner._tp_checkpoint_layouts = layouts

    logger.info("Swapped %d HF MoE blocks for llmtuner MoE blocks", len(replacements))
    return len(replacements)
