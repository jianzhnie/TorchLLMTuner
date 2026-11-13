"""Plan and install native MoE replacements before FSDP wrapping.

Explicit models supply decoder and MoE slot declarations. Target resolution
lives in ``plan.py``; alias ownership lives in ``ownership.py``. Weight
conversion belongs to ``block_conversion.py`` or ``kimi_conversion.py``.
This module installs validated replacements after weights are loaded.

All replacements, shared-parameter bindings and TP checkpoint registrations
are planned before any decoder slot is replaced. Unsupported aliases or layouts
therefore fail without leaving a mixture of HF and native blocks. Failed planning
also restores auxiliary-loss group counts.

Installation preserves repeated block references and marks decoder ownership
with ``moe_enabled`` and a non-registered ``moe`` alias. FSDP uses those markers
to shard experts over the sparse mesh rather than the dense DP mesh. The alias
must remain outside the module registry to avoid duplicate state-dict keys.

``on_replacement`` receives the planned module and parameter mapping before
installation; callers use it to migrate metadata tied to the original objects.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from llmtuner.config.parallel_config import (
    ExpertParallelConfig,
    validate_router_aux_loss_coefficient,
)
from llmtuner.errors import UnsupportedCombinationError

from ...models.common.aux_loss import AuxLoss
from ...models.common.moe.layer import MoE, declared_moe_block
from ...models.contracts import DecoderStructure
from ...models.kimi.kimi_k3.moe import KimiSparseMoE
from ...utils.logging import get_logger
from ..plan import ParallelPlan, ReplacementObserver
from .block_conversion import (
    convert_block,
    validate_moe_conversion,
)
from .kimi_conversion import convert_kimi_block
from .ownership import (
    bind_shared_projection,
    conversion_roots,
    external_state_owners,
    plan_tp_checkpoint_layouts,
    validate_kimi_expert_aliases,
)
from .plan import (
    MoEConversionTarget,
    declared_router_output,
    decoder_layers_for_ep,
    resolve_ep_plan,
    validate_ep_model_geometry,
    validate_ep_model_options,
    validate_packed_expert_aliases,
    validate_router_output_conversion,
)

logger = get_logger(__name__)

__all__ = ["resolve_ep_plan", "swap_moe_blocks"]


def swap_moe_blocks(
    model: nn.Module,
    *,
    ep_group: dist.ProcessGroup | None = None,
    router_aux_loss_coef: float | None = None,
    quantile_balancing: bool = False,
    fixed_router: bool = False,
    token_dispatcher: str = "alltoall",
    torchao_pad_multiple: int = 16,
    tp_enabled: bool = False,
    on_replacement: ReplacementObserver | None = None,
    resolved_plan: ParallelPlan | None = None,
) -> int:
    """Replace every HF MoE block in ``model`` with llmtuner's MoE, in place.

    Args:
        model: an explicit model (anything exposing ``.layers``).
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
        resolved_plan: optional target plan resolved for this exact model and
            EP degree/backend. Stale module or parameter bindings are rejected.
        on_replacement: receive module and fused-parameter mappings after all
            replacements are planned, before any block is installed. Used by
            assembly to preserve programmatic references across layout changes.

    Returns:
        The number of blocks swapped. Mixed sparse/dense models (e.g.
        Qwen3Moe's ``decoder_sparse_step``) swap only the sparse layers.

    Raises:
        UnsupportedCombinationError: if sparse slots lack explicit contracts or
            request unsupported routing, expert storage or dispatch semantics.
    """
    decoder_layers_for_ep(model)
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

    hf_config = getattr(model, "config", None)
    validate_ep_model_options(hf_config, token_dispatcher)
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

    # Bind targets after preceding stages (TP/PP) have stabilized this instance.
    degree = 1 if ep_group is None else dist.get_world_size(ep_group)
    execution_plan = resolved_plan or resolve_ep_plan(
        model,
        degree=degree,
        implementation=token_dispatcher,
        tp_enabled=tp_enabled,
        quantile_balancing=quantile_balancing,
        has_ep_group=ep_group is not None,
    )
    if (
        execution_plan.stage != "ep"
        or execution_plan.degree != degree
        or execution_plan.implementation != token_dispatcher
    ):
        raise ValueError("EP plan does not match the requested execution")
    execution_plan.validate_binding(model)
    candidates = []
    routing_bindings = {}
    packed_bindings = {}
    for target in execution_plan.targets:
        binding = target.spec
        if target.kind != "moe" or not isinstance(binding, MoEConversionTarget):
            raise ValueError("EP plan requires resolved MoE conversion targets")
        if binding.owner._modules.get(binding.attribute) is not target.module:
            raise ValueError(f"Stale EP decoder slot: {target.path}")
        # Module identity alone cannot detect a model changing a sparse slot
        # to dense, or declaring a different child after plan resolution.
        if hasattr(binding.owner, "moe_block_name") != binding.declared_slot or (
            binding.declared_slot
            and getattr(binding.owner, "moe_block_name", None) != binding.attribute
        ):
            raise ValueError(f"Stale EP decoder declaration: {target.path}")
        if isinstance(target.module, KimiSparseMoE) and (
            target.module.expert_parallel_math_signature() != binding.kimi_math
        ):
            raise ValueError(f"Stale EP Kimi mathematics: {target.path}")
        if getattr(target.module, "moe_contract", None) != binding.contract:
            raise ValueError(f"Stale EP model contract: {target.path}")
        if binding.contract is not None:
            try:
                current_output = declared_router_output(target.module)
            except ValueError as error:
                raise ValueError(
                    f"Stale EP router output contract: {target.path}"
                ) from error
            if current_output != binding.router_logits:
                raise ValueError(f"Stale EP router output contract: {target.path}")
        validate_router_output_conversion(
            target.module, binding.router_logits, hf_config
        )
        candidates.append(
            (binding.owner, binding.attribute, target.module, binding.router_logits)
        )
        if isinstance(target.module, KimiSparseMoE):
            if target.layout != "kimi_per_expert":
                raise ValueError("EP Kimi target requires its per-expert layout")
            # Revalidate all targets before constructing the first replacement:
            # execution options/config can change without replacing tensors.
            target.module.validate_expert_parallel(
                degree=degree,
                tp_enabled=tp_enabled,
                quantile_balancing=quantile_balancing,
                token_dispatcher=token_dispatcher,
            )
        else:
            if target.layout != "fused_gate_up" or binding.packed_experts is None:
                raise ValueError("EP plan requires a resolved fused gate/up binding")
            if binding.routing is None or binding.routing.source is not target.module:
                raise ValueError("EP plan requires a matching routing binding")
            validate_moe_conversion(
                target.module,
                binding.routing,
                tp_enabled=tp_enabled,
                quantile_balancing=quantile_balancing,
                token_dispatcher=token_dispatcher,
                has_ep_group=ep_group is not None,
            )
            routing_bindings[id(target.module)] = binding.routing
            binding.packed_experts.validate(target.module)
            packed_bindings[id(target.module)] = binding.packed_experts
    if not candidates:
        raise ValueError("EP plan must contain conversion targets")
    if isinstance(getattr(model, "decoder_structure", None), DecoderStructure):
        # EP replaces every declared sparse slot, in decoder order. A caller
        # can rebuild a frozen plan with missing/repeated targets while all
        # module snapshots remain valid; that must not produce a mixed model.
        expected_slots = []
        for layer in decoder_layers_for_ep(model):
            declared = declared_moe_block(layer)
            if declared is None:
                raise ValueError("Explicit EP requires declared decoder MoE slots")
            attribute, block = declared
            if block is not None:
                expected_slots.append((id(layer), attribute, id(block)))
        actual_slots = [
            (id(owner), attribute, id(block))
            for owner, attribute, block, _ in candidates
        ]
        if actual_slots != expected_slots:
            raise ValueError("EP plan must cover declared MoE slots in decoder order")
    validate_ep_model_geometry(model, degree=degree, tp_enabled=tp_enabled)

    module_paths = list(model.named_modules(remove_duplicate=False))
    replaced_roots = conversion_roots(candidates, module_paths)
    validate_packed_expert_aliases(
        execution_plan.targets, module_paths, replaced_roots, degree=degree
    )
    external_parameters, external_buffers = external_state_owners(
        module_paths, replaced_roots
    )

    replacements: list[tuple[nn.Module, str, MoE]] = []
    converted_blocks: dict[int, MoE] = {}
    converted_parameters: dict[int, tuple[nn.Parameter, ...]] = {}
    stacked_projection_sources: dict[int, tuple[int, ...]] = {}
    original_parameters = (
        {id(parameter): parameter for parameter in model.parameters()}
        if on_replacement is not None
        else {}
    )
    converted_biases: dict[int, torch.Tensor] = {}
    aux_group_counts = dict(AuxLoss._group_counts)
    validate_kimi_expert_aliases(candidates, module_paths, external_parameters)
    try:
        for layer, attr, block, wants_router_logits in candidates:
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
            router_gate = routing_bindings[id(block)].router
            load_balance_coeff = (
                float(default_coeff)
                if getattr(router_gate, "e_score_correction_bias", None) is not None
                else None
            )
            moe = convert_block(
                block,
                routing=routing_bindings[id(block)],
                packed_experts=packed_bindings.get(id(block)),
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
            projections: list[
                tuple[nn.Module, tuple[str, ...], tuple[torch.Tensor, ...]]
            ] = [
                (moe.router.gate, ("weight",), (router_gate.weight,)),
                (grouped, ("w1_EFD", "w3_EFD"), (block.experts.gate_up_proj,)),
                (grouped, ("w2_EDF",), (block.experts.down_proj,)),
            ]
            for owner, names, sources in projections:
                bind_shared_projection(
                    owner,
                    names,
                    sources,
                    converted_parameters=converted_parameters,
                    stacked_projection_sources=stacked_projection_sources,
                    external_parameters=external_parameters,
                )
            # Shared bias consumers contribute counts or histograms to one
            # policy-specific update in bias_update.py.
            source_bias = getattr(router_gate, "e_score_correction_bias", None)
            if isinstance(source_bias, torch.Tensor) and moe.expert_bias_E is not None:
                if id(source_bias) not in converted_biases:
                    target_bias = moe.expert_bias_E
                    if (
                        source_bias.dtype == target_bias.dtype
                        and source_bias.device == target_bias.device
                    ):
                        # Bias updates must remain visible to registered consumers
                        # outside the replaced subtree, just like shared weights.
                        moe.expert_bias_E = source_bias
                    elif id(source_bias) in external_buffers:
                        raise UnsupportedCombinationError(
                            "MoE conversion cannot preserve an external router bias "
                            "alias while changing its dtype or device: "
                            f"{external_buffers[id(source_bias)]}"
                        )
                    converted_biases[id(source_bias)] = moe.expert_bias_E
                else:
                    moe.expert_bias_E = converted_biases[id(source_bias)]
            # Fixed assignment and constant scores belong to the router,
            # shared by every dispatcher.
            moe.router.fixed_router = fixed_router
            # Keep the host layer's unpacking contract intact: the block it was
            # written against returns a tuple on 4.x Mixtral/OLMoE/GraniteMoe.
            moe.returns_router_logits = wants_router_logits
            converted_blocks[id(block)] = moe
            replacements.append((layer, attr, moe))
        converted_layouts = plan_tp_checkpoint_layouts(
            module_paths, candidates, converted_blocks, replaced_roots
        )

        if on_replacement is not None:
            module_targets = {}
            for _, _, block, _ in candidates:
                moe = converted_blocks[id(block)]
                module_targets[block] = moe
                module_targets[block.experts] = moe.routed_experts.inner_experts
                router = (
                    block.gate
                    if isinstance(block, KimiSparseMoE)
                    else routing_bindings[id(block)].router
                )
                module_targets[router] = moe.router.gate
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

    model.ep_parallel_plan = execution_plan
    logger.info("Converted %d model MoE blocks for EP execution", len(replacements))
    return len(replacements)
