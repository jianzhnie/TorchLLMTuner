"""Resolve EP targets and validate ownership before model mutation.

Plans retain concrete bindings and model-owned routing declarations. Resolution
never installs blocks, allocates replacement weights or executes collectives.
Pipeline validation compares global semantics with freshly bound local targets.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ...config.parallel_config import validate_expert_dispatcher
from ...errors import UnsupportedCombinationError
from ...models.common.moe.layer import (
    MoE,
    declared_expert_layout,
    declared_moe_block,
)
from ...models.common.moe.layout import (
    MoERoutingBinding,
    PackedExpertBinding,
    resolve_moe_routing,
    resolve_packed_experts,
)
from ...models.contracts import DecoderStructure, MoEBlockContract
from ..plan import (
    FullPlanBindings,
    ParallelPlan,
    PlanTarget,
    first_paths_by_identity,
)
from .block_conversion import validate_moe_conversion
from .model_conversion import uses_model_expert_conversion
from .ownership import (
    conversion_roots,
    external_state_owners,
    validate_model_expert_ownership,
)


def declared_router_output(block: nn.Module) -> bool:
    """Resolve explicit output semantics without coercing or inspecting forward.

    A retained per-block flag must agree with the model-owned MoE contract.
    Contract-only declarations do not need to duplicate the flag.
    """
    contract = getattr(block, "moe_contract", None)
    expected = (
        contract.router_logits_in_output
        if isinstance(contract, MoEBlockContract)
        else False
    )
    declared = getattr(block, "router_logits_in_output", expected)
    if type(declared) is not bool:
        raise ValueError("MoE router_logits_in_output must be boolean")
    if isinstance(contract, MoEBlockContract) and declared != expected:
        raise ValueError("MoE router output flag conflicts with its declared contract")
    return declared


def validate_router_output_conversion(
    block: nn.Module, wants_router_logits: bool, config: Any
) -> None:
    """Preserve the output restriction even when config changes after parsing."""
    if wants_router_logits and getattr(config, "output_router_logits", False):
        raise UnsupportedCombinationError(
            f"{type(block).__name__} returns router logits and its config "
            "enables output_router_logits, but the swapped-in llmtuner MoE "
            "keeps its router scores internal, so it could only hand back "
            "None. Turn output_router_logits off -- llmtuner runs its own "
            "load-balancing loss -- before swapping."
        )


def decoder_layers_for_ep(model: nn.Module) -> Iterable[nn.Module]:
    """Bind explicit local decoder ownership without requiring PP edge modules.

    A declared model must not recover from a malformed contract by reading a
    similarly named convenience property. Resolve current registered children
    so PP copies and parameter replacement cannot leave cached references.
    """
    structure = getattr(model, "decoder_structure", None)
    if not isinstance(structure, DecoderStructure):
        raise TypeError(
            "EP requires a model-owned decoder_structure (DecoderStructure)"
        )
    return structure.bind_blocks(model)


def find_moe_candidates(
    model: nn.Module, layers: Iterable[nn.Module], hf_config: Any
) -> list[tuple[nn.Module, str, nn.Module, bool]]:
    """Resolve model-declared sparse slots before allocating replacement weights."""
    candidates = []
    for layer in layers:
        declared = declared_moe_block(layer)
        if declared is None:
            raise ValueError(
                f"Explicit decoder {type(layer).__name__} requires "
                "moe_block_name (None for a dense layer)"
            )
        attr, block = declared
        if block is None:
            continue
        if not isinstance(block, MoE) and not uses_model_expert_conversion(block):
            if not isinstance(getattr(block, "moe_contract", None), MoEBlockContract):
                raise UnsupportedCombinationError(
                    f"Declared MoE slot {type(layer).__name__}.{attr} requires a "
                    "model-owned MoEBlockContract"
                )
        wants_router_logits = declared_router_output(block)
        validate_router_output_conversion(block, wants_router_logits, hf_config)
        candidates.append((layer, attr, block, wants_router_logits))

    if not candidates:
        raise UnsupportedCombinationError(
            f"no declared or compatible MoE block found on {type(model).__name__}; "
            "EP requires a sparse decoder slot with a supported expert layout. "
            "Declare moe_block_name and a model-owned expert layout adapter."
        )
    return candidates


@dataclass(frozen=True, slots=True)
class MoEConversionTarget:
    """Decoder slot, output convention and routing choices fixed by parsing."""

    owner: nn.Module
    attribute: str
    router_logits: bool
    routing: MoERoutingBinding | None = None
    packed_experts: PackedExpertBinding | None = None
    contract: MoEBlockContract | None = None
    model_math: tuple[object, ...] | None = None
    declared_slot: bool = False
    model_conversion: bool = False


def validate_packed_expert_aliases(
    targets: Iterable[PlanTarget],
    module_paths: Iterable[tuple[str, nn.Module]],
    replaced_roots: list[str],
    *,
    degree: int,
) -> None:
    """Reject aliases that packed splitting or expert sharding cannot retain.

    Same-parameter sharing between converted blocks remains supported. Local
    down projections keep their Parameter and need no alias restriction. Read
    storage metadata only; do not allocate expert views or replacements here.
    """
    changed: set[int] = set()
    storages: dict[int, int] = {}
    for target in targets:
        binding = target.spec
        if not isinstance(binding, MoEConversionTarget):
            continue
        packed = binding.packed_experts
        if packed is None:
            continue
        for parameter in (packed.gate_up, *((packed.down,) if degree > 1 else ())):
            changed.add(id(parameter))
            if type(parameter) is nn.Parameter and parameter.layout == torch.strided:
                storage = parameter.untyped_storage()._cdata
                previous = storages.setdefault(storage, id(parameter))
                if previous != id(parameter):
                    raise UnsupportedCombinationError(
                        "EP cannot preserve packed expert parameters sharing storage"
                    )
    if not changed:
        return
    for path, module in module_paths:
        local = any(
            path == root or path.startswith(root + ".") for root in replaced_roots
        )
        for is_buffer, tensors in (
            (False, module.named_parameters(recurse=False, remove_duplicate=False)),
            (True, module.named_buffers(recurse=False, remove_duplicate=False)),
        ):
            for name, tensor in tensors:
                aliases_changed = id(tensor) in changed
                if (
                    type(tensor) in (torch.Tensor, nn.Parameter)
                    and tensor.layout == torch.strided
                ):
                    owner = storages.get(tensor.untyped_storage()._cdata)
                    aliases_changed |= owner is not None and owner != id(tensor)
                if aliases_changed and (
                    is_buffer or not local or id(tensor) not in changed
                ):
                    raise UnsupportedCombinationError(
                        "EP cannot preserve packed expert storage alias at "
                        f"{path + '.' if path else ''}{name}; split or sharded "
                        "expert state requires an adapted consumer"
                    )


def validate_ep_model_options(config: Any, implementation: str) -> None:
    """Reject unsupported execution choices before binding or changing a model.

    Parsing and execution share this check: a supplied plan does not authorize
    options changed after parsing, including score multiplication placement.
    """
    validate_expert_dispatcher(implementation)
    if getattr(config, "experts_implementation", "native") == "pre_down":
        raise UnsupportedCombinationError(
            "EP MoE replacement cannot preserve experts_implementation='pre_down'; "
            "use native experts or retain the local/DP/FSDP expert path. "
            "This also applies to single-rank fixed-router replacement."
        )


def validate_ep_model_geometry(
    model: nn.Module, *, degree: int, tp_enabled: bool
) -> None:
    """Validate native geometry while the model still owns unsharded storage.

    Reuse this before execution: config fields can change without invalidating
    module or tensor bindings. TP validates native geometry before sharding;
    never apply that native contract to already transformed parameters.
    """
    validate = getattr(model, "validate_parallel", None)
    if (
        callable(validate)
        and not tp_enabled
        and getattr(model, "parallel_plan", None) is None
        and getattr(model, "ep_parallel_plan", None) is None
    ):
        validate(tp_degree=1, ep_degree=degree)


def resolve_ep_plan(
    model: nn.Module,
    *,
    degree: int = 1,
    implementation: str = "alltoall",
    tp_enabled: bool = False,
    quantile_balancing: bool = False,
    has_ep_group: bool | None = None,
    bindings: FullPlanBindings | None = None,
) -> ParallelPlan:
    """Bind EP conversion targets without allocating weights or running collectives.

    Generic routing restrictions are checked with the supplied execution options.
    Group presence defaults to degree > 1; pass it explicitly for size-one groups.
    Execution revalidates restrictions defensively if options or metadata change.
    Generic conversion consumes explicit routing and packed storage declarations
    rather than rediscovering them during execution. Re-resolve after config,
    TP or PP changes. Kimi retains its dedicated conversion contract.
    """
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError("EP degree must be a positive integer")
    config = getattr(model, "config", None)
    validate_ep_model_options(config, implementation)
    layers = decoder_layers_for_ep(model)
    candidates = find_moe_candidates(model, layers, config)
    modules = (
        tuple(model.named_modules(remove_duplicate=False))
        if bindings is None
        else bindings["module_bindings"]
    )
    first_paths = first_paths_by_identity(modules)
    targets = []
    for owner, attribute, block, router_logits in candidates:
        owner_path = first_paths.get(id(owner))
        if owner_path is None:
            raise ValueError("EP decoder owner is not registered in this model")
        routing = None
        packed_experts = None
        model_conversion = uses_model_expert_conversion(block)
        if model_conversion:
            block.validate_expert_parallel(
                degree=degree,
                tp_enabled=tp_enabled,
                quantile_balancing=quantile_balancing,
                token_dispatcher=implementation,
            )
        else:
            routing = resolve_moe_routing(block)
            packed_experts = resolve_packed_experts(block, degree=degree)
            validate_moe_conversion(
                block,
                routing,
                tp_enabled=tp_enabled,
                quantile_balancing=quantile_balancing,
                token_dispatcher=implementation,
                has_ep_group=degree > 1 if has_ep_group is None else has_ep_group,
            )
        targets.append(
            PlanTarget(
                f"{owner_path}.{attribute}" if owner_path else attribute,
                "moe",
                block,
                declared_expert_layout(block) or "fused_gate_up",
                MoEConversionTarget(
                    owner,
                    attribute,
                    router_logits,
                    routing,
                    packed_experts,
                    getattr(block, "moe_contract", None),
                    (
                        block.expert_parallel_math_signature()
                        if callable(
                            getattr(block, "expert_parallel_math_signature", None)
                        )
                        else None
                    ),
                    declared_slot=hasattr(owner, "moe_block_name"),
                    model_conversion=model_conversion,
                ),
            )
        )
    module_paths = list(modules)
    roots = conversion_roots(candidates, module_paths)
    validate_packed_expert_aliases(targets, module_paths, roots, degree=degree)
    if any(
        callable(getattr(block, "validate_expert_parallel_ownership", None))
        for _, _, block, _ in candidates
    ):
        external_parameters, _ = external_state_owners(module_paths, roots)
        validate_model_expert_ownership(candidates, module_paths, external_parameters)
    validate_ep_model_geometry(model, degree=degree, tp_enabled=tp_enabled)
    return ParallelPlan(
        stage="ep",
        targets=tuple(targets),
        degree=degree,
        implementation=implementation,
        **(
            {
                "module_bindings": modules,
                "parameter_bindings": tuple(
                    model.named_parameters(remove_duplicate=False)
                ),
                "buffer_bindings": tuple(model.named_buffers(remove_duplicate=False)),
            }
            if bindings is None
            else bindings
        ),
    )
