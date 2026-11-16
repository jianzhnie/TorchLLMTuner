"""Apply TP/EP/CP and memory transformations to explicit whole models.

Stages follow the single STAGE_ORDER and return final local layout metadata.
Pipeline execution is isolated in experiments/pipeline_parallel.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext

import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh

from llmtuner.config.activation_checkpoint_config import (
    MemoryBudgetACConfig,
    RegionACConfig,
    SelectiveACConfig,
)
from llmtuner.config.compile_config import CompileConfig
from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.config.training_config import validate_activation_swap_options

from ..accelerator.collectives import synchronize_local_errors
from ..memory.activation_checkpoint import apply_ac, validate_ac_options
from ..memory.activation_swap import apply_activation_swap
from ..models.contracts import DecoderStructure
from .assembly import ModelPartLayout, ParallelAssembly
from .compatibility import chunked_loss_fsdp
from .context_parallel.context_parallel import apply_cp
from .context_parallel.plan import resolve_cp_plan
from .execution import (
    ExecutionLayout,
    resolve_execution_layout,
)
from .expert_parallel.expert_parallel import apply_ep
from .expert_parallel.mc2_ops import validate_mc2_world_size
from .expert_parallel.plan import resolve_ep_plan
from .fully_shard_parallel.fully_shard_parallel import (
    apply_fsdp,
    validate_fsdp_environment,
)
from .fully_shard_parallel.ownership import FSDPPatternReferences
from .model_compile import apply_compile, validate_compile_environment
from .plan import (
    ParallelPlan,
    model_bindings_from_plan,
)
from .stages import STAGE_ORDER
from .tensor_parallel.tensor_parallel import (
    apply_tp,
    resolve_tp_plan,
)
from .topology import ParallelTopology, ParallelTopologySnapshot

__all__ = ["parallelize_model"]


def parallelize_model(
    model: nn.Module,
    *,
    cfg: ParallelConfig,
    mesh: DeviceMesh | None,
    parallel_dims: ParallelTopology | None,
    error_group: ProcessGroup | None = None,
    compile: bool = False,
    compile_config: CompileConfig | None = None,
    activation_checkpoint: str = "none",
    activation_swap: bool = False,
    activation_swap_sync_release: bool = False,
    activation_swap_max_host_bytes: int = 0,
    selective_ac: SelectiveACConfig | None = None,
    memory_budget_ac: MemoryBudgetACConfig | None = None,
    region_ac: RegionACConfig | None = None,
    chunked_loss_num_chunks: int = 1,
    on_model_acquired: Callable[[nn.Module], None] | None = None,
) -> ParallelAssembly:
    """Transform an explicit model using parallel config and training options.

    Compilation and checkpointing transform the whole model. Return the final
    execution module and its local layout; PP is an experimental entry point.

    ``error_group`` names every peer that must stop on local preflight failure.
    It must include all ranks participating in subsequent transforms. Without
    an explicit group, existing callers retain default-group coordination;
    CP preflight uses the topology world when available.

    ``on_model_acquired`` retains cleanup ownership of each
    transformed wrapper before subsequent stages can fail. It must perform
    only local reference bookkeeping, without device work or collectives.
    """

    with (
        synchronize_local_errors("Parallel option validation", group=error_group)
        if error_group is not None
        else nullcontext()
    ):
        if on_model_acquired is not None:
            on_model_acquired(model)
        validate_activation_swap_options(
            activation_swap=activation_swap,
            compile=compile,
            activation_checkpoint=activation_checkpoint,
        )
        validate_ac_options(
            activation_checkpoint,
            selective=selective_ac,
            memory_budget=memory_budget_ac,
            region=region_ac,
            compile_enabled=compile,
        )
    # Inspect original model metadata before parameter replacement.
    needs_blocks = (
        activation_checkpoint not in ("none", "memory_budget")
        or (activation_swap and activation_checkpoint != "full")
        or (compile and compile_config is not None and compile_config.per_block)
    )
    if needs_blocks:
        with synchronize_local_errors(
            "Decoder transformation target validation", group=error_group
        ):
            structure = getattr(model, "decoder_structure", None)
            if structure is not None:
                if not isinstance(structure, DecoderStructure):
                    raise TypeError(
                        "Decoder transformations require a DecoderStructure"
                    )
                structure.bind_blocks(model)
    validate_fsdp_environment(cfg, parallel_dims, error_group=error_group)
    # Backend registration and optional codegen imports are rank-local. Stop
    # every peer before any model transform if one environment is unavailable.
    if compile:
        with synchronize_local_errors(
            "Parallel compile environment validation", group=error_group
        ):
            validate_compile_environment(compile_config)

    if chunked_loss_num_chunks > 1 and compile and parallel_dims is not None:
        if parallel_dims.dp_cp_enabled:
            chunked_loss_fsdp(chunked_loss_num_chunks, compile=True)

    with synchronize_local_errors("Parallel topology snapshot", group=error_group):
        topology_snapshot = (
            ParallelTopologySnapshot.capture(parallel_dims)
            if parallel_dims is not None
            else ParallelTopologySnapshot()
        )
    layouts: list[ModelPartLayout] = []

    def apply_stages(
        part: nn.Module,
        *,
        dense_mesh: DeviceMesh | None,
        tp_mesh: DeviceMesh | None,
        ep_group: ProcessGroup | None,
        ep_error_group: ProcessGroup | None,
        execution_layout: ExecutionLayout,
        part_cfg: ParallelConfig | None = None,
        resolved_tp_plan: ParallelPlan | None = None,
    ) -> nn.Module:
        """Bind stage options once, then transform the model in declared order."""
        fsdp_cfg = cfg if part_cfg is None else part_cfg
        layout = ModelPartLayout(
            execution=execution_layout,
            topology=topology_snapshot,
        )
        references = (
            FSDPPatternReferences(part, fsdp_cfg, replacements=layout.replacements)
            if fsdp_cfg.fsdp_config.fsdp_ignored_modules
            or any(
                options.get("hook_module") is not None or options.get("ignored_params")
                for options in fsdp_cfg.fsdp_config.fsdp_apply_modules.values()
            )
            else None
        )

        def record_replacements(modules, parameters):
            if references is not None:
                references.record_replacements(modules, parameters)
            else:
                layout.replacements.record_replacements(modules, parameters)

        activation_swap_enabled = activation_swap and activation_checkpoint != "full"
        for name in STAGE_ORDER:
            if name == "compile" and not compile:
                continue
            if name == "activation_swap" and not activation_swap_enabled:
                continue
            if name == "tp":
                part = apply_tp(
                    part,
                    mesh=dense_mesh,
                    cfg=cfg,
                    resolved_plan=resolved_tp_plan,
                    on_replacement=record_replacements,
                )
            elif name == "ep":
                # TP may have replaced parameters referenced by FSDP options.
                if references is not None:
                    references = FSDPPatternReferences(
                        part, references.bind(part), replacements=layout.replacements
                    )
                part = apply_ep(
                    part,
                    cfg=cfg,
                    ep_group=ep_group,
                    error_group=ep_error_group,
                    on_replacement=record_replacements,
                )
            elif name == "cp":
                part = apply_cp(part, mesh=dense_mesh, cfg=cfg)
            elif name == "ac":
                part = apply_ac(
                    part,
                    mode=activation_checkpoint,
                    selective=selective_ac,
                    memory_budget=memory_budget_ac,
                    region=region_ac,
                    compile_enabled=compile,
                )
            elif name == "activation_swap":
                part = apply_activation_swap(
                    part,
                    sync_release=activation_swap_sync_release,
                    max_host_bytes=activation_swap_max_host_bytes,
                )
            elif name == "compile":
                part = apply_compile(
                    part, compile_config=compile_config, tp_mesh=tp_mesh
                )
            elif name == "fsdp":
                layout.parameter_bindings = dict(
                    part.named_parameters(remove_duplicate=False)
                )
                part = apply_fsdp(
                    part,
                    cfg=fsdp_cfg if references is None else references.bind(part),
                    parallel_dims=parallel_dims,
                    chunked_loss_num_chunks=chunked_loss_num_chunks,
                )
            else:
                raise ValueError(f"Unknown parallel assembly stage: {name!r}")
            if on_model_acquired is not None:
                on_model_acquired(part)
        with synchronize_local_errors(
            "Transformed model layout validation", group=error_group
        ):
            execution_layout.validate_model(part)
            layout.refresh(part)
        layouts.append(layout)
        return part

    # EP groups come from the sparse mesh.
    ep_group = None
    ep_error_group = None
    if cfg.ep_size > 1:
        if parallel_dims is None:
            raise ValueError(
                f"ep={cfg.ep_size} needs a process group, but this run is "
                "single-process (parallel_dims is None). EP requires "
                "world_size > 1."
            )
        ep_mesh = parallel_dims.get_optional_mesh("ep")
        if ep_mesh is None:
            raise ValueError(
                f"ep={cfg.ep_size} but parallel_dims has no multi-rank 'ep' axis."
            )
        ep_group = ep_mesh.get_group()
        if cfg.ep_config.ep_token_dispatcher in ("domino", "mc2"):
            # The stage-local loss view spans DP/CP/TP, the same ranks that
            # share dense and sparse FSDP communication after expert execution.
            ep_error_group = parallel_dims.get_mesh("loss").get_group()
    tp_mesh = None if parallel_dims is None else parallel_dims.get_optional_mesh("tp")
    # Validate the original model before TP/EP transforms and shared FSDP.
    # Resolve the already-owned world mesh/group outside the local-only body.
    cp_error_group = error_group
    if cp_error_group is None and cfg.cp_size > 1 and parallel_dims is not None:
        cp_error_group = parallel_dims.build_device_mesh().get_group()
    global_bindings = None
    global_cp_plan = None
    if cfg.cp_size > 1:
        with (
            synchronize_local_errors("CP backend validation", group=cp_error_group)
            if cp_error_group is not None
            else nullcontext()
        ):
            # Metadata-only preflight. Later stages resolve current bindings
            # after TP/EP establish the actual local instance.
            global_cp_plan = resolve_cp_plan(model, cfg=cfg)
            if global_cp_plan is not None:
                global_bindings = model_bindings_from_plan(global_cp_plan)

    global_ep_plan = None
    if cfg.ep_size > 1 or cfg.ep_config.fixed_router:
        # Reject invalid expert layouts before TP replaces parameters.
        # This is metadata-only; the execution stage
        # resolves again against its post-TP instance, never this snapshot.
        # Include the world, rather than only one EP group, because other
        # stages/ranks may subsequently enter shared FSDP collectives.
        with synchronize_local_errors(
            "Expert parallel global plan validation", group=error_group
        ):
            if cfg.ep_config.ep_token_dispatcher == "mc2":
                validate_mc2_world_size(cfg.ep_size)
            global_ep_plan = resolve_ep_plan(
                model,
                degree=cfg.ep_size,
                implementation=cfg.ep_config.ep_token_dispatcher,
                tp_enabled=cfg.tp_size > 1,
                quantile_balancing=cfg.ep_config.moe_quantile_balancing,
                has_ep_group=ep_group is not None,
                bindings=global_bindings,
            )
            if global_ep_plan is not None:
                global_bindings = model_bindings_from_plan(global_ep_plan)

    resolved_tp_plan = None
    if cfg.tp_size > 1 and mesh is not None:
        # Local inspection only: coordinate failures before TP accesses its
        # communication group. The executor consumes exactly these bindings.
        with synchronize_local_errors(
            "Tensor parallel plan validation", group=error_group
        ):
            resolved_tp_plan = resolve_tp_plan(model, cfg=cfg, bindings=global_bindings)
    with synchronize_local_errors("Execution layout validation", group=error_group):
        execution_layout = resolve_execution_layout(
            model,
            tp_size=cfg.tp_size,
            cp_size=cfg.cp_size,
            tp_plan=resolved_tp_plan,
            chunked_loss=chunked_loss_num_chunks > 1,
            compile_enabled=compile,
            activation_swap=activation_swap and activation_checkpoint != "full",
        )
    execution = apply_stages(
        model,
        dense_mesh=mesh,
        resolved_tp_plan=resolved_tp_plan,
        tp_mesh=tp_mesh,
        ep_group=ep_group,
        ep_error_group=ep_error_group,
        execution_layout=execution_layout,
    )
    return ParallelAssembly(execution, tuple(layouts))
