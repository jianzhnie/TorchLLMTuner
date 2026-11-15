"""Apply parallel and memory transformations to explicitly declared models.

Both dense and pipeline paths follow ``stages.STAGE_ORDER``:
TP → EP → CP → activation checkpoint → activation swap → compile → FSDP. PP
splits the model first, transforms each local part, then binds the pipeline schedule.

Both paths return ``ParallelAssembly`` with a module or ``PipelineParallelSetup``
as execution and explicit local layouts. Weight
ties are retained and handled by FSDP grouping. EP consumes model-owned MoE
contracts through ``expert_parallel/model_replacement.py``; single-block conversion
lives in ``block_conversion.py`` and preserves initialization. TP plans come
from model declarations.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist
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
from ..components.loss_contract import LossSum
from ..memory.activation_checkpoint import apply_ac, validate_ac_options
from ..memory.activation_swap import apply_activation_swap
from ..models.contracts import DecoderStructure
from .assembly import ModelPartLayout, ParallelAssembly
from .compatibility import chunked_loss_fsdp
from .context_parallel.context_parallel import apply_cp
from .context_parallel.plan import resolve_cp_plan, validate_pipeline_cp_plan
from .execution import (
    ExecutionLayout,
    resolve_execution_layout,
    validate_execution_chain,
)
from .expert_parallel.expert_parallel import apply_ep
from .expert_parallel.mc2_ops import validate_mc2_world_size
from .expert_parallel.plan import resolve_ep_plan, validate_pipeline_ep_plan
from .fully_shard_parallel.fully_shard_parallel import (
    apply_fsdp,
    validate_fsdp_environment,
)
from .fully_shard_parallel.ownership import FSDPPatternReferences
from .fully_shard_parallel.plan import resolve_pipeline_fsdp_configs
from .model_compile import apply_compile, validate_compile_environment
from .pipeline_parallel.microbatch_drivers import validate_pipeline_schedule
from .pipeline_parallel.ownership import PipelineOwnership
from .pipeline_parallel.pipeline_parallel import (
    PipelineParallelSetup,
    apply_pp,
    build_pipeline_schedule,
    validate_microbatches,
)
from .plan import (
    ParallelPlan,
    model_bindings_from_plan,
    snapshot_model_bindings,
)
from .stages import STAGE_ORDER
from .tensor_parallel.sharding import resolve_tp_overrides
from .tensor_parallel.tensor_parallel import (
    apply_tp,
    resolve_tp_plan,
    validate_pipeline_tp_plan,
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
    device: torch.device | None = None,
    compile: bool = False,
    compile_config: CompileConfig | None = None,
    activation_checkpoint: str = "none",
    activation_swap: bool = False,
    activation_swap_sync_release: bool = False,
    activation_swap_max_host_bytes: int = 0,
    selective_ac: SelectiveACConfig | None = None,
    memory_budget_ac: MemoryBudgetACConfig | None = None,
    region_ac: RegionACConfig | None = None,
    global_batch_size: int | None = None,
    chunked_loss_num_chunks: int = 1,
    loss_sum: LossSum | None = None,
    pipeline_schedule_factory: Callable[..., Any] | None = None,
    on_model_acquired: Callable[[nn.Module], None] | None = None,
) -> ParallelAssembly:
    """Transform an explicit model using parallel config and training options.

    Compilation and activation-checkpoint options apply identically to dense
    models and PP parts. Memory-budget checkpointing requires compilation;
    region checkpointing requires the optional torch_remat dependency.

    Return a ParallelAssembly with execution objects and one layout per local
    part. Its execution is a PipelineParallelSetup under PP, otherwise a module.
    PP requires ``global_batch_size`` for microbatch validation.

    ``error_group`` names every peer that must stop on local preflight failure.
    It must include all ranks participating in subsequent transforms. Without
    an explicit group, existing callers retain default-group coordination;
    CP preflight uses the topology world when available.

    ``on_model_acquired`` retains cleanup ownership of each local PP part and
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
    # Inspect only metadata on the original model, before any TP/EP/PP
    # replacement. Local stages bind again after PP establishes ownership.
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
        pipeline_ownership: PipelineOwnership | None = None,
    ) -> nn.Module:
        """Bind stage options once, then transform the model in declared order."""
        fsdp_cfg = cfg if part_cfg is None else part_cfg
        layout = ModelPartLayout(
            pipeline_ownership=(
                None
                if pipeline_ownership is None
                else pipeline_ownership.parameter_ownership()
            ),
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

    # EP groups come from the sparse mesh in both assembly paths. Under PP
    # this slice contains only ranks in the current pipeline stage.
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
    # Include PP peers: a stage-local CP group cannot stop the other pipeline
    # stages before they enter FSDP/schedule setup. Validate the original model
    # before TP/EP transforms or PP partitioning, not each transformed part.
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
            # after PP/TP/EP establish the actual local instance.
            global_cp_plan = resolve_cp_plan(model, cfg=cfg)
            if global_cp_plan is not None:
                global_bindings = model_bindings_from_plan(global_cp_plan)

    global_ep_plan = None
    if cfg.ep_size > 1 or cfg.ep_config.fixed_router:
        # Reject invalid expert layouts before TP replaces parameters or PP
        # copies/splits modules. This is metadata-only; the execution stage
        # resolves again against its post-TP/PP instance, never this snapshot.
        # Include the world, rather than only one EP group, because other
        # stages/ranks may subsequently enter shared FSDP or PP collectives.
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

    if parallel_dims is not None and parallel_dims.pp_enabled:
        global_tp_plan = None
        # All PP stages must stop before splitting if any rank's global plan
        # is invalid. This body only inspects local metadata.
        with synchronize_local_errors(
            "Pipeline global plan validation", group=error_group
        ):
            if global_batch_size is None:
                raise ValueError(
                    "pp > 1 needs global_batch_size for microbatch validation; "
                    "the trainer passes cfg.training.global_batch_size."
                )
            if cfg.tp_size > 1:
                resolve_tp_overrides(model, cfg=cfg)
                global_tp_plan = resolve_tp_plan(
                    model, cfg=cfg, bindings=global_bindings
                )
            validate_microbatches(parallel_dims, cfg, global_batch_size)
            global_execution_layout = resolve_execution_layout(
                model,
                tp_size=cfg.tp_size,
                cp_size=cfg.cp_size,
                pipeline=True,
                tp_plan=global_tp_plan,
                chunked_loss=chunked_loss_num_chunks > 1,
                compile_enabled=compile,
                activation_swap=activation_swap and activation_checkpoint != "full",
            )
        pipeline_ownership_plan = (
            global_tp_plan
            or global_cp_plan
            or global_ep_plan
            or ParallelPlan(
                stage="pp",
                targets=(),
                degree=cfg.pp_size,
                implementation="deepcopy",
                **snapshot_model_bindings(model),
            )
        )
        # Split first; use this stage's dense mesh for all local transforms.
        stages, model_parts, has_first_stage, has_last_stage = apply_pp(
            model,
            parallel_dims=parallel_dims,
            cfg=cfg,
            device=device if device is not None else next(model.parameters()).device,
            global_batch_size=global_batch_size,
        )
        if on_model_acquired is not None:
            for part in model_parts:
                on_model_acquired(part)
        dense_mesh = parallel_dims.spmd_dense_mesh()
        # Stage construction deep-copies the model. Resolve fresh local bindings;
        # references from the global preflight deliberately cannot cross this boundary.
        resolved_tp_plans = [None] * len(model_parts)
        if global_tp_plan is not None:
            with synchronize_local_errors(
                "Pipeline local TP plan validation", group=error_group
            ):
                resolved_tp_plans = [
                    resolve_tp_plan(part, cfg=cfg) for part in model_parts
                ]
                for part, local_plan in zip(
                    model_parts, resolved_tp_plans, strict=True
                ):
                    validate_pipeline_tp_plan(global_tp_plan, local_plan, part)
        local_ownership = []
        with synchronize_local_errors(
            "Pipeline local ownership validation", group=error_group
        ):
            for part, local_tp_plan in zip(model_parts, resolved_tp_plans, strict=True):
                local_ownership_plan = local_tp_plan or ParallelPlan(
                    stage="pp",
                    targets=(),
                    degree=cfg.pp_size,
                    implementation="deepcopy",
                    **snapshot_model_bindings(part),
                )
                ownership = PipelineOwnership(
                    pipeline_ownership_plan, local_ownership_plan
                )
                ownership.validate_aliases()
                local_ownership.append(ownership)
        if global_cp_plan is not None:
            with synchronize_local_errors(
                "Pipeline local CP plan validation", group=error_group
            ):
                for part, ownership in zip(model_parts, local_ownership, strict=True):
                    local_cp_plan = resolve_cp_plan(
                        part,
                        cfg=cfg,
                        bindings=model_bindings_from_plan(ownership.local_plan),
                    )
                    validate_pipeline_cp_plan(global_cp_plan, local_cp_plan, part)
        if global_ep_plan is not None:
            # PP stage configuration can change declarations without replacing
            # their tensors. Check every local part before any TP/EP transform.
            with synchronize_local_errors(
                "Pipeline local EP plan validation", group=error_group
            ):
                for part, ownership in zip(model_parts, local_ownership, strict=True):
                    local_ep_plan = resolve_ep_plan(
                        part,
                        degree=cfg.ep_size,
                        implementation=cfg.ep_config.ep_token_dispatcher,
                        tp_enabled=cfg.tp_size > 1,
                        quantile_balancing=cfg.ep_config.moe_quantile_balancing,
                        has_ep_group=ep_group is not None,
                        bindings=model_bindings_from_plan(ownership.local_plan),
                    )
                    validate_pipeline_ep_plan(global_ep_plan, local_ep_plan, part)
        # Part ownership is rank-local, but failure must stop all stages
        # before TP/EP/FSDP begin communicating on the resulting parts.
        with synchronize_local_errors("Pipeline FSDP plan binding", group=error_group):
            fsdp_configs = (
                resolve_pipeline_fsdp_configs(
                    model, model_parts, cfg, ownership=local_ownership
                )
                if cfg.fsdp_config.fsdp_apply_modules
                else [cfg] * len(model_parts)
            )
        with synchronize_local_errors(
            "Pipeline execution layout validation", group=error_group
        ):
            execution_layouts = [
                resolve_execution_layout(
                    part,
                    tp_size=cfg.tp_size,
                    cp_size=cfg.cp_size,
                    pipeline=True,
                    tp_plan=tp_plan,
                    chunked_loss=chunked_loss_num_chunks > 1,
                    compile_enabled=compile,
                    activation_swap=activation_swap and activation_checkpoint != "full",
                )
                for part, tp_plan in zip(model_parts, resolved_tp_plans, strict=True)
            ]
        # Interleaved schedules own nonadjacent global stages on each rank.
        # Gather declarations on the PP axis and validate the actual chain
        # before any parameter transforms or schedule resources are created.
        pp_mesh = parallel_dims.get_mesh("pp")
        with synchronize_local_errors(
            "Pipeline boundary metadata preparation", group=error_group
        ):
            local_boundaries = [
                (stage.stage_index, layout)
                for stage, layout in zip(stages, execution_layouts, strict=True)
            ]
        gathered_boundaries = [None] * pp_mesh.size()
        dist.all_gather_object(
            gathered_boundaries, local_boundaries, group=pp_mesh.get_group()
        )
        with synchronize_local_errors(
            "Pipeline execution chain validation", group=error_group
        ):
            boundaries = sorted(
                (
                    entry
                    for rank_entries in gathered_boundaries
                    for entry in rank_entries
                ),
                key=lambda entry: entry[0],
            )
            indices = [index for index, _ in boundaries]
            if indices != list(range(len(boundaries))):
                raise ValueError("Pipeline stage indices must be unique and contiguous")
            validate_execution_chain(tuple(layout for _, layout in boundaries))
        for i, part in enumerate(model_parts):
            model_parts[i] = apply_stages(
                part,
                dense_mesh=dense_mesh,
                tp_mesh=tp_mesh,
                ep_group=ep_group,
                ep_error_group=ep_error_group,
                execution_layout=execution_layouts[i],
                part_cfg=fsdp_configs[i],
                resolved_tp_plan=resolved_tp_plans[i],
                pipeline_ownership=local_ownership[i],
            )
            # Rebind the stage's submodule in case a transform replaced the chunk.
            stages[i].submod = model_parts[i]
        schedule_factory = (
            build_pipeline_schedule
            if pipeline_schedule_factory is None
            else pipeline_schedule_factory
        )
        execution = PipelineParallelSetup(
            # The last stage's loss needs the same TP vocabulary as training.
            schedule=schedule_factory(
                stages,
                cfg=cfg,
                loss_sum=loss_sum,
                logits_width=(
                    None
                    if global_execution_layout.vocabulary_sharded
                    else global_execution_layout.vocabulary_size
                ),
                **global_execution_layout.loss_kwargs(
                    None if tp_mesh is None else tp_mesh.get_group()
                ),
            ),
            stages=stages,
            model_parts=model_parts,
            has_first_stage=has_first_stage,
            has_last_stage=has_last_stage,
        )

        # Constructors may communicate. Validate only after they return, then
        # propagate local contract failures before any rank starts training.
        with synchronize_local_errors(
            "Pipeline schedule contract validation", group=error_group
        ):
            validate_pipeline_schedule(execution.schedule, stages)
        return ParallelAssembly(execution, tuple(layouts))

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
            pipeline=False,
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
