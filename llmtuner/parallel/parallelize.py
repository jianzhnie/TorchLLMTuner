"""Apply parallel and memory transformations to explicitly declared models.

Both dense and pipeline paths follow ``stages.STAGE_ORDER``:
TP → EP → CP → activation checkpoint → activation swap → compile → FSDP. PP
splits the model first, transforms each local part, then binds the pipeline schedule.

Dense runs return a module; PP runs return ``PipelineParallelSetup``. Weight
ties are retained and handled by FSDP grouping. EP consumes model-owned MoE
contracts through ``expert_parallel/model_replacement.py``; single-block conversion
lives in ``block_conversion.py`` and preserves initialization. TP plans come
from model declarations.
"""

from __future__ import annotations

from contextlib import nullcontext
from functools import partial

import torch
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
from .compatibility import chunked_loss_fsdp
from .context_parallel.context_parallel import apply_cp
from .context_parallel.plan import resolve_cp_plan, validate_pipeline_cp_plan
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
from .pipeline_parallel.pipeline_parallel import (
    PipelineParallelSetup,
    apply_pp,
    build_pipeline_schedule,
    validate_microbatches,
)
from .plan import ParallelPlan
from .stages import STAGE_ORDER
from .tensor_parallel.sharding import resolve_tp_overrides
from .tensor_parallel.tensor_parallel import (
    apply_tp,
    resolve_tp_plan,
    validate_pipeline_tp_plan,
)
from .topology import ParallelTopology

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
) -> nn.Module | PipelineParallelSetup:
    """Transform an explicit model using parallel config and training options.

    Compilation and activation-checkpoint options apply identically to dense
    models and PP parts. Memory-budget checkpointing requires compilation;
    region checkpointing requires the optional torch_remat dependency.

    PP requires ``global_batch_size`` for microbatch validation and returns a
    ``PipelineParallelSetup`` containing local parts, stages, and schedule.
    Other runs return the transformed module.

    ``error_group`` names every peer that must stop on local preflight failure.
    It must include all ranks participating in subsequent transforms. Without
    an explicit group, existing callers retain default-group coordination;
    CP preflight uses the topology world when available.
    """

    with (
        synchronize_local_errors("Parallel option validation", group=error_group)
        if error_group is not None
        else nullcontext()
    ):
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

    def apply_stages(
        part: nn.Module,
        *,
        dense_mesh: DeviceMesh | None,
        part_cfg: ParallelConfig | None = None,
        resolved_tp_plan: ParallelPlan | None = None,
    ) -> nn.Module:
        """Bind stage options once, then transform the model in declared order."""
        fsdp_cfg = cfg if part_cfg is None else part_cfg
        references = (
            FSDPPatternReferences(part, fsdp_cfg)
            if fsdp_cfg.fsdp_config.fsdp_ignored_modules
            or any(
                options.get("hook_module") is not None or options.get("ignored_params")
                for options in fsdp_cfg.fsdp_config.fsdp_apply_modules.values()
            )
            else None
        )
        tp_options = {}
        if resolved_tp_plan is not None:
            tp_options["resolved_plan"] = resolved_tp_plan
        if references is not None:
            tp_options["on_replacement"] = references.record_replacements
        runners = {
            "tp": partial(
                apply_tp,
                mesh=dense_mesh,
                cfg=cfg,
                **tp_options,
            ),
            "ep": partial(
                apply_ep, cfg=cfg, ep_group=ep_group, error_group=ep_error_group
            ),
            "cp": partial(apply_cp, mesh=dense_mesh, cfg=cfg),
            "ac": partial(
                apply_ac,
                mode=activation_checkpoint,
                selective=selective_ac,
                memory_budget=memory_budget_ac,
                region=region_ac,
                compile_enabled=compile,
            ),
            "activation_swap": partial(
                apply_activation_swap,
                sync_release=activation_swap_sync_release,
                max_host_bytes=activation_swap_max_host_bytes,
            ),
            "compile": partial(
                apply_compile, compile_config=compile_config, tp_mesh=tp_mesh
            ),
            "fsdp": partial(
                apply_fsdp,
                cfg=fsdp_cfg,
                parallel_dims=parallel_dims,
                chunked_loss_num_chunks=chunked_loss_num_chunks,
            ),
        }
        activation_swap_enabled = activation_swap and activation_checkpoint != "full"
        for name in STAGE_ORDER:
            if name == "compile" and not compile:
                continue
            if name == "activation_swap" and not activation_swap_enabled:
                continue
            if name == "ep" and references is not None:
                # TP may already have replaced Parameters before EP's
                # explicit split/rename mapping is emitted.
                references = FSDPPatternReferences(part, references.bind(part))
                part = runners[name](
                    part, on_replacement=references.record_replacements
                )
            elif name == "fsdp" and references is not None:
                part = runners[name](part, cfg=references.bind(part))
            else:
                part = runners[name](part)
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
            )

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
                global_tp_plan = resolve_tp_plan(model, cfg=cfg)
            validate_microbatches(parallel_dims, cfg, global_batch_size)
        # Split first; use this stage's dense mesh for all local transforms.
        stages, model_parts, has_first_stage, has_last_stage = apply_pp(
            model,
            parallel_dims=parallel_dims,
            cfg=cfg,
            device=device if device is not None else next(model.parameters()).device,
            global_batch_size=global_batch_size,
        )
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
        if global_cp_plan is not None:
            with synchronize_local_errors(
                "Pipeline local CP plan validation", group=error_group
            ):
                for part in model_parts:
                    local_cp_plan = resolve_cp_plan(part, cfg=cfg)
                    validate_pipeline_cp_plan(global_cp_plan, local_cp_plan, part)
        if global_ep_plan is not None:
            # PP stage configuration can change declarations without replacing
            # their tensors. Check every local part before any TP/EP transform.
            with synchronize_local_errors(
                "Pipeline local EP plan validation", group=error_group
            ):
                for part in model_parts:
                    local_ep_plan = resolve_ep_plan(
                        part,
                        degree=cfg.ep_size,
                        implementation=cfg.ep_config.ep_token_dispatcher,
                        tp_enabled=cfg.tp_size > 1,
                        quantile_balancing=cfg.ep_config.moe_quantile_balancing,
                        has_ep_group=ep_group is not None,
                    )
                    validate_pipeline_ep_plan(global_ep_plan, local_ep_plan, part)
        # Part ownership is rank-local, but failure must stop all stages
        # before TP/EP/FSDP begin communicating on the resulting parts.
        with synchronize_local_errors("Pipeline FSDP plan binding", group=error_group):
            fsdp_configs = (
                resolve_pipeline_fsdp_configs(model, model_parts, cfg)
                if cfg.fsdp_config.fsdp_apply_modules
                else [cfg] * len(model_parts)
            )
        for i, part in enumerate(model_parts):
            model_parts[i] = apply_stages(
                part,
                dense_mesh=dense_mesh,
                part_cfg=fsdp_configs[i],
                resolved_tp_plan=resolved_tp_plans[i],
            )
            # Rebind the stage's submodule in case a transform replaced the chunk.
            stages[i].submod = model_parts[i]
        return PipelineParallelSetup(
            # The last stage's loss needs the same TP vocabulary as training.
            schedule=build_pipeline_schedule(
                stages,
                cfg=cfg,
                tp_group=None if tp_mesh is None else tp_mesh.get_group(),
                global_vocab_size=(
                    None if tp_mesh is None else getattr(model, "vocab_size", None)
                ),
            ),
            stages=stages,
            model_parts=model_parts,
            has_first_stage=has_first_stage,
            has_last_stage=has_last_stage,
        )

    resolved_tp_plan = None
    if cfg.tp_size > 1 and mesh is not None:
        # Local inspection only: coordinate failures before TP accesses its
        # communication group. The executor consumes exactly these bindings.
        with synchronize_local_errors(
            "Tensor parallel plan validation", group=error_group
        ):
            resolved_tp_plan = resolve_tp_plan(model, cfg=cfg)
    return apply_stages(model, dense_mesh=mesh, resolved_tp_plan=resolved_tp_plan)
