"""Apply parallel and memory transformations to Hugging Face models.

Both dense and pipeline paths follow ``stages.STAGE_ORDER``:
TP → EP → CP → activation checkpoint → compile → FSDP. PP splits the model
first, transforms each local part, then binds the pipeline schedule.

Dense runs return a module; PP runs return ``PipelineParallelSetup``. Weight
ties are retained and handled by FSDP grouping. EP replaces supported HF MoE
blocks through ``expert_parallel/model_replacement.py``; single-block conversion
lives in ``block_conversion.py`` and preserves initialization. TP plans come
from the model registry.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from llmtuner.config.activation_checkpoint_config import (
    MemoryBudgetACConfig,
    RegionACConfig,
    SelectiveACConfig,
)
from llmtuner.config.compile_config import CompileConfig
from llmtuner.config.parallel_config import ParallelConfig

from ..memory.activation_checkpoint import apply_ac
from .context_parallel.context_parallel import apply_cp
from .expert_parallel.expert_parallel import apply_ep
from .fully_shard_parallel.fully_shard_parallel import apply_fsdp
from .model_compile import apply_compile
from .pipeline_parallel.pipeline_parallel import (
    PipelineParallelSetup,
    apply_pp,
    build_pipeline_schedule,
)
from .stages import STAGE_ORDER
from .tensor_parallel.tensor_parallel import apply_tp
from .topology import ParallelTopology

__all__ = ["parallelize_hf_transformers"]


def parallelize_hf_transformers(
    model: nn.Module,
    *,
    cfg: ParallelConfig,
    mesh: DeviceMesh | None,
    parallel_dims: ParallelTopology | None,
    device: torch.device | None = None,
    compile: bool = False,
    compile_config: CompileConfig | None = None,
    activation_checkpoint: str = "none",
    selective_ac: SelectiveACConfig | None = None,
    memory_budget_ac: MemoryBudgetACConfig | None = None,
    region_ac: RegionACConfig | None = None,
    global_batch_size: int | None = None,
) -> nn.Module | PipelineParallelSetup:
    """Transform a model using parallel config and explicit training options.

    Compilation and activation-checkpoint options apply identically to dense
    models and PP parts. Memory-budget checkpointing requires compilation;
    region checkpointing requires the optional torch_remat dependency.

    PP requires ``global_batch_size`` for microbatch validation and returns a
    ``PipelineParallelSetup`` containing local parts, stages, and schedule.
    Other runs return the transformed module.
    """

    def apply_stages(
        part: nn.Module,
        *,
        dense_mesh: DeviceMesh | None,
    ) -> nn.Module:
        """Bind stage options once, then transform the model in declared order."""
        runners = {
            "tp": partial(apply_tp, mesh=dense_mesh, cfg=cfg),
            "ep": partial(apply_ep, cfg=cfg, ep_group=ep_group),
            "cp": partial(
                apply_cp, mesh=dense_mesh, cfg=cfg, parallel_dims=parallel_dims
            ),
            "ac": partial(
                apply_ac,
                mode=activation_checkpoint,
                selective=selective_ac,
                memory_budget=memory_budget_ac,
                region=region_ac,
                compile_enabled=compile,
            ),
            "compile": partial(
                apply_compile, compile_config=compile_config, tp_mesh=tp_mesh
            ),
            "fsdp": partial(apply_fsdp, cfg=cfg, parallel_dims=parallel_dims),
        }
        for name in STAGE_ORDER:
            if name != "compile" or compile:
                part = runners[name](part)
        return part

    # EP groups come from the sparse mesh in both assembly paths. Under PP
    # this slice contains only ranks in the current pipeline stage.
    ep_group = None
    if cfg.ep > 1:
        if parallel_dims is None:
            raise ValueError(
                f"ep={cfg.ep} needs a process group, but this run is "
                "single-process (parallel_dims is None). EP requires "
                "world_size > 1."
            )
        ep_mesh = parallel_dims.get_optional_mesh("ep")
        if ep_mesh is None:
            raise ValueError(
                f"ep={cfg.ep} but parallel_dims has no multi-rank 'ep' axis."
            )
        ep_group = ep_mesh.get_group()
    tp_mesh = None if parallel_dims is None else parallel_dims.get_optional_mesh("tp")

    if parallel_dims is not None and parallel_dims.pp_enabled:
        if global_batch_size is None:
            raise ValueError(
                "pp > 1 needs global_batch_size for microbatch validation; "
                "the trainer passes cfg.training.global_batch_size."
            )
        # Split first; use this stage's dense mesh for all local transforms.
        stages, model_parts, has_first_stage, has_last_stage = apply_pp(
            model,
            parallel_dims=parallel_dims,
            cfg=cfg,
            device=device if device is not None else next(model.parameters()).device,
            global_batch_size=global_batch_size,
        )
        dense_mesh = parallel_dims.spmd_dense_mesh()
        for i, part in enumerate(model_parts):
            model_parts[i] = apply_stages(part, dense_mesh=dense_mesh)
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

    return apply_stages(model, dense_mesh=mesh)
