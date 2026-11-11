"""Resolve once, then execute TP -> EP -> FSDP through existing engines."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch.distributed as dist

from llmtuner.accelerator.collectives import synchronize_local_errors
from llmtuner.accelerator.device import device_type
from llmtuner.config import ParallelConfig
from llmtuner.parallel.expert_parallel import apply_ep
from llmtuner.parallel.fully_shard_parallel import apply_fsdp
from llmtuner.parallel.tensor_parallel import apply_tp
from llmtuner.parallel.topology import ParallelTopology, build_device_mesh

from .resolver import ExecutionPlan, resolve_plan

if TYPE_CHECKING:
    from .model import Qwen3MoeModel


@dataclass(slots=True)
class AssemblyResult:
    model: Qwen3MoeModel
    topology: ParallelTopology
    execution: ExecutionPlan


def assemble_model(
    model: Qwen3MoeModel,
    config: ParallelConfig,
    *,
    world_size: int,
) -> AssemblyResult:
    """Apply the supported experiment plan without model-name branches."""
    with synchronize_local_errors("Experiment plan preflight"):
        execution = resolve_plan(model, config, world_size=world_size)
        execution.validate_binding(model)
        actual_device = next(model.parameters()).device.type
        if world_size > 1 and actual_device != device_type:
            raise ValueError(
                f"Distributed experiment requires mesh device {device_type}, "
                f"got {actual_device}; use a matching environment"
            )
    topology = ParallelTopology.from_config(config, world_size)
    if world_size == 1:
        return AssemblyResult(model, topology, execution)
    config = replace(config, data_parallel_shard_size=topology.dp_shard)
    mesh = build_device_mesh(topology)
    model = apply_tp(model, mesh, config, plan=dict(execution.tp_plan))
    ep_mesh = topology.get_optional_mesh("ep") if world_size > 1 else None
    model = apply_ep(
        model,
        config,
        ep_group=None if ep_mesh is None else ep_mesh.get_group(),
        error_group=(
            dist.group.WORLD if config.ep_size > 1 and dist.is_initialized() else None
        ),
    )
    model = apply_fsdp(model, config, topology if world_size > 1 else None)
    return AssemblyResult(model=model, topology=topology, execution=execution)
