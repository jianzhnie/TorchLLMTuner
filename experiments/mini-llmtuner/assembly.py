"""Resolve first, then apply TP → EP → native FSDP2 before building optimizer."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from .config import ParallelConfig
from .ownership import ParameterMapping, bind_parameter_mappings
from .parallel import ExpertDispatcher, ParallelRuntime
from .resolver import ExecutionPlan, resolve_plan


@dataclass(slots=True)
class AssemblyResult:
    model: torch.nn.Module
    execution: ExecutionPlan
    runtime: ParallelRuntime = field(default_factory=ParallelRuntime)

    @property
    def parameter_mappings(self) -> tuple[ParameterMapping, ...]:
        """Expose provenance and local ownership after all parameter transforms."""
        return self.runtime.parameter_mappings

    @property
    def old_to_new_parameters(self) -> Mapping[int, torch.nn.Parameter]:
        """Original object IDs remain usable after full-size sources are freed."""
        return MappingProxyType(
            {entry.source.source_id: entry.target for entry in self.parameter_mappings}
        )


class LocalParallelAdapter:
    """Single-process execution with no process-group creation."""

    def __init__(self) -> None:
        self.runtime = ParallelRuntime()

    def apply(self, model: torch.nn.Module, plan: ExecutionPlan) -> torch.nn.Module:
        return model


class PyTorchParallelAdapter(LocalParallelAdapter):
    """Explicit meshes, model-owned TP math, generic EP transport, native FSDP2."""

    def __init__(self, config: ParallelConfig) -> None:
        super().__init__()
        self.config = config

    def apply(self, model: torch.nn.Module, plan: ExecutionPlan) -> torch.nn.Module:
        if not dist.is_initialized():
            raise RuntimeError(
                "Parallel execution requires an initialized process group"
            )
        tp, ep, fsdp = plan.degrees
        device = next(model.parameters()).device.type
        if fsdp > 1 and device == "cpu":
            raise ValueError("This prototype requires NPU/CUDA for native FSDP2")
        mesh = init_device_mesh(
            device, (fsdp, ep, tp), mesh_dim_names=("fsdp", "ep", "tp")
        )
        data_mesh = mesh["fsdp", "ep"]._flatten(mesh_dim_name="data")
        runtime = ParallelRuntime(
            tp_group=mesh["tp"].get_group(),
            ep_group=mesh["ep"].get_group(),
            data_group=data_mesh.get_group(),
            norm_group=dist.group.WORLD,
            tp_size=tp,
            ep_size=ep,
            fsdp_size=fsdp,
            data_rank=data_mesh.get_local_rank(),
        )
        self.runtime = runtime
        if tp > 1:
            model.apply_tensor_parallel(runtime, plan.tp_rules)
        if ep > 1:
            for target in plan.targets:
                if target.role != "moe":
                    continue
                block = model.get_submodule(target.path)
                block.dispatcher = ExpertDispatcher(block.experts, runtime.ep_group)
        if fsdp > 1:
            for layer in model.layers:
                fully_shard(layer, mesh=mesh["fsdp"])
            fully_shard(model, mesh=mesh["fsdp"])
        return model


def assemble_model(
    model: torch.nn.Module,
    config: ParallelConfig,
    *,
    world_size: int,
    adapter: LocalParallelAdapter | None = None,
) -> AssemblyResult:
    error = None
    execution = None
    try:
        execution = resolve_plan(model, config, world_size=world_size)
        execution.validate_binding(model)
    except Exception as caught:
        error = caught
    if world_size > 1 and dist.is_initialized():
        # Coordinate only after pure local parsing; no collective is hidden in
        # the resolver. A bad local model must not leave peers building meshes.
        failed = torch.tensor(
            int(error is not None), device=next(model.parameters()).device
        )
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if int(failed):
            if error is not None:
                raise error
            raise RuntimeError("Parallel preflight failed on another rank")
    elif error is not None:
        raise error
    selected = adapter or (
        PyTorchParallelAdapter(config) if world_size > 1 else LocalParallelAdapter()
    )
    assembled = selected.apply(model, execution)
    runtime = selected.runtime
    runtime.parameter_mappings = bind_parameter_mappings(
        assembled,
        execution.parameter_specs,
        tp=runtime.tp_size,
        ep=runtime.ep_size,
        fsdp=runtime.fsdp_size,
        tp_rank=dist.get_rank(runtime.tp_group) if runtime.tp_size > 1 else 0,
        ep_rank=dist.get_rank(runtime.ep_group) if runtime.ep_size > 1 else 0,
    )
    return AssemblyResult(assembled, execution, runtime)
