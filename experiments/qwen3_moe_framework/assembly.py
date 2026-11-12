"""Apply a resolved plan through a small, replaceable execution adapter."""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from .config import ParallelConfig
from .resolver import ExecutionPlan, resolve_plan


@dataclass(slots=True)
class AssemblyResult:
    model: torch.nn.Module
    execution: ExecutionPlan


class LocalParallelAdapter:
    """Single-process adapter; distributed backends can implement this API."""

    def apply(self, model: torch.nn.Module, plan: ExecutionPlan) -> torch.nn.Module:
        del plan
        return model


class PyTorchParallelAdapter(LocalParallelAdapter):
    """Native PyTorch 2.1+ adapter for the prototype's declared plan.

    TP and FSDP2 are applied directly to the model.  PP/CP/EP require model
    specific schedules and are represented by explicit hooks rather than
    silently changing tensor semantics.
    """

    def __init__(self, config: ParallelConfig) -> None:
        self.config = config

    def apply(self, model: torch.nn.Module, plan: ExecutionPlan) -> torch.nn.Module:
        if not dist.is_initialized():
            raise RuntimeError(
                "distributed process group is required for parallel execution"
            )
        if self.config.cp_size > 1:
            raise NotImplementedError("CP needs a model attention/context schedule")
        if self.config.pp_size > 1:
            raise NotImplementedError(
                "PP needs a pipeline schedule and stage input contract"
            )
        mesh = self._mesh(model)
        if self.config.tp_size > 1:
            model = self._apply_tp(model, mesh["tp"])
        if self.config.ep_size > 1:
            model = self._apply_ep(model, mesh)
        if self.config.data_parallel_shard_size > 1:
            from torch.distributed.fsdp import fully_shard

            for layer in model.layers:
                fully_shard(layer, mesh=mesh["fsdp"])
            fully_shard(model, mesh=mesh["fsdp"])
        return model

    def _mesh(self, model: torch.nn.Module):
        from torch.distributed.device_mesh import init_device_mesh

        device = next(model.parameters()).device.type
        tp = self.config.tp_size
        fsdp = self.config.data_parallel_shard_size
        if tp * fsdp != dist.get_world_size():
            raise ValueError("TP x FSDP degree must equal distributed world size")
        return init_device_mesh(device, (tp, fsdp), mesh_dim_names=("tp", "fsdp"))

    def _apply_tp(self, model: torch.nn.Module, mesh) -> torch.nn.Module:
        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            RowwiseParallel,
            parallelize_module,
        )

        plan = {}
        for path, layout in model.contract.tp_plan.items():
            if layout == "packed_colwise":
                raise NotImplementedError(
                    "packed expert TP needs a grouped-matmul parameter sharder"
                )
            if layout in {"colwise", "packed_colwise"}:
                plan[path] = ColwiseParallel()
            elif layout == "rowwise":
                plan[path] = RowwiseParallel()
        return parallelize_module(model, mesh, plan)

    def _apply_ep(self, model: torch.nn.Module, mesh) -> torch.nn.Module:
        # Expert routing is model-specific; expose an explicit hook contract.
        hook = getattr(model, "apply_expert_parallel", None)
        if hook is None:
            raise NotImplementedError(
                "model must implement apply_expert_parallel(mesh)"
            )
        hook(mesh)
        return model


def assemble_model(
    model: torch.nn.Module,
    config: ParallelConfig,
    *,
    world_size: int,
    adapter: LocalParallelAdapter | None = None,
) -> AssemblyResult:
    execution = resolve_plan(model, config, world_size=world_size)
    execution.validate_binding(model)
    selected = adapter or (
        PyTorchParallelAdapter(config) if world_size > 1 else LocalParallelAdapter()
    )
    return AssemblyResult(selected.apply(model, execution), execution)
