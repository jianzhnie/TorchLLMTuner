"""Pure, framework-independent parallel plan resolution."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import ParallelConfig
from .ownership import ParameterSpec, resolve_parameter_specs

if TYPE_CHECKING:
    from .model import Qwen3MoeModel


@dataclass(frozen=True, slots=True)
class PlanTarget:
    path: str
    role: str
    layout: str
    module_id: int


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    operations: tuple[str, ...]
    targets: tuple[PlanTarget, ...]
    degrees: tuple[int, ...]
    tp_rules: tuple[tuple[str, str], ...]
    parameters: tuple[tuple[str, int, tuple[int, ...]], ...]
    parameter_specs: tuple[ParameterSpec, ...]

    def validate_binding(self, model: "Qwen3MoeModel") -> None:
        current = tuple(
            (name, id(value), tuple(value.shape))
            for name, value in model.named_parameters(remove_duplicate=False)
        )
        if current != self.parameters:
            raise ValueError("Stale plan parameter binding or geometry")
        for spec in self.parameter_specs:
            if (
                spec.source() is None
                or spec.source().requires_grad != spec.requires_grad
            ):
                raise ValueError("Stale plan parameter trainability")
        for target in self.targets:
            try:
                module = model.get_submodule(target.path)
            except AttributeError as error:
                raise ValueError(f"Stale plan binding: {target.path}") from error
            if id(module) != target.module_id:
                raise ValueError(f"Stale plan binding: {target.path}")


ResolvedTarget = PlanTarget


def resolve_plan(
    model: "Qwen3MoeModel", config: ParallelConfig, *, world_size: int
) -> ExecutionPlan:
    """Validate geometry without mutating model parameters."""
    if config.recompute or config.quantization is not None:
        raise NotImplementedError("Recompute/quantization execution is not implemented")
    if config.cp_size != 1 or config.pp_size != 1:
        raise NotImplementedError("CP/PP schedules are not implemented")
    if config.tp_size < 1 or config.ep_size < 1 or config.data_parallel_shard_size < 1:
        raise ValueError("parallel degrees must be positive")
    if world_size == 1 and any(
        degree > 1
        for degree in (
            config.tp_size,
            config.ep_size,
            config.data_parallel_shard_size,
            config.cp_size,
            config.pp_size,
        )
    ):
        raise ValueError(
            "parallel degree > 1 requires torchrun with matching world size"
        )
    if world_size != config.tp_size * config.ep_size * config.data_parallel_shard_size:
        if world_size != 1:
            raise ValueError("world size must equal TP x EP x FSDP degrees")
    model.validate_parallel(tp_degree=config.tp_size, ep_degree=config.ep_size)
    targets = tuple(
        PlanTarget(path, role, layout, id(model.get_submodule(path)))
        for path, role, layout in model.contract.targets
    )
    operations = tuple(
        name
        for name, enabled in (
            ("tp", config.tp_size > 1),
            ("ep", config.ep_size > 1),
            ("fsdp", config.data_parallel_shard_size > 1),
            ("cp", config.cp_size > 1),
            ("pp", config.pp_size > 1),
        )
        if enabled
    )
    return ExecutionPlan(
        operations,
        targets,
        (config.tp_size, config.ep_size, config.data_parallel_shard_size),
        tuple(model.contract.tp_plan.items()),
        tuple(
            (name, id(value), tuple(value.shape))
            for name, value in model.named_parameters(remove_duplicate=False)
        ),
        resolve_parameter_specs(
            model,
            model.contract.tp_plan.items(),
            model.contract.targets,
            tp=config.tp_size,
            ep=config.ep_size,
            fsdp=config.data_parallel_shard_size,
            gradient_sums=getattr(model.contract, "tp_gradient_sums", ()),
        ),
    )
