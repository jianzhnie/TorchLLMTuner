"""Pure, framework-independent parallel plan resolution."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import ParallelConfig

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

    def validate_binding(self, model: "Qwen3MoeModel") -> None:
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
    )
