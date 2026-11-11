"""Pure preflight: resolve explicit targets and ownership before mutation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.parallel.plan import ParallelPlan, PlanTarget
from llmtuner.parallel.topology import ParallelTopology

if TYPE_CHECKING:
    from .model import Qwen3MoeModel

ExecutionPlan = ParallelPlan
ResolvedTarget = PlanTarget


def resolve_plan(
    model: Qwen3MoeModel, config: ParallelConfig, *, world_size: int
) -> ExecutionPlan:
    """Validate a bounded experiment feature set without touching parameters."""
    topology = ParallelTopology.from_config(config, world_size)
    if config.cp_size != 1 or config.pp_size != 1:
        raise ValueError("This experiment implements TP/EP/FSDP only")
    if (
        config.tp_config.tp_colwise_modules
        or config.tp_config.tp_rowwise_modules
        or config.tp_config.tp_sequence_modules
    ):
        raise ValueError(
            "Experiment plans do not accept unresolved TP pattern overrides"
        )
    if config.fsdp_config.fsdp_apply_modules:
        raise ValueError("Experiment uses declared decoder units, not FSDP patterns")
    if config.ep_config.fixed_router or config.ep_config.moe_quantile_balancing:
        raise ValueError("Experiment requires the model's native top-k routing")
    if config.ep_config.ep_token_dispatcher != "alltoall":
        raise ValueError("Experiment currently validates only alltoall EP dispatch")
    model.validate_parallel(tp_degree=config.tp_size, ep_degree=config.ep_size)
    targets = tuple(
        PlanTarget(path, role, model.get_submodule(path), layout)
        for path, role, layout in model.contract.targets
    )
    aliases: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    shared = tuple(tuple(names) for names in aliases.values() if len(names) > 1)
    if shared and config.tp_size > 1:
        raise ValueError(
            "Tied weights with TP require a separate experiment acceptance"
        )
    operations = tuple(
        name
        for name, enabled in (
            ("tp", config.tp_size > 1),
            ("ep", config.ep_size > 1),
            ("fsdp", topology.dp_shard > 1 or topology.dp_replicate > 1),
        )
        if enabled
    )
    return ExecutionPlan(
        stage="tp,ep,fsdp",
        targets=targets,
        degree=config.tp_size,
        implementation="native",
        degrees=(
            config.tp_size,
            config.ep_size,
            topology.dp_shard,
            config.data_parallel_replicate_size,
        ),
        shared_parameters=shared,
        operations=operations,
        tp_plan=tuple(model.contract.tp_plan.items()),
    )
