"""Pipeline-aware optimizer assembly over shared update and checkpoint logic.

This constructor coordinates parameter-rule matches across pipeline stages.
Optimizer factories, parameter planning, updates and state serialization use
the production optimizer implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.optim import Optimizer

from llmtuner.accelerator.collectives import (
    all_reduce,
    synchronize_local_errors,
)
from llmtuner.accelerator.distributed import get_comm_device
from llmtuner.components.optimizer.optimizer import (
    OptimizersContainer as BaseOptimizersContainer,
)
from llmtuner.components.optimizer.optimizer import (
    build_optimizer_factories,
    disable_unsupported_fused_npu_updates,
)
from llmtuner.components.optimizer.state_residency import (
    install_state_residency,
)
from llmtuner.parallel.assembly import ModelPartLayout

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup
    from torch.distributed.device_mesh import DeviceMesh

    # Type-only. ``llmtuner.config`` does not import this package at runtime (it
    # describes the object; the trainer builds it), so there is no cycle to
    # break -- this matches ``lr_scheduler.py``'s import of ``LRSchedulerConfig``.
    from llmtuner.config.optimizer_config import OptimizerConfig

__all__ = ["OptimizersContainer"]


class OptimizersContainer(BaseOptimizersContainer):
    """One optimizer per (model part, optimizer name), driven as a single one.

    The training loop should not know how many optimizers exist, so this is
    itself an ``Optimizer``: ``step`` and ``zero_grad`` (the two methods the
    loop calls) fan out to the inner ones, and everything else -- gradient
    clipping, checkpointing -- reaches the parameters through ``param_groups``,
    which ``Optimizer.__init__`` merges across the inner optimizers.

    ``_initialize_optimizer_interface`` populates ``param_groups`` for gradient
    clipping. Step pre-hooks (MoE balancing and aux-loss roll-up) need the machinery
    that only ``Optimizer.__init__`` sets up. Calling it with an empty options
    dict gives both without registering any hyperparameters of its own.

    The number of inner optimizers follows from two independent splits:

    * one per **local model part**, because each rank owns its assigned pipeline
      stages/chunks. A rank can hold several parts under an interleaved schedule
      -- which matters, because a step pre-hook fires once
      per ``step()`` call, i.e. once per container, not once per inner optimizer.
    * one per **optimizer name** within a part, because different parameter
      groups may name different optimizer classes.

    Parameters are matched to groups by regex against their FQN, first pattern
    wins; that is what lets norm and bias parameters take a different weight
    decay from the rest. A ``ValueError`` is raised if a pattern matches nothing
    across all local model parts (a typo'd pattern is otherwise a silent no-op)
    or if a trainable parameter ends up in no group at all. Shared parameters
    across local model parts and colliding checkpoint FQNs are rejected before
    constructing any inner optimizer.

    Rules are validated across local parts, or across pipeline ranks when an
    explicit pipeline_mesh is supplied. All pipeline ranks must then construct
    the container together with the same ordered rules.

    Args:
        config: the run's optimizer configuration. ``param_groups`` is the list
            of ``ParamGroupConfig`` patterns; ``implementation`` selects
            ``fused`` / ``foreach`` / ``for-loop`` for every inner optimizer.
        model_parts: the model chunks to optimize, one per pipeline stage.
        pipeline_mesh: optional PP axis for checking rules over every stage.
        initialization_group: optional group coordinating local planning and
            construction errors; defaults to the PP group. Trainer supplies world
            so failures also reach other data replicas. All members must construct
            together, while rule matching continues to use the PP group.
        on_acquired: optional owner callback, invoked before inner resources are
            created. It must retain this container without raising or device work;
            the owner can retry close if construction and cleanup both fail.
    """

    optimizers: list[Optimizer]
    model_parts: list[nn.Module]
    sharded_tensor_parallel: bool = False
    sharded_expert_parallel: bool = False

    def __init__(
        self,
        config: OptimizerConfig,
        *,
        model_parts: list[nn.Module],
        pipeline_mesh: DeviceMesh | None = None,
        initialization_group: ProcessGroup | None = None,
        sharded_tensor_parallel: bool = False,
        sharded_expert_parallel: bool = False,
        runtime_layouts: tuple[ModelPartLayout, ...] = (),
        on_acquired: Callable[[OptimizersContainer], None] | None = None,
    ) -> None:
        self.optimizers = []
        self.model_parts = list(model_parts)
        self.sharded_tensor_parallel = sharded_tensor_parallel
        self.sharded_expert_parallel = sharded_expert_parallel
        self.state_residency = config.state_residency
        # close() is usable once the inner-owner list exists. Hand off before
        # planning or constructors can fail, so failed cleanup remains reachable.
        if on_acquired is not None:
            on_acquired(self)

        pipeline_group = (
            pipeline_mesh.get_group()
            if pipeline_mesh is not None and pipeline_mesh.size() > 1
            else None
        )
        error_group = (
            initialization_group if initialization_group is not None else pipeline_group
        )
        with (
            synchronize_local_errors("Optimizer parameter planning", group=error_group)
            if error_group is not None
            else nullcontext()
        ):
            self._bind_runtime_layouts(
                runtime_layouts, validate_restore_groups=self._validate_restore_policy
            )
            impl_kwargs = self._build_impl_kwargs(config)
            model_group_plans = [
                self._build_param_groups(
                    layout,
                    param_group_configs=config.param_groups,
                    impl_kwargs=impl_kwargs,
                )
                for layout in self.runtime_layouts
            ]
        matched_configs = {
            index
            for plan in model_group_plans
            for groups in plan.values()
            for index in groups.config_indices
        }
        if pipeline_group is not None:
            group = pipeline_group
            matches = torch.tensor(
                [index in matched_configs for index in range(len(config.param_groups))],
                dtype=torch.int32,
                device=get_comm_device(group),
            )
            all_reduce(matches, op="max", group=group)
            matched_configs = {
                index for index, matched in enumerate(matches.tolist()) if matched
            }
        with (
            self._construction_resources(),
            (
                synchronize_local_errors("Optimizer initialization", group=error_group)
                if error_group is not None
                else nullcontext()
            ),
        ):
            all_params = self._validate_parameter_plan(
                model_group_plans,
                param_group_configs=config.param_groups,
                matched_configs=matched_configs,
            )

            optimizer_factories = build_optimizer_factories(config)
            for groups_by_opt_name in model_group_plans:
                for name in groups_by_opt_name:
                    if name not in optimizer_factories:
                        raise NotImplementedError(f"Optimizer {name} not added.")
                    if name == "DistMuon" and config.state_residency != "device":
                        raise ValueError(
                            "DistMuon requires optimizer.state_residency='device'; "
                            "npu_virtual currently supports Adam/AdamW moments only."
                        )

            for part_idx, groups_by_opt_name in enumerate(model_group_plans):
                for opt_name, groups in groups_by_opt_name.items():
                    disable_unsupported_fused_npu_updates(opt_name, groups.param_groups)
                    optimizer = optimizer_factories[opt_name](groups.param_groups)
                    # Own returned resources before validation or residency
                    # installation can fail. Factories own partial objects.
                    self.optimizers.append(optimizer)
                    # PyTorch accepts arbitrary group metadata, so misspelled
                    # update options otherwise silently leave defaults active.
                    # initial_lr is the scheduler's intentional group metadata.
                    for group in optimizer.param_groups:
                        unknown = (
                            group.keys()
                            - optimizer.defaults.keys()
                            - {"params", "param_names", "initial_lr"}
                        )
                        if unknown:
                            raise ValueError(
                                f"{opt_name} optimizer_kwargs contain unknown options: "
                                f"{sorted(unknown)}"
                            )
                    install_state_residency(optimizer, config.state_residency)
                    self._log_optimizer(optimizer, part_idx, groups.patterns)

            self._initialize_optimizer_interface(all_params)
            self._validate_runtime_layouts()
