"""Compose model-part optimizers and expose one FQN-keyed checkpoint state.

Adapted from torchtitan ``components/optimizer/optimizer.py``. The container
owns one optimizer per model part and optimizer name; parameter
patterns use first-match ownership. Step and zero_grad fan out to inner
optimizers, while the container exposes their parameters to clipping and hooks.

Checkpoint state is materialized and flattened by ``state_dict.py``. Expert
states are gathered/sliced through the EP checkpoint adapter. Fused kernel
selection falls back according to device support. DistMuon is implemented in
``flex_shard`` and selected through explicit parameter-group configuration.
AdamWSwap performs explicit batched state transfers; ``state_residency`` instead
installs Host-backed moments on native Adam/AdamW. Their storage and checkpoint
constraints remain separate. Configurable BF16 optimizer-state precision and
upstream optimizer factory metadata are not implemented here.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from llmtuner.accelerator.collectives import (
    all_reduce,
    synchronize_local_errors,
)
from llmtuner.accelerator.device import get_current_device
from llmtuner.accelerator.distributed import get_comm_device
from llmtuner.components.optimizer.adamw_swap import AdamWSwap
from llmtuner.components.optimizer.flex_shard.dist_muon import DistMuon
from llmtuner.components.optimizer.optimizer import (
    OptimizersContainer as BaseOptimizersContainer,
)
from llmtuner.components.optimizer.optimizer import (
    _OptimizerGroups,
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
    from llmtuner.config.optimizer_config import OptimizerConfig, ParamGroupConfig

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

            def build_dist_muon(groups: list[dict]) -> DistMuon:
                ns_devices = {group.get("ns_device") for group in groups}
                if len(ns_devices) != 1:
                    raise ValueError(
                        "DistMuon ns_device must be identical across parameter groups"
                    )
                for group in groups:
                    group.pop("ns_device", None)
                return DistMuon(
                    groups,
                    compute_sharding_by_fqn=config.compute_sharding_by_fqn,
                    bucket_configs=config.bucket_configs,
                    ns_device=ns_devices.pop(),
                )

            optimizer_factories: dict[str, Callable[..., Optimizer]] = {
                "Adam": torch.optim.Adam,
                "AdamW": torch.optim.AdamW,
                "AdamWSwap": lambda groups: AdamWSwap(
                    groups,
                    max_device_bytes=config.swap_max_device_bytes,
                    max_host_bytes=config.swap_max_host_bytes,
                    prefetch=config.swap_prefetch,
                    memory_fraction=config.swap_memory_fraction,
                    update_device=get_current_device(),
                ),
                "DistMuon": build_dist_muon,
            }
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

    @staticmethod
    def _build_param_groups(
        layout: ModelPartLayout,
        *,
        param_group_configs: list[ParamGroupConfig],
        impl_kwargs: dict[str, Any],
    ) -> dict[str, _OptimizerGroups]:
        """Partition a model's parameters into per-optimizer param groups.

        Each parameter is claimed by the first ``ParamGroupConfig`` whose pattern
        any of its canonical registered aliases matches, so order matters and
        a catch-all belongs last. Shared parameters are assigned once by identity.

        Returns groups keyed by optimizer name, paired with diagnostic patterns.
        The patterns stay out of the group dict on purpose -- they are for the
        log line only, and a saved optimizer state dict would carry them forever.

        Each group dict also carries ``param_names`` (canonical FQNs aligned with
        ``params``). PyTorch records those on the group, and the checkpoint
        helpers read them back to key optimizer state by FQN.
        """
        groups: dict[str, _OptimizerGroups] = defaultdict(_OptimizerGroups)
        claimed: set[int] = set()
        state_names = layout.state_parameter_names
        trainable_parameters = [
            (state_names[path], param)
            for path, param in layout.parameter_bindings.items()
            if param.requires_grad
        ]

        layout_names = layout.parameter_state_names

        for config_index, param_group_config in enumerate(param_group_configs):
            reserved = {
                "params",
                "param_names",
            } & param_group_config.optimizer_kwargs.keys()
            if reserved:
                raise ValueError(
                    "optimizer_kwargs cannot override managed parameter fields: "
                    f"{sorted(reserved)}"
                )
            pattern = re.compile(param_group_config.pattern)
            params: list[nn.Parameter] = []
            param_names: list[str] = []
            for name, param in trainable_parameters:
                if id(param) not in claimed and pattern.search(name):
                    # Match any alias, but checkpoint under the first registered
                    # name so group pattern changes do not rename optimizer state.
                    params.append(param)
                    param_names.append(layout_names[param])
                    claimed.add(id(param))

            if not params:
                # A rule may target another local pipeline chunk. The
                # constructor checks unmatched rules over all local parts.
                continue

            optimizer_groups = groups[param_group_config.optimizer_name]
            group_impl_kwargs = (
                {} if param_group_config.optimizer_name == "DistMuon" else impl_kwargs
            )
            optimizer_groups.param_groups.append(
                {
                    "params": params,
                    "param_names": param_names,
                    **group_impl_kwargs,
                    **param_group_config.optimizer_kwargs,
                }
            )
            optimizer_groups.patterns.append(param_group_config.pattern)
            optimizer_groups.config_indices.append(config_index)

        return groups
