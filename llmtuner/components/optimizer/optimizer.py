"""Compose model-part optimizers and expose one FQN-keyed checkpoint state.

Adapted from torchtitan ``components/optimizer/optimizer.py``. The container
owns one Adam/AdamW optimizer per model part and optimizer name; parameter
patterns use first-match ownership. Step and zero_grad fan out to inner
optimizers, while the container exposes their parameters to clipping and hooks.

Checkpoint state is materialized and flattened by ``state_dict.py``. Expert
states are gathered/sliced through the EP checkpoint adapter. Fused kernel
selection falls back according to device support. DistMuon, bf16 optimizer
states, and upstream optimizer factory metadata are not implemented here.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer

from ...accelerator.collectives import all_reduce, synchronize_local_errors
from ...accelerator.device import device_type
from ...accelerator.distributed import get_comm_device
from ...parallel.expert_parallel.checkpointing import (
    expert_shard_map,
    gather_expert_state,
    load_expert_state,
)
from ...utils.logging import get_logger
from ...utils.module_names import canonical_fqn
from .state_dict import (
    get_flat_optim_state_dict,
    init_optim_state,
    unflatten_optim_state_dict,
)

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup
    from torch.distributed.device_mesh import DeviceMesh

    # Type-only. ``llmtuner.config`` does not import this package at runtime (it
    # describes the object; the trainer builds it), so there is no cycle to
    # break -- this matches ``lr_scheduler.py``'s import of ``LRSchedulerConfig``.
    from ...config.optimizer_config import OptimizerConfig, ParamGroupConfig

logger = get_logger(__name__)

__all__ = ["OptimizersContainer"]


@dataclass(slots=True)
class _OptimizerGroups:
    """Constructor groups and their matching diagnostic patterns."""

    param_groups: list[dict[str, Any]] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    config_indices: list[int] = field(default_factory=list)


def fused_kernel_device_types() -> frozenset[str]:
    """Device types torch ships fused Adam/AdamW kernels for.

    torch keeps this list internally (``torch.optim.adam``); asking it beats
    hard-coding, so a build without an XPU kernel, or a vendor build that adds
    its own, is described correctly. The fallback is the pair those kernels
    have always covered, for the day the private helper moves.
    """
    try:
        from torch.optim.adam import _get_fused_kernels_supported_devices
    except ImportError:  # pragma: no cover - private helper moved
        return frozenset({"cuda", "xpu"})
    return frozenset(_get_fused_kernels_supported_devices())


class OptimizersContainer(Optimizer, Stateful):
    """One optimizer per (model part, optimizer name), driven as a single one.

    The training loop should not know how many optimizers exist, so this is
    itself an ``Optimizer``: ``step`` and ``zero_grad`` (the two methods the
    loop calls) fan out to the inner ones, and everything else -- gradient
    clipping, checkpointing -- reaches the parameters through ``param_groups``,
    which ``Optimizer.__init__`` merges across the inner optimizers.

    ``_initialize_optimizer_interface`` populates ``param_groups`` for gradient
    clipping. The step
    pre-hooks (MoE load balancing, the aux-loss roll-up) need the hook machinery
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
    or if a trainable parameter ends up in no group at all.

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
    """

    optimizers: list[Optimizer]
    model_parts: list[nn.Module]

    def __init__(
        self,
        config: OptimizerConfig,
        *,
        model_parts: list[nn.Module],
        pipeline_mesh: DeviceMesh | None = None,
        initialization_group: ProcessGroup | None = None,
    ) -> None:
        all_params: list[nn.Parameter] = []
        self.optimizers = []
        self.model_parts = list(model_parts)

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
            impl_kwargs = self._build_impl_kwargs(config)
            model_group_plans = [
                self._build_param_groups(
                    model,
                    param_group_configs=config.param_groups,
                    impl_kwargs=impl_kwargs,
                )
                for model in self.model_parts
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
            synchronize_local_errors("Optimizer initialization", group=error_group)
            if error_group is not None
            else nullcontext()
        ):
            for index, param_group_config in enumerate(config.param_groups):
                if index not in matched_configs:
                    raise ValueError(
                        "optimizer.param_groups pattern "
                        f"{param_group_config.pattern!r} matched no parameters "
                        "across model parts in the pipeline"
                    )
            for groups_by_opt_name in model_group_plans:
                for groups in groups_by_opt_name.values():
                    for group in groups.param_groups:
                        all_params.extend(group["params"])
            self._validate_params(all_params)
            checkpoint_names: set[str] = set()
            for part_index, groups_by_opt_name in enumerate(model_group_plans):
                for groups in groups_by_opt_name.values():
                    for group in groups.param_groups:
                        names = set(group["param_names"])
                        collisions = checkpoint_names & names
                        if collisions:
                            raise ValueError(
                                "Optimizer checkpoint FQNs collide across model parts "
                                f"at part {part_index}: {sorted(collisions)}. "
                                "Model parts must use unique parameter FQNs."
                            )
                        checkpoint_names.update(names)
            optimizer_factories: dict[str, Callable[..., Optimizer]] = {
                "Adam": torch.optim.Adam,
                "AdamW": torch.optim.AdamW,
            }
            for groups_by_opt_name in model_group_plans:
                for name in groups_by_opt_name:
                    if name not in optimizer_factories:
                        raise NotImplementedError(f"Optimizer {name} not added.")

            for part_idx, groups_by_opt_name in enumerate(model_group_plans):
                for opt_name, groups in groups_by_opt_name.items():
                    optimizer = optimizer_factories[opt_name](groups.param_groups)
                    self.optimizers.append(optimizer)
                    self._log_optimizer(optimizer, part_idx, groups.patterns)

            self._initialize_optimizer_interface(all_params)

    @staticmethod
    def _build_impl_kwargs(config: OptimizerConfig) -> dict[str, Any]:
        """The implementation kwargs (``fused`` / ``foreach``) applied to all groups.

        An ``optimizer_kwargs`` entry on a ``ParamGroupConfig`` overrides these --
        the update below is per group, so a group can opt out (``fused=False``).
        """
        implementation = config.implementation
        if implementation not in ("fused", "foreach", "for-loop"):
            raise ValueError(
                f"Unknown optimizer implementation {implementation!r}; expected "
                "one of 'fused', 'foreach', 'for-loop'."
            )
        has_fused_kernel = device_type in fused_kernel_device_types()
        if implementation == "fused" and not has_fused_kernel:
            # See the module docstring: the flag is a preference, and the
            # default one has to leave a CPU run working.
            logger.info(
                "optimizer.implementation='fused' has no kernel on %s; using "
                "the for-loop implementation instead.",
                device_type,
            )
            implementation = "for-loop"
        return {
            "fused": implementation == "fused",
            "foreach": implementation == "foreach",
        }

    @staticmethod
    def _build_param_groups(
        model: nn.Module,
        *,
        param_group_configs: list[ParamGroupConfig],
        impl_kwargs: dict[str, Any],
    ) -> dict[str, _OptimizerGroups]:
        """Partition a model's parameters into per-optimizer param groups.

        Each parameter is claimed by the first ``ParamGroupConfig`` whose pattern
        any of its registered names matches, so order matters and a catch-all
        belongs last. Shared parameters are assigned once by identity.

        Returns groups keyed by optimizer name, paired with diagnostic patterns.
        The patterns stay out of the group dict on purpose -- they are for the
        log line only, and a saved optimizer state dict would carry them forever.

        Each group dict also carries ``param_names`` (canonical FQNs aligned with
        ``params``). PyTorch records those on the group, and the checkpoint
        helpers read them back to key optimizer state by FQN.
        """
        groups: dict[str, _OptimizerGroups] = defaultdict(_OptimizerGroups)
        claimed: set[int] = set()
        canonical_names: dict[str, str] = {}
        trainable_parameters = [
            (name, param)
            for name, param in model.named_parameters(remove_duplicate=False)
            if param.requires_grad
        ]

        primary_names: dict[int, str] = {}
        for name, param in trainable_parameters:
            primary_names.setdefault(id(param), name)

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
                    primary_name = primary_names[id(param)]
                    canonical_name = canonical_fqn(primary_name)
                    if canonical_name in canonical_names:
                        raise ValueError(
                            "Optimizer parameter FQNs collide after wrapper removal: "
                            f"{canonical_names[canonical_name]!r} and {name!r} "
                            f"both map to {canonical_name!r}"
                        )
                    canonical_names[canonical_name] = primary_name
                    params.append(param)
                    param_names.append(canonical_name)
                    claimed.add(id(param))

            if not params:
                # A rule may target another local pipeline chunk. The
                # constructor checks unmatched rules over all local parts.
                continue

            optimizer_groups = groups[param_group_config.optimizer_name]
            optimizer_groups.param_groups.append(
                {
                    "params": params,
                    "param_names": param_names,
                    **impl_kwargs,
                    **param_group_config.optimizer_kwargs,
                }
            )
            optimizer_groups.patterns.append(param_group_config.pattern)
            optimizer_groups.config_indices.append(config_index)

        return groups

    def _log_optimizer(
        self, optimizer: Optimizer, part_idx: int, patterns: list[str]
    ) -> None:
        """Log one inner optimizer's group assignments.

        The patterns are logged here and nowhere else -- they are deliberately
        not stored on the param groups, so this line is the only record of which
        pattern produced which group.
        """
        key_kwargs = (
            "lr",
            "weight_decay",
            "betas",
            "eps",
            "momentum",
            "nesterov",
            "fused",
            "foreach",
        )
        optimizer_name = type(optimizer).__name__
        for group, pattern in zip(optimizer.param_groups, patterns, strict=True):
            kwargs = {key: group[key] for key in key_kwargs if key in group}
            logger.info(
                "Optimizer %s (model_part=%d): %d params [%s] %s",
                optimizer_name,
                part_idx,
                len(group["params"]),
                pattern,
                kwargs,
            )

    def _validate_params(self, all_params: list[nn.Parameter]) -> None:
        """Every trainable parameter must land in exactly one group.

        Upstream asserts equality of the id sets; this names *which* invariant
        broke, because the two directions have different fixes and upstream's
        count-only message leaves them indistinguishable. User-reachable both
        ways, which is why this raises ``ValueError`` rather than asserting:

        * missing -- explicit ``param_groups`` without a catch-all, so a
          parameter would silently stay frozen at its initial value.
        * assigned twice -- shared parameters or repeated modules across local
          model parts. First-match-wins prevents duplicates within one part,
          but each part constructs its own groups.
        """
        registered = {id(param) for param in all_params}
        missing = [
            (name, param)
            for model in self.model_parts
            for name, param in model.named_parameters()
            if param.requires_grad and id(param) not in registered
        ]
        if missing:
            names = ", ".join(
                f"{name} ({tuple(param.shape)})" for name, param in missing
            )
            raise ValueError(
                f"optimizer.param_groups left {len(missing)} trainable parameter(s) "
                f"unassigned, so they would keep their initial values: {names}. "
                "Add a catch-all ParamGroupConfig(pattern='.*') last."
            )

        duplicates = len(all_params) - len(registered)
        if duplicates:
            raise ValueError(
                f"optimizer.param_groups assigned {duplicates} parameter(s) to more "
                f"than one group. Check for repeated model parts or parameters "
                f"shared across local model parts."
            )

    def __iter__(self) -> Iterator[Optimizer]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Advance every inner optimizer.

        ``closure`` is rejected rather than ignored: this container cannot
        support one, and silently dropping it would skip the user's loss
        recomputation. Returning ``None`` matches ``Optimizer.step``.
        """
        if closure is not None:
            raise ValueError("OptimizersContainer does not support closures")
        for optimizer in self.optimizers:
            optimizer.step()
        return None

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """A flat, FQN-keyed state dict covering every inner optimizer.

        Side effect: if an inner optimizer has not stepped yet,
        ``init_optim_state`` materializes its state first (a zero-gradient,
        zero-lr step) so DCP has tensors to read, and to write into on load. It
        leaves parameters alone, and is a no-op once state exists.
        """
        result: dict[str, Any] = {}
        for optimizer in self.optimizers:
            init_optim_state(optimizer)
            optimizer_state = get_flat_optim_state_dict(optimizer)
            duplicated = result.keys() & optimizer_state.keys()
            if duplicated:
                raise ValueError(
                    "Optimizer checkpoint keys overlap across inner optimizers: "
                    f"{sorted(duplicated)}. Model parts must use unique parameter FQNs."
                )
            result.update(optimizer_state)
        # Expert optimizer states are rank-heterogeneous like the expert
        # weights themselves; gather them to the full tensors the checkpoint
        # stores (parallel/expert_parallel/checkpointing.py).
        return gather_expert_state(
            result,
            shard_map=expert_shard_map(self.model_parts),
            model_parts=self.model_parts,
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # init_optim_state must run first: the unflattening step reads each
        # optimizer's live state to learn which state tensors to expect, so a
        # fresh optimizer would find nothing to write into.
        state_dict = load_expert_state(
            state_dict,
            shard_map=expert_shard_map(self.model_parts),
            model_parts=self.model_parts,
        )
        for optimizer in self.optimizers:
            init_optim_state(optimizer)
        # Validate every inner state before applying any checkpoint values.
        optimizer_states = [
            unflatten_optim_state_dict(optimizer, flat_state=state_dict)
            for optimizer in self.optimizers
        ]
        for optimizer, optimizer_state in zip(
            self.optimizers, optimizer_states, strict=True
        ):
            optimizer.load_state_dict(optimizer_state)

    def _initialize_optimizer_interface(self, all_params: list[nn.Parameter]) -> None:
        # ``Optimizer.__init__`` is what populates ``param_groups`` and sets up
        # the hook machinery that ``register_step_pre_hook`` needs. The empty
        # options dict is deliberate: the container's own ``param_groups`` is
        # only a view over the inner optimizers' parameters, and each inner
        # optimizer already holds the hyperparameters for its groups.
        Optimizer.__init__(self, all_params, {})
