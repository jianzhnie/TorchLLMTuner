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
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ...accelerator.collectives import synchronize_local_errors
from ...accelerator.device import device_type, get_current_device
from ...parallel.assembly import ModelPartLayout
from ...utils.logging import get_logger
from .adamw_swap import AdamWSwap
from .checkpoint_state import OptimizerCheckpointState
from .flex_shard.dist_muon import DistMuon
from .state_residency import (
    install_state_residency,
    validate_virtual_optimizer_groups,
)

try:
    from torch.optim.optimizer import _get_fused_kernels_supported_devices
except ImportError:  # Private helper is absent on some supported torch builds.
    _get_fused_kernels_supported_devices = None

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

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
    if _get_fused_kernels_supported_devices is None:
        return frozenset({"cuda", "xpu"})
    return frozenset(_get_fused_kernels_supported_devices())


def disable_unsupported_fused_npu_updates(
    optimizer_name: str, param_groups: list[dict[str, Any]]
) -> None:
    if device_type != "npu" or optimizer_name not in ("Adam", "AdamW", "AdamWSwap"):
        return
    # The upstream fused-device list is shared by Adam/AdamW, but vendor
    # registrations may implement only one. In particular, NPU Adam otherwise
    # silently copies to CPU.
    operator = (
        "aten::_fused_adam_" if optimizer_name == "Adam" else "aten::_fused_adamw_"
    )
    if torch._C._dispatch_has_kernel_for_dispatch_key(operator, "PrivateUse1"):
        return
    for group in param_groups:
        if group.get("fused"):
            logger.info(
                "%s has no native fused NPU kernel; using for-loop to avoid "
                "CPU fallback.",
                optimizer_name,
            )
            group["fused"] = False


class OptimizersContainer(OptimizerCheckpointState, Optimizer):
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

    Rules are validated across local parts. Distributed construction errors
    coordinate through the explicit initialization group.

    Args:
        config: the run's optimizer configuration. ``param_groups`` is the list
            of ``ParamGroupConfig`` patterns; ``implementation`` selects
            ``fused`` / ``foreach`` / ``for-loop`` for every inner optimizer.
        model_parts: the model chunks to optimize, one per pipeline stage.
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

        error_group = initialization_group
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

    def _validate_runtime_layouts(self) -> None:
        super()._validate_runtime_layouts()
        # Initial binding validates models before any optimizer is constructed.
        if self._layout_snapshots is None:
            return
        expected = {
            parameter: name
            for layout in self.runtime_layouts
            for parameter, name in layout.parameter_state_names.items()
            if parameter.requires_grad
        }
        seen: set[nn.Parameter] = set()
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                parameters = group["params"]
                names = group.get("param_names")
                if names is None or len(names) != len(parameters):
                    raise ValueError(
                        "Optimizer parameter names must match assembly ownership"
                    )
                for parameter, name in zip(parameters, names, strict=True):
                    if parameter not in expected or expected[parameter] != name:
                        raise ValueError(
                            "Optimizer parameter names disagree with assembly "
                            "ownership; rebuild the optimizer after changing groups"
                        )
                    if parameter in seen:
                        raise ValueError(
                            "Optimizer groups duplicate parameter ownership"
                        )
                    seen.add(parameter)
        if seen != set(expected):
            raise ValueError("Optimizer groups no longer cover assembly ownership")

    @contextmanager
    def _construction_resources(self) -> Iterator[None]:
        """Release returned inner optimizers on local or coordinated failure."""
        try:
            yield
        except BaseException as construction_error:
            try:
                self.close()
            except BaseException as cleanup_error:
                try:
                    logger.exception("Optimizer cleanup failed during construction")
                except BaseException:
                    pass
                raise construction_error from cleanup_error
            raise

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

    def _validate_parameter_plan(
        self,
        model_group_plans: list[dict[str, _OptimizerGroups]],
        *,
        param_group_configs: list[ParamGroupConfig],
        matched_configs: set[int],
    ) -> list[nn.Parameter]:
        """Validate rules, ownership and checkpoint names before construction.

        ``matched_configs`` includes rule matches from other pipeline ranks.
        Parameter identity and checkpoint FQNs are checked across local parts.
        This phase does not instantiate optimizers or modify model parameters.
        """
        all_params: list[nn.Parameter] = []
        for index, param_group_config in enumerate(param_group_configs):
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

        return all_params

    def __iter__(self) -> Iterator[Optimizer]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Advance every inner optimizer.

        ``closure`` is rejected rather than ignored: this container cannot
        support one, and silently dropping it would skip the user's loss
        recomputation. Returning ``None`` matches ``Optimizer.step``.

        Updates are not atomic across optimizers or step hooks. An exception
        may leave earlier optimizers and pre-hook state already advanced; a
        caller must restore a complete training checkpoint before retrying.
        """
        if closure is not None:
            raise ValueError("OptimizersContainer does not support closures")
        for optimizer in self.optimizers:
            optimizer.step()
        return None

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    @property
    def observation_metrics(self) -> dict[str, float]:
        """Sum additive local counters declared by inner optimizer services.

        Inner services own names, units and counter lifetimes. Equal metric
        names must represent additive values across model parts; this property
        neither synchronizes ranks nor resets the underlying counters.
        """
        metrics: dict[str, float] = {}
        for optimizer in self.optimizers:
            for name, value in getattr(optimizer, "observation_metrics", {}).items():
                metrics[name] = metrics.get(name, 0.0) + float(value)
        return metrics

    def close(self) -> None:
        """Close owned optimizer resources, retaining failed owners for retry."""
        first_error: BaseException | None = None
        pending = getattr(self, "_pending_close", list(reversed(self.optimizers)))
        failed: list[Optimizer] = []
        for optimizer in pending:
            close = getattr(optimizer, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except BaseException as error:
                failed.append(optimizer)
                if first_error is None:
                    first_error = error
        self._pending_close = failed
        if first_error is not None:
            raise first_error

    def _validate_restore_policy(self, groups: Iterable[Mapping[str, Any]]) -> None:
        """Apply optimizer-owned residency constraints before checkpoint writes."""
        if self.state_residency == "npu_virtual":
            validate_virtual_optimizer_groups(groups)

    def _initialize_optimizer_interface(
        self, all_params: Sequence[torch.Tensor]
    ) -> None:
        # ``Optimizer.__init__`` is what populates ``param_groups`` and sets up
        # the hook machinery that ``register_step_pre_hook`` needs. The empty
        # options dict is deliberate: the container's own ``param_groups`` is
        # only a view over the inner optimizers' parameters, and each inner
        # optimizer already holds the hyperparameters for its groups.
        # An empty tensor scope still needs the hook/container interface;
        # an explicit empty group is supported by torch without dummy tensors.
        Optimizer.__init__(self, all_params or [{"params": []}], {})
