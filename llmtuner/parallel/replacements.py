"""Compose explicit identity mappings emitted by parallel transformations.

Mappings retain object references, including split and merged parameters. They
do not infer ownership from names or shapes and never mutate the model. One
instance belongs to one local model part and one assembly attempt.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

ReplacementObserver = Callable[
    [dict[nn.Module, nn.Module], dict[nn.Parameter, tuple[nn.Parameter, ...]]],
    None,
]


class ReplacementMap:
    """Track original and intermediate objects through successive conversions.

    Each event maps objects entering a transformation to its outputs. An empty
    parameter tuple represents removal. Composition deduplicates converging
    targets while preserving their first occurrence.
    Tensor storage sharing is separate from object replacement and remains
    described by the ownership snapshots in ``ParallelPlan``.
    """

    def __init__(self) -> None:
        self.module_targets: dict[nn.Module, nn.Module] = {}
        self.parameter_targets: dict[nn.Parameter, tuple[nn.Parameter, ...]] = {}

    def record_replacements(
        self,
        modules: dict[nn.Module, nn.Module],
        parameters: dict[nn.Parameter, tuple[nn.Parameter, ...]],
    ) -> None:
        """Compose one simultaneous event without following intra-event edges.

        Build both results before updating the map so an invalid event cannot
        leave a partially composed mapping. Consumers needing their own checks
        should perform them before recording the event.
        """
        module_targets = {
            source: modules.get(target, target)
            for source, target in self.module_targets.items()
        } | modules
        parameter_targets = {
            source: tuple(
                dict.fromkeys(
                    replacement
                    for target in targets
                    for replacement in parameters.get(target, (target,))
                )
            )
            for source, targets in self.parameter_targets.items()
        } | parameters
        self.module_targets = module_targets
        self.parameter_targets = parameter_targets
