"""Explicit local layout results shared by parallel and training assembly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from torch import nn

from .execution import ExecutionLayout
from .pipeline_parallel.ownership import PipelineOwnership
from .replacements import ReplacementMap

if TYPE_CHECKING:
    from .pipeline_parallel.pipeline_parallel import PipelineParallelSetup


@dataclass(slots=True)
class ModelPartLayout:
    """One local part's PP origin, transformations, and registered aliases.

    Owns metadata only. Refresh after parameter materialization before creating
    optimizers; parameter identity changes are tracked by stable registrations,
    never by shapes or storage addresses.
    """

    execution: ExecutionLayout | None = None
    pipeline_ownership: PipelineOwnership | None = None
    replacements: ReplacementMap = field(default_factory=ReplacementMap)
    parameter_bindings: dict[str, nn.Parameter] = field(default_factory=dict)

    def refresh(self, model: nn.Module) -> None:
        current = dict(model.named_parameters(remove_duplicate=False))
        replacements: dict[nn.Parameter, tuple[nn.Parameter, ...]] = {}
        for path, source in self.parameter_bindings.items():
            if path not in current:
                raise ValueError(f"Layout refresh lost parameter registration {path!r}")
            target = current[path]
            if target is not source:
                previous = replacements.setdefault(source, (target,))
                if previous[0] is not target:
                    raise ValueError(
                        f"Layout refresh split a parameter alias at {path!r}"
                    )
        self.replacements.record_replacements({}, replacements)
        self.parameter_bindings = current

    @property
    def parameter_aliases(self) -> Mapping[nn.Parameter, tuple[str, ...]]:
        """Read-only parameter identities and all final registered paths."""
        aliases: dict[nn.Parameter, list[str]] = {}
        for path, parameter in self.parameter_bindings.items():
            aliases.setdefault(parameter, []).append(path)
        return MappingProxyType({p: tuple(paths) for p, paths in aliases.items()})

    def validate(self, model: nn.Module) -> None:
        """Reject stale bindings before consumers create optimizer/state views."""
        current = dict(model.named_parameters(remove_duplicate=False))
        if current.keys() != self.parameter_bindings.keys() or any(
            current[path] is not parameter
            for path, parameter in self.parameter_bindings.items()
        ):
            raise ValueError("Model layout parameter bindings are stale")
        registered = set(current.values())
        if any(
            target not in registered
            for targets in self.replacements.parameter_targets.values()
            for target in targets
        ):
            raise ValueError("Model layout replacement target is no longer registered")

    def global_parameter_targets(
        self, source: nn.Parameter
    ) -> tuple[nn.Parameter, ...]:
        """Resolve global PP source ownership through subsequent TP/EP/FSDP."""
        local = (
            self.pipeline_ownership.parameter_bindings({source})
            if self.pipeline_ownership is not None
            else {source}
        )
        return tuple(
            dict.fromkeys(
                target
                for parameter in local
                for target in self.replacements.parameter_targets.get(
                    parameter, (parameter,)
                )
            )
        )


@dataclass(frozen=True, slots=True)
class ParallelAssembly:
    """Transformed execution objects plus layouts in local model-part order."""

    execution: nn.Module | PipelineParallelSetup
    layouts: tuple[ModelPartLayout, ...]
