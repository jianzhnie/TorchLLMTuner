"""Explicit local layout results shared by parallel and training assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from torch import nn

from ..utils.module_names import canonical_fqn
from .execution import ExecutionLayout
from .pipeline_parallel.ownership import PipelineParameterOwnership
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
    pipeline_ownership: PipelineParameterOwnership | None = None
    replacements: ReplacementMap = field(default_factory=ReplacementMap)
    parameter_bindings: dict[str, nn.Parameter] = field(default_factory=dict)

    def refresh(self, model: nn.Module) -> None:
        current = dict(model.named_parameters(remove_duplicate=False))
        targets: dict[nn.Parameter, nn.Parameter] = {}
        for path, source in self.parameter_bindings.items():
            if path not in current:
                raise ValueError(f"Layout refresh lost parameter registration {path!r}")
            target = current[path]
            # Unchanged registrations also constrain alias identity: replacing
            # only one path of a shared parameter must not silently split it.
            previous = targets.setdefault(source, target)
            if previous is not target:
                raise ValueError(f"Layout refresh split a parameter alias at {path!r}")
        replacements = {
            source: (target,)
            for source, target in targets.items()
            if target is not source
        }
        self.replacements.record_replacements({}, replacements)
        self.parameter_bindings = current
        if self.pipeline_ownership is not None:
            self.pipeline_ownership = self.pipeline_ownership.remap_parameters(
                self.replacements.parameter_targets
            )

    @property
    def parameter_aliases(self) -> Mapping[nn.Parameter, tuple[str, ...]]:
        """Read-only parameter identities and all final registered paths."""
        aliases: dict[nn.Parameter, list[str]] = {}
        for path, parameter in self.parameter_bindings.items():
            aliases.setdefault(parameter, []).append(path)
        return MappingProxyType({p: tuple(paths) for p, paths in aliases.items()})

    @property
    def state_parameter_names(self) -> Mapping[str, str]:
        """Model-state names preserve every registration, including aliases."""
        names: dict[str, str] = {}
        owners: dict[str, nn.Parameter] = {}
        for path, parameter in self.parameter_bindings.items():
            normalized = canonical_fqn(path)
            if owners.setdefault(normalized, parameter) is not parameter:
                raise ValueError(
                    "Layout parameter names collide after wrapper removal: "
                    f"{normalized!r}"
                )
            names[path] = normalized
        return MappingProxyType(names)

    @property
    def canonical_parameter_names(self) -> Mapping[str, str]:
        """Optimizer-state names choose one registration per parameter."""
        names: dict[str, str] = {}
        state_names = self.state_parameter_names
        for paths in self.parameter_aliases.values():
            primary = state_names[paths[0]]
            for path in paths:
                names[path] = primary
                names[state_names[path]] = primary
        return MappingProxyType(names)

    def validate(self, model: nn.Module) -> None:
        """Reject stale identities and alias order before creating state views."""
        current = dict(model.named_parameters(remove_duplicate=False))
        # The first registration names optimizer and EMA state for a shared
        # parameter. Set equality would miss a reorder that changes that name.
        if tuple(current) != tuple(self.parameter_bindings) or any(
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
        if self.pipeline_ownership is not None and any(
            target not in registered
            for targets in self.pipeline_ownership.local_parameters.values()
            for target in targets
        ):
            raise ValueError("Pipeline parameter target is no longer registered")
        # Every state consumer relies on an unambiguous wrapper-free name,
        # including consumers which only validate ownership before construction.
        _ = self.state_parameter_names

    def validate_parameter_snapshot(
        self, snapshot: Mapping[str, nn.Parameter], *, consumer: str
    ) -> None:
        """Reject rebinding since a state consumer captured its parameter scope."""
        current = self.parameter_bindings
        if tuple(current) != tuple(snapshot) or any(
            current[path] is not parameter for path, parameter in snapshot.items()
        ):
            raise ValueError(
                f"{consumer} parameter ownership changed; rebuild the consumer "
                "against the final model layout"
            )


@dataclass(frozen=True, slots=True)
class ParallelAssembly:
    """Transformed execution objects plus layouts in local model-part order."""

    execution: nn.Module | PipelineParallelSetup
    layouts: tuple[ModelPartLayout, ...]


def capture_model_part_layouts(
    model_parts: Sequence[nn.Module],
) -> tuple[ModelPartLayout, ...]:
    """Capture local bindings for standalone consumers without assembly metadata.

    This does not reconstruct transformation history or execution capabilities.
    Consumers retain their own snapshots and validate within their error boundary.
    """
    layouts = []
    for model in model_parts:
        layout = ModelPartLayout()
        layout.refresh(model)
        layouts.append(layout)
    return tuple(layouts)


def validate_model_part_layouts(
    model_parts: Sequence[nn.Module], layouts: tuple[ModelPartLayout, ...]
) -> None:
    """Validate local ownership, including frozen parameters and state names."""
    if len(model_parts) != len(layouts):
        raise ValueError("Model layouts must match local model parts")
    parameter_owners: dict[nn.Parameter, int] = {}
    name_owners: dict[str, int] = {}
    for index, (model, layout) in enumerate(zip(model_parts, layouts, strict=True)):
        if not isinstance(layout, ModelPartLayout):
            raise TypeError("Model layouts must be ModelPartLayout")
        layout.validate(model)
        for parameter in layout.parameter_aliases:
            previous = parameter_owners.setdefault(parameter, index)
            if previous != index:
                raise ValueError(
                    f"Parameter ownership overlaps across model parts {previous} "
                    f"and {index}; cross-part shared parameters are unsupported"
                )
        for name in layout.state_parameter_names.values():
            previous = name_owners.setdefault(name, index)
            if previous != index:
                raise ValueError(
                    f"Parameter checkpoint name {name!r} collides across model "
                    f"parts {previous} and {index}"
                )
