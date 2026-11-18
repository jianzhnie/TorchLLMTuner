"""Explicit local layout results shared by parallel and training assembly."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from torch import nn

from ..utils.module_names import canonical_fqn
from .execution import ExecutionLayout
from .replacements import ReplacementMap
from .topology import ParallelTopologySnapshot


@dataclass(slots=True)
class ModelPartLayout:
    """One local model's transformations and registered aliases.

    Owns metadata only. Refresh after parameter materialization before creating
    optimizers; parameter identity changes are tracked by stable registrations,
    never by shapes or storage addresses.
    """

    execution: ExecutionLayout | None = None
    topology: ParallelTopologySnapshot | None = None
    replacements: ReplacementMap = field(default_factory=ReplacementMap)
    parameter_bindings: dict[str, nn.Parameter] = field(default_factory=dict)
    _bindings_captured: bool = field(default=False, init=False, repr=False)

    def refresh(self, model: nn.Module) -> None:
        """Rebind parameters while preserving registration order and aliases.

        Structural split/merge transformations belong to explicit replacement
        events before this snapshot, not to materialization inferred from FQNs.
        """
        current = dict(model.named_parameters(remove_duplicate=False))
        if (self._bindings_captured or self.parameter_bindings) and tuple(
            current
        ) != tuple(self.parameter_bindings):
            raise ValueError(
                "Layout refresh changed parameter registration paths or order"
            )
        targets: dict[nn.Parameter, nn.Parameter] = {}
        sources: dict[nn.Parameter, nn.Parameter] = {}
        for path, source in self.parameter_bindings.items():
            target = current[path]
            # Unchanged registrations also constrain alias identity: replacing
            # only one path of a shared parameter must not silently split it.
            previous = targets.setdefault(source, target)
            if previous is not target:
                raise ValueError(f"Layout refresh split a parameter alias at {path!r}")
            previous_source = sources.setdefault(target, source)
            if previous_source is not source:
                raise ValueError(
                    f"Layout refresh merged distinct parameters at {path!r}"
                )
        replacements = {
            source: (target,)
            for source, target in targets.items()
            if target is not source
        }
        self.replacements.record_replacements({}, replacements)
        self.parameter_bindings = current
        self._bindings_captured = True

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
        return MappingProxyType(
            {path: name for path, name, _ in self._normalized_parameter_bindings()}
        )

    def _normalized_parameter_bindings(
        self,
    ) -> Iterator[tuple[str, str, nn.Parameter]]:
        """Normalize each registered path and reject ambiguous checkpoint owners."""
        owners: dict[str, nn.Parameter] = {}
        for path, parameter in self.parameter_bindings.items():
            normalized = canonical_fqn(path)
            if owners.setdefault(normalized, parameter) is not parameter:
                raise ValueError(
                    "Layout parameter names collide after wrapper removal: "
                    f"{normalized!r}"
                )
            yield path, normalized, parameter

    @property
    def parameter_state_names(self) -> Mapping[nn.Parameter, str]:
        """One checkpoint name per parameter, chosen by registration order."""
        names: dict[nn.Parameter, str] = {}
        for _, name, parameter in self._normalized_parameter_bindings():
            names.setdefault(parameter, name)
        return MappingProxyType(names)

    def validate(self, model: nn.Module) -> Mapping[str, str]:
        """Validate ownership and return its unambiguous model-state names.

        The returned mapping belongs to this validation pass, not a cache: a
        later refresh or parameter replacement requires validation again.
        """
        if self.topology is not None and not isinstance(
            self.topology, ParallelTopologySnapshot
        ):
            raise TypeError("Model layout requires a resolved topology snapshot")
        if self.execution is not None:
            if not isinstance(self.execution, ExecutionLayout):
                raise TypeError("Model layout execution must be an ExecutionLayout")
            self.execution.validate_model(model)
            if self.topology is not None:
                axes = tuple(
                    name for name in ("cp", "tp") if getattr(self.topology, name) > 1
                )
                if self.execution.sequence_shard_axes != axes:
                    raise ValueError(
                        "Execution sequence layout disagrees with topology"
                    )
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
        # Every state consumer relies on an unambiguous wrapper-free name,
        # including consumers which only validate ownership before construction.
        return self.state_parameter_names

    def snapshot(self) -> ModelPartLayoutSnapshot:
        """Bind a state consumer to final parameters and execution metadata."""
        return ModelPartLayoutSnapshot(
            parameters=MappingProxyType(dict(self.parameter_bindings)),
            topology=self.topology,
            execution=self.execution,
        )


@dataclass(frozen=True, slots=True, eq=False)
class ModelPartLayoutSnapshot:
    """Stable consumer scope after assembly and parameter materialization."""

    parameters: Mapping[str, nn.Parameter]
    topology: ParallelTopologySnapshot | None
    execution: ExecutionLayout | None

    def validate(self, layout: ModelPartLayout, *, consumer: str) -> None:
        current = layout.parameter_bindings
        if tuple(current) != tuple(self.parameters) or any(
            current[path] is not parameter
            for path, parameter in self.parameters.items()
        ):
            raise ValueError(
                f"{consumer} parameter ownership changed; rebuild the consumer "
                "against the final model layout"
            )
        if layout.topology != self.topology or layout.execution != self.execution:
            raise ValueError(
                f"{consumer} layout metadata changed; rebuild the consumer "
                "against the final model layout"
            )


@dataclass(frozen=True, slots=True)
class ParallelAssembly:
    """Transformed execution objects plus layouts in local model-part order."""

    execution: nn.Module
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
    if any(not isinstance(layout, ModelPartLayout) for layout in layouts):
        raise TypeError("Model layouts must be ModelPartLayout")
    if len({layout.topology for layout in layouts}) > 1:
        raise ValueError("Model parts must share the same declared topology snapshot")
    parameter_owners: dict[nn.Parameter, int] = {}
    name_owners: dict[str, int] = {}
    for index, (model, layout) in enumerate(zip(model_parts, layouts, strict=True)):
        state_names = layout.validate(model)
        for parameter in dict.fromkeys(layout.parameter_bindings.values()):
            previous = parameter_owners.setdefault(parameter, index)
            if previous != index:
                raise ValueError(
                    f"Parameter ownership overlaps across model parts {previous} "
                    f"and {index}; cross-part shared parameters are unsupported"
                )
        for name in state_names.values():
            previous = name_owners.setdefault(name, index)
            if previous != index:
                raise ValueError(
                    f"Parameter checkpoint name {name!r} collides across model "
                    f"parts {previous} and {index}"
                )
