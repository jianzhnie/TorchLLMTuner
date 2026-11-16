"""Read-only global-to-local ownership after pipeline partitioning.

Only retained registration paths bind copied objects. Identity placeholders are
nonlocal modules; shared storage and shared object identity remain distinct.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from weakref import ReferenceType, ref

from torch import nn

from llmtuner.parallel.plan import (
    ParallelPlan,
    model_bindings_from_plan,
    snapshot_model_bindings,
)


@dataclass(frozen=True, slots=True)
class PipelineParameterOwnership:
    """Final parameter-only PP provenance without retaining the global model.

    Global source objects are weakly referenced: a caller retaining a source
    Parameter can still resolve it, while final layouts do not keep the full
    pre-split model or its storage alive. Local bindings are advanced to final
    targets when ModelPartLayout refreshes, including split/merged parameters.
    """

    source_parameters: Mapping[int, tuple[ReferenceType[nn.Parameter], tuple[str, ...]]]
    local_parameters: Mapping[str, tuple[nn.Parameter, ...]]

    def remap_parameters(
        self, replacements: Mapping[nn.Parameter, tuple[nn.Parameter, ...]]
    ) -> PipelineParameterOwnership:
        """Advance local ownership before releasing pre-transformation owners."""
        return PipelineParameterOwnership(
            self.source_parameters,
            MappingProxyType(
                {
                    path: tuple(
                        dict.fromkeys(
                            target
                            for parameter in parameters
                            for target in replacements.get(parameter, (parameter,))
                        )
                    )
                    for path, parameters in self.local_parameters.items()
                }
            ),
        )

    def parameter_bindings(self, parameters: set[nn.Parameter]) -> set[nn.Parameter]:
        """Resolve live global identities to retained local registrations."""
        rebound = set()
        for parameter in parameters:
            source, paths = self.source_parameters[id(parameter)]
            if source() is not parameter:
                raise KeyError("Parameter is not part of this pipeline source")
            candidates = {
                tuple(map(id, self.local_parameters[path])): self.local_parameters[path]
                for path in paths
                if path in self.local_parameters
            }
            if len(candidates) > 1:
                raise ValueError(
                    f"Pipeline parameter aliases no longer share one owner: {paths}"
                )
            rebound.update(next(iter(candidates.values()), ()))
        return rebound


@dataclass(frozen=True, slots=True)
class PipelineOwnership:
    """One copied part's bindings to the unchanged global model snapshot.

    Instances borrow modules and parameters, without copying storage. Resolve
    again after a transformation; these bindings describe the PP split only.
    """

    global_plan: ParallelPlan
    local_plan: ParallelPlan
    module_paths: Mapping[nn.Module, tuple[str, ...]] = field(init=False, repr=False)
    parameter_paths: Mapping[nn.Parameter, tuple[str, ...]] = field(
        init=False, repr=False
    )
    local_modules: Mapping[str, nn.Module] = field(init=False, repr=False)
    local_parameters: Mapping[str, nn.Parameter] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        source = model_bindings_from_plan(self.global_plan)
        local = model_bindings_from_plan(self.local_plan)
        for name, bindings in (
            ("module_paths", source["module_bindings"]),
            ("parameter_paths", source["parameter_bindings"]),
        ):
            paths = {}
            for path, value in bindings:
                paths.setdefault(value, []).append(path)
            object.__setattr__(
                self,
                name,
                MappingProxyType(
                    {value: tuple(names) for value, names in paths.items()}
                ),
            )
        object.__setattr__(
            self, "local_modules", MappingProxyType(dict(local["module_bindings"]))
        )
        object.__setattr__(
            self,
            "local_parameters",
            MappingProxyType(dict(local["parameter_bindings"])),
        )

    def validate_aliases(self) -> None:
        """Reject broken or newly merged aliases among retained global paths."""
        _validate_aliases(self.global_plan, self.local_plan)

    def parameter_ownership(self) -> PipelineParameterOwnership:
        """Deliver only the parameter provenance needed after assembly."""
        return PipelineParameterOwnership(
            MappingProxyType(
                {
                    id(parameter): (ref(parameter), paths)
                    for parameter, paths in self.parameter_paths.items()
                }
            ),
            MappingProxyType(
                {
                    path: (parameter,)
                    for path, parameter in self.local_parameters.items()
                }
            ),
        )

    def module_binding(self, source: nn.Module) -> tuple[str, nn.Module] | None:
        """Bind one global module, or return None when its paths are nonlocal."""
        candidates = {
            self.local_modules[path]: path
            for path in reversed(self.module_paths[source])
            if path in self.local_modules
            and not (
                isinstance(self.local_modules[path], nn.Identity)
                and not isinstance(source, nn.Identity)
            )
        }
        if len(candidates) > 1:
            raise ValueError(
                "Pipeline module aliases no longer share one owner: "
                f"{self.module_paths[source]}"
            )
        return next(((path, module) for module, path in candidates.items()), None)

    def parameter_bindings(self, parameters: set[nn.Parameter]) -> set[nn.Parameter]:
        """Bind retained parameters by all aliases, excluding nonlocal paths."""
        rebound = set()
        for parameter in parameters:
            candidates = {
                self.local_parameters[path]
                for path in self.parameter_paths[parameter]
                if path in self.local_parameters
            }
            if len(candidates) > 1:
                raise ValueError(
                    "Pipeline parameter aliases no longer share one owner: "
                    f"{self.parameter_paths[parameter]}"
                )
            rebound.update(candidates)
        return rebound


def snapshot_pipeline_ownership(
    model: nn.Module, parts: list[nn.Module]
) -> list[PipelineOwnership]:
    """Snapshot copied parts for callers outside the full parallel assembler."""
    global_plan = ParallelPlan(
        stage="pp",
        targets=(),
        degree=1,
        implementation="deepcopy",
        **snapshot_model_bindings(model),
    )
    return [
        PipelineOwnership(
            global_plan,
            ParallelPlan(
                stage="pp",
                targets=(),
                degree=1,
                implementation="deepcopy",
                **snapshot_model_bindings(part),
            ),
        )
        for part in parts
    ]


def _validate_aliases(global_plan: ParallelPlan, local_plan: ParallelPlan) -> None:
    """Preserve module/tensor alias groups projected onto retained PP paths.

    Compare names, never global/local tensor identity: PP deep-copies parameters.
    Only kinds with ownership snapshots on both sides provide this guarantee.
    Tensor subclasses retain their own distributed storage contract.
    """

    def projected(
        groups: Iterable[tuple[str, ...]], retained: set[str]
    ) -> set[frozenset[str]]:
        result = set()
        for paths in groups:
            group = frozenset(path for path in paths if path in retained)
            if len(group) > 1:
                result.add(group)
        return result

    def identity_groups(
        bindings: Iterable[tuple[str, object]],
    ) -> tuple[tuple[str, ...], ...]:
        groups: dict[int, list[str]] = {}
        for path, value in bindings:
            groups.setdefault(id(value), []).append(path)
        return tuple(tuple(paths) for paths in groups.values() if len(paths) > 1)

    retained_storage = set()
    for kind, global_bindings, local_bindings in (
        ("parameter", global_plan.parameter_bindings, local_plan.parameter_bindings),
        ("buffer", global_plan.buffer_bindings, local_plan.buffer_bindings),
    ):
        if global_bindings is None or local_bindings is None:
            continue
        retained = {path for path, _ in global_bindings} & {
            path for path, _ in local_bindings
        }
        retained_storage.update(f"{kind}:{path}" for path in retained)

        global_aliases = (
            global_plan.shared_parameters
            if kind == "parameter"
            else identity_groups(global_bindings)
        )
        local_aliases = (
            local_plan.shared_parameters
            if kind == "parameter"
            else identity_groups(local_bindings)
        )
        if projected(global_aliases, retained) != projected(local_aliases, retained):
            raise ValueError(f"Pipeline {kind} alias ownership changed")
    if projected(global_plan.storage_aliases, retained_storage) != projected(
        local_plan.storage_aliases, retained_storage
    ):
        raise ValueError("Pipeline storage alias ownership changed")

    if (
        global_plan.module_bindings is not None
        and local_plan.module_bindings is not None
    ):
        global_modules = dict(global_plan.module_bindings)
        local_modules = dict(local_plan.module_bindings)
        retained_modules = {
            path
            for path in global_modules.keys() & local_modules.keys()
            if not (
                isinstance(local_modules[path], nn.Identity)
                and not isinstance(global_modules[path], nn.Identity)
            )
        }
        if projected(
            identity_groups(global_plan.module_bindings), retained_modules
        ) != (projected(identity_groups(local_plan.module_bindings), retained_modules)):
            raise ValueError("Pipeline module alias ownership changed")
