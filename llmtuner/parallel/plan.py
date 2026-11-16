"""Immutable metadata plans with live bindings, resolved before model mutation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal, TypedDict

from torch import Tensor, device, dtype, nn, strided

from ..models.contracts import DecoderStructure, registered_submodule

__all__ = [
    "ParallelPlan",
    "PlanTarget",
    "first_paths_by_identity",
    "model_bindings_from_plan",
    "snapshot_model_bindings",
    "unique_module_bindings",
]


class FullPlanBindings(TypedDict):
    module_bindings: tuple[tuple[str, nn.Module], ...]
    parameter_bindings: tuple[tuple[str, nn.Parameter], ...]
    buffer_bindings: tuple[tuple[str, Tensor], ...]


def snapshot_model_bindings(model: nn.Module) -> FullPlanBindings:
    """Capture registered ownership paths without modifying tensors or modules."""
    return {
        "module_bindings": tuple(model.named_modules(remove_duplicate=False)),
        "parameter_bindings": tuple(model.named_parameters(remove_duplicate=False)),
        "buffer_bindings": tuple(model.named_buffers(remove_duplicate=False)),
    }


def first_paths_by_identity(
    module_bindings: tuple[tuple[str, nn.Module], ...],
) -> dict[int, str]:
    """Return one stable registered path per module object.

    Aliased modules have several names, but resolution needs one owner path.
    Keeping this rule in the binding layer prevents each resolver from
    rebuilding subtly different alias indexes.
    """
    paths: dict[int, str] = {}
    for path, module in module_bindings:
        paths.setdefault(id(module), path)
    return paths


def unique_module_bindings(
    bindings: Iterable[tuple[str, nn.Module]],
) -> tuple[tuple[str, nn.Module], ...]:
    """Keep one execution target per module, using its first registered path."""
    modules: dict[int, tuple[str, nn.Module]] = {}
    for path, module in bindings:
        modules.setdefault(id(module), (path, module))
    return tuple(modules.values())


def model_bindings_from_plan(plan: ParallelPlan) -> FullPlanBindings:
    """Reuse a complete ownership snapshot without another model traversal."""
    if (
        plan.module_bindings is None
        or plan.parameter_bindings is None
        or plan.buffer_bindings is None
    ):
        raise ValueError("Plan does not contain complete model bindings")
    return {
        "module_bindings": plan.module_bindings,
        "parameter_bindings": plan.parameter_bindings,
        "buffer_bindings": plan.buffer_bindings,
    }


TargetKind = Literal[
    "dense",
    "moe",
    "vocab",
    "attention",
    "decoder",
    "fsdp_unit",
]


@dataclass(frozen=True, slots=True)
class PlanTarget:
    """One resolved module path and its role, with an optional live binding."""

    path: str
    kind: TargetKind
    module: nn.Module | None = None
    layout: str = ""
    spec: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        # The empty FQN denotes the root; all other targets are concrete module
        # paths, never user patterns waiting to be resolved during execution.
        if not isinstance(self.path, str) or (
            self.path and any(not part.strip() for part in self.path.split("."))
        ):
            raise ValueError("Plan target must have a concrete module path")
        if self.module is not None and not isinstance(self.module, nn.Module):
            raise ValueError("Plan target binding must be a module")


TensorMetadata = tuple[tuple[int, ...], dtype, device, tuple[int, ...], int, bool]


def tensor_metadata(value: Tensor) -> TensorMetadata:
    return (
        tuple(value.shape),
        value.dtype,
        value.device,
        tuple(value.stride()),
        value.storage_offset(),
        value.requires_grad,
    )


@dataclass(frozen=True, slots=True)
class ParallelPlan:
    """Resolved work and diagnostics without copying parameter or buffer storage.

    Optional ownership and TP fields are added only by consumers that need them;
    the minimal TP plan does not have to manufacture placeholder capabilities.
    """

    stage: str
    targets: tuple[PlanTarget, ...]
    degree: int
    implementation: str
    shared_parameters: tuple[tuple[str, ...], ...] = field(
        default=(), init=False, repr=False, compare=False
    )
    token_history_requirements: tuple[tuple[str, bool | None], ...] = ()
    module_bindings: tuple[tuple[str, nn.Module], ...] | None = field(
        default=None, repr=False, compare=False
    )
    parameter_bindings: tuple[tuple[str, nn.Parameter], ...] | None = field(
        default=None, repr=False, compare=False
    )

    buffer_bindings: tuple[tuple[str, Tensor], ...] | None = field(
        default=None, repr=False, compare=False
    )

    parameter_metadata: tuple[TensorMetadata, ...] = field(
        default=(), init=False, repr=False, compare=False
    )

    buffer_metadata: tuple[TensorMetadata, ...] = field(
        default=(), init=False, repr=False, compare=False
    )

    buffer_persistence: tuple[tuple[str, frozenset[str]], ...] = field(
        default=(), init=False, repr=False, compare=False
    )

    forward_bindings: tuple[tuple[str, object, object, object], ...] = field(
        default=(), init=False, repr=False, compare=False
    )

    storage_aliases: tuple[tuple[str, ...], ...] = field(
        default=(), init=False, repr=False, compare=False
    )
    decoder_structure: DecoderStructure | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def parameter_alias_groups(self) -> tuple[tuple[str, ...], ...]:
        """Return every registered path sharing the same Parameter object."""
        groups: dict[int, list[str]] = {}
        for path, parameter in self.parameter_bindings or ():
            groups.setdefault(id(parameter), []).append(path)
        return tuple(tuple(paths) for paths in groups.values() if len(paths) > 1)

    def storage_alias_groups(self) -> tuple[tuple[str, ...], ...]:
        """Describe ordinary strided storage sharing without retaining storage.

        Storage handles work for meta tensors as well as real allocations.
        Tensor subclasses own their distributed storage semantics; do not
        inspect their local shards through private attributes here.
        """
        groups: dict[int, list[str]] = {}
        for kind, bindings in (
            ("parameter", self.parameter_bindings),
            ("buffer", self.buffer_bindings),
        ):
            for path, value in bindings or ():
                if type(value) not in (Tensor, nn.Parameter) or value.layout != strided:
                    continue
                groups.setdefault(value.untyped_storage()._cdata, []).append(
                    f"{kind}:{path}"
                )
        return tuple(tuple(paths) for paths in groups.values() if len(paths) > 1)

    def __post_init__(self) -> None:
        # Frozen attributes do not freeze caller-owned lists. Snapshot plan
        # containers before deriving ownership metadata; retain tensor identity.
        object.__setattr__(self, "targets", tuple(self.targets))
        for name in (
            "token_history_requirements",
            "module_bindings",
            "parameter_bindings",
            "buffer_bindings",
        ):
            values = getattr(self, name)
            if values is not None:
                object.__setattr__(self, name, tuple(tuple(value) for value in values))
        if not isinstance(self.stage, str) or not self.stage.strip():
            raise ValueError("Parallel plan stage must not be empty")
        if type(self.degree) is not int or self.degree < 1:
            raise ValueError("Parallel plan degree must be a positive integer")
        if not isinstance(self.implementation, str) or not self.implementation.strip():
            raise ValueError("Parallel plan implementation must not be empty")
        object.__setattr__(self, "shared_parameters", self.parameter_alias_groups())
        object.__setattr__(self, "storage_aliases", self.storage_alias_groups())
        if self.module_bindings is not None:
            # Accessing a bound method creates a fresh method object. Snapshot
            # its function and receiver instead of that ephemeral object's ID.
            # A target can delegate its math to nested modules. The complete
            # ownership snapshot must protect their execution bindings too.
            forward_bindings = []
            buffer_persistence = []
            for path, module in self.module_bindings:
                forward = module.forward
                forward_bindings.append(
                    (
                        path,
                        getattr(forward, "__func__", forward),
                        getattr(forward, "__self__", None),
                        module._compiled_call_impl,
                    )
                )
                buffer_persistence.append(
                    (path, frozenset(module._non_persistent_buffers_set))
                )
            object.__setattr__(self, "forward_bindings", tuple(forward_bindings))
            object.__setattr__(self, "buffer_persistence", tuple(buffer_persistence))
            for path, module in self.module_bindings:
                if path == "":
                    # Absence is part of the resolved semantics too. A minimal
                    # plan without a root snapshot has no such guarantee.
                    structure = getattr(module, "decoder_structure", None)
                    if isinstance(structure, DecoderStructure):
                        object.__setattr__(self, "decoder_structure", structure)
                    break
        if self.buffer_bindings is not None:
            object.__setattr__(
                self,
                "buffer_metadata",
                tuple(tensor_metadata(value) for _, value in self.buffer_bindings),
            )
        if self.parameter_bindings is not None:
            object.__setattr__(
                self,
                "parameter_metadata",
                tuple(tensor_metadata(value) for _, value in self.parameter_bindings),
            )

    def count(self, kind: TargetKind) -> int:
        """Return the number of targets with one resolved role."""
        return sum(target.kind == kind for target in self.targets)

    def validate_binding(self, model: nn.Module) -> None:
        """Reject a plan whose live module references no longer match the model."""
        bound_modules = (
            dict(self.module_bindings) if self.module_bindings is not None else None
        )
        decoder_structure_bound = bound_modules is not None and "" in bound_modules
        if decoder_structure_bound and (
            getattr(model, "decoder_structure", None) != self.decoder_structure
        ):
            # Registered modules can stay identical while model-owned paths
            # change. Both TP and EP must reject that stale structural meaning.
            raise ValueError("Stale plan decoder structure")
        for kind, expected, actual in (
            ("module", self.module_bindings, model.named_modules),
            ("parameter", self.parameter_bindings, model.named_parameters),
            ("buffer", self.buffer_bindings, model.named_buffers),
        ):
            if expected is None:
                continue
            current = tuple(actual(remove_duplicate=False))
            if len(current) != len(expected):
                raise ValueError(f"Stale plan {kind} ownership")
            for (path, value), (bound_path, bound_value) in zip(
                current, expected, strict=True
            ):
                if path != bound_path or value is not bound_value:
                    raise ValueError(f"Stale plan {kind} binding: {path}")
        if self.module_bindings is not None:
            for path, function, receiver, compiled_call in self.forward_bindings:
                module = bound_modules[path]
                forward = module.forward
                if (
                    getattr(forward, "__func__", forward) is not function
                    or getattr(forward, "__self__", None) is not receiver
                ):
                    raise ValueError(f"Stale plan forward binding: {path or '<root>'}")
                # Module.compile changes __call__ without replacing forward or
                # parameter identities. An existing plan must not bypass a new
                # compiled graph or reuse one after it has been removed.
                if module._compiled_call_impl is not compiled_call:
                    raise ValueError(
                        f"Stale plan compiled call binding: {path or '<root>'}"
                    )
            # register_buffer can change checkpoint ownership without replacing
            # its tensor or changing any tensor geometry metadata.
            for (path, module), (_, nonpersistent) in zip(
                self.module_bindings, self.buffer_persistence, strict=True
            ):
                if module._non_persistent_buffers_set != nonpersistent:
                    raise ValueError(
                        f"Stale plan buffer persistence: {path or '<root>'}"
                    )
        for path, expected in self.token_history_requirements:
            try:
                module = (
                    bound_modules[path]
                    if bound_modules is not None
                    else registered_submodule(model, path)
                )
            except (AttributeError, KeyError) as error:
                raise ValueError(
                    f"Stale plan token history requirement: {path}"
                ) from error
            current = getattr(module, "requires_full_token_history", None)
            if type(current) is not type(expected) or current != expected:
                raise ValueError(f"Stale plan token history requirement: {path}")
        # Parameter identity survives `.data` reassignment. Keep immutable
        # geometry metadata so resized storage cannot reuse an old shard plan.
        # Buffer identity also survives resize_ and .data replacement. Snapshot
        # layout, not values: legitimate in-place cache/router updates stay valid.
        # Device, dtype, layout and trainability matter even when swap-based
        # materialization preserves tensor identities.
        for kind, bindings, metadata in (
            ("parameter", self.parameter_bindings, self.parameter_metadata),
            ("buffer", self.buffer_bindings, self.buffer_metadata),
        ):
            if bindings is None:
                continue
            for (path, value), expected in zip(bindings, metadata, strict=True):
                if kind == "parameter" and tuple(value.shape) != expected[0]:
                    raise ValueError(f"Stale plan parameter shape: {path}")
                if tensor_metadata(value) != expected:
                    raise ValueError(f"Stale plan {kind} metadata: {path}")
        if self.storage_alias_groups() != self.storage_aliases:
            raise ValueError("Stale plan storage aliases")
        for target in self.targets:
            if target.module is None:
                continue
            try:
                current = (
                    bound_modules[target.path]
                    if bound_modules is not None
                    else registered_submodule(model, target.path)
                )
            except (AttributeError, KeyError) as error:
                raise ValueError(f"Stale plan binding: {target.path}") from error
            if current is not target.module:
                raise ValueError(f"Stale plan binding: {target.path}")
