"""Immutable metadata plans with live bindings, resolved before model mutation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict

from torch import Tensor, device, dtype, nn, strided

from ..models.contracts import DecoderStructure, registered_submodule

__all__ = [
    "ParallelPlan",
    "PlanTarget",
    "model_bindings_from_plan",
    "snapshot_model_bindings",
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

    forward_bindings: tuple[tuple[object, object], ...] = field(
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
            forward_paths = {target.path for target in self.targets} | {""}
            object.__setattr__(
                self,
                "forward_bindings",
                tuple(
                    (
                        path,
                        getattr(module.forward, "__func__", module.forward),
                        getattr(module.forward, "__self__", None),
                    )
                    for path, module in self.module_bindings
                    if not self.targets or path in forward_paths
                ),
            )
            for path, module in self.module_bindings:
                if path == "":
                    # Absence is part of the resolved semantics too. A minimal
                    # plan without a root snapshot has no such guarantee.
                    structure = getattr(module, "decoder_structure", None)
                    if isinstance(structure, DecoderStructure):
                        object.__setattr__(self, "decoder_structure", structure)
                    break
            object.__setattr__(
                self,
                "buffer_persistence",
                tuple(
                    (path, frozenset(module._non_persistent_buffers_set))
                    for path, module in self.module_bindings
                ),
            )
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
        bound_modules = None
        decoder_structure_bound = self.module_bindings is not None and any(
            path == "" for path, _ in self.module_bindings
        )
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
            bound_modules = dict(self.module_bindings)
            for path, function, receiver in self.forward_bindings:
                forward = bound_modules[path].forward
                if (
                    getattr(forward, "__func__", forward) is not function
                    or getattr(forward, "__self__", None) is not receiver
                ):
                    raise ValueError(f"Stale plan forward binding: {path or '<root>'}")
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
        if self.parameter_bindings is not None:
            for (path, parameter), metadata in zip(
                self.parameter_bindings, self.parameter_metadata, strict=True
            ):
                if tuple(parameter.shape) != metadata[0]:
                    raise ValueError(f"Stale plan parameter shape: {path}")
                # Device, dtype, layout and trainability affect backend support,
                # shard views and ownership even when identity survives. In
                # particular, swap-based materialization preserves Parameter IDs.
                if tensor_metadata(parameter) != metadata:
                    raise ValueError(f"Stale plan parameter metadata: {path}")
        # Buffer identity also survives resize_ and .data replacement. Snapshot
        # layout, not values: legitimate in-place cache/router updates stay valid.
        if self.buffer_bindings is not None:
            for (path, buffer), metadata in zip(
                self.buffer_bindings, self.buffer_metadata, strict=True
            ):
                if tensor_metadata(buffer) != metadata:
                    raise ValueError(f"Stale plan buffer metadata: {path}")
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


def validate_pipeline_targets(
    global_plan: ParallelPlan, local_plan: ParallelPlan, model: nn.Module
) -> tuple[tuple[PlanTarget, PlanTarget], ...]:
    """Validate retained PP ownership and return global/local metadata pairs.

    Missing modules and removed-stage Identity placeholders are nonlocal targets.
    The returned global bindings are only references for semantic comparison;
    execution must consume the independently resolved local plan.
    """
    local_plan.validate_binding(model)
    label = f"Pipeline {global_plan.stage.upper()}"
    # validate_binding checked the complete module table when supplied.
    modules = dict(
        model.named_modules(remove_duplicate=False)
        if local_plan.module_bindings is None
        else local_plan.module_bindings
    )
    expected = {}
    for target in global_plan.targets:
        module = modules.get(target.path)
        if module is None or (
            isinstance(module, nn.Identity)
            and not isinstance(target.module, nn.Identity)
        ):
            continue
        key = (target.path, target.kind)
        if key in expected:
            raise ValueError(f"{label} global plan contains duplicate targets")
        expected[key] = target
    actual = {(target.path, target.kind): target for target in local_plan.targets}
    if len(actual) != len(local_plan.targets):
        raise ValueError(f"{label} local plan contains duplicate targets")
    if expected.keys() != actual.keys():
        raise ValueError(
            f"{label} target ownership changed: "
            f"missing={sorted(expected.keys() - actual.keys())}, "
            f"unexpected={sorted(actual.keys() - expected.keys())}"
        )
    return tuple((expected[key], target) for key, target in actual.items())
