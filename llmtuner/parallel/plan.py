"""Immutable metadata plans with live bindings, resolved before model mutation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal

from torch import Tensor, device, dtype, nn, strided

from ..models.contracts import DecoderStructure, registered_submodule

__all__ = ["ParallelPlan", "PlanTarget"]

ReplacementObserver = Callable[
    [dict[nn.Module, nn.Module], dict[nn.Parameter, tuple[nn.Parameter, ...]]],
    None,
]

TargetKind = Literal[
    "dense",
    "moe",
    "vocab",
    "attention",
    "decoder",
    "embedding",
    "norm",
    "head",
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
    degrees: tuple[int, ...] = ()
    shared_parameters: tuple[tuple[str, ...], ...] = ()
    operations: tuple[str, ...] = ()
    tp_plan: tuple[tuple[str, str], ...] = ()
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

    parameter_shapes: tuple[tuple[str, tuple[int, ...]], ...] = field(
        default=(), init=False, repr=False, compare=False
    )

    parameter_metadata: tuple[tuple[dtype, device, tuple[int, ...], int, bool], ...] = (
        field(default=(), init=False, repr=False, compare=False)
    )

    buffer_metadata: tuple[
        tuple[tuple[int, ...], dtype, device, tuple[int, ...], int, bool], ...
    ] = field(default=(), init=False, repr=False, compare=False)

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
    decoder_structure_bound: bool = field(
        default=False, init=False, repr=False, compare=False
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
        for name in ("targets", "degrees", "operations"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in (
            "shared_parameters",
            "tp_plan",
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
        if any(type(value) is not int or value < 1 for value in self.degrees):
            raise ValueError("Parallel plan degrees must be positive integers")
        if not isinstance(self.implementation, str) or not self.implementation.strip():
            raise ValueError("Parallel plan implementation must not be empty")
        object.__setattr__(self, "storage_aliases", self.storage_alias_groups())
        if self.module_bindings is not None:
            # Accessing a bound method creates a fresh method object. Snapshot
            # its function and receiver instead of that ephemeral object's ID.
            object.__setattr__(
                self,
                "forward_bindings",
                tuple(
                    (
                        getattr(module.forward, "__func__", module.forward),
                        getattr(module.forward, "__self__", None),
                    )
                    for _, module in self.module_bindings
                ),
            )
            for path, module in self.module_bindings:
                if path == "":
                    # Absence is part of the resolved semantics too. A minimal
                    # plan without a root snapshot has no such guarantee.
                    object.__setattr__(self, "decoder_structure_bound", True)
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
                tuple(
                    (
                        tuple(value.shape),
                        value.dtype,
                        value.device,
                        tuple(value.stride()),
                        value.storage_offset(),
                        value.requires_grad,
                    )
                    for _, value in self.buffer_bindings
                ),
            )
        if self.parameter_bindings is not None:
            object.__setattr__(
                self,
                "parameter_shapes",
                tuple(
                    (path, tuple(value.shape))
                    for path, value in self.parameter_bindings
                ),
            )
            object.__setattr__(
                self,
                "parameter_metadata",
                tuple(
                    (
                        value.dtype,
                        value.device,
                        tuple(value.stride()),
                        value.storage_offset(),
                        value.requires_grad,
                    )
                    for _, value in self.parameter_bindings
                ),
            )

    def count(self, kind: TargetKind) -> int:
        """Return the number of targets with one resolved role."""
        return sum(target.kind == kind for target in self.targets)

    def validate_binding(self, model: nn.Module) -> None:
        """Reject a plan whose live module references no longer match the model."""
        if self.decoder_structure_bound and (
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
            for (path, module), (function, receiver) in zip(
                self.module_bindings, self.forward_bindings, strict=True
            ):
                forward = module.forward
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
                module = registered_submodule(model, path)
            except AttributeError as error:
                raise ValueError(
                    f"Stale plan token history requirement: {path}"
                ) from error
            current = getattr(module, "requires_full_token_history", None)
            if type(current) is not type(expected) or current != expected:
                raise ValueError(f"Stale plan token history requirement: {path}")
        # Parameter identity survives `.data` reassignment. Keep immutable
        # geometry metadata so resized storage cannot reuse an old shard plan.
        if self.parameter_bindings is not None:
            for (path, parameter), (_, shape) in zip(
                self.parameter_bindings, self.parameter_shapes, strict=True
            ):
                if tuple(parameter.shape) != shape:
                    raise ValueError(f"Stale plan parameter shape: {path}")
            for (path, parameter), metadata in zip(
                self.parameter_bindings, self.parameter_metadata, strict=True
            ):
                # Device, dtype, layout and trainability affect backend support,
                # shard views and ownership even when identity survives. In
                # particular, swap-based materialization preserves Parameter IDs.
                current_metadata = (
                    parameter.dtype,
                    parameter.device,
                    tuple(parameter.stride()),
                    parameter.storage_offset(),
                    parameter.requires_grad,
                )
                if current_metadata != metadata:
                    raise ValueError(f"Stale plan parameter metadata: {path}")
        # Buffer identity also survives resize_ and .data replacement. Snapshot
        # layout, not values: legitimate in-place cache/router updates stay valid.
        if self.buffer_bindings is not None:
            for (path, buffer), metadata in zip(
                self.buffer_bindings, self.buffer_metadata, strict=True
            ):
                current_metadata = (
                    tuple(buffer.shape),
                    buffer.dtype,
                    buffer.device,
                    tuple(buffer.stride()),
                    buffer.storage_offset(),
                    buffer.requires_grad,
                )
                if current_metadata != metadata:
                    raise ValueError(f"Stale plan buffer metadata: {path}")
        if self.storage_alias_groups() != self.storage_aliases:
            raise ValueError("Stale plan storage aliases")
        for target in self.targets:
            if target.module is None:
                continue
            try:
                current = registered_submodule(model, target.path)
            except AttributeError as error:
                raise ValueError(f"Stale plan binding: {target.path}") from error
            if current is not target.module:
                raise ValueError(f"Stale plan binding: {target.path}")


def validate_pipeline_aliases(
    global_plan: ParallelPlan, local_plan: ParallelPlan
) -> None:
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

        if projected(identity_groups(global_bindings), retained) != projected(
            identity_groups(local_bindings), retained
        ):
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


def validate_pipeline_targets(
    global_plan: ParallelPlan, local_plan: ParallelPlan, model: nn.Module
) -> tuple[tuple[PlanTarget, PlanTarget], ...]:
    """Validate retained PP ownership and return global/local metadata pairs.

    Missing modules and removed-stage Identity placeholders are nonlocal targets.
    The returned global bindings are only references for semantic comparison;
    execution must consume the independently resolved local plan.
    """
    local_plan.validate_binding(model)
    validate_pipeline_aliases(global_plan, local_plan)
    label = f"Pipeline {global_plan.stage.upper()}"
    modules = dict(model.named_modules(remove_duplicate=False))
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
