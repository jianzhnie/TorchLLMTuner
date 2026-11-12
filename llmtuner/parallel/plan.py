"""Immutable, tensor-free plans produced before parallel model mutation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from torch import Tensor, device, dtype, nn, strided

__all__ = ["ParallelPlan", "PlanTarget"]

ReplacementObserver = Callable[
    [dict[nn.Module, nn.Module], dict[nn.Parameter, tuple[nn.Parameter, ...]]],
    None,
]

TargetKind = Literal[
    "dense", "moe", "vocab", "attention", "decoder", "embedding", "norm", "head"
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

    @property
    def fqn(self) -> str:
        """Compatibility spelling for consumers that use fully qualified names."""
        return self.path


@dataclass(frozen=True, slots=True)
class ParallelPlan:
    """Resolved parallel work and diagnostics without tensor payloads.

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

    storage_aliases: tuple[tuple[str, ...], ...] = field(
        default=(), init=False, repr=False, compare=False
    )

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
            current = getattr(
                model.get_submodule(path), "requires_full_token_history", None
            )
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
                current = model.get_submodule(target.path)
            except AttributeError as error:
                raise ValueError(f"Stale plan binding: {target.path}") from error
            if current is not target.module:
                raise ValueError(f"Stale plan binding: {target.path}")
