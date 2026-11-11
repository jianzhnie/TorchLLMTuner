"""Immutable, tensor-free plans produced before parallel model mutation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from torch import nn

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
    module_bindings: tuple[tuple[str, nn.Module], ...] | None = field(
        default=None, repr=False, compare=False
    )
    parameter_bindings: tuple[tuple[str, nn.Parameter], ...] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.stage:
            raise ValueError("Parallel plan stage must not be empty")
        if self.degree < 1:
            raise ValueError("Parallel plan degree must be positive")
        if not self.implementation:
            raise ValueError("Parallel plan implementation must not be empty")

    def count(self, kind: TargetKind) -> int:
        """Return the number of targets with one resolved role."""
        return sum(target.kind == kind for target in self.targets)

    def validate_binding(self, model: nn.Module) -> None:
        """Reject a plan whose live module references no longer match the model."""
        for kind, expected, actual in (
            ("module", self.module_bindings, model.named_modules),
            ("parameter", self.parameter_bindings, model.named_parameters),
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
        for target in self.targets:
            if (
                target.module is not None
                and model.get_submodule(target.path) is not target.module
            ):
                raise ValueError(f"Stale plan binding: {target.path}")
