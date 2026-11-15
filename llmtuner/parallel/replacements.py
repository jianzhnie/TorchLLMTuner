"""Compose explicit identity mappings emitted by parallel transformations.

Mappings weakly reference source identities and retain final targets, including
split and merged parameters. They never infer ownership from names or shapes.
One instance belongs to one local model part and one assembly attempt.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import TypeVar
from weakref import ReferenceType, ref

from torch import nn

ReplacementObserver = Callable[
    [dict[nn.Module, nn.Module], dict[nn.Parameter, tuple[nn.Parameter, ...]]],
    None,
]

_Source = TypeVar("_Source")
_Target = TypeVar("_Target")


class _WeakIdentityMapping(Mapping[_Source, _Target]):
    """Weak source lookup without invoking Tensor equality.

    WeakKeyDictionary compares weak references using referent equality, which
    is not scalar-valued for Tensor keys. Index by id and verify referent
    identity instead; callbacks guard against reusing an expired object's id.
    """

    def __init__(self, values: Mapping[_Source, _Target]) -> None:
        self._entries: dict[int, tuple[ReferenceType[_Source], _Target]] = {}
        owner = ref(self)
        for source, target in values.items():
            identity = id(source)

            def remove(source_ref, identity=identity, owner=owner):
                mapping = owner()
                if mapping is not None:
                    entry = mapping._entries.get(identity)
                    if entry is not None and entry[0] is source_ref:
                        del mapping._entries[identity]

            self._entries[identity] = (ref(source, remove), target)

    def __getitem__(self, source: _Source) -> _Target:
        entry = self._entries.get(id(source))
        if entry is None or entry[0]() is not source:
            raise KeyError("Unknown replacement source identity")
        return entry[1]

    def __iter__(self) -> Iterator[_Source]:
        for source_ref, _ in tuple(self._entries.values()):
            source = source_ref()
            if source is not None:
                yield source

    def __len__(self) -> int:
        return len(self._entries)


class ReplacementMap:
    """Track original and intermediate objects through successive conversions.

    Each event maps objects entering a transformation to its outputs. An empty
    parameter tuple represents removal. Composition deduplicates converging
    targets while preserving their first occurrence.
    Tensor storage sharing is separate from object replacement and remains
    described by the ownership snapshots in ``ParallelPlan``.
    """

    def __init__(self) -> None:
        self._module_targets: Mapping[nn.Module, nn.Module] = _WeakIdentityMapping({})
        self._parameter_targets: Mapping[nn.Parameter, tuple[nn.Parameter, ...]] = (
            _WeakIdentityMapping({})
        )

    @property
    def module_targets(self) -> Mapping[nn.Module, nn.Module]:
        """Read-only final targets for source modules still owned by callers."""
        return self._module_targets

    @property
    def parameter_targets(self) -> Mapping[nn.Parameter, tuple[nn.Parameter, ...]]:
        """Read-only final targets for source parameters still owned by callers."""
        return self._parameter_targets

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
        weak_modules = _WeakIdentityMapping(module_targets)
        weak_parameters = _WeakIdentityMapping(parameter_targets)
        self._module_targets = weak_modules
        self._parameter_targets = weak_parameters
