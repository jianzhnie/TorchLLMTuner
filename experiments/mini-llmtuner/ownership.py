"""Parameter provenance and rank-local ownership without retaining source storage.

Partition ranges describe each stage's input coordinates: TP → EP → FSDP.
Packed gate/up uses two ranges, not one incorrect contiguous half. This is
runtime layout metadata; it does not implement checkpoint resharding.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from weakref import ReferenceType, ref

import torch
from torch import nn
from torch.distributed.tensor import DTensor, Shard


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """One original parameter, all registered aliases and its planned axes."""

    names: tuple[str, ...]
    source_id: int
    shape: tuple[int, ...]
    requires_grad: bool
    source: ReferenceType = field(repr=False, compare=False)
    tp_axis: int | None = None
    packed: bool = False
    ep_axis: int | None = None
    tp_gradient_sum: bool = False


@dataclass(frozen=True, slots=True)
class Partition:
    """A shard of the previous stage's tensor; ranges are half-open."""

    stage: str
    axis: int
    degree: int
    rank: int
    ranges: tuple[tuple[int, int], ...]
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]

    def select(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the recorded shard to a reference tensor for layout validation."""
        if tuple(value.shape) != self.input_shape:
            raise ValueError(f"{self.stage} ownership input shape differs")
        return torch.cat(
            [
                value.narrow(self.axis, start, stop - start)
                for start, stop in self.ranges
            ],
            dim=self.axis,
        )


def partition(
    stage: str,
    shape: tuple[int, ...],
    axis: int,
    degree: int,
    rank: int,
    *,
    packed: bool = False,
) -> Partition:
    """Describe chunk-based sharding, including uneven and empty FSDP shards."""
    if degree < 1 or not 0 <= rank < degree:
        raise ValueError("Invalid partition rank or degree")
    halves = 2 if packed else 1
    if shape[axis] % halves:
        raise ValueError("Packed ownership requires two equal halves")
    width = shape[axis] // halves
    chunk = (width + degree - 1) // degree
    begin, end = min(rank * chunk, width), min((rank + 1) * chunk, width)
    ranges = tuple(
        (offset * width + begin, offset * width + end) for offset in range(halves)
    )
    output = list(shape)
    output[axis] = (end - begin) * halves
    return Partition(stage, axis, degree, rank, ranges, shape, tuple(output))


@dataclass(frozen=True, slots=True)
class ParameterMapping:
    """Original identity → final runtime parameter, with explicit alias ownership.

    The source is weak: replaced full-size parameters may be freed after
    assembly. The final target is already owned by the model; no clone is kept.
    """

    source: ParameterSpec
    target: nn.Parameter = field(repr=False, compare=False)
    partitions: tuple[Partition, ...]
    tp_replicas: int
    ep_replicas: int
    fsdp_degree: int

    @property
    def canonical_name(self) -> str:
        return self.source.names[0]

    @property
    def local_shape(self) -> tuple[int, ...]:
        return (
            self.partitions[-1].output_shape if self.partitions else self.source.shape
        )

    def select_local(self, original: torch.Tensor) -> torch.Tensor:
        """Project a global reference using the recorded stage layouts."""
        if tuple(original.shape) != self.source.shape:
            raise ValueError("Original parameter geometry differs from mapping")
        for entry in self.partitions:
            original = entry.select(original)
        return original


def resolve_parameter_specs(
    model: nn.Module,
    rules: Iterable[tuple[str, str]],
    targets: Iterable[tuple[str, str, str]],
    *,
    tp: int,
    ep: int,
    fsdp: int,
    gradient_sums: tuple[str, ...] = (),
) -> tuple[ParameterSpec, ...]:
    """Bind axes from explicit declarations and reject unpreservable aliases.

    No allocation of model weights, mutation, process groups or RNG operations.
    Sparse expert field names belong to the declared fused_gate_up layout.
    """
    named = dict(model.named_parameters(remove_duplicate=False))
    layouts: dict[str, tuple[int, bool]] = {}
    for path, layout in rules:
        if tp == 1 or layout in (
            "replicated",
            "embedding",
            "embedding_rowwise",
            "replicated_with_grad_allreduce",
        ):
            continue
        if layout not in ("colwise", "rowwise", "packed_colwise"):
            raise ValueError(f"Unsupported TP layout {layout!r} at {path}")
        if path in named:
            layouts[path] = (
                1 if layout == "packed_colwise" else 2,
                layout == "packed_colwise",
            )
        else:
            module = model.get_submodule(path)
            if not isinstance(module, nn.Linear):
                raise ValueError(
                    f"TP target {path} must be Linear or declared parameter"
                )
            layouts[f"{path}.weight"] = (0 if layout == "colwise" else 1, False)
            if module.bias is not None and layout == "colwise":
                layouts[f"{path}.bias"] = (0, False)
    ep_names = set()
    for path, role, layout in targets:
        if role == "moe" and ep > 1:
            if layout != "fused_gate_up":
                raise ValueError(f"Unsupported EP ownership layout {layout!r}")
            ep_names.update(
                f"{path}.experts.{name}" for name in ("gate_up_proj", "down_proj")
            )
    missing = (set(layouts) | ep_names | set(gradient_sums)) - named.keys()
    if missing:
        raise ValueError(f"Unbound parameter ownership: {sorted(missing)}")
    groups: dict[int, list[str]] = {}
    for name, value in named.items():
        groups.setdefault(id(value), []).append(name)
    specs = []
    storages: dict[int, int] = {}
    for names in groups.values():
        value = named[names[0]]
        transformed = any(name in layouts or name in ep_names for name in names)
        if len(names) > 1 and transformed:
            raise ValueError(
                f"Sharded parameter aliases need an explicit adapter: {names}"
            )
        # FSDP/TP/EP create new storage and cannot preserve distinct Parameter
        # objects backed by one storage. Buffer aliases need the same refusal.
        if tp * ep * fsdp > 1 and type(value) is nn.Parameter:
            storage = value.untyped_storage()._cdata
            previous = storages.setdefault(storage, id(value))
            if previous != id(value):
                raise ValueError(
                    "Parallel assembly cannot preserve parameter storage aliases"
                )
        axis, packed = layouts.get(names[0], (None, False))
        specs.append(
            ParameterSpec(
                tuple(names),
                id(value),
                tuple(value.shape),
                value.requires_grad,
                ref(value),
                axis,
                packed,
                0 if names[0] in ep_names else None,
                any(name in gradient_sums for name in names),
            )
        )
    if tp * ep * fsdp > 1:
        for _, buffer in model.named_buffers(remove_duplicate=False):
            if buffer.untyped_storage()._cdata in storages:
                raise ValueError(
                    "Parallel assembly cannot preserve parameter/buffer storage aliases"
                )
    return tuple(specs)


def bind_parameter_mappings(
    model: nn.Module,
    specs: tuple[ParameterSpec, ...],
    *,
    tp: int,
    ep: int,
    fsdp: int,
    tp_rank: int,
    ep_rank: int,
) -> tuple[ParameterMapping, ...]:
    """Validate the executed layouts and bind final targets, never recreate them."""
    named = dict(model.named_parameters(remove_duplicate=False))
    if set(named) != {name for spec in specs for name in spec.names}:
        raise ValueError("Assembly changed parameter FQNs without a state adapter")
    mappings = []
    used: set[int] = set()
    for spec in specs:
        target = named[spec.names[0]]
        if any(named[name] is not target for name in spec.names):
            raise ValueError(f"Assembly broke parameter aliases: {spec.names}")
        if id(target) in used:
            raise ValueError("Assembly merged distinct source parameters")
        used.add(id(target))
        if target.requires_grad != spec.requires_grad:
            raise ValueError("Assembly changed parameter trainability")
        shape, partitions = spec.shape, []
        for stage, axis, degree, rank in (
            ("tp", spec.tp_axis, tp, tp_rank),
            ("ep", spec.ep_axis, ep, ep_rank),
        ):
            if axis is not None:
                entry = partition(
                    stage,
                    shape,
                    axis,
                    degree,
                    rank,
                    packed=stage == "tp" and spec.packed,
                )
                partitions.append(entry)
                shape = entry.output_shape
        if tuple(target.shape) != shape:
            raise ValueError(f"Executed parameter layout differs at {spec.names[0]}")
        if fsdp > 1:
            if not isinstance(target, DTensor) or len(target.placements) != 1:
                raise ValueError("FSDP ownership requires a one-dimensional DTensor")
            placement = target.placements[0]
            if not isinstance(placement, Shard) or target.device_mesh.size() != fsdp:
                raise ValueError("FSDP ownership requires the declared shard mesh")
            entry = partition(
                "fsdp", shape, placement.dim, fsdp, target.device_mesh.get_local_rank()
            )
            partitions.append(entry)
            if tuple(target.to_local().shape) != entry.output_shape:
                raise ValueError("FSDP local shape differs from ownership")
        mappings.append(
            ParameterMapping(
                spec,
                target,
                tuple(partitions),
                1 if spec.tp_axis is not None else tp,
                1 if spec.ep_axis is not None else ep,
                fsdp,
            )
        )
    return tuple(mappings)
