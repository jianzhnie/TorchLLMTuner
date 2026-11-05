# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data contracts for DistMuon storage and compute layouts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn, cast

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard

from llmtuner.config.optimizer_layout import BlockShard, ComputeLayout, Owned

# ``ComputeLayout`` first lowers shard order to DTensor placements, then
# resolves each mesh axis independently, and finally keeps BlockShard explicit
# so the planner can preserve matrix boundaries.
_LoweredComputeSharding = Owned | Replicate | Shard | _StridedShard | BlockShard
_AxisComputeSharding = Replicate | Shard | _StridedShard | BlockShard
_ResolvedComputeSharding = Owned | Replicate | Shard | BlockShard


@dataclass(frozen=True, slots=True)
class _NoRedistributionTransition:
    pass


@dataclass(frozen=True, slots=True)
class _RedistributionTransition:
    pass


_StorageToComputeTransition = _NoRedistributionTransition | _RedistributionTransition


@dataclass(frozen=True, slots=True)
class _ParameterComputeLayout:
    fqn: str
    param: DTensor
    group_index: int
    storage_mesh_ranks: tuple[int, ...]
    storage_layout_signature: tuple[Any, ...]
    local_storage_signature: tuple[Any, ...]
    compute_sharding: _ResolvedComputeSharding
    storage_to_compute_transition: _StorageToComputeTransition
    resolved_compute_layout_signature: tuple[Any, ...]
    redistribution_storage_mesh_axis: int | None
    compute_dtype: torch.dtype

    @property
    def lr_reference_shape(self) -> tuple[int, ...]:
        """Return the same LR reference shape on every rank."""
        if type(self.compute_sharding) is BlockShard:
            return (sum(self.compute_sharding.block_sizes), self.param.shape[-1])
        return self.param.shape[-2:]

    @property
    def storage_is_compute_ready(self) -> bool:
        return isinstance(
            self.storage_to_compute_transition, _NoRedistributionTransition
        )


@dataclass(frozen=True, slots=True)
class _UnsupportedStoragePlacement:
    type_name: str
    representation: str


@dataclass(frozen=True, slots=True)
class _ResolvedStorageToComputeTransition:
    compute_sharding: _ResolvedComputeSharding
    storage_to_compute_transition: _StorageToComputeTransition
    resolved_compute_layout_signature: tuple[Any, ...]
    redistribution_storage_mesh_axis: int | None = None


def _validate_matrix_batch_storage_placements(
    fqn: str,
    param: DTensor,
) -> None:
    """Validate placements supported by matrix-batch storage."""
    for mesh_axis_size, placement in zip(
        param.device_mesh.shape,
        param.placements,
        strict=True,
    ):
        if mesh_axis_size == 1 or type(placement) is Replicate:
            continue
        if type(placement) is not Shard:
            raise ValueError(
                f"Muon parameter {fqn!r} with matrix-batch storage "
                "requires exact Shard or Replicate placements"
            )
        if placement.dim % param.ndim != 0:
            raise ValueError(
                f"Muon parameter {fqn!r} with matrix-batch storage "
                "requires shards along tensor dimension 0"
            )


def _row_intervals_by_mesh_axis_coordinate(
    num_rows: int,
    sharding: _AxisComputeSharding,
    *,
    mesh_axis_size: int,
) -> tuple[tuple[int, int], ...]:
    """Resolve the global rows owned by each mesh-axis coordinate.

    Placement names alone do not show whether communication is needed:
    ``Shard(0)`` and ``BlockShard(0, (R,))`` may own the same rows when shard
    boundaries align, or different rows when a storage shard splits a block.
    Comparing these intervals distinguishes a local view from redistribution.
    """
    if type(sharding) is Replicate:
        return ((0, num_rows),) * mesh_axis_size

    if type(sharding) is BlockShard:
        assert sharding.dim == 0
        num_sharding_units = sharding.num_blocks(num_rows)
    else:
        assert type(sharding) is Shard and sharding.dim == 0
        num_sharding_units = num_rows

    intervals = []
    for axis_coordinate in range(mesh_axis_size):
        local_num_units, unit_offset = Shard.local_shard_size_and_offset(
            num_sharding_units,
            mesh_axis_size,
            axis_coordinate,
        )
        if type(sharding) is BlockShard:
            start = sharding.block_start(unit_offset)
            end = sharding.block_start(unit_offset + local_num_units)
        else:
            start, end = unit_offset, unit_offset + local_num_units
        intervals.append((start, end))
    return tuple(intervals)


def _resolve_storage_to_compute_redistribution_requirement(
    fqn: str,
    param: DTensor,
    block_shard: BlockShard | None,
    target_sharding_by_storage_mesh_axis: Mapping[int, _AxisComputeSharding],
    declared_storage_mesh_axes: Sequence[int],
) -> tuple[int, ...]:
    """Compare actual storage with the target compute tensor."""
    if block_shard is None:
        changed_storage_mesh_axes = []
        mesh_axis_names = param.device_mesh.mesh_dim_names
        assert mesh_axis_names is not None
        for storage_mesh_axis in sorted(declared_storage_mesh_axes):
            source_sharding = _normalize_storage_placement(
                param.placements[storage_mesh_axis],
                ndim=param.ndim,
                mesh_axis_size=param.device_mesh.size(storage_mesh_axis),
            )
            if type(source_sharding) is _UnsupportedStoragePlacement:
                raise NotImplementedError(
                    f"Muon parameter {fqn!r} has unsupported storage placement "
                    f"{source_sharding.type_name!r} "
                    f"({source_sharding.representation}) on mesh axis "
                    f"{mesh_axis_names[storage_mesh_axis]!r}"
                )
            target_sharding = target_sharding_by_storage_mesh_axis.get(
                storage_mesh_axis
            )
            if target_sharding is not None and source_sharding != target_sharding:
                changed_storage_mesh_axes.append(storage_mesh_axis)
        redistribution_storage_mesh_axes = tuple(changed_storage_mesh_axes)
    else:
        mesh_axis_names = param.device_mesh.mesh_dim_names
        assert mesh_axis_names is not None
        active_block_shard_mesh_axes = tuple(
            storage_mesh_axis
            for storage_mesh_axis, target_sharding in (
                target_sharding_by_storage_mesh_axis.items()
            )
            if type(target_sharding) is BlockShard
            and param.device_mesh.size(storage_mesh_axis) > 1
        )
        if len(active_block_shard_mesh_axes) > 1:
            axis_names = [
                mesh_axis_names[axis] for axis in active_block_shard_mesh_axes
            ]
            raise NotImplementedError(
                f"Muon parameter {fqn!r} requests matrix-batch compute on "
                f"multiple active mesh axes {axis_names}; only one active "
                "BlockShard axis is supported"
            )

        redistribution_storage_mesh_axis = (
            active_block_shard_mesh_axes[0] if active_block_shard_mesh_axes else None
        )
        nonreplicated_other_mesh_axes = []
        for storage_mesh_axis, placement in enumerate(param.placements):
            if storage_mesh_axis == redistribution_storage_mesh_axis:
                continue
            storage_sharding = _normalize_storage_placement(
                placement,
                ndim=param.ndim,
                mesh_axis_size=param.device_mesh.size(storage_mesh_axis),
            )
            if type(storage_sharding) is not Replicate:
                nonreplicated_other_mesh_axes.append(mesh_axis_names[storage_mesh_axis])
        if nonreplicated_other_mesh_axes:
            if redistribution_storage_mesh_axis is None:
                raise NotImplementedError(
                    f"Muon parameter {fqn!r} matrix-batch compute requires an "
                    "active BlockShard target for every non-replicated storage "
                    f"mesh axis; non-replicated axes: {nonreplicated_other_mesh_axes}"
                )
            raise NotImplementedError(
                f"Muon parameter {fqn!r} matrix-batch compute along mesh "
                f"axis {mesh_axis_names[redistribution_storage_mesh_axis]!r} "
                "requires every other storage mesh axis to be replicated; "
                f"non-replicated axes: {nonreplicated_other_mesh_axes}"
            )

        if redistribution_storage_mesh_axis is None:
            redistribution_storage_mesh_axes = ()
        else:
            storage_sharding = cast(
                Replicate | Shard,
                _normalize_storage_placement(
                    param.placements[redistribution_storage_mesh_axis],
                    ndim=param.ndim,
                    mesh_axis_size=param.device_mesh.size(
                        redistribution_storage_mesh_axis
                    ),
                ),
            )
            compute_sharding = target_sharding_by_storage_mesh_axis[
                redistribution_storage_mesh_axis
            ]
            assert type(compute_sharding) is BlockShard
            mesh_axis_size = param.device_mesh.size(redistribution_storage_mesh_axis)
            storage_intervals = _row_intervals_by_mesh_axis_coordinate(
                param.shape[0],
                storage_sharding,
                mesh_axis_size=mesh_axis_size,
            )
            compute_intervals = _row_intervals_by_mesh_axis_coordinate(
                param.shape[0],
                compute_sharding,
                mesh_axis_size=mesh_axis_size,
            )
            redistribution_storage_mesh_axes = (
                ()
                if storage_intervals == compute_intervals
                else (redistribution_storage_mesh_axis,)
            )

    return redistribution_storage_mesh_axes


# A per-mesh-axis compute sharding takes three successive forms while
# ``ComputeLayout`` resolves. The layout itself declares ``Owned``,
# ``Replicate``, ``Shard``, or ``BlockShard`` on each axis it names.


def _lower_shard_order_to_strided_shards(
    param: DTensor,
    compute_layout: ComputeLayout,
    storage_axis_by_name: Mapping[str, int],
    shardings_by_storage_mesh_axis: Mapping[int, _LoweredComputeSharding],
) -> dict[int, _LoweredComputeSharding]:
    """Encode a declared shard order as DTensor placements.

    DTensor applies same-dimension shard axes in storage-mesh order, so an axis
    that the compute layout applies later than that default becomes a
    ``_StridedShard`` whose split factor is the product of the mesh sizes of the
    axes it now follows but that sit to its right in the storage mesh. Axes the
    layout declares for other mesh variants are absent here and drop out of the
    order.
    """
    lowered_shardings = dict(shardings_by_storage_mesh_axis)
    for tensor_dim, axis_names in compute_layout.shard_order_by_tensor_dim.items():
        ordered_mesh_axes = [
            storage_axis_by_name[axis_name]
            for axis_name in axis_names
            if axis_name in storage_axis_by_name
        ]
        for order_index, storage_mesh_axis in enumerate(ordered_mesh_axes):
            split_factor = math.prod(
                param.device_mesh.size(preceding_mesh_axis)
                for preceding_mesh_axis in ordered_mesh_axes[:order_index]
                if preceding_mesh_axis > storage_mesh_axis
            )
            if split_factor > 1:
                lowered_shardings[storage_mesh_axis] = _StridedShard(
                    tensor_dim, split_factor=split_factor
                )
    return lowered_shardings


def _validate_shard_order_compute_targets(
    fqn: str,
    param: DTensor,
    target_sharding_by_storage_mesh_axis: Mapping[int, _AxisComputeSharding],
    owned_storage_mesh_axes: Sequence[int],
) -> None:
    """Reject declared shard orders that DistMuon cannot lower to a transport plan.

    DistMuon supports reordering a mesh axis behind exactly one later mesh axis
    that shards the same tensor dimension, because that axis is the one whose
    storage ownership the redistribution preserves.
    """
    mesh_axis_names = param.device_mesh.mesh_dim_names
    assert mesh_axis_names is not None
    owned_axis_set = set(owned_storage_mesh_axes)
    for (
        storage_mesh_axis,
        target_sharding,
    ) in target_sharding_by_storage_mesh_axis.items():
        if type(target_sharding) is not _StridedShard:
            continue
        target_dim = _normalize_dim(target_sharding.dim, param.ndim)
        rightward_shard_axes = []
        for rightward_mesh_axis in range(
            storage_mesh_axis + 1,
            param.device_mesh.ndim,
        ):
            if rightward_mesh_axis in owned_axis_set:
                continue
            rightward_sharding = target_sharding_by_storage_mesh_axis.get(
                rightward_mesh_axis
            )
            if rightward_sharding is None:
                rightward_sharding = _normalize_storage_placement(
                    param.placements[rightward_mesh_axis],
                    ndim=param.ndim,
                    mesh_axis_size=param.device_mesh.size(rightward_mesh_axis),
                )
            if (
                type(rightward_sharding) is Shard
                and _normalize_dim(rightward_sharding.dim, param.ndim) == target_dim
            ):
                rightward_shard_axes.append(rightward_mesh_axis)

        if len(rightward_shard_axes) != 1:
            raise ValueError(
                f"Muon parameter {fqn!r} orders mesh axis "
                f"{mesh_axis_names[storage_mesh_axis]!r} after another axis on "
                f"tensor dimension {target_dim}; DistMuon requires exactly one "
                "later mesh axis to shard that dimension"
            )
        preserved_mesh_axis_size = param.device_mesh.size(rightward_shard_axes[0])
        if target_sharding.split_factor != preserved_mesh_axis_size:
            raise ValueError(
                f"Muon parameter {fqn!r} must order mesh axis "
                f"{mesh_axis_names[storage_mesh_axis]!r} directly after "
                f"{mesh_axis_names[rightward_shard_axes[0]]!r} on tensor "
                f"dimension {target_dim}; DistMuon does not support ordering it "
                "after further mesh axes"
            )


def _is_supported_orthogonal_dim0_shard_redistribution(
    *,
    ndim: int,
    block_shard: BlockShard | None,
    source_storage_placement: object,
    target_compute_sharding: _AxisComputeSharding | None,
    preserved_storage_placement: object,
) -> bool:
    if block_shard is not None or ndim < 3:
        return False
    if (
        type(source_storage_placement) is not Shard
        or type(target_compute_sharding) is not _StridedShard
        or type(preserved_storage_placement) is not Shard
    ):
        return False

    storage_dim = _normalize_dim(source_storage_placement.dim, ndim)
    compute_dim = _normalize_dim(target_compute_sharding.dim, ndim)
    preserved_dim = _normalize_dim(preserved_storage_placement.dim, ndim)
    return storage_dim == ndim - 2 and compute_dim == preserved_dim == 0


def _resolve_matrix_block_shard(
    fqn: str,
    param: DTensor,
    shardings: Mapping[int, _LoweredComputeSharding],
    owned_axes: tuple[int, ...],
) -> BlockShard | None:
    """Validate matrix-batch ownership and preserve explicit block boundaries."""
    mesh_axis_names = param.device_mesh.mesh_dim_names
    assert mesh_axis_names is not None
    # Unit mesh axes normalize to Replicate but retain their declared blocks.
    block_shards = tuple(
        sharding for sharding in shardings.values() if type(sharding) is BlockShard
    )
    if not block_shards:
        return None
    if param.ndim != 2:
        raise ValueError(
            f"Muon parameter {fqn!r} BlockShard currently requires a "
            f"2D row-concatenated parameter; got shape {tuple(param.shape)}"
        )
    normalized_dims = tuple(
        _normalize_dim(sharding.dim, param.ndim) for sharding in block_shards
    )
    if any(dim != 0 for dim in normalized_dims):
        raise ValueError(
            f"Muon parameter {fqn!r} matrix-batch BlockShard must shard "
            "tensor dimension 0"
        )
    block_sizes = block_shards[0].block_sizes
    if any(sharding.block_sizes != block_sizes for sharding in block_shards):
        raise ValueError(
            f"Muon parameter {fqn!r} must use the same BlockShard block sizes "
            "across mesh axes"
        )
    block_shard = BlockShard(dim=0, block_sizes=block_sizes)
    if block_shard.num_blocks(param.shape[0]) == 0:
        raise ValueError(f"Muon parameter {fqn!r} requires at least one matrix block")
    _validate_matrix_batch_storage_placements(fqn, param)
    if owned_axes:
        raise ValueError(
            f"Muon owned compute for parameter {fqn!r} requires a 2D matrix"
        )
    shard_axes = [
        mesh_axis_names[storage_mesh_axis]
        for storage_mesh_axis, sharding in shardings.items()
        if type(sharding) in (Shard, _StridedShard)
    ]
    if shard_axes:
        raise ValueError(
            f"Muon parameter {fqn!r} with matrix-batch compute requires "
            f"BlockShard instead of Shard on mesh axes {shard_axes}"
        )

    return block_shard


def _resolve_storage_to_compute_transition(
    fqn: str,
    param: DTensor,
    compute_layout: ComputeLayout,
) -> _ResolvedStorageToComputeTransition:
    """Validate one storage layout and resolve its concrete compute transition."""
    local = param.to_local()
    if param.ndim < 2:
        raise ValueError(
            f"Muon parameter {fqn!r} compute shape "
            f"{tuple(param.shape)} must be a matrix or matrix batch"
        )
    if torch.is_complex(param) or not local.is_contiguous():
        _raise_unsupported_layout(fqn)

    mesh_axis_names = param.device_mesh.mesh_dim_names
    if mesh_axis_names is None:
        raise ValueError(
            f"Muon parameter {fqn!r} requires a storage mesh with named axes"
        )
    storage_axis_by_name = {
        axis_name: storage_mesh_axis
        for storage_mesh_axis, axis_name in enumerate(mesh_axis_names)
    }
    applicable_compute_shardings_by_storage_mesh_axis: dict[
        int, _LoweredComputeSharding
    ] = {
        storage_axis_by_name[axis_name]: sharding
        for axis_name, sharding in compute_layout.shardings_by_mesh_axis.items()
        if axis_name in storage_axis_by_name
    }
    if not applicable_compute_shardings_by_storage_mesh_axis:
        declared_axes = sorted(compute_layout.shardings_by_mesh_axis)
        raise ValueError(
            f"Muon compute layout for parameter {fqn!r} declares no axis in "
            f"storage mesh {list(mesh_axis_names)}; declared axes: {declared_axes}"
        )
    applicable_compute_shardings_by_storage_mesh_axis = (
        _lower_shard_order_to_strided_shards(
            param,
            compute_layout,
            storage_axis_by_name,
            applicable_compute_shardings_by_storage_mesh_axis,
        )
    )

    applicable_owned_storage_mesh_axes = tuple(
        storage_mesh_axis
        for storage_mesh_axis, sharding in (
            applicable_compute_shardings_by_storage_mesh_axis.items()
        )
        if type(sharding) is Owned
    )

    block_shard = _resolve_matrix_block_shard(
        fqn,
        param,
        applicable_compute_shardings_by_storage_mesh_axis,
        applicable_owned_storage_mesh_axes,
    )

    replicated_axes = [
        mesh_axis_names[storage_mesh_axis]
        for storage_mesh_axis, sharding in (
            applicable_compute_shardings_by_storage_mesh_axis.items()
        )
        if type(sharding) is Replicate
    ]
    if replicated_axes:
        raise NotImplementedError(
            f"Muon parameter {fqn!r} requests explicit replicated compute on "
            f"mesh axes {replicated_axes}; replicated compute is not implemented"
        )

    normalized_target_sharding_by_storage_mesh_axis: dict[
        int, _AxisComputeSharding
    ] = {}
    declared_shard_dims = []
    for (
        storage_mesh_axis,
        sharding,
    ) in applicable_compute_shardings_by_storage_mesh_axis.items():
        if type(sharding) is Owned:
            continue
        placement = cast(_AxisComputeSharding, sharding)
        if type(placement) in (Shard, _StridedShard):
            declared_shard_dims.append(_normalize_dim(placement.dim, param.ndim))
        elif type(placement) is BlockShard:
            declared_shard_dims.append(0)
        target_sharding = _normalize_compute_placement(
            placement,
            ndim=param.ndim,
            mesh_axis_size=param.device_mesh.size(storage_mesh_axis),
        )
        normalized_target_sharding_by_storage_mesh_axis[storage_mesh_axis] = (
            target_sharding
        )

    _validate_shard_order_compute_targets(
        fqn,
        param,
        normalized_target_sharding_by_storage_mesh_axis,
        applicable_owned_storage_mesh_axes,
    )
    changed_storage_mesh_axes = _resolve_storage_to_compute_redistribution_requirement(
        fqn,
        param,
        block_shard,
        normalized_target_sharding_by_storage_mesh_axis,
        tuple(applicable_compute_shardings_by_storage_mesh_axis),
    )
    active_owned_storage_mesh_axes = tuple(
        storage_mesh_axis
        for storage_mesh_axis in applicable_owned_storage_mesh_axes
        if param.device_mesh.size(storage_mesh_axis) > 1
    )
    transport_mesh_axes = tuple(
        sorted(set(changed_storage_mesh_axes).union(active_owned_storage_mesh_axes))
    )
    if len(transport_mesh_axes) > 1:
        axis_names = [mesh_axis_names[axis] for axis in transport_mesh_axes]
        raise NotImplementedError(
            f"Muon parameter {fqn!r} requires compute redistribution or "
            f"owned compute on multiple mesh axes {axis_names}; multi-axis "
            "transport is not implemented"
        )

    redistribution_storage_mesh_axis = (
        transport_mesh_axes[0] if transport_mesh_axes else None
    )
    uses_supported_orthogonal_shard_redistribution = False
    if redistribution_storage_mesh_axis is not None:
        redistribution_axis_name = mesh_axis_names[redistribution_storage_mesh_axis]
        for storage_mesh_axis, placement in enumerate(param.placements):
            if storage_mesh_axis == redistribution_storage_mesh_axis:
                if type(placement) not in (Replicate, Shard):
                    raise NotImplementedError(
                        f"Muon parameter {fqn!r} cannot redistribute "
                        f"{type(placement).__name__} storage on mesh axis "
                        f"{redistribution_axis_name!r}"
                    )
            else:
                preserved_storage_sharding = _normalize_storage_placement(
                    placement,
                    ndim=param.ndim,
                    mesh_axis_size=param.device_mesh.size(storage_mesh_axis),
                )
                if type(preserved_storage_sharding) is Replicate:
                    continue
                redistribution_storage_placement = param.placements[
                    redistribution_storage_mesh_axis
                ]
                redistribution_compute_sharding = (
                    normalized_target_sharding_by_storage_mesh_axis.get(
                        redistribution_storage_mesh_axis
                    )
                )
                if (
                    not uses_supported_orthogonal_shard_redistribution
                    and _is_supported_orthogonal_dim0_shard_redistribution(
                        ndim=param.ndim,
                        block_shard=block_shard,
                        source_storage_placement=redistribution_storage_placement,
                        target_compute_sharding=redistribution_compute_sharding,
                        preserved_storage_placement=placement,
                    )
                ):
                    uses_supported_orthogonal_shard_redistribution = True
                    continue
                if (
                    type(redistribution_storage_placement) is Shard
                    and type(redistribution_compute_sharding)
                    in (Shard, _StridedShard, BlockShard)
                    and type(placement) is Shard
                ):
                    target_compute_shard = cast(
                        Shard | _StridedShard | BlockShard,
                        redistribution_compute_sharding,
                    )
                    storage_dim = _normalize_dim(
                        redistribution_storage_placement.dim, param.ndim
                    )
                    target_dim = _normalize_dim(target_compute_shard.dim, param.ndim)
                    preserved_dim = _normalize_dim(placement.dim, param.ndim)
                    if (
                        storage_dim == param.ndim - 2
                        and target_dim == preserved_dim == 0
                        and type(redistribution_compute_sharding) is Shard
                        and redistribution_storage_mesh_axis < storage_mesh_axis
                    ):
                        preserved_axis_name = mesh_axis_names[storage_mesh_axis]
                        raise ValueError(
                            f"Muon parameter {fqn!r} must declare "
                            "shard_order_by_tensor_dim={0: "
                            f"({preserved_axis_name!r}, "
                            f"{redistribution_axis_name!r})}} when preserving "
                            f"Shard(0) on mesh axis {preserved_axis_name!r}"
                        )
                    raise NotImplementedError(
                        f"Muon parameter {fqn!r} cannot redistribute storage on "
                        f"mesh axis {redistribution_axis_name!r} from "
                        f"Shard({storage_dim}) to "
                        f"{redistribution_compute_sharding!r} while "
                        f"preserving Shard({preserved_dim}) storage on mesh axis "
                        f"{mesh_axis_names[storage_mesh_axis]!r}; orthogonal-shard "
                        "redistribution is not implemented"
                    )
                raise NotImplementedError(
                    f"Muon parameter {fqn!r} cannot redistribute mesh axis "
                    f"{redistribution_axis_name!r} while storage mesh axis "
                    f"{mesh_axis_names[storage_mesh_axis]!r} has non-replicated "
                    f"placement {placement}; this implementation requires every "
                    "other storage mesh axis to be replicated"
                )

    reordered_axis_names = [
        mesh_axis_names[storage_mesh_axis]
        for storage_mesh_axis, target_sharding in (
            normalized_target_sharding_by_storage_mesh_axis.items()
        )
        if type(target_sharding) is _StridedShard
    ]
    if (
        reordered_axis_names
        and redistribution_storage_mesh_axis is not None
        and not uses_supported_orthogonal_shard_redistribution
    ):
        raise ValueError(
            f"Muon parameter {fqn!r} has an unsupported shard order on mesh "
            f"axes {reordered_axis_names}; DistMuon currently reorders a mesh "
            "axis only when a preceding redistribution axis preserves one "
            "rightward Shard axis on the same tensor dimension"
        )

    resolved_target_signature = []
    resolved_shard_dims = []
    owned_axis_set = set(applicable_owned_storage_mesh_axes)
    for storage_mesh_axis, axis_name in enumerate(mesh_axis_names):
        if storage_mesh_axis in owned_axis_set:
            target_sharding: (
                Owned | _AxisComputeSharding | _UnsupportedStoragePlacement
            ) = Owned()
        elif storage_mesh_axis in normalized_target_sharding_by_storage_mesh_axis:
            target_sharding = normalized_target_sharding_by_storage_mesh_axis[
                storage_mesh_axis
            ]
        else:
            target_sharding = _normalize_storage_placement(
                param.placements[storage_mesh_axis],
                ndim=param.ndim,
                mesh_axis_size=param.device_mesh.size(storage_mesh_axis),
            )
        if type(target_sharding) is _UnsupportedStoragePlacement:
            raise NotImplementedError(
                f"Muon parameter {fqn!r} has unsupported compute placement "
                f"{target_sharding.type_name!r} "
                f"({target_sharding.representation}) on mesh axis {axis_name!r}"
            )
        resolved_target_signature.append((axis_name, target_sharding))
        if type(target_sharding) is Shard:
            resolved_shard_dims.append(target_sharding.dim)
        elif type(target_sharding) is _StridedShard:
            resolved_shard_dims.append(target_sharding.dim)
        elif type(target_sharding) is BlockShard:
            resolved_shard_dims.append(0)

    resolved_compute_layout_signature = tuple(resolved_target_signature)
    compute_shard_dims = [*resolved_shard_dims, *declared_shard_dims]
    if applicable_owned_storage_mesh_axes and block_shard is not None:
        raise ValueError(
            f"Muon owned compute for parameter {fqn!r} requires a native "
            "matrix or matrix-batch tensor"
        )
    if active_owned_storage_mesh_axes:
        compute_sharding: _ResolvedComputeSharding = Owned()
    elif compute_shard_dims:
        if block_shard is None and param.ndim == 2:
            raise ValueError(
                f"Muon parameter {fqn!r}: 2D Muon compute cannot use Shard; "
                "use Owned() for one matrix or "
                "BlockShard(dim=0, block_sizes=(R,)) for row-concatenated matrices"
            )
        if (block_shard is None and param.ndim < 3) or any(
            shard_dim != 0 for shard_dim in compute_shard_dims
        ):
            raise ValueError(
                f"Muon sharded compute for parameter {fqn!r} requires a native "
                "matrix batch sharded only on tensor dimension 0"
            )
        compute_sharding = block_shard if block_shard is not None else Shard(0)
    elif applicable_owned_storage_mesh_axes:
        compute_sharding = Owned()
    else:
        raise ValueError(f"unsupported storage-to-compute layout for {fqn!r}")

    if redistribution_storage_mesh_axis is None:
        return _ResolvedStorageToComputeTransition(
            compute_sharding=compute_sharding,
            storage_to_compute_transition=_NoRedistributionTransition(),
            resolved_compute_layout_signature=resolved_compute_layout_signature,
        )

    return _ResolvedStorageToComputeTransition(
        compute_sharding=compute_sharding,
        storage_to_compute_transition=_RedistributionTransition(),
        resolved_compute_layout_signature=resolved_compute_layout_signature,
        redistribution_storage_mesh_axis=redistribution_storage_mesh_axis,
    )


def _raise_unsupported_layout(fqn: str) -> NoReturn:
    raise ValueError(f"unsupported storage-to-compute layout for {fqn!r}")


def _normalize_compute_placement(
    placement: _AxisComputeSharding,
    *,
    ndim: int,
    mesh_axis_size: int,
) -> _AxisComputeSharding:
    if type(placement) is Replicate:
        return Replicate()
    if type(placement) is Shard:
        normalized_dim = _normalize_dim(placement.dim, ndim)
        if mesh_axis_size == 1:
            return Replicate()
        return Shard(normalized_dim)
    if type(placement) is _StridedShard:
        normalized_dim = _normalize_dim(placement.dim, ndim)
        if mesh_axis_size == 1:
            return Replicate()
        if placement.split_factor == 1:
            return Shard(normalized_dim)
        return _StridedShard(
            normalized_dim,
            split_factor=placement.split_factor,
        )
    assert type(placement) is BlockShard
    normalized_dim = _normalize_dim(placement.dim, ndim)
    if mesh_axis_size == 1:
        return Replicate()
    return BlockShard(normalized_dim, placement.block_sizes)


def _normalize_storage_placement(
    placement: object,
    *,
    ndim: int,
    mesh_axis_size: int,
) -> Replicate | Shard | _StridedShard | _UnsupportedStoragePlacement:
    if mesh_axis_size == 1 or type(placement) is Replicate:
        return Replicate()
    if type(placement) is Shard:
        return Shard(_normalize_dim(placement.dim, ndim))
    if type(placement) is _StridedShard:
        normalized_dim = _normalize_dim(placement.dim, ndim)
        if placement.split_factor == 1:
            return Shard(normalized_dim)
        return _StridedShard(
            normalized_dim,
            split_factor=placement.split_factor,
        )
    return _UnsupportedStoragePlacement(
        type_name=type(placement).__name__,
        representation=repr(placement),
    )


def _normalize_dim(dim: int, ndim: int) -> int:
    normalized = dim if dim >= 0 else dim + ndim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"dimension {dim} is invalid for a rank-{ndim} tensor")
    return normalized
