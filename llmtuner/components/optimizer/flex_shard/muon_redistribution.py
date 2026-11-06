# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Resolve DistMuon compute layouts into redistribution plans."""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence

import torch
from torch.distributed.tensor import Shard

from llmtuner.config.optimizer_layout import BlockShard, Owned

from .muon_layout import (
    _NoRedistributionTransition,
    _ParameterComputeLayout,
    _RedistributionTransition,
)
from .optimizer_reshard_schedule import (
    _BucketPlanningContext,
    _build_dim0_shard_redistribution_plan,
    _build_whole_tensor_redistribution_plan,
    _dtensor_storage_regions,
    _ParticipantPartition,
    _RedistributionGroup,
    _RedistributionPlan,
    _require_valid_plan,
    _RouteEndpoint,
    _StorageRegionMapping,
    _TensorRegion,
    _TensorRegionRoute,
)


def _resolve_muon_redistribution_plans(
    contexts: tuple[_BucketPlanningContext[_ParameterComputeLayout], ...],
    *,
    ns_steps_by_group: Sequence[int],
) -> tuple[tuple[_RedistributionPlan | None, ...], ...]:
    """Resolve Muon compute shardings directly into transport plans."""
    cumulative_loads_by_participants: dict[tuple[int, ...], tuple[int, ...]] = {}
    specs_by_bucket = []
    unique_specs = {}
    for context in contexts:
        participants = context.group.participants
        initial_loads = cumulative_loads_by_participants.setdefault(
            participants,
            (0,) * len(participants),
        )
        owner_ranks, cumulative_loads = _assign_balanced_owner_ranks(
            context.items,
            participants=participants,
            cumulative_loads=initial_loads,
            ns_steps_by_group=ns_steps_by_group,
        )
        cumulative_loads_by_participants[participants] = cumulative_loads
        bucket_specs = []
        for layout, owner_rank in zip(context.items, owner_ranks, strict=True):
            if layout.storage_is_compute_ready:
                bucket_specs.append(None)
                continue
            spec = (
                layout.storage_layout_signature,
                tuple(layout.param.device_mesh.shape),
                layout.storage_mesh_ranks,
                layout.redistribution_storage_mesh_axis,
                context.group.participants,
                context.group.mesh_axis_participants,
                layout.compute_sharding,
                owner_rank,
            )
            bucket_specs.append(spec)
            unique_specs.setdefault(spec, (layout, context.group, owner_rank))
        specs_by_bucket.append(bucket_specs)

    plans_by_spec = {
        spec: _build_parameter_redistribution_plan(*args)
        for spec, args in unique_specs.items()
    }
    return tuple(
        tuple(None if spec is None else plans_by_spec[spec] for spec in bucket)
        for bucket in specs_by_bucket
    )


def _assign_balanced_owner_ranks(
    compute_layouts: Sequence[_ParameterComputeLayout],
    *,
    participants: tuple[int, ...],
    cumulative_loads: Sequence[int],
    ns_steps_by_group: Sequence[int],
) -> tuple[tuple[int | None, ...], tuple[int, ...]]:
    """Balance temporary compute ownership within and across ordered buckets."""
    assignments: list[int | None] = [None] * len(compute_layouts)
    candidates = tuple(
        (index, layout)
        for index, layout in enumerate(compute_layouts)
        if type(layout.compute_sharding) is Owned
    )
    candidate_partitions, updated_cumulative_loads = _balance_loads_across_partitions(
        tuple(
            (
                _estimate_muon_compute_cost(
                    layout.param.shape,
                    ns_steps_by_group[layout.group_index],
                ),
                layout.param.numel() * layout.param.element_size(),
                layout.fqn,
            )
            for _index, layout in candidates
        ),
        initial_cumulative_primary_loads=cumulative_loads,
    )
    for (index, _layout), partition in zip(
        candidates,
        candidate_partitions,
        strict=True,
    ):
        assignments[index] = participants[partition]
    return tuple(assignments), updated_cumulative_loads


def _estimate_muon_compute_cost(matrix_shape: torch.Size, ns_steps: int) -> int:
    *batch_shape, rows, columns = matrix_shape
    num_matrices = math.prod(batch_shape)
    short_dim, long_dim = sorted((rows, columns))
    # Each NS step has two s^2 * l matmuls and one s^3 matmul.
    return num_matrices * ns_steps * short_dim * short_dim * (2 * long_dim + short_dim)


def _balance_loads_across_partitions(
    loads: Sequence[tuple[int, int, str]],
    *,
    initial_cumulative_primary_loads: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Balance keyed loads with a deterministic LPT heuristic.

    Each load is ``(primary, secondary, stable_key)``. Assignments are
    partition indices aligned with those loads. Each call balances its primary
    and then secondary loads before cumulative primary load. Stable keys make
    ordering deterministic. This is not an exact partition optimum.
    """
    num_partitions = len(initial_cumulative_primary_loads)
    assignments = [0] * len(loads)
    partition_loads = [
        (0, 0, cumulative_load, partition)
        for partition, cumulative_load in enumerate(initial_cumulative_primary_loads)
    ]
    heapq.heapify(partition_loads)
    ordered_loads = sorted(
        enumerate(loads),
        key=lambda indexed_load: (
            -indexed_load[1][0],
            -indexed_load[1][1],
            indexed_load[1][2],
        ),
    )
    for load_index, (primary, secondary, _stable_key) in ordered_loads:
        (
            current_primary,
            current_secondary,
            cumulative_primary,
            partition,
        ) = partition_loads[0]
        assignments[load_index] = partition
        heapq.heapreplace(
            partition_loads,
            (
                current_primary + primary,
                current_secondary + secondary,
                cumulative_primary + primary,
                partition,
            ),
        )

    updated_cumulative_primary_loads = [0] * num_partitions
    for _primary, _secondary, cumulative_primary, partition in partition_loads:
        updated_cumulative_primary_loads[partition] = cumulative_primary
    return tuple(assignments), tuple(updated_cumulative_primary_loads)


def _build_parameter_redistribution_plan(
    compute_layout: _ParameterComputeLayout,
    group: _RedistributionGroup,
    owner_rank: int | None,
) -> _RedistributionPlan | None:
    transition = compute_layout.storage_to_compute_transition
    if isinstance(transition, _NoRedistributionTransition):
        return None
    assert isinstance(transition, _RedistributionTransition)

    group_local_storage_shape, storage_regions = _dtensor_storage_regions(
        compute_layout.param,
        group.participants,
        required_storage_mesh_axis=compute_layout.redistribution_storage_mesh_axis,
    )
    compute_sharding = compute_layout.compute_sharding
    if type(compute_sharding) is Owned:
        assert owner_rank is not None
        assert owner_rank in group.participants
        return _build_whole_tensor_redistribution_plan(
            storage_regions,
            participants=group.participants,
            compute_participants=(owner_rank,),
            logical_shape=tuple(compute_layout.param.shape),
        )

    assert owner_rank is None
    if type(compute_sharding) is Shard:
        return _build_dim0_shard_redistribution_plan(
            storage_regions,
            participants=group.participants,
            shard_participants=group.mesh_axis_participants,
            logical_shape=group_local_storage_shape,
        )

    assert type(compute_sharding) is BlockShard
    return _build_batched_matrix_redistribution_plan(
        storage_regions,
        participants=group.participants,
        shard_participants=group.mesh_axis_participants,
        storage_shape=tuple(compute_layout.param.shape),
        block_shard=compute_sharding,
    )


def _build_batched_matrix_redistribution_plan(
    storage_regions: Sequence[_StorageRegionMapping],
    *,
    participants: tuple[int, ...],
    shard_participants: tuple[int, ...],
    storage_shape: tuple[int, ...],
    block_shard: BlockShard,
) -> _RedistributionPlan:
    """Build flat routes whose destination shards contain complete matrices."""
    _require_valid_plan(
        len(storage_shape) == 2 and block_shard.dim == 0,
        "block redistribution requires flat 2D storage and row blocks",
    )
    _require_valid_plan(
        set(shard_participants) == set(participants)
        and len(shard_participants) == len(participants),
        "block shard participants must match the redistribution group",
    )

    num_matrices = block_shard.num_blocks(storage_shape[0])
    matrix_columns = storage_shape[1]
    storage_by_participant = {}
    storage_endpoints = []
    for holders, logical_region in storage_regions:
        _require_valid_plan(
            bool(holders)
            and len(logical_region.shape) == 2
            and logical_region.offsets[1] == 0
            and logical_region.shape[1] == matrix_columns,
            "matrix-batch redistribution requires row-sharded 2D storage",
        )
        storage_endpoints.append((holders, logical_region))
        for participant in holders:
            _require_valid_plan(
                participant in participants
                and participant not in storage_by_participant,
                "matrix-batch storage holders must partition participants",
            )
            storage_by_participant[participant] = logical_region

    _require_valid_plan(
        set(storage_by_participant) == set(participants),
        "matrix-batch storage must cover every participant",
    )
    storage_partitions = tuple(
        _ParticipantPartition(
            participant=participant,
            tensor_shape=storage_by_participant[participant].shape,
            logical_regions=(storage_by_participant[participant],),
        )
        for participant in participants
    )

    compute_endpoints = []
    compute_partitions = []
    shard_index_by_participant = {
        participant: index for index, participant in enumerate(shard_participants)
    }
    for participant in participants:
        local_num_matrices, matrix_offset = Shard.local_shard_size_and_offset(
            num_matrices,
            len(participants),
            shard_index_by_participant[participant],
        )
        logical_region = _TensorRegion(
            offsets=(block_shard.block_start(matrix_offset), 0),
            shape=(
                block_shard.block_start(matrix_offset + local_num_matrices)
                - block_shard.block_start(matrix_offset),
                matrix_columns,
            ),
        )
        compute_endpoints.append((participant, matrix_offset, local_num_matrices))
        compute_partitions.append(
            _ParticipantPartition(
                participant=participant,
                tensor_shape=logical_region.shape,
                logical_regions=(logical_region,),
            )
        )

    storage_to_compute_routes = []
    for source_holders, storage_region in storage_endpoints:
        storage_row_offset = storage_region.offsets[0]
        storage_row_end = storage_row_offset + storage_region.shape[0]
        for participant, matrix_offset, local_num_matrices in compute_endpoints:
            for local_matrix_index in range(local_num_matrices):
                matrix_index = matrix_offset + local_matrix_index
                matrix_row_offset = block_shard.block_start(matrix_index)
                route_row_offset = max(storage_row_offset, matrix_row_offset)
                route_row_end = min(
                    storage_row_end,
                    block_shard.block_start(matrix_index + 1),
                )
                route_rows = route_row_end - route_row_offset
                if route_rows <= 0:
                    continue

                logical_region = _TensorRegion(
                    offsets=(route_row_offset, 0),
                    shape=(route_rows, matrix_columns),
                )
                storage_tensor_region = _TensorRegion(
                    offsets=(route_row_offset - storage_row_offset, 0),
                    shape=(route_rows, matrix_columns),
                )
                compute_tensor_region = _TensorRegion(
                    offsets=(
                        route_row_offset - block_shard.block_start(matrix_offset),
                        0,
                    ),
                    shape=(route_rows, matrix_columns),
                )
                storage_to_compute_routes.append(
                    _TensorRegionRoute(
                        logical_region=logical_region,
                        source=_RouteEndpoint(storage_tensor_region, source_holders),
                        destination=_RouteEndpoint(
                            compute_tensor_region, (participant,)
                        ),
                    )
                )

    return _RedistributionPlan(
        participants=participants,
        logical_shape=storage_shape,
        storage_partitions=storage_partitions,
        compute_partitions=tuple(compute_partitions),
        storage_to_compute_routes=tuple(storage_to_compute_routes),
    )
