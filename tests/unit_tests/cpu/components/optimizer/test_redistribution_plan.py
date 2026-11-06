"""Transport-neutral redistribution plan construction contracts."""

import pytest

from llmtuner.components.optimizer.flex_shard.muon_redistribution import (
    _build_batched_matrix_redistribution_plan,
)
from llmtuner.components.optimizer.flex_shard.optimizer_reshard_schedule import (
    _build_dim0_shard_redistribution_plan,
    _build_whole_tensor_redistribution_plan,
    _TensorRegion,
)
from llmtuner.config.optimizer_layout import BlockShard


def test_plan_builders_share_storage_partitions_and_route_regions():
    storage_regions = (
        ((0,), _TensorRegion((0, 0), (2, 3))),
        ((1,), _TensorRegion((2, 0), (2, 3))),
    )
    whole = _build_whole_tensor_redistribution_plan(
        storage_regions,
        participants=(0, 1),
        compute_participants=(0, 1),
        logical_shape=(4, 3),
    )
    sharded = _build_dim0_shard_redistribution_plan(
        storage_regions,
        participants=(0, 1),
        shard_participants=(0, 1),
        logical_shape=(4, 3),
    )

    assert whole.storage_partitions == sharded.storage_partitions
    assert [route.logical_region for route in whole.storage_to_compute_routes] == [
        region for _, region in storage_regions
    ]
    assert [route.logical_region for route in sharded.storage_to_compute_routes] == [
        region for _, region in storage_regions
    ]
    assert whole.compute_to_storage_routes == tuple(
        route.inverse() for route in whole.storage_to_compute_routes
    )


def test_plan_builders_reject_multiple_regions_per_storage_participant():
    storage_regions = (
        ((0,), _TensorRegion((0,), (1,))),
        ((0,), _TensorRegion((1,), (1,))),
        ((1,), _TensorRegion((2,), (2,))),
    )
    with pytest.raises(RuntimeError, match="multiple storage regions"):
        _build_whole_tensor_redistribution_plan(
            storage_regions,
            participants=(0, 1),
            compute_participants=(0, 1),
            logical_shape=(4,),
        )
    with pytest.raises(RuntimeError, match="multiple storage regions"):
        _build_dim0_shard_redistribution_plan(
            storage_regions,
            participants=(0, 1),
            shard_participants=(0, 1),
            logical_shape=(4,),
        )


def test_batched_matrix_routes_use_local_offsets_across_split_matrices():
    plan = _build_batched_matrix_redistribution_plan(
        (
            ((0,), _TensorRegion((0, 0), (3, 4))),
            ((1,), _TensorRegion((3, 0), (4, 4))),
        ),
        participants=(0, 1),
        shard_participants=(0, 1),
        storage_shape=(7, 4),
        block_shard=BlockShard(dim=0, block_sizes=(2, 3, 2)),
    )

    routes = plan.storage_to_compute_routes
    assert [route.logical_region.offsets for route in routes] == [
        (0, 0),
        (2, 0),
        (3, 0),
        (5, 0),
    ]
    assert [route.logical_region.shape for route in routes] == [
        (2, 4),
        (1, 4),
        (2, 4),
        (2, 4),
    ]
    assert [route.source.tensor_region.offsets for route in routes] == [
        (0, 0),
        (2, 0),
        (0, 0),
        (2, 0),
    ]
    assert [route.destination.tensor_region.offsets for route in routes] == [
        (0, 0),
        (2, 0),
        (3, 0),
        (0, 0),
    ]
