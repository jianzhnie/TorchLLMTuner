"""Transport-neutral redistribution plan construction contracts."""

import pytest
import torch

from llmtuner.components.optimizer.flex_shard._optimizer_reshard_schedule import (
    _build_packed_all_to_all_schedule,
    _build_region_redistribution_plan,
    _RedistributionDirection,
    _TensorRegion,
)


def test_plan_builders_share_storage_partitions_and_route_regions():
    storage_regions = (
        ((0,), _TensorRegion((0, 0), (2, 3))),
        ((1,), _TensorRegion((2, 0), (2, 3))),
    )
    whole = _build_region_redistribution_plan(
        storage_regions,
        ((0, _TensorRegion((0, 0), (4, 3))),),
        participants=(0, 1),
        logical_shape=(4, 3),
    )
    sharded = _build_region_redistribution_plan(
        storage_regions,
        tuple((holders[0], region) for holders, region in storage_regions),
        participants=(0, 1),
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
        _build_region_redistribution_plan(
            storage_regions,
            ((0, _TensorRegion((0,), (4,))),),
            participants=(0, 1),
            logical_shape=(4,),
        )


def test_batched_matrix_routes_use_local_offsets_across_split_matrices():
    plan = _build_region_redistribution_plan(
        (
            ((0,), _TensorRegion((0, 0), (3, 4))),
            ((1,), _TensorRegion((3, 0), (4, 4))),
        ),
        (
            (0, _TensorRegion((0, 0), (2, 4))),
            (0, _TensorRegion((2, 0), (3, 4))),
            (1, _TensorRegion((5, 0), (2, 4))),
        ),
        participants=(0, 1),
        logical_shape=(7, 4),
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


@pytest.mark.parametrize("direction", list(_RedistributionDirection))
def test_packed_schedule_matches_peer_splits_and_preserves_region_offsets(direction):
    plan = _build_region_redistribution_plan(
        (
            ((2,), _TensorRegion((0, 0), (3, 4))),
            ((7,), _TensorRegion((3, 0), (4, 4))),
        ),
        (
            (2, _TensorRegion((0, 0), (2, 4))),
            (7, _TensorRegion((2, 0), (3, 4))),
            (2, _TensorRegion((5, 0), (2, 4))),
        ),
        participants=(2, 7),
        logical_shape=(7, 4),
    )
    assert plan.compute_partitions[0].tensor_shape == (4, 4)
    assert len(plan.compute_partitions[0].logical_regions) == 2
    schedules = [
        _build_packed_all_to_all_schedule(
            (plan,), direction=direction, process_group=None, local_participant=rank
        )
        for rank in plan.participants
    ]
    for source, sender in enumerate(schedules):
        assert sender.has_remote_transfers
        for destination, receiver in enumerate(schedules):
            assert (
                sender.input_split_sizes[destination]
                == (receiver.output_split_sizes[source])
            )
        assert sum(sender.input_split_sizes) == sum(
            span.region.numel for span in sender.input_spans_by_parameter[0]
        )
    assert sum(sum(schedule.input_split_sizes) for schedule in schedules) == 28

    # Replay the packed exchange without a process group, checking physical
    # offsets as well as matching peer counts in each direction.
    logical = torch.arange(28).reshape(7, 4)
    storage = (logical[:3], logical[3:])
    compute = (torch.cat((logical[:2], logical[5:])), logical[2:5])
    inputs, expected = (
        (storage, compute)
        if direction is _RedistributionDirection.STORAGE_TO_COMPUTE
        else (compute, storage)
    )
    packed = []
    outputs = [torch.empty_like(tensor) for tensor in expected]
    for tensor, schedule in zip(inputs, schedules, strict=True):
        buffer = torch.empty(sum(schedule.input_split_sizes), dtype=tensor.dtype)
        for span in schedule.input_spans_by_parameter[0]:
            slices = tuple(
                slice(offset, offset + size)
                for offset, size in zip(
                    span.region.offsets, span.region.shape, strict=True
                )
            )
            buffer[span.buffer_offset : span.buffer_offset + span.numel] = tensor[
                slices
            ].flatten()
        packed.append(buffer.split(schedule.input_split_sizes))
    for destination, schedule in enumerate(schedules):
        received = torch.cat([buffer[destination] for buffer in packed])
        for span in schedule.output_spans_by_parameter[0]:
            slices = tuple(
                slice(offset, offset + size)
                for offset, size in zip(
                    span.region.offsets, span.region.shape, strict=True
                )
            )
            outputs[destination][slices] = received[
                span.buffer_offset : span.buffer_offset + span.numel
            ].reshape(span.region.shape)
        torch.testing.assert_close(outputs[destination], expected[destination])
