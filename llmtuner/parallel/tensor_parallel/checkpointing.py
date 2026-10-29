"""Serialize physical TP slices as full tensors, including DP DTensor shards.

Model, optimizer and EMA share the same path-based layout contract. Full tensor
materialization costs memory on every TP rank; it prevents DCP treating distinct
local TP slices as replicas. Registration lives on modules, surviving FSDP's
Parameter replacement and canonical AC/compile wrapper names.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor, distribute_tensor

from ...utils.module_names import canonical_fqn
from ..expert_parallel.checkpointing import EP_GROUP_ATTR, expert_shard_map

TPLayout = tuple[int, bool, dist.ProcessGroup]


def shard_disjoint_tp_ep_state(
    state: dict[str, Any], *, model_parts: list[nn.Module]
) -> dict[str, Any]:
    """Describe dense TP and expert EP chunks with disjoint parameter ownership."""
    from ..expert_parallel.checkpointing import shard_expert_state

    with _coordinate_chunk_errors(
        model_parts, "Joint TP/EP checkpoint ownership", include_expert_groups=True
    ):
        validate_disjoint_tp_ep_checkpoint_ownership(model_parts)
    # TP touches only its registered parameters. Expert entries remain in their
    # local shape until the EP transform assigns global expert coordinates.
    result = shard_tensor_parallel_state(state, model_parts=model_parts)
    return shard_expert_state(
        result, shard_map=expert_shard_map(model_parts), model_parts=model_parts
    )


def restore_disjoint_tp_ep_state(
    state: dict[str, Any],
    *,
    local_state: dict[str, Any],
    model_parts: list[nn.Module],
) -> dict[str, Any]:
    """Validate all disjoint TP/EP chunks before restoring local storage."""
    from ...components.checkpointer.chunked_tensor import ChunkedCheckpointTensor

    destinations = shard_disjoint_tp_ep_state(local_state, model_parts=model_parts)
    with _coordinate_chunk_errors(
        model_parts, "Joint TP/EP checkpoint restore", include_expert_groups=True
    ):
        pending = _validate_chunk_destinations(state, destinations)
    for destination, source in pending:
        destination.copy_(source)
    result = dict(state)
    for key, destination in destinations.items():
        if isinstance(destination, ChunkedCheckpointTensor):
            result[key] = local_state[key]
    return result


def validate_disjoint_tp_ep_checkpoint_ownership(
    model_parts: list[nn.Module],
) -> None:
    """Reject overlapping TP and EP parameter ownership before chunk creation."""
    layouts, parameters = _index_layouts(model_parts)
    experts = expert_shard_map(model_parts)
    expert_ids = {id(parameters[name]) for name in experts if name in parameters}
    overlapping = [
        name
        for name in layouts
        if name in parameters and id(parameters[name]) in expert_ids
    ]
    if overlapping:
        raise ValueError(
            "Joint checkpoint chunks require disjoint TP and EP parameters: "
            + ", ".join(sorted(overlapping))
        )


@contextmanager
def _coordinate_chunk_errors(model_parts, phase, *, include_expert_groups=False):
    """Coordinate local errors across the selected TP and EP ownership groups."""
    error = None
    try:
        yield
    except BaseException as caught:
        error = caught
    groups = {
        layout[2]
        for model in model_parts
        for module in model.modules()
        for layout in module.__dict__.get("_tp_checkpoint_layouts", {}).values()
    }
    if include_expert_groups:
        groups.update(
            group
            for model in model_parts
            for module in model.modules()
            if (group := getattr(module, EP_GROUP_ATTR, None)) is not None
        )
    groups = sorted(
        groups, key=lambda group: tuple(dist.get_process_group_ranks(group))
    )
    failed = error is not None
    # Joint owners form overlapping TP/EP groups. Repeated reports propagate
    # a failure through that component without involving unrelated PP stages.
    # The world size bounds its diameter; use the same bound on every owner.
    rounds = dist.get_world_size() if include_expert_groups and groups else 1
    for _ in range(rounds):
        for group in groups:
            reports = [None] * dist.get_world_size(group)
            dist.all_gather_object(reports, not failed, group=group)
            failed = failed or not all(reports)
    if error is not None:
        raise error
    if failed:
        raise RuntimeError(f"{phase} failed on another rank")


def _index_layouts(model_parts: list[nn.Module]):
    layouts: dict[str, TPLayout] = {}
    parameters: dict[str, torch.Tensor] = {}
    for model in model_parts:
        for path, module in model.named_modules():
            prefix = canonical_fqn(path)
            registered = module.__dict__.get("_tp_checkpoint_layouts", {})
            for name, layout in registered.items():
                name = canonical_fqn(f"{prefix}.{name}" if prefix else name)
                if name in layouts and layouts[name] != layout:
                    raise ValueError(f"Conflicting TP checkpoint layout: {name!r}")
                layouts[name] = layout
        for name, parameter in model.named_parameters(remove_duplicate=False):
            name = canonical_fqn(name)
            if name in parameters and parameters[name] is not parameter:
                raise ValueError(f"Conflicting TP checkpoint parameter: {name!r}")
            parameters[name] = parameter
    return layouts, parameters


def _entries(state: dict[str, Any], layouts, parameters):
    entries = []
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        for name, layout in layouts.items():
            if key != name and not key.startswith(f"state.{name}."):
                continue
            if key != name and value.ndim == 0:
                continue  # Optimizer step counters are replicated scalars.
            entries.append((key, layout, parameters[name]))
            break
    return sorted(entries, key=lambda entry: entry[0])


def _prepare_state(state, model_parts, *, loading: bool):
    # Discover communication ownership before validating canonical names. Even
    # a rank with invalid parameter metadata must join its peers' error reports.
    groups = sorted(
        {
            layout[2]
            for model in model_parts
            for module in model.modules()
            for layout in module.__dict__.get("_tp_checkpoint_layouts", {}).values()
        },
        key=lambda group: tuple(dist.get_process_group_ranks(group)),
    )
    local_error = None
    entries = []
    try:
        layouts, parameters = _index_layouts(model_parts)
        expert_groups = expert_shard_map(model_parts)
        parameter_names = {
            id(parameter): name for name, parameter in parameters.items()
        }
        entries = _entries(state, layouts, parameters)
        for key, (dim, packed, group), parameter in entries:
            if state[key].is_meta:
                raise ValueError(
                    f"TP checkpoint {key!r} requires materialized tensor storage"
                )
            if state[key].layout != torch.strided:
                raise ValueError(
                    f"TP checkpoint {key!r} requires a strided tensor layout"
                )
            expected = list(parameter.shape)
            expert_group = expert_groups.get(parameter_names[id(parameter)])
            if expert_group is not None:
                # EP is gathered before TP, and sliced after TP on restore.
                # Its expert axis therefore stays global during both TP passes.
                if dim == 0:
                    raise ValueError("TP and EP cannot shard the same expert axis")
                expected[0] *= dist.get_world_size(expert_group)
            if not 0 <= dim < parameter.ndim:
                raise ValueError(f"Invalid TP checkpoint dimension for {key!r}")
            if loading:
                expected[dim] *= dist.get_world_size(group)
            if tuple(state[key].shape) != tuple(expected):
                raise ValueError(
                    f"TP checkpoint {key!r} has shape {tuple(state[key].shape)}, "
                    f"expected {'global' if loading else 'local'} shape "
                    f"{tuple(expected)}. Legacy single-rank TP slices cannot "
                    "restore global weights."
                )
            if packed and parameter.shape[dim] % 2:
                raise ValueError(f"TP checkpoint {key!r} needs equal gate/up halves")
    except BaseException as error:
        local_error = error
    peer_error = None
    for group in groups:
        manifest = (
            [
                (
                    key,
                    dim,
                    packed,
                    tuple(state[key].shape),
                    str(state[key].dtype),
                    str(state[key].layout),
                    state[key].device.type,
                    isinstance(state[key], DTensor),
                    tuple(parameter.shape),
                )
                for key, (dim, packed, owner), parameter in entries
                if owner is group
            ]
            if local_error is None
            else []
        )
        reports = [None] * dist.get_world_size(group)
        dist.all_gather_object(reports, (local_error is None, manifest), group=group)
        if not all(ready for ready, _ in reports):
            peer_error = RuntimeError(
                "TP checkpoint preparation failed on another rank"
            )
        elif any(report[1] != reports[0][1] for report in reports[1:]):
            peer_error = ValueError(
                "TP checkpoint entries or layouts differ across ranks"
            )
    if local_error is not None:
        raise local_error
    if peer_error is not None:
        raise peer_error
    return entries


def gather_tensor_parallel_state(
    state: dict[str, Any], *, model_parts: list[nn.Module]
) -> dict[str, Any]:
    """Gather global TP weights and parameter-shaped optimizer/EMA state."""
    entries = _prepare_state(state, model_parts, loading=False)
    result = dict(state)
    groups = sorted(
        {layout[2] for _, layout, _ in entries},
        key=lambda group: tuple(dist.get_process_group_ranks(group)),
    )
    for group in groups:
        owned = [entry for entry in entries if entry[1][2] is group]
        for key, (dim, packed, _), _ in owned:
            value = state[key]
            if isinstance(value, DTensor):
                value = value.full_tensor()
            pieces = [
                torch.empty_like(value) for _ in range(dist.get_world_size(group))
            ]
            dist.all_gather(pieces, value.contiguous(), group=group)
            if packed:
                halves = [piece.chunk(2, dim=dim) for piece in pieces]
                pieces = [torch.cat([half[i] for half in halves], dim) for i in (0, 1)]
            result[key] = torch.cat(pieces, dim=dim)
    return result


def shard_tensor_parallel_state(
    state: dict[str, Any], *, model_parts: list[nn.Module]
) -> dict[str, Any]:
    """Describe global TP coordinates without gathering TP or DP tensor data.

    EP expert entries must already follow the existing global expert-axis
    contract. This representation is not yet the default manager save path.
    """
    entries = _prepare_state(state, model_parts, loading=False)
    with _coordinate_chunk_errors(model_parts, "TP checkpoint chunk construction"):
        result = _build_chunked_state(state, entries)
    return result


def _build_chunked_state(state, entries):
    from torch.distributed.tensor import Partial
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    from ...components.checkpointer.chunked_tensor import ChunkedCheckpointTensor

    result = dict(state)
    for key, (dim, packed, group), _ in entries:
        value = state[key]
        shape = list(value.shape)
        offsets = [0] * value.ndim
        local = value
        if isinstance(value, DTensor):
            if any(isinstance(placement, Partial) for placement in value.placements):
                raise ValueError("TP checkpoint cannot serialize partial DP values")
            _, offsets = compute_local_shape_and_global_offset(
                value.shape, value.device_mesh, value.placements
            )
            offsets = list(offsets)
            local = value._local_tensor
        world, rank = dist.get_world_size(group), dist.get_rank(group)
        width = shape[dim] // (2 if packed else 1)
        shape[dim] *= world
        chunks = []
        start, stop = offsets[dim], offsets[dim] + local.shape[dim]
        half_count = (2 if packed else 1) if local.numel() else 0
        for half in range(half_count):
            lower, upper = max(start, half * width), min(stop, (half + 1) * width)
            if lower >= upper:
                continue
            chunk_offset = list(offsets)
            chunk_offset[dim] = (
                half * width * world + rank * width + lower - half * width
            )
            chunk = local.narrow(dim, lower - start, upper - lower)
            chunks.append((chunk_offset, chunk))
        result[key] = ChunkedCheckpointTensor(
            shape, chunks, dtype=value.dtype, device=value.device
        )
    return result


def restore_sharded_tensor_parallel_state(
    state: dict[str, Any],
    *,
    local_state: dict[str, Any],
    model_parts: list[nn.Module],
) -> dict[str, Any]:
    """Copy checkpoint chunks into caller-owned local TP/DP destinations.

    Validate every local layout before copying any value. The caller supplies
    parameter-shaped destinations, including the existing EP-axis convention.
    Destination storage and DTensor placements are preserved.
    """
    from ...components.checkpointer.chunked_tensor import ChunkedCheckpointTensor

    destinations = shard_tensor_parallel_state(local_state, model_parts=model_parts)
    with _coordinate_chunk_errors(model_parts, "TP checkpoint chunk restore"):
        pending = _validate_chunk_destinations(state, destinations)
    for destination, source in pending:
        destination.copy_(source)
    result = dict(state)
    for key, destination in destinations.items():
        if isinstance(destination, ChunkedCheckpointTensor):
            result[key] = local_state[key]
    return result


def _validate_chunk_destinations(state, destinations):
    from ...components.checkpointer.chunked_tensor import ChunkedCheckpointTensor

    pending = []
    for key, destination in destinations.items():
        if not isinstance(destination, ChunkedCheckpointTensor):
            continue
        source = state.get(key)
        if not isinstance(source, ChunkedCheckpointTensor):
            raise ValueError(f"TP checkpoint {key!r} requires chunked source storage")
        source_layout = [(offset, value.shape) for offset, value in source.chunks]
        target_layout = [(offset, value.shape) for offset, value in destination.chunks]
        if source.shape != destination.shape or source_layout != target_layout:
            raise ValueError(
                f"TP checkpoint {key!r} chunk layout differs from destination"
            )
        if source.dtype != destination.dtype:
            raise ValueError(f"TP checkpoint {key!r} dtype differs from destination")
        pending.append((destination, source))
    return pending


def load_tensor_parallel_state(
    state: dict[str, Any], *, model_parts: list[nn.Module]
) -> dict[str, Any]:
    """Slice global tensors and restore each parameter's live DP layout."""
    entries = _prepare_state(state, model_parts, loading=True)
    expert_groups = expert_shard_map(model_parts)
    expert_parameters = {
        id(parameter)
        for name, parameter in _index_layouts(model_parts)[1].items()
        if name in expert_groups
    }
    result = dict(state)
    for key, (dim, packed, group), parameter in entries:
        value = state[key]
        rank, world = dist.get_rank(group), dist.get_world_size(group)
        if packed:
            value = torch.cat(
                [half.chunk(world, dim)[rank] for half in value.chunk(2, dim)], dim
            )
        else:
            value = value.chunk(world, dim)[rank]
        value = value.contiguous()
        if isinstance(parameter, DTensor) and id(parameter) not in expert_parameters:
            value = distribute_tensor(
                value, parameter.device_mesh, parameter.placements
            )
        result[key] = value
    return result
