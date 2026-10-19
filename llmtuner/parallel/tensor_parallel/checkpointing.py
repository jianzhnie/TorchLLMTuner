"""Serialize physical TP slices as full tensors, including DP DTensor shards.

Model, optimizer and EMA share the same path-based layout contract. Full tensor
materialization costs memory on every TP rank; it prevents DCP treating distinct
local TP slices as replicas. Registration lives on modules, surviving FSDP's
Parameter replacement and canonical AC/compile wrapper names.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor, distribute_tensor

from ...utils.module_names import canonical_fqn
from ..expert_parallel.checkpointing import expert_shard_map

TPLayout = tuple[int, bool, dist.ProcessGroup]


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
