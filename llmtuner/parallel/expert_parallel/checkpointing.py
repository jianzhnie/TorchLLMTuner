"""EP-aware expert-state serialization for checkpoints.

The EP swap leaves expert weights as rank-heterogeneous tensors: each rank's
``GroupedExperts`` holds only its own ``num_local = E / ep`` experts, under the
same FQN every rank reports. A checkpoint backend that treats same-FQN tensors
as replicated collapses all ranks onto one expert slice, so EP runs refused
checkpointing outright (the retired ``ep_checkpoint`` matrix row).

The representation chosen here is the full-tensor one: on save, each expert
entry is all-gathered along the expert dim across the EP group into the
complete ``(E, F, D)`` / ``(E, D, F)`` tensor (upstream's logical view, where
the expert weight is one DTensor sharded on the EP axis); on load, every rank
slices its own ``[ep_rank * num_local, (ep_rank + 1) * num_local)`` piece back
out -- the same per-rank slicing the swap itself performs. Because the stored
tensor is the complete expert weight under the original FQN, a checkpoint is
independent of the EP degree: resuming with a different ``ep`` re-slices the
same full tensor.

Two value forms are handled:

* plain tensors (EP without FSDP): the live value is the local slice; gather
  concatenates, load narrows;
* ``DTensor`` (EP with FSDP2, experts sharded on the efsdp axis): the value is
  first materialized with ``full_tensor()`` per EP rank, then gathered; on
  load the slice is re-distributed with the live parameter's own mesh and
  placements so ``load_state_dict`` can copy it into the sharded parameter.

The transform applies to any flat, FQN-keyed dict: the model wrapper's (key
``fqn``), the optimizer container's (keys ``state.{fqn}.{state_name}`` -- the
shape match against the live parameter keeps scalar entries like ``step``
untouched), and the EMA's (``state.{fqn}.ema_params``).

Known cost: every save materializes the full expert weights on every rank
(and every load plans against them). That is the price of a degree-
independent, replication-safe layout; a sharded write plan is future work.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.nn as nn

from ...accelerator.distributed import get_rank, get_world_size
from ...utils.module_names import canonical_fqn

try:  # torch >= 2.4; the FSDP path (DTensor expert weights) needs it.
    from torch.distributed.tensor import DTensor, distribute_tensor
except ImportError:  # pragma: no cover - old torch, plain-tensor path only
    DTensor = None
    distribute_tensor = None

__all__ = [
    "EP_GROUP_ATTR",
    "expert_shard_map",
    "gather_expert_state",
    "load_expert_state",
    "mark_experts_ep_sharded",
]

EP_GROUP_ATTR = "_ep_checkpoint_group"
"""Attribute the EP swap sets on a ``GroupedExperts`` whose weights are
expert-dim-sharded across that process group. Absent attribute == ep == 1 ==
nothing to transform."""


def mark_experts_ep_sharded(grouped: nn.Module, group: dist.ProcessGroup) -> None:
    """Flag ``grouped``'s direct parameters as expert-dim-sharded over ``group``."""
    setattr(grouped, EP_GROUP_ATTR, group)


def expert_shard_map(
    model_parts: list[nn.Module],
) -> dict[str, dist.ProcessGroup]:
    """Map each EP-sharded expert parameter FQN to its EP process group."""
    shard_map: dict[str, dist.ProcessGroup] = {}
    for model in model_parts:
        for module_path, module in model.named_modules():
            group = getattr(module, EP_GROUP_ATTR, None)
            if group is None:
                continue
            for name, _ in module.named_parameters(recurse=False):
                fqn = f"{module_path}.{name}" if module_path else name
                fqn = canonical_fqn(fqn)
                if fqn in shard_map and shard_map[fqn] is not group:
                    raise ValueError(
                        f"EP checkpoint FQN {fqn!r} belongs to different "
                        "expert process groups"
                    )
                shard_map[fqn] = group
    return shard_map


def index_checkpoint_parameters(
    model_parts: list[nn.Module],
) -> dict[str, torch.Tensor]:
    """Index live parameters without silently overwriting canonical FQNs."""
    params: dict[str, torch.Tensor] = {}
    for model in model_parts:
        for fqn, param in model.named_parameters():
            name = canonical_fqn(fqn)
            if name in params and params[name] is not param:
                raise ValueError(
                    f"EP checkpoint parameter FQN {name!r} identifies "
                    "different parameters across model parts or wrapper paths"
                )
            params[name] = param
    return params


def iter_expert_state_entries(
    state_dict: dict[str, Any],
    *,
    shard_map: dict[str, dist.ProcessGroup],
    params: dict[str, torch.Tensor],
    direction: Literal["save", "load"],
) -> Iterator[tuple[str, dist.ProcessGroup, torch.Tensor]]:
    """Yield ``(key, group, live_param)`` for entries holding expert state.

    An entry matches when its key is the expert FQN itself (model dicts) or a
    nested state entry under it (``state.{fqn}.{name}``, optimizer/EMA dicts).
    The shape gate is direction-aware:

    * ``direction="save"``: the live value is the local slice, so it must
      have the live parameter's shape. This keeps replicated scalars
      (``state.{fqn}.step``) and unrelated state out of the transform.
    * ``direction="load"``: the stored value is the *full* tensor, so the
      local-shape gate would match nothing. Match on rank instead: same
      ndim, same trailing dims; the leading-dim check (full vs
      ``num_local * world``) is the explicit ValueError in
      ``load_expert_state``. Scalars fall out on ndim.
    """
    for key, value in state_dict.items():
        candidates = ((key, shard_map[key]),) if key in shard_map else shard_map.items()
        for fqn, group in candidates:
            if key != fqn and not key.startswith(f"state.{fqn}."):
                continue
            param = params.get(fqn)
            if param is None or not isinstance(value, torch.Tensor):
                continue
            shape_matches = (
                value.shape == param.shape
                if direction == "save"
                else value.ndim == param.ndim and value.shape[1:] == param.shape[1:]
            )
            if not shape_matches:
                if key == fqn:
                    raise ValueError(
                        f"EP checkpoint weight {key!r} has incompatible shape "
                        f"{tuple(value.shape)} for live parameter {tuple(param.shape)} "
                        f"during {direction}"
                    )
                continue
            yield key, group, param
            break


def gather_expert_state(
    state_dict: dict[str, Any],
    *,
    shard_map: dict[str, dist.ProcessGroup],
    model_parts: list[nn.Module],
) -> dict[str, Any]:
    """Return ``state_dict`` with every expert entry all-gathered to the full tensor.

    The result presents one replicated ``(E, ...)`` tensor per expert weight,
    which every checkpoint backend can save (and plan a load against) with
    its ordinary replicated-tensor semantics.
    """
    if not shard_map:
        return state_dict
    params = index_checkpoint_parameters(model_parts)
    out = dict(state_dict)
    for key, group, _ in iter_expert_state_entries(
        state_dict, shard_map=shard_map, params=params, direction="save"
    ):
        value = out[key]
        if DTensor is not None and isinstance(value, DTensor):
            # Per-EP-rank materialization of the efsdp-sharded weight.
            value = value.full_tensor()
        world = get_world_size(group)
        gathered = value.new_empty((world * value.shape[0],) + value.shape[1:])
        dist.all_gather_into_tensor(gathered, value.contiguous(), group=group)
        out[key] = gathered
    return out


def load_expert_state(
    state_dict: dict[str, Any],
    *,
    shard_map: dict[str, dist.ProcessGroup],
    model_parts: list[nn.Module],
) -> dict[str, Any]:
    """Return ``state_dict`` with every full expert entry sliced back to this rank.

    The inverse of ``gather_expert_state``: after the backend has filled the
    full tensors, each rank keeps its own ``num_local`` experts -- the same
    slice the EP swap copied out of the HF block. A live ``DTensor`` parameter
    receives a re-distributed tensor with its own mesh and placements so the
    downstream ``load_state_dict`` copies shard-for-shard.
    """
    if not shard_map:
        return state_dict
    params = index_checkpoint_parameters(model_parts)
    out = dict(state_dict)
    for key, group, param in iter_expert_state_entries(
        state_dict, shard_map=shard_map, params=params, direction="load"
    ):
        full = out[key]
        world = get_world_size(group)
        rank = get_rank(group)
        num_local = param.shape[0]
        if full.shape[0] != num_local * world:
            raise ValueError(
                f"checkpoint entry {key!r} holds {full.shape[0]} experts, but "
                f"this rank expects {num_local} x ep={world} = "
                f"{num_local * world}. The checkpoint was saved with a "
                "different expert layout."
            )
        local = full.narrow(0, rank * num_local, num_local)
        if DTensor is not None and isinstance(param, DTensor):
            local = distribute_tensor(
                local.contiguous(), param.device_mesh, param.placements
            )
        out[key] = local
    return out
