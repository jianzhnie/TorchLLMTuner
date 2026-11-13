"""Hook-owned FSDP buffers with PyTorch 2.14 collective semantics.

Adapts FSDPTurbo fine_grained_fully_shard/param_group.py. Native parameter
groups still perform dtype conversion, collectives, accumulation and offload;
this extension releases other hooks' retained buffers after their events.
Group initialization follows PyTorch 2.14 _fsdp_init.py::_init_param_group,
using this subclass without replacing existing objects or monkeypatching torch.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple
from weakref import ReferenceType

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy
from torch.distributed.fsdp._fully_shard._fsdp_collectives import AllGatherResult
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    DataParallelMeshInfo,
    FSDPMeshInfo,
    HSDPMeshInfo,
    ShardPlacementFnResult,
    TrainingState,
    _dynamo_disable,
    resolve_shard_placement,
)
from torch.distributed.fsdp._fully_shard._fsdp_init import (
    _get_post_forward_mesh_info,
)
from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
    FSDPCommContext,
    FSDPParamGroup,
)

if TYPE_CHECKING:
    from .fine_grained_state import FineGrainedFSDPState


class HookAllGatherState(NamedTuple):
    all_gather_result: AllGatherResult
    event: torch.Event | None
    hook_module: ReferenceType[nn.Module]


class HookReduceScatterState(NamedTuple):
    reduce_scatter_input: torch.Tensor
    event: torch.Event | None
    hook_module: ReferenceType[nn.Module]


class FineGrainedFSDPParamGroup(FSDPParamGroup):
    """Release foreign-hook buffers while retaining same-hook overlap."""

    hook_module: ReferenceType[nn.Module]
    communication_contexts: list[FSDPCommContext]

    def _reset_iter_state(self) -> None:
        super()._reset_iter_state()
        # Native reset waits for pending reductions but retains parameter grads.
        # A failed iteration is discarded, including no-sync accumulation.
        for param in self.fsdp_params:
            if param.grad_offload_event is not None:
                param.grad_offload_event.synchronize()
                param.grad_offload_event = None
            param.sharded_param.grad = None
            if hasattr(param, "_unsharded_param"):
                param._unsharded_param.grad = None
            param.unsharded_accumulated_grad = None

    def wait_for_unshard(self) -> None:
        if not self._all_gather_result:
            return
        if self._training_state == TrainingState.FORWARD:
            for context in self.communication_contexts:
                previous = context.all_gather_state
                if (
                    previous is not None
                    and previous.hook_module() is not self.hook_module()
                ):
                    if previous.event is not None:
                        self._wait_all_gather_streams_on_event(previous.event)
                    context.all_gather_state = None
        super().wait_for_unshard()
        if state := self.comm_ctx.all_gather_state:
            self.comm_ctx.all_gather_state = HookAllGatherState(
                state.all_gather_result, state.event, self.hook_module
            )

    @_dynamo_disable
    def post_backward(self, *unused: Any) -> None:
        if self.reduce_grads:
            for context in self.communication_contexts:
                if context is self.comm_ctx:
                    continue  # Native retention limits govern the local queue.
                retained = []
                for state in context.reduce_scatter_states:
                    if state.hook_module() is self.hook_module():
                        retained.append(state)
                    elif state.event is not None:
                        self.device_handle.current_stream().wait_event(state.event)
                context.reduce_scatter_states[:] = retained
        super().post_backward(*unused)
        for index, state in enumerate(self.comm_ctx.reduce_scatter_states):
            if not isinstance(state, HookReduceScatterState):
                self.comm_ctx.reduce_scatter_states[index] = HookReduceScatterState(
                    state.reduce_scatter_input, state.event, self.hook_module
                )


def init_fine_grained_param_groups(
    state: FineGrainedFSDPState,
    params: list[nn.Parameter],
    modules: tuple[nn.Module, ...],
    mesh_info: DataParallelMeshInfo,
    post_forward_mesh_info: FSDPMeshInfo | None,
    device: torch.device,
    shard_placement_fn: Callable[[nn.Parameter], ShardPlacementFnResult] | None,
    mp_policy: MixedPrecisionPolicy,
    offload_policy: OffloadPolicy,
    *,
    reshard_after_forward: bool | int,
) -> None:
    """Partition parameters by shard and replication groups before construction."""
    if not params:
        return
    if shard_placement_fn is None:
        groups = [(mesh_info, params)]
    else:
        if not isinstance(mesh_info, FSDPMeshInfo):
            raise ValueError(
                "Per-param mesh via shard_placement_fn requires FSDPMeshInfo"
            )
        by_process_group: dict[
            tuple[dist.ProcessGroup, dist.ProcessGroup | None],
            tuple[FSDPMeshInfo, list[nn.Parameter]],
        ] = {}
        for param in params:
            param_mesh = resolve_shard_placement(
                shard_placement_fn(param), mesh_info
            ).mesh_info
            key = (
                param_mesh.shard_process_group,
                param_mesh.replicate_process_group
                if isinstance(param_mesh, HSDPMeshInfo)
                else None,
            )
            if key not in by_process_group:
                by_process_group[key] = (param_mesh, [param])
            else:
                if by_process_group[key][0] is not param_mesh:
                    raise ValueError(
                        "Params sharing a process group must use the same "
                        "FSDPMeshInfo object"
                    )
                by_process_group[key][1].append(param)
        groups = list(by_process_group.values())
    for group_mesh, group_params in groups:
        group_post_forward = (
            post_forward_mesh_info
            if group_mesh is mesh_info
            else _get_post_forward_mesh_info(reshard_after_forward, group_mesh)
        )
        state._fsdp_param_groups.append(
            FineGrainedFSDPParamGroup(
                group_params,
                modules,
                group_mesh,
                group_post_forward,
                device,
                shard_placement_fn,
                mp_policy,
                offload_policy,
            )
        )
