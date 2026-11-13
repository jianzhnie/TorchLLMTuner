"""Fine-grained hook and communication lifecycles for PyTorch 2.14 FSDP.

Adapted from FSDPTurbo's fine_grained_fully_shard/state.py. Parameter groups
and collectives retain PyTorch 2.14 semantics, including per-parameter meshes.
Contexts retain independent buffers/events while sharing communication streams
to preserve collective ordering, as in the upstream multi-context engine.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from weakref import ref

import torch
import torch.nn as nn
from torch.distributed._composable import contract
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard._fsdp_api import (
    DataParallelMeshDims,
    OffloadPolicy,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    FSDPMeshInfo,
    ShardPlacementFnResult,
    TrainingState,
    _dynamo_disable,
)
from torch.distributed.fsdp._fully_shard._fsdp_init import (
    _apply_to_module,
    _get_device_from_mesh,
    _get_mesh_info,
    _get_modules_and_states,
    _get_post_forward_mesh_info,
    _init_default_mesh,
    _validate_mesh,
    _validate_module,
)
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPCommContext
from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
from torch.distributed.fsdp._fully_shard._fully_shard import (
    FSDPModule,
    _unimplemented_deepcopy,
    get_cls_to_fsdp_cls,
)

from llmtuner.errors import EnvironmentUnsupportedError
from llmtuner.models.common.aux_loss import in_checkpoint_recompute

from .fine_grained_param_group import init_fine_grained_param_groups

if not hasattr(FSDPState(), "_fsdp_param_groups") or not hasattr(
    FSDPState, "_reset_iter_state"
):
    raise EnvironmentUnsupportedError(
        "custom FSDP requires PyTorch 2.14 multiple parameter groups and "
        "iteration reset APIs; install a compatible build or select "
        "fsdp_implementation='native'"
    )


class FineGrainedFSDPState(FSDPState):
    """Keep managed-module discovery separate from forward hook ownership."""

    def init(
        self,
        modules: tuple[nn.Module, ...],
        device: torch.device,
        mp_policy: MixedPrecisionPolicy,
        auto_reshard_after_forward: bool,
        hook_module: nn.Module | None = None,
    ) -> None:
        super().init(modules, device, mp_policy, auto_reshard_after_forward)
        self.hook_module = ref(hook_module if hook_module is not None else modules[0])
        self.communication_contexts = [self._comm_ctx]
        if hook_module is not None:
            self._pre_forward_hook_handle.remove()
            self._post_forward_hook_handle.remove()
            self._modules_to_run_forward.clear()
            self._pre_forward_hook_handle = hook_module.register_forward_pre_hook(
                self._pre_forward, prepend=True, with_kwargs=True
            )
            self._post_forward_hook_handle = hook_module.register_forward_hook(
                self._post_forward, prepend=False
            )

    def _lazy_init(self) -> None:
        if self._is_root is None and self.hook_module() not in set(
            self._modules[0].modules()
        ):
            raise RuntimeError(
                "relocated FSDP hooks require running forward through a "
                "fine-grained FSDP root first"
            )
        if self._state_ctx.all_states and not isinstance(
            self._state_ctx.all_states[0], FineGrainedFSDPState
        ):
            raise RuntimeError("fine-grained FSDP requires a custom FSDP root")
        super()._lazy_init()

    def _init_shared_state(self) -> None:
        if any(
            not isinstance(state, FineGrainedFSDPState)
            for state in self._state_ctx.all_states
        ):
            raise RuntimeError("do not mix native and custom FSDP units in one model")
        super()._init_shared_state()
        contexts = [self._comm_ctx]
        hook_counts: dict[nn.Module, int] = {}
        assignments = []
        for state in self._state_ctx.all_states:
            target = state.hook_module()
            index = hook_counts.get(target, 0)
            hook_counts[target] = index + 1
            if index == len(contexts):
                context = FSDPCommContext()
                # Share root streams directly: lazy_init would allocate four
                # new device streams that are immediately overwritten below.
                for attribute in (
                    "device_handle",
                    "all_gather_copy_in_stream",
                    "all_gather_stream",
                    "reduce_scatter_stream",
                    "all_reduce_stream",
                    "post_forward_order",
                ):
                    setattr(context, attribute, getattr(self._comm_ctx, attribute))
                context.all_gather_state = None
                context.reduce_scatter_states = []
                context._last_post_reduce_events = {}
                context.reduce_scatter_max_input_buffers = 1
                contexts.append(context)
            context = contexts[index]
            state._comm_ctx = context
            state.communication_contexts = contexts
            for group in state._fsdp_param_groups:
                group.comm_ctx = context
                group.hook_module = state.hook_module
                group.communication_contexts = contexts
            assignments.append((state, context))
        for context in contexts:
            context.reduce_scatter_max_input_buffers = max(
                (
                    group.reduce_scatter_max_input_buffers
                    for state, assigned in assignments
                    if assigned is context
                    for group in state._fsdp_param_groups
                ),
                default=1,
            )

    @_dynamo_disable
    def _pre_forward(self, module, args, kwargs):
        # Pipeline microbatches may leave a relocated state IDLE after another
        # forward/backward used the same ancestor. Checkpoint replay must use
        # the backward path, never register a new forward lifecycle inside SAC.
        if in_checkpoint_recompute.get():
            self._training_state = TrainingState.PRE_BACKWARD
        return super()._pre_forward(module, args, kwargs)

    @_dynamo_disable
    def _post_forward(self, module: nn.Module, input: Any, output: Any) -> Any:
        is_forward_root = (
            self._state_ctx.iter_forward_root is self
            and self._training_state != TrainingState.PRE_BACKWARD
        )
        output = super()._post_forward(module, input, output)
        if is_forward_root:
            for context in self.communication_contexts:
                if state := context.all_gather_state:
                    context.all_gather_copy_in_stream.wait_event(state.event)
                    context.all_gather_stream.wait_event(state.event)
                    context.all_gather_state = None
        return output

    @_dynamo_disable
    def _root_post_backward_final_callback(self) -> None:
        # Native finalization covers every parameter group and the callback's
        # context. Complete the other contexts after all groups have reduced.
        super()._root_post_backward_final_callback()
        if self._state_ctx.is_last_backward:
            for context in self.communication_contexts:
                context.post_forward_order.clear()
                for state in context.reduce_scatter_states:
                    if state.event is not None:
                        self._device_handle.current_stream().wait_event(state.event)
                context.reduce_scatter_states.clear()

    def _reset_iter_state(self) -> None:
        if self._is_root is False:
            raise RuntimeError(
                "reset_iter_state must be called on the root FSDP module"
            )
        current_stream = self._device_handle.current_stream()
        for context in self.communication_contexts:
            if state := context.all_gather_state:
                if state.event is not None:
                    current_stream.wait_event(state.event)
                context.all_gather_state = None
            for state in context.reduce_scatter_states:
                if state.event is not None:
                    current_stream.wait_event(state.event)
            context.reduce_scatter_states.clear()
            for event in context._last_post_reduce_events.values():
                current_stream.wait_event(event)
            context._last_post_reduce_events.clear()
            context.post_forward_order.clear()
        super()._reset_iter_state()


@contract(state_cls=FineGrainedFSDPState)
def apply_fine_grained_fsdp(
    module: nn.Module | list[nn.Module],
    *,
    hook_module: nn.Module | None = None,
    mesh: DeviceMesh | None = None,
    reshard_after_forward: bool | int | None = None,
    shard_placement_fn: Callable[[nn.Parameter], ShardPlacementFnResult] | None = None,
    mp_policy: MixedPrecisionPolicy | None = None,
    offload_policy: OffloadPolicy | None = None,
    ignored_params: set[nn.Parameter] | None = None,
    dp_mesh_dims: DataParallelMeshDims | None = None,
) -> nn.Module | list[nn.Module]:
    """Shard parameter groups with custom hook and communication ownership.

    Initialization follows PyTorch 2.14's fully_shard, including per-parameter
    meshes, automatic root reshard policy and grouped shared parameters.
    Apply this strategy bottom-up, including the root module. A relocated
    hook target must be an ancestor that executes all managed modules.
    """
    _validate_module(module, "apply_fine_grained_fsdp")
    if mp_policy is None:
        mp_policy = MixedPrecisionPolicy()
    if offload_policy is None:
        offload_policy = OffloadPolicy()
    mesh = mesh or _init_default_mesh()
    _validate_mesh(mesh, dp_mesh_dims)
    mesh_info = _get_mesh_info(mesh, dp_mesh_dims)
    device = _get_device_from_mesh(mesh)
    auto_reshard = reshard_after_forward is None
    policy = True if auto_reshard else reshard_after_forward
    if isinstance(mesh_info, FSDPMeshInfo):
        if (
            mesh_info.is_spmd_mesh
            and isinstance(policy, int)
            and not isinstance(policy, bool)
        ):
            raise NotImplementedError(
                "reshard_after_forward as int is not supported with dp_mesh_dims"
            )
        post_forward_mesh_info = _get_post_forward_mesh_info(policy, mesh_info)
    else:
        post_forward_mesh_info = None
    original, modules, managed, params, buffers = _get_modules_and_states(
        module, device, ignored_params
    )
    state = apply_fine_grained_fsdp.state(modules[0])
    state.init(modules, device, mp_policy, auto_reshard, hook_module)
    init_fine_grained_param_groups(
        state,
        params,
        modules,
        mesh_info,
        post_forward_mesh_info,
        device,
        shard_placement_fn,
        mp_policy,
        offload_policy,
        reshard_after_forward=policy,
    )
    for managed_module in managed:
        managed_module._is_fsdp_managed_module = True
        managed_module._fsdp_use_orig_params = True
    _apply_to_module(
        modules, get_cls_to_fsdp_cls(), FSDPModule, "FSDP", _unimplemented_deepcopy
    )
    return original
