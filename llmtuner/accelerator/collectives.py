"""Collective operations, and gradient-norm clipping that respects the meshes.

The reductions and ``clip_grad_norm_`` are vendored from torchtitan
``distributed/utils.py``. What changed:

* The ``extra_pg`` argument is gone. torchtitan threads an extra process group
  through every reduction to reach ranks a mesh does not model (its odd-sized TP
  cases); llmtuner's meshes cover every rank, so the mesh argument is the whole
  addressing story.
* EP-aware clipping is adapted to llmtuner's physical expert partition: callers
  pass the exact local expert parameters plus the EP mesh. Dense parameters are
  counted once, while the p-th powers of local expert norms are reduced across
  EP ranks. This avoids upstream's requirement that every parameter be a
  DTensor carrying an explicit ``"ep"`` mesh axis.
* ``dist_sum`` / ``dist_max`` / ``dist_mean`` / ``dist_sum_tensor`` are not
  re-created. Upstream they are one-line ``funcol.all_reduce`` wrappers naming
  a reduction and its mesh; llmtuner's counterpart is this module's
  ``all_reduce`` (cast to the comm device + in-place c10d collective, vendored
  from mmengine's ``dist.all_reduce``), called at the site.
  ``trainer/trainer.py`` and ``trainer/validate.py`` reduce their loss/token
  denominators over ``dp_mesh`` / ``loss_mesh`` with it, and
  ``components/metrics.py`` reduces nothing at all (see its docstring), so
  there is no caller that would read better with a named helper. This also
  covers upstream's ``all_gather_entries`` and friends, which exist for
  bucketed per-module metrics llmtuner does not report.

The single non-obvious line kept from upstream is the ``DTensor`` branch in
``clip_grad_norm_``; it carries a comment explaining why it exists.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from ..utils.logger_utils import get_logger
from .device import device_module
from .dist_utils import (
    get_comm_device,
    get_default_group,
    get_world_size,
)
from .tensor_transfer import cast_data_device, get_data_device

logger = get_logger(__name__)

__all__ = [
    "all_reduce",
    "clip_grad_norm_",
    "set_pg_timeouts",
]


def set_pg_timeouts(
    timeout: timedelta,
    parallel_dims,
    *,
    device: torch.device | None = None,
) -> None:
    """Lower every process group's timeout, once startup is behind the run.

    Called after the first completed train step. The groups are created with a
    long timeout because startup -- model build, the first collective, compile --
    is what genuinely takes minutes; left there, a later hang looks the same as a
    slow start and the job waits out the startup value. By this point that work
    is done, so the timeout can become the one a stall should be measured
    against.

    ``device`` is the rank's device, used only for the barrier's ``device_ids``
    and the device-side sync (NCCL needs both; gloo rejects ``device_ids``, so
    ``None`` -- the default -- omits them).

    The barrier before the change is the point of the whole function: a slow rank
    may still be inside an operation permitted by the OLD timeout while a fast
    rank moves on and issues a collective under the new, shorter one, and
    times out waiting for it. Synchronizing first means every rank crosses the
    reduction together.
    """
    if device is not None and device.type != "cpu":
        # NCCL accepts device_ids; HCCL selects the current NPU and rejects the
        # CUDA-specific argument on some torch-npu releases.
        if device.type == "cuda":
            dist.barrier(device_ids=[device.index])
        else:
            dist.barrier()
        # Derive the sync call from the passed device rather than the
        # import-time global.
        getattr(torch, device.type, device_module).synchronize(device)
    else:
        dist.barrier()

    # torch-npu 2.10 exposes the c10d compatibility API, but HCCL does not
    # implement the operation (it emits one warning per group and changes
    # nothing). The synchronization above is still useful; retain the startup
    # timeout and make this limitation explicit once per rank.
    if device is not None and device.type == "npu":
        logger.warning(
            "HCCL cannot change process-group timeouts at runtime; continuing "
            "with the startup timeout"
        )
        return

    # ``None`` names the default (world) group, which is not part of any mesh.
    groups = [
        mesh.get_group()
        for mesh in parallel_dims.get_all_one_dimensional_meshes().values()
    ]
    logger.info(
        "Adjusting the timeout of %d process group(s) plus the default to %s",
        len(groups),
        timeout,
    )
    set_timeout = getattr(dist, "set_timeout", None)
    if set_timeout is None:
        # PyTorch 2.10 exposes this operation only from distributed_c10d.
        # Keep the compatibility detail here rather than forcing the trainer to
        # know which torch release it is running on.
        set_timeout = getattr(dist.distributed_c10d, "_set_pg_timeout", None)
    if set_timeout is None:
        logger.warning(
            "This PyTorch build cannot change process-group timeouts at runtime; "
            "continuing with the startup timeout"
        )
        return

    for group in groups:
        set_timeout(timeout, group)
    set_timeout(timeout)


def clip_grad_norm_(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh=None,
    ep_mesh=None,
    expert_parameters: Iterable[torch.Tensor] | None = None,
    tp_mesh=None,
    tp_sharded_parameters: Iterable[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Clip the gradient norm of an iterable of parameters, over the whole model.

    ``torch.nn.utils.clip_grad_norm_`` handles DTensor/FSDP sharding, but this
    project's TP weights are local tensors. Their squared norms must be summed
    over TP, while TP-replicated parameters are counted only once. PP stages
    hold disjoint parameters, so the assembled norm is reduced over PP too.

    Args:
        parameters: an iterable of tensors (or one tensor) to normalize.
        max_norm: max norm of the gradients. A non-positive value skips the clip
            but still returns the norm -- which is how the training loop reports
            ``grad_norm`` without paying for a clip nobody asked for.
        norm_type: type of the used p-norm. ``'inf'`` for infinity norm.
        error_if_nonfinite: throw if the total norm is nan/inf.
        foreach: use the faster foreach implementation (``None`` lets torch pick).
        pp_mesh: pipeline-parallel mesh; when present the norm is reduced across
            stages before clipping.
        ep_mesh: expert-parallel mesh. When present, only the norm contribution
            from ``expert_parameters`` is reduced over this mesh.
        expert_parameters: parameters physically partitioned across EP ranks.
            Required exactly when ``ep_mesh`` is provided.
        tp_mesh: tensor-parallel mesh for local TP weight shards.
        tp_sharded_parameters: parameters physically sharded across TP ranks.
            Required exactly when ``tp_mesh`` is provided.

    Returns:
        The total norm of the parameter gradients (viewed as one vector).

    NOTE: intentionally not ``torch.no_grad()`` -- ``get_total_norm`` must keep
    its autograd history so the clip's effect propagates. Do not add it.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)  # do not exhaust a generator

    if (ep_mesh is None) != (expert_parameters is None):
        raise ValueError(
            "ep_mesh and expert_parameters must either both be provided or both be None"
        )
    if (tp_mesh is None) != (tp_sharded_parameters is None):
        raise ValueError(
            "tp_mesh and tp_sharded_parameters must either both be provided "
            "or both be None"
        )

    # Preserve the plain single-device path. Under TP, separate the local
    # weight shards from replicated parameters before any norm calculation.
    # An EP expert is handled by the EP reduction, even if its id also appears
    # in the TP-sharded set (EP can consume ranks along the dense TP axis).
    tp_ids = set() if tp_mesh is None else {id(p) for p in tp_sharded_parameters}
    expert_ids = set() if ep_mesh is None else {id(p) for p in expert_parameters}
    tp_grads = [
        p.grad
        for p in parameters
        if id(p) in tp_ids and id(p) not in expert_ids and p.grad is not None
    ]
    dense_grads = [
        p.grad
        for p in parameters
        if id(p) not in tp_ids and id(p) not in expert_ids and p.grad is not None
    ]
    expert_grads = [
        p.grad for p in parameters if id(p) in expert_ids and p.grad is not None
    ]

    def _local_norm(grads: list[torch.Tensor]) -> torch.Tensor:
        norm = torch.nn.utils.get_total_norm(
            grads, norm_type, error_if_nonfinite, foreach
        )
        if isinstance(norm, DTensor):
            norm = norm.full_tensor()
        return norm

    if tp_mesh is not None or ep_mesh is not None:
        dense_norm = _local_norm(dense_grads)
        tp_norm = _local_norm(tp_grads)
        expert_norm = _local_norm(expert_grads)
        # get_total_norm([]) returns a CPU zero. Every rank still joins the
        # collective, including a PP stage with no expert or TP weight grads.
        device = next(
            (p.grad.device for p in parameters if p.grad is not None),
            parameters[0].device if parameters else dense_norm.device,
        )
        dense_norm = dense_norm.to(device)
        tp_norm = tp_norm.to(device)
        expert_norm = expert_norm.to(device)

        if math.isinf(norm_type):
            if tp_mesh is not None:
                dist.all_reduce(
                    tp_norm, op=dist.ReduceOp.MAX, group=tp_mesh.get_group()
                )
            if ep_mesh is not None:
                dist.all_reduce(
                    expert_norm, op=dist.ReduceOp.MAX, group=ep_mesh.get_group()
                )
            total_norm = torch.maximum(torch.maximum(dense_norm, tp_norm), expert_norm)
        else:
            tp_power = tp_norm.pow(norm_type)
            expert_power = expert_norm.pow(norm_type)
            if tp_mesh is not None:
                dist.all_reduce(
                    tp_power, op=dist.ReduceOp.SUM, group=tp_mesh.get_group()
                )
            if ep_mesh is not None:
                dist.all_reduce(
                    expert_power, op=dist.ReduceOp.SUM, group=ep_mesh.get_group()
                )
            total_norm = (
                dense_norm.pow(norm_type) + tp_power + expert_power
            ).pow(1.0 / norm_type)

    else:
        total_norm = _local_norm(dense_grads)

    # Under FSDP/TP the norm comes back as a DTensor with a partial (sum)
    # placement: it must be materialized both to be correct along those axes and
    # to return a tensor whose ``.item()`` is the real global value.
    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()

    if pp_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            # A norm does not survive an all-reduce directly: sum the p-th
            # powers, reduce, then take the p-th root.
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    if max_norm > 0:
        # torch's foreach clip batches DTensor grads together by device and
        # dtype, but a model can have dense and expert FSDP parameters on
        # different meshes. DTensor cannot multiply such a mixed list in one
        # foreach op. Group by mesh while using the same global clipping norm.
        clip_groups: dict[int | None, list[torch.Tensor]] = {}
        for parameter in parameters:
            grad = parameter.grad
            if grad is None:
                continue
            mesh_key = id(grad.device_mesh) if isinstance(grad, DTensor) else None
            clip_groups.setdefault(mesh_key, []).append(parameter)
        for group_parameters in clip_groups.values():
            torch.nn.utils.clip_grads_with_norm_(
                group_parameters,
                max_norm,
                total_norm,
                # A scalar DTensor norm can still be paired with a different
                # mesh by foreach's fused dispatcher. The scalar loop has the
                # same clipping arithmetic and accepts the already-materialized
                # global norm for every mesh group.
                False if len(clip_groups) > 1 else foreach,
            )
    return total_norm


_REDUCE_OPS = {
    "sum": dist.ReduceOp.SUM,
    "product": dist.ReduceOp.PRODUCT,
    "min": dist.ReduceOp.MIN,
    "max": dist.ReduceOp.MAX,
    "band": dist.ReduceOp.BAND,
    "bor": dist.ReduceOp.BOR,
    "bxor": dist.ReduceOp.BXOR,
}


def all_reduce(
    data: torch.Tensor, op: str = "sum", group: dist.ProcessGroup | None = None
) -> None:
    """In-place all-reduce; a no-op outside a distributed environment.

    Vendored from mmengine's ``dist.all_reduce``. Unlike the raw c10d call it
    casts the tensor to the group's comm device first and copies the result
    back, so a CPU tensor reduces correctly over an NCCL/HCCL group (the
    trainer's loss/token denominators are CPU scalars). ``op="mean"`` is
    emulated as sum + divide because c10d has no mean reduction.
    """
    world_size = get_world_size(group)
    if world_size <= 1:
        return
    if group is None:
        group = get_default_group()

    input_device = get_data_device(data)
    backend_device = get_comm_device(group)
    data_on_device = cast_data_device(data, backend_device)

    if op.lower() == "mean":
        dist.all_reduce(data_on_device, _REDUCE_OPS["sum"], group)
        # true_divide because int64 tensors reject in-place true division.
        data_on_device = torch.true_divide(data_on_device, world_size)
    elif op.lower() in _REDUCE_OPS:
        dist.all_reduce(data_on_device, _REDUCE_OPS[op.lower()], group)
    else:
        raise ValueError(
            f"reduce op should be one of {list(_REDUCE_OPS)} or 'mean', but got {op}"
        )

    cast_data_device(data_on_device, input_device, out=data)
