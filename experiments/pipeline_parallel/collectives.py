"""Pipeline gradient clipping over disjoint stage parameters.

Shared collectives and local norm helpers live in llmtuner.accelerator.collectives.
This module retains the PP norm reduction before scaling local gradients.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from llmtuner.accelerator.collectives import (
    _clip_offloaded_gradients,
    _offloaded_gradient_norm,
    _parameter_precision_norm,
    build_packed_gradient_groups,
)
from llmtuner.accelerator.distributed import get_comm_device

__all__ = ["clip_grad_norm_"]


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
    norm_precision: str = "float32",
    packed_parameters: Iterable[tuple[torch.Tensor, torch.Tensor]] | None = None,
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
            stages before clipping. Stage norms use a common FP32 accumulator,
            promoted to FP64 when any stage supplies an FP64 norm.
        ep_mesh: expert-parallel mesh. When present, only the norm contribution
            from ``expert_parameters`` is reduced over this mesh.
        expert_parameters: parameters physically partitioned across EP ranks.
            Required exactly when ``ep_mesh`` is provided.
        tp_mesh: tensor-parallel mesh for local TP weight shards.
        tp_sharded_parameters: parameters physically sharded across TP ranks.
            Required exactly when ``tp_mesh`` is provided.
        norm_precision: ``float32`` promotes low-precision per-gradient norms;
            ``parameter`` preserves native dtype through norm aggregation and
            clipping coefficients, matching the NPU baseline for finite positive p.
        packed_parameters: gate/up parameter pairs treated as one axis-1 packed
            tensor for ``parameter`` norm rounding. Scaling still uses the
            original parameters; the default FP32 norm is unaffected.

    Returns:
        The total norm of the parameter gradients (viewed as one vector).

    Nonfinite checks run after the global norm is assembled, before clipping.
    """
    norm_type = float(norm_type)
    if norm_precision not in ("float32", "parameter"):
        raise ValueError("norm_precision must be float32 or parameter")
    if norm_precision == "parameter" and (
        not math.isfinite(norm_type) or norm_type <= 0
    ):
        raise ValueError(
            "parameter norm precision requires a finite positive norm_type"
        )
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
    packed_gradients = build_packed_gradient_groups(
        parameters, packed_parameters, expert_ids=expert_ids, tp_ids=tp_ids
    )

    def _local_norm(grads: list[torch.Tensor]) -> torch.Tensor:
        if norm_precision == "parameter":
            return _parameter_precision_norm(grads, norm_type, packed_gradients)
        # FSDP exclusions remain plain tensors. Reduce each DTensor mesh's
        # scalar norm before combining it with replicated local contributions;
        # foreach cannot mix tensors/DTensors or independent meshes.
        groups: dict[tuple[int | None, torch.device], list[torch.Tensor]] = {}
        for grad in grads:
            key = (
                id(grad.device_mesh) if isinstance(grad, DTensor) else None,
                grad.device,
            )
            groups.setdefault(key, []).append(grad)
        group_norms = []
        for grads in groups.values():
            if any(
                isinstance(grad, DTensor)
                and grad.device.type == "cpu"
                and grad.device_mesh.device_type != "cpu"
                for grad in grads
            ):
                norms = [_offloaded_gradient_norm(grad, norm_type) for grad in grads]
                device = get_comm_device(grads[0].device_mesh.get_group(0))
                group_norms.append(
                    torch.nn.utils.get_total_norm(
                        [norm.to(device) for norm in norms],
                        norm_type,
                        False,
                        foreach,
                    )
                )
                continue
            # Check only after every mesh reduction, so one bad local shard does
            # not raise while its peers are still waiting in a collective.
            if any(grad.dtype in (torch.float16, torch.bfloat16) for grad in grads):
                # Keep only scalar FP32 norms, rather than retaining FP32 copies
                # of every gradient at once. DTensor preserves the reduction layout.
                norms = [
                    torch.linalg.vector_norm(
                        grad,
                        norm_type,
                        dtype=torch.float32
                        if grad.dtype in (torch.float16, torch.bfloat16)
                        else None,
                    )
                    for grad in grads
                ]
                norm = torch.nn.utils.get_total_norm(norms, norm_type, False, foreach)
            else:
                norm = torch.nn.utils.get_total_norm(grads, norm_type, False, foreach)
            if isinstance(norm, DTensor):
                norm = norm.full_tensor()
            # Squaring a finite fp16 norm (e.g. 300) already overflows. Mesh
            # accumulation must use at least fp32 before taking p-th powers.
            if (
                pp_mesh is not None or tp_mesh is not None or ep_mesh is not None
            ) and norm.dtype in (torch.float16, torch.bfloat16):
                norm = norm.float()
            group_norms.append(norm)
        if len(group_norms) == 1:
            return group_norms[0]
        if group_norms:
            norm_device = next(
                (norm.device for norm in group_norms if norm.device.type != "cpu"),
                group_norms[0].device,
            )
            group_norms = [norm.to(norm_device) for norm in group_norms]
        return torch.nn.utils.get_total_norm(group_norms, norm_type, False, foreach)

    if tp_mesh is not None or ep_mesh is not None:
        dense_norm = _local_norm(dense_grads)
        tp_norm = _local_norm(tp_grads)
        expert_norm = _local_norm(expert_grads)
        # get_total_norm([]) returns a CPU zero. Every rank still joins the
        # collective, including a PP stage with no expert or TP weight grads.
        device = next(
            (p.grad.device for p in parameters if p.grad is not None),
            parameters[0].device
            if parameters
            else get_comm_device(
                (tp_mesh if tp_mesh is not None else ep_mesh).get_group()
            ),
        )
        if device.type == "cpu":
            device = get_comm_device(
                (tp_mesh if tp_mesh is not None else ep_mesh).get_group()
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
            total_norm = (dense_norm.pow(norm_type) + tp_power + expert_power).pow(
                1.0 / norm_type
            )

    else:
        total_norm = _local_norm(dense_grads)

    # Under FSDP/TP the norm comes back as a DTensor with a partial (sum)
    # placement: it must be materialized both to be correct along those axes and
    # to return a tensor whose ``.item()`` is the real global value.
    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()

    if pp_mesh is not None:
        # Empty gradient lists produce a CPU zero; NCCL/HCCL still require
        # this stage to participate using its parameter device.
        device = next(
            (p.grad.device for p in parameters if p.grad is not None),
            parameters[0].device
            if parameters
            else get_comm_device(pp_mesh.get_group()),
        )
        if device.type == "cpu":
            device = get_comm_device(pp_mesh.get_group())
        # Empty stages return FP32, while native local norms may be BF16/FP16
        # or FP64. Agree before reducing: c10d requires identical wire dtypes.
        # Promote low precision before powers and retain FP64 if any stage uses it.
        group = pp_mesh.get_group()
        needs_double = torch.tensor(
            int(total_norm.dtype == torch.float64), device=device, dtype=torch.int32
        )
        dist.all_reduce(needs_double, op=dist.ReduceOp.MAX, group=group)
        total_norm = total_norm.to(
            device=device,
            dtype=torch.float64 if needs_double.item() else torch.float32,
        )
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            # A norm does not survive an all-reduce directly: sum the p-th
            # powers, reduce, then take the p-th root.
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    if error_if_nonfinite and not torch.isfinite(total_norm):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )

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
        offloaded_gradients = [
            parameter.grad
            for group_parameters in clip_groups.values()
            for parameter in group_parameters
            if isinstance(parameter.grad, DTensor)
            and parameter.grad.device.type == "cpu"
            and parameter.grad.device_mesh.device_type != "cpu"
        ]
        offloaded_ids = _clip_offloaded_gradients(
            offloaded_gradients, max_norm=max_norm, total_norm=total_norm
        )
        for group_parameters in clip_groups.values():
            group_parameters = [
                parameter
                for parameter in group_parameters
                if id(parameter.grad) not in offloaded_ids
            ]
            if not group_parameters:
                continue
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
