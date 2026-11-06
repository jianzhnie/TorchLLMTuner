"""Gradient synchronization, clipping, and finite-step checks."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

import torch
from torch.distributed.tensor import DTensor

from ...accelerator.collectives import all_reduce, clip_grad_norm_
from ...accelerator.distributed import get_comm_device
from ...models.common.moe.layer import MoE
from ...parallel.expert_parallel.npu_baseline_experts import NpuBaselineGroupedExperts
from ...parallel.tensor_parallel.tensor_parallel import tp_sharded_param_ids
from .reductions import TrainingReductionMeshes

if TYPE_CHECKING:
    from ..trainer import Trainer


def unique_parameters(
    model_parts: Iterable[torch.nn.Module],
) -> Iterator[torch.nn.Parameter]:
    """Yield registered parameters once, preserving model-part order."""
    seen: set[int] = set()
    for part in model_parts:
        for parameter in part.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                yield parameter


def expert_parameter_ids(model_parts: Iterable[torch.nn.Module]) -> set[int]:
    """Return identities of parameters owned by routed experts."""
    return {
        id(parameter)
        for part in model_parts
        for module in part.modules()
        if isinstance(module, MoE)
        for parameter in module.routed_experts.inner_experts.parameters()
    }


def allreduce_fsdp_excluded_grads(trainer: Trainer) -> None:
    """Sum non-FSDP gradients over their dense or expert replica axes."""
    if trainer.parallel_dims is None:
        return
    expert_ids = (
        expert_parameter_ids(trainer.model_parts)
        if trainer.parallel_dims.ep_enabled
        else set()
    )
    meshes = {
        False: [trainer.parallel_dims.get_optional_mesh(axis) for axis in ("dp", "cp")],
        True: [
            trainer.parallel_dims.get_optional_mesh(axis)
            for axis in ("dp_replicate", "efsdp")
        ],
    }
    for parameter in unique_parameters(trainer.model_parts):
        if not parameter.requires_grad or isinstance(parameter, DTensor):
            continue
        for mesh in meshes[id(parameter) in expert_ids]:
            if mesh is None or mesh.size() <= 1:
                continue
            group = mesh.get_group()
            present = torch.tensor(
                int(parameter.grad is not None),
                device=parameter.device,
                dtype=torch.int32,
            )
            all_reduce(present, group=group)
            if present.item() == 0:
                continue
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            all_reduce(parameter.grad, group=group)


def allreduce_replicated_tp_grads(trainer: Trainer) -> None:
    """Sum gradients of TP-replicated parameters over partial token shards."""
    tp_mesh = (
        None
        if trainer.parallel_dims is None
        else trainer.parallel_dims.get_optional_mesh("tp")
    )
    if tp_mesh is None:
        return
    sharded_ids = tp_sharded_param_ids(trainer.model_parts)
    group = tp_mesh.get_group()
    parameters = [
        param
        for param in unique_parameters(trainer.model_parts)
        if param.requires_grad and id(param) not in sharded_ids
    ]
    if not parameters:
        return
    # Every TP rank owns the same ordered replicated parameters. Agree on
    # participation once, before any gradient collective, including when a
    # rank's local branch did not use a parameter.
    present = torch.tensor(
        [int(param.grad is not None) for param in parameters],
        device=get_comm_device(group),
        dtype=torch.int32,
    )
    all_reduce(present, group=group)
    for param, count in zip(parameters, present.tolist(), strict=True):
        if count == 0:
            continue
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        grad = param.grad
        if isinstance(grad, DTensor):
            all_reduce(grad.to_local(), group=group)
        else:
            all_reduce(grad, group=group)


def check_finite(trainer: Trainer, step_is_finite: torch.Tensor) -> None:
    """Stop before updating parameters when the global step is non-finite."""
    torch._assert_async(
        step_is_finite,
        f"Loss or gradient norm is not finite at step {trainer.step}. "
        "Stopping before the optimizer update, since every later number "
        "would be garbage.",
    )


def grad_norm_options(trainer: Trainer) -> dict:
    if getattr(trainer.cfg, "grad_norm_precision", "float32") == "float32":
        return {}
    packed_parameters = {}
    for part in trainer.model_parts:
        for module in part.modules():
            if isinstance(module, NpuBaselineGroupedExperts):
                pair = (module.w1_EFD, module.w3_EFD)
                if all(parameter.grad is None for parameter in pair):
                    continue
                packed_parameters.setdefault(tuple(map(id, pair)), pair)
    return {
        "norm_precision": trainer.cfg.grad_norm_precision,
        "packed_parameters": list(packed_parameters.values()),
    }


def clip_and_check_finite(
    trainer: Trainer,
    *,
    reduction_meshes: TrainingReductionMeshes,
    loss_is_finite: torch.Tensor,
) -> torch.Tensor:
    """Clip gradients and produce one rank-uniform finite-step verdict."""
    trainer.before_gradient_clipping()
    pp_mesh = reduction_meshes.pp_mesh
    parameters = list(unique_parameters(trainer.model_parts))
    expert_ids = expert_parameter_ids(trainer.model_parts)
    expert_parameters = [p for p in parameters if id(p) in expert_ids]
    ep_mesh = (
        trainer.parallel_dims.get_optional_mesh("ep")
        if trainer.parallel_dims is not None and trainer.parallel_dims.ep_enabled
        else None
    )
    tp_mesh = (
        trainer.parallel_dims.get_optional_mesh("tp")
        if trainer.parallel_dims is not None and trainer.parallel_dims.tp_enabled
        else None
    )
    tp_sharded_ids = tp_sharded_param_ids(trainer.model_parts) if tp_mesh else set()
    norm_options = grad_norm_options(trainer)
    grad_norm = clip_grad_norm_(
        parameters,
        max_norm=trainer.cfg.max_norm,
        foreach=True,
        pp_mesh=pp_mesh,
        ep_mesh=ep_mesh,
        expert_parameters=expert_parameters if ep_mesh is not None else None,
        tp_mesh=tp_mesh,
        tp_sharded_parameters=(
            [p for p in parameters if id(p) in tp_sharded_ids]
            if tp_mesh is not None
            else None
        ),
        **norm_options,
    )

    step_is_finite = loss_is_finite
    step_is_finite.logical_and_(torch.isfinite(grad_norm).all())
    all_reduce(step_is_finite, op="min")
    check_finite(trainer, step_is_finite)
    return grad_norm
