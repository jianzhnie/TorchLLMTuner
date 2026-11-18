"""Gradient synchronization, clipping, and finite-step checks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch.distributed.tensor import DTensor

from experiments.pipeline_parallel.collectives import (
    clip_grad_norm_ as storage_clip_grad_norm_,
)
from experiments.pipeline_parallel.reductions import TrainingReductionMeshes
from llmtuner.accelerator.collectives import all_reduce
from llmtuner.errors import UnsupportedCombinationError
from llmtuner.parallel.tensor_parallel.tensor_parallel import tp_sharded_param_ids
from llmtuner.trainer.runtime.gradients import (
    check_finite,
    expert_parameter_ids,
    grad_norm_options,
    reference_norm_gradients,
    unique_parameters,
)

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from experiments.pipeline_parallel.topology import ParallelTopology


def allreduce_fsdp_excluded_grads(
    model_parts: Sequence[torch.nn.Module], *, parallel_dims: ParallelTopology | None
) -> None:
    """Sum non-FSDP gradients over their dense or expert replica axes."""
    if parallel_dims is None:
        return
    expert_ids = (
        expert_parameter_ids(model_parts) if parallel_dims.ep_enabled else set()
    )
    meshes = {
        False: [parallel_dims.get_optional_mesh(axis) for axis in ("dp", "cp")],
        True: [
            parallel_dims.get_optional_mesh(axis) for axis in ("dp_replicate", "efsdp")
        ],
    }
    for parameter in unique_parameters(model_parts):
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


def clip_grad_norm_(
    parameters, *args, reference_model_parts=None, reference_error_group=None, **kwargs
):
    """Use storage clipping unless model-declared reference groups are selected."""
    if reference_model_parts is None:
        return storage_clip_grad_norm_(parameters, *args, **kwargs)
    if args:
        raise ValueError("reference norm requires keyword clipping options")
    if kwargs.get("norm_precision") != "parameter" or any(
        kwargs.get(name) is not None for name in ("pp_mesh", "ep_mesh", "tp_mesh")
    ):
        raise UnsupportedCombinationError(
            "reference norm layout requires plain DP parameter precision"
        )
    parameters = list(parameters)
    norm_gradients = reference_norm_gradients(
        parameters, reference_model_parts, error_group=reference_error_group
    )
    # Match native per-parameter norm inputs without retaining copies of all
    # logical expert gradients at once.
    norms = [
        torch.linalg.vector_norm(gradient.contiguous()) for gradient in norm_gradients
    ]
    total_norm = torch.linalg.vector_norm(torch.stack(norms))
    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()
    if kwargs.get("error_if_nonfinite", False) and not torch.isfinite(total_norm):
        raise RuntimeError("reference gradient norm is non-finite")
    if kwargs["max_norm"] > 0:
        torch.nn.utils.clip_grads_with_norm_(
            parameters, kwargs["max_norm"], total_norm, foreach=False
        )
    return total_norm


def clip_and_check_finite(
    model_parts: Sequence[torch.nn.Module],
    *,
    parallel_dims: ParallelTopology | None,
    max_norm: float,
    precision: str,
    expert_reduce_layout: str,
    step: int,
    error_group: ProcessGroup | None = None,
    reduction_meshes: TrainingReductionMeshes,
    loss_is_finite: torch.Tensor,
) -> torch.Tensor:
    """Clip gradients and produce one rank-uniform finite-step verdict."""
    pp_mesh = reduction_meshes.pp_mesh
    parameters = list(unique_parameters(model_parts))
    expert_ids = expert_parameter_ids(model_parts)
    expert_parameters = [p for p in parameters if id(p) in expert_ids]
    ep_mesh = (
        parallel_dims.get_optional_mesh("ep")
        if parallel_dims is not None and parallel_dims.ep_enabled
        else None
    )
    tp_mesh = (
        parallel_dims.get_optional_mesh("tp")
        if parallel_dims is not None and parallel_dims.tp_enabled
        else None
    )
    tp_sharded_ids = tp_sharded_param_ids(model_parts) if tp_mesh else set()
    norm_options = grad_norm_options(
        model_parts, precision=precision, expert_reduce_layout=expert_reduce_layout
    )
    grad_norm = clip_grad_norm_(
        parameters,
        max_norm=max_norm,
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
        reference_error_group=error_group,
        **norm_options,
    )

    step_is_finite = loss_is_finite
    step_is_finite.logical_and_(torch.isfinite(grad_norm).all())
    all_reduce(step_is_finite, op="min", group=error_group)
    check_finite(step, step_is_finite)
    return grad_norm
