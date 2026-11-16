"""Gradient synchronization, clipping, and finite-step checks."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from experiment.pipeline_parallel.collectives import (
    all_reduce,
    synchronize_local_errors,
)
from experiment.pipeline_parallel.collectives import (
    clip_grad_norm_ as storage_clip_grad_norm_,
)
from experiment.pipeline_parallel.reductions import TrainingReductionMeshes
from llmtuner.accelerator.distributed import get_comm_device
from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.common.gradient_norm import ReferenceNormGroup
from llmtuner.parallel.tensor_parallel.tensor_parallel import tp_sharded_param_ids

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from experiment.pipeline_parallel.topology import ParallelTopology


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
    """Collect model-declared expert ownership without probing expert storage."""
    parameter_ids: set[int] = set()
    for part in model_parts:
        for module in part.modules():
            declare_parameters = getattr(module, "expert_parallel_parameters", None)
            if callable(declare_parameters):
                parameter_ids.update(
                    id(parameter) for parameter in declare_parameters()
                )
    return parameter_ids


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


def allreduce_replicated_tp_grads(
    model_parts: Sequence[torch.nn.Module], *, parallel_dims: ParallelTopology | None
) -> None:
    """Sum gradients of TP-replicated parameters over partial token shards."""
    tp_mesh = None if parallel_dims is None else parallel_dims.get_optional_mesh("tp")
    if tp_mesh is None:
        return
    sharded_ids = tp_sharded_param_ids(model_parts)
    group = tp_mesh.get_group()
    parameters = [
        param
        for param in unique_parameters(model_parts)
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


def check_step_condition(condition: torch.Tensor, message: str) -> None:
    """Enforce eager step guards without unsupported NPU assertion fallback.

    torch-npu 2.12 has no native aten::_assert_async.msg implementation. Read
    the scalar explicitly at the step boundary; do not disable the check or
    register a global ATen override. Other backends retain their async check.
    This helper belongs outside compiled model graphs.
    """
    if condition.device.type == "npu":
        if condition.numel() != 1:
            raise RuntimeError("Step guard requires exactly one tensor value")
        if not condition.item():
            raise RuntimeError(message)
        return
    torch._assert_async(condition, message)


def check_finite(step: int, step_is_finite: torch.Tensor) -> None:
    """Stop before updating parameters when the global step is non-finite."""
    check_step_condition(
        step_is_finite,
        f"Loss or gradient norm is not finite at step {step}. "
        "Stopping before the optimizer update, since every later number "
        "would be garbage.",
    )


def grad_norm_options(
    model_parts: Sequence[torch.nn.Module],
    *,
    precision: str,
    expert_reduce_layout: str,
) -> dict[str, Any]:
    """Resolve norm representation from explicit numerical/layout policies."""
    if precision == "float32":
        return {}
    if expert_reduce_layout == "split":
        return {
            "norm_precision": "parameter",
            "reference_model_parts": model_parts,
        }
    packed_parameters = {}
    for part in model_parts:
        for module in part.modules():
            declare_groups = getattr(module, "gradient_norm_parameter_groups", None)
            if callable(declare_groups):
                for group in declare_groups():
                    if all(parameter.grad is None for parameter in group):
                        continue
                    packed_parameters.setdefault(tuple(map(id, group)), group)
    return {
        "norm_precision": precision,
        "packed_parameters": list(packed_parameters.values()),
    }


def _collect_reference_groups(parameters, model_parts):
    """Bind and validate model-declared groups before gradient collectives."""
    parameter_ids = {id(parameter) for parameter in parameters}
    groups = {}
    occupied = set()
    for part in model_parts:
        for module in part.modules():
            declare_groups = getattr(module, "reference_norm_groups", None)
            if not callable(declare_groups):
                continue
            for group in declare_groups():
                if not isinstance(group, ReferenceNormGroup):
                    raise TypeError(
                        "reference_norm_groups must return ReferenceNormGroup"
                    )
                identities = tuple(map(id, group.parameters))
                if identities in groups:
                    continue
                if not identities or len(set(identities)) != len(identities):
                    raise ValueError("reference norm groups require unique parameters")
                if not set(identities) <= parameter_ids:
                    raise ValueError(
                        "reference norm group contains untracked parameters"
                    )
                if occupied.intersection(identities):
                    raise ValueError("reference norm parameter groups overlap")
                groups[identities] = group
                occupied.update(identities)
    if not groups:
        raise UnsupportedCombinationError(
            "reference norm found no declared parameter groups"
        )
    for group in groups.values():
        group.validate_gradients()
    return groups


@torch.no_grad()
def reference_norm_gradients(parameters, model_parts, *, error_group=None):
    """Expand declared logical parameter groups into their gradient tensors."""
    with synchronize_local_errors(
        "Reference gradient norm preflight", group=error_group
    ):
        parameters = list(parameters)
        groups = _collect_reference_groups(parameters, model_parts)
        positions = {id(parameter): index for index, parameter in enumerate(parameters)}
        signature = tuple(
            (
                group.layout,
                tuple(
                    (
                        positions[id(parameter)],
                        tuple(parameter.shape),
                        str(parameter.dtype),
                        tuple(parameter.grad.shape),
                        str(parameter.grad.dtype),
                        (
                            tuple(map(str, parameter.grad.placements))
                            if isinstance(parameter.grad, DTensor)
                            else None
                        ),
                    )
                    for parameter in group.parameters
                ),
            )
            for group in groups.values()
        )
    if dist.is_initialized() and dist.get_world_size(error_group) > 1:
        signatures = [None] * dist.get_world_size(error_group)
        dist.all_gather_object(signatures, signature, group=error_group)
        if any(candidate != signature for candidate in signatures):
            raise ValueError(
                "Reference norm ranks declared incompatible ordered layouts"
            )
    replacements = {}
    for identities, group in groups.items():
        gradients = group.logical_gradients()
        if len(gradients) != len(identities):
            raise ValueError(
                "reference norm contributions disagree with parameter count"
            )
        for identity, contributions in zip(identities, gradients, strict=True):
            if any(not isinstance(value, torch.Tensor) for value in contributions):
                raise TypeError("reference norm contributions must be tensors")
            replacements[identity] = contributions
    return [
        gradient
        for parameter in parameters
        if parameter.grad is not None
        for gradient in replacements.get(id(parameter), (parameter.grad,))
    ]


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
