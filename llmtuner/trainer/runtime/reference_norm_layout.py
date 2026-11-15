"""Explicit reference parameter grouping for native low-precision norm rounding."""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from ...accelerator.collectives import clip_grad_norm_ as storage_clip_grad_norm_
from ...accelerator.collectives import synchronize_local_errors
from ...errors import UnsupportedCombinationError
from ...models.common.gradient_norm import ReferenceNormGroup


def _collect_reference_groups(parameters, model_parts):
    """Bind and validate local declarations before gradient collectives."""
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
    """Consume model declarations without inferring projection or expert storage."""
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
    """Keep storage behavior unless the explicit reference layout is selected."""
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
    gradients = reference_norm_gradients(
        parameters, reference_model_parts, error_group=reference_error_group
    )
    # Match the reference's contiguous per-parameter kernel inputs without
    # retaining copies of every expert half (several GiB for a full model).
    # This is get_total_norm(foreach=False)'s native norm/stack/norm sequence.
    norms = [torch.linalg.vector_norm(gradient.contiguous()) for gradient in gradients]
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
