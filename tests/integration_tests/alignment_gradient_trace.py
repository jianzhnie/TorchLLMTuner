"""Opt-in local gradient evidence before and after the real clipping call.

Never gather DTensors here: a 30B trace must not materialize full expert weights.
Samples are diagnostic, not a replacement for full-state acceptance.
"""

from __future__ import annotations

from functools import wraps
from math import prod
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor


def _flat_sample(local, size):
    if local.ndim == 0:
        return local.reshape(1)[:size].cpu().clone()
    indices = torch.arange(min(size, local.numel()), device=local.device)
    coordinates = (
        tuple(
            (indices // prod(local.shape[axis + 1 :])) % width
            for axis, width in enumerate(local.shape)
        )
        if local.numel()
        else (indices,)
    )
    return local[coordinates].cpu().clone().reshape(-1)[:size]


def local_gradient_record(gradient, *, sample_size=8192):
    local = gradient.to_local() if isinstance(gradient, DTensor) else gradient
    local = local.detach()
    record = {
        "global_shape": tuple(gradient.shape),
        "local_shape": tuple(local.shape),
        "dtype": str(local.dtype),
        "placements": [str(p) for p in gradient.placements]
        if isinstance(gradient, DTensor)
        else [],
        "mesh_ranks": gradient.device_mesh.mesh.tolist()
        if isinstance(gradient, DTensor)
        else [],
        "sample": _flat_sample(local, sample_size),
        "native_norm": torch.linalg.vector_norm(local).float().cpu(),
        "float32_norm": torch.linalg.vector_norm(local, dtype=torch.float32).cpu(),
    }
    # Packed gate/up uses axis 1. Preserve both halves' prefixes for comparison
    # with split expert parameters rather than sampling the gate half alone.
    if local.ndim == 3 and local.shape[1] % 2 == 0:
        record["half_samples"] = [
            _flat_sample(half, sample_size) for half in local.chunk(2, dim=1)
        ]
    return record


def observe_gradient_clipping(original, trainer, root: Path):
    names = {
        id(parameter): f"part{index}.{name}"
        for index, model in enumerate(trainer.model_parts)
        for name, parameter in model.named_parameters()
        if "layers.0." in name
    }

    @wraps(original)
    def observed(parameters, *args, **kwargs):
        parameters = list(parameters)

        def snapshot():
            return {
                names[id(parameter)]: local_gradient_record(parameter.grad)
                for parameter in parameters
                if id(parameter) in names and parameter.grad is not None
            }

        records = {"before": snapshot()}
        # Persist before clipping so even a failing collective retains evidence.
        path = root / f"gradient-clip-step-{trainer.step}.pt"
        torch.save(records, path)
        result = original(parameters, *args, **kwargs)
        records["after"] = snapshot()
        norm = result.to_local() if isinstance(result, DTensor) else result
        records["returned_norm"] = norm.detach().cpu().clone()
        torch.save(records, path)
        return result

    return observed
