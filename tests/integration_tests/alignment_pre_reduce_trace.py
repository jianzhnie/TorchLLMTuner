"""Observe actual FSDP foreach_reduce inputs without changing their arithmetic."""

import inspect
import json
from functools import wraps
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor

from tests.integration_tests.alignment_parameter_hash import canonical_parameter_views


def pre_reduce_target_records(fsdp_params, gradients, targets):
    records = {}
    for parameter, gradient in zip(fsdp_params, gradients, strict=True):
        name = parameter._param_fqn
        if name is None:
            raise ValueError("FSDP pre-reduce trace requires actual parameter FQNs")
        if not any(token in name for token in ("inner_experts.", "mlp.experts.")):
            continue
        local = gradient.to_local() if isinstance(gradient, DTensor) else gradient
        if local.ndim != 3 or local.shape[0] != 128:
            raise ValueError("pre-reduce trace requires full DP8 expert dimension")
        for canonical, projection in canonical_parameter_views(name, local):
            for selected, expert, start, end in targets:
                if selected != canonical:
                    continue
                if not (
                    0 <= expert < projection.shape[0]
                    and 0 <= start < end <= projection.shape[1]
                ):
                    raise ValueError(
                        "pre-reduce target exceeds actual gradient geometry"
                    )
                key = f"{canonical}/expert{expert}/rows{start}:{end}"
                if key in records:
                    raise ValueError("duplicate pre-reduce target ownership")
                records[key] = {
                    "source_parameter": name,
                    "full_gradient_shape": tuple(local.shape),
                    "full_gradient_stride": tuple(local.stride()),
                    "gradient_dtype": str(local.dtype),
                    "gradient": projection[expert, start:end].detach().cpu().clone(),
                }
    return records


def install_pre_reduce_trace(trainer, root: Path, selection):
    from torch.distributed.fsdp._fully_shard import _fsdp_param_group as owner

    original = owner.foreach_reduce
    signature = inspect.signature(original)
    targets = json.loads(Path(selection).read_text())
    captured = {}

    @wraps(original)
    def observed(*args, **kwargs):
        if trainer.step == 1:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            parameters = bound.arguments["fsdp_params"]
            gradients = bound.arguments["unsharded_grads"]
            rows = pre_reduce_target_records(parameters, gradients, targets)
            if captured.keys() & rows.keys():
                raise ValueError("pre-reduce target captured by multiple FSDP groups")
            configuration = {
                key: str(value)
                for key, value in bound.arguments.items()
                if key
                in (
                    "orig_dtype",
                    "reduce_dtype",
                    "gradient_divide_factor",
                    "force_sum_reduction_for_comms",
                )
            }
            for row in rows.values():
                row["reduction_configuration"] = configuration
            captured.update(rows)
            if rows:
                torch.save(captured, root / "pre-reduce-step-1.pt")
        return original(*args, **kwargs)

    owner.foreach_reduce = observed

    def restore():
        if owner.foreach_reduce is observed:
            owner.foreach_reduce = original

    return restore
