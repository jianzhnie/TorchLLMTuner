"""Opt-in local gradient evidence before and after the real clipping call.

Never gather DTensors here: a 30B trace must not materialize full expert weights.
Samples are diagnostic, not a replacement for full-state acceptance.
"""

from __future__ import annotations

import os
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
    native_norm = torch.linalg.vector_norm(local)
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
        "native_norm": native_norm.float().cpu(),
        "native_norm_dtype": str(native_norm.dtype),
        "float32_norm": torch.linalg.vector_norm(local, dtype=torch.float32).cpu(),
    }
    # Packed gate/up uses axis 1. Preserve both halves' prefixes for comparison
    # with split expert parameters rather than sampling the gate half alone.
    if local.ndim == 3 and local.shape[1] % 2 == 0:
        record["half_samples"] = [
            _flat_sample(half, sample_size) for half in local.chunk(2, dim=1)
        ]
    return record


def _is_qkv_parameter(name):
    return any(
        name.endswith(f".{projection}.{kind}")
        for projection in (
            "attention.qkv_linear.wqkv",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
        )
        for kind in ("weight", "bias")
    )


def _is_outer_parameter(name):
    return name in {
        "tok_embeddings.weight",
        "norm.weight",
        "lm_head.weight",
        "output.weight",
        "model.model.embed_tokens.weight",
        "model.model.norm.weight",
        "model.lm_head.weight",
    }


def observe_gradient_clipping(original, trainer, root: Path):
    restore_pre_reduce = None
    if os.environ.get("LLMTUNER_ALIGNMENT_PRE_REDUCE_TRACE") == "1":
        from tests.integration_tests.alignment_pre_reduce_trace import (
            install_pre_reduce_trace,
        )

        restore_pre_reduce = install_pre_reduce_trace(
            trainer, root, os.environ["LLMTUNER_ALIGNMENT_TARGET_CHUNKS"]
        )
    selection = os.environ.get("LLMTUNER_ALIGNMENT_GRADIENT_LAYERS", "0")
    if not selection or any(not item.isdecimal() for item in selection.split(",")):
        raise ValueError("gradient trace layers must be comma-separated indices")
    prefixes = tuple(f"layers.{int(item)}." for item in selection.split(","))
    all_names = {
        id(parameter): f"part{index}.{name}"
        for index, model in enumerate(trainer.model_parts)
        for name, parameter in model.named_parameters()
    }
    full_qkv = os.environ.get("LLMTUNER_ALIGNMENT_FULL_QKV") == "1"
    full_outer = os.environ.get("LLMTUNER_ALIGNMENT_FULL_OUTER") == "1"
    if full_qkv and full_outer:
        raise ValueError("full QKV and outer captures must be separate")
    names = {
        id(parameter): all_names[id(parameter)]
        for model in trainer.model_parts
        for name, parameter in model.named_parameters()
        if (
            _is_outer_parameter(name)
            if full_outer
            else any(prefix in name for prefix in prefixes)
            and (not full_qkv or _is_qkv_parameter(name))
        )
    }

    @wraps(original)
    def observed(parameters, *args, **kwargs):
        parameters = list(parameters)
        if restore_pre_reduce is not None and trainer.step >= 2:
            restore_pre_reduce()
        if os.environ.get(
            "LLMTUNER_ALIGNMENT_PARAMETER_HASH"
        ) == "1" and trainer.step in (1, 2):
            from tests.integration_tests.alignment_parameter_hash import (
                save_parameter_hashes,
            )

            save_parameter_hashes(
                parameters,
                all_names,
                root / f"parameter-hash-step-{trainer.step}.jsonl",
            )

        def snapshot():
            targets = os.environ.get("LLMTUNER_ALIGNMENT_TARGET_CHUNKS")
            if targets:
                from tests.integration_tests.alignment_targeted_chunks import (
                    targeted_expert_records,
                )

                return targeted_expert_records(parameters, all_names, trainer, targets)
            if os.environ.get("LLMTUNER_ALIGNMENT_PARAMETER_HASH") == "1" and not (
                full_qkv or full_outer
            ):
                return {}
            records = {}
            for parameter in parameters:
                if id(parameter) not in names or parameter.grad is None:
                    continue
                record = local_gradient_record(parameter.grad)
                if os.environ.get("LLMTUNER_ALIGNMENT_PARAMETER_SAMPLES") == "1":
                    local = (
                        parameter.to_local()
                        if isinstance(parameter, DTensor)
                        else parameter
                    ).detach()
                    record["parameter_sample"] = _flat_sample(local, 8192)
                    if local.ndim == 3 and local.shape[1] % 2 == 0:
                        record["parameter_half_samples"] = [
                            _flat_sample(half, 8192) for half in local.chunk(2, dim=1)
                        ]
                if full_qkv or full_outer:
                    gradient = parameter.grad
                    local_gradient = (
                        gradient.to_local()
                        if isinstance(gradient, DTensor)
                        else gradient
                    )
                    local_parameter = (
                        parameter.to_local()
                        if isinstance(parameter, DTensor)
                        else parameter
                    )
                    record["full_local_gradient"] = (
                        local_gradient.detach().cpu().clone()
                    )
                    record["full_local_parameter"] = (
                        local_parameter.detach().cpu().clone()
                    )
                records[names[id(parameter)]] = record
            return records

        records = {
            "before": snapshot(),
            "parameter_order": [
                {
                    "name": all_names[id(parameter)],
                    "shape": tuple(parameter.shape),
                    "dtype": str(parameter.dtype),
                    "gradient_dtype": str(parameter.grad.dtype)
                    if parameter.grad is not None
                    else None,
                }
                for parameter in parameters
            ],
        }
        # Persist before clipping so even a failing collective retains evidence.
        path = root / f"gradient-clip-step-{trainer.step}.pt"
        torch.save(records, path)
        result = original(parameters, *args, **kwargs)
        records["after"] = snapshot()
        norm = result.to_local() if isinstance(result, DTensor) else result
        records["returned_norm"] = norm.detach().cpu().clone()
        max_norm = args[0] if args else kwargs["max_norm"]
        records["max_norm"] = max_norm
        records["clip_coefficient_from_returned_norm"] = (
            (max_norm / (norm.detach() + 1e-6)).clamp(max=1).cpu()
        )
        torch.save(records, path)
        return result

    return observed
