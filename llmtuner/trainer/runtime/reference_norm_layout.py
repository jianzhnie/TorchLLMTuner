"""Explicit reference parameter grouping for native low-precision norm rounding."""

import torch
from torch.distributed.tensor import DTensor, Shard

from ...accelerator.collectives import clip_grad_norm_ as storage_clip_grad_norm_
from ...errors import UnsupportedCombinationError


def interleave_qkv(query, key, value, *, kv_heads, head_dim):
    """Use the actual converter's [KV group, Q heads + K + V, dim] order."""
    if key.shape != value.shape or query.ndim != 2 or key.ndim != 2:
        raise ValueError("reference norm requires matching matrix Q/K/V gradients")
    if query.shape[1] != key.shape[1] or key.shape[0] != kv_heads * head_dim:
        raise ValueError("reference norm QKV geometry is incompatible")
    hidden = query.shape[1]
    return torch.cat(
        (
            query.reshape(kv_heads, -1, head_dim, hidden),
            key.reshape(kv_heads, 1, head_dim, hidden),
            value.reshape(kv_heads, 1, head_dim, hidden),
        ),
        dim=1,
    ).flatten(0, 2)


@torch.no_grad()
def reference_norm_gradients(parameters, model_parts):
    """Return temporary logical gradients; never replace Parameters or .grad."""
    parameter_ids = {id(parameter) for parameter in parameters}
    replacements = {}
    skipped = set()
    for part in model_parts:
        for module in part.modules():
            if all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj")):
                projections = [
                    getattr(module, name) for name in ("q_proj", "k_proj", "v_proj")
                ]
                weights = [projection.weight for projection in projections]
                if id(weights[0]) in replacements:
                    continue
                if not all(
                    id(weight) in parameter_ids and weight.grad is not None
                    for weight in weights
                ):
                    raise ValueError("reference norm requires complete QKV gradients")
                gradients = [weight.grad for weight in weights]
                if any(projection.bias is not None for projection in projections):
                    raise UnsupportedCombinationError(
                        "reference QKV norm requires bias-free projections"
                    )
                if any(
                    not isinstance(g, DTensor) or g.placements != (Shard(0),)
                    for g in gradients
                ):
                    raise UnsupportedCombinationError(
                        "reference QKV norm requires DP Shard(0)"
                    )
                mesh = gradients[0].device_mesh
                if mesh.ndim != 1 or any(g.device_mesh != mesh for g in gradients):
                    raise UnsupportedCombinationError(
                        "reference QKV norm requires one shared mesh"
                    )
                full = [gradient.full_tensor() for gradient in gradients]
                combined = interleave_qkv(
                    *full,
                    kv_heads=module.config.num_key_value_heads,
                    head_dim=module.head_dim,
                )
                rows = combined.shape[0]
                if rows % mesh.size():
                    raise UnsupportedCombinationError(
                        "reference QKV norm requires evenly sharded rows"
                    )
                local_rows = rows // mesh.size()
                local = combined.narrow(
                    0, mesh.get_local_rank() * local_rows, local_rows
                ).clone()
                logical = DTensor.from_local(
                    local,
                    mesh,
                    [Shard(0)],
                    run_check=False,
                    shape=combined.shape,
                    stride=combined.stride(),
                )
                replacements[id(weights[0])] = [logical]
                skipped.update(id(weight) for weight in weights[1:])
                del full, combined
            if hasattr(module, "gate_up_proj") and hasattr(module, "down_proj"):
                gate_up, down = module.gate_up_proj, module.down_proj
                if not isinstance(gate_up, torch.Tensor) or not isinstance(
                    down, torch.Tensor
                ):
                    continue
                if gate_up.grad is None or down.grad is None:
                    raise ValueError(
                        "reference norm requires complete expert gradients"
                    )
                if gate_up.grad.ndim != 3 or gate_up.grad.shape[1] % 2:
                    raise ValueError(
                        "reference expert norm requires packed gate/up geometry"
                    )
                gate, up = gate_up.grad.chunk(2, dim=1)
                replacements[id(gate_up)] = [gate]
                replacements[id(down)] = [down.grad, up]
    if not replacements:
        raise UnsupportedCombinationError(
            "reference norm found no supported parameter groups"
        )
    return [
        gradient
        for parameter in parameters
        if parameter.grad is not None and id(parameter) not in skipped
        for gradient in replacements.get(id(parameter), [parameter.grad])
    ]


def clip_grad_norm_(parameters, *args, reference_model_parts=None, **kwargs):
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
    gradients = reference_norm_gradients(parameters, reference_model_parts)
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
