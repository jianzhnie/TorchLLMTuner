"""Model-declared logical gradients for reference norm rounding."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import torch
from torch import nn
from torch.distributed.tensor import DTensor, Shard

from ...errors import UnsupportedCombinationError

LogicalGradients = tuple[tuple[torch.Tensor, ...], ...]


@dataclass(frozen=True, slots=True)
class ReferenceNormGroup:
    """Bind parameters to ordered logical gradients without mutating storage.

    Each returned tuple replaces one parameter's norm contributions. An empty
    tuple means its gradient was incorporated in another group's contribution.
    The declaration is ephemeral and must be rebound after transformations.
    """

    parameters: tuple[nn.Parameter, ...]
    logical_gradients: Callable[[], LogicalGradients]
    validate_gradients: Callable[[], None]
    layout: str

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, tuple) or not self.parameters:
            raise TypeError("Reference norm parameters must be a nonempty tuple")
        if any(
            not isinstance(parameter, nn.Parameter) for parameter in self.parameters
        ):
            raise TypeError("Reference norm groups must bind Parameters")
        if not callable(self.logical_gradients):
            raise TypeError("Reference norm groups must provide a gradient builder")
        if not callable(self.validate_gradients):
            raise TypeError("Reference norm groups must provide a local validator")
        if not isinstance(self.layout, str) or not self.layout:
            raise ValueError("Reference norm groups require an explicit layout name")


def interleave_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Use [KV group, Q heads + K + V, dim] reference packing order."""
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


def _validate_qkv_gradients(
    projections: tuple[nn.Module, nn.Module, nn.Module],
    *,
    kv_heads: int,
    head_dim: int,
) -> None:
    gradients = [projection.weight.grad for projection in projections]
    if any(gradient is None for gradient in gradients):
        raise ValueError("reference norm requires complete QKV gradients")
    if any(projection.bias is not None for projection in projections):
        raise UnsupportedCombinationError(
            "reference QKV norm requires bias-free projections"
        )
    if any(
        not isinstance(gradient, DTensor) or gradient.placements != (Shard(0),)
        for gradient in gradients
    ):
        raise UnsupportedCombinationError("reference QKV norm requires DP Shard(0)")
    mesh = gradients[0].device_mesh
    if mesh.ndim != 1 or any(gradient.device_mesh != mesh for gradient in gradients):
        raise UnsupportedCombinationError("reference QKV norm requires one shared mesh")
    if type(kv_heads) is not int or type(head_dim) is not int:
        raise TypeError("reference QKV head geometry must contain integers")
    if kv_heads <= 0 or head_dim <= 0:
        raise ValueError("reference QKV head geometry must be positive")
    query, key, value = gradients
    if any(gradient.ndim != 2 for gradient in gradients):
        raise ValueError("reference norm requires matrix Q/K/V gradients")
    if (
        key.shape != value.shape
        or query.shape[1] != key.shape[1]
        or key.shape[0] != kv_heads * head_dim
        or query.shape[0] % (kv_heads * head_dim)
        or any(gradient.dtype != query.dtype for gradient in gradients)
    ):
        raise ValueError("reference norm QKV geometry or dtype is incompatible")
    if sum(gradient.shape[0] for gradient in gradients) % mesh.size():
        raise UnsupportedCombinationError(
            "reference QKV norm requires evenly sharded rows"
        )


def _qkv_gradients(
    projections: tuple[nn.Module, nn.Module, nn.Module],
    *,
    kv_heads: int,
    head_dim: int,
) -> LogicalGradients:
    gradients = [projection.weight.grad for projection in projections]
    mesh = gradients[0].device_mesh
    combined = interleave_qkv(
        *(gradient.full_tensor() for gradient in gradients),
        kv_heads=kv_heads,
        head_dim=head_dim,
    )
    rows = combined.shape[0]
    if rows % mesh.size():
        raise UnsupportedCombinationError(
            "reference QKV norm requires evenly sharded rows"
        )
    local_rows = rows // mesh.size()
    local = combined.narrow(0, mesh.get_local_rank() * local_rows, local_rows).clone()
    logical = DTensor.from_local(
        local,
        mesh,
        [Shard(0)],
        run_check=False,
        shape=combined.shape,
        stride=combined.stride(),
    )
    return ((logical,), (), ())


def qkv_reference_norm_group(
    projections: tuple[nn.Module, nn.Module, nn.Module], *, kv_heads: int, head_dim: int
) -> ReferenceNormGroup:
    """Declare separate Q/K/V projections with KV-group reference packing."""
    return ReferenceNormGroup(
        tuple(projection.weight for projection in projections),
        partial(_qkv_gradients, projections, kv_heads=kv_heads, head_dim=head_dim),
        partial(
            _validate_qkv_gradients, projections, kv_heads=kv_heads, head_dim=head_dim
        ),
        "kv_group_qkv",
    )


def _validate_expert_gradients(gate_up: nn.Parameter, down: nn.Parameter) -> None:
    if gate_up.grad is None or down.grad is None:
        raise ValueError("reference norm requires complete expert gradients")
    if gate_up.grad.ndim != 3 or gate_up.grad.shape[1] % 2:
        raise ValueError("reference expert norm requires packed gate/up geometry")


def _expert_gradients(gate_up: nn.Parameter, down: nn.Parameter) -> LogicalGradients:
    gate, up = gate_up.grad.chunk(2, dim=1)
    return ((gate,), (down.grad, up))


def packed_expert_reference_norm_group(
    gate_up: nn.Parameter, down: nn.Parameter
) -> ReferenceNormGroup:
    """Declare consecutive gate/up halves with reference gate/down/up order."""
    return ReferenceNormGroup(
        (gate_up, down),
        partial(_expert_gradients, gate_up, down),
        partial(_validate_expert_gradients, gate_up, down),
        "packed_gate_up",
    )
