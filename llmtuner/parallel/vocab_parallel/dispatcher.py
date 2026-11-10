"""Route token IDs to vocabulary owners with variable-size all-to-all."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from .comm import alltoallv, alltoallv_with_gradient_divide_factor

__all__ = ["vocab_parallel_dispatch_forward"]


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    while isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return tensor


def _exchange_counts(
    send_counts: torch.Tensor, group: dist.ProcessGroup | None
) -> torch.Tensor:
    if group is None or dist.get_world_size(group) == 1:
        return send_counts
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts.contiguous(), group=group)
    return recv_counts


def _permutation(owners: torch.Tensor, vp_size: int) -> torch.Tensor:
    # Integer argsort falls back to AI CPU on NPU. Owner ranks are exact in
    # float32 within this bound; IDs themselves remain int64 throughout.
    keys = owners
    if owners.device.type == "npu" and vp_size <= 2**24:
        keys = owners.to(torch.float32)
    return torch.argsort(keys)


def _inverse_permutation(permutation: torch.Tensor) -> torch.Tensor:
    if permutation.device.type == "npu" and permutation.numel() < 2**24:
        return torch.argsort(permutation.to(torch.float32))
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel(), device=permutation.device)
    return inverse


def vocab_parallel_dispatch_forward(
    module: nn.Embedding,
    input_ids: torch.Tensor,
    vp_group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """Return owner-routed embeddings in the original token order.

    The VP group is also a data-parallel group: every rank enters with different
    tokens and leaves with its own tokens' embeddings. The module weight must be
    sharded along vocabulary dim 0 over the same group.
    """
    vp_size = 1 if vp_group is None else dist.get_world_size(vp_group)
    vp_rank = 0 if vp_group is None else dist.get_rank(vp_group)
    local_vocab_size = module.num_embeddings // vp_size
    flat_ids = input_ids.reshape(-1).long()
    owners = torch.div(flat_ids, local_vocab_size, rounding_mode="floor")
    permutation = _permutation(owners, vp_size)
    sorted_owners = owners[permutation]
    sorted_local_ids = flat_ids[permutation] - sorted_owners * local_vocab_size

    owner_ids = torch.arange(vp_size, device=owners.device)
    send_counts_tensor = (
        (owners.unsqueeze(-1) == owner_ids).sum(dim=0).to(dtype=torch.int64)
    )
    recv_counts_tensor = _exchange_counts(send_counts_tensor, vp_group)
    send_counts = send_counts_tensor.cpu().tolist()
    recv_counts = recv_counts_tensor.cpu().tolist()

    received_ids = alltoallv(sorted_local_ids, vp_group, send_counts, recv_counts)
    local_padding_idx = None
    if module.padding_idx is not None:
        vocab_start = vp_rank * local_vocab_size
        if vocab_start <= module.padding_idx < vocab_start + local_vocab_size:
            local_padding_idx = module.padding_idx - vocab_start
    local_embeddings = F.embedding(
        received_ids,
        _local_tensor(module.weight),
        padding_idx=local_padding_idx,
        norm_type=module.norm_type,
    )
    output_dtype = module.vp_output_dtype
    if output_dtype is not None and local_embeddings.dtype != output_dtype:
        local_embeddings = local_embeddings.to(dtype=output_dtype)
    sorted_embeddings = alltoallv_with_gradient_divide_factor(
        local_embeddings,
        vp_group,
        recv_counts,
        send_counts,
        gradient_divide_factor=module.vp_gradient_divide_factor,
    )
    output = sorted_embeddings[_inverse_permutation(permutation)]
    return output.view(*input_ids.shape, module.embedding_dim)
