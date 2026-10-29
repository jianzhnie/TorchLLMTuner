"""Shard model inputs across CP/TP and attention masks across CP.

This module is the one concentration point for llmtuner's use of torch's private
context-parallel helpers (``_context_parallel_shard`` and the load balancers).
They are private and may move between torch releases; keeping every import here
means a torch upgrade breaks exactly one file, and breaks it with an error that
says why.

Two sharding rules, both inherited from torchtitan's
``distributed/context_parallel/api.py``:

* token-carrying tensors (input ids, labels, positions) shard along their
  sequence dim, all with ONE shared load balancer so a token, its label and
  its position travel together;
* a ``BlockMask`` is built over the full sequence and shards only along its Q
  axis -- the KV axis stays full because every rank's queries attend against
  the gathered, full-length K/V.

With no load balancer each rank holds the contiguous ``T / cp`` block of the
sequence. With ``"headtail"`` each rank holds one head chunk and one tail
chunk, which balances the causal-mask triangle across ranks but leaves the
sequence *rearranged*: rank-local tokens are no longer contiguous in the
original order, so callers must not assume they are. The rearrangement is
consistent across every tensor sharded with the same balancer parameters, and
the gathered K/V come out in the same rearranged order, which is what keeps
the sharded BlockMask's indexing correct.

``_context_parallel_shard`` only exists from torch 2.12 on. The contiguous
*unbalanced* tensor partition has an exact public-API equivalent -- chunk along
the sequence dim and take this rank's chunk, in mesh rank order -- which
``_shard_buffers`` falls back to when the private helper is missing (the case
on the torch 2.9 Ascend/``torch_npu`` stack). Load-balanced partitions and
``BlockMask`` Q-sharding have no such equivalent and keep requiring the private
helper, loudly.
"""

from __future__ import annotations

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.nn.attention.flex_attention import BlockMask

from llmtuner.models.common.attention.masks import create_attention_mask

from .. import compatibility

__all__ = [
    "MASK_Q_SEQ_DIM",
    "shard_attention_mask_for_cp",
    "shard_batch_for_cp",
    "shard_batch_for_tp",
    "shard_padding_mask_for_cp",
    "shard_padding_mask_for_tp",
]

# BlockMask is (B, H, Q, KV); only the Q axis is sequence-sharded. The KV axis
# stays full: every rank attends its query shard against the full keys.
MASK_Q_SEQ_DIM = 2

try:
    from torch.distributed.tensor.experimental._attention import (
        _HeadTailLoadBalancer,
    )
except ImportError:  # pragma: no cover - depends on the torch build
    _HeadTailLoadBalancer = None

try:
    from torch.distributed.tensor.experimental._context_parallel import (
        _context_parallel_shard,
    )
except ImportError:  # pragma: no cover - exercised only on torch builds without these
    _context_parallel_shard = None


def require_torch_input_sharding() -> None:
    """Report a missing shard helper before attempting to call it.

    Only the paths without a public-API equivalent call this: a load-balanced
    partition, or ``BlockMask`` Q-sharding. Plain contiguous tensor partitions
    go through :func:`_shard_buffers`, which falls back instead.
    """
    if _context_parallel_shard is None:
        raise ImportError(
            "TP/CP input sharding requires torch's private helper "
            "``torch.distributed.tensor.experimental._context_parallel."
            "_context_parallel_shard``, which this torch build does not provide. "
            "Install a compatible torch build or disable TP/CP input sharding. "
            "The contiguous (unbalanced) tensor partition does not need it and "
            "is handled by the public-API fallback."
        )


def _contiguous_sequence_partition(
    buffers: list[torch.Tensor],
    seq_dims: list[int],
    mesh: DeviceMesh,
) -> tuple[torch.Tensor, ...]:
    """Public-API equivalent of torch's contiguous ``_context_parallel_shard``.

    Chunk every buffer along its sequence dim and keep this rank's chunk. Mesh
    rank order is the order the fused TP collectives reassemble the sequence in
    (``tensor_parallel/linear_ops.py``) and the order the CP K/V gather assumes,
    so this is the same partition torch's private helper produces when no load
    balancer is attached.
    """
    size = mesh.size()
    coord = mesh.get_local_rank()
    partitions = []
    for buffer, dim in zip(buffers, seq_dims, strict=True):
        length = buffer.shape[dim]
        if length % size != 0:
            raise ValueError(
                f"sequence dim {dim} has length {length}, not divisible by the "
                f"mesh size {size}"
            )
        partitions.append(torch.chunk(buffer, size, dim=dim)[coord])
    return tuple(partitions)


def _shard_buffers(
    *,
    mesh: DeviceMesh,
    buffers: list[torch.Tensor],
    seq_dims: list[int],
    load_balancer: object | None,
) -> tuple[torch.Tensor, ...]:
    """Partition token buffers across ``mesh``, using torch's helper when present.

    Torch's private helper wins whenever it exists, so the partition stays
    upstream-exact on builds that have it. The fallback covers only the
    contiguous unbalanced partition; a load balancer still needs the helper
    (its rearrangement is not reproducible from public APIs here).
    """
    if _context_parallel_shard is not None:
        return tuple(
            _context_parallel_shard(
                mesh=mesh,
                buffers=buffers,
                seq_dims=seq_dims,
                load_balancer=load_balancer,
            )
        )
    if load_balancer is not None:
        require_torch_input_sharding()
    return _contiguous_sequence_partition(buffers, seq_dims, mesh)


def resolve_load_balancer(load_balancer: str | None, seq_len: int, cp_mesh: DeviceMesh):
    """Instantiate the named load balancer for one sequence length.

    The balancers are deterministic in ``(seq_len, cp_size, device)``, so the
    trainer sharding the batch and the wrapper sharding the BlockMask construct
    them independently and still agree on the rearrangement.
    """
    cp_size = cp_mesh.size()
    # DTensor permits uneven shards, but CP attention communicates equal-sized
    # sequence partitions. Reject before sharding can create rank-local shapes.
    if seq_len % cp_size != 0:
        raise ValueError(
            f"sequence length {seq_len} is not divisible by cp={cp_size}; "
            "seq_len must be a multiple of the CP degree."
        )
    if load_balancer is None:
        return None
    if load_balancer == "headtail":
        if _HeadTailLoadBalancer is None:
            raise ImportError(
                "headtail load balancing requires torch's private "
                "_HeadTailLoadBalancer; install a compatible torch build "
                "or set parallel.context_parallel_load_balancer=None."
            )
        if seq_len % (2 * cp_size) != 0:
            raise ValueError(
                f"'headtail' load balancing requires seq_len ({seq_len}) to be "
                f"divisible by 2 * cp ({2 * cp_size}): each rank takes one head "
                "chunk and one tail chunk."
            )
        return _HeadTailLoadBalancer(seq_len, cp_size, cp_mesh.device_type)
    if load_balancer == "ptrr":
        compatibility.ptrr_load_balancer_backstop()
    raise ValueError(
        f"Unknown CP load balancer {load_balancer!r}; expected 'headtail' or None."
    )


def shard_batch_for_cp(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    positions: torch.Tensor,
    cp_mesh: DeviceMesh,
    load_balancer: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shard the flattened batch along the sequence dim across the CP mesh.

    Args:
        input_ids: ``(T,)`` flat token ids (the wrapper's forward contract).
        labels: ``(T,)`` next-token labels, sharded identically so the loss
            pairs each logit with the right target.
        positions: ``(T,)`` full-length positions. Sharded identically so RoPE
            follows each token to its rank.
        cp_mesh: the CP axis mesh.
        load_balancer: ``"headtail"`` for causal load balancing, ``None`` for a
            plain contiguous split.

    Returns:
        The sharded ``(input_ids, labels, positions)`` triple, each of length
        ``T / cp``. Under ``"headtail"`` they are rearranged (see the module
        docstring); the sum-reduced loss is permutation-invariant, so the
        trainer does not need to undo it.
    """
    balancer = resolve_load_balancer(load_balancer, input_ids.shape[0], cp_mesh)
    return _shard_buffers(
        mesh=cp_mesh,
        buffers=[input_ids, labels, positions],
        seq_dims=[0, 0, 0],
        load_balancer=balancer,
    )


def shard_padding_mask_for_cp(
    padding_mask: torch.Tensor,
    cp_mesh: DeviceMesh,
    load_balancer: str | None = None,
) -> torch.Tensor:
    """Shard a flat ``(T,)`` padding mask exactly as ``shard_batch_for_cp``.

    Kept a separate entry point rather than another buffer in
    ``shard_batch_for_cp`` so that function's triple return -- unpacked at
    every call site -- stays stable. The rearrangement still agrees with the
    batch shard: the load balancers are deterministic in
    ``(seq_len, cp_size, device)`` (see ``resolve_load_balancer``), so
    constructing one here for the same sequence lands the same tokens on this
    rank.
    """
    balancer = resolve_load_balancer(load_balancer, padding_mask.shape[0], cp_mesh)
    return _shard_buffers(
        mesh=cp_mesh,
        buffers=[padding_mask],
        seq_dims=[0],
        load_balancer=balancer,
    )[0]


def shard_padding_mask_for_tp(
    padding_mask: torch.Tensor,
    tp_mesh: DeviceMesh,
) -> torch.Tensor:
    """Shard a flat ``(T,)`` padding mask exactly as ``shard_batch_for_tp``.

    The same contiguous cut, for the same reason: the mask must follow the
    token stream it describes into this rank's ``T / tp`` slice.
    """
    return _shard_tp_buffers([padding_mask], tp_mesh)[0]


def _shard_tp_buffers(
    buffers: list[torch.Tensor], tp_mesh: DeviceMesh
) -> tuple[torch.Tensor, ...]:
    """Apply one contiguous sequence partition to related token buffers."""
    seq_len = buffers[0].shape[0]
    if seq_len % tp_mesh.size() != 0:
        raise ValueError(
            f"sequence length {seq_len} is not divisible by "
            f"tp={tp_mesh.size()}; seq_len must be a multiple of the TP degree."
        )
    return _shard_buffers(
        mesh=tp_mesh,
        buffers=buffers,
        seq_dims=[0] * len(buffers),
        load_balancer=None,
    )


def shard_batch_for_tp(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    tp_mesh: DeviceMesh,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shard the flattened batch along the sequence dim across the TP mesh.

    This is what makes sequence parallelism's premise true for TP: llmtuner's
    fused TP GEMMs (``AllGatherLinear`` / ``LinearReduceScatter``) all-gather
    the sequence *inside* the projection, so every rank must enter the forward
    holding only its ``T / tp`` slice. Without this cut the gather concatenates
    ``tp`` copies of the full sequence and every sharded weight gradient comes
    out ``tp`` times too large.

    Args:
        input_ids: ``(T,)`` flat token ids; already CP-sharded when CP is on,
            so the TP slice lands inside this rank's CP shard and the
            in-projection all-gather reassembles exactly that shard.
        labels: ``(T,)`` next-token labels, sharded identically so the loss
            pairs each logit with the right target.
        tp_mesh: the TP axis mesh.

    Returns:
        The sharded ``(input_ids, labels)`` pair, each of length ``T / tp``.

    Positions are deliberately not taken: after the in-projection gather each
    rank's attention and RoPE see the assembled sequence (the CP shard, or the
    full sequence with CP off), so the positions the forward needs are the
    CP-sharded -- or full -- ones, never a TP slice. The same holds for the
    attention mask, which is why Q-sharding stays a CP-only operation.

    Contiguous, never load-balanced: the causal-triangle imbalance a load
    balancer smooths exists between the assembled CP shards, so balancing is
    the CP shard's business; slicing inside it evenly keeps every TP rank at
    the same GEMM size.
    """
    input_shard, label_shard = _shard_tp_buffers([input_ids, labels], tp_mesh)
    return input_shard, label_shard


def shard_attention_mask_for_cp(
    mask: BlockMask,
    cp_mesh: DeviceMesh,
    load_balancer: str | None = None,
) -> BlockMask:
    """Q-shard a full-length BlockMask for CP.

    The mask must be built over the FULL sequence (``Q_LEN == KV_LEN == T``)
    before this call -- the document structure of a packed batch is a global
    property, so a mask built from a rank's local positions shard would be
    wrong. Sharding then keeps the KV axis full and slices only Q, and torch's
    ``_create_cp_block_mask`` rewrites the mask mod so the rank's local query
    indices map back to the right global ones (load-balancer rearrangement
    included).

    Note this inherits torch's constraint that ``Q_LEN`` be divisible by
    ``cp * BLOCK_SIZE`` (128): torch rebuilds the mask per rank rather than
    slicing the block indices.
    """
    if not isinstance(mask, BlockMask):
        raise TypeError(
            f"CP can only shard BlockMask attention masks, got {type(mask).__name__}."
        )
    # No public-API fallback: rebuilding a rank's BlockMask from the full-length
    # one is torch's ``_create_cp_block_mask``, not a slice.
    require_torch_input_sharding()
    balancer = resolve_load_balancer(load_balancer, mask.seq_lengths[0], cp_mesh)
    if cp_mesh.device_type == "npu":
        query_length, key_length = mask.seq_lengths
        local_length = query_length // cp_mesh.size()
        if query_length % (cp_mesh.size() * 128):
            raise NotImplementedError(
                "CP BlockMask query length must be divisible by CP size * 128"
            )
        offset = cp_mesh.get_local_rank() * local_length
        indices = (
            balancer._generate_indices(restore=False) if balancer is not None else None
        )

        def local_mask(batch, head, query, key):
            query = query + offset
            if indices is not None:
                batch_index = 0 if indices.size(0) == 1 else batch
                query = indices[batch_index, query]
                key = indices[batch_index, key]
            return mask.mask_mod(batch, head, query, key)

        return create_attention_mask(
            local_mask,
            B=mask.shape[0],
            H=mask.shape[1],
            Q_LEN=local_length,
            KV_LEN=key_length,
            device=cp_mesh.device_type,
        )
    return _shard_buffers(
        mesh=cp_mesh,
        buffers=[mask],
        seq_dims=[MASK_Q_SEQ_DIM],
        load_balancer=balancer,
    )[0]
