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
"""

from __future__ import annotations

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.nn.attention.flex_attention import BlockMask

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
    """Report a missing shard helper before attempting to call it."""
    if _context_parallel_shard is None:
        raise ImportError(
            "TP/CP input sharding requires torch's private helper "
            "``torch.distributed.tensor.experimental._context_parallel."
            "_context_parallel_shard``, which this torch build does not provide. "
            "Install a compatible torch build or disable TP/CP input sharding."
        )


def resolve_load_balancer(load_balancer: str | None, seq_len: int, cp_mesh: DeviceMesh):
    """Instantiate the named load balancer for one sequence length.

    The balancers are deterministic in ``(seq_len, cp_size, device)``, so the
    trainer sharding the batch and the wrapper sharding the BlockMask construct
    them independently and still agree on the rearrangement.
    """
    require_torch_input_sharding()
    if load_balancer is None:
        return None
    cp_size = cp_mesh.size()
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
    sharded = _context_parallel_shard(
        mesh=cp_mesh,
        buffers=[input_ids, labels, positions],
        seq_dims=[0, 0, 0],
        load_balancer=balancer,
    )
    return tuple(sharded)


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
    return _context_parallel_shard(
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
    require_torch_input_sharding()
    seq_len = buffers[0].shape[0]
    if seq_len % tp_mesh.size() != 0:
        raise ValueError(
            f"sequence length {seq_len} is not divisible by "
            f"tp={tp_mesh.size()}; seq_len must be a multiple of the TP degree."
        )
    return tuple(
        _context_parallel_shard(
            mesh=tp_mesh,
            buffers=buffers,
            seq_dims=[0] * len(buffers),
            load_balancer=None,
        )
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
    balancer = resolve_load_balancer(load_balancer, mask.seq_lengths[0], cp_mesh)
    return _context_parallel_shard(
        mesh=cp_mesh,
        buffers=[mask],
        seq_dims=[MASK_Q_SEQ_DIM],
        load_balancer=balancer,
    )[0]
