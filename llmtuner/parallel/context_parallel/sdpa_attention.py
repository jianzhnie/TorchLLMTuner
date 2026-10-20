"""CP attention on the sdpa backend: the same redistribution, a tensor mask.

``attention.py``'s kernel ends in flex, which the NPU builds do not have:
llmtuner selects the flex backend on CUDA only (``models/hf/attention_backend``),
and its kv path additionally needs torch's private ``flex_cp_allgather``, which
this torch does not carry. The *redistribution* halves are backend independent
-- ``tests/integration_tests/cp_equivalence.py`` verifies them on CPU/gloo,
including the all-to-all round trip -- so this module reuses those shapes and
replaces only the last step and what it consumes.

Two things differ from the flex kernel, both forced by the backend:

* flex takes a ``BlockMask``; sdpa takes a boolean tensor. For kv_allgather on
  a single causal document the mask follows from ``(cp_rank, cp_size, local Q
  len)`` alone, so it is built here rather than sharded.
* the K/V gather carries its own backward. ``torch.distributed.nn.functional.
  all_gather``'s backward is the DP-style one -- it assumes every rank holds the
  same loss over the gathered tensor -- and on this stack it did not deliver the
  CP-correct gradient (measured against a full-sequence reference: the gathered
  tensor's gradient carried only the local queries' contribution, magnitude
  2.0065 against the reference's 3.2348). CP's contract is explicit here: every
  rank's K/V rows are attended by every rank's queries, so the gradient of the
  block a rank owns is the SUM over ranks of their gradient for that block --
  which is what :class:`_GatherSequence` does (reduce the whole gathered
  gradient, then slice this rank's block).

Both strategies are expressed, each with the mask its redistribution makes
derivable: ``kv_allgather`` rebuilds the causal mask from the CP offsets (its
forward cost is a single gather), while ``ulysses`` attends the full sequence
after the head swap, so the plain causal mask -- or the wrapper's full-length
document mask for a packed corpus -- applies unsharded.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

from .attention import HeadToSeq, SeqToHead

__all__ = ["CPSdpaKernel", "CPSdpaUlyssesKernel", "causal_mask_for_local_queries"]

# q/k/v arrive HF-shaped: (batch, heads, seq, dim); the CP input sharding split
# the sequence axis, dim 2 of this layout.
_SEQ_DIM = 2


class _GatherSequence(torch.autograd.Function):
    """All-gather along the sequence dim, with the CP gradient contract.

    Forward concatenates every rank's block in group order. Backward reduces the
    *whole* gathered gradient across the group and then keeps this rank's block:
    each rank's loss depends on every block of the gathered tensor, so the
    gradient of the block this rank owns is the sum over ranks of their
    contribution to that block.

    Reducing before slicing matters: slicing first and reducing afterwards adds
    together different blocks, which produced sign-flipped values in the
    hand-checkable case (seq=2: ``-0.1738`` against a reference of ``+0.1738``).
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group, ranks: tuple[int, ...]):
        ctx.group = group
        ctx.ranks = ranks
        ctx.world = len(ranks)
        local = tensor.contiguous()
        local_blocks: list[torch.Tensor] = [torch.empty_like(local) for _ in ranks]
        dist.all_gather(local_blocks, local, group=group)
        return torch.cat(local_blocks, dim=_SEQ_DIM)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        total = grad_output.contiguous().clone()
        dist.all_reduce(total, group=ctx.group)
        rank = ctx.ranks.index(dist.get_rank())
        local_len = total.shape[_SEQ_DIM] // ctx.world
        block = total[:, :, rank * local_len : (rank + 1) * local_len]
        return block, None, None


def causal_mask_for_local_queries(
    *,
    local_len: int,
    full_len: int,
    offset: int,
    device: torch.device,
) -> torch.Tensor:
    """``(1, 1, local_len, full_len)`` bool: query ``i`` attends KV ``j <= offset + i``.

    The contiguous CP split gives a rank its token block starting at
    ``offset = cp_rank * local_len``, so causality for its queries is the
    full-sequence condition shifted by the block start. Shape and dtype follow
    sdpa's convention: boolean, ``True`` means the pair takes part.
    """
    rows = torch.arange(local_len, device=device).unsqueeze(1) + offset
    cols = torch.arange(full_len, device=device).unsqueeze(0)
    return (cols <= rows).view(1, 1, local_len, full_len)


def full_length_causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """``(1, 1, length, length)`` bool lower-triangular mask for the ulysses swap.

    After the token-to-head all-to-all every rank holds the *full* sequence, so
    causality is the plain unshifted condition -- no CP offset enters here, which
    is exactly why this strategy's mask does not need the flex BlockMask (or the
    flex backend).
    """
    rows = torch.arange(length, device=device).unsqueeze(1)
    cols = torch.arange(length, device=device).unsqueeze(0)
    return (cols <= rows).view(1, 1, length, length)


class CPSdpaKernel(nn.Module):
    """Redistribute q/k/v for CP, attend with sdpa, hand the shard back.

    Attached by ``apply_cp`` in place of the flex kernel when the model runs the
    sdpa backend. The forward contract matches the flex kernel's: q/k/v arrive
    HF-shaped and already sequence-sharded, and the output is this rank's query
    shard in the same layout.
    """

    def __init__(self, *, cp_mesh: DeviceMesh, packed: bool = False) -> None:
        super().__init__()
        if packed:
            raise NotImplementedError(
                "The sdpa CP kernel does not express a packed corpus's document "
                "mask: sdpa takes a dense boolean mask and the document "
                "structure would have to be materialized as one. Run packing "
                "with the flex backend (CUDA), or cp=1."
            )
        self.cp_size = cp_mesh.size()
        self.cp_rank = cp_mesh.get_local_rank()
        self._cp_group = cp_mesh.get_group()
        # Resolved on first forward, not here: construction stays usable with a
        # duck-typed mesh (the flex kernel's arrangement, and what the unit tests
        # build), and a rank list is only meaningful once a collective runs.
        self._cp_ranks: tuple[int, ...] | None = None
        # Deterministic in the offsets, so one entry per shape per layer.
        self._masks: dict[tuple[int, int, str], torch.Tensor] = {}

    def _ranks(self) -> tuple[int, ...]:
        if self._cp_ranks is None:
            self._cp_ranks = tuple(dist.get_process_group_ranks(self._cp_group))
        return self._cp_ranks

    def _mask(
        self, local_len: int, full_len: int, device: torch.device
    ) -> torch.Tensor:
        key = (local_len, full_len, str(device))
        mask = self._masks.get(key)
        if mask is None:
            mask = causal_mask_for_local_queries(
                local_len=local_len,
                full_len=full_len,
                offset=self.cp_rank * local_len,
                device=device,
            )
            self._masks[key] = mask
        return mask

    def forward(self, query, key, value, *, module=None, block_mask=None, **kwargs):
        """Gather K/V, attend the local queries against them, return the shard."""
        ranks = self._ranks()
        key = _GatherSequence.apply(key, self._cp_group, ranks)
        value = _GatherSequence.apply(value, self._cp_group, ranks)
        mask = self._mask(query.shape[_SEQ_DIM], key.shape[_SEQ_DIM], query.device)
        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=0.0,
            scale=kwargs.get("scaling"),
            enable_gqa=query.shape[1] != key.shape[1],
        )
        # HF's interface contract is (batch, seq, heads, dim).
        return out.transpose(1, 2)


class CPSdpaUlyssesKernel(nn.Module):
    """Ulysses CP on sdpa: swap the token shard for a head shard, attend, swap back.

    The flex kernel's ulysses path ends in a flex call whose mask is a
    ``BlockMask``; this one needs none of that, because the swap puts the *full*
    sequence on every rank -- the mask is the plain lower-triangular one built
    from the length alone, plus (for a packed corpus) the dense document mask the
    wrapper hands over, which is already causal-and-same-document.

    The all-to-all pair is ``attention.py``'s ``SeqToHead`` / ``HeadToSeq``: the
    same exchange the flex kernel drives, with the same backward. Only the final
    attention call differs, which is what makes this expressible without flex.

    Refusals that stay in ``apply_cp``: a load balancer (the rearrangement lands
    in the mask's query order, which a length-derived mask cannot express) and
    the dual-axis form (its mask is not written for sdpa).
    """

    def __init__(self, *, cp_mesh: DeviceMesh, packed: bool = False) -> None:
        super().__init__()
        self.cp_size = cp_mesh.size()
        self.packed = packed
        self._cp_group = cp_mesh.get_group()
        # Deterministic in the length, so one entry per shape per layer.
        self._masks: dict[tuple[int, str], torch.Tensor] = {}

    def _causal(self, length: int, device: torch.device) -> torch.Tensor:
        key = (length, str(device))
        mask = self._masks.get(key)
        if mask is None:
            mask = full_length_causal_mask(length, device)
            self._masks[key] = mask
        return mask

    def forward(self, query, key, value, *, module=None, block_mask=None, **kwargs):
        """Swap to heads, attend the full sequence, swap back to this rank's shard."""
        q = SeqToHead.apply(query.contiguous(), self._cp_group)
        k = SeqToHead.apply(key.contiguous(), self._cp_group)
        v = SeqToHead.apply(value.contiguous(), self._cp_group)

        length = q.shape[_SEQ_DIM]
        if block_mask is not None and block_mask.dim() == 4:
            # A packed corpus's document mask, built full-length by the wrapper:
            # it already carries causality and the document split.
            mask = block_mask
        else:
            if self.packed:
                raise ValueError(
                    "ulysses CP received no document mask for a packed corpus: "
                    "the document structure cannot be recovered from a length "
                    "alone. The wrapper builds it full-length for this kernel."
                )
            mask = self._causal(length, q.device)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            scale=kwargs.get("scaling"),
            enable_gqa=q.shape[1] != k.shape[1],
        )
        out = HeadToSeq.apply(out.contiguous(), self._cp_group)
        # HF's interface contract is (batch, seq, heads, dim).
        return out.transpose(1, 2)
