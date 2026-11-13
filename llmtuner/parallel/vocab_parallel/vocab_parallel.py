"""Vocabulary-parallel embeddings for the tensor-parallel sequence layout."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import _functional_collectives as funcol

from ...accelerator.linear_collectives import all_gather_along

__all__ = ["VocabParallelEmbedding", "vocab_parallel_embedding_forward"]


def vocab_parallel_embedding_forward(
    module: nn.Embedding, input_ids: torch.Tensor
) -> torch.Tensor:
    """Look up this rank's token shard using only its vocabulary slice.

    TP first gathers the token shards because each vocabulary owner needs every
    token. Out-of-range IDs select a zero row and are masked after the lookup;
    summing selects the owner. The output then returns to the original token
    shard consumed by the decoder.
    """
    group = module.tp_group
    world_size = 1 if group is None else dist.get_world_size(group)
    rank = 0 if group is None else dist.get_rank(group)
    gathered_ids = (
        input_ids if world_size == 1 else all_gather_along(input_ids, -1, group)
    )
    local_ids = gathered_ids - module.vocab_start_index
    local_vocab_size = module.vocab_end_index - module.vocab_start_index
    invalid = (local_ids < 0) | (local_ids >= local_vocab_size)
    local_ids = torch.where(invalid, 0, local_ids)
    output = F.embedding(
        local_ids,
        module.weight,
        padding_idx=module.local_padding_idx,
        norm_type=module.norm_type,
    )
    output = output.masked_fill(invalid.unsqueeze(-1), 0)
    if world_size > 1:
        output = funcol.all_reduce(output.contiguous(), dist.ReduceOp.SUM.name, group)
        output = output.chunk(world_size, dim=-2)[rank].contiguous()
    return output


class VocabParallelEmbedding(nn.Embedding):
    """An ``nn.Embedding`` holding a contiguous vocabulary shard.

    ``weight`` is already this rank's slice of the original ``[V, H]`` table.
    Inputs and outputs retain their original token-sharded layout. ``max_norm``,
    sparse gradients and frequency-scaled gradients are rejected because they
    require coordination beyond the single output reduction.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        source: nn.Embedding,
        vocab_start_index: int,
        vocab_end_index: int,
        tp_group: dist.ProcessGroup | None,
    ) -> None:
        if source.max_norm is not None:
            raise ValueError("Vocab parallel does not support nn.Embedding.max_norm")
        if source.scale_grad_by_freq:
            raise ValueError(
                "Vocab parallel does not support nn.Embedding.scale_grad_by_freq"
            )
        if source.sparse:
            raise ValueError(
                "Vocab parallel does not support sparse embedding gradients"
            )
        super().__init__(
            source.num_embeddings,
            source.embedding_dim,
            padding_idx=None,
            device=weight.device,
            dtype=weight.dtype,
        )
        self.weight = nn.Parameter(weight, requires_grad=source.weight.requires_grad)
        self.vocab_start_index = vocab_start_index
        self.vocab_end_index = vocab_end_index
        self.tp_group = tp_group
        self.local_padding_idx = (
            source.padding_idx - vocab_start_index
            if source.padding_idx is not None
            and vocab_start_index <= source.padding_idx < vocab_end_index
            else None
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return vocab_parallel_embedding_forward(self, input_ids)
