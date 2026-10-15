"""Token alignment through actual PyTorch CP and TP input partitions.

The fake process group supplies mesh coordinates; no collective or attention
kernel is validated by these local DTensor partitions.
"""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed.fake_pg import FakeStore

from llmtuner.parallel.context_parallel.input_sharding import (
    shard_batch_for_cp,
    shard_batch_for_tp,
    shard_padding_mask_for_cp,
    shard_padding_mask_for_tp,
)


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("load_balancer", [None, "headtail"])
def test_cp_then_tp_preserves_token_fields(rank, load_balancer):
    dist.init_process_group("fake", store=FakeStore(), rank=rank, world_size=4)
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("cp", "tp"))
        cp_rank, tp_rank = divmod(rank, 2)
        tokens = torch.arange(16)
        labels = tokens + 100
        positions = tokens % 5
        padding = tokens % 3 == 1
        original = [x.clone() for x in (tokens, labels, positions, padding)]
        if load_balancer is None:
            cp_indices = torch.arange(cp_rank * 8, (cp_rank + 1) * 8)
        else:
            # Four chunks: each CP rank owns its head and opposite tail.
            cp_indices = torch.cat(
                [
                    torch.arange(cp_rank * 4, (cp_rank + 1) * 4),
                    torch.arange((3 - cp_rank) * 4, (4 - cp_rank) * 4),
                ]
            )
        tokens_cp, labels_cp, positions_cp = shard_batch_for_cp(
            tokens, labels, positions, mesh["cp"], load_balancer
        )
        padding_cp = shard_padding_mask_for_cp(padding, mesh["cp"], load_balancer)
        for actual, source in zip(
            (tokens_cp, labels_cp, positions_cp, padding_cp), original, strict=True
        ):
            torch.testing.assert_close(actual, source[cp_indices])
        tokens_tp, labels_tp = shard_batch_for_tp(tokens_cp, labels_cp, mesh["tp"])
        padding_tp = shard_padding_mask_for_tp(padding_cp, mesh["tp"])
        final_indices = cp_indices[tp_rank * 4 : (tp_rank + 1) * 4]
        for actual, source in zip(
            (tokens_tp, labels_tp, padding_tp),
            (tokens, labels, padding),
            strict=True,
        ):
            torch.testing.assert_close(actual, source[final_indices])
        # RoPE consumes the CP sequence after projection gathers the TP slices.
        torch.testing.assert_close(positions_cp, positions[cp_indices])
        for source, snapshot in zip(
            (tokens, labels, positions, padding), original, strict=True
        ):
            torch.testing.assert_close(source, snapshot)
    finally:
        dist.destroy_process_group()
