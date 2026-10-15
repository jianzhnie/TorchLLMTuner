"""Four-rank CPU/Gloo dual CP mesh membership and collective checks.

Run: PYTHONPATH=. torchrun --nproc_per_node=4 \
    tests/integration_tests/dual_cp_topology.py

Checks KV-major/Ulysses-minor rank order, subgroup SUM and mask indices.
This does not run attention kernels or establish target-device numerical equivalence.
"""

from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.attention.flex_attention import and_masks, create_block_mask

from llmtuner.models.common.attention.masks import get_document_mask_mod
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.parallel.context_parallel.input_sharding import (
    shard_attention_mask_for_cp,
)
from llmtuner.parallel.topology import ParallelTopology


def main() -> None:
    dist.init_process_group("gloo")
    try:
        assert dist.get_world_size() == 4
        rank = dist.get_rank()
        topology = ParallelTopology(
            dp_replicate=1,
            dp_shard=1,
            cp=4,
            tp=1,
            pp=1,
            ep=1,
            world_size=4,
            cp_kv=2,
            cp_ulysses=2,
        )
        topology.build_device_mesh()
        expected = {
            "cp": [0, 1, 2, 3],
            "cp_kv": [rank % 2, rank % 2 + 2],
            "cp_ulysses": [rank // 2 * 2, rank // 2 * 2 + 1],
        }
        for axis, ranks in expected.items():
            mesh = topology.get_optional_mesh(axis)
            assert mesh is not None
            assert mesh.mesh.tolist() == ranks
            assert mesh.get_local_rank() == ranks.index(rank)
            value = torch.tensor(float(rank + 1))
            dist.all_reduce(value, group=mesh.get_group())
            assert value.item() == sum(member + 1 for member in ranks)
        kv = topology.get_optional_mesh("cp_kv")
        ulysses = topology.get_optional_mesh("cp_ulysses")
        assert set(kv.mesh.tolist()) & set(ulysses.mesh.tolist()) == {rank}

        def causal(batch, head, query, key):
            return query >= key

        mask = create_block_mask(
            causal, B=1, H=1, Q_LEN=512, KV_LEN=512, device="cpu", _compile=False
        )
        local_mask = shard_attention_mask_for_cp(mask, kv)
        assert local_mask.seq_lengths == (256, 512)
        queries = torch.arange(256).view(-1, 1)
        keys = torch.arange(512).view(1, -1)
        actual = local_mask.mask_mod(torch.tensor(0), torch.tensor(0), queries, keys)
        expected_mask = queries + (rank // 2) * 256 >= keys
        assert torch.equal(actual, expected_mask)

        def build_mask(*, positions):
            assert torch.equal(positions, torch.arange(512))
            return mask

        stub = SimpleNamespace(
            uses_dsa=False,
            model=SimpleNamespace(config=SimpleNamespace(attn_mask_type="causal")),
            cp_mesh=topology.get_optional_mesh("cp"),
            _cp_mask_mesh=kv,
            _cp_strategy="kv_allgather",
            _cp_load_balancer=None,
            get_attention_masks=build_mask,
        )
        rebuilt = HFTransformerModel._get_cp_attention_masks(
            stub, positions=torch.arange(rank * 128, (rank + 1) * 128)
        )
        assert rebuilt.seq_lengths == (256, 512)
        assert torch.equal(
            rebuilt.mask_mod(torch.tensor(0), torch.tensor(0), queries, keys),
            expected_mask,
        )
        lengths = (190, 145, 177)
        positions = torch.cat([torch.arange(length) for length in lengths])
        document_ids = torch.repeat_interleave(torch.arange(3), torch.tensor(lengths))
        packed = create_block_mask(
            and_masks(causal, get_document_mask_mod(positions)),
            B=1,
            H=1,
            Q_LEN=512,
            KV_LEN=512,
            device="cpu",
            _compile=False,
        )
        packed_local = shard_attention_mask_for_cp(packed, kv)
        global_queries = queries + rank // 2 * 256
        expected_packed = (global_queries >= keys) & (
            document_ids[global_queries] == document_ids[keys]
        )
        actual_packed = packed_local.mask_mod(
            torch.tensor(0), torch.tensor(0), queries, keys
        )
        assert torch.equal(actual_packed, expected_packed)
        assert not torch.equal(actual_packed, expected_mask)
        if rank == 0:
            print(
                "PASS dual CP: rank order, orthogonal groups, "
                "SUM, causal and packed masks",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
