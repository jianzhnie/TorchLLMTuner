"""Vocab-parallel embedding forward and gradient equivalence under TP.

Run with:
    LLMTUNER_TEST_DEVICE=cpu PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/vocab_parallel_embedding_equivalence.py

The same script selects the accelerator and its native backend when available.
"""

import torch
import torch.distributed as dist
import torch.nn as nn

from llmtuner.config import ParallelConfig
from llmtuner.parallel.tensor_parallel import apply_tp
from llmtuner.parallel.topology import ParallelTopology, build_device_mesh
from tests.integration_tests.harness import init_distributed


class TokenModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


def main() -> None:
    device = init_distributed()
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    torch.manual_seed(7)
    reference = TokenModel(16, 8).to(device)
    torch.manual_seed(7)
    model = TokenModel(16, 8).to(device)
    topology = ParallelTopology(
        dp_replicate=1, dp_shard=1, cp=1, tp=world, pp=1, ep=1, world_size=world
    )
    model = apply_tp(
        model,
        build_device_mesh(topology),
        ParallelConfig(tensor_parallel_size=world),
        plan={"embedding": "embedding_rowwise"},
    )

    generator = torch.Generator(device="cpu").manual_seed(11)
    ids = torch.randint(1, 16, (8,), generator=generator).to(device)
    ids[::5] = 0
    local_ids = ids.chunk(world)[rank].contiguous()
    local_width = reference.embedding.num_embeddings // world
    start = rank * local_width
    torch.testing.assert_close(
        model.embedding.weight,
        reference.embedding.weight[start : start + local_width],
    )
    expected = reference(ids)
    actual = model(local_ids)
    torch.testing.assert_close(actual, expected.chunk(world)[rank])

    weights = torch.linspace(-0.5, 1.5, expected.numel(), device=device).reshape(
        expected.shape
    )
    local_weights = weights.chunk(world)[rank].contiguous()
    (actual * local_weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(
        model.embedding.weight.grad,
        reference.embedding.weight.grad[start : start + local_width],
    )
    gathered = [torch.empty_like(model.embedding.weight) for _ in range(world)]
    dist.all_gather(gathered, model.embedding.weight.detach())
    torch.testing.assert_close(torch.cat(gathered), reference.embedding.weight)

    if rank == 0:
        print(f"vocab-parallel embedding: all checks passed on {device.type}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
