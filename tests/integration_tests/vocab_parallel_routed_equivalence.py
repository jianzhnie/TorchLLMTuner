"""Independent VP and VP+VFSDP numerical/checkpoint equivalence.

Run with two ranks for pure VP or four ranks for 2x2 VP+VFSDP:
    LLMTUNER_TEST_DEVICE=cpu PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
        tests/integration_tests/vocab_parallel_routed_equivalence.py
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh

from llmtuner.parallel.vocab_parallel import (
    RoutedVocabParallelEmbedding,
    VocabParallelPlan,
    gather_vocab_parallel_weight,
    load_vocab_parallel_weight,
    vocab_fully_shard_modules,
    vocab_parallelize_modules,
)
from tests.integration_tests.harness import init_distributed


class TokenModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8, padding_idx=0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


def _local_grad(parameter: torch.Tensor) -> torch.Tensor:
    while hasattr(parameter, "to_local"):
        parameter = parameter.to_local()
    return parameter


def main() -> None:
    device = init_distributed()
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world in (2, 4)
    mesh = init_device_mesh(
        device.type, (2, world // 2), mesh_dim_names=("vp", "vfsdp")
    )
    vp_mesh, vfsdp_mesh = mesh["vp"], mesh["vfsdp"]
    torch.manual_seed(7)
    reference = TokenModel().to(device)
    torch.manual_seed(7)
    model = TokenModel() if world == 2 else TokenModel().to(device)
    divide_factor = float(world) if device.type != "cpu" or world == 2 else 1.0
    plan = VocabParallelPlan(
        apply_modules=["embedding"],
        gradient_divide_factor=divide_factor,
        output_dtype=torch.bfloat16 if world == 2 else None,
    )
    model = vocab_parallelize_modules(model, vp_mesh, plan)
    assert isinstance(model.embedding, RoutedVocabParallelEmbedding)
    model = vocab_fully_shard_modules(model, vfsdp_mesh, plan)

    generator = torch.Generator(device="cpu").manual_seed(11 + rank)
    ids = torch.randint(1, 16, (7,), generator=generator).to(device)
    ids[::4] = 0
    expected = reference(ids)
    if plan.output_dtype is not None:
        expected = expected.to(plan.output_dtype)
    actual = model(ids)
    torch.testing.assert_close(actual, expected)

    reference_group = dist.group.WORLD if vfsdp_mesh.size() > 1 else vp_mesh.get_group()
    reference_size = dist.get_world_size(reference_group)
    peer_ids = [torch.empty_like(ids) for _ in range(reference_size)]
    dist.all_gather(peer_ids, ids, group=reference_group)
    combined_ids = torch.cat(peer_ids)
    reference_output = reference(combined_ids)
    if plan.output_dtype is not None:
        reference_output = reference_output.to(plan.output_dtype)
    weights = torch.linspace(-0.25, 0.75, actual.numel(), device=device).reshape(
        actual.shape
    )
    combined_weights = torch.cat([weights] * reference_size)
    (actual * weights).sum().backward()
    (reference_output * combined_weights).sum().backward()

    vf_rank = vfsdp_mesh.get_local_rank()
    vf_size = vfsdp_mesh.size()
    vp_rank = vp_mesh.get_local_rank()
    expected_shard = reference.embedding.weight.grad.chunk(2, dim=0)[vp_rank]
    expected_divisor = divide_factor if divide_factor != 1.0 else vf_size
    expected_local = expected_shard.chunk(vf_size, dim=0)[vf_rank] / expected_divisor
    torch.testing.assert_close(_local_grad(model.embedding.weight.grad), expected_local)

    restored = gather_vocab_parallel_weight(model.embedding)
    torch.testing.assert_close(restored, reference.embedding.weight)
    with torch.no_grad():
        reference.embedding.weight.add_(0.125)
    load_vocab_parallel_weight(model.embedding, reference.embedding.weight)
    torch.testing.assert_close(
        gather_vocab_parallel_weight(model.embedding), reference.embedding.weight
    )

    if rank == 0:
        print(
            f"routed vocab parallel: VP=2 VFSDP={vf_size} passed on {device.type}; "
            f"local weight={tuple(_local_grad(model.embedding.weight).shape)}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
