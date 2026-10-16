"""Two CPU ranks: real FSDP hooks around emulated MC2 expert computation.

Run with PYTHONPATH=. torchrun --nproc_per_node=2. This does not run NPU
collectives: fused operators are CPU emulation with local world_size=1.
"""

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from llmtuner.models.common.moe.experts import GroupedExperts
from llmtuner.parallel.expert_parallel.mc2_dispatch import MC2GroupedExperts
from llmtuner.parallel.fully_shard_parallel.fine_grained import fully_shard_fine_grained
from tests.unit_tests.cpu.models.test_mc2_ops import EmulatedOps


class ExpertOwner(torch.nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.experts = experts

    def forward(self, inputs):
        return self.experts(inputs, EmulatedOps(), "test", 1, [2, 3], [2, 3])


def main():
    dist.init_process_group("gloo")
    try:
        assert dist.get_world_size() == 2
        mesh = init_device_mesh("cpu", (2,))
        for custom in (False, True):
            for relocated in (False, True) if custom else (False,):
                torch.manual_seed(12)
                experts = MC2GroupedExperts(4, 3, 2, use_grouped_mm=False).double()
                for parameter in experts.parameters():
                    torch.nn.init.normal_(parameter, std=0.1)
                reference = GroupedExperts(4, 3, 2, use_grouped_mm=False).double()
                reference.load_state_dict(experts.state_dict())
                root = ExpertOwner(experts)
                shard = fully_shard_fine_grained if custom else fully_shard
                options = {"hook_module": root} if relocated else {}
                shard(experts, mesh=mesh, **options)
                shard(root, mesh=mesh)
                inputs = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
                ref_inputs = inputs.detach().clone().requires_grad_()
                counts = [2, 3]
                actual = root(inputs)
                expected = reference(ref_inputs, torch.tensor(counts))
                actual.square().sum().backward()
                expected.square().sum().backward()
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
                torch.testing.assert_close(
                    inputs.grad, ref_inputs.grad, atol=1e-12, rtol=1e-12
                )
                for got, want in zip(
                    experts.parameters(), reference.parameters(), strict=True
                ):
                    torch.testing.assert_close(
                        got.grad.full_tensor(), want.grad, atol=1e-12, rtol=1e-12
                    )
                if dist.get_rank() == 0:
                    print(
                        f"custom={custom} relocated={relocated}: "
                        "FSDP MC2 lifecycle passed"
                    )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
