"""Pattern FSDP with one child aliased under distinct parent units.

Run with torchrun --standalone --nproc_per_node=2. The shared harness selects
NPU/HCCL by default; LLMTUNER_TEST_DEVICE=cpu selects CPU/Gloo.
Four ranks additionally exercise HSDP with two replicas and two shards.
"""

import copy
import sys

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule

from llmtuner.config.parallel_config import FSDPConfig, ParallelConfig
from llmtuner.parallel.fully_shard_parallel.fully_shard_parallel import (
    apply_fsdp_to_decoder,
    force_sum_grad_reduction,
)
from llmtuner.parallel.fully_shard_parallel.pattern import apply_fsdp_by_pattern
from tests.integration_tests.harness import init_distributed


class AliasedChildModel(nn.Module):
    def __init__(self):
        super().__init__()
        shared = nn.Linear(4, 4)
        self.left = nn.Sequential(shared, nn.Tanh())
        self.right = nn.Sequential(shared, nn.Sigmoid())
        self.head = nn.Linear(4, 2)

    def forward(self, inputs):
        return self.head(self.right(self.left(inputs)))


class RootAliasedDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Identity()
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(4, 4), nn.Tanh())])
        self.external = self.layers[0][0]
        self.norm = nn.Identity()
        self.lm_head = nn.Linear(4, 2)
        self.enable_weight_tying = False

    def forward(self, inputs):
        return self.lm_head(self.external(self.layers[0](inputs)))


def main():
    device = init_distributed()
    try:
        world_size = dist.get_world_size()
        assert world_size in (2, 4)
        rank = dist.get_rank()
        torch.manual_seed(41)
        decoder = "--decoder" in sys.argv
        model = (RootAliasedDecoder() if decoder else AliasedChildModel()).to(device)
        reference = copy.deepcopy(model)
        mesh = init_device_mesh(
            device.type,
            (2, 2) if world_size == 4 else (2,),
            mesh_dim_names=("dp_replicate", "dp_shard")
            if world_size == 4
            else ("dp_shard",),
        )
        cfg = ParallelConfig(
            fsdp_config=FSDPConfig(fsdp_apply_modules={"left": {}, "right": {}})
        )
        if decoder:
            apply_fsdp_to_decoder(
                model,
                mesh,
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                pp_enabled=False,
            )
            assert model.external is model.layers[0][0]
            # Native fully_shard collapses ancestor/descendant groups to the
            # root; the layer must not acquire an independent hook lifecycle.
            assert isinstance(model, FSDPModule)
            assert not isinstance(model.layers[0], FSDPModule)
        else:
            apply_fsdp_by_pattern(
                model,
                dp_mesh=mesh,
                cfg=cfg,
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
            )
            assert model.left[0] is model.right[0]
            assert model.left._get_fsdp_state() is model.right._get_fsdp_state()
        force_sum_grad_reduction(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        reference_optimizer = torch.optim.SGD(
            reference.parameters(), lr=0.05, momentum=0.9
        )
        initial = next(reference.parameters()).detach().clone()
        for step in range(3):
            torch.manual_seed(100 + step)
            inputs = torch.randn(8, 4, device=device)
            targets = torch.randn(8, 2, device=device)
            optimizer.zero_grad()
            reference_optimizer.zero_grad()
            expected = reference(inputs)
            (expected - targets).square().sum().backward()
            actual = model(inputs.chunk(world_size)[rank])
            torch.testing.assert_close(
                actual, expected.detach().chunk(world_size)[rank]
            )
            (actual - targets.chunk(world_size)[rank]).square().sum().backward()
            reference_parameters = dict(reference.named_parameters())
            for name, parameter in model.named_parameters():
                target = reference_parameters[name]
                torch.testing.assert_close(
                    parameter.grad.full_tensor(), target.grad, rtol=1e-5, atol=1e-6
                )
                assert target.grad.count_nonzero() > 0, name
            optimizer.step()
            reference_optimizer.step()
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(
                    parameter.full_tensor(),
                    reference_parameters[name],
                    rtol=1e-5,
                    atol=1e-6,
                )
            print(
                f"PASS rank={rank} step={step + 1}: "
                f"aliased-child FSDP decoder={decoder}",
                flush=True,
            )
        assert not torch.equal(initial, next(reference.parameters()))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
