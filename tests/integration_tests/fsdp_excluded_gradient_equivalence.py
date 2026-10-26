"""Verify Trainer exclusion reduction against independent dense AdamW updates.

PYTHONPATH=. torchrun --nproc_per_node=2 \
    tests/integration_tests/fsdp_excluded_gradient_equivalence.py [native|custom]

CPU/Gloo FP64, DP2. Includes rank-local and globally unused excluded gradients.
Calls the production parallelizer and Trainer reducer; this is not full Trainer
or EP/TP/CP coverage. Use four ranks and append hsdp for replicate2/shard2.
"""

import copy
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from llmtuner.accelerator.collectives import clip_grad_norm_
from llmtuner.config.parallel_config import FSDPConfig, ParallelConfig
from llmtuner.parallel.parallelize import parallelize_hf_transformers
from llmtuner.parallel.topology import ParallelTopology
from llmtuner.trainer.trainer import Trainer


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        self.optional = nn.Parameter(torch.ones(4))
        self.unused = nn.Parameter(torch.ones(4))

    def forward(self, inputs, use_optional):
        output = self.linear(inputs)
        return output + self.optional if use_optional else output


def main():
    dist.init_process_group("gloo")
    try:
        world = dist.get_world_size()
        hsdp = "hsdp" in sys.argv[2:]
        assert world == (4 if hsdp else 2)
        implementation = sys.argv[1] if len(sys.argv) > 1 else "native"
        torch.manual_seed(315)
        model = Model().double()
        reference = copy.deepcopy(model)
        cfg = ParallelConfig(
            data_parallel_shard_size=2,
            data_parallel_replicate_size=2 if hsdp else 1,
            fsdp_config=FSDPConfig(
                fsdp_implementation=implementation,
                fsdp_reduce_dtype="float64",
                fsdp_apply_modules={
                    "": {"ignored_params": {model.optional, model.unused}}
                },
            ),
        )
        dims = ParallelTopology.from_config(cfg, world_size=world)
        dims.build_device_mesh()
        model = parallelize_hf_transformers(
            model, cfg=cfg, mesh=dims.spmd_dense_mesh(), parallel_dims=dims
        )
        assert not isinstance(model.optional, DTensor)
        assert isinstance(model.linear.weight, DTensor)
        if hsdp:
            assert model.linear.weight.device_mesh.shape == (2, 2)
            assert dims.get_mesh("dp").size() == 4
            assert dims.get_mesh("dp_replicate").size() == 2
        runtime = SimpleNamespace(parallel_dims=dims, model_parts=[model])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
        dense_optimizer = torch.optim.AdamW(
            reference.parameters(), lr=0.01, weight_decay=0.1
        )
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)
            losses = []
            for rank in range(world):
                inputs = torch.arange(8, dtype=torch.float64).reshape(2, 4) / 8
                inputs = inputs + rank / 3 + step / 5
                use_optional = step == 2 or (step == 0 and rank == 0)
                expected = reference(inputs, use_optional)
                losses.append(expected.square().sum() / 16)
                if rank == dist.get_rank():
                    actual = model(inputs, use_optional)
                    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                    (actual.square().sum() / 16).backward()
            sum(losses).backward()
            Trainer._allreduce_fsdp_excluded_grads(runtime)
            for name, parameter in model.named_parameters():
                expected = dict(reference.named_parameters())[name]
                if expected.grad is None:
                    assert parameter.grad is None, name
                else:
                    gradient = parameter.grad
                    if isinstance(gradient, DTensor):
                        gradient = gradient.full_tensor()
                    torch.testing.assert_close(
                        gradient, expected.grad, rtol=1e-12, atol=1e-12
                    )
            expected_norm = torch.linalg.vector_norm(
                torch.cat(
                    [
                        p.grad.flatten()
                        for p in reference.parameters()
                        if p.grad is not None
                    ]
                )
            )
            norm = clip_grad_norm_(model.parameters(), 0.5)
            torch.testing.assert_close(norm, expected_norm, rtol=1e-12, atol=1e-12)
            factor = (0.5 / (expected_norm + 1e-6)).clamp(max=1)
            for parameter in reference.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(factor)
            optimizer.step()
            dense_optimizer.step()
            for actual, expected in zip(
                model.parameters(), reference.parameters(), strict=True
            ):
                if isinstance(actual, DTensor):
                    actual = actual.full_tensor()
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            print(f"rank={dist.get_rank()} {implementation} step={step}: passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
