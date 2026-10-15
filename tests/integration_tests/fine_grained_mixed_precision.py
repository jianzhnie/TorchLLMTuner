"""Compare custom relocated FSDP mixed precision with native FSDP.

Run: PYTHONPATH=. torchrun --nproc_per_node=2 \
    tests/integration_tests/fine_grained_mixed_precision.py

CPU/Gloo checks dtype, outputs, full gradients and three SGD updates.
Append --checkpoint and/or --compile-backend aot_eager to transform the
custom model before FSDP; native FSDP remains an eager numerical reference.
This does not prove CUDA/NPU communication overlap or performance.
"""

import argparse
import copy
from itertools import product

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from llmtuner.parallel.fully_shard_parallel.fine_grained import fully_shard_fine_grained
from llmtuner.parallel.fully_shard_parallel.fully_shard_parallel import (
    force_sum_grad_reduction,
)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(8, 8)
        self.last = nn.Linear(8, 8)

    def forward(self, value):
        return self.last(self.first(value).tanh())


class PromotedTanhBlock(Block):
    """Independent reference retaining tanh's FP32 intermediate for backward."""

    def forward(self, value):
        hidden = self.first(value)
        return self.last(hidden.float().tanh().to(hidden.dtype))


def local_checkpoint_probe():
    """Isolate the mixed-precision compile discrepancy without FSDP or a group."""
    for checkpointed in (False, True):
        torch._dynamo.reset()
        torch.manual_seed(42)
        reference = Block().bfloat16()
        promoted = PromotedTanhBlock().bfloat16()
        promoted.load_state_dict(reference.state_dict())
        candidate = copy.deepcopy(reference)
        if checkpointed:
            from llmtuner.memory.activation_checkpoint import wrap_full

            candidate = wrap_full(candidate)
        candidate = torch.compile(candidate, backend="inductor")
        inputs = torch.randn(4, 8, generator=torch.Generator().manual_seed(100))
        actual = candidate(inputs.bfloat16())
        expected = reference(inputs.bfloat16())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.float().square().sum().backward()
        expected.float().square().sum().backward()
        promoted(inputs.bfloat16()).float().square().sum().backward()
        import torch._inductor.config as inductor_config

        oracle = (
            promoted
            if checkpointed and not inductor_config.emulate_precision_casts
            else reference
        )
        for actual_param, oracle_param in zip(
            candidate.parameters(), oracle.parameters(), strict=True
        ):
            torch.testing.assert_close(
                actual_param.grad, oracle_param.grad, rtol=0, atol=0
            )
        for (name, actual_param), expected_param in zip(
            candidate.named_parameters(), reference.parameters(), strict=True
        ):
            print(
                "LOCAL",
                checkpointed,
                name,
                "gradient_max_abs",
                (actual_param.grad.float() - expected_param.grad.float())
                .abs()
                .max()
                .item(),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--compile-backend", choices=("aot_eager", "inductor"))
    parser.add_argument("--local-probe", action="store_true")
    args = parser.parse_args()
    if args.local_probe:
        local_checkpoint_probe()
        return
    dist.init_process_group("gloo")
    try:
        assert dist.get_world_size() == 2
        mesh = init_device_mesh("cpu", (2,))
        for param_dtype, reduce_dtype, output_dtype, reshard, input_grad in product(
            (torch.bfloat16, torch.float16),
            (torch.float32, torch.bfloat16),
            (torch.float32, torch.bfloat16),
            (False, True),
            (False, True),
        ):
            print(
                "CASE",
                dist.get_rank(),
                param_dtype,
                reduce_dtype,
                output_dtype,
                reshard,
                input_grad,
                flush=True,
            )
            torch.manual_seed(42)
            custom = Block()
            native = copy.deepcopy(custom)
            if args.checkpoint:
                from llmtuner.memory.activation_checkpoint import wrap_full

                custom = wrap_full(custom)
            if args.compile_backend:
                from torch._dynamo.utils import counters

                torch._dynamo.reset()
                counters.clear()
                custom = torch.compile(custom, backend=args.compile_backend)
            policy = MixedPrecisionPolicy(
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                output_dtype=output_dtype,
            )
            for model, shard in (
                (custom, fully_shard_fine_grained),
                (native, fully_shard),
            ):
                # Native owns the same Block forward boundary as the relocated
                # custom units. Native per-Linear hooks would cast before tanh
                # at a different boundary and define a different computation.
                if model is custom:
                    for layer in (model.first, model.last):
                        shard(
                            layer,
                            mesh=mesh,
                            mp_policy=policy,
                            reshard_after_forward=reshard,
                            hook_module=model,
                        )
                shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=reshard)
                force_sum_grad_reduction(model)
            optimizers = [
                torch.optim.SGD(m.parameters(), lr=0.01) for m in (custom, native)
            ]
            generator = torch.Generator().manual_seed(100 + dist.get_rank())
            for _ in range(3):
                inputs = torch.randn(4, 8, generator=generator)
                values = [inputs.clone().requires_grad_(input_grad) for _ in range(2)]
                outputs = [
                    model(value)
                    for model, value in zip((custom, native), values, strict=True)
                ]
                assert all(output.dtype == output_dtype for output in outputs)
                torch.testing.assert_close(*outputs)
                for output in outputs:
                    output.float().square().sum().backward()
                if input_grad:
                    torch.testing.assert_close(values[0].grad, values[1].grad)
                for a, b in zip(custom.parameters(), native.parameters(), strict=True):
                    assert a.dtype == b.dtype == torch.float32
                    assert a.grad.dtype == b.grad.dtype == torch.float32
                    torch.testing.assert_close(
                        a.grad.full_tensor(), b.grad.full_tensor()
                    )
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for a, b in zip(custom.parameters(), native.parameters(), strict=True):
                    torch.testing.assert_close(a.full_tensor(), b.full_tensor())
            if args.compile_backend:
                assert counters["stats"]["unique_graphs"] > 0
            print(
                "PASS mixed precision",
                dist.get_rank(),
                param_dtype,
                reduce_dtype,
                output_dtype,
                reshard,
                input_grad,
                args.checkpoint,
                args.compile_backend,
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
