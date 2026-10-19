"""Two-rank CPU/Gloo DCP resume for relocated custom FSDP mixed precision.

Run with PYTHONPATH=. torchrun --nproc_per_node=2 and this script.
BF16/FP16 compute, FP32 reduction/master parameters, reshard on/off, and full
activation checkpoint on/off, and source/target AMSGrad on/off (32 cases).
New runtime restores model, AdamW, scheduler and
EMA exactly, then takes the same next update. This does not test device kernels,
compile, asynchronous transport, topology changes or data/RNG checkpoint state.
"""

from itertools import product
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy

from llmtuner.components.checkpointer.distributed_checkpoint import (
    DistributedCheckpointManager,
)
from llmtuner.components.optimizer.ema import EMA
from llmtuner.components.optimizer.lr_scheduler import build_lr_scheduler
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config import CheckpointConfig, LRSchedulerConfig, OptimizerConfig
from llmtuner.config.optimizer_config import ParamGroupConfig
from llmtuner.memory.activation_checkpoint import wrap_full
from llmtuner.parallel.fully_shard_parallel.fine_grained import fully_shard_fine_grained
from llmtuner.parallel.fully_shard_parallel.fully_shard_parallel import (
    force_sum_grad_reduction,
)
from tests.integration_tests.fine_grained_mixed_precision import Block


def runtime(mesh, dtype, reshard, recompute, seed, amsgrad):
    torch.manual_seed(seed)
    model = Block()
    if recompute:
        model = wrap_full(model)
    policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32)
    for layer in (model.first, model.last):
        fully_shard_fine_grained(
            layer,
            mesh=mesh,
            mp_policy=policy,
            reshard_after_forward=reshard,
            hook_module=model,
        )
    fully_shard_fine_grained(
        model, mesh=mesh, mp_policy=policy, reshard_after_forward=reshard
    )
    force_sum_grad_reduction(model)
    optimizer = OptimizersContainer(
        OptimizerConfig(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"amsgrad": amsgrad, "lr": 3e-4},
                )
            ],
        ),
        model_parts=[model],
    )
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(decay_ratio=1.0),
        optimizers=optimizer.optimizers,
        training_steps=4,
    )
    ema = EMA(model_parts=[model], decay=0.5)
    return model, optimizer, scheduler, ema


def update(runtime, value, step):
    model, optimizer, scheduler, ema = runtime
    optimizer.zero_grad(set_to_none=True)
    output = model(value)
    output.float().square().sum().backward()
    gradients = [p.grad.full_tensor().clone() for p in model.parameters()]
    optimizer.step()
    scheduler.step()
    ema.step(step)
    return output.detach(), gradients


def manager(runtime, folder):
    model, optimizer, scheduler, ema = runtime
    return DistributedCheckpointManager(
        CheckpointConfig(enable=True, interval=1, keep_latest_k=0),
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=scheduler,
        ema=ema,
        states={},
        folder=folder,
    )


def compare(left, right):
    for a, b in zip(left[0].parameters(), right[0].parameters(), strict=True):
        torch.testing.assert_close(a.full_tensor(), b.full_tensor(), rtol=0, atol=0)
    for a, b in zip(left[1:], right[1:], strict=True):
        torch.testing.assert_close(a.state_dict(), b.state_dict(), rtol=0, atol=0)


def main():
    dist.init_process_group("gloo")
    try:
        assert dist.get_world_size() == 2
        mesh = init_device_mesh("cpu", (2,))
        for dtype, reshard, recompute, amsgrad, target_amsgrad in product(
            (torch.bfloat16, torch.float16),
            (False, True),
            (False, True),
            (False, True),
            (False, True),
        ):
            with TemporaryDirectory() as local:
                paths = [local if dist.get_rank() == 0 else None]
                dist.broadcast_object_list(paths, src=0)
                original = runtime(mesh, dtype, reshard, recompute, 42, amsgrad)
                generator = torch.Generator().manual_seed(100 + dist.get_rank())
                for step in (1, 2):
                    update(original, torch.randn(4, 8, generator=generator), step)
                writer = manager(original, paths[0])
                try:
                    assert writer.save(2)
                finally:
                    writer.close()
                restored = runtime(mesh, dtype, reshard, recompute, 987, target_amsgrad)
                reader = manager(restored, paths[0])
                try:
                    assert reader.load(2)
                finally:
                    reader.close()
                compare(original, restored)
                value = torch.randn(4, 8, generator=generator)
                expected = update(original, value, 3)
                actual = update(restored, value, 3)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                compare(original, restored)
                dist.barrier()
                print(
                    f"PASS rank={dist.get_rank()} {dtype=} {reshard=} {recompute=} "
                    f"{amsgrad=} {target_amsgrad=}"
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
