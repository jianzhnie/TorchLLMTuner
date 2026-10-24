"""Feature-TP MoE fresh-runtime DCP resume with AdamW, scheduler, EMA and RNG.

Run using PYTHONPATH=. torchrun --nproc_per_node=2 on CPU/Gloo.
Exercises lower-level TP conversion, not full Trainer or an independent OS job.
"""

import argparse
import pickle
import random
from contextlib import nullcontext
from copy import deepcopy
from itertools import product
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from llmtuner.components.checkpointer.checkpoint_keys import RNG_STATE
from llmtuner.components.optimizer.ema import EMA
from llmtuner.components.optimizer.lr_scheduler import build_lr_scheduler
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config.compile_config import CompileConfig
from llmtuner.config.lr_scheduler_config import LRSchedulerConfig
from llmtuner.config.optimizer_config import OptimizerConfig
from llmtuner.config.parallel_config import FSDPConfig, ParallelConfig
from llmtuner.memory.activation_checkpoint import wrap_full
from llmtuner.models.common.aux_loss import AuxLoss
from llmtuner.parallel.expert_parallel.model_replacement import swap_hf_moe_blocks
from llmtuner.parallel.fully_shard_parallel.fine_grained import fully_shard_fine_grained
from llmtuner.parallel.fully_shard_parallel.plan import apply_fsdp_by_pattern
from llmtuner.parallel.model_compile import apply_compile
from llmtuner.parallel.tensor_parallel.linear_ops import functional_collectives
from llmtuner.parallel.tensor_parallel.tensor_parallel import realize_moe_blocks_tp
from tests.integration_tests.ep_wiring_equivalence import _model
from tests.integration_tests.fixed_router_tp_equivalence import (
    MoEDecoder,
    MoEModel,
    SharedExpert,
)
from tests.integration_tests.rng_checkpoint_equivalence import manager


def runtime(seed, recompute, compile_enabled, tp_group, dp_mesh=None, custom=False):
    if compile_enabled:
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
    target = _model(seed).train()
    block = target.layers[0].mlp
    block.shared_expert = SharedExpert()
    path = next(name for name, module in target.named_modules() if module is block)
    realize_moe_blocks_tp(
        [(path, block)], tp_size=2, tp_rank=dist.get_rank(tp_group), group=tp_group
    )
    target._tp_checkpoint_layouts = {
        f"{path}.experts.gate_up_proj": (1, True, tp_group),
        f"{path}.experts.down_proj": (2, False, tp_group),
        **{
            f"{path}.shared_expert.{name}.weight": (dimension, False, tp_group)
            for name, dimension in (("gate_proj", 0), ("up_proj", 0), ("down_proj", 1))
        },
    }
    swap_hf_moe_blocks(
        target, fixed_router=True, tp_enabled=True, router_aux_loss_coef=0.03
    )
    decoder = MoEDecoder(target.layers[0].mlp)
    if recompute:
        decoder = wrap_full(decoder)
    execution = MoEModel(decoder)
    if compile_enabled:
        execution = apply_compile(
            execution, compile_config=CompileConfig(backend="aot_eager")
        )
    if dp_mesh is not None:
        shard = fully_shard_fine_grained if custom else fully_shard
        if custom:
            cfg = ParallelConfig(
                data_parallel_shard_size=2,
                fsdp_config=FSDPConfig(
                    fsdp_implementation="custom",
                    fsdp_apply_modules={
                        "layers.0.mlp.routed_experts.inner_experts": {
                            "hook_module": decoder
                        },
                        "layers.0": {},
                    },
                    fsdp_forward_prefetch_count=2,
                    fsdp_backward_prefetch_count=2,
                ),
            )
            apply_fsdp_by_pattern(
                execution,
                dp_mesh=dp_mesh,
                cfg=cfg,
                param_dtype=torch.float64,
                reduce_dtype=torch.float64,
            )
        else:
            shard(decoder, mesh=dp_mesh)
        assert isinstance(target.layers[0].mlp.router.gate.weight, DTensor)
    target._checkpoint_test_custom_fsdp = custom
    optimizer = OptimizersContainer(
        OptimizerConfig(learning_rate=0.01, implementation="for-loop"),
        model_parts=[target],
    )
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(), optimizers=optimizer.optimizers, training_steps=4
    )
    return (
        target,
        execution,
        optimizer,
        scheduler,
        EMA(model_parts=[target], decay=0.75),
    )


def update(state, tp_group):
    model, execution, optimizer, scheduler, ema = state
    block = model.layers[0].mlp
    optimizer.zero_grad(set_to_none=True)
    x = torch.randn(2, 4, 64, dtype=torch.float64)
    x *= 0.5 + random.random() + np.random.random()
    mask = torch.arange(8).reshape(2, 4) % 3 == dist.get_rank()
    denominator = (~mask).sum()
    dist.all_reduce(denominator)
    AuxLoss.set_step_denominator(denominator, model_parts=[model])
    block.set_padding_mask(mask.reshape(-1))
    loss = execution(x).square().sum()
    loss.backward()
    decoder = execution.layers[0]
    if model._checkpoint_test_custom_fsdp:
        assert len(decoder._get_fsdp_state().communication_contexts) == 2
    gradient = block.router.gate.weight.grad
    if isinstance(gradient, DTensor):
        gradient = gradient.to_local()
    dist.all_reduce(gradient, group=tp_group)
    optimizer.step()
    scheduler.step()
    expected_averages = []
    for ema_optimizer in ema.optimizers:
        for group in ema_optimizer.param_groups:
            for tensor in group["params"]:
                average = ema_optimizer.state[tensor]["ema_params"]
                local_average = (
                    average.to_local() if isinstance(average, DTensor) else average
                )
                local_tensor = (
                    tensor.to_local() if isinstance(tensor, DTensor) else tensor
                )
                # Independent scalar-tensor lerp, outside the foreach grouping.
                expected_averages.append(
                    (
                        average,
                        torch.lerp(
                            local_average.detach().clone(), local_tensor.detach(), 0.25
                        ),
                    )
                )
    ema.step(scheduler.state_dict()["last_epoch"])
    for average, expected in expected_averages:
        actual = average.to_local() if isinstance(average, DTensor) else average
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    return loss.detach().clone()


def snapshot(state):
    return deepcopy(
        (
            state[0].state_dict(),
            state[2].state_dict(),
            state[3].state_dict(),
            state[4].state_dict(),
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsdp", action="store_true", help="Use four ranks: DP2 x TP2")
    parser.add_argument(
        "--custom-fsdp", action="store_true", help="Use custom FSDP; requires --fsdp"
    )
    args = parser.parse_args()
    if args.custom_fsdp and not args.fsdp:
        parser.error("--custom-fsdp requires --fsdp")
    dist.init_process_group("gloo")
    try:
        assert dist.get_world_size() == (4 if args.fsdp else 2)
        dp_mesh = None
        tp_group = dist.group.WORLD
        if args.fsdp:
            mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
            dp_mesh = mesh["dp"]
            tp_group = mesh["tp"].get_group()
        functional_collectives()
        rank = dist.get_rank()
        context = TemporaryDirectory() if rank == 0 else nullcontext(None)
        with context as root:
            paths = [root]
            dist.broadcast_object_list(paths, src=0)
            for recompute, compiled, mode in product(
                (False, True), (False, True), ("disabled", "async")
            ):
                original = runtime(
                    42, recompute, compiled, tp_group, dp_mesh, args.custom_fsdp
                )
                random.seed(100 + rank)
                np.random.seed(200 + rank)
                torch.manual_seed(300 + rank)
                for _ in range(2):
                    update(original, tp_group)
                boundary = snapshot(original)
                folder = f"{paths[0]}/{recompute}_{compiled}_{mode}"
                writer = manager(original, folder, async_mode=mode)
                try:
                    rng = writer.states[RNG_STATE]
                    saved_rng = pickle.loads(rng.state_dict()[rng.key])
                    if compiled:
                        assert (
                            torch._dynamo.utils.counters["stats"]["unique_graphs"] > 0
                        )
                    assert writer.save(2)
                    if mode == "async":
                        assert writer.save_future is not None
                    writer.maybe_wait_for_staging()
                    losses = [update(original, tp_group) for _ in range(2)]
                finally:
                    writer.close()
                resumed = runtime(
                    123, recompute, compiled, tp_group, dp_mesh, args.custom_fsdp
                )
                reader = manager(resumed, folder, async_mode=mode)
                try:
                    assert reader.load(2)
                    rng = reader.states[RNG_STATE]
                    loaded_rng = pickle.loads(rng.state_dict()[rng.key])
                    for key in saved_rng:
                        if key == "numpy":
                            np.testing.assert_equal(loaded_rng[key], saved_rng[key])
                        elif key == "cpu":
                            torch.testing.assert_close(
                                loaded_rng[key], saved_rng[key], atol=0, rtol=0
                            )
                        else:
                            assert loaded_rng[key] == saved_rng[key]

                    torch.testing.assert_close(
                        snapshot(resumed), boundary, atol=0, rtol=0
                    )
                finally:
                    reader.close()
                actual = [update(resumed, tp_group) for _ in range(2)]
                if compiled:
                    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] > 0
                torch.testing.assert_close(actual, losses, atol=0, rtol=0)
                torch.testing.assert_close(
                    snapshot(resumed), snapshot(original), atol=0, rtol=0
                )
                dist.barrier()
                if rank == 0:
                    print(
                        f"feature TP resume {recompute=} {compiled=} {mode=}: exact",
                        flush=True,
                    )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
