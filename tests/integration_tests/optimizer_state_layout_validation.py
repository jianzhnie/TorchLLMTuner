"""Reject rank-local optimizer state layout errors before any state is loaded."""

import argparse
from contextlib import nullcontext
from copy import deepcopy

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import (
    DTensor,
    Partial,
    Replicate,
    Shard,
    distribute_tensor,
)
from torch.distributed.tensor.debug import CommDebugMode

from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config import OptimizerConfig
from tests.integration_tests.harness import init_distributed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-counter-ranks", action="store_true")
    parser.add_argument("--empty-shard", action="store_true")
    args = parser.parse_args()
    device = init_distributed()
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        mesh = init_device_mesh(device.type, (2,), mesh_dim_names=("dp_shard",))
        rows = 1 if args.empty_shard else 4
        shape = torch.Size((rows, 4))
        model = nn.Module()
        model.weight = nn.Parameter(
            distribute_tensor(
                torch.arange(rows * 4, device=device).float().reshape(shape),
                mesh,
                [Shard(0)],
            )
        )
        optimizer = OptimizersContainer(
            OptimizerConfig(implementation="for-loop"), model_parts=[model]
        )
        checkpoint = deepcopy(optimizer.state_dict())
        key = "state.weight.exp_avg"
        # Redistribution is collective, so prepare the invalid candidate on both
        # ranks before entering the rank-local validation phase.
        replicated = checkpoint[key].redistribute(placements=[Replicate()])
        reversed_mesh = DeviceMesh(device.type, [1, 0], mesh_dim_names=("dp_shard",))
        wrong_mesh = DTensor.from_local(
            checkpoint[key].to_local(),
            reversed_mesh,
            [Shard(0)],
            shape=shape,
            stride=(4, 1),
        )
        wrong_local_shape = DTensor.from_local(
            checkpoint[key].to_local()[:0],
            mesh,
            [Shard(0)],
            shape=shape,
            stride=(4, 1),
        )
        candidates = {
            "plain": torch.ones(shape, device=device),
            # Invalid checkpoint metadata can be rejected without sparse NPU
            # kernels or copying the malformed tensor onto the device.
            "sparse": torch.ones(shape, device="cpu").to_sparse(),
            "replicated": replicated,
            "wrong_mesh": wrong_mesh,
            "wrong_local_shape": wrong_local_shape,
        }
        cases = [(kind, key, candidate) for kind, candidate in candidates.items()]
        partial_counter = DTensor.from_local(
            torch.tensor(float(rank + 1), device=device), mesh, [Partial()]
        )
        sharded_counter = DTensor.from_local(
            torch.zeros(1 if rank == 0 else 0, device=device),
            mesh,
            [Shard(0)],
            shape=torch.Size((1,)),
            stride=(1,),
        )
        cases.extend(
            [
                ("counter_Partial", "state.weight.step", partial_counter),
                ("counter_Shard", "state.weight.step", sharded_counter),
            ]
        )
        # Exercise actual NPU scalar reads as well as layout rejection. Only
        # rank zero receives these corrupt counters in the default run.
        for kind, value in (
            ("negative", -1.0),
            ("fractional", 0.5),
            ("nan", float("nan")),
            ("infinite", float("inf")),
            ("boolean", True),
        ):
            cases.append(
                (
                    f"counter_{kind}",
                    "state.weight.step",
                    torch.tensor(value, device=device),
                )
            )
        for kind, key, candidate in cases:
            malformed = dict(checkpoint)
            all_counter_ranks = args.all_counter_ranks and key.endswith(".step")
            if rank == 0 or all_counter_ranks:
                malformed[key] = candidate
            inner = optimizer.optimizers[0]
            original = inner.state[model.weight]
            values = deepcopy(original)
            error = None
            try:
                with (
                    CommDebugMode() if all_counter_ranks else nullcontext(None)
                ) as debug:
                    optimizer.load_state_dict(malformed)
            except (ValueError, RuntimeError) as caught:
                error = caught
            if all_counter_ranks:
                print(
                    f"counter validation communications rank={rank}: "
                    f"{debug.get_comm_counts()}"
                )
                assert debug.get_total_counts() == 1, debug.get_comm_counts()
            outcomes = [None, None]
            dist.all_gather_object(outcomes, (type(error).__name__, str(error)))
            assert outcomes[0][0] == "ValueError", (kind, outcomes)
            assert key in outcomes[0][1], outcomes
            if kind == "sparse":
                assert "tensor layout" in outcomes[0][1], outcomes
            if all_counter_ranks:
                assert outcomes[1][0] == "ValueError", outcomes
            else:
                assert outcomes[1][0] == "RuntimeError", outcomes
                assert "failed on another rank" in outcomes[1][1], outcomes
            assert inner.state[model.weight] is original
            torch.testing.assert_close(original, values, rtol=0, atol=0)
            optimizer.load_state_dict(checkpoint)
            print(f"PASS optimizer moment layout rejection/retry {kind} rank={rank}")
        # Preserve the existing fully replicated counter contract. Validation
        # reads its local tensor, while loading retains the original DTensor.
        replica_counter = DTensor.from_local(
            torch.tensor(0.0, device=device), mesh, [Replicate()]
        )
        valid = dict(checkpoint)
        valid["state.weight.step"] = replica_counter
        optimizer.load_state_dict(valid)
        restored_step = optimizer.optimizers[0].state[model.weight]["step"]
        assert isinstance(restored_step, DTensor)
        assert restored_step.placements == (Replicate(),)
        model.weight.grad = torch.ones_like(model.weight)
        before = model.weight.to_local().detach().clone()
        optimizer.step()
        assert torch.isfinite(model.weight.to_local()).all().item()
        local_weight = model.weight.to_local()
        if local_weight.numel():
            assert not torch.equal(local_weight, before)
        else:
            assert args.empty_shard and rank == 1
            assert torch.equal(local_weight, before)
        assert restored_step.to_local().item() == 1
        print(f"PASS replicated counter load and next update rank={rank}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
