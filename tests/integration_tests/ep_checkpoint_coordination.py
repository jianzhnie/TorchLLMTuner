"""EP checkpoint preflight with real collective and asymmetric state failures.

PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
    tests/integration_tests/ep_checkpoint_coordination.py

Also supports four ranks: each PP-like pair owns a separate EP group, so
stage-local state names must not be compared across the world group.
"""

from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist

from llmtuner.parallel.expert_parallel import checkpointing
from llmtuner.parallel.expert_parallel.checkpointing import (
    expert_shard_map,
    gather_expert_state,
    mark_experts_ep_sharded,
)


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    try:
        world = dist.get_world_size()
        assert world in (2, 4)
        rank = dist.get_rank()
        groups = [dist.new_group([start, start + 1]) for start in range(0, world, 2)]
        group = groups[rank // 2]
        stage = rank // 2
        for case in ("missing", "empty", "dtype", "shape", "local_shape", "collision"):
            for failing_rank in (0, 1):
                model = torch.nn.Module()
                name = f"stage{stage}_weight"
                model.register_parameter(name, torch.nn.Parameter(torch.zeros(1, 2)))
                mark_experts_ep_sharded(model, group)
                parts = [model]
                state = {name: torch.full((1, 2), float(rank))}
                state[f"state.{name}.ema_params"] = state[name].clone()
                failed = rank % 2 == failing_rank
                if failed:
                    if case == "missing":
                        state.pop(f"state.{name}.ema_params")
                    elif case == "empty":
                        state.clear()
                    elif case == "dtype":
                        state[name] = state[name].double()
                    elif case == "shape":
                        model._parameters[name] = torch.nn.Parameter(torch.zeros(2, 2))
                        state = {
                            name: torch.zeros(2, 2),
                            f"state.{name}.ema_params": torch.zeros(2, 2),
                        }
                    elif case == "local_shape":
                        state[name] = torch.zeros(3, 2)
                    else:
                        duplicate = torch.nn.Module()
                        duplicate.register_parameter(
                            name, torch.nn.Parameter(torch.zeros(1, 2))
                        )
                        parts.append(duplicate)
                with patch.object(dist, "all_gather_into_tensor") as gather:
                    try:
                        gather_expert_state(
                            state,
                            shard_map=expert_shard_map([model]),
                            model_parts=parts,
                        )
                    except ValueError as error:
                        if case in ("local_shape", "collision"):
                            assert failed
                            assert "incompatible shape" in str(
                                error
                            ) or "identifies different parameters" in str(error)
                        else:
                            assert "entries differ across ranks" in str(error)
                    except RuntimeError as error:
                        assert case in ("local_shape", "collision") and not failed
                        assert "preparation failed on another rank" in str(error)
                    else:
                        raise AssertionError(f"rank={rank} {case}: preflight succeeded")
                    gather.assert_not_called()
                dist.barrier()
        for error_type in (KeyboardInterrupt, SystemExit):
            for failing_rank in (0, 1):
                model = torch.nn.Linear(2, 2, bias=False)
                mark_experts_ep_sharded(model, group)
                original = error_type("interrupted EP checkpoint planning")
                index_parameters = checkpointing.index_checkpoint_parameters

                def interrupted_index(
                    parts,
                    failing_rank=failing_rank,
                    original=original,
                    index_parameters=index_parameters,
                ):
                    if rank % 2 == failing_rank:
                        raise original
                    return index_parameters(parts)

                with (
                    patch.object(
                        checkpointing, "index_checkpoint_parameters", interrupted_index
                    ),
                    patch.object(dist, "all_gather_into_tensor") as gather,
                ):
                    try:
                        gather_expert_state(
                            model.state_dict(),
                            shard_map=expert_shard_map([model]),
                            model_parts=[model],
                        )
                    except BaseException as error:
                        if rank % 2 == failing_rank:
                            assert error is original
                        else:
                            assert isinstance(error, RuntimeError)
                            assert "preparation failed on another rank" in str(error)
                    else:
                        raise AssertionError("EP interruption was accepted")
                    gather.assert_not_called()
                dist.barrier()
        print(f"rank={rank}: EP planning interrupts coordinated", flush=True)
        # A fresh valid snapshot on the same EP groups must still work.
        model = torch.nn.Module()
        name = f"stage{stage}_weight"
        model.register_parameter(name, torch.nn.Parameter(torch.zeros(1, 2)))
        mark_experts_ep_sharded(model, group)
        result = gather_expert_state(
            {name: torch.full((1, 2), float(rank))},
            shard_map=expert_shard_map([model]),
            model_parts=[model],
        )
        expected = torch.tensor([[2.0 * stage] * 2, [2.0 * stage + 1] * 2])
        torch.testing.assert_close(result[name], expected, rtol=0, atol=0)
        if world == 4:
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.tensor import Shard, distribute_tensor

            mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("efsdp", "ep"))
            ep_rank = mesh["ep"].get_local_rank()
            logical = torch.tensor([[10.0 * ep_rank] * 2, [10.0 * ep_rank + 1] * 2])
            sharded = torch.nn.Module()
            sharded.weight = torch.nn.Parameter(
                distribute_tensor(logical, mesh["efsdp"], [Shard(0)])
            )
            mark_experts_ep_sharded(sharded, mesh["ep"].get_group())
            full = gather_expert_state(
                sharded.state_dict(),
                shard_map=expert_shard_map([sharded]),
                model_parts=[sharded],
            )
            torch.testing.assert_close(
                full["weight"],
                torch.tensor([[0.0] * 2, [1.0] * 2, [10.0] * 2, [11.0] * 2]),
                rtol=0,
                atol=0,
            )
        dist.barrier()
        print(
            f"rank={rank}: EP checkpoint preflight and healthy retry passed", flush=True
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
