"""Run with two CPU/Gloo ranks: local TP planning failures reach every peer."""

from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn

from llmtuner.parallel.tensor_parallel.checkpointing import (
    gather_tensor_parallel_state,
    load_tensor_parallel_state,
)

dist.init_process_group("gloo", timeout=timedelta(seconds=30))
try:
    rank = dist.get_rank()
    for loading in (False, True):
        for failure in (
            "shape",
            "missing_parameter",
            "packed",
            "manifest",
            "sparse",
            "meta",
        ):
            model = nn.Linear(2, 2, bias=False)
            model._tp_checkpoint_layouts = {"weight": (0, False, dist.group.WORLD)}
            state = {"weight": torch.ones(4 if loading else 2, 2)}
            if rank == 0:
                if failure == "shape":
                    state["weight"] = torch.ones(3, 2)
                elif failure == "missing_parameter":
                    model.register_parameter("weight", None)
                elif failure == "packed":
                    model.weight = nn.Parameter(torch.ones(3, 2))
                    model._tp_checkpoint_layouts["weight"] = (0, True, dist.group.WORLD)
                    state["weight"] = torch.ones(6 if loading else 3, 2)
                elif failure == "meta":
                    state["weight"] = torch.empty_like(state["weight"], device="meta")
                elif failure == "sparse":
                    state["weight"] = state["weight"].to_sparse()
                else:
                    state["weight"] = state["weight"].double()
            operation = (
                load_tensor_parallel_state if loading else gather_tensor_parallel_state
            )
            try:
                operation(state, model_parts=[model])
            except (ValueError, RuntimeError, KeyError):
                pass
            else:
                raise AssertionError(f"rank {rank} accepted {loading=} {failure=}")
            dist.barrier()
            print(f"rank {rank}: coordinated {loading=} {failure=}", flush=True)
    model = nn.Linear(2, 2, bias=False)
    model._tp_checkpoint_layouts = {"weight": (0, False, dist.group.WORLD)}
    state = {"weight": torch.full((2, 2), float(rank))}
    full = gather_tensor_parallel_state(state, model_parts=[model])
    restored = load_tensor_parallel_state(full, model_parts=[model])
    torch.testing.assert_close(restored["weight"], state["weight"], rtol=0, atol=0)
    print(f"rank {rank}: healthy retry passed", flush=True)
finally:
    dist.destroy_process_group()
