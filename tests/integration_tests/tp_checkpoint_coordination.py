"""Two CPU or accelerator ranks: TP planning failures reach every peer."""

from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn

from llmtuner.parallel.tensor_parallel import checkpointing
from llmtuner.parallel.tensor_parallel.checkpointing import (
    gather_tensor_parallel_state,
    load_tensor_parallel_state,
)
from tests.integration_tests.harness import init_distributed

device = init_distributed(timeout=timedelta(seconds=90))
try:
    assert dist.get_world_size() == 2
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
            model = nn.Linear(2, 2, bias=False, device=device)
            model._tp_checkpoint_layouts = {"weight": (0, False, dist.group.WORLD)}
            state = {"weight": torch.ones(4 if loading else 2, 2, device=device)}
            if rank == 0:
                if failure == "shape":
                    state["weight"] = torch.ones(3, 2, device=device)
                elif failure == "missing_parameter":
                    model.register_parameter("weight", None)
                elif failure == "packed":
                    model.weight = nn.Parameter(torch.ones(3, 2, device=device))
                    model._tp_checkpoint_layouts["weight"] = (0, True, dist.group.WORLD)
                    state["weight"] = torch.ones(6 if loading else 3, 2, device=device)
                elif failure == "meta":
                    state["weight"] = torch.empty_like(state["weight"], device="meta")
                elif failure == "sparse":
                    state["weight"] = state["weight"].cpu().to_sparse()
                else:
                    state["weight"] = state["weight"].to(torch.bfloat16)
            operation = (
                load_tensor_parallel_state if loading else gather_tensor_parallel_state
            )
            error = None
            try:
                operation(state, model_parts=[model])
            except (ValueError, RuntimeError, KeyError) as caught:
                error = caught
            outcomes = [None, None]
            dist.all_gather_object(outcomes, (type(error).__name__, str(error)))
            if failure == "manifest":
                assert all(kind == "ValueError" for kind, _ in outcomes), outcomes
                assert all(
                    "differ across ranks" in message for _, message in outcomes
                ), outcomes
            else:
                expected = (
                    "KeyError" if failure == "missing_parameter" else "ValueError"
                )
                assert outcomes[0][0] == expected, outcomes
                assert outcomes[1][0] == "RuntimeError", outcomes
                assert "preparation failed on another rank" in outcomes[1][1], outcomes
            dist.barrier()
            print(f"rank {rank}: coordinated {loading=} {failure=}", flush=True)
    for operation in (gather_tensor_parallel_state, load_tensor_parallel_state):
        for error_type in (KeyboardInterrupt, SystemExit):
            for failing_rank in (0, 1):
                model = nn.Linear(2, 2, bias=False, device=device)
                model._tp_checkpoint_layouts = {"weight": (0, False, dist.group.WORLD)}
                original = error_type("interrupted TP checkpoint planning")
                index_layouts = checkpointing._index_layouts

                def interrupted_index(
                    parts,
                    failing_rank=failing_rank,
                    original=original,
                    index_layouts=index_layouts,
                ):
                    if rank == failing_rank:
                        raise original
                    return index_layouts(parts)

                with patch.object(checkpointing, "_index_layouts", interrupted_index):
                    try:
                        operation(
                            {
                                "weight": torch.ones(
                                    4 if operation is load_tensor_parallel_state else 2,
                                    2,
                                    device=device,
                                )
                            },
                            model_parts=[model],
                        )
                    except BaseException as error:
                        if rank == failing_rank:
                            assert error is original
                        else:
                            assert isinstance(error, RuntimeError)
                            assert "preparation failed on another rank" in str(error)
                    else:
                        raise AssertionError("TP interruption was accepted")
                dist.barrier()
    print(f"rank {rank}: TP planning interrupts coordinated", flush=True)
    model = nn.Linear(2, 2, bias=False, device=device)
    model._tp_checkpoint_layouts = {"weight": (0, False, dist.group.WORLD)}
    state = {"weight": torch.full((2, 2), float(rank), device=device)}
    full = gather_tensor_parallel_state(state, model_parts=[model])
    restored = load_tensor_parallel_state(full, model_parts=[model])
    torch.testing.assert_close(restored["weight"], state["weight"], rtol=0, atol=0)
    print(f"rank {rank}: healthy retry passed", flush=True)
finally:
    dist.destroy_process_group()
