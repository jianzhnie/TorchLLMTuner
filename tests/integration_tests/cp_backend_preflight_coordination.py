"""Reject removed CP backends on every rank before any parallel transform.

PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
    tests/integration_tests/cp_backend_preflight_coordination.py

Use the integration harness device selection for CPU/Gloo or NPU/HCCL.
"""

from contextlib import ExitStack
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.parallel import parallelize
from llmtuner.parallel.context_parallel import plan as cp_plan
from tests.integration_tests.harness import init_distributed


def main() -> None:
    device = init_distributed(timeout=timedelta(seconds=45))
    try:
        if dist.get_world_size() != 2:
            raise ValueError("This test requires two ranks")
        rank = dist.get_rank()
        mesh = init_device_mesh(device.type, (2,), mesh_dim_names=("world",))
        topology = SimpleNamespace(
            pp_enabled=False,
            get_optional_mesh=lambda name: None,
            build_device_mesh=lambda: mesh,
        )
        for failing_rank in (0, 1):
            model = nn.Linear(2, 2, device=device)
            if rank != failing_rank:
                model.context_parallel_backend = "compressor"
            parameters = tuple(model.parameters())
            values = tuple(parameter.detach().clone() for parameter in parameters)
            rng = torch.get_rng_state().clone()
            with ExitStack() as stack:
                # This test isolates cross-rank rejection of removed backends;
                # Compressor geometry is covered by its own resolver tests.
                stack.enter_context(
                    patch.object(cp_plan, "resolve_compressor_plan", return_value=None)
                )
                transforms = [
                    stack.enter_context(
                        patch.object(
                            parallelize,
                            name,
                            side_effect=AssertionError("Parallel transform began"),
                        )
                    )
                    for name in (
                        "apply_tp",
                        "apply_ep",
                        "apply_cp",
                        "apply_pp",
                        "apply_ac",
                        "apply_fsdp",
                    )
                ]
                try:
                    parallelize.parallelize_model(
                        model,
                        cfg=ParallelConfig(context_parallel_kv_allgather_size=2),
                        mesh=None,
                        parallel_dims=topology,
                    )
                except NotImplementedError as error:
                    assert rank == failing_rank
                    assert "kernels have been removed" in str(error)
                except RuntimeError as error:
                    assert rank != failing_rank
                    assert str(error) == "CP backend validation failed on another rank"
                else:
                    raise AssertionError("Unsupported CP unexpectedly succeeded")
                for transform in transforms:
                    transform.assert_not_called()
            for current, original, value in zip(
                model.parameters(), parameters, values, strict=True
            ):
                assert current is original
                torch.testing.assert_close(current, value, rtol=0, atol=0)
            assert torch.equal(torch.get_rng_state(), rng)
            dist.barrier()
            if rank == 0:
                print(
                    f"failing_rank={failing_rank}: CP backend rejected on both ranks "
                    "before transforms; parameters and RNG unchanged",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
