"""CPU/Gloo regression for compiled full-AC all-to-all ownership.

Run with ``torchrun --standalone --nproc_per_node=2 -m
tests.integration_tests.full_ac_collective_replay``. Check output and gradients
against an analytic permutation, and count real exchanges through forward and
backward. The checkpoint must reuse the forward exchange during replay.
"""

from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import (
    all_to_all_single_autograd,
    wait_tensor,
)
from torch.utils.checkpoint import DefaultDeviceType

from llmtuner.memory.activation_checkpoint import wrap_full


def exchange(values: torch.Tensor) -> torch.Tensor:
    received = all_to_all_single_autograd(values, None, None, dist.group.WORLD)
    return wait_tensor(received).sin()


class Exchange(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return exchange(values)


def main() -> None:
    previous_device = DefaultDeviceType.get_device_type()
    DefaultDeviceType.set_device_type("cpu")
    try:
        dist.init_process_group("gloo", timeout=timedelta(seconds=60))
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        compiled = torch.compile(
            wrap_full(Exchange()), backend="aot_eager", fullgraph=True
        )
        for step in range(2):
            values = (
                torch.arange(8, dtype=torch.float32).reshape(4, 2) / 10
                + rank
                + step / 5
            ).requires_grad_()
            expected = torch.cat(
                [
                    (
                        torch.arange(8, dtype=torch.float32).reshape(4, 2) / 10
                        + source
                        + step / 5
                    )[rank * 2 : (rank + 1) * 2]
                    for source in range(2)
                ]
            ).sin()
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as communication:
                output = compiled(values)
                output.square().sum().backward()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                values.grad, 2 * values.detach().sin() * values.detach().cos()
            )
            counts = {event.key: event.count for event in communication.key_averages()}
            exchanges = counts.get("c10d::alltoall_base_", 0)
            assert exchanges == 2, (
                f"Expected two exchanges, observed {exchanges}"
            )  # One forward and its inverse backward.
        dist.barrier()
        print(f"rank {rank}: compiled full-AC exchange and gradients passed")
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            DefaultDeviceType.set_device_type(previous_device)


if __name__ == "__main__":
    main()
