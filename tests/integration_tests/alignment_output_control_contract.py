"""Real two-rank contract for diagnostic output layout and gradient ownership."""

from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from tests.integration_tests.alignment_output_control import (
    install_output_layout_control,
    local_token_logits,
)
from tests.integration_tests.harness import init_distributed


def main():
    init_distributed()
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != 2:
            raise ValueError("Run this contract with exactly two ranks")
        device = "npu" if dist.get_backend() == "hccl" else "cpu"
        dtype = torch.float32 if device == "npu" else torch.float64
        mesh = init_device_mesh(device, (world,))
        torch.manual_seed(103)
        head = torch.nn.Linear(7, 12, bias=False, device=device, dtype=dtype)
        reference = torch.nn.Linear(7, 12, bias=False, device=device, dtype=dtype)
        reference.load_state_dict(head.state_dict())
        inputs = torch.randn(8, 7, device=device, dtype=dtype)
        labels = torch.tensor([1, 4, -100, 0, 8, 3, 11, 2], device=device)
        full_input = inputs.detach().clone().requires_grad_()
        local_input = inputs.chunk(world)[rank].detach().clone().requires_grad_()
        dims = SimpleNamespace(tp=world, cp=1, pp=1, get_mesh=lambda name: mesh)
        trainer = SimpleNamespace(
            parallel_dims=dims,
            model_parts=[SimpleNamespace(lm_head=head)],
            param_context=nullcontext,
        )
        install_output_layout_control(trainer)
        with trainer.param_context():
            output = head(local_input)
            original = reference(full_input)
            torch.testing.assert_close(
                local_token_logits(output, mesh), original.chunk(world)[rank]
            )
            loss = trainer.loss_sum(output, labels.chunk(world)[rank])
            expected = torch.nn.functional.cross_entropy(
                original.float(), labels, reduction="sum"
            )
            torch.testing.assert_close(loss * world, expected)
            (loss / 7).backward()
            (expected / 7).backward()
        dist.all_reduce(head.weight.grad)
        torch.testing.assert_close(head.weight.grad, reference.weight.grad)
        torch.testing.assert_close(local_input.grad, full_input.grad.chunk(world)[rank])
        print(f"PASS rank{rank}: logits, loss accounting, input and weight gradients")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
