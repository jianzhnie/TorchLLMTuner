"""Two NPU ranks: packed BF16 sharded norm, clipping and alias rejection.

PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
    tests/integration_tests/npu_packed_grad_norm.py

Uses an independent local packed norm and explicit HCCL sum as the reference.
This checks TorchLLMTuner contracts, not TorchTitan-NPU equivalence.
"""

from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from llmtuner.accelerator.collectives import clip_grad_norm_
from llmtuner.parallel.expert_parallel.npu_baseline_experts import (
    NpuBaselineGroupedExperts,
)
from llmtuner.trainer.runtime.gradients import clip_and_check_finite
from llmtuner.trainer.runtime.reductions import TrainingReductionMeshes
from tests.caps import require_env
from tests.integration_tests.harness import init_distributed


def main():
    require_env("torch_npu")
    device = init_distributed()
    assert device.type == "npu" and dist.get_world_size() == 2
    try:
        mesh = init_device_mesh("npu", (2,))
        # Two physical gate/up parameters represent one packed norm group.
        # Reduce native BF16 local norms in FP32, independently of the helper.
        pair = []
        for offset in (0.25, -0.5):
            values = torch.arange(256, device="npu").reshape(4, 8, 8).float()
            values = (values / 257 + offset).to(torch.bfloat16)
            gradient = distribute_tensor(values, mesh, [Shard(0)])
            parameter = torch.nn.Parameter(torch.zeros_like(gradient))
            parameter.grad = gradient
            pair.append(parameter)
        original = [p.grad.to_local().clone() for p in pair]
        for order in (1.0, 2.0):
            for parameter, value in zip(pair, original, strict=True):
                parameter.grad.to_local().copy_(value)
            packed = torch.cat(original, dim=1)
            power = torch.linalg.vector_norm(packed, order).float().pow(order)
            dist.all_reduce(power)
            expected = power.pow(1 / order)
            actual = clip_grad_norm_(
                pair,
                0.5,
                norm_type=order,
                norm_precision="parameter",
                packed_parameters=[tuple(pair)],
            )
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            coefficient = (0.5 / (expected + 1e-6)).clamp(max=1)
            for parameter, value in zip(pair, original, strict=True):
                torch.testing.assert_close(
                    parameter.grad.to_local(), value * coefficient, atol=0, rtol=0
                )
            assert actual.item() > 0.5
        # Exercise Trainer's collector with one real expert module registered
        # in two model parts. Its gate/up pair must reach clipping only once.
        experts = NpuBaselineGroupedExperts(8, 8, 4, use_grouped_mm=False)
        experts.w1_EFD, experts.w3_EFD = pair
        experts.w2_EDF.requires_grad_(False)
        parts = [torch.nn.Module(), torch.nn.Module()]
        for part in parts:
            part.experts = experts
        # A real but unused expert must not create a packed pair requiring
        # nonexistent gradients. Keep its parameters on the active device.
        parts[0].unused_experts = NpuBaselineGroupedExperts(
            8, 8, 4, use_grouped_mm=False
        ).to(device=device, dtype=torch.bfloat16)
        for parameter, value in zip(pair, original, strict=True):
            parameter.grad.to_local().copy_(value)
        trainer = SimpleNamespace(
            model_parts=parts,
            parallel_dims=None,
            cfg=SimpleNamespace(max_norm=0.5, grad_norm_precision="parameter"),
            step=1,
            before_gradient_clipping=lambda: None,
        )
        actual = clip_and_check_finite(
            trainer.model_parts,
            parallel_dims=trainer.parallel_dims,
            max_norm=trainer.cfg.max_norm,
            precision=trainer.cfg.grad_norm_precision,
            expert_reduce_layout="storage",
            step=trainer.step,
            reduction_meshes=TrainingReductionMeshes(None, None, None, None),
            loss_is_finite=torch.ones((), dtype=torch.int32, device=device),
        )
        assert all(
            parameter.grad is None for parameter in parts[0].unused_experts.parameters()
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for parameter, value in zip(pair, original, strict=True):
            torch.testing.assert_close(
                parameter.grad.to_local(), value * coefficient, atol=0, rtol=0
            )
        before = pair[0].grad.to_local().clone()
        try:
            clip_grad_norm_(
                pair,
                0.01,
                norm_precision="parameter",
                packed_parameters=[(pair[0], pair[0])],
            )
        except ValueError as error:
            assert "disjoint" in str(error)
        else:
            raise AssertionError("packed gradient alias was accepted")
        torch.testing.assert_close(pair[0].grad.to_local(), before, atol=0, rtol=0)
        print(
            f"PASS rank={dist.get_rank()}: packed BF16 norm, clipping, alias rejection"
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
