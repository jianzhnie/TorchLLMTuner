"""Real NPU Adam/AdamW dispatch and empty-shard swap regression.

PYTHONPATH=. torchrun --standalone --nproc-per-node=2 \
  tests/integration_tests/npu_optimizer_kernel_selection.py
"""

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from tests.caps import require_env

require_env("torch_npu")

from llmtuner.accelerator.device import get_current_device, set_device
from llmtuner.components.optimizer.adamw_swap import AdamWSwap
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.components.optimizer.state_residency import install_state_residency
from llmtuner.config.optimizer_config import OptimizerConfig


def main() -> None:
    device = get_current_device()
    set_device(device)
    dist.init_process_group("hccl")
    try:
        for name in ("Adam", "AdamW"):
            model = torch.nn.Linear(3, 2, bias=False, device=device)
            reference = torch.nn.Parameter(model.weight.detach().clone())
            optimizer = OptimizersContainer(
                OptimizerConfig(optimizer_type=name, implementation="fused"),
                model_parts=[model],
            )
            operator = "aten::_fused_adam_" if name == "Adam" else "aten::_fused_adamw_"
            supported = torch._C._dispatch_has_kernel_for_dispatch_key(
                operator, "PrivateUse1"
            )
            inner = optimizer.optimizers[0]
            assert all(
                bool(group["fused"]) == supported for group in inner.param_groups
            )
            native = getattr(torch.optim, name)(
                [reference], lr=3e-4, weight_decay=0.0, fused=supported, foreach=False
            )
            for step in range(3):
                gradient = torch.full_like(reference, 0.2 * (step + 1))
                model.weight.grad = gradient.clone()
                reference.grad = gradient.clone()
                optimizer.step()
                native.step()
                torch.testing.assert_close(model.weight, reference, rtol=0, atol=0)
            print(
                f"PASS rank={dist.get_rank()} {name} native_fused={supported}",
                flush=True,
            )

        mesh = init_device_mesh(device.type, (dist.get_world_size(),))
        # A global single element leaves some ranks with genuinely empty local
        # ownership. Do not require staging capacity on those ranks.
        rank = dist.get_rank()
        payload = torch.ones(1 if rank == 0 else 0, device=device)
        parameter = torch.nn.Parameter(
            DTensor.from_local(
                payload,
                mesh,
                [Shard(0)],
                shape=torch.Size([1]),
                stride=(1,),
                run_check=False,
            )
        )
        swap = AdamWSwap([parameter], prefetch=True, max_device_bytes=8 if rank else 32)
        parameter.grad = DTensor.from_local(
            torch.ones_like(payload),
            mesh,
            [Shard(0)],
            shape=torch.Size([1]),
            stride=(1,),
            run_check=False,
        )
        swap.step()
        assert swap.state[parameter]["step"].item() == 1
        if rank:
            assert swap.staging_peak_payload_bytes == swap.host_state_bytes == 0

        # A globally matching moment must still have the destination's local
        # shard shape. Reject malformed input before installing virtual state.
        virtual = torch.optim.AdamW([parameter], foreach=False)
        install_state_residency(virtual, "npu_virtual")
        checkpoint = virtual.state_dict()
        wrong = DTensor(
            torch.zeros(payload.numel() + 1, device=device),
            parameter._spec,
            requires_grad=False,
        )
        checkpoint["state"][0] = {
            "step": torch.tensor(1.0),
            "exp_avg": wrong,
            "exp_avg_sq": wrong,
        }
        try:
            virtual.load_state_dict(checkpoint)
        except ValueError as error:
            assert "ownership mismatch" in str(error)
        else:
            raise AssertionError("wrong local ownership accepted")
        assert not virtual.state
        print(
            f"PASS rank={rank} empty ownership and virtual restore validation",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
