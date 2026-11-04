"""Two-rank shared snapshot ownership, conversion failure and retry.

Run with torchrun --nproc-per-node=2; pass --backend hccl for real NPU tensors.
"""

import argparse
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from tests.caps import require_env
from tests.integration_tests.npu_baseline_training_capture import shared_hf_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gloo", "hccl"), default="gloo")
    args = parser.parse_args()
    device = "cpu"
    if args.backend == "hccl":
        require_env("torch_npu")
        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
        device = "npu"
    dist.init_process_group(args.backend)
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        mesh = init_device_mesh(device, (2,))
        full = torch.arange(24, dtype=torch.float32, device=device).reshape(6, 4)
        model = nn.Module()
        model.register_parameter(
            "weight", nn.Parameter(distribute_tensor(full.clone(), mesh, [Shard(0)]))
        )

        class Adapter:
            fail = False

            def to_hf(self, state):
                assert rank == 0, "nonwriters must not convert or retain CPU state"
                assert state["weight"].device.type == "cpu"
                if self.fail:
                    raise ValueError("injected conversion failure")
                return {"canonical.weight": state["weight"]}

        adapter = Adapter()
        first, first_count = shared_hf_snapshot(model, adapter)
        assert first_count == 1, "all ranks need the canonical tensor count"
        if rank == 0:
            torch.testing.assert_close(first["canonical.weight"], full.cpu())
        else:
            assert first is None
        with torch.no_grad():
            model.weight.to_local().add_(7)
        adapter.fail = True
        try:
            shared_hf_snapshot(model, adapter)
        except RuntimeError as error:
            assert "injected conversion failure" in str(error)
        else:
            raise AssertionError("all ranks must receive conversion failure")
        adapter.fail = False
        restored, restored_count = shared_hf_snapshot(model, adapter)
        assert restored_count == first_count
        if rank == 0:
            torch.testing.assert_close(restored["canonical.weight"], full.cpu() + 7)
            # Earlier observations own their storage and remain unchanged.
            torch.testing.assert_close(first["canonical.weight"], full.cpu())
        else:
            assert restored is None
        print(f"PASS rank={rank}: writer ownership, conversion failure and retry")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
