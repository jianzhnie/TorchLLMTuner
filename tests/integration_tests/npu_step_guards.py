"""Real HCCL step-boundary guards, including a globally rejected step.

Run with two NPU ranks. This auxiliary test does not replace model training.
"""

import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode

from llmtuner.trainer.runtime.gradients import check_finite, check_step_condition
from tests.caps import require_env
from tests.integration_tests.harness import init_distributed


class RejectUnsupportedAssertion(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (
            torch.ops.aten._assert_async.default,
            torch.ops.aten._assert_async.msg,
        ):
            raise AssertionError("NPU guard invoked unsupported ATen assertion")
        return func(*args, **(kwargs or {}))


def main() -> None:
    require_env("torch_npu")
    device = init_distributed()
    try:
        assert device.type == "npu"
        rank = dist.get_rank()
        assert dist.get_world_size() == 2
        with RejectUnsupportedAssertion():
            check_step_condition(torch.tensor(2, device=device), "unexpected failure")
            # Rank 1 injects a failure; all ranks reject the same global condition.
            finite = torch.tensor(rank == 0, dtype=torch.int32, device=device)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            try:
                check_finite(7, finite.bool())
            except RuntimeError as error:
                assert "not finite at step 7" in str(error)
            else:
                raise AssertionError("Invalid global step was accepted")
            try:
                check_step_condition(torch.tensor(0, device=device), "zero targets")
            except RuntimeError as error:
                assert "zero targets" in str(error)
            else:
                raise AssertionError("Zero-target step was accepted")
        dist.barrier()
        if rank == 0:
            print("NPU step guards passed on 2 ranks; unsupported assertion not used")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
