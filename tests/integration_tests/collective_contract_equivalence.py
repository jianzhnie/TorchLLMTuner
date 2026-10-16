"""Two-rank Gloo reductions, strided writeback and local failure recovery.

Run with PYTHONPATH=. torchrun --standalone --nproc_per_node=2
    tests/integration_tests/collective_contract_equivalence.py
"""

from datetime import timedelta

import torch
import torch.distributed as dist

from llmtuner.accelerator.collectives import (
    PartialToInvariantAllReduce,
    all_reduce,
    synchronize_local_errors,
)


def main() -> None:
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2
    try:
        for op, expected in [
            ("sum", [3.0, 5.0]),
            ("mean", [1.5, 2.5]),
            ("product", [2.0, 6.0]),
            ("min", [1.0, 2.0]),
            ("max", [2.0, 3.0]),
        ]:
            storage = torch.zeros(2, 2, dtype=torch.float64)
            values = storage[:, 0]
            values.copy_(torch.tensor([rank + 1, rank + 2]))
            all_reduce(values, op=op)
            torch.testing.assert_close(
                values, torch.tensor(expected, dtype=torch.float64), rtol=0, atol=0
            )
            assert torch.equal(storage[:, 1], torch.zeros(2, dtype=torch.float64))
        x = torch.tensor([rank + 1.0], requires_grad=True)
        y = PartialToInvariantAllReduce.apply(x, None)
        torch.testing.assert_close(y, torch.tensor([3.0]), rtol=0, atol=0)
        y.square().sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([6.0]), rtol=0, atol=0)
        assert x.item() == rank + 1
        for kind in (ValueError, KeyboardInterrupt, SystemExit):
            original = kind("rank local")
            try:
                with synchronize_local_errors("contract review"):
                    if rank == 1:
                        raise original
            except BaseException as error:
                if rank == 1:
                    assert error is original
                else:
                    assert type(error) is RuntimeError and "contract review" in str(
                        error
                    )
            else:
                raise AssertionError("missing coordinated error")
            probe = torch.tensor(rank + 1.0)
            all_reduce(probe)
            assert probe.item() == 3
        print(
            f"rank {rank}: reductions, strided copy, partial autograd, "
            "interrupt coordination/retry passed",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
