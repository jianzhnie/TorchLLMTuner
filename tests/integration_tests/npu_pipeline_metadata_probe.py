"""Four ranks: isolate repeated PP-group object or tensor P2P without a Trainer.

Groups [0, 2] and [1, 3] match PP2 over two data ranks. Select visible devices
explicitly to test intra-node or cross-node transport. Non-symmetric payloads
and repeated exchanges detect stale messages, not just successful startup.
"""

import argparse
from datetime import timedelta

import torch
import torch.distributed as dist

from tests.integration_tests.harness import init_distributed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--tensor-elements", type=int, default=2048)
    parser.add_argument(
        "--dtype", choices=("float32", "int32", "int64"), default="float32"
    )
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--unbatched", action="store_true")
    parser.add_argument("--tensor-exchange", action="store_true")
    parser.add_argument(
        "--ordered", action="store_true", help="Higher peer receives first (unbatched)."
    )
    args = parser.parse_args()
    if args.iterations < 1 or args.tensor_elements < 1:
        parser.error("--iterations and --tensor-elements must be positive")
    if not args.tensor_exchange and (
        args.dtype != "float32" or args.tensor_elements != 2048
    ):
        parser.error("--dtype and --tensor-elements require --tensor-exchange")
    if args.ordered and not (args.unbatched and args.tensor_exchange):
        parser.error("--ordered requires --unbatched --tensor-exchange")
    device = init_distributed(timeout=timedelta(seconds=45))
    try:
        assert dist.get_world_size() == 4
        rank = dist.get_rank()
        groups = [dist.new_group(ranks) for ranks in ([0, 2], [1, 3])]
        group = groups[rank % 2]
        if args.warmup:
            token = torch.ones(1, device=device)
            dist.all_reduce(token, group=group)
            assert token.item() == 2
        if args.tensor_exchange:
            peer = rank + 2 if rank < 2 else rank - 2
            for iteration in range(args.iterations):
                sent = torch.full(
                    (args.tensor_elements,),
                    rank + iteration,
                    device=device,
                    dtype=getattr(torch, args.dtype),
                )
                received = torch.empty_like(sent)
                ops = [
                    dist.P2POp(dist.isend, sent, peer, group),
                    dist.P2POp(dist.irecv, received, peer, group),
                ]
                if args.ordered and rank > peer:
                    ops.reverse()
                if args.unbatched:
                    works = [op.op(op.tensor, op.peer, op.group) for op in ops]
                else:
                    works = dist.batch_isend_irecv(ops)
                for work in works:
                    work.wait()
                torch.testing.assert_close(
                    received,
                    torch.full_like(received, peer + iteration),
                    atol=0,
                    rtol=0,
                )
            dist.barrier()
            print(
                f"PASS rank={rank} tensor_exchange=True "
                f"unbatched={args.unbatched} ordered={args.ordered} "
                f"dtype={args.dtype} elements={args.tensor_elements} "
                f"iterations={args.iterations}"
            )
            return
        for iteration in range(args.iterations):
            payload = {"shape": (1, 16, 32), "dtype": "float32", "iteration": iteration}
            if rank < 2:
                dist.send_object_list(
                    [payload],
                    dst=rank + 2,
                    group=group,
                    device=device,
                    use_batch=not args.unbatched,
                )
            else:
                objects = [None]
                dist.recv_object_list(
                    objects,
                    src=rank - 2,
                    group=group,
                    device=device,
                    use_batch=not args.unbatched,
                )
                assert objects == [payload]
        dist.barrier()
        print(
            f"PASS rank={rank} warmup={args.warmup} unbatched={args.unbatched} "
            f"iterations={args.iterations}"
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
