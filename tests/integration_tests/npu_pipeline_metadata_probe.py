"""Four ranks: isolate cross-node HCCL object P2P used by pipeline metadata."""

import argparse
from datetime import timedelta

import torch
import torch.distributed as dist

from tests.integration_tests.harness import init_distributed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--unbatched", action="store_true")
    args = parser.parse_args()
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
        payload = {"shape": (1, 16, 32), "dtype": "float32"}
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
        print(f"PASS rank={rank} warmup={args.warmup} unbatched={args.unbatched}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
