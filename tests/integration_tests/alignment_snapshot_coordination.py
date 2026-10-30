"""Verify shared alignment snapshots and writer failures with real collectives.

Run with torchrun --nproc_per_node=2 on CPU/Gloo or NPU/HCCL.
"""

from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.distributed as dist

from tests.integration_tests.alignment_history import save_shared_snapshot
from tests.integration_tests.harness import init_distributed


def main() -> None:
    init_distributed(timeout=timedelta(seconds=30))
    try:
        rank = dist.get_rank()
        assert dist.get_world_size() >= 2
        with TemporaryDirectory() if rank == 0 else nullcontext(None) as directory:
            directories = [directory]
            dist.broadcast_object_list(directories, src=0)
            root = Path(directories[0])
            state = {"weight": torch.arange(4)}
            save_shared_snapshot(state, root / "initial-hf.pt")
            actual = torch.load(root / "initial-hf.pt", weights_only=True)
            torch.testing.assert_close(actual["weight"], state["weight"])

            def reject(value):
                raise ValueError("injected HF mismatch")

            for failure in ("validation", "write"):
                path = root / f"{failure}.pt"
                context = (
                    patch("torch.save", side_effect=OSError("injected disk full"))
                    if failure == "write"
                    else nullcontext()
                )
                with context:
                    try:
                        save_shared_snapshot(
                            state,
                            path,
                            validate=reject if failure == "validation" else None,
                        )
                    except RuntimeError as error:
                        message = str(error)
                    else:
                        raise AssertionError(f"Rank {rank} accepted {failure} failure")
                messages = [None] * dist.get_world_size()
                dist.all_gather_object(messages, message)
                assert len(set(messages)) == 1, messages
                assert not path.exists()

            save_shared_snapshot(state, root / "final-hf.pt")
            dist.barrier()
            if rank == 0:
                print("PASS shared snapshot publication and coordinated failures")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
