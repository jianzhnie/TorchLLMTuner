"""Two-rank checkpoint constructors reject backend/enable/async disagreement.

Run with torchrun --nproc_per_node=2 on CPU/Gloo.
"""

from datetime import timedelta

import torch
import torch.distributed as dist

from llmtuner.components.checkpointer.distributed_checkpoint import (
    DistributedCheckpointManager,
)
from llmtuner.components.checkpointer.torch_checkpointing_backend import (
    TorchCheckpointingManager,
)
from llmtuner.config.checkpoint_config import CheckpointConfig


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=5))
    manager = None
    try:
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        for case in (
            "load_only",
            "optional_load_only",
            "enable",
            "async",
            "optional_enable",
            "backend",
        ):
            config = CheckpointConfig(
                enable=dist.get_rank() == 0
                if case in {"enable", "optional_enable"}
                else True,
                load_only=bool(dist.get_rank()) if "load_only" in case else False,
                async_mode="async"
                if case == "async" and dist.get_rank()
                else "disabled",
            )
            manager_class = (
                TorchCheckpointingManager
                if case in {"optional_enable", "optional_load_only"}
                or (case == "backend" and dist.get_rank())
                else DistributedCheckpointManager
            )
            try:
                manager = manager_class(
                    config,
                    model_parts=[model],
                    optimizer=optimizer,
                    lr_scheduler=torch.optim.lr_scheduler.LambdaLR(
                        optimizer, lambda _: 1
                    ),
                    states={},
                    folder="/tmp/llmtuner-constructor-319",
                )
            except RuntimeError as error:
                assert "configuration differs across ranks" in str(error), str(error)
            else:
                raise AssertionError(f"Mismatched {case} settings were accepted")
            dist.barrier()
            print(f"rank={dist.get_rank()} {case} mismatch refused before setup")
        for manager_class in (DistributedCheckpointManager, TorchCheckpointingManager):
            manager = manager_class(
                CheckpointConfig(enable=False),
                model_parts=[model],
                optimizer=optimizer,
                lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
                states={},
                folder="/tmp/llmtuner-constructor-319",
            )
            assert not manager.save(1)
            assert not manager.load()
            manager.close()
            manager = None
            dist.barrier()
    finally:
        if manager is not None:
            manager.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
