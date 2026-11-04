"""Two NPU ranks reject a local model/category collision before DCP IO.

Run explicitly with torchrun --nproc-per-node=2 -m
tests.integration_tests.checkpoint_flatten_coordination.
"""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist

from llmtuner.components.checkpointer.backends.distributed_checkpoint import (
    DistributedCheckpointManager,
)
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config import CheckpointConfig, OptimizerConfig
from tests.integration_tests.harness import init_distributed


def main():
    device = init_distributed()
    try:
        assert device.type == "npu" and dist.get_world_size() == 2
        rank = dist.get_rank()
        torch.manual_seed(91)
        model = torch.nn.Linear(4, 2, device=device)
        if rank == 0:
            model.register_buffer("extra", torch.ones(2, device=device))
        optimizer = OptimizersContainer(
            OptimizerConfig(implementation="for-loop"), model_parts=[model]
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer.optimizers[0], lambda _: 1
        )
        with TemporaryDirectory() as local_folder:
            folders = [local_folder]
            dist.broadcast_object_list(folders, src=0)
            manager = DistributedCheckpointManager(
                CheckpointConfig(enable=True, interval=1, keep_latest_k=0),
                model_parts=[model],
                optimizer=optimizer,
                lr_scheduler=scheduler,
                states={"extra": {"value": torch.ones(1)}},
                folder=folders[0],
            )
            try:
                before = deepcopy(model.state_dict())
                caught = None
                try:
                    manager.save(1)
                except (ValueError, RuntimeError) as error:
                    caught = error
                outcomes = [None, None]
                dist.all_gather_object(outcomes, (type(caught).__name__, str(caught)))
                assert outcomes[0][0] == "ValueError", outcomes
                assert "overlap training state categories" in outcomes[0][1], outcomes
                assert outcomes[1][0] == "RuntimeError", outcomes
                assert "flattening failed on another rank" in outcomes[1][1], outcomes
                assert not (Path(manager.folder) / "step-1" / ".metadata").exists()
                torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
                del manager.states["extra"]
                assert manager.save(1)
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.add_(1)
                assert manager.load(1)
                torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
                print(f"PASS checkpoint flatten rejection/retry/restore rank={rank}")
            finally:
                manager.close()
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
