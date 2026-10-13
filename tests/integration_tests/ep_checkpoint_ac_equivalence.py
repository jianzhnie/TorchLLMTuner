"""EP checkpoint with activation wrappers: real DCP model/optimizer/EMA restore.

    PYTHONPATH=. torchrun --nproc_per_node=2 \
        tests/integration_tests/ep_checkpoint_ac_equivalence.py

Also restore the full expert checkpoint into EP=1 on each rank.
"""

import shutil
import tempfile

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)

from llmtuner.components.checkpointer.model_state import ModelCheckpointState
from llmtuner.components.optimizer import EMA, OptimizersContainer
from llmtuner.config import OptimizerConfig
from llmtuner.models.common.moe.experts import GroupedExperts
from llmtuner.parallel.expert_parallel.checkpointing import mark_experts_ep_sharded


def build(rank, *, wrapped, ep):
    model = GroupedExperts(8, 12, 2 if ep else 4, use_grouped_mm=False).double()
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.fill_(index + rank * 10 + 1)
    if ep:
        mark_experts_ep_sharded(model, dist.group.WORLD)
    if wrapped:
        model = checkpoint_wrapper(model)
    optimizer = OptimizersContainer(
        OptimizerConfig(learning_rate=0.01, implementation="for-loop"),
        model_parts=[model],
    )
    ema = EMA(model_parts=[model], decay=0.5)
    return model, optimizer, ema


def main():
    dist.init_process_group("gloo")
    folder = None
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        folders = [tempfile.mkdtemp(prefix="llmtuner-ep-ac-") if rank == 0 else None]
        dist.broadcast_object_list(folders)
        folder = folders[0]
        model, optimizer, ema = build(rank, wrapped=True, ep=True)
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, rank + 0.5)
        optimizer.step()
        ema.step(1)
        states = {
            "model": ModelCheckpointState(model),
            "optimizer": optimizer,
            "ema": ema,
        }
        expected = {
            category: {
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in state.state_dict().items()
            }
            for category, state in states.items()
        }
        assert all(value.shape[0] == 4 for value in expected["model"].values())
        dcp.save(states, checkpoint_id=folder)
        for wrapped, ep in ((True, True), (False, True), (False, False)):
            restored_model, restored_optimizer, restored_ema = build(
                rank, wrapped=wrapped, ep=ep
            )
            restored = {
                "model": ModelCheckpointState(restored_model),
                "optimizer": restored_optimizer,
                "ema": restored_ema,
            }
            dcp.load(restored, checkpoint_id=folder)
            for category, state in restored.items():
                actual = state.state_dict()
                assert actual.keys() == expected[category].keys()
                for key, value in actual.items():
                    target = expected[category][key]
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(value, target, rtol=0, atol=0)
                    else:
                        assert value == target
            if rank == 0:
                print(
                    f"EP DCP restore wrapped={wrapped}, ep={2 if ep else 1}: exact",
                    flush=True,
                )
        dist.barrier()
    finally:
        if folder is not None and dist.get_rank() == 0:
            shutil.rmtree(folder)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
