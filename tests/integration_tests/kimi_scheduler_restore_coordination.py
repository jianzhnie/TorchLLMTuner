"""Two-rank Kimi layout migration coordinates local scheduler load failures.

Run with PYTHONPATH=. torchrun --nproc_per_node=2 on CPU/Gloo.
"""

from contextlib import nullcontext
from datetime import timedelta
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from llmtuner.components.checkpointer.kimi_layout import load_changed_kimi_layout
from llmtuner.components.checkpointer.model_state import ModelCheckpointState
from llmtuner.components.checkpointer.random_state import RNGCheckpointState
from llmtuner.models.hf.state_dict_adapter import HFTransformerStateDictAdapter
from llmtuner.parallel.expert_parallel.model_replacement import swap_hf_moe_blocks
from tests.unit_tests.cpu.parallel.test_kimi_conversion import make_model


class SchedulerState:
    def __init__(self, fail=False):
        self.fail = fail
        self.step = 0

    def state_dict(self):
        return {"step": self.step}

    def load_state_dict(self, state):
        if self.fail:
            raise ValueError("injected scheduler failure")
        self.step = state["step"]


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=10))
    try:
        rank = dist.get_rank()
        assert dist.get_world_size() == 2
        context = TemporaryDirectory() if rank == 0 else nullcontext(None)
        with context as root:
            paths = [root]
            dist.broadcast_object_list(paths, src=0)
            for source_grouped in (False, True):
                torch.manual_seed(11)
                source = torch.nn.Module()
                source.model = make_model(latent=True, shared=1)
                config = source.model.layers[0].block_sparse_moe.config
                if source_grouped:
                    swap_hf_moe_blocks(source.model)
                saved = dict(source.state_dict())
                saved["rng_state"] = RNGCheckpointState(torch.device("cpu"))
                saved["lr_scheduler"] = SchedulerState()
                folder = f"{paths[0]}/{source_grouped}"
                dcp.save(saved, checkpoint_id=folder)
                target = torch.nn.Module()
                target.model = make_model(latent=True, shared=1)
                if not source_grouped:
                    swap_hf_moe_blocks(target.model)
                scheduler = SchedulerState(fail=rank == 0)
                states = {
                    "model": ModelCheckpointState(target),
                    "lr_scheduler": scheduler,
                    "rng_state": RNGCheckpointState(torch.device("cpu")),
                }
                adapter = HFTransformerStateDictAdapter(
                    config, None, kimi_expert_parallel=not source_grouped
                )
                try:
                    load_changed_kimi_layout(folder, states=states, adapter=adapter)
                except (ValueError, RuntimeError) as error:
                    expected = (
                        "injected scheduler failure"
                        if rank == 0
                        else "Kimi checkpoint lr_scheduler restoration failed "
                        "on another rank"
                    )
                    assert str(error) == expected, str(error)
                else:
                    raise AssertionError("scheduler failure did not propagate")
                dist.barrier()
                scheduler.fail = False
                assert load_changed_kimi_layout(folder, states=states, adapter=adapter)
                dist.barrier()
                if rank == 0:
                    print(
                        f"source_grouped={source_grouped}: "
                        "scheduler failure coordinated"
                    )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
