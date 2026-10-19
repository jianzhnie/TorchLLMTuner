"""HF exports survive distributed consolidation/publication failures.

PYTHONPATH=. torchrun --nproc_per_node=2 \
    tests/integration_tests/hf_checkpoint_equivalence.py

CPU/Gloo, native/custom FSDP, one/two output shards. Every case verifies a
fresh DTensor model loads the previously published weights after one rank
fails, and subsequently loads a successful rewrite exactly. This exercises
local storage and CPU tensor I/O, not CUDA/NPU or remote publication.
"""

from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn as nn
from fsdp_checkpoint_equivalence import topology_for
from fsdp_mesh_equivalence import _build_model, _full
from torch.distributed.checkpoint.api import CheckpointException

from llmtuner.components.checkpointer import hf_storage
from llmtuner.components.checkpointer.distributed_checkpoint import (
    DistributedCheckpointManager,
)
from llmtuner.components.optimizer.lr_scheduler import build_lr_scheduler
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config.checkpoint_config import CheckpointConfig
from llmtuner.config.lr_scheduler_config import LRSchedulerConfig
from llmtuner.config.optimizer_config import OptimizerConfig
from llmtuner.models.hf.state_dict_adapter import HFTransformerStateDictAdapter
from llmtuner.parallel.fully_shard_parallel.fully_shard_parallel import apply_fsdp


def runtime(cfg, topology, seed):
    model = nn.ModuleDict({"model": apply_fsdp(_build_model(seed), cfg, topology)})
    optimizer = OptimizersContainer(OptimizerConfig(), model_parts=[model])
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(), optimizers=optimizer.optimizers, training_steps=5
    )
    return model, optimizer, scheduler


def manager_for(runtime_parts, config, folder, adapter):
    model, optimizer, scheduler = runtime_parts
    return DistributedCheckpointManager(
        config,
        model_parts=[model],
        optimizer=optimizer,
        lr_scheduler=scheduler,
        states={},
        folder=str(folder),
        sd_adapter=adapter,
    )


def full_state(model):
    return {name: _full(value).clone() for name, value in model.state_dict().items()}


def check_case(root, implementation, indexed, failure_phase):
    cfg, topology = topology_for("shard", implementation)
    parts = runtime(cfg, topology, 0)
    adapter = HFTransformerStateDictAdapter(SimpleNamespace(), None)
    if indexed:
        adapter.fqn_to_index_mapping = {
            name.removeprefix("model."): index % 2 + 1
            for index, name in enumerate(sorted(parts[0].state_dict()))
        }
    folder = Path(root) / f"{implementation}_{indexed}_{failure_phase}"
    manager = manager_for(
        parts,
        CheckpointConfig(
            enable=True, interval=1, last_save_in_hf=True, keep_latest_k=0
        ),
        folder,
        adapter,
    )
    try:
        expected = full_state(parts[0])
        assert manager.save(1)
        assert manager.save(1, last_step=True)
        export = folder / "checkpoint" / "step-1" / "model-export"
        assert manager._is_resumable_checkpoint(str(export.parent))
        with torch.no_grad():
            for parameter in parts[0].parameters():
                parameter.add_(5)
        consolidate = hf_storage.hf_consolidation._consolidate_safetensors_files

        def fail_consolidation(*args, **kwargs):
            result = consolidate(*args, **kwargs)
            if dist.get_rank() == 1:
                raise OSError("injected HF consolidation failure on rank 1")
            return result

        def fail_publication(*args, **kwargs):
            raise OSError("injected HF publication failure on rank 0")

        target = (
            patch.object(
                hf_storage.hf_consolidation,
                "_consolidate_safetensors_files",
                fail_consolidation,
            )
            if failure_phase == "consolidation"
            else patch.object(hf_storage.os, "replace", fail_publication)
        )
        with target:
            try:
                manager.save(1, last_step=True)
            except CheckpointException as error:
                assert "injected HF" in str(error)
            else:
                raise AssertionError("Export failure must reach every rank")
        # Start with distinct parameters and a new manager; export loads are
        # explicitly model-only and must not become automatic training resumes.
        restored = runtime(cfg, topology, 123)
        loader = manager_for(
            restored,
            CheckpointConfig(
                enable=True,
                initial_load_path=str(export),
                initial_load_in_hf=True,
                keep_latest_k=0,
            ),
            folder / "imported",
            adapter,
        )
        try:
            assert loader.load(-1)
            torch.testing.assert_close(
                full_state(restored[0]), expected, rtol=0, atol=0
            )
            rewritten = full_state(parts[0])
            assert manager.save(1, last_step=True)
            assert loader.load(-1)
            torch.testing.assert_close(
                full_state(restored[0]), rewritten, rtol=0, atol=0
            )
        finally:
            loader.close()
        if dist.get_rank() == 0:
            print(
                f"{implementation}/indexed={indexed}/{failure_phase}: exact HF restore"
            )
    finally:
        manager.close()


def main():
    dist.init_process_group("gloo")
    try:
        assert dist.get_world_size() == 2
        context = TemporaryDirectory() if dist.get_rank() == 0 else nullcontext(None)
        with context as root:
            paths = [root]
            dist.broadcast_object_list(paths, src=0)
            for implementation in ("native", "custom"):
                for indexed in (False, True):
                    for failure_phase in ("consolidation", "publication"):
                        check_case(paths[0], implementation, indexed, failure_phase)
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
