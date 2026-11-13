"""Reject rank-local HF source index corruption during trainer assembly.

PYTHONPATH=. torchrun --nproc_per_node=2 \
    tests/integration_tests/checkpoint_adapter_construction.py

CPU/Gloo, actual assembly helper, adapter file IO and DCP manager construction.
Passes explicit runtime owners without constructing a Trainer.
"""

import json
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
import torch.distributed as dist

from llmtuner.config.checkpoint_config import CheckpointConfig
from llmtuner.config.optimizer_config import OptimizerConfig
from llmtuner.trainer.runtime.assembly import (
    ModelAssembly,
    OptimizerStack,
    build_checkpointer,
)
from tests.integration_tests.checkpoint_data_preflight import loader
from tests.integration_tests.rng_checkpoint_equivalence import runtime


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=10))
    try:
        rank = dist.get_rank()
        assert dist.get_world_size() == 2
        with TemporaryDirectory() if rank == 0 else nullcontext(None) as root:
            roots = [root]
            dist.broadcast_object_list(roots, src=0)
            for failing_rank in (0, 1):
                for malformed in ("json", "weight_map"):
                    assets = Path(roots[0]) / f"{failing_rank}-{malformed}-{rank}"
                    assets.mkdir()
                    index = assets / "model.safetensors.index.json"
                    index.write_text(
                        "{"
                        if rank == failing_rank and malformed == "json"
                        else json.dumps(
                            {
                                "weight_map": {}
                                if rank == failing_rank
                                else {"weight": "weights.safetensors"}
                            }
                        )
                    )
                    model, _, optimizer, scheduler, ema = runtime(42)
                    cfg = SimpleNamespace(
                        checkpoint=CheckpointConfig(enable=True, keep_latest_k=0),
                        optimizer=OptimizerConfig(),
                        dump_folder=f"{roots[0]}/out-{failing_rank}-{malformed}",
                        hf_model=str(assets),
                        parallel=SimpleNamespace(ep_size=1),
                    )
                    dependencies = dict(
                        model_assembly=ModelAssembly(model=model, model_parts=[model]),
                        optimizer_stack=OptimizerStack(optimizer, scheduler, ema),
                        dataloader=loader(2),
                        train_state=None,
                        device=torch.device("cpu"),
                        parallel_dims=None,
                    )
                    caught = None
                    try:
                        build_checkpointer(
                            cfg, SimpleNamespace(model_type="llama"), **dependencies
                        )
                    except BaseException as error:
                        caught = error
                    if rank == failing_rank:
                        assert isinstance(caught, ValueError), caught
                        if malformed == "json":
                            assert isinstance(caught, json.JSONDecodeError)
                        else:
                            assert "non-empty weight_map" in str(caught)
                    else:
                        assert isinstance(caught, RuntimeError), caught
                        assert str(caught) == (
                            "Checkpoint services preparation failed on another rank"
                        )
                    dist.barrier()
                    index.write_text(
                        json.dumps({"weight_map": {"weight": "weights.safetensors"}})
                    )
                    checkpointer = build_checkpointer(
                        cfg, SimpleNamespace(model_type="llama"), **dependencies
                    )
                    try:
                        assert checkpointer._initialized
                    finally:
                        checkpointer.close()
                    dist.barrier()
                    if rank == 0:
                        print(f"PASS {failing_rank=} {malformed=}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
