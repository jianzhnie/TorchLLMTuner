"""Reject rank-local HF source index corruption during trainer assembly.

PYTHONPATH=. torchrun --nproc_per_node=2 \
    tests/integration_tests/checkpoint_adapter_construction.py

CPU/Gloo, actual assembly helper, adapter file IO and DCP manager construction.
Uses a minimal trainer owner rather than the full model/data assembly pipeline.
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
from llmtuner.trainer.runtime.assembly import _build_checkpointer
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
                    trainer = SimpleNamespace(
                        cfg=SimpleNamespace(
                            checkpoint=CheckpointConfig(enable=True, keep_latest_k=0),
                            dump_folder=f"{roots[0]}/out-{failing_rank}-{malformed}",
                            hf_model=str(assets),
                            parallel=SimpleNamespace(ep_size=1),
                        ),
                        dataloader=loader(2),
                        device=torch.device("cpu"),
                        parallel_dims=None,
                        model_parts=[model],
                        optimizer=optimizer,
                        lr_scheduler=scheduler,
                        ema=ema,
                    )
                    caught = None
                    try:
                        _build_checkpointer(
                            trainer, SimpleNamespace(model_type="llama")
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
                    assert not hasattr(trainer, "checkpointer")
                    dist.barrier()
                    index.write_text(
                        json.dumps({"weight_map": {"weight": "weights.safetensors"}})
                    )
                    _build_checkpointer(trainer, SimpleNamespace(model_type="llama"))
                    try:
                        assert trainer.checkpointer._initialized
                    finally:
                        trainer.checkpointer.close()
                    dist.barrier()
                    if rank == 0:
                        print(f"PASS {failing_rank=} {malformed=}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
