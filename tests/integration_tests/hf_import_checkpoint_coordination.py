"""Two NPU ranks coordinate HF adaptation and destination validation failures.

Run with torchrun --nproc-per-node=2 -m
tests.integration_tests.hf_import_checkpoint_coordination.
Uses real safetensors reads and DCP load; only injects rank-local adapter faults.
"""

import json
import shutil
from pathlib import Path
from tempfile import mkdtemp
from types import SimpleNamespace

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from llmtuner.components.checkpointer.checkpoint_keys import MODEL
from llmtuner.components.checkpointer.distributed_checkpoint import (
    DistributedCheckpointManager,
)
from llmtuner.config import CheckpointConfig
from llmtuner.models.hf.state_dict_adapter import HFTransformerStateDictAdapter
from tests.integration_tests.harness import init_distributed


def main():
    device = init_distributed()
    directory = None
    rank = None
    try:
        assert device.type == "npu" and dist.get_world_size() == 2
        rank = dist.get_rank()
        roots = [
            mkdtemp(prefix="llmtuner-hf-load-coordination-") if rank == 0 else None
        ]
        dist.broadcast_object_list(roots, src=0)
        directory = Path(roots[0])
        source = directory / "source"
        if rank == 0:
            source.mkdir()
            weights = {
                "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
                "bias": torch.tensor([0.25, -0.5]),
            }
            save_file(weights, str(source / "model.safetensors"))
            (source / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {},
                        "weight_map": {
                            "weight": "model.safetensors",
                            "bias": "model.safetensors",
                        },
                    }
                ),
                encoding="utf-8",
            )
        dist.barrier()

        model = torch.nn.Module()
        model.model = torch.nn.Linear(3, 2, device=device)
        with torch.no_grad():
            model.model.weight.zero_()
            model.model.bias.zero_()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        adapter = HFTransformerStateDictAdapter(
            SimpleNamespace(model_type="llama", tie_word_embeddings=False),
            str(source),
        )
        manager = DistributedCheckpointManager(
            CheckpointConfig(
                enable=True,
                interval=1,
                keep_latest_k=0,
                initial_load_path=str(source),
                initial_load_in_hf=True,
            ),
            model_parts=[model],
            optimizer=optimizer,
            lr_scheduler=scheduler,
            states={},
            folder=str(directory / "checkpoints"),
            sd_adapter=adapter,
        )
        original = adapter.from_hf
        original_reader = adapter.get_hf_storage_reader
        initial = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        expected = {
            "model.weight": torch.arange(6, device=device, dtype=torch.float32).reshape(
                2, 3
            ),
            "model.bias": torch.tensor([0.25, -0.5], device=device),
        }
        try:
            scenarios = (
                ("reader", 1),
                ("adapter", 0),
                ("destination", 1),
            )
            for failure, failing_rank in scenarios:

                def injected(state, failure=failure, failing_rank=failing_rank):
                    result = original(state)
                    if rank == failing_rank:
                        if failure == "adapter":
                            raise ValueError("injected rank-local HF adapter failure")
                        result["model.weight"] = result["model.weight"][:1]
                    return result

                adapter.from_hf = injected
                if failure == "reader":

                    def injected_reader(path, failing_rank=failing_rank, **kwargs):
                        if rank == failing_rank:
                            raise ValueError("injected rank-local HF reader failure")
                        return original_reader(path, **kwargs)

                    adapter.get_hf_storage_reader = injected_reader
                error = None
                try:
                    manager._load_checkpoint(
                        {MODEL: manager.states[MODEL]},
                        str(source),
                        from_hf=True,
                        from_quantized=False,
                    )
                except (ValueError, RuntimeError) as caught:
                    error = caught
                outcomes = [None, None]
                dist.all_gather_object(outcomes, (type(error).__name__, str(error)))
                local_phase = {
                    "reader": "HF checkpoint reader preparation",
                    "adapter": "HF checkpoint state adaptation",
                    "destination": "HF checkpoint model validation",
                }[failure]
                assert outcomes[failing_rank][0] == "ValueError", outcomes
                expected_error = {
                    "reader": "injected rank-local HF reader failure",
                    "adapter": "injected rank-local HF adapter failure",
                    "destination": "shape",
                }[failure]
                assert expected_error in outcomes[failing_rank][1], outcomes
                peer = outcomes[1 - failing_rank]
                assert peer[0] == "RuntimeError", outcomes
                assert f"{local_phase} failed on another rank" in peer[1], outcomes
                # The HF reader writes live tensors only after reader
                # preparation. Later adapter/shape failures do not roll back.
                failed_state = initial if failure == "reader" else expected
                torch.testing.assert_close(
                    model.state_dict(), failed_state, rtol=0, atol=0
                )
                adapter.from_hf = original
                adapter.get_hf_storage_reader = original_reader
                manager._load_checkpoint(
                    {MODEL: manager.states[MODEL]},
                    str(source),
                    from_hf=True,
                    from_quantized=False,
                )
                torch.testing.assert_close(model.state_dict(), expected, rtol=0, atol=0)
                print(f"PASS HF {failure} failure/retry rank={rank}", flush=True)
        finally:
            adapter.from_hf = original
            adapter.get_hf_storage_reader = original_reader
            manager.close()
        dist.barrier()
    finally:
        dist.destroy_process_group()
        if directory is not None and rank == 0:
            shutil.rmtree(directory)


if __name__ == "__main__":
    main()
