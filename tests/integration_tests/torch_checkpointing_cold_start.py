"""Exercise optional backend EMA cold-start coordination with real Gloo ranks.

PYTHONPATH=. torchrun --nproc_per_node=2 \
    tests/integration_tests/torch_checkpointing_cold_start.py

The backend reader is substituted; this checks llmtuner's post-load control
flow only, not torch_checkpointing I/O or its distributed protocol.
"""

import pickle
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from llmtuner.components.checkpointer import torch_checkpointing_backend as tc
from llmtuner.components.checkpointer.checkpoint_keys import (
    DATALOADER,
    EMA,
    LR_SCHEDULER,
    MODEL,
    RNG_STATE,
    TRAIN_STATE,
)
from llmtuner.components.checkpointer.random_state import RNGCheckpointState
from llmtuner.components.checkpointer.torch_checkpointing_backend import (
    TorchCheckpointingManager,
)
from llmtuner.components.optimizer.ema import EMA as MovingAverage


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    try:
        assert dist.get_world_size() == 2
        model = torch.nn.Linear(2, 2)
        ema = MovingAverage(model_parts=[model], decay=0.75)
        manager = TorchCheckpointingManager.__new__(TorchCheckpointingManager)
        manager._initialized = False
        manager._backend = SimpleNamespace(
            HuggingFaceSafetensorsDistributedMetadataFormat=None
        )
        manager._manager = SimpleNamespace(load=lambda path, *, into, strict: into)
        manager._manager_config = SimpleNamespace(storage_config=None)
        manager._is_resumable_checkpoint = lambda path: True
        manager.states = {MODEL: model, EMA: ema}
        for key in (LR_SCHEDULER, DATALOADER, TRAIN_STATE, RNG_STATE):
            failure = ValueError(f"rank-local {key} restore failed")

            class LocalState:
                def state_dict(self):
                    return {}

                def load_state_dict(self, state, failure=failure):
                    if dist.get_rank() == 0:
                        raise failure

            caught = None
            try:
                manager._load_checkpoint(
                    {key: LocalState()}, "unused", from_hf=False, from_quantized=False
                )
            except (ValueError, RuntimeError) as error:
                caught = error
            assert caught is not None
            if dist.get_rank() == 0:
                assert caught is failure
            else:
                assert f"Checkpoint {key} restoration" in str(caught)
            dist.barrier()

        rng = RNGCheckpointState(torch.device("cpu"))
        saved_rng = rng.state_dict()
        if dist.get_rank() == 0:
            payload = pickle.loads(saved_rng[rng.key])
            payload["version"] = 2
            saved_rng[rng.key] = pickle.dumps(payload)
        torch.rand(5)
        before_rng = torch.get_rng_state().clone()
        with (
            patch.object(rng, "state_dict", return_value=saved_rng),
            patch.object(tc, "sync_save_config", return_value=None),
            patch.object(tc, "default_backend_config", return_value=None),
            patch.object(
                tc,
                "temporary_backend_manager",
                side_effect=lambda config: nullcontext(manager._manager),
            ),
        ):
            caught = None
            try:
                manager._load_checkpoint(
                    {RNG_STATE: rng}, "unused", from_hf=False, from_quantized=False
                )
            except (ValueError, RuntimeError) as error:
                caught = error
            assert caught is not None
            assert torch.equal(torch.get_rng_state(), before_rng), (
                "valid peer rewound RNG before another rank rejected its payload"
            )
        dist.barrier()

        original_reset = ema.reset_from_model
        original_error = ValueError("rank-local EMA reseed failure")
        calls = []

        def reset():
            calls.append("reset")
            if dist.get_rank() == 0:
                raise original_error
            original_reset()

        # The old public load path calls reset too; the regression must expose
        # missing peer failure propagation rather than merely an API change.
        with patch.object(ema, "reset_from_model", reset):
            caught = None
            try:
                manager._load_checkpoint(
                    {MODEL: model}, "unused", from_hf=False, from_quantized=False
                )
            except (ValueError, RuntimeError) as error:
                caught = error
            assert caught is not None, "peer returned successfully after reseed failure"
            if dist.get_rank() == 0:
                assert caught is original_error
            else:
                assert "Checkpoint EMA cold start" in str(caught)
            assert calls == ["reset"]
        dist.barrier()

        # A local reseed failure is not atomic. A full load/reseed retry must
        # still work on every rank once the cause has been removed.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(3)
        manager._load_checkpoint(
            {MODEL: model}, "unused", from_hf=False, from_quantized=False
        )
        for averages, parameters in ema._tracked_groups():
            for average, parameter in zip(averages, parameters, strict=True):
                torch.testing.assert_close(average, parameter, atol=0, rtol=0)
        dist.barrier()
        if dist.get_rank() == 0:
            print("Optional backend EMA reseed failure coordinated; full retry passed.")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
