"""Runtime metrics consume explicit services and scalar training state."""

from types import SimpleNamespace

import pytest
import torch

from llmtuner.trainer.runtime.metrics import (
    build_step_metrics,
    collect_resource_metrics,
)
from llmtuner.trainer.runtime.reductions import TrainingReductionMeshes
from llmtuner.trainer.runtime.results import AccumulationResult, UpdateResult
from llmtuner.trainer.runtime.training_observer import DispatchTimer


def test_step_metrics_need_no_trainer_or_engine():
    result = build_step_metrics(
        ntokens_seen=17,
        device=torch.device("cpu"),
        pp_has_last_stage=False,
        accumulation=AccumulationResult(
            torch.tensor(9.0), torch.ones((), dtype=torch.int32)
        ),
        global_valid_tokens=torch.tensor(3),
        local_valid_tokens=3,
        local_valid_tokens_tensor=torch.tensor(3),
        reduction_meshes=TrainingReductionMeshes(None, None, None, None),
        update=UpdateResult(1, torch.tensor(2.0), True, True, False),
        lr_metrics={"lr/AdamW": 0.01},
    )
    assert result == {
        "loss": 3.0,
        "max_loss": 3.0,
        "grad_norm": 2.0,
        "n_tokens_seen": 17.0,
        "lr/AdamW": 0.01,
    }


def test_reporting_rejects_nonlogging_accumulation_result():
    with pytest.raises(ValueError, match="accumulation result with loss_sum"):
        build_step_metrics(
            ntokens_seen=17,
            device=torch.device("cpu"),
            pp_has_last_stage=False,
            accumulation=AccumulationResult(None, torch.ones((), dtype=torch.int32)),
            global_valid_tokens=torch.tensor(3),
            local_valid_tokens=3,
            local_valid_tokens_tensor=torch.tensor(3),
            reduction_meshes=TrainingReductionMeshes(None, None, None, None),
            update=UpdateResult(1, None, True, True, False),
            lr_metrics={},
        )


def test_resource_metrics_consume_only_injected_timing_window():
    timer = DispatchTimer()
    timer.cpu_seconds = 0.25
    checkpoint_metrics = {"checkpoint/write_seconds": 0.5}
    optimizer_metrics = {"optimizer/local_transfer_bytes": 64.0}
    options = {
        "activation_swap": None,
        "optimizer": SimpleNamespace(observation_metrics=optimizer_metrics),
        "checkpointer": SimpleNamespace(observation_metrics=checkpoint_metrics),
        "dispatch_timers": [timer],
    }

    first = collect_resource_metrics(**options)
    second = collect_resource_metrics(**options)

    assert first["moe/local_forward_dispatch_seconds_window"] == 0.25
    assert second["moe/local_forward_dispatch_seconds_window"] == 0.0
    assert first["checkpoint/write_seconds"] == 0.5
    assert first["optimizer/local_transfer_bytes"] == 64.0
    first["optimizer/local_transfer_bytes"] = 0.0
    assert optimizer_metrics == {"optimizer/local_transfer_bytes": 64.0}
    first["checkpoint/write_seconds"] = 9
    assert checkpoint_metrics == {"checkpoint/write_seconds": 0.5}
