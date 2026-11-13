"""Runtime metrics consume explicit services and scalar training state."""

from types import SimpleNamespace

import torch

from llmtuner.trainer.runtime.metrics import (
    build_step_metrics,
    collect_resource_metrics,
)
from llmtuner.trainer.runtime.reductions import TrainingReductionMeshes
from llmtuner.trainer.runtime.training_observer import DispatchTimer


def test_step_metrics_need_no_trainer_or_engine():
    result = build_step_metrics(
        ntokens_seen=17,
        device=torch.device("cpu"),
        parallel_dims=None,
        model_parts=[],
        pp_has_last_stage=False,
        accumulated_loss=torch.tensor(9.0),
        global_valid_tokens=torch.tensor(3),
        local_valid_tokens=3,
        local_valid_tokens_tensor=torch.tensor(3),
        reduction_meshes=TrainingReductionMeshes(None, None, None, None),
        grad_norm=torch.tensor(2.0),
        lr_metrics={"lr/AdamW": 0.01},
    )
    assert result == {
        "loss": 3.0,
        "max_loss": 3.0,
        "grad_norm": 2.0,
        "n_tokens_seen": 17.0,
        "lr/AdamW": 0.01,
    }


def test_resource_metrics_consume_only_injected_timing_window():
    timer = DispatchTimer()
    timer.cpu_seconds = 0.25
    checkpoint_metrics = {"checkpoint/write_seconds": 0.5}
    options = {
        "activation_swap_enabled": False,
        "optimizer": SimpleNamespace(optimizers=[]),
        "checkpointer": SimpleNamespace(observation_metrics=checkpoint_metrics),
        "dispatch_timers": [timer],
    }

    first = collect_resource_metrics(**options)
    second = collect_resource_metrics(**options)

    assert first["moe/local_forward_dispatch_seconds_window"] == 0.25
    assert second["moe/local_forward_dispatch_seconds_window"] == 0.0
    assert first["checkpoint/write_seconds"] == 0.5
    first["checkpoint/write_seconds"] = 9
    assert checkpoint_metrics == {"checkpoint/write_seconds": 0.5}
