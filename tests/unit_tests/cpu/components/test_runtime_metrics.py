"""Runtime metrics consume explicit services and scalar training state."""

from types import SimpleNamespace

from llmtuner.trainer.runtime.metrics import (
    collect_resource_metrics,
)
from llmtuner.trainer.runtime.training_observer import DispatchTimer


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
