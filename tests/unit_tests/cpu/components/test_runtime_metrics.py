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


def test_step_reporting_preserves_accumulation_and_normalization(monkeypatch):
    import torch

    from llmtuner.trainer.runtime import metrics
    from llmtuner.trainer.runtime.reductions import TrainingReductionMeshes
    from llmtuner.trainer.runtime.results import AccumulationResult, UpdateResult

    for sequence_sharded in (False, True):
        groups = {name: object() for name in ("dp", "loss", "sequence")}
        calls = []

        def reduce(tensor, *, group, op="sum", calls=calls):
            calls.append((group, op))
            if op == "sum":
                tensor.mul_(2)

        monkeypatch.setattr(metrics, "all_reduce", reduce)
        meshes = {
            name: SimpleNamespace(
                get_group=lambda name=name, groups=groups: groups[name]
            )
            for name in groups
        }
        loss = torch.tensor(12.0)
        denominator = torch.tensor(8)
        update = UpdateResult(1, torch.tensor(3.0), True, True, False)
        result = metrics.build_step_metrics(
            ntokens_seen=16,
            device=torch.device("cpu"),
            accumulation=AccumulationResult(loss, torch.tensor(1, dtype=torch.int32)),
            global_valid_tokens=denominator,
            local_valid_tokens=4,
            reduction_meshes=TrainingReductionMeshes(
                meshes["dp"],
                meshes["loss"],
                meshes["sequence"] if sequence_sharded else None,
            ),
            update=update,
            lr_metrics={"lr": 0.01},
        )
        assert result == {
            "loss": 3.0,
            "max_loss": 6.0 if sequence_sharded else 3.0,
            "n_tokens_seen": 32.0,
            "grad_norm": 3.0,
            "lr": 0.01,
        }
        assert loss.item() == 12.0
        assert denominator.item() == 8
        assert update.grad_norm.item() == 3.0
        expected = [(groups["sequence"], "sum")] if sequence_sharded else []
        assert calls == expected + [
            (groups["loss"], "sum"),
            (groups["dp"], "max"),
            (groups["loss"], "sum"),
        ]
