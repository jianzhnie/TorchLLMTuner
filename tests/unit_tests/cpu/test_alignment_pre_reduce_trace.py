from types import SimpleNamespace

import pytest
import torch

from tests.integration_tests.alignment_pre_reduce_trace import pre_reduce_target_records


def test_actual_reduce_input_target_views_preserve_values_and_own_storage():
    gate = torch.randn(128, 4, 3)
    up = torch.randn_like(gate)
    targets = [["layers.2.experts.gate", 7, 1, 4], ["layers.2.experts.up", 96, 2, 4]]
    baseline = pre_reduce_target_records(
        [
            SimpleNamespace(
                _param_fqn="layers.2.moe.routed_experts.inner_experts.w1_EFD"
            ),
            SimpleNamespace(
                _param_fqn="layers.2.moe.routed_experts.inner_experts.w3_EFD"
            ),
        ],
        [gate, up],
        targets,
    )
    packed = torch.cat((gate, up), dim=1)
    candidate = pre_reduce_target_records(
        [SimpleNamespace(_param_fqn="model.model.layers.2.mlp.experts.gate_up_proj")],
        [packed],
        targets,
    )
    assert baseline.keys() == candidate.keys()
    for key in baseline:
        torch.testing.assert_close(
            baseline[key]["gradient"], candidate[key]["gradient"], atol=0, rtol=0
        )
        assert candidate[key]["gradient_dtype"] == "torch.float32"
    packed.zero_()
    key = "layers.2.experts.up/expert96/rows2:4"
    torch.testing.assert_close(candidate[key]["gradient"], up[96, 2:4])


def test_pre_reduce_rejects_already_sharded_expert_gradients():
    with pytest.raises(ValueError, match="full DP8"):
        pre_reduce_target_records(
            [
                SimpleNamespace(
                    _param_fqn="layers.2.moe.routed_experts.inner_experts.w1_EFD"
                )
            ],
            [torch.zeros(16, 4, 3)],
            [],
        )


def test_reduce_observer_passes_original_objects_and_restores(monkeypatch, tmp_path):
    import json

    from torch.distributed.fsdp._fully_shard import _fsdp_param_group as owner

    from tests.integration_tests.alignment_pre_reduce_trace import (
        install_pre_reduce_trace,
    )

    calls = []
    sentinel = object()

    def original(fsdp_params, unsharded_grads, reduce_dtype=None):
        calls.append((fsdp_params, unsharded_grads, reduce_dtype))
        return sentinel

    monkeypatch.setattr(owner, "foreach_reduce", original)
    target = tmp_path / "targets.json"
    target.write_text(json.dumps([["layers.2.experts.gate", 7, 1, 3]]))
    trainer = SimpleNamespace(step=1)
    restore = install_pre_reduce_trace(trainer, tmp_path, target)
    parameters = [
        SimpleNamespace(_param_fqn="layers.2.moe.routed_experts.inner_experts.w1_EFD")
    ]
    gradients = [torch.randn(128, 4, 3)]
    try:
        assert owner.foreach_reduce(parameters, gradients, torch.float32) is sentinel
        assert calls[0][0] is parameters and calls[0][1] is gradients
        record = torch.load(tmp_path / "pre-reduce-step-1.pt", weights_only=True)
        row = record["layers.2.experts.gate/expert7/rows1:3"]
        torch.testing.assert_close(row["gradient"], gradients[0][7, 1:3])
        assert row["reduction_configuration"]["reduce_dtype"] == "torch.float32"
    finally:
        restore()
    assert owner.foreach_reduce is original
