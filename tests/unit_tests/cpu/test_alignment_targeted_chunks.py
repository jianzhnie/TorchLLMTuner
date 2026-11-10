import json
from types import SimpleNamespace

import torch

from tests.integration_tests.alignment_targeted_chunks import targeted_expert_records


def test_targeted_packed_chunks_capture_before_after_gradient_moments_and_update(
    tmp_path,
):
    parameter = torch.nn.Parameter(torch.arange(48.0).reshape(2, 8, 3) / 100)
    parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.AdamW([parameter], lr=0.01, foreach=False, fused=False)
    trainer = SimpleNamespace(optimizer=optimizer)
    names = {id(parameter): "part0.model.model.layers.2.mlp.experts.gate_up_proj"}
    path = tmp_path / "targets.json"
    path.write_text(json.dumps([["layers.2.experts.up", 1, 2, 4]]))
    first = targeted_expert_records([parameter], names, trainer, path)
    key = "layers.2.experts.up/expert1/rows2:4"
    assert set(first) == {key}
    assert not first[key]["optimizer_state_present"]
    torch.testing.assert_close(first[key]["parameter"], parameter[1, 6:8])
    torch.nn.utils.clip_grad_norm_([parameter], 0.5)
    clipped = targeted_expert_records([parameter], names, trainer, path)
    torch.testing.assert_close(clipped[key]["gradient"], parameter.grad[1, 6:8])
    optimizer.step()
    updated = targeted_expert_records([parameter], names, trainer, path)
    assert updated[key]["optimizer_step"].item() == 1
    for field in ("exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            updated[key][field], optimizer.state[parameter][field][1, 6:8]
        )
    assert not torch.equal(first[key]["parameter"], updated[key]["parameter"])
    original = first[key]["parameter"].clone()
    with torch.no_grad():
        parameter.zero_()
    torch.testing.assert_close(first[key]["parameter"], original)


def test_targeted_split_experts_match_packed_logical_rows(tmp_path):
    gate = torch.nn.Parameter(torch.randn(2, 4, 3))
    up = torch.nn.Parameter(torch.randn_like(gate))
    packed = torch.nn.Parameter(torch.cat((gate, up), dim=1))
    targets = [["layers.2.experts.gate", 0, 1, 3], ["layers.2.experts.up", 1, 0, 4]]
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(targets))
    trainer = SimpleNamespace(optimizer=None)
    baseline = targeted_expert_records(
        [gate, up],
        {
            id(gate): "part0.layers.2.moe.routed_experts.inner_experts.w1_EFD",
            id(up): "part0.layers.2.moe.routed_experts.inner_experts.w3_EFD",
        },
        trainer,
        path,
    )
    candidate = targeted_expert_records(
        [packed],
        {id(packed): "part0.model.model.layers.2.mlp.experts.gate_up_proj"},
        trainer,
        path,
    )
    assert baseline.keys() == candidate.keys()
    for key in baseline:
        torch.testing.assert_close(
            baseline[key]["parameter"], candidate[key]["parameter"], atol=0, rtol=0
        )
