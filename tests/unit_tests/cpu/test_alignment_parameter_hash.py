import pytest
import torch

from tests.integration_tests.alignment_parameter_hash import parameter_chunk_hashes


def chunk_map(records):
    return {
        (c["canonical"], c["expert"], c["row_start"], c["row_end"]): c["sha256"]
        for record in records
        for c in record["chunks"]
    }


def test_packed_experts_hash_all_noncontiguous_halves_in_canonical_order():
    torch.manual_seed(4)
    gate = torch.randn(3, 7, 5, dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    packed = torch.cat((gate, up), dim=1)
    baseline = [
        parameter_chunk_hashes(
            "part0.layers.4._checkpoint_wrapped_module.moe.routed_experts.inner_experts."
            + role,
            tensor,
            chunk_elements=10,
        )
        for role, tensor in (("w1_EFD", gate), ("w3_EFD", up))
    ]
    actual = parameter_chunk_hashes(
        "part0.model.model.layers.4._checkpoint_wrapped_module.mlp.experts.gate_up_proj",
        packed,
        chunk_elements=10,
    )
    assert chunk_map(baseline) == chunk_map([actual])
    packed[2, 13, 4] += 1
    changed = parameter_chunk_hashes(
        "model.model.layers.4.mlp.experts.gate_up_proj", packed, chunk_elements=10
    )
    differences = [
        key
        for key, value in chunk_map([actual]).items()
        if chunk_map([changed])[key] != value
    ]
    assert differences == [("layers.4.experts.up", 2, 6, 7)]


@pytest.mark.parametrize(
    "source,target",
    [
        ("attention.wo", "self_attn.o_proj"),
        ("moe.router.gate", "mlp.gate"),
        ("attention_norm", "input_layernorm"),
        ("ffn_norm", "post_attention_layernorm"),
    ],
)
def test_regular_parameter_canonical_hashes_cover_full_tensor(source, target):
    value = torch.arange(35, dtype=torch.float32).reshape(7, 5)
    first = parameter_chunk_hashes(
        f"part0.layers.1.{source}.weight", value, chunk_elements=10
    )
    second = parameter_chunk_hashes(
        f"part0.model.model.layers.1.{target}.weight", value, chunk_elements=10
    )
    assert chunk_map([first]) == chunk_map([second])
    assert sum(c["shape"][0] * c["shape"][1] for c in first["chunks"]) == value.numel()


def test_qkv_is_explicitly_excluded_from_chunk_capture():
    value = torch.ones(4, 3)
    for name in (
        "layers.0.attention.qkv_linear.wqkv.weight",
        "model.model.layers.0.self_attn.k_proj.weight",
    ):
        assert parameter_chunk_hashes(name, value)["chunks"] == []


def test_invalid_geometry_and_chunk_bound_fail_explicitly():
    with pytest.raises(ValueError, match="positive"):
        parameter_chunk_hashes("norm.weight", torch.ones(3), chunk_elements=0)
    with pytest.raises(ValueError, match="even"):
        parameter_chunk_hashes("layers.0.mlp.experts.gate_up_proj", torch.ones(2, 3, 4))


def test_hash_observer_records_first_update_without_repeating_prefix_samples(
    monkeypatch, tmp_path
):
    import json
    from types import SimpleNamespace

    from tests.integration_tests.alignment_gradient_trace import (
        observe_gradient_clipping,
    )

    monkeypatch.setenv("LLMTUNER_ALIGNMENT_PARAMETER_HASH", "1")
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Linear(3, 2, bias=False)])
    parameter = model.layers[0].weight
    parameter.grad = torch.ones_like(parameter)
    trainer = SimpleNamespace(model_parts=[model], step=1)
    observed = observe_gradient_clipping(
        torch.nn.utils.clip_grad_norm_, trainer, tmp_path
    )
    observed(model.parameters(), 1.0)
    with torch.no_grad():
        parameter.add_(1)
    trainer.step = 2
    observed(model.parameters(), 1.0)
    first = json.loads((tmp_path / "parameter-hash-step-1.jsonl").read_text())
    second = json.loads((tmp_path / "parameter-hash-step-2.jsonl").read_text())
    assert first["chunks"][0]["sha256"] != second["chunks"][0]["sha256"]
    record = torch.load(tmp_path / "gradient-clip-step-2.pt", weights_only=True)
    assert record["before"] == record["after"] == {}
