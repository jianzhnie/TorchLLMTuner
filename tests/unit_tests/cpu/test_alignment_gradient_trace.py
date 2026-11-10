from types import SimpleNamespace

import torch

from tests.integration_tests.alignment_gradient_trace import (
    local_gradient_record,
    observe_gradient_clipping,
)


def test_local_samples_are_bounded_independent_copies():
    gradient = torch.arange(64.0).reshape(2, 4, 8)
    record = local_gradient_record(gradient, sample_size=3)
    assert record["sample"].numel() == 3
    assert all(value.numel() == 3 for value in record["half_samples"])
    torch.testing.assert_close(record["half_samples"][1], gradient[0, 2, :3])
    gradient.zero_()
    torch.testing.assert_close(record["sample"], torch.arange(3.0))


def test_observer_preserves_real_clip_and_preclip_evidence(tmp_path):
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])
    parameter = model.layers[0].weight
    parameter.grad = torch.full_like(parameter, 3.0)
    trainer = SimpleNamespace(model_parts=[model], step=1)
    observed = observe_gradient_clipping(
        torch.nn.utils.clip_grad_norm_, trainer, tmp_path
    )
    norm = observed(model.parameters(), 1.0)
    torch.testing.assert_close(norm, torch.tensor(6.0))
    record = torch.load(tmp_path / "gradient-clip-step-1.pt", weights_only=True)
    name = "part0.layers.0.weight"
    torch.testing.assert_close(record["before"][name]["sample"], torch.full((4,), 3.0))
    torch.testing.assert_close(
        record["after"][name]["sample"], parameter.grad.flatten()
    )
    torch.testing.assert_close(record["returned_norm"], norm)


def test_gradient_trace_selects_exact_layer_indices(monkeypatch, tmp_path):
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(
        [torch.nn.Linear(2, 2, bias=False) for _ in range(11)]
    )
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    monkeypatch.setenv("LLMTUNER_ALIGNMENT_GRADIENT_LAYERS", "1,10")
    trainer = SimpleNamespace(model_parts=[model], step=1)
    observed = observe_gradient_clipping(
        torch.nn.utils.clip_grad_norm_, trainer, tmp_path
    )
    observed(model.parameters(), 1.0)
    record = torch.load(tmp_path / "gradient-clip-step-1.pt", weights_only=True)
    assert set(record["before"]) == {"part0.layers.1.weight", "part0.layers.10.weight"}


def test_observer_preserves_bf16_norm_and_real_clipped_gradients(tmp_path):
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(
        [torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)]
    )
    parameter = model.layers[0].weight
    parameter.grad = torch.full_like(parameter, 3.0)
    reference = torch.nn.Parameter(parameter.detach().clone())
    reference.grad = parameter.grad.clone()
    expected_norm = torch.nn.utils.clip_grad_norm_([reference], 1.0)
    trainer = SimpleNamespace(model_parts=[model], step=1)
    observed = observe_gradient_clipping(
        torch.nn.utils.clip_grad_norm_, trainer, tmp_path
    )
    actual_norm = observed(model.parameters(), max_norm=1.0)
    torch.testing.assert_close(actual_norm, expected_norm, atol=0, rtol=0)
    torch.testing.assert_close(parameter.grad, reference.grad, atol=0, rtol=0)
    record = torch.load(tmp_path / "gradient-clip-step-1.pt", weights_only=True)
    assert record["returned_norm"].dtype == torch.bfloat16
    assert record["before"]["part0.layers.0.weight"]["native_norm_dtype"] == (
        "torch.bfloat16"
    )
    assert record["parameter_order"][0]["gradient_dtype"] == "torch.bfloat16"
    torch.testing.assert_close(
        record["clip_coefficient_from_returned_norm"],
        (1.0 / (expected_norm + 1e-6)).clamp(max=1),
        atol=0,
        rtol=0,
    )


def test_parameter_samples_distinguish_update_from_clipping(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMTUNER_ALIGNMENT_PARAMETER_SAMPLES", "1")
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])
    parameter = model.layers[0].weight
    parameter.grad = torch.full_like(parameter, 3.0)
    initial = parameter.detach().flatten().clone()
    trainer = SimpleNamespace(model_parts=[model], step=1)
    observed = observe_gradient_clipping(
        torch.nn.utils.clip_grad_norm_, trainer, tmp_path
    )
    observed(model.parameters(), 1.0)
    with torch.no_grad():
        parameter.add_(1.0)
    trainer.step = 2
    observed(model.parameters(), 1.0)
    first = torch.load(tmp_path / "gradient-clip-step-1.pt", weights_only=True)
    second = torch.load(tmp_path / "gradient-clip-step-2.pt", weights_only=True)
    name = "part0.layers.0.weight"
    for phase in ("before", "after"):
        torch.testing.assert_close(first[phase][name]["parameter_sample"], initial)
        torch.testing.assert_close(
            second[phase][name]["parameter_sample"], initial + 1.0
        )


def test_full_qkv_capture_filters_parameters_and_owns_local_copies(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("LLMTUNER_ALIGNMENT_FULL_QKV", "1")
    model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    layer.self_attn.q_proj = torch.nn.Linear(3, 2, bias=False)
    layer.self_attn.k_proj = torch.nn.Linear(3, 1, bias=False)
    layer.self_attn.v_proj = torch.nn.Linear(3, 1, bias=False)
    layer.self_attn.o_proj = torch.nn.Linear(3, 3, bias=False)
    model.layers = torch.nn.ModuleList([layer])
    originals = {}
    for name, parameter in model.named_parameters():
        parameter.grad = torch.ones_like(parameter)
        originals[name] = parameter.detach().clone()
    trainer = SimpleNamespace(model_parts=[model], step=1)
    observed = observe_gradient_clipping(
        torch.nn.utils.clip_grad_norm_, trainer, tmp_path
    )
    observed(model.parameters(), 1.0)
    records = torch.load(tmp_path / "gradient-clip-step-1.pt", weights_only=True)
    assert len(records["before"]) == 3
    assert not any("o_proj" in name for name in records["before"])
    for name, parameter in model.named_parameters():
        if "o_proj" in name:
            continue
        before = records["before"]["part0." + name]
        after = records["after"]["part0." + name]
        assert before["full_local_gradient"].shape == parameter.shape
        torch.testing.assert_close(
            before["full_local_gradient"], torch.ones_like(parameter)
        )
        torch.testing.assert_close(after["full_local_gradient"], parameter.grad)
        with torch.no_grad():
            parameter.zero_()
        torch.testing.assert_close(before["full_local_parameter"], originals[name])


def test_full_qkv_name_selection_matches_actual_converter():
    from tests.integration_tests.alignment_gradient_trace import _is_qkv_parameter

    assert _is_qkv_parameter(
        "layers.0._checkpoint_wrapped_module.attention.qkv_linear.wqkv.weight"
    )
    assert _is_qkv_parameter(
        "model.layers.47._checkpoint_wrapped_module.self_attn.k_proj.weight"
    )
    assert not _is_qkv_parameter("layers.0.self_attn.q_norm.weight")
    assert not _is_qkv_parameter("layers.0.mlp.gate.weight")


def test_full_outer_capture_selects_actual_names_and_complete_tensors(
    monkeypatch, tmp_path
):
    from tests.integration_tests.alignment_gradient_trace import _is_outer_parameter

    for name in (
        "tok_embeddings.weight",
        "norm.weight",
        "lm_head.weight",
        "output.weight",
        "model.model.embed_tokens.weight",
        "model.model.norm.weight",
        "model.lm_head.weight",
    ):
        assert _is_outer_parameter(name)
    assert not _is_outer_parameter("layers.0.input_layernorm.weight")
    monkeypatch.setenv("LLMTUNER_ALIGNMENT_FULL_OUTER", "1")
    model = torch.nn.Module()
    model.tok_embeddings = torch.nn.Embedding(7, 3)
    model.norm = torch.nn.LayerNorm(3, bias=False)
    model.lm_head = torch.nn.Linear(3, 7, bias=False)
    model.layers = torch.nn.ModuleList([torch.nn.Linear(3, 3, bias=False)])
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    trainer = SimpleNamespace(model_parts=[model], step=1)
    observed = observe_gradient_clipping(
        torch.nn.utils.clip_grad_norm_, trainer, tmp_path
    )
    observed(model.parameters(), 1.0)
    record = torch.load(tmp_path / "gradient-clip-step-1.pt", weights_only=True)
    assert set(record["before"]) == {
        "part0.tok_embeddings.weight",
        "part0.norm.weight",
        "part0.lm_head.weight",
    }
    for name, parameter in model.named_parameters():
        if name.startswith("layers."):
            continue
        before = record["before"]["part0." + name]
        after = record["after"]["part0." + name]
        torch.testing.assert_close(before["full_local_parameter"], parameter)
        torch.testing.assert_close(after["full_local_gradient"], parameter.grad)


def test_parameter_hash_and_full_qkv_capture_can_coexist(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMTUNER_ALIGNMENT_PARAMETER_HASH", "1")
    monkeypatch.setenv("LLMTUNER_ALIGNMENT_FULL_QKV", "1")
    model = torch.nn.Module()
    attention = torch.nn.Module()
    attention.q_proj = torch.nn.Linear(2, 2, bias=False)
    layer = torch.nn.Module()
    layer.self_attn = attention
    model.layers = torch.nn.ModuleList([layer])
    parameter = attention.q_proj.weight
    parameter.grad = torch.ones_like(parameter)
    trainer = SimpleNamespace(model_parts=[model], step=1)
    observed = observe_gradient_clipping(
        torch.nn.utils.clip_grad_norm_, trainer, tmp_path
    )
    observed(model.parameters(), 1.0)
    assert (tmp_path / "parameter-hash-step-1.jsonl").exists()
    record = torch.load(tmp_path / "gradient-clip-step-1.pt", weights_only=True)
    assert torch.equal(
        record["before"]["part0.layers.0.self_attn.q_proj.weight"][
            "full_local_gradient"
        ],
        torch.ones_like(parameter),
    )
