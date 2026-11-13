"""Native DeepSeek V4 construction and random-training contracts."""

from pathlib import Path

import pytest
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as reference
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Experts as NativeDeepseekV4Experts,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4ForCausalLM as NativeDeepseekV4ForCausalLM,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Model as NativeDeepseekV4Model,
)

from llmtuner.components.tokenizer.huggingface import HuggingFaceTokenizer
from llmtuner.config.llmtuner_config import LLMTunerConfig
from llmtuner.config.model_config import ModelConfig
from llmtuner.datasets.text.processors import require_token_prefix
from llmtuner.errors import UnsupportedCombinationError
from llmtuner.memory.materialization import materialize_meta_model
from llmtuner.models.deepseek.deepseek_v4.attention import DeepseekV4GroupedLinear
from llmtuner.models.deepseek.deepseek_v4.compression import DeepseekV4CSACompressor
from llmtuner.models.deepseek.deepseek_v4.hyperconnection import (
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
)
from llmtuner.models.deepseek.deepseek_v4.indexer import DeepseekV4Indexer
from llmtuner.models.deepseek.deepseek_v4.modeling import (
    DeepseekV4Config,
    DeepseekV4Experts,
    DeepseekV4ForCausalLM,
    DeepseekV4MLP,
    DeepseekV4Model,
)
from llmtuner.models.deepseek.deepseek_v4.normalization import (
    DeepseekV4RMSNorm,
    DeepseekV4UnweightedRMSNorm,
)
from llmtuner.models.deepseek.deepseek_v4.rotary import (
    DeepseekV4RotaryEmbedding,
    apply_rotary_pos_emb,
)
from llmtuner.models.deepseek.deepseek_v4.routing import (
    DeepseekV4HashRouter,
    DeepseekV4TopKRouter,
)
from llmtuner.parallel.expert_parallel.model_replacement import resolve_ep_plan
from llmtuner.trainer.runtime.assembly import build_model


def test_grouped_projection_fullgraph_dynamic_forward_and_backward():
    projection = DeepseekV4GroupedLinear(4, 6, 3)
    compiled = torch.compile(
        projection, backend="aot_eager", fullgraph=True, dynamic=True
    )
    for tokens in (2, 5, 0):
        values = torch.randn(tokens, 4, 3).transpose(1, 2).requires_grad_()
        expected_values = values.detach().clone().requires_grad_()
        expected_weight = projection.weight.detach().clone().requires_grad_()
        weights = expected_weight.reshape(3, 2, 4)
        expected = torch.stack(
            [expected_values[:, group] @ weights[group].T for group in range(3)],
            dim=1,
        )
        actual = compiled(values)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        upstream = torch.randn_like(actual)
        observed = torch.autograd.grad(actual, (values, projection.weight), upstream)
        baseline = torch.autograd.grad(
            expected, (expected_values, expected_weight), upstream
        )
        for gradient, reference_gradient in zip(observed, baseline, strict=True):
            torch.testing.assert_close(gradient, reference_gradient)
            if tokens:
                assert gradient.count_nonzero()


@pytest.mark.parametrize("shape", [(), (12,), (2, 6, 2), (2, 1, 4), (2, 3, 8)])
def test_grouped_projection_rejects_wrong_input_layout(shape):
    projection = DeepseekV4GroupedLinear(4, 6, 3)
    with pytest.raises(ValueError, match="expects trailing dimensions"):
        projection(torch.empty(shape))


def test_grouped_projection_noncontiguous_input_matches_separate_groups():
    projection = DeepseekV4GroupedLinear(4, 6, 3)
    values = torch.randn(2, 4, 3).transpose(1, 2).requires_grad_()
    assert not values.is_contiguous()
    actual = projection(values)
    weights = projection.weight.reshape(3, 2, 4)
    expected = torch.stack(
        [values[:, group] @ weights[group].T for group in range(3)], dim=1
    )
    torch.testing.assert_close(actual, expected)
    probe = torch.randn_like(actual)
    observed = torch.autograd.grad(actual, (values, projection.weight), probe)
    baseline = torch.autograd.grad(expected, (values, projection.weight), probe)
    for gradient, reference_gradient in zip(observed, baseline, strict=True):
        torch.testing.assert_close(gradient, reference_gradient)
        assert gradient.count_nonzero()


@pytest.mark.parametrize("shape", [(0, 3, 4), (2, 0, 3, 4)])
def test_grouped_projection_preserves_empty_token_shape_and_gradients(shape):
    projection = DeepseekV4GroupedLinear(4, 6, 3)
    values = torch.empty(shape, requires_grad=True)
    output = projection(values)
    assert output.shape == (*shape[:-1], 2)
    assert output.numel() == 0
    output.sum().backward()
    assert values.grad is not None and values.grad.shape == values.shape
    assert projection.weight.grad is not None
    assert not projection.weight.grad.count_nonzero()


@pytest.mark.parametrize(
    "model_class",
    [
        DeepseekV4Model,
        DeepseekV4ForCausalLM,
        DeepseekV4HashRouter,
        DeepseekV4TopKRouter,
    ],
)
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("num_local_experts", 0),
        ("num_experts_per_tok", 0),
        ("num_experts_per_tok", 5),
        ("num_experts_per_tok", True),
        ("scoring_func", "unknown"),
    ],
)
def test_invalid_routing_rejected_without_allocation(
    monkeypatch, model_class, name, value
):
    config = small_config()
    setattr(config, name, value)
    rng = torch.get_rng_state().clone()

    def reject_allocation(*args, **kwargs):
        raise AssertionError("Invalid routing must fail before allocation")

    monkeypatch.setattr(torch, "empty", reject_allocation)
    with pytest.raises(ValueError, match="DeepSeek V4"):
        model_class(config)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("model_class", [DeepseekV4Model, DeepseekV4ForCausalLM])
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("layer_types", ["sliding_attention"]),
        ("layer_types", ["unknown", "sliding_attention"]),
        ("mlp_layer_types", ["moe"]),
        ("mlp_layer_types", ["hash_moe", "unknown"]),
        ("compress_rates", {}),
        ("compress_rates", {"compressed_sparse_attention": 0}),
        ("compress_rates", {"compressed_sparse_attention": True}),
    ],
)
def test_invalid_schedule_rejected_before_allocation(
    monkeypatch, model_class, name, value
):
    config = small_config()
    setattr(config, name, value)
    rng = torch.get_rng_state().clone()

    def reject_allocation(*args, **kwargs):
        raise AssertionError("Invalid schedule must fail before allocation")

    monkeypatch.setattr(torch.nn.Embedding, "__init__", reject_allocation)
    with pytest.raises(ValueError, match="DeepSeek V4"):
        model_class(config)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize(
    "geometry", [(0, 6, 3), (4, 0, 3), (4, 6, 0), (4, 7, 3), (4, 6, True)]
)
def test_grouped_projection_rejects_geometry_without_rng_changes(geometry):
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="Grouped projection"):
        DeepseekV4GroupedLinear(*geometry)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("model_class", [DeepseekV4Model, DeepseekV4ForCausalLM])
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("o_groups", 0),
        ("o_groups", 3),
        ("o_groups", True),
        ("o_lora_rank", -1),
        ("head_dim", 0),
        ("num_attention_heads", 0),
    ],
)
def test_output_geometry_rejected_before_allocation(
    monkeypatch, model_class, name, value
):
    config = small_config()
    setattr(config, name, value)

    def reject_allocation(*args, **kwargs):
        raise AssertionError("Invalid geometry must fail before embedding allocation")

    monkeypatch.setattr(torch.nn.Embedding, "__init__", reject_allocation)
    with pytest.raises(ValueError, match="DeepSeek V4"):
        model_class(config)


def test_owned_initialization_preserves_loaded_parameters_and_buffers():
    config = small_config()
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    model = DeepseekV4ForCausalLM(config)
    tensors = dict(model.named_parameters()) | dict(model.named_buffers())
    with torch.no_grad():
        for tensor in tensors.values():
            tensor.fill_(0.37 if tensor.is_floating_point() else 1)
            tensor._is_hf_initialized = True
    expected = {name: tensor.clone() for name, tensor in tensors.items()}
    rng = torch.get_rng_state().clone()
    model.apply(model._init_weights)
    assert torch.equal(torch.get_rng_state(), rng)
    torch.testing.assert_close(tensors, expected, atol=0, rtol=0)


@pytest.mark.parametrize("sequence", [1, 7])
def test_owned_csa_preserves_empty_slots_and_indexer_mask(monkeypatch, sequence):
    config = small_config()
    torch.manual_seed(257)
    native = reference.DeepseekV4CSACompressor(config)
    owned = DeepseekV4CSACompressor(config)
    with torch.no_grad():
        for parameter in native.parameters():
            parameter.normal_(std=0.2)
    owned.load_state_dict(native.state_dict(), strict=True)
    hidden = torch.randn(2, sequence, config.hidden_size)
    residual = torch.randn(2, sequence, config.q_lora_rank)
    positions = torch.arange(sequence).unsqueeze(0).expand(2, -1)
    expected = native(hidden, residual, positions, None, 0)

    def reject(*args, **kwargs):
        raise AssertionError("Explicit CSA must own its forward")

    monkeypatch.setattr(reference.DeepseekV4CSACompressor, "forward", reject)
    actual = owned(hidden, residual, positions, None, 0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    entries = sequence // owned.compress_rate
    assert actual[0].shape == (2, 1, entries, config.head_dim)
    assert actual[1].shape == (2, 1, sequence, entries)
    if entries:
        assert actual[0].isfinite().all() and actual[0].count_nonzero()
        assert actual[1][:, :, 0].isneginf().all()
        assert (actual[1][:, :, -1] == 0).any()


@pytest.mark.parametrize("sequence", [1, 7])
def test_owned_indexer_preserves_empty_and_causal_selection(monkeypatch, sequence):
    config = small_config()
    torch.manual_seed(251)
    native = reference.DeepseekV4Indexer(config)
    owned = DeepseekV4Indexer(config)
    with torch.no_grad():
        for parameter in native.parameters():
            parameter.normal_(std=0.2)
    owned.load_state_dict(native.state_dict(), strict=True)
    hidden = torch.randn(2, sequence, config.hidden_size)
    residual = torch.randn(2, sequence, config.q_lora_rank)
    positions = torch.arange(sequence).unsqueeze(0).expand(2, -1)
    expected = native(hidden, residual, positions, None, 0)

    def reject(*args, **kwargs):
        raise AssertionError("Explicit indexer must own its forward")

    monkeypatch.setattr(reference.DeepseekV4Indexer, "forward", reject)
    actual = owned(hidden, residual, positions, None, 0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    if sequence == 1:
        assert actual.shape == (2, 1, 0)
    else:
        assert (actual[:, 0] == -1).all()
        assert (actual[:, -1] >= 0).any()
        threshold = (positions + 1) // owned.compress_rate
        assert ((actual < threshold.unsqueeze(-1)) | (actual == -1)).all()


@pytest.mark.parametrize("std", [0.0, 0.02])
@pytest.mark.parametrize("tied", [False, True])
def test_owned_deepseek_construction_preserves_complete_state_and_rng(
    monkeypatch, std, tied
):
    config = small_config()
    config.initializer_range = std
    config.tie_word_embeddings = tied
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    torch.manual_seed(241)
    native = NativeDeepseekV4ForCausalLM(config)
    rng = torch.get_rng_state().clone()

    def reject(*args, **kwargs):
        raise AssertionError("Explicit V4 must own initialization")

    monkeypatch.setattr(reference.DeepseekV4PreTrainedModel, "_init_weights", reject)
    torch.manual_seed(241)
    owned = DeepseekV4ForCausalLM(config)
    assert torch.equal(torch.get_rng_state(), rng)
    torch.testing.assert_close(owned.state_dict(), native.state_dict(), atol=0, rtol=0)
    assert owned.enable_weight_tying == tied


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("conjugate", [False, True])
def test_owned_rotary_application_matches_value_and_angle_gradients(dtype, conjugate):
    torch.manual_seed(239)
    values = torch.randn(2, 3, 4, 10, dtype=dtype, requires_grad=True)
    cosine = torch.randn(2, 4, 3, requires_grad=True)
    sine = torch.randn(2, 4, 3, requires_grad=True)
    actual_values = values.detach().clone().requires_grad_()
    actual_cosine = cosine.detach().clone().requires_grad_()
    actual_sine = sine.detach().clone().requires_grad_()
    sign = -1 if conjugate else 1
    expected = reference.apply_rotary_pos_emb(values, cosine, sign * sine)
    actual = apply_rotary_pos_emb(actual_values, actual_cosine, sign * actual_sine)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual[..., :4], values[..., :4], atol=0, rtol=0)
    probe = torch.randn_like(actual)
    actual.backward(probe)
    expected.backward(probe)
    for observed, baseline in (
        (actual_values.grad, values.grad),
        (actual_cosine.grad, cosine.grad),
        (actual_sine.grad, sine.grad),
    ):
        torch.testing.assert_close(observed, baseline, atol=0, rtol=0)
        assert observed.isfinite().all() and observed.count_nonzero()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_owned_grouped_projection_preserves_rng_layout_and_gradients(
    monkeypatch, dtype
):
    torch.manual_seed(233)
    native = reference.DeepseekV4GroupedLinear(8, 12, 3).to(dtype)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(233)
    owned = DeepseekV4GroupedLinear(8, 12, 3).to(dtype)
    assert torch.equal(torch.get_rng_state(), rng)
    torch.testing.assert_close(owned.state_dict(), native.state_dict(), atol=0, rtol=0)
    values = torch.randn(2, 4, 3, 8, dtype=dtype, requires_grad=True)
    actual_values = values.detach().clone().requires_grad_()
    expected = native(values)

    def reject(*args, **kwargs):
        raise AssertionError("Explicit grouped projection must own its math")

    monkeypatch.setattr(reference.DeepseekV4GroupedLinear, "forward", reject)
    actual = owned(actual_values)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    probe = torch.randn_like(actual)
    actual.backward(probe)
    expected.backward(probe)
    torch.testing.assert_close(actual_values.grad, values.grad, atol=0, rtol=0)
    torch.testing.assert_close(owned.weight.grad, native.weight.grad, atol=0, rtol=0)
    assert owned.weight.grad.count_nonzero()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_owned_rotary_preserves_both_frequency_sets(monkeypatch, dtype):
    config = small_config()
    native = reference.DeepseekV4RotaryEmbedding(config)
    owned = DeepseekV4RotaryEmbedding(config)
    torch.testing.assert_close(
        dict(owned.named_buffers()), dict(native.named_buffers()), atol=0, rtol=0
    )
    assert not owned.state_dict()
    values = torch.ones(2, 4, config.hidden_size, dtype=dtype)
    positions = torch.tensor([[0, 1, 7, 13], [2, 5, 9, 17]])
    expected = {
        kind: native(values, positions, layer_type=kind)
        for kind in ("main", "compress")
    }

    def reject(*args, **kwargs):
        raise AssertionError("Explicit rotary must own its forward")

    monkeypatch.setattr(reference.DeepseekV4RotaryEmbedding, "forward", reject)
    for kind, baseline in expected.items():
        actual = owned(values, positions, layer_type=kind)
        torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
        assert all(part.dtype == dtype and part.isfinite().all() for part in actual)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_owned_norm_preserves_cast_order(monkeypatch, dtype, weighted, empty):
    native_cls = (
        reference.DeepseekV4RMSNorm
        if weighted
        else reference.DeepseekV4UnweightedRMSNorm
    )
    cls = DeepseekV4RMSNorm if weighted else DeepseekV4UnweightedRMSNorm
    args = (8,) if weighted else ()
    native, owned = native_cls(*args), cls(*args)
    if weighted:
        with torch.no_grad():
            native.weight.copy_(torch.linspace(0.2, 1.7, 8))
        owned.load_state_dict(native.state_dict())
    torch.manual_seed(229)
    values = torch.randn(2, 0 if empty else 3, 8, dtype=dtype, requires_grad=True)
    actual_values = values.detach().clone().requires_grad_()
    expected = native(values)

    def reject(*args, **kwargs):
        raise AssertionError("Explicit norm must own its cast order")

    monkeypatch.setattr(native_cls, "forward", reject)
    actual = owned(actual_values)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    probe = torch.randn_like(actual)
    actual.backward(probe)
    expected.backward(probe)
    torch.testing.assert_close(actual_values.grad, values.grad, atol=0, rtol=0)
    if weighted:
        torch.testing.assert_close(
            owned.weight.grad, native.weight.grad, atol=0, rtol=0
        )
    if not empty:
        assert actual.isfinite().all() and actual.count_nonzero()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("head", [False, True])
def test_owned_hyperconnection_matches_stream_math(monkeypatch, dtype, head):
    config = small_config()
    config.hc_mult = 3
    config.hc_sinkhorn_iters = 3
    native_cls = (
        reference.DeepseekV4HyperHead if head else reference.DeepseekV4HyperConnection
    )
    owned_cls = DeepseekV4HyperHead if head else DeepseekV4HyperConnection
    native, owned = native_cls(config).to(dtype), owned_cls(config).to(dtype)
    torch.manual_seed(227)
    with torch.no_grad():
        for parameter in native.parameters():
            parameter.normal_(std=0.3)
    owned.load_state_dict(native.state_dict(), strict=True)
    values = torch.randn(2, 4, 3, config.hidden_size, dtype=dtype, requires_grad=True)
    actual_values = values.detach().clone().requires_grad_()
    expected = native(values)

    def reject(*args, **kwargs):
        raise AssertionError("Explicit hyperconnection must own its math")

    monkeypatch.setattr(native_cls, "forward", reject)
    actual = owned(actual_values)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual_parts = (actual,) if head else actual
    expected_parts = (expected,) if head else expected
    probes = [torch.randn_like(part) for part in actual_parts]
    sum(
        (part * probe).sum() for part, probe in zip(actual_parts, probes, strict=True)
    ).backward()
    sum(
        (part * probe).sum() for part, probe in zip(expected_parts, probes, strict=True)
    ).backward()
    torch.testing.assert_close(actual_values.grad, values.grad, atol=0, rtol=0)
    assert actual_values.grad.isfinite().all() and actual_values.grad.count_nonzero()
    for name, parameter in owned.named_parameters():
        torch.testing.assert_close(
            parameter.grad, dict(native.named_parameters())[name].grad, atol=0, rtol=0
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
def test_owned_shared_mlp_preserves_initialization_and_math(monkeypatch, dtype, bias):
    config = small_config()
    config.mlp_bias = bias
    config.swiglu_limit = 0.1
    torch.manual_seed(223)
    native = reference.DeepseekV4MLP(config).to(dtype)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(223)
    owned = DeepseekV4MLP(config).to(dtype)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    torch.testing.assert_close(owned.state_dict(), native.state_dict(), atol=0, rtol=0)
    values = torch.randn(2, 3, config.hidden_size, dtype=dtype, requires_grad=True)
    actual_values = values.detach().clone().requires_grad_()
    expected = native(values)

    def reject(*args, **kwargs):
        raise AssertionError("Explicit shared MLP must own its math")

    monkeypatch.setattr(reference.DeepseekV4MLP, "forward", reject)
    actual = owned(actual_values)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.isfinite().all() and actual.count_nonzero()
    probe = torch.randn_like(actual)
    actual.backward(probe)
    expected.backward(probe)
    torch.testing.assert_close(actual_values.grad, values.grad, atol=0, rtol=0)
    for name, parameter in owned.named_parameters():
        torch.testing.assert_close(
            parameter.grad, dict(native.named_parameters())[name].grad, atol=0, rtol=0
        )
    for module in (native, owned):
        torch.optim.SGD(module.parameters(), lr=0.03, foreach=False).step()
    torch.testing.assert_close(owned.state_dict(), native.state_dict(), atol=0, rtol=0)


@pytest.mark.parametrize("hash_routing", [False, True])
def test_owned_router_matches_nontrivial_selection_and_gradients(hash_routing):
    config = small_config()
    torch.manual_seed(211)
    cls = (
        reference.DeepseekV4HashRouter
        if hash_routing
        else reference.DeepseekV4TopKRouter
    )
    native = cls(config)
    owned = (DeepseekV4HashRouter if hash_routing else DeepseekV4TopKRouter)(config)
    with torch.no_grad():
        native.weight.normal_(std=0.3)
        if hash_routing:
            native.tid2eid.copy_(
                torch.arange(config.vocab_size * 2).reshape(-1, 2)
                % config.num_local_experts
            )
        else:
            native.e_score_correction_bias.copy_(torch.tensor([-0.7, 0.2, 0.9, -0.1]))
    owned.load_state_dict(native.state_dict(), strict=True)
    values = torch.randn(2, 3, config.hidden_size, requires_grad=True)
    actual_values = values.detach().clone().requires_grad_()
    extra = (torch.tensor([[1, 3, 7], [5, 4, 8]]),) if hash_routing else ()
    expected = native(values, *extra)
    actual = owned(actual_values, *extra)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    (actual[0].square().sum() + actual[1].square().sum()).backward()
    (expected[0].square().sum() + expected[1].square().sum()).backward()
    torch.testing.assert_close(actual_values.grad, values.grad, atol=0, rtol=0)
    torch.testing.assert_close(owned.weight.grad, native.weight.grad, atol=0, rtol=0)
    assert owned.weight.grad.count_nonzero()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_owned_experts_match_clamp_score_and_gradient_contract(monkeypatch, dtype):
    config = small_config()
    config._experts_implementation = "eager"
    config.swiglu_limit = 0.1
    torch.manual_seed(199)
    native = NativeDeepseekV4Experts(config).to(dtype)
    owned = DeepseekV4Experts(config).to(dtype)
    with torch.no_grad():
        for parameter in native.parameters():
            parameter.normal_(std=0.4)
    owned.load_state_dict(native.state_dict(), strict=True)
    values = torch.randn(3, config.hidden_size, dtype=dtype, requires_grad=True)
    actual_values = values.detach().clone().requires_grad_()
    weights = torch.tensor([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]], dtype=dtype)
    weights.requires_grad_()
    actual_weights = weights.detach().clone().requires_grad_()
    indices = torch.tensor([[0, 1], [1, 0], [0, 1]])
    expected = native(values, indices, weights)

    def reject(*args, **kwargs):
        raise AssertionError("Explicit experts must own their math")

    monkeypatch.setattr(NativeDeepseekV4Experts, "forward", reject)
    actual = owned(actual_values, indices, actual_weights)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.isfinite().all() and actual.count_nonzero()
    probe = torch.randn_like(actual)
    actual.backward(probe)
    expected.backward(probe)
    for observed, baseline in (
        (actual_values.grad, values.grad),
        (actual_weights.grad, weights.grad),
        (owned.gate_up_proj.grad, native.gate_up_proj.grad),
        (owned.down_proj.grad, native.down_proj.grad),
    ):
        torch.testing.assert_close(observed, baseline, atol=0, rtol=0)
        assert observed.isfinite().all() and observed.count_nonzero()
    assert not owned.down_proj.grad[2:].count_nonzero()


def small_config():
    return DeepseekV4Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        q_lora_rank=16,
        o_lora_rank=8,
        o_groups=1,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=12,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=4,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        compress_rates={
            "compressed_sparse_attention": 2,
            "heavily_compressed_attention": 4,
        },
        layer_types=["compressed_sparse_attention", "sliding_attention"],
        mlp_layer_types=["hash_moe", "moe"],
        max_position_embeddings=64,
    )


def test_v4_meta_initialization_and_forward_backward(monkeypatch):
    config = small_config()
    config.attn_mask_type = "causal"
    with torch.device("meta"):
        model = DeepseekV4ForCausalLM(config)

    def reject(*args, **kwargs):
        raise AssertionError("Materialization must construct model-owned buffers")

    monkeypatch.setattr(reference.DeepseekV4RotaryEmbedding, "__init__", reject)
    monkeypatch.setattr(reference.DeepseekV4HashRouter, "__init__", reject)
    parameters = dict(model.named_parameters())
    device = torch.device("cpu")
    materialize_meta_model(
        model, device, runtime_buffers=model.materialization_buffers(device)
    )
    assert all(
        dict(model.named_parameters())[name] is p for name, p in parameters.items()
    )
    model.initialize_parameters()
    hash_router = model.model.layers[0].mlp.gate
    table = hash_router.tid2eid
    assert table.shape == (64, 2)
    assert torch.all((0 <= table) & (table < 4))
    assert torch.any(table != 0)
    assert torch.all(table[:, 0] != table[:, 1])
    counts = torch.bincount(table.flatten(), minlength=4)
    assert counts.max() - counts.min() <= 1
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    tokens = torch.arange(8) % 64
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, foreach=False)
    before = model.lm_head.weight.detach().clone()
    output = model(tokens, positions=torch.arange(8)).logits
    assert output.shape == (8, 64)
    output.float().square().mean().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(bool(gradient.count_nonzero()) for gradient in gradients)
    optimizer.step()
    assert not torch.equal(model.lm_head.weight, before)


def test_explicit_v4_model_matches_native_hf_forward() -> None:
    config = small_config()
    native = NativeDeepseekV4ForCausalLM(config).eval()
    explicit = DeepseekV4ForCausalLM(config).eval()
    explicit.load_state_dict(native.state_dict(), strict=True)
    tokens = torch.arange(8) % 64
    with torch.no_grad():
        expected = native(input_ids=tokens.unsqueeze(0), use_cache=False).logits
        actual = explicit(input_ids=tokens.unsqueeze(0), use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("return_cache", [False, True])
@pytest.mark.parametrize(
    "compressed_type", ["compressed_sparse_attention", "heavily_compressed_attention"]
)
def test_owned_decoder_preserves_caller_cache_continuation(
    monkeypatch, return_cache, compressed_type
):
    config = small_config()
    config.layer_types[0] = compressed_type
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    torch.manual_seed(197)
    native = NativeDeepseekV4Model(config).eval()
    explicit = DeepseekV4ForCausalLM(config).model.eval()
    explicit.load_state_dict(native.state_dict(), strict=True)
    expected_cache = DynamicCache(config=config)
    actual_cache = DynamicCache(config=config)
    chunks = (
        torch.tensor([[1, 3, 7, 2], [5, 4, 8, 9]]),
        torch.tensor([[6], [11]]),
        torch.tensor([[12, 13], [14, 15]]),
    )
    expected = []
    with torch.no_grad():
        for tokens in chunks:
            output = native(
                tokens, past_key_values=expected_cache, use_cache=return_cache
            )
            expected.append(output.last_hidden_state.clone())

    def reject(*args, **kwargs):
        raise AssertionError("Explicit decoder must own cached execution")

    monkeypatch.setattr(NativeDeepseekV4Model, "forward", reject)
    with torch.no_grad():
        for tokens, baseline in zip(chunks, expected, strict=True):
            output = explicit(
                tokens, past_key_values=actual_cache, use_cache=return_cache
            )
            torch.testing.assert_close(
                output.last_hidden_state, baseline, atol=0, rtol=0
            )
            assert output.last_hidden_state.isfinite().all()
            assert output.last_hidden_state.count_nonzero()
            assert output.past_key_values is (actual_cache if return_cache else None)
    for index in range(config.num_hidden_layers):
        assert actual_cache.get_seq_length(index) == expected_cache.get_seq_length(
            index
        )
        assert actual_cache.get_seq_length(index) > 0
        for name in (
            "keys",
            "values",
            "buffer_kv",
            "buffer_gate",
            "compressed_kv",
            "entry_count",
            "overlap_kv",
            "overlap_gate",
        ):
            baseline = getattr(expected_cache.layers[index], name, None)
            observed = getattr(actual_cache.layers[index], name, None)
            torch.testing.assert_close(observed, baseline, atol=0, rtol=0)


@pytest.mark.parametrize("return_dict", [False, True])
@pytest.mark.parametrize("logits_to_keep", [0, 2])
@pytest.mark.parametrize("router_outputs", [False, True])
@pytest.mark.parametrize(
    "compressed_type", ["compressed_sparse_attention", "heavily_compressed_attention"]
)
def test_owned_lm_matches_reference_without_forward_delegation(
    monkeypatch, return_dict, logits_to_keep, router_outputs, compressed_type
):
    config = small_config()
    config.layer_types[0] = compressed_type
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    torch.manual_seed(191)
    native = NativeDeepseekV4ForCausalLM(config).eval()
    explicit = DeepseekV4ForCausalLM(config).eval()
    explicit.load_state_dict(native.state_dict(), strict=True)
    tokens = torch.tensor([[1, 3, 7, 2], [5, 4, 8, 9]])
    labels = tokens if logits_to_keep == 0 else None
    expected = native(
        tokens,
        labels=labels,
        logits_to_keep=logits_to_keep,
        output_router_logits=router_outputs,
        output_hidden_states=True,
        output_attentions=True,
        use_cache=False,
    )

    def reject(*args, **kwargs):
        raise AssertionError("Explicit LM must own its forward")

    monkeypatch.setattr(NativeDeepseekV4ForCausalLM, "forward", reject)
    monkeypatch.setattr(NativeDeepseekV4Model, "forward", reject)
    monkeypatch.setattr(reference.DeepseekV4TopKRouter, "forward", reject)
    monkeypatch.setattr(reference.DeepseekV4HashRouter, "forward", reject)
    monkeypatch.setattr(reference.DeepseekV4MLP, "forward", reject)
    explicit.config.return_dict = return_dict
    monkeypatch.setattr(reference.DeepseekV4HyperConnection, "forward", reject)
    monkeypatch.setattr(reference.DeepseekV4HyperHead, "forward", reject)
    monkeypatch.setattr(reference.DeepseekV4Attention, "forward", reject)
    monkeypatch.setattr(reference.DeepseekV4GroupedLinear, "forward", reject)
    monkeypatch.setattr(reference.DeepseekV4HCACompressor, "forward", reject)
    monkeypatch.setattr(reference.DeepseekV4IndexerScorer, "forward", reject)
    monkeypatch.setattr(reference.DeepseekV4Indexer, "forward", reject)
    monkeypatch.setattr(reference.DeepseekV4CSACompressor, "forward", reject)
    actual = explicit(
        tokens,
        labels=labels,
        logits_to_keep=logits_to_keep,
        output_router_logits=router_outputs,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=return_dict,
    )
    actual_logits = (
        actual.logits
        if return_dict
        else actual[int(labels is not None) + int(router_outputs)]
    )
    if return_dict and router_outputs:
        assert actual.router_logits
        torch.testing.assert_close(actual.aux_loss, expected.aux_loss, atol=0, rtol=0)
        for observed, baseline in zip(
            actual.router_logits, expected.router_logits, strict=True
        ):
            torch.testing.assert_close(observed, baseline, atol=0, rtol=0)
    torch.testing.assert_close(actual_logits, expected.logits, atol=0, rtol=0)
    if return_dict:
        assert actual.hidden_states
        assert actual.attentions
        for name in ("hidden_states", "attentions"):
            for observed, baseline in zip(
                getattr(actual, name), getattr(expected, name), strict=True
            ):
                torch.testing.assert_close(observed, baseline, atol=0, rtol=0)
    actual_loss = actual.loss if return_dict else actual[0]
    if labels is not None:
        torch.testing.assert_close(actual_loss, expected.loss, atol=0, rtol=0)
        actual_loss.backward()
        expected.loss.backward()
    else:
        actual_logits.square().sum().backward()
        expected.logits.square().sum().backward()
    for name, parameter in explicit.named_parameters():
        torch.testing.assert_close(
            parameter.grad, dict(native.named_parameters())[name].grad, atol=0, rtol=0
        )


def test_eager_training_initializes_hash_buffers_without_replaying_weights(tmp_path):
    config = small_config()
    config.save_pretrained(tmp_path)
    cfg = LLMTunerConfig(model=ModelConfig(model_name_or_path=str(tmp_path)))
    model, _, load_initial_weights = build_model(
        cfg, parallel_dims=None, device=torch.device("cpu")
    )
    assert not load_initial_weights
    table = model.model.layers[0].mlp.gate.tid2eid
    assert table.count_nonzero()
    parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    model.initialize_training_buffers()
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(p, parameters[name]) for name, p in model.named_parameters())
    restored = DeepseekV4ForCausalLM(config)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(restored.model.layers[0].mlp.gate.tid2eid, table)


def test_initial_weight_loading_skips_random_training_buffers(tmp_path, monkeypatch):
    small_config().save_pretrained(tmp_path)
    cfg = LLMTunerConfig(model=ModelConfig(model_name_or_path=str(tmp_path)))
    cfg.checkpoint.enable = True
    cfg.checkpoint.initial_load_in_hf = True
    cfg.checkpoint.initial_load_path = str(tmp_path)

    def reject(self):
        raise AssertionError("Checkpoint targets must not initialize training buffers")

    monkeypatch.setattr(DeepseekV4ForCausalLM, "initialize_training_buffers", reject)
    model, _, load_initial_weights = build_model(
        cfg, parallel_dims=None, device=torch.device("cpu")
    )
    assert load_initial_weights
    assert all(parameter.is_meta for parameter in model.parameters())


def test_v4_recipe_chat_template_preserves_supervision_boundary():
    root = Path("/home/jianzhnie/llmtuner/hfhub/models/deepseek-ai/DeepSeek-V4-Flash")
    if not root.is_dir():
        pytest.skip("Official V4 tokenizer is unavailable")
    tokenizer = HuggingFaceTokenizer(tokenizer_path=str(root))
    template = Path("examples/deepseek_v4_pro/chat_template.jinja").read_text(
        encoding="utf-8"
    )
    tokenizer.set_chat_template(template)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Question"}], add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ],
        add_generation_prompt=False,
    )
    assert prompt.endswith("<｜Assistant｜></think>")
    assert full == prompt + "Answer<｜end▁of▁sentence｜>"
    require_token_prefix(
        tokenizer.encode(full, add_bos=True, add_eos=False),
        tokenizer.encode(prompt, add_bos=True, add_eos=False),
    )


def test_parallel_validation_rejects_unimplemented_tp_ep_without_mutation():
    model = DeepseekV4ForCausalLM(small_config())
    parameters = tuple(model.parameters())
    state = {name: value.clone() for name, value in model.state_dict().items()}
    forwards = tuple(module.forward for module in model.modules())
    rng = torch.get_rng_state().clone()

    with pytest.raises(
        UnsupportedCombinationError, match="does not declare a default TP plan"
    ):
        model.validate_parallel(tp_degree=2, ep_degree=1)
    with pytest.raises(UnsupportedCombinationError, match="clamped expert math"):
        model.validate_parallel(tp_degree=1, ep_degree=2)
    with pytest.raises(
        UnsupportedCombinationError, match="requires a model-owned MoEBlockContract"
    ):
        resolve_ep_plan(model, degree=2)

    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert forwards == tuple(module.forward for module in model.modules())
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
