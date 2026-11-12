"""Model-owned GDN decay initialization cannot introduce log-zero infinities."""

from copy import deepcopy

import pytest
import torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoePreTrainedModel,
)

from llmtuner.models.qwen.qwen3_5_moe.model import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextModel,
)


@pytest.mark.parametrize("constructor", [Qwen3_5MoeTextModel, Qwen3_5MoeForCausalLM])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_delta_initializer_remains_finite(constructor, dtype):
    config = Qwen3_5MoeTextConfig()
    with torch.device("meta"):
        model = constructor(config)
    delta = next(m for m in model.modules() if isinstance(m, Qwen3_5MoeGatedDeltaNet))
    delta.A_log = torch.nn.Parameter(torch.empty(8192, dtype=dtype))
    delta.dt_bias = torch.nn.Parameter(torch.empty(8192, dtype=dtype))
    identity = id(delta.A_log)
    torch.manual_seed(73)
    model._init_weights(delta)
    assert id(delta.A_log) == identity
    assert delta.A_log.isfinite().all()
    assert delta.A_log.float().exp().min() > 0
    assert delta.A_log.count_nonzero() > 8000
    assert torch.equal(delta.dt_bias, torch.ones_like(delta.dt_bias))
    torch.manual_seed(73)
    reference = torch.empty(8192).uniform_(0, 16)
    reference.clamp_min_(torch.finfo(torch.float32).tiny).log_()
    torch.testing.assert_close(delta.A_log, reference.to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize("constructor", [Qwen3_5MoeTextModel, Qwen3_5MoeForCausalLM])
@pytest.mark.parametrize(
    "option,value",
    [
        ("linear_attention_backend", "npu_triton"),
        ("gdn_numerical_policy", "split_qkv_fp32_norm"),
        ("gdn_allow_right_padding", True),
        ("gdn_deterministic_reduction", True),
    ],
)
def test_unimplemented_gdn_policy_fails_before_allocation(
    constructor, option, value, monkeypatch
):
    config = Qwen3_5MoeTextConfig()
    setattr(config, option, value)
    before = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid policy must fail before parameter allocation")

    monkeypatch.setattr(torch.nn.Embedding, "__init__", forbidden)
    with pytest.raises(ValueError, match="explicit"):
        constructor(config)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("decay_loaded", [False, True])
@pytest.mark.parametrize("bias_loaded", [False, True])
def test_delta_initializer_preserves_individually_loaded_parameters(
    decay_loaded, bias_loaded
):
    with torch.device("meta"):
        model = Qwen3_5MoeForCausalLM(Qwen3_5MoeTextConfig())
    delta = next(m for m in model.modules() if isinstance(m, Qwen3_5MoeGatedDeltaNet))
    delta.A_log = torch.nn.Parameter(torch.full((32,), 0.5, dtype=torch.bfloat16))
    delta.dt_bias = torch.nn.Parameter(torch.full((32,), 3.0, dtype=torch.bfloat16))
    delta.A_log._is_hf_initialized = decay_loaded
    delta.dt_bias._is_hf_initialized = bias_loaded
    decay, bias = delta.A_log, delta.dt_bias
    rng = torch.get_rng_state().clone()
    model._init_weights(delta)
    assert delta.A_log is decay and delta.dt_bias is bias
    assert torch.equal(delta.dt_bias, torch.full_like(bias, 3 if bias_loaded else 1))
    if decay_loaded:
        assert torch.equal(delta.A_log, torch.full_like(decay, 0.5))
        assert torch.equal(torch.get_rng_state(), rng)
    else:
        assert not torch.equal(torch.get_rng_state(), rng)
        assert decay.isfinite().all() and not torch.equal(
            decay, torch.full_like(decay, 0.5)
        )


@pytest.mark.parametrize("constructor", [Qwen3_5MoeTextModel, Qwen3_5MoeForCausalLM])
@pytest.mark.parametrize("decay_loaded", [False, True])
@pytest.mark.parametrize("bias_loaded", [False, True])
def test_fp32_delta_owned_initializer_preserves_reference_rng_and_partial_load(
    monkeypatch, constructor, decay_loaded, bias_loaded
):
    with torch.device("meta"):
        model = constructor(Qwen3_5MoeTextConfig())
    delta = next(m for m in model.modules() if isinstance(m, Qwen3_5MoeGatedDeltaNet))
    delta.A_log = torch.nn.Parameter(torch.full((32,), 0.5))
    delta.dt_bias = torch.nn.Parameter(torch.full((32,), 3.0))
    delta.A_log._is_hf_initialized = decay_loaded
    delta.dt_bias._is_hf_initialized = bias_loaded
    expected = deepcopy(delta)
    # Parameter deepcopy intentionally drops custom tensor attributes.
    expected.A_log._is_hf_initialized = decay_loaded
    expected.dt_bias._is_hf_initialized = bias_loaded
    torch.manual_seed(139)
    Qwen3_5MoePreTrainedModel._init_weights(model, expected)
    expected_rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        pytest.fail("Model-owned GDN initialization delegated to reference")

    monkeypatch.setattr(Qwen3_5MoePreTrainedModel, "_init_weights", forbidden)
    decay, bias = delta.A_log, delta.dt_bias
    torch.manual_seed(139)
    model._init_weights(delta)
    assert delta.A_log is decay and delta.dt_bias is bias
    assert torch.equal(torch.get_rng_state(), expected_rng)
    torch.testing.assert_close(delta.A_log, expected.A_log, rtol=0, atol=0)
    torch.testing.assert_close(delta.dt_bias, expected.dt_bias, rtol=0, atol=0)
