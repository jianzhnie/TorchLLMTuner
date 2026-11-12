"""Model-owned assembly preserves reference initialization and training."""

from copy import deepcopy

import pytest
import torch
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM as ReferenceModel,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeModel as ReferenceDecoder,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoePreTrainedModel as ReferencePreTrainedModel,
)

from llmtuner.models.configuration import resolve_model_class
from llmtuner.models.qwen.qwen3.model import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
)


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("experts", [None, "eager"])
def test_owned_qwen3_matches_reference_initialization_and_update(
    tied, backend, experts, monkeypatch
):
    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        moe_intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        mlp_only_layers=[1],
        tie_word_embeddings=tied,
        attention_dropout=0.0,
    )
    config._attn_implementation = backend
    config._experts_implementation = experts
    assert resolve_model_class(config) is Qwen3MoeForCausalLM
    torch.manual_seed(37)
    expected = ReferenceModel(deepcopy(config))
    reference_rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("reference model-specific initialization")

    monkeypatch.setattr(ReferencePreTrainedModel, "_init_weights", forbidden)
    torch.manual_seed(37)
    actual = Qwen3MoeForCausalLM(deepcopy(config))
    assert (
        actual.config._experts_implementation == expected.config._experts_implementation
    )
    assert torch.equal(torch.get_rng_state(), reference_rng)
    assert type(actual.model.layers[0]) is Qwen3MoeDecoderLayer
    assert (actual.model.embed_tokens.weight is actual.lm_head.weight) == tied
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for name, value in expected.state_dict().items():
        assert torch.equal(actual.state_dict()[name], value), name

    tokens = torch.tensor([[1, 3, 7, 2], [5, 4, 2, 9]])
    outputs = []
    for model in (expected, actual):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03, foreach=False)
        output = model(input_ids=tokens, labels=tokens, use_cache=False)
        assert torch.isfinite(output.loss) and output.loss.item() > 0
        output.loss.backward()
        outputs.append(output)
        optimizer.step()
    assert torch.equal(outputs[0].logits, outputs[1].logits)
    assert torch.equal(outputs[0].loss, outputs[1].loss)
    for (name, reference), (_, owned) in zip(
        expected.named_parameters(), actual.named_parameters(), strict=True
    ):
        assert (reference.grad is None) == (owned.grad is None), name
        if reference.grad is not None:
            assert torch.equal(reference.grad, owned.grad), name
        assert torch.equal(reference, owned), name


@pytest.mark.parametrize(
    ("owned", "reference"),
    [(Qwen3MoeModel, ReferenceDecoder), (Qwen3MoeForCausalLM, ReferenceModel)],
)
def test_owned_experts_capability_delegates_to_reference(owned, reference):
    assert owned._can_set_experts_implementation() == (
        reference._can_set_experts_implementation()
    )


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("sliding_window", [None, 2])
def test_owned_decoder_matches_masks_positions_capture_and_cached_decode(
    backend, sliding_window, monkeypatch
):
    """Exercise decoder semantics independently of the inherited LM forward."""
    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        moe_intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        attention_dropout=0.0,
        sliding_window=sliding_window,
    )
    config._attn_implementation = backend
    config._experts_implementation = "eager"
    torch.manual_seed(51)
    reference = ReferenceDecoder(deepcopy(config))
    owned = Qwen3MoeModel(deepcopy(config))
    owned.load_state_dict(reference.state_dict())
    tokens = torch.tensor([[0, 3, 7, 2], [0, 0, 5, 9]])
    mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]])
    positions = torch.tensor([[0, 0, 1, 2], [0, 0, 0, 1]])
    expected = reference(
        tokens,
        attention_mask=mask,
        position_ids=positions,
        use_cache=False,
        output_hidden_states=True,
        output_router_logits=True,
    )

    def forbidden_reference_forward(*args, **kwargs):
        raise AssertionError("explicit decoder must own its execution")

    monkeypatch.setattr(ReferenceDecoder, "forward", forbidden_reference_forward)
    actual = owned(
        tokens,
        attention_mask=mask,
        position_ids=positions,
        use_cache=False,
        output_hidden_states=True,
        output_router_logits=True,
    )
    torch.testing.assert_close(
        actual.last_hidden_state, expected.last_hidden_state, rtol=0, atol=0
    )
    for field in ("hidden_states", "router_logits"):
        assert getattr(actual, field) and getattr(expected, field)
        for value, baseline in zip(
            getattr(actual, field), getattr(expected, field), strict=True
        ):
            torch.testing.assert_close(value, baseline, rtol=0, atol=0)
    weights = torch.arange(16, dtype=torch.float32) + 1
    for output in (expected, actual):
        (output.last_hidden_state * weights).square().sum().backward()
    for name, parameter in owned.named_parameters():
        baseline = dict(reference.named_parameters())[name]
        assert (parameter.grad is None) == (baseline.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, baseline.grad, rtol=0, atol=0)
    monkeypatch.undo()

    # Decoder caching remains an explicit component capability even though
    # the training-facing causal model always disables use_cache.
    with torch.no_grad():
        expected_prefill = reference(tokens, use_cache=True)
        actual_prefill = owned(tokens, use_cache=True)
        next_tokens = torch.tensor([[11], [13]])
        expected_next = reference(
            next_tokens,
            past_key_values=expected_prefill.past_key_values,
            use_cache=True,
        )
        actual_next = owned(
            next_tokens, past_key_values=actual_prefill.past_key_values, use_cache=True
        )
    assert actual_next.past_key_values.get_seq_length() == tokens.shape[1] + 1
    torch.testing.assert_close(
        actual_next.last_hidden_state, expected_next.last_hidden_state, rtol=0, atol=0
    )


def test_owned_decoder_requires_exactly_one_input():
    config = Qwen3MoeConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=8,
        moe_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=2,
        num_experts_per_tok=1,
    )
    model = Qwen3MoeModel(config)
    with pytest.raises(ValueError, match="exactly one"):
        model(use_cache=False)
    with pytest.raises(ValueError, match="exactly one"):
        model(
            torch.ones(1, 2, dtype=torch.long),
            inputs_embeds=torch.ones(1, 2, 8),
            use_cache=False,
        )


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("return_dict", [False, True])
@pytest.mark.parametrize("with_loss", [False, True])
def test_owned_lm_forward_matches_reference_without_reference_execution(
    backend, return_dict, with_loss, monkeypatch
):
    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        moe_intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        mlp_only_layers=[1],
        attention_dropout=0.0,
        output_router_logits=True,
    )
    config._attn_implementation = backend
    config._experts_implementation = "eager"
    torch.manual_seed(83)
    reference = ReferenceModel(deepcopy(config))
    owned = Qwen3MoeForCausalLM(deepcopy(config))
    owned.load_state_dict(reference.state_dict())
    tokens = torch.tensor([[0, 3, 7, 2], [0, 0, 5, 9]])
    mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]])
    kwargs = {
        "attention_mask": mask,
        "return_dict": return_dict,
        "output_hidden_states": True,
        "use_cache": False,
    }
    if with_loss:
        kwargs["labels"] = tokens.masked_fill(mask == 0, -100)
    else:
        kwargs["logits_to_keep"] = torch.tensor([1, 3])
    expected = reference(tokens, **kwargs)

    def forbidden_reference_forward(*args, **kwargs):
        raise AssertionError("explicit model must not call inherited forward")

    monkeypatch.setattr(ReferenceModel, "forward", forbidden_reference_forward)
    monkeypatch.setattr(ReferenceDecoder, "forward", forbidden_reference_forward)
    actual = owned(tokens, **kwargs)
    if return_dict:
        assert actual.aux_loss > 0
        expected_values, actual_values = expected.to_tuple(), actual.to_tuple()
    else:
        expected_values, actual_values = expected, actual
    torch.testing.assert_close(actual_values, expected_values, rtol=0, atol=0)
    if with_loss:
        expected_loss = expected.loss if return_dict else expected[0]
        actual_loss = actual.loss if return_dict else actual[0]
        assert actual_loss > 0
        expected_loss.backward()
        actual_loss.backward()
        for name, parameter in owned.named_parameters():
            baseline = dict(reference.named_parameters())[name]
            assert (parameter.grad is None) == (baseline.grad is None), name
            if parameter.grad is not None:
                torch.testing.assert_close(
                    parameter.grad, baseline.grad, rtol=0, atol=0
                )
        assert owned.model.layers[0].mlp.gate.weight.grad.count_nonzero() > 0


@pytest.mark.parametrize("return_dict", [False, True])
@pytest.mark.parametrize("configured_dict", [False, True])
def test_owned_flat_logits_and_hidden_only_ignore_tuple_setting(
    return_dict, configured_dict
):
    """Flat outputs keep their token axis; hidden-only never executes the head."""
    config = Qwen3MoeConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=8,
        moe_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=2,
        num_experts_per_tok=1,
    )
    config.return_dict = configured_dict
    model = Qwen3MoeForCausalLM(config)
    tokens = torch.tensor([1, 3, 5])
    output = model(tokens, return_dict=return_dict)
    logits = output.logits if return_dict else output[0]
    assert logits.shape == (3, config.vocab_size)
    head_calls = []
    handle = model.lm_head.register_forward_hook(lambda *args: head_calls.append(True))
    try:
        hidden = model(tokens, skip_lm_head=True, return_dict=return_dict)
    finally:
        handle.remove()
    assert not head_calls
    assert hidden.shape == (3, config.hidden_size)
    torch.testing.assert_close(model.lm_head(hidden), logits, rtol=0, atol=0)
