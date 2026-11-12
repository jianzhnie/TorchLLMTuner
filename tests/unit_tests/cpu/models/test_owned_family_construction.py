"""Explicit family construction preserves initialization and training APIs."""

from copy import deepcopy
from functools import partial

import pytest
import torch
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4ForCausalLM as ReferenceDeepseek,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM as ReferenceQwen,
)

from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.configuration import resolve_model_class
from llmtuner.models.deepseek.deepseek_v4.modeling import DeepseekV4ForCausalLM
from llmtuner.models.kimi.kimi_k3.configuration import KimiLinearConfig
from llmtuner.models.kimi.kimi_k3.model import KimiLinearForCausalLM
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.models.qwen.qwen3_5_moe.model import Qwen3_5MoeForCausalLM
from llmtuner.models.registry import build_registered_model
from tests.unit_tests.cpu.models.test_qwen3_model_contract import model_config


def small_config():
    return ReferenceDeepseek.config_class(
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


def text_config(config_type, *, dynamic=False):
    rope = {
        "rope_type": "dynamic" if dynamic else "default",
        "rope_theta": 10_000_000.0,
        "partial_rotary_factor": 0.5,
        "mrope_section": [11, 11, 10],
    }
    if dynamic:
        rope["factor"] = 2.0
    return config_type(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        layer_types=["linear_attention"],
        rope_parameters=rope,
    )


@pytest.mark.parametrize("family", ["deepseek", "qwen35", "qwen3", "kimi"])
@pytest.mark.parametrize("flat", [False, True])
def test_padding_mask_preserves_batch_dimensions_and_keep_mask_semantics(family, flat):
    if family == "deepseek":
        model = DeepseekV4ForCausalLM(small_config())
    elif family == "qwen35":
        config = text_config(ReferenceQwen.config_class)
        config.layer_types = ["full_attention"]
        model = Qwen3_5MoeForCausalLM(config)
    elif family == "kimi":
        model = KimiLinearForCausalLM(
            KimiLinearConfig(
                vocab_size=32,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                linear_attn_config={
                    "kda_layers": [1],
                    "full_attn_layers": [],
                    "head_dim": 4,
                    "num_heads": 2,
                    "short_conv_kernel_size": 3,
                },
            )
        )
    else:
        model = Qwen3MoeForCausalLM(model_config())
    model.eval()
    tokens = torch.tensor([[1, 3, 2, 7], [5, 4, 6, 2]])
    padding = torch.tensor([[False, False, False, True], [False, False, True, True]])
    if flat:
        tokens, padding = tokens[0], padding[0]
    keep = (~padding).long()
    if flat:
        keep = keep.unsqueeze(0)
    with torch.no_grad():
        expected = model(tokens, attention_masks=keep).logits
        actual = model(tokens, padding_mask=padding).logits
    assert actual.shape == (*tokens.shape, model.config.vocab_size)
    assert torch.isfinite(actual).all()
    assert actual.count_nonzero()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    head_calls = []
    handle = model.lm_head.register_forward_pre_hook(
        lambda module, args: head_calls.append(True)
    )
    try:
        hidden = model(tokens, padding_mask=padding, skip_lm_head=True)
    finally:
        handle.remove()
    assert head_calls == []
    assert hidden.shape == (*tokens.shape, model.config.hidden_size)
    torch.testing.assert_close(model.lm_head(hidden), actual, atol=0, rtol=0)
    hidden.square().mean().backward()
    assert model.tok_embeddings.weight.grad is not None
    assert model.tok_embeddings.weight.grad.count_nonzero()


@pytest.mark.parametrize("family", ["deepseek", "qwen35"])
def test_owned_family_binds_declared_structure(family):
    config = (
        small_config()
        if family == "deepseek"
        else text_config(ReferenceQwen.config_class)
    )
    model = build_registered_model(config, explicit_only=True)
    bound = model.decoder_structure.bind(model)
    assert model.tok_embeddings is bound.embedding
    assert model.layers is bound.blocks
    assert model.norm is bound.norm
    assert model.lm_head is bound.head
    assert not any(key.startswith("_decoder.") for key in model.state_dict())


@pytest.mark.parametrize("family", ["deepseek", "qwen35"])
@pytest.mark.parametrize("expert_backend", [None, "eager"])
def test_explicit_family_preserves_initialization_forward_and_checkpoint(
    family, expert_backend
):
    if family == "deepseek":
        config = small_config()
        reference_type, owned_type = ReferenceDeepseek, DeepseekV4ForCausalLM
    else:
        config = text_config(ReferenceQwen.config_class)
        config.num_hidden_layers = 2
        config.layer_types = ["linear_attention", "full_attention"]
        reference_type, owned_type = ReferenceQwen, Qwen3_5MoeForCausalLM
    config._attn_implementation = "eager"
    config._experts_implementation = expert_backend
    assert resolve_model_class(config) is owned_type
    torch.manual_seed(91)
    reference = reference_type(deepcopy(config))
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(91)
    constructor = (
        partial(build_registered_model, explicit_only=True)
        if family == "deepseek"
        else owned_type
    )
    owned = constructor(deepcopy(config))
    assert type(owned) is owned_type
    assert (
        owned.config._experts_implementation == reference.config._experts_implementation
    )
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert type(owned.model) is not type(reference.model)
    assert type(owned.model.layers[0]) is not type(reference.model.layers[0])
    assert owned.model.layers[0].mlp.router_logits_in_output is False
    assert reference.state_dict().keys() == owned.state_dict().keys()
    for name, value in reference.state_dict().items():
        assert torch.equal(value, owned.state_dict()[name]), name
    tokens = torch.tensor([[1, 3, 2, 7, 4, 6, 8, 5]])
    outputs = []
    for model in (reference, owned):
        result = model(input_ids=tokens, labels=tokens, use_cache=False)
        assert torch.isfinite(result.loss) and result.loss.item() > 0
        result.loss.backward()
        outputs.append(result)
    assert torch.equal(outputs[0].logits, outputs[1].logits)
    assert torch.equal(outputs[0].loss, outputs[1].loss)
    gradients = 0
    for (name, expected), (_, actual) in zip(
        reference.named_parameters(), owned.named_parameters(), strict=True
    ):
        assert (expected.grad is None) == (actual.grad is None), name
        if expected.grad is not None:
            gradients += int(bool(expected.grad.count_nonzero()))
            assert torch.equal(expected.grad, actual.grad), name
    assert gradients > 0
    restored = constructor(deepcopy(config))
    restored.load_state_dict(owned.state_dict(), strict=True)
    assert torch.equal(
        restored(input_ids=tokens, use_cache=False).logits, outputs[1].logits
    )


@pytest.mark.parametrize("family", ["deepseek", "qwen35"])
@pytest.mark.parametrize("tied", [False, True])
def test_owned_family_declares_moe_slots_and_actual_weight_identity(family, tied):
    config = (
        small_config()
        if family == "deepseek"
        else text_config(ReferenceQwen.config_class)
    )
    config.tie_word_embeddings = tied
    constructor = (
        DeepseekV4ForCausalLM if family == "deepseek" else Qwen3_5MoeForCausalLM
    )
    model = constructor(config)
    assert model.enable_weight_tying is tied
    bound = model.decoder_structure.bind(model)
    for layer in bound.blocks:
        assert layer.moe_block_name == "mlp"
        assert layer._modules[layer.moe_block_name] is layer.mlp
    # The contract follows live Parameter identity, including manual untie/retie.
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    assert model.enable_weight_tying is False
    model.lm_head.weight = model.model.embed_tokens.weight
    assert model.enable_weight_tying is True
    model.lm_head = None
    assert model.enable_weight_tying is False
    model.lm_head = bound.head
    model.model.embed_tokens = torch.nn.Identity()
    assert model.enable_weight_tying is False


@pytest.mark.parametrize("backend", ["flex", "eager_flex"])
@pytest.mark.parametrize("family", ["qwen3", "qwen35", "deepseek", "kimi"])
def test_removed_attention_adapter_rejects_before_parameter_allocation(
    monkeypatch, backend, family
):
    if family == "qwen3":
        config, constructor = model_config(), Qwen3MoeForCausalLM
    elif family == "qwen35":
        config = text_config(ReferenceQwen.config_class)
        constructor = Qwen3_5MoeForCausalLM
    elif family == "deepseek":
        config, constructor = small_config(), DeepseekV4ForCausalLM
    else:
        config, constructor = KimiLinearConfig(), KimiLinearForCausalLM
    config.attention_backend = backend
    rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported attention backend reached parameter allocation")

    monkeypatch.setattr(torch.nn.Module, "register_parameter", forbidden)
    with pytest.raises(UnsupportedCombinationError, match="no explicit-model adapter"):
        constructor(config)
    assert torch.equal(rng, torch.get_rng_state())
