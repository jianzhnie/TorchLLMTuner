"""Explicit Kimi construction, input semantics and initialization lifecycle."""

import pytest
import torch

from llmtuner.config import LLMTunerConfig, ModelConfig, ParallelConfig
from llmtuner.errors import UnsupportedCombinationError
from llmtuner.memory.materialization import materialize_meta_model
from llmtuner.models.configuration import (
    build_model_config,
    build_model_config_for,
    resolve_model_class,
)
from llmtuner.models.kimi.kimi_k3.configuration import KimiLinearConfig
from llmtuner.models.kimi.kimi_k3.delta_attention import KimiDeltaAttention
from llmtuner.models.kimi.kimi_k3.model import KimiLinearForCausalLM, KimiTextDecoder
from llmtuner.models.kimi.kimi_k3.moe import KimiMoEGate
from llmtuner.models.kimi.kimi_k3.normalization import KimiRMSNorm
from llmtuner.parallel.expert_parallel.model_replacement import swap_moe_blocks
from llmtuner.trainer.runtime.assembly import build_model


def test_kimi_router_rejects_unknown_scoring_before_allocation(monkeypatch):
    config = KimiLinearConfig(hidden_size=8, num_experts=4, num_experts_per_token=2)
    config.moe_router_activation_func = "unknown"
    rng = torch.get_rng_state().clone()

    def reject(*args, **kwargs):
        raise AssertionError("Invalid scoring must fail before weight allocation")

    monkeypatch.setattr(torch, "empty", reject)
    with pytest.raises(ValueError, match="moe_router_activation_func"):
        KimiMoEGate(config)
    assert torch.equal(rng, torch.get_rng_state())


def test_kimi_binds_declared_structure():
    config = KimiLinearConfig(
        vocab_size=16,
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
    wrapper = KimiLinearForCausalLM(config)
    bound = wrapper.decoder_structure.bind(wrapper)
    assert wrapper.layers is bound.blocks
    assert wrapper.tok_embeddings is bound.embedding
    assert wrapper.lm_head is bound.head
    assert wrapper.enable_weight_tying is (bound.head.weight is bound.embedding.weight)
    bound.head.weight = bound.embedding.weight
    assert wrapper.enable_weight_tying is True
    wrapper.lm_head = None
    assert wrapper.enable_weight_tying is False
    wrapper.lm_head = bound.head
    wrapper.model.embed_tokens = torch.nn.Identity()
    assert wrapper.enable_weight_tying is False


def test_kimi_decoder_rejects_batched_document_resets_and_conflicting_masks():

    config = KimiLinearConfig(
        vocab_size=16,
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
    model = KimiLinearForCausalLM(config)
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    with pytest.raises(ValueError, match="single batch row"):
        model(tokens, position_ids=torch.tensor([[0, 1, 0, 1], [0, 1, 2, 3]]))
    with pytest.raises(ValueError, match="disagree"):
        model(
            tokens,
            attention_mask=torch.ones_like(tokens),
            valid_token_mask=torch.zeros_like(tokens, dtype=torch.bool),
        )
    # Position resets in the explicitly masked suffix are padding, not docs.
    output = model(
        tokens,
        position_ids=torch.tensor([[0, 1, 0, 0], [0, 1, 2, 3]]),
        attention_mask=torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]]),
    )
    assert torch.isfinite(output.logits).all()


def test_registered_kimi_architecture_builds_from_training_config():

    config = build_model_config_for(
        LLMTunerConfig(model=ModelConfig(model_name_or_path="kimi_linear"))
    )

    assert isinstance(config, KimiLinearConfig)
    assert resolve_model_class(config) is KimiLinearForCausalLM


def test_kimi_dense_kda_training_and_state_roundtrip():
    config = build_model_config(
        "kimi_linear",
        seq_len=8,
        arch_overrides={
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "linear_attn_config": {
                "kda_layers": [1, 2],
                "full_attn_layers": [],
                "head_dim": 4,
                "num_heads": 2,
                "short_conv_kernel_size": 3,
            },
        },
    )
    config.attn_mask_type = "causal"
    model = KimiLinearForCausalLM(config)
    tokens, positions = torch.tensor([1, 2, 3, 4]), torch.arange(4)
    output = model(tokens, positions=positions).logits
    assert output.shape == (4, 32)
    output.square().sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    restored = KimiLinearForCausalLM(config)
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(tokens, positions=positions).logits, output)


def test_kimi_hybrid_padding_matches_unpadded_valid_tokens():

    config = build_model_config(
        "kimi_linear",
        seq_len=8,
        arch_overrides={
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 3,
            "qk_rope_head_dim": 1,
            "v_head_dim": 3,
            "mla_use_nope": True,
            "linear_attn_config": {
                "kda_layers": [1],
                "full_attn_layers": [2],
                "head_dim": 4,
                "num_heads": 2,
                "short_conv_kernel_size": 3,
            },
        },
    )
    config._attn_implementation = "sdpa"
    model = resolve_model_class(config)(config)
    tokens = torch.tensor([[0, 1, 0, 2, 3]])
    valid = torch.tensor([[0, 1, 0, 1, 1]])
    padded = model(
        input_ids=tokens,
        attention_mask=valid,
        position_ids=torch.tensor([[0, 0, 0, 1, 2]]),
    ).logits
    plain = model(input_ids=tokens[:, valid[0].bool()]).logits
    torch.testing.assert_close(padded[:, valid[0].bool()], plain, atol=1e-6, rtol=1e-6)
    padded.square().sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_kimi_padding_matches_compacted_sequence():
    config = build_model_config(
        "kimi_linear",
        seq_len=8,
        arch_overrides={
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "linear_attn_config": {
                "kda_layers": [1],
                "full_attn_layers": [],
                "head_dim": 4,
                "num_heads": 2,
                "short_conv_kernel_size": 3,
            },
        },
    )
    config.attn_mask_type = "causal"
    model = KimiLinearForCausalLM(config)
    tokens = torch.tensor([0, 1, 0, 2, 3])
    padding = torch.tensor([True, False, True, False, False])
    positions = torch.tensor([0, 0, 0, 1, 2])
    actual = model(tokens, positions=positions, padding_mask=padding).logits
    expected = model(tokens[~padding], positions=torch.arange(3)).logits
    torch.testing.assert_close(actual[~padding], expected, atol=1e-6, rtol=1e-6)


def test_kimi_decoder_rejects_cache_instead_of_silently_ignoring_it():

    config = KimiLinearConfig(
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [],
            "head_dim": 4,
            "num_heads": 2,
            "short_conv_kernel_size": 3,
        },
    )
    decoder = KimiTextDecoder(config)
    with pytest.raises(ValueError, match="cache"):
        decoder(input_ids=torch.tensor([[1, 2]]), past_key_values=object())
    with pytest.raises(ValueError, match="position_ids"):
        decoder(input_ids=torch.tensor([[1, 2]]), position_ids=torch.arange(3)[None])


def test_kimi_moe_layer_selection_matches_configured_dense_prefix_and_frequency():

    config = KimiLinearConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_experts=4,
        num_experts_per_token=2,
        moe_intermediate_size=12,
        first_k_dense_replace=1,
        moe_layer_freq=2,
        linear_attn_config={
            "kda_layers": [1, 2, 3, 4],
            "full_attn_layers": [],
            "head_dim": 4,
            "num_heads": 2,
            "short_conv_kernel_size": 3,
        },
    )
    model = KimiTextDecoder(config)
    assert [layer.moe_block_name for layer in model.layers] == [
        None,
        None,
        "block_sparse_moe",
        None,
    ]
    assert [hasattr(layer, "block_sparse_moe") for layer in model.layers] == [
        False,
        False,
        True,
        False,
    ]


def test_kimi_meta_materialization_initializes_custom_parameters():

    config = build_model_config(
        "kimi_linear",
        seq_len=8,
        arch_overrides={
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_experts": 4,
            "num_experts_per_token": 2,
            "moe_intermediate_size": 6,
            "linear_attn_config": {
                "kda_layers": [1],
                "full_attn_layers": [],
                "head_dim": 4,
                "num_heads": 2,
                "short_conv_kernel_size": 3,
            },
        },
    )
    config.attn_mask_type = "causal"
    with torch.device("meta"):
        model = KimiLinearForCausalLM(config)
    materialize_meta_model(
        model,
        torch.device("cpu"),
        runtime_buffers=model.materialization_buffers(torch.device("cpu")),
    )
    model.initialize_parameters()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert all(
        torch.count_nonzero(module.weight) == module.weight.numel()
        for module in model.modules()
        if isinstance(module, KimiRMSNorm)
    )
    gate = next(module for module in model.modules() if isinstance(module, KimiMoEGate))
    assert torch.count_nonzero(gate.weight) > 0
    assert torch.count_nonzero(gate.e_score_correction_bias) == 0
    attention = next(
        module for module in model.modules() if isinstance(module, KimiDeltaAttention)
    )
    assert torch.all(attention.A_log <= torch.log(torch.tensor(16.0)))
    assert torch.all(attention.A_log >= 0)
    assert torch.count_nonzero(attention.dt_bias) == 0


def test_converted_kimi_initialization_rejects_before_mutating_weights_or_rng():
    config = KimiLinearConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_experts=4,
        num_experts_per_token=2,
        moe_intermediate_size=6,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [],
            "head_dim": 4,
            "num_heads": 2,
            "short_conv_kernel_size": 3,
        },
    )
    model = KimiLinearForCausalLM(config)
    assert swap_moe_blocks(model) == 1
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    with pytest.raises(TypeError, match="ownership-aware initialization"):
        model.initialize_parameters()
    assert torch.equal(rng, torch.get_rng_state())
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)


def test_kimi_ep_meta_random_initialization_rejects_before_model_allocation(
    monkeypatch,
):
    cfg = LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="kimi_linear", initialization_device="meta"
        ),
        parallel=ParallelConfig(expert_parallel_size=2),
    )

    def reject(*args, **kwargs):
        raise AssertionError("Unsupported initialization must fail before allocation")

    monkeypatch.setattr("llmtuner.trainer.runtime.assembly.resolve_model_class", reject)
    with pytest.raises(UnsupportedCombinationError, match="ownership-aware"):
        build_model(cfg, parallel_dims=None, device=torch.device("cpu"))


def test_kda_decay_parameters_remain_fp32_with_bf16_default():

    config = KimiLinearConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [],
            "head_dim": 4,
            "num_heads": 256,
            "short_conv_kernel_size": 3,
        },
    )
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            model = KimiLinearForCausalLM(config)
        model.to_empty(device="cpu")
        torch.manual_seed(42)
        model.initialize_parameters()
    finally:
        torch.set_default_dtype(previous)
    attention = next(m for m in model.modules() if isinstance(m, KimiDeltaAttention))
    assert attention.q_proj.weight.dtype == torch.bfloat16
    assert attention.A_log.dtype == attention.dt_bias.dtype == torch.float32
    assert torch.all(
        (attention.A_log >= 0) & (attention.A_log <= torch.log(torch.tensor(16.0)))
    )
    assert torch.count_nonzero(attention.dt_bias) == 0
