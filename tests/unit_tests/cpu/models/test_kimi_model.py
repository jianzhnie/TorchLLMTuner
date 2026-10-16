"""Local Kimi factory and wrapper forward/backward integration."""

import torch

from llmtuner.models.hf.factory import build_model_config
from llmtuner.models.hf.model import HFTransformerModel


def test_kimi_dense_kda_wrapper_training_and_state_roundtrip():
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
    model = HFTransformerModel(config)
    tokens, positions = torch.tensor([1, 2, 3, 4]), torch.arange(4)
    output = model(tokens, positions=positions)
    assert output.shape == (4, 32)
    output.square().sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    restored = HFTransformerModel(config)
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(tokens, positions=positions), output)


def test_kimi_hybrid_padding_matches_unpadded_valid_tokens():
    from llmtuner.models.hf.factory import resolve_model_class

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


def test_kimi_wrapper_padding_matches_compacted_sequence():
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
    model = HFTransformerModel(config)
    tokens = torch.tensor([0, 1, 0, 2, 3])
    padding = torch.tensor([True, False, True, False, False])
    positions = torch.tensor([0, 0, 0, 1, 2])
    actual = model(tokens, positions=positions, padding_mask=padding)
    expected = model(tokens[~padding], positions=torch.arange(3))
    torch.testing.assert_close(actual[~padding], expected, atol=1e-6, rtol=1e-6)


def test_kimi_decoder_rejects_cache_instead_of_silently_ignoring_it():
    import pytest

    from llmtuner.models.kimi.kimi_linear_config import KimiLinearConfig
    from llmtuner.models.kimi.model import KimiTextDecoder

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
