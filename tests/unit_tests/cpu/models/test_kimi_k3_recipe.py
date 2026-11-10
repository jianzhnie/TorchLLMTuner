"""K3 example configuration and official tokenizer integration."""

from pathlib import Path

import pytest

from examples.kimi_k3.train import (
    build_argument_parser,
    build_config,
    load_text_overrides,
)
from llmtuner.errors import ConfigError

MODEL = Path("/home/jianzhnie/llmtuner/hfhub/models/moonshotai/Kimi-K3")


def test_k3_reduces_only_layers_and_attention_schedule():
    if not MODEL.is_dir():
        pytest.skip("Official K3 configuration is not available")
    original = load_text_overrides(str(MODEL))
    reduced = load_text_overrides(str(MODEL), 4)
    assert original["num_hidden_layers"] == 93
    assert reduced["num_hidden_layers"] == 4
    assert reduced["linear_attn_config"]["kda_layers"] == [1, 2, 3]
    assert reduced["linear_attn_config"]["full_attn_layers"] == [4]
    for key in ("hidden_size", "num_experts", "moe_intermediate_size"):
        assert reduced[key] == original[key]
    with pytest.raises(ConfigError, match="layer count"):
        load_text_overrides(str(MODEL), 0)


def test_k3_flops_cover_reduced_kda_mla_and_latent_moe():
    if not MODEL.is_dir():
        pytest.skip("Official K3 configuration is not available")
    from llmtuner.models.kimi.kimi_k3.configuration import KimiLinearConfig
    from llmtuner.utils.flops import flops_per_token

    one = KimiLinearConfig(**load_text_overrides(str(MODEL), 1))
    two = KimiLinearConfig(**load_text_overrides(str(MODEL), 2))
    four = KimiLinearConfig(**load_text_overrides(str(MODEL), 4))
    counts = [flops_per_token(config, seq_len=2048) for config in (one, two, four)]
    assert all(count > 0 for count in counts)
    assert counts[0] < counts[1] < counts[2]


def test_k3_recipe_uses_local_model_and_explicit_official_tokenizer():
    if not MODEL.is_dir():
        pytest.skip("Official K3 configuration is not available")
    args = build_argument_parser().parse_args(
        ["--model", str(MODEL), "--data", "limo.jsonl", "--output", "output"]
    )
    config = build_config(args)
    assert config.model.model_name_or_path == "kimi_linear"
    assert config.model.arch_overrides["num_hidden_layers"] == 93
    assert config.dataloader.tokenizer_backend == "kimi_k3"
    assert config.dataloader.chat_template_kwargs == {"thinking": False}
    assert config.optimizer.state_residency == "npu_virtual"


def test_k3_official_tokenizer_matches_segmented_chat_encoding():
    if not MODEL.is_dir():
        pytest.skip("Official K3 tokenizer is not available")
    pytest.importorskip("tiktoken")
    from llmtuner.components.tokenizer.kimi import KimiK3Tokenizer

    tokenizer = KimiK3Tokenizer(tokenizer_path=str(MODEL))
    messages = [{"role": "user", "content": "Explain 1+1."}]
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
    assert tokenizer.encode(rendered) == tokenizer.backend.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    assert tokenizer.get_vocab_size() == 163840
