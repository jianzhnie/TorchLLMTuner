"""Native DeepSeek V4 construction and random-training contracts."""

import torch
from transformers import AutoConfig

from llmtuner.models.hf.materialization import (
    initialize_random_hf_model,
    materialize_meta_model,
)
from llmtuner.models.hf.model import HFTransformerModel


def small_config():
    return AutoConfig.for_model(
        "deepseek_v4",
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


def test_v4_meta_initialization_and_forward_backward():
    config = small_config()
    config.attn_mask_type = "causal"
    with torch.device("meta"):
        model = HFTransformerModel(config)
    materialize_meta_model(model, torch.device("cpu"))
    initialize_random_hf_model(model.model)
    hash_router = model.model.model.layers[0].mlp.gate
    table = hash_router.tid2eid
    assert table.shape == (64, 2)
    assert torch.all((0 <= table) & (table < 4))
    assert torch.any(table != 0)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    tokens = torch.arange(8) % 64
    output = model(tokens, positions=torch.arange(8))
    assert output.shape == (8, 64)
    output.float().square().mean().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_v4_recipe_chat_template_preserves_supervision_boundary():
    from pathlib import Path

    from llmtuner.components.tokenizer.huggingface import HuggingFaceTokenizer
    from llmtuner.datasets.text.processors import require_token_prefix

    root = Path("/home/jianzhnie/llmtuner/hfhub/models/deepseek-ai/DeepSeek-V4-Flash")
    if not root.is_dir():
        import pytest

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
