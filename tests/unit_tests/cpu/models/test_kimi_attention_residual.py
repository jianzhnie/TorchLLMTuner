"""Attention residual scores and unnormalized values against token enumeration."""

from copy import deepcopy

import pytest
import torch

from llmtuner.models.kimi.attention_residual import attention_residual


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("history", [0, 3])
def test_attention_residual_matches_independent_per_token_reference(history, seed):
    torch.manual_seed(seed)
    tensors = [
        torch.randn(2, 4, requires_grad=True),
        torch.randn(2, history, 4, requires_grad=True),
        torch.randn(1, 4, requires_grad=True),
        torch.randn(4, requires_grad=True),
    ]
    reference = [t.detach().clone().requires_grad_() for t in tensors]
    prefix, blocks, projection, norm = reference
    rows = []
    for token in range(2):
        values = [*blocks[token].unbind(0), prefix[token]]
        scores = torch.stack(
            [
                ((v / (v.square().mean() + 1e-6).sqrt()) * norm * projection[0]).sum()
                for v in values
            ]
        )
        weights = scores.softmax(0)
        rows.append(sum(w * v for w, v in zip(weights, values, strict=True)))
    expected = torch.stack(rows) + blocks.sum() * 0
    actual = attention_residual(*tensors, 1e-6)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    upstream = torch.randn_like(actual)
    a = torch.autograd.grad(actual, tensors, upstream)
    b = torch.autograd.grad(expected, reference, upstream)
    for left, right in zip(a, b, strict=True):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("seed", range(32))
@pytest.mark.parametrize("block_size", [1, 2])
def test_kimi_attention_residual_model_runs_all_parameter_gradients(block_size, seed):
    from llmtuner.models.kimi.kimi_linear_config import KimiLinearConfig
    from llmtuner.models.kimi.model import KimiLinearForCausalLM

    config = KimiLinearConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=3,
        num_attention_heads=2,
        attn_res_block_size=block_size,
        linear_attn_config={
            "kda_layers": [1, 2, 3],
            "full_attn_layers": [],
            "head_dim": 4,
            "num_heads": 2,
            "short_conv_kernel_size": 3,
        },
    )
    torch.manual_seed(seed)
    model = KimiLinearForCausalLM(config)
    reference = deepcopy(model)
    tokens = torch.tensor([[1, 2, 3]])
    prefix = reference.model.embed_tokens(tokens)
    history = []

    def mix(candidates, projection, norm):
        # FSDPTurbo 0af4eca ops/cpu/attn_res.py FP32 reduction contract.
        # Stack the independently maintained history list. A per-token scalar
        # sum changes forward rounding and amplifies gradient differences through
        # repeated RMSNorms; its mathematical formula is checked separately above.
        values = torch.stack(candidates, dim=-2)
        shape = values.shape
        values = values.reshape(-1, shape[-2], shape[-1]).float()
        variance = values.pow(2).mean(-1, keepdim=True)
        normalized = values * torch.rsqrt(variance + norm.variance_epsilon)
        score_weight = norm.weight.float() * projection.weight.squeeze(0).float()
        scores = (normalized * score_weight).sum(-1)
        probabilities = scores.softmax(-1).unsqueeze(1)
        return (
            torch.matmul(probabilities, values)
            .squeeze(1)
            .reshape(*shape[:-2], shape[-1])
        )

    for index, layer in enumerate(reference.model.layers):
        attention_input = (
            mix(
                [*history, prefix],
                layer.self_attention_res_proj,
                layer.self_attention_res_norm,
            )
            if history
            else prefix
        )
        boundary = index % block_size == 0
        if boundary:
            history.append(prefix)
        attention_output = layer.self_attn(layer.input_layernorm(attention_input))
        prefix = attention_output if boundary else prefix + attention_output
        mlp_input = mix([*history, prefix], layer.mlp_res_proj, layer.mlp_res_norm)
        prefix = prefix + layer.mlp(layer.post_attention_layernorm(mlp_input))
    final = mix(
        [*history, prefix],
        reference.model.output_attn_res_proj,
        reference.model.output_attn_res_norm,
    )
    expected = reference.lm_head(reference.model.norm(final))
    output = model(input_ids=tokens)
    torch.testing.assert_close(output.logits, expected, atol=0, rtol=0)
    expected.square().sum().backward()
    output.logits.square().sum().backward()
    for (name, parameter), (_, other) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        if other.grad is None:
            assert parameter.grad is None, name
        else:
            torch.testing.assert_close(
                parameter.grad,
                other.grad,
                atol=2e-6,
                rtol=2e-6,
                msg=lambda detail, name=name: f"{name}: {detail}",
            )
    unused = {name for name, p in model.named_parameters() if p.grad is None}
    # The first attention sees only the initial embedding, so upstream skips
    # scoring an empty history; exactly these two weights are intentionally unused.
    assert unused == {
        "model.layers.0.self_attention_res_norm.weight",
        "model.layers.0.self_attention_res_proj.weight",
    }
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
