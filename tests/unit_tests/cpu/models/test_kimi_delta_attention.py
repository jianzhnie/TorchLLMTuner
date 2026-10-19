"""KDA production projection path and independent one-step recurrence."""

import pytest
import torch

from llmtuner.models.kimi.delta_attention import KimiDeltaAttention, recurrent_kda
from llmtuner.models.kimi.kimi_linear_config import KimiLinearConfig


@pytest.mark.parametrize("full_gate", [False, True])
def test_kimi_delta_attention_has_finite_complete_training_gradients(full_gate):
    config = KimiLinearConfig(
        hidden_size=8,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [],
            "head_dim": 4,
            "num_heads": 2,
            "short_conv_kernel_size": 3,
            "use_full_rank_gate": full_gate,
        },
    )
    model = KimiDeltaAttention(config, 0)
    inputs = torch.randn(1, 5, 8, requires_grad=True)
    output = model(inputs)
    output.square().sum().backward()
    assert output.shape == inputs.shape
    assert torch.isfinite(inputs.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_kimi_recurrence_first_token_has_no_history_or_gate_dependence():
    torch.manual_seed(5)
    q, k, v, g = [torch.randn(1, 1, 2, 3) for _ in range(4)]
    beta = torch.randn(1, 1, 2)
    actual = recurrent_kda(q, k, v, g, beta, torch.zeros(2), torch.zeros(6))
    inner = (q * k).sum(-1) / (
        (q.square().sum(-1) + 1e-6).sqrt() * (k.square().sum(-1) + 1e-6).sqrt()
    )
    expected = inner.unsqueeze(-1) * beta.sigmoid().unsqueeze(-1) * v / 3**0.5
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_kimi_packed_documents_do_not_share_convolution_or_delta_history():
    config = KimiLinearConfig(
        hidden_size=8,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [],
            "head_dim": 4,
            "num_heads": 2,
            "short_conv_kernel_size": 3,
        },
    )
    model = KimiDeltaAttention(config, 0)
    inputs = torch.randn(1, 6, 8, requires_grad=True)
    result = model(inputs, cu_seqlens=torch.tensor([0, 2, 6]))
    expected = torch.cat([model(inputs[:, :2]), model(inputs[:, 2:])], 1)
    torch.testing.assert_close(result, expected)
    result[:, 2:].sum().backward()
    torch.testing.assert_close(inputs.grad[:, :2], torch.zeros_like(inputs[:, :2]))


def test_kimi_padding_unpads_each_row_and_preserves_zero_gradients():
    config = KimiLinearConfig(
        hidden_size=8,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [],
            "head_dim": 4,
            "num_heads": 2,
            "short_conv_kernel_size": 3,
        },
    )
    model = KimiDeltaAttention(config, 0)
    inputs = torch.randn(2, 5, 8, requires_grad=True)
    mask = torch.tensor([[False, True, True, False, True], [False] * 5])
    result = model(inputs, attention_mask=mask)
    torch.testing.assert_close(result[0, mask[0]], model(inputs[:1, mask[0]])[0])
    torch.testing.assert_close(result[~mask], torch.zeros_like(result[~mask]))
    result.square().sum().backward()
    torch.testing.assert_close(inputs.grad[~mask], torch.zeros_like(inputs.grad[~mask]))
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("lower_bound", [None, -3.0])
def test_kimi_recurrence_matches_affine_state_transition_product(lower_bound):
    # Independent oracle: express each token as an affine matrix transform of
    # every value column, then multiply augmented transforms across time.
    torch.manual_seed(19)
    values = [torch.randn(1, 5, 2, 3, requires_grad=True) for _ in range(4)]
    values += [
        torch.randn(1, 5, 2, requires_grad=True),
        torch.randn(2, requires_grad=True),
        torch.randn(6, requires_grad=True),
    ]
    reference = [x.detach().clone().requires_grad_() for x in values]
    q, k, v, raw, beta, rate, bias = reference
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    raw = raw + bias.reshape(2, 3)
    decay_log = (
        -rate.exp()[None, None, :, None] * torch.nn.functional.softplus(raw)
        if lower_bound is None
        else lower_bound * (rate.exp()[None, None, :, None] * raw).sigmoid()
    )
    identity = torch.eye(3).expand(1, 2, 3, 3)
    prefix = torch.eye(4).expand(1, 2, 3, 4, 4)
    outputs = []
    for t in range(5):
        key, strength = k[:, t], beta[:, t].sigmoid()
        transition = (
            identity - strength[..., None, None] * key[..., :, None] * key[..., None, :]
        ) @ torch.diag_embed(decay_log[:, t].exp())
        write = strength[..., None, None] * key[..., :, None] * v[:, t, :, None, :]
        top = torch.cat(
            (
                transition[:, :, None].expand(-1, -1, 3, -1, -1),
                write.transpose(-1, -2).unsqueeze(-1),
            ),
            -1,
        )
        bottom = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(1, 2, 3, 1, 4)
        prefix = torch.cat((top, bottom), -2) @ prefix
        state = prefix[..., :3, 3].transpose(-1, -2)
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], state) / 3**0.5)
    expected = torch.stack(outputs, 1)
    actual = recurrent_kda(*values, lower_bound)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    upstream = torch.randn_like(actual)
    gradients = torch.autograd.grad(actual, values, upstream)
    oracle_gradients = torch.autograd.grad(expected, reference, upstream)
    for actual_grad, expected_grad in zip(gradients, oracle_gradients, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=3e-6, rtol=3e-6)
