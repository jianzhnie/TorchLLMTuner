"""Packed vision reference math must respect both Q and KV document boundaries."""

import pytest
import torch

from llmtuner.models.kimi.kimi_k3.multimodal import eager_attention


def reference(q, k, v, q_offsets, k_offsets):
    outputs = []
    for qs, qe, ks, ke in zip(
        q_offsets[:-1], q_offsets[1:], k_offsets[:-1], k_offsets[1:], strict=True
    ):
        query, key, value = (x.transpose(0, 1) for x in (q[qs:qe], k[ks:ke], v[ks:ke]))
        scores = query @ key.transpose(-2, -1) / q.shape[-1] ** 0.5
        weights = scores.softmax(-1, dtype=torch.float32).to(q.dtype)
        outputs.append((weights @ value).transpose(0, 1))
    return torch.cat(outputs).flatten(-2)


@pytest.mark.parametrize(
    "q_offsets,k_offsets",
    [([0, 2, 5], [0, 4, 7]), ([0, 2, 5], [0, 2, 5]), ([0, 0, 5], [0, 2, 7])],
)
@pytest.mark.parametrize(
    "dtype,atol,rtol",
    [
        (torch.float32, 1e-6, 1e-5),
        (torch.float16, 2e-3, 2e-3),
        (torch.bfloat16, 2e-2, 2e-2),
    ],
)
def test_paired_documents_match_independent_output_and_gradients(
    q_offsets, k_offsets, dtype, atol, rtol
):
    generator = torch.Generator().manual_seed(37)
    values = [
        torch.randn(n, 2, 4, generator=generator, dtype=dtype, requires_grad=True)
        for n in (q_offsets[-1], k_offsets[-1], k_offsets[-1])
    ]
    baseline = [x.detach().clone().requires_grad_() for x in values]
    actual = eager_attention(*values, torch.tensor(q_offsets), torch.tensor(k_offsets))
    expected = reference(*baseline, q_offsets, k_offsets)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    upstream = torch.randn(actual.shape, generator=generator, dtype=dtype)
    actual.backward(upstream)
    expected.backward(upstream)
    for value, other in zip(values, baseline, strict=True):
        assert value.grad is not None and value.grad.count_nonzero()
        torch.testing.assert_close(value.grad, other.grad, atol=atol, rtol=rtol)


def test_unspecified_offsets_mean_one_document():
    values = torch.randn(3, 2, 4)
    torch.testing.assert_close(
        eager_attention(values, values, values),
        reference(values, values, values, [0, 3], [0, 3]),
    )


def test_nonempty_queries_cannot_attend_empty_document():
    q = torch.randn(3, 2, 4)
    kv = torch.randn(2, 2, 4)
    with pytest.raises(ValueError, match="nonempty keys"):
        eager_attention(q, kv, kv, torch.tensor([0, 1, 3]), torch.tensor([0, 0, 2]))


@pytest.mark.parametrize("key_tokens", [0, 3])
def test_empty_query_keeps_all_input_gradient_paths(key_tokens):
    q = torch.empty(0, 2, 4, requires_grad=True)
    k = torch.randn(key_tokens, 2, 4, requires_grad=True)
    v = torch.randn(key_tokens, 2, 4, requires_grad=True)
    output = eager_attention(q, k, v)
    assert output.shape == (0, 8)
    output.sum().backward()
    for value in (q, k, v):
        assert value.grad is not None
        assert value.grad.shape == value.shape
        assert not value.grad.count_nonzero()


def test_packed_documents_do_not_leak_values_or_gradients():
    generator = torch.Generator().manual_seed(74)
    q, k, v = [
        torch.randn(5, 2, 4, generator=generator, requires_grad=True) for _ in range(3)
    ]
    offsets = torch.tensor([0, 2, 5])
    output = eager_attention(q, k, v, offsets, offsets)
    changed = v.detach().clone()
    changed[2:] += 100
    other = eager_attention(q, k, changed, offsets, offsets)
    torch.testing.assert_close(output[:2], other[:2], atol=0, rtol=0)
    output[:2].square().sum().backward()
    for value in (q, k, v):
        assert value.grad[:2].count_nonzero()
        assert not value.grad[2:].count_nonzero()


def test_host_boundaries_include_leading_zero_and_preserve_documents():
    generator = torch.Generator().manual_seed(18)
    values = [torch.randn(5, 2, 4, generator=generator) for _ in range(3)]
    offsets = [0, 2, 5]
    torch.testing.assert_close(
        eager_attention(*values, offsets, offsets),
        reference(*values, offsets, offsets),
        atol=1e-6,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"num_heads": 0}, "positive integer"),
        ({"num_heads": True}, "positive integer"),
        ({"hidden_dim": -1}, "positive integer"),
        ({"mlp_dim": 1.5}, "positive integer"),
        ({"qkv_hidden_size": 7}, "divisible"),
        ({"attn_implementation": "missing"}, "backend"),
    ],
)
def test_vision_layer_rejects_bad_geometry_before_parameters(
    monkeypatch, overrides, match
):
    from llmtuner.models.kimi.kimi_k3.multimodal import MoonViTEncoderLayer

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid vision geometry reached parameter allocation")

    monkeypatch.setattr(torch.nn.LayerNorm, "__init__", unexpected)
    before = torch.get_rng_state().clone()
    arguments = dict(num_heads=2, hidden_dim=8, mlp_dim=16, attn_implementation="eager")
    with pytest.raises(ValueError, match=match):
        MoonViTEncoderLayer(**(arguments | overrides))
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("heads,width", [(0, 8), (True, 8), (2, 7), (2, 12), (2, 0)])
def test_encoder_rejects_invalid_rotary_geometry_before_blocks(
    monkeypatch, heads, width
):
    from llmtuner.models.kimi.kimi_k3 import multimodal

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid encoder geometry allocated a decoder block")

    monkeypatch.setattr(multimodal, "MoonViTEncoderLayer", unexpected)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="positive|divisible"):
        multimodal.MoonViT3dEncoder(
            hidden_dim=8,
            num_layers=1,
            block_cfg=dict(hidden_dim=8, num_heads=heads, qkv_hidden_size=width),
        )
    assert torch.equal(before, torch.get_rng_state())
