from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llmtuner.trainer.runtime.reference_norm_layout import (
    clip_grad_norm_,
    interleave_qkv,
    reference_norm_gradients,
)


def test_interleave_uses_kv_groups_not_naive_projection_concatenation():
    query = torch.arange(16).reshape(8, 2)
    key = torch.arange(100, 108).reshape(4, 2)
    value = torch.arange(200, 208).reshape(4, 2)
    actual = interleave_qkv(query, key, value, kv_heads=2, head_dim=2)
    expected = torch.cat((query[:4], key[:2], value[:2], query[4:], key[2:], value[2:]))
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, torch.cat((query, key, value)))


def test_expert_logical_norm_preserves_gradients_parameters_and_scaling():
    model = nn.Module()
    model.gate_up_proj = nn.Parameter(torch.zeros(3, 4, 2))
    model.down_proj = nn.Parameter(torch.zeros(3, 2, 2))
    torch.manual_seed(71)
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    parameters = list(model.parameters())
    originals = [p.grad.clone() for p in parameters]
    identities = [id(p) for p in parameters]
    logical = reference_norm_gradients(parameters, [model])
    expected = [originals[0][:, :2], originals[1], originals[0][:, 2:]]
    assert all(torch.equal(a, b) for a, b in zip(logical, expected, strict=True))
    norm = torch.nn.utils.get_total_norm(expected, foreach=False)
    observed = clip_grad_norm_(
        parameters,
        max_norm=1.0,
        norm_precision="parameter",
        reference_model_parts=[model],
    )
    assert torch.equal(norm, observed)
    coefficient = (1 / (norm + 1e-6)).clamp(max=1)
    assert [id(p) for p in parameters] == identities
    assert all(
        torch.equal(p.grad, g * coefficient)
        for p, g in zip(parameters, originals, strict=True)
    )


def test_qkv_rejects_unsharded_or_partial_projection_groups():
    model = nn.Module()
    for name in ("q_proj", "k_proj", "v_proj"):
        setattr(model, name, nn.Linear(2, 2, bias=False))
    model.config = SimpleNamespace(num_key_value_heads=1)
    model.head_dim = 2
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    with pytest.raises(NotImplementedError, match="Shard"):
        reference_norm_gradients(list(model.parameters()), [model])
