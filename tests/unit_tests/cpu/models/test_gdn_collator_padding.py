"""Collator → model-owned preparation → real GDN training, using CPU math.

Native padding/prefix VJP and AC replay are covered by
``test_qwen35_collator_padding``. This module checks the Trainer preparation
boundary and an optimizer update, without reviving removed wrapper guards.
"""

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from llmtuner.errors import UnsupportedCombinationError
from tests.unit_tests.cpu.models.test_qwen35_collator_padding import (
    actual_model,
    collated,
)


@pytest.mark.parametrize("valid", [2, 93, 128])
def test_prepared_collator_batch_matches_direct_training_update(valid):
    torch.manual_seed(71)
    model = actual_model()
    reference = deepcopy(model)
    batch = collated(valid)
    original = deepcopy(batch)
    inputs, labels, extra = model.prepare_inputs(batch, parallel_dims=None)
    assert torch.equal(inputs, batch["input"])
    assert torch.equal(labels, batch["labels"])
    assert batch["padding_mask"].sum().item() == 128 - valid
    assert batch["positions"][:valid].tolist() == list(range(valid))
    assert batch["positions"][valid:].tolist() == list(range(128 - valid))

    actual = model(inputs, **extra).logits
    expected = reference(
        batch["input"],
        positions=batch["positions"],
        padding_mask=batch["padding_mask"],
    ).logits
    assert actual.isfinite().all() and actual.count_nonzero() > 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    loss = F.cross_entropy(actual, labels)
    reference_loss = F.cross_entropy(expected, batch["labels"])
    assert loss.isfinite() and loss > 0
    torch.testing.assert_close(loss, reference_loss, rtol=0, atol=0)
    loss.backward()
    reference_loss.backward()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    reference_parameters = dict(reference.named_parameters())
    nonzero = 0
    for name, parameter in model.named_parameters():
        other = reference_parameters[name]
        torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
        if parameter.grad is not None:
            assert parameter.grad.isfinite().all(), name
            nonzero += bool(parameter.grad.count_nonzero())
    assert nonzero > 0
    for selected in (model, reference):
        torch.optim.AdamW(selected.parameters(), lr=1e-3, foreach=False).step()
    assert any(
        not torch.equal(parameter, before[name])
        for name, parameter in model.named_parameters()
    )
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter, reference_parameters[name], rtol=0, atol=0
        )
    for name, value in batch.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, original[name]), name
        else:
            assert value == original[name], name


def test_collator_packed_request_rejects_before_preparation(monkeypatch):
    model = actual_model()
    batch = collated()
    model.config.attn_mask_type = "block_causal"

    def forbidden(*args, **kwargs):
        raise AssertionError("Packed recurrent input reached generic preparation")

    monkeypatch.setattr(
        "llmtuner.models.qwen.qwen3_5_moe.model.prepare_text_inputs", forbidden
    )
    with pytest.raises(UnsupportedCombinationError, match="packed documents"):
        model.prepare_inputs(batch, parallel_dims=None)
