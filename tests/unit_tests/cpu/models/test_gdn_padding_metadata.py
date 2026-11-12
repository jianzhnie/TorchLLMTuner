"""Native Qwen3.5 token-mask semantics, without the removed GDN wrapper.

The old custom backend restricted batches to one right-padded row. Native GDN
supports multiple rows and other token masks; compare its actual math instead
of retaining restrictions belonging to that removed implementation.
"""

from types import SimpleNamespace

import pytest
import torch

from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.qwen.qwen3_5_moe.model import validate_recurrent_mask
from tests.unit_tests.cpu.models.test_qwen35_owned_decoder import models


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.float32])
@pytest.mark.parametrize("pattern", ["full", "right", "left", "hole", "empty"])
def test_native_token_masks_match_reference_outputs_and_gradients(
    batch_size, dtype, pattern
):
    baseline, owned = models()
    baseline.eval()
    owned.eval()
    tokens = torch.tensor([[1, 5, 2, 7], [3, 6, 4, 8]])[:batch_size]
    keep = {
        "full": [1, 1, 1, 1],
        "right": [1, 1, 1, 0],
        "left": [0, 1, 1, 1],
        "hole": [1, 0, 1, 1],
        "empty": [0, 0, 0, 0],
    }[pattern]
    mask = torch.tensor([keep] * batch_size, dtype=dtype)
    if batch_size == 2:
        mask[1] = torch.tensor([1, 1, 0, 0], dtype=dtype)
    original = mask.clone()
    expected = baseline(tokens, attention_mask=mask, use_cache=False).last_hidden_state
    actual = owned(tokens, attention_mask=mask, use_cache=False).last_hidden_state
    assert actual.isfinite().all() and actual.count_nonzero() > 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    probe = torch.randn_like(actual)
    actual.backward(probe)
    expected.backward(probe)
    references = dict(baseline.named_parameters())
    nonzero = 0
    for name, parameter in owned.named_parameters():
        torch.testing.assert_close(
            parameter.grad, references[name].grad, rtol=0, atol=0
        )
        if parameter.grad is not None:
            assert parameter.grad.isfinite().all(), name
            nonzero += bool(parameter.grad.count_nonzero())
    assert nonzero > 0
    assert torch.equal(mask, original)


@pytest.mark.parametrize("metadata", [{"deltanet": None}, object(), [1, 1, 0, 0]])
def test_structured_metadata_is_not_a_native_token_mask(metadata):
    config = SimpleNamespace(layer_types=["linear_attention"])
    with pytest.raises(UnsupportedCombinationError, match="token keep mask"):
        validate_recurrent_mask(config, metadata)
