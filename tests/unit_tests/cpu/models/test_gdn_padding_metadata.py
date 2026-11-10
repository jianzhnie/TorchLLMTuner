"""Single-row padding contract, independent of production GDN/backend imports."""

import pytest
import torch

from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.qwen.qwen3_5.padding_metadata import validate_gdn_padding_metadata


def check(**kwargs):
    return validate_gdn_padding_metadata(batch_size=1, sequence_length=128, **kwargs)


def test_defaults_preserve_no_padding_contract():
    check()
    check(padding_mask=torch.zeros(128, dtype=torch.bool))
    with pytest.raises(UnsupportedCombinationError):
        check(attention_mask=torch.ones(1, 128, dtype=torch.bool))


@pytest.mark.parametrize("valid", [1, 93, 128])
def test_right_padding_polarity_and_no_mutation(valid):
    keep = (torch.arange(128) < valid)[None, :]
    padding = ~keep
    original = keep.clone()
    check(
        attention_mask=keep,
        padding_mask=padding,
        positions=torch.arange(128)[None, :],
        allow_right_padding=True,
    )
    assert torch.equal(keep, original)
    if valid < 128:
        with pytest.raises(UnsupportedCombinationError):
            check(padding_mask=padding)


@pytest.mark.parametrize(
    "mask",
    [
        torch.zeros(1, 128, dtype=torch.bool),
        torch.arange(128)[None, :] >= 35,
        (torch.arange(128)[None, :] != 50),
        torch.ones(128, dtype=torch.bool),
        torch.ones(1, 128),
        torch.full((1, 128), 2, dtype=torch.long),
    ],
)
def test_empty_left_hole_and_malformed_masks_rejected(mask):
    with pytest.raises(UnsupportedCombinationError):
        check(attention_mask=mask, allow_right_padding=True)


def test_b2_and_position_resets_rejected():
    with pytest.raises(UnsupportedCombinationError):
        validate_gdn_padding_metadata(
            batch_size=2, sequence_length=128, allow_right_padding=True
        )
    positions = torch.arange(128)
    positions[93:] = torch.arange(35)
    with pytest.raises(UnsupportedCombinationError):
        check(positions=positions, allow_right_padding=True)


def test_metadata_disagreement_rejected():
    with pytest.raises(UnsupportedCombinationError):
        check(
            attention_mask=torch.ones(1, 128, dtype=torch.bool),
            padding_mask=torch.arange(128) >= 93,
            allow_right_padding=True,
        )


@pytest.mark.parametrize("metadata", [{"deltanet": None}, object()])
def test_structured_metadata_not_treated_as_hf_mask(metadata):
    with pytest.raises(UnsupportedCombinationError):
        check(attention_mask=metadata, allow_right_padding=True)
