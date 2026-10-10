"""Public device-transfer imports and nested output validation."""

import pytest
import torch

from llmtuner.accelerator import cast_data_device, get_data_device
from llmtuner.accelerator.dist_utils import (
    cast_data_device as legacy_cast_data_device,
)
from llmtuner.accelerator.tensor_transfer import (
    cast_data_device as transfer_cast_data_device,
)


def test_transfer_imports_share_one_implementation() -> None:
    assert cast_data_device is legacy_cast_data_device is transfer_cast_data_device
    values = {"tokens": [torch.tensor([1, 2])]}
    assert get_data_device(values) == torch.device("cpu")
    assert cast_data_device(values, torch.device("cpu")) == values


def test_output_type_and_sequence_length_are_checked() -> None:
    tensor = torch.tensor([1])
    with pytest.raises(TypeError, match="out is <class 'list'>"):
        cast_data_device(tensor, torch.device("cpu"), out=[tensor])
    with pytest.raises(ValueError, match=r"zip\(\) argument 2 is shorter"):
        cast_data_device([tensor, tensor], torch.device("cpu"), out=[tensor])
