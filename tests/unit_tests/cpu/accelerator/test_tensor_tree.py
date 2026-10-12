"""Public device-transfer imports and nested output validation."""

import pytest
import torch

from llmtuner.accelerator import get_tensor_tree_device, move_tensor_tree
from llmtuner.accelerator.distributed import (
    move_tensor_tree as legacy_move_tensor_tree,
)
from llmtuner.accelerator.tensor_tree import (
    move_tensor_tree as transfer_move_tensor_tree,
)


def test_transfer_imports_share_one_implementation() -> None:
    assert move_tensor_tree is legacy_move_tensor_tree is transfer_move_tensor_tree
    values = {"tokens": [torch.tensor([1, 2])]}
    assert get_tensor_tree_device(values) == torch.device("cpu")
    assert move_tensor_tree(values, torch.device("cpu")) == values


def test_output_type_and_sequence_length_are_checked() -> None:
    tensor = torch.tensor([1])
    with pytest.raises(TypeError, match="out is <class 'list'>"):
        move_tensor_tree(tensor, torch.device("cpu"), out=[tensor])
    with pytest.raises(ValueError, match="container lengths"):
        move_tensor_tree([tensor, tensor], torch.device("cpu"), out=[tensor])


def test_namedtuple_and_defaultdict_preserve_structure() -> None:
    from collections import defaultdict, namedtuple

    Pair = namedtuple("Pair", "inputs labels")
    data = defaultdict(list, batch=Pair(torch.tensor([1]), torch.tensor([2])))
    moved = move_tensor_tree(data, torch.device("cpu"))
    assert isinstance(moved, defaultdict)
    assert moved.default_factory is list
    assert isinstance(moved["batch"], Pair)
    assert torch.equal(moved["batch"].labels, torch.tensor([2]))


def test_invalid_nested_output_does_not_modify_earlier_leaves() -> None:
    data = {"first": torch.tensor([1]), "nested": {"tokens": torch.tensor([2])}}
    out = {"first": torch.tensor([0]), "nested": {"wrong": torch.tensor([0])}}
    with pytest.raises(ValueError, match="mapping keys"):
        move_tensor_tree(data, torch.device("cpu"), out=out)
    assert out["first"].item() == 0


def test_mapping_output_matches_keys_independently_of_order() -> None:
    data = {"first": torch.tensor([1]), "second": [torch.tensor([2])]}
    out = {"second": [torch.tensor([0])], "first": torch.tensor([0])}
    move_tensor_tree(data, torch.device("cpu"), out=out)
    assert out["first"].item() == 1
    assert out["second"][0].item() == 2


@pytest.mark.parametrize("data", [b"text", "text", iter([torch.tensor([1])])])
def test_unsupported_iterables_raise_clear_errors(data) -> None:
    with pytest.raises(TypeError, match="tensor container"):
        get_tensor_tree_device(data)
    with pytest.raises(TypeError, match="tensor container"):
        move_tensor_tree(data, torch.device("cpu"))


def test_mixed_devices_and_output_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="consistent"):
        get_tensor_tree_device([torch.ones(1), torch.ones(1, device="meta")])
    with pytest.raises(ValueError, match="shapes"):
        move_tensor_tree(torch.ones(2), torch.device("cpu"), out=torch.zeros(1))
