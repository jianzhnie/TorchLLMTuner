"""Inspect and move tensor trees while preserving their container structure.

Leaves must be tensors. Containers may be mappings, lists, tuples (including
namedtuples), or sets. One-shot iterators and scalar metadata are unsupported.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from copy import copy
from typing import Any

import torch
from torch import Tensor


def _children(data: Any) -> list[Any]:
    """Read one container level using the same contract for every operation."""
    if isinstance(data, Mapping):
        children = list(data.values())
    elif isinstance(data, list | tuple | set):
        children = list(data)
    else:
        raise TypeError(f"Expected a tensor or tensor container, got {type(data)}")
    if not children:
        raise ValueError("data should not be empty")
    return children


def get_tensor_tree_device(data: Any) -> torch.device:
    """Return the common device, rejecting empty or mixed-device trees."""
    if isinstance(data, Tensor):
        return data.device
    devices = {get_tensor_tree_device(child) for child in _children(data)}
    if len(devices) != 1:
        raise ValueError(f"Tensor devices must be consistent, got {devices}")
    return devices.pop()


def _validate_output(data: Any, out: Any) -> None:
    """Check the entire output tree before copying any tensor into it."""
    if type(data) is not type(out):
        raise TypeError(
            "out should be the same type with data, but got data is "
            f"{type(data)} and out is {type(out)}"
        )
    if isinstance(data, Tensor):
        if data.shape != out.shape:
            raise ValueError("data and out tensor shapes must match")
        return
    if isinstance(data, set):
        raise TypeError("out should not be a set")
    children = _children(data)
    if isinstance(data, Mapping) and data.keys() != out.keys():
        raise ValueError("data and out mapping keys must match")
    outputs = (
        [out[key] for key in data] if isinstance(data, Mapping) else _children(out)
    )
    if len(children) != len(outputs):
        raise ValueError("data and out container lengths must match")
    for child, output in zip(children, outputs, strict=True):
        _validate_output(child, output)


def _move(data: Any, device: torch.device, out: Any = None) -> Any:
    if isinstance(data, Tensor):
        moved = data.to(device)
        if out is not None:
            out.copy_(moved)
        return moved
    children = _children(data)
    if out is None:
        outputs = [None] * len(children)
    elif isinstance(data, Mapping):
        outputs = [out[key] for key in data]
    else:
        outputs = _children(out)
    moved = [
        _move(child, device, output)
        for child, output in zip(children, outputs, strict=True)
    ]
    if isinstance(data, Mapping):
        items = dict(zip(data, moved, strict=True))
        if isinstance(data, MutableMapping):
            result = copy(data)
            # Keep existing keys and use the mapping's assignment protocol.
            # ModelOutput forbids update() and synchronizes attributes here.
            for key, value in items.items():
                result[key] = value
            return result
        return type(data)(items)
    if isinstance(data, tuple) and hasattr(data, "_fields"):
        return type(data)(*moved)
    return type(data)(moved)


def move_tensor_tree(data: Any, device: torch.device | str, *, out: Any = None) -> Any:
    """Move tensor leaves, preserving containers and optionally copying to out.

    ``out`` must have identical container types, keys, lengths, and tensor shapes.
    Structural errors are detected before any output tensors are modified.
    Returned tensors are on ``device``; output tensors retain their own devices.
    """
    if out is not None:
        _validate_output(data, out)
    return _move(data, torch.device(device), out)
