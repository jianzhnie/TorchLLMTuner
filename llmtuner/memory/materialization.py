"""Model-neutral storage allocation for explicit meta models."""

from collections.abc import Mapping

import torch
from torch import nn


def materialize_meta_model(
    model: nn.Module,
    device: torch.device,
    *,
    parameter_device: torch.device | None = None,
    offloaded_parameters: set[nn.Parameter] | None = None,
    runtime_buffers: Mapping[tuple[nn.Module, str], torch.Tensor] | None = None,
) -> None:
    """Allocate model storage, preserving identities and registered aliases.

    Model owners explicitly supply reconstructed non-checkpoint buffers. No
    model identification, RoPE construction or weight initialization happens
    here. Unknown meta runtime buffers fail before changing model storage.
    """
    # Module._apply may allocate each registered path independently when
    # leaving meta. Record aliases before allocation so tied parameters and
    # buffers remain one object when optimizers capture them afterwards.
    tensor_paths: dict[int, list[tuple[nn.Module, str]]] = {}
    parameters: dict[tuple[nn.Module, str], nn.Parameter] = {}
    meta_buffers: list[tuple[nn.Module, str]] = []
    restored_buffers = dict(runtime_buffers or {})
    owners = set(model.modules())
    for (owner, name), value in restored_buffers.items():
        if owner not in owners or name not in owner._buffers:
            raise ValueError("Runtime buffer target must belong to this model")
        original = owner._buffers[name]
        if (
            not isinstance(original, torch.Tensor)
            or not isinstance(value, torch.Tensor)
            or value.is_meta
            or value.shape != original.shape
            or value.dtype != original.dtype
            or value.device != device
        ):
            raise ValueError(f"Invalid reconstructed runtime buffer {name!r}")
    moved_buffers: dict[int, torch.Tensor] = {}
    for module in model.modules():
        for name, parameter in module.named_parameters(
            recurse=False, remove_duplicate=False
        ):
            parameters[module, name] = parameter
            tensor_paths.setdefault(id(parameter), []).append((module, name))
        for name, buffer in module.named_buffers(recurse=False, remove_duplicate=False):
            tensor_paths.setdefault(id(buffer), []).append((module, name))
            if buffer.is_meta:
                if name in module._non_persistent_buffers_set:
                    meta_buffers.append((module, name))
            else:
                # to_empty replaces real buffers too; preserve constructor state.
                if id(buffer) not in moved_buffers:
                    moved_buffers[id(buffer)] = buffer.to(device)
                restored_buffers.setdefault((module, name), moved_buffers[id(buffer)])

    # A recognized buffer may have arbitrary registered aliases, including
    # in other modules. Reconstruct the physical tensor once and resolve all
    # its buffer paths before rejecting unknown non-persistent storage.
    for paths in tensor_paths.values():
        restored_paths = [path for path in paths if path in restored_buffers]
        if restored_paths:
            value = restored_buffers[restored_paths[0]]
            if any(restored_buffers[path] is not value for path in restored_paths):
                raise ValueError("Conflicting runtime buffer aliases")
            for module, name in paths:
                if name in module._buffers:
                    restored_buffers[module, name] = value

    unresolved = [
        f"{type(module).__name__}.{name}"
        for module, name in meta_buffers
        if (module, name) not in restored_buffers
    ]
    if unresolved:
        raise RuntimeError(
            "meta materialization has unsupported non-checkpoint buffers: "
            + ", ".join(unresolved)
        )
    # Persistent meta buffers are populated by the checkpoint load. Unknown
    # non-persistent buffers have now failed before model storage is changed.
    model.to_empty(device=parameter_device or device)
    for (module, name), parameter in parameters.items():
        allocated = getattr(module, name)
        if allocated is not parameter:
            torch.utils.swap_tensors(parameter, allocated)
            setattr(module, name, parameter)
    # Offloaded parameters stay on CPU; buffers execute with the model.
    if parameter_device is not None and parameter_device != device:
        if offloaded_parameters is not None:
            for parameter in model.parameters():
                if parameter not in offloaded_parameters:
                    parameter.data = parameter.data.to(device)
        for module in model.modules():
            for name, buffer in module.named_buffers(
                recurse=False, remove_duplicate=False
            ):
                if (module, name) not in restored_buffers:
                    setattr(module, name, buffer.to(device))
    for (module, name), value in restored_buffers.items():
        setattr(module, name, value)
    for paths in tensor_paths.values():
        if len(paths) > 1:
            owner, name = paths[0]
            tensor = getattr(owner, name)
            for alias_owner, alias_name in paths[1:]:
                setattr(alias_owner, alias_name, tensor)
