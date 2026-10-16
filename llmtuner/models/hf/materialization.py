"""Materialize meta model storage and restore runtime buffers and aliases."""

from __future__ import annotations

import torch
from torch import nn

from ..common.moe.layer import MoE

__all__ = ["materialize_meta_model"]


def materialize_meta_model(model: nn.Module, device: torch.device) -> None:
    """Materialize an already-parallelized HF model without a full copy.

    FSDP turns meta parameters into sharded meta ``DTensor`` objects; calling
    ``to_empty`` afterwards allocates only their local shards. Non-persistent
    buffers are not present in the HF checkpoint, so they must be reconstructed
    rather than left as uninitialized ``to_empty`` storage. Decoder RoPE and
    native MoE token counters are reconstructed explicitly; unknown buffers
    fail loudly so a new architecture cannot train on garbage state.
    """
    # Module._apply may allocate each registered path independently when
    # leaving meta. Record aliases before allocation so tied parameters and
    # buffers remain one object when optimizers capture them afterwards.
    tensor_paths: dict[int, list[tuple[nn.Module, str]]] = {}
    meta_buffers: list[tuple[nn.Module, str]] = []
    restored_buffers: dict[tuple[nn.Module, str], torch.Tensor] = {}
    for module in model.modules():
        for name, parameter in module.named_parameters(
            recurse=False, remove_duplicate=False
        ):
            tensor_paths.setdefault(id(parameter), []).append((module, name))
        for name, buffer in module.named_buffers(recurse=False, remove_duplicate=False):
            tensor_paths.setdefault(id(buffer), []).append((module, name))
            if buffer.is_meta:
                if name in module._non_persistent_buffers_set:
                    meta_buffers.append((module, name))
            else:
                # to_empty replaces real buffers too; preserve constructor state.
                restored_buffers[module, name] = buffer.to(device)

    rope_modules = {module for module, name in meta_buffers if name == "inv_freq"}
    for module, name in meta_buffers:
        if isinstance(module, MoE) and name == "tokens_per_expert_E":
            # Per-step routing statistics start empty and are intentionally
            # excluded from HF weights and native checkpoint state.
            restored_buffers[module, name] = torch.zeros(
                module.router.num_experts, dtype=torch.float32, device=device
            )
    for module in rope_modules:
        if not hasattr(module, "config"):
            continue
        fresh = type(module)(module.config, device=device)
        for name, value in fresh.named_buffers(recurse=False, remove_duplicate=False):
            if name in {"inv_freq", "original_inv_freq"}:
                restored_buffers[module, name] = value

    # A recognized buffer may have arbitrary registered aliases, including
    # in other modules. Reconstruct the physical tensor once and resolve all
    # its buffer paths before rejecting unknown non-persistent storage.
    for paths in tensor_paths.values():
        restored_paths = [path for path in paths if path in restored_buffers]
        if restored_paths:
            value = restored_buffers[restored_paths[0]]
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
    model.to_empty(device=device)
    for (module, name), value in restored_buffers.items():
        setattr(module, name, value)
    for paths in tensor_paths.values():
        if len(paths) > 1:
            owner, name = paths[0]
            tensor = getattr(owner, name)
            for alias_owner, alias_name in paths[1:]:
                setattr(alias_owner, alias_name, tensor)
