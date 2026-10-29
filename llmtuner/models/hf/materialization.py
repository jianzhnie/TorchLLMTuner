"""Materialize meta model storage and restore runtime buffers and aliases."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from llmtuner.errors import UnsupportedCombinationError

from ..common.aux_loss import AuxLoss
from ..common.moe.layer import MoE

if TYPE_CHECKING:
    from transformers import PretrainedConfig

__all__ = ["materialize_meta_model"]


def validate_rotary_initialization(config: PretrainedConfig) -> None:
    """Reject target-device initialization outside its implemented contract."""
    policy = getattr(config, "rotary_init_device", "constructor")
    if policy not in ("constructor", "target"):
        raise ValueError("model.rotary_init_device must be 'constructor' or 'target'")
    if policy == "constructor":
        return
    if config.model_type != "qwen3" or config.rope_parameters["rope_type"] != "default":
        raise UnsupportedCombinationError(
            "model.rotary_init_device='target' requires dense Qwen3 default RoPE"
        )


def initialize_rotary_on_device(model: nn.Module, device: torch.device) -> None:
    """Rebuild opted-in Qwen3 runtime frequencies before training starts."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    replacements: dict[int, torch.Tensor] = {}
    updates = []
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is None or getattr(config, "rotary_init_device", "constructor") != (
            "target"
        ):
            continue
        validate_rotary_initialization(config)
        if not isinstance(module, Qwen3RotaryEmbedding):
            continue
        if type(module) is not Qwen3RotaryEmbedding:
            raise UnsupportedCombinationError(
                "target RoPE initialization requires native Qwen3RotaryEmbedding"
            )
        fresh = Qwen3RotaryEmbedding(config, device=device)
        for name in ("inv_freq", "original_inv_freq"):
            replacements[id(getattr(module, name))] = getattr(fresh, name)
        updates.append((module, fresh))
    for module in model.modules():
        for name, buffer in module.named_buffers(recurse=False, remove_duplicate=False):
            replacement = replacements.get(id(buffer))
            if replacement is not None:
                setattr(module, name, replacement)
    for module, fresh in updates:
        module.attention_scaling = fresh.attention_scaling
        module.max_seq_len_cached = fresh.max_seq_len_cached
        module.original_max_seq_len = fresh.original_max_seq_len


def materialize_meta_model(
    model: nn.Module,
    device: torch.device,
    *,
    parameter_device: torch.device | None = None,
    offloaded_parameters: set[nn.Parameter] | None = None,
) -> None:
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
    parameters: dict[tuple[nn.Module, str], nn.Parameter] = {}
    meta_buffers: list[tuple[nn.Module, str]] = []
    restored_buffers: dict[tuple[nn.Module, str], torch.Tensor] = {}
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
                restored_buffers[module, name] = buffer.to(device)

    rope_modules = {module for module, name in meta_buffers if name == "inv_freq"}
    for module, name in meta_buffers:
        if isinstance(module, AuxLoss) and name == "instance_acc":
            restored_buffers[module, name] = torch.zeros(
                (), dtype=torch.float32, device=device
            )
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
    initialize_rotary_on_device(model, device)
