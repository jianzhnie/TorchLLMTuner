"""Construction-time normalization policies for supported HF architectures."""

from __future__ import annotations

from types import MethodType

import torch
import torch.nn as nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm

from llmtuner.errors import UnsupportedCombinationError

from .normalization import forward_aten_rms_norm
from .residual_normalization import (
    apply_joint_residual_norm,
    forward_joint_residual_block,
    forward_joint_residual_norm,
    isolate_norm_input,
    validate_joint_residual_norm,
)

try:
    import torch_npu
except ImportError:
    torch_npu = None

# Retain the former hook name for callers holding ownership checks by identity.
_isolate_norm_input = isolate_norm_input


def forward_torch_rms_norm(module: nn.Module, hidden_states: torch.Tensor):
    return torch.nn.functional.rms_norm(
        hidden_states,
        (module.weight.numel(),),
        module.weight,
        module.variance_epsilon,
    )


def forward_npu_rms_norm(module: nn.Module, hidden_states: torch.Tensor):
    if hidden_states.device.type != "npu":
        raise RuntimeError("model.norm_backend='npu' requires NPU tensors")
    return torch_npu.npu_rms_norm(
        hidden_states, module.weight, module.variance_epsilon
    )[0]


def stabilize_residual_norm_gradients(model: nn.Module) -> None:
    """Select normalization arithmetic and known residual gradient boundaries."""
    model_type = getattr(model.config, "model_type", None)
    backend = getattr(model.config, "norm_backend", "native")
    if backend not in ("native", "torch", "aten", "joint", "npu"):
        raise ValueError(
            "model.norm_backend must be 'native', 'torch', 'aten', 'joint', or 'npu'"
        )
    if backend == "joint" and model_type != "qwen3":
        raise ValueError("model.norm_backend='joint' requires dense Qwen3")
    if backend in ("aten", "torch", "npu") and model_type not in ("qwen3", "qwen3_moe"):
        raise ValueError(f"model.norm_backend='{backend}' requires qwen3 or qwen3_moe")
    if model_type == "qwen3":
        norm_type = Qwen3RMSNorm
    elif model_type == "qwen3_moe":
        norm_type = Qwen3MoeRMSNorm
    else:
        return
    owned_forwards = {
        "aten": forward_aten_rms_norm,
        "torch": forward_torch_rms_norm,
        "npu": forward_npu_rms_norm,
        "joint": forward_joint_residual_norm,
    }
    replace_norm_forward = backend in ("aten", "torch", "npu")
    if replace_norm_forward:
        for module in model.modules():
            if (
                type(module) is norm_type
                and "forward" in module.__dict__
                and getattr(module.forward, "__func__", None)
                not in owned_forwards.values()
            ):
                raise UnsupportedCombinationError(
                    f"model.norm_backend={backend!r} cannot replace a custom "
                    "RMSNorm forward; use native or adapt it explicitly."
                )
    if backend == "joint":
        validate_joint_residual_norm(
            model, allowed_norm_forwards=tuple(owned_forwards.values())
        )
    norm_forward = owned_forwards.get(backend)
    for module in model.modules():
        if type(module) is not norm_type:
            continue
        for key, hook in list(module._forward_pre_hooks.items()):
            if hook is isolate_norm_input:
                del module._forward_pre_hooks[key]
                module._forward_pre_hooks_with_kwargs.pop(key, None)
        installed = getattr(module.forward, "__func__", None)
        if replace_norm_forward:
            module.forward = MethodType(norm_forward, module)
        elif installed in owned_forwards.values() and installed is not norm_forward:
            del module.forward
    if backend != "joint":
        for layer in model.model.layers:
            if getattr(layer.forward, "__func__", None) is forward_joint_residual_block:
                del layer.forward
    if replace_norm_forward:
        return
    for layer in model.model.layers:
        for name in ("input_layernorm", "post_attention_layernorm"):
            norm = getattr(layer, name, None)
            if type(norm) is norm_type and isolate_norm_input not in (
                norm._forward_pre_hooks.values()
            ):
                norm.register_forward_pre_hook(isolate_norm_input, with_kwargs=True)
    if backend == "joint":
        apply_joint_residual_norm(model)
