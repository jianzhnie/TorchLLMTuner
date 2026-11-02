"""Construction-time normalization policies for supported HF architectures."""

from __future__ import annotations

import torch
import torch.nn as nn


def _isolate_norm_input(module, args, kwargs):
    if args:
        value = args[0]
        return (value.view_as(value), *args[1:]), kwargs
    value = kwargs["hidden_states"]
    return args, {**kwargs, "hidden_states": value.view_as(value)}


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
    import torch_npu

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
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

        norm_type = Qwen3RMSNorm
    elif model_type == "qwen3_moe":
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm

        norm_type = Qwen3MoeRMSNorm
    else:
        return
    for module in model.modules():
        if type(module) is not norm_type:
            continue
        for key, hook in list(module._forward_pre_hooks.items()):
            if hook is _isolate_norm_input:
                del module._forward_pre_hooks[key]
                module._forward_pre_hooks_with_kwargs.pop(key, None)
        if backend != "npu" and getattr(module.forward, "__func__", None) is (
            forward_npu_rms_norm
        ):
            del module.forward
        if (
            backend != "aten"
            and "forward" in module.__dict__
            and getattr(module.forward, "__module__", None)
            == f"{__package__}.normalization"
        ):
            from .normalization import forward_aten_rms_norm

            if getattr(module.forward, "__func__", None) is forward_aten_rms_norm:
                del module.forward
        if backend != "torch" and getattr(module.forward, "__func__", None) is (
            forward_torch_rms_norm
        ):
            del module.forward
        if backend != "joint" and getattr(module.forward, "__module__", None) == (
            f"{__package__}.residual_normalization"
        ):
            from .residual_normalization import forward_joint_residual_norm

            if getattr(module.forward, "__func__", None) is forward_joint_residual_norm:
                del module.forward
    if backend != "joint":
        for layer in model.model.layers:
            if getattr(layer.forward, "__module__", None) == (
                f"{__package__}.residual_normalization"
            ):
                from .residual_normalization import forward_joint_residual_block

                if (
                    getattr(layer.forward, "__func__", None)
                    is forward_joint_residual_block
                ):
                    del layer.forward
    if backend in ("aten", "torch", "npu"):
        from types import MethodType

        if backend == "aten":
            from .normalization import forward_aten_rms_norm as norm_forward
        elif backend == "npu":
            norm_forward = forward_npu_rms_norm
        else:
            norm_forward = forward_torch_rms_norm
        for module in model.modules():
            if type(module) is norm_type:
                module.forward = MethodType(norm_forward, module)
        return
    for layer in model.model.layers:
        for name in ("input_layernorm", "post_attention_layernorm"):
            norm = getattr(layer, name, None)
            if type(norm) is norm_type and _isolate_norm_input not in (
                norm._forward_pre_hooks.values()
            ):
                norm.register_forward_pre_hook(_isolate_norm_input, with_kwargs=True)
    if backend == "joint":
        from .residual_normalization import apply_joint_residual_norm

        apply_joint_residual_norm(model)
