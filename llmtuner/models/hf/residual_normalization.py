"""First-order Qwen3 normalization with an explicit residual merge boundary."""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
from torch import nn
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.distributed.tensor.experimental import register_sharding
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm

from llmtuner.errors import UnsupportedCombinationError


def _validate_dtype(hidden_states: torch.Tensor) -> None:
    if hidden_states.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("joint RMSNorm supports float16, bfloat16, and float32 inputs")


@torch.library.custom_op("llmtuner::joint_rms_norm_backward", mutates_args=())
def joint_rms_norm_backward(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    upstream: torch.Tensor,
    residual: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    value = hidden_states.float()
    scale = torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + epsilon)
    normalized = (value * scale).to(hidden_states.dtype)
    weight_gradient = upstream * normalized
    if hidden_states.ndim > 1:
        weight_gradient = weight_gradient.sum(tuple(range(hidden_states.ndim - 1)))
    weight_gradient = weight_gradient.to(weight.dtype)
    numerator_gradient = (upstream * weight).to(hidden_states.dtype).float()
    direct = numerator_gradient * scale
    scale_gradient = (numerator_gradient * value).sum(-1, keepdim=True)
    variance_gradient = (-0.5 * scale_gradient) * scale.pow(3)
    indirect = (variance_gradient.expand_as(value) / value.shape[-1]) * (2 * value)
    if hidden_states.dtype == torch.float32:
        input_gradient = (direct + residual) + indirect
    else:
        input_gradient = (direct + indirect).to(hidden_states.dtype) + residual
    return input_gradient.contiguous(), weight_gradient.contiguous()


@joint_rms_norm_backward.register_fake
def _backward_fake(hidden_states, weight, upstream, residual, epsilon):
    return (
        torch.empty_like(hidden_states, memory_format=torch.contiguous_format),
        torch.empty_like(weight, memory_format=torch.contiguous_format),
    )


@torch.library.custom_op("llmtuner::joint_rms_norm", mutates_args=())
def joint_rms_norm(
    hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_dtype(hidden_states)
    value = hidden_states.float()
    scale = torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + epsilon)
    return weight * (value * scale).to(hidden_states.dtype), hidden_states.clone()


@joint_rms_norm.register_fake
def _forward_fake(hidden_states, weight, epsilon):
    _validate_dtype(hidden_states)
    return (
        torch.empty_like(
            hidden_states, dtype=torch.promote_types(hidden_states.dtype, weight.dtype)
        ),
        torch.empty_like(hidden_states),
    )


def _setup_context(ctx, inputs, output):
    hidden_states, weight, ctx.epsilon = inputs
    ctx.save_for_backward(hidden_states, weight)


def _backward(ctx, upstream, residual):
    input_gradient, weight_gradient = joint_rms_norm_backward(
        *ctx.saved_tensors, upstream, residual, ctx.epsilon
    )
    return input_gradient, weight_gradient, None


joint_rms_norm.register_autograd(_backward, setup_context=_setup_context)


@register_sharding(torch.ops.llmtuner.joint_rms_norm.default)
def _forward_sharding(hidden_states, weight, epsilon):
    return [([Replicate(), Replicate()], [Replicate(), Replicate(), None])] + [
        ([Shard(dimension), Shard(dimension)], [Shard(dimension), Replicate(), None])
        for dimension in range(hidden_states.ndim - 1)
    ]


@register_sharding(torch.ops.llmtuner.joint_rms_norm_backward.default)
def _backward_sharding(hidden_states, weight, upstream, residual, epsilon):
    return [
        (
            [Replicate(), Replicate()],
            [Replicate(), Replicate(), Replicate(), Replicate(), None],
        )
    ] + [
        (
            [Shard(dimension), Partial()],
            [Shard(dimension), Replicate(), Shard(dimension), Shard(dimension), None],
        )
        for dimension in range(hidden_states.ndim - 1)
    ]


def forward_joint_residual_norm(
    module: nn.Module, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized input and a nonaliasing residual for the paired decoder."""
    _validate_norm_hooks(module)
    return joint_rms_norm(hidden_states, module.weight, module.variance_epsilon)


def _validate_norm_hooks(module: nn.Module) -> None:
    from .model_setup import _isolate_norm_input

    if (
        any(
            hook is not _isolate_norm_input
            for hook in module._forward_pre_hooks.values()
        )
        or module._forward_hooks
        or module._backward_pre_hooks
        or module._backward_hooks
    ):
        raise UnsupportedCombinationError(
            "model.norm_backend='joint' cannot change residual norm hook contracts"
        )


def forward_joint_residual_block(
    module: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_values: Any = None,
    use_cache: bool = False,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Preserve Qwen3 decoder operations while owning both residual branches."""
    hidden_states, residual = module.input_layernorm(hidden_states)
    hidden_states, _ = module.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states
    hidden_states, residual = module.post_attention_layernorm(hidden_states)
    return residual + module.mlp(hidden_states)


def apply_joint_residual_norm(model: nn.Module) -> None:
    """Install paired norm/block forwards only after validating every instance."""
    if getattr(model.config, "norm_backend", "native") != "joint":
        raise ValueError("model.norm_backend must be 'joint' for joint installation")
    if model.config.model_type != "qwen3":
        raise UnsupportedCombinationError(
            "model.norm_backend='joint' requires dense Qwen3"
        )
    layers = list(model.model.layers)
    for layer in layers:
        if type(layer) is not Qwen3DecoderLayer or (
            "forward" in layer.__dict__
            and getattr(layer.forward, "__func__", None)
            is not forward_joint_residual_block
        ):
            raise UnsupportedCombinationError(
                "model.norm_backend='joint' cannot replace a custom decoder forward"
            )
        for module in (layer.input_layernorm, layer.post_attention_layernorm):
            if type(module) is not Qwen3RMSNorm or (
                "forward" in module.__dict__
                and getattr(module.forward, "__func__", None)
                is not forward_joint_residual_norm
            ):
                raise UnsupportedCombinationError(
                    "model.norm_backend='joint' requires native residual "
                    "RMSNorm modules"
                )
            _validate_norm_hooks(module)
    for layer in layers:
        layer.forward = MethodType(forward_joint_residual_block, layer)
        for module in (layer.input_layernorm, layer.post_attention_layernorm):
            module.forward = MethodType(forward_joint_residual_norm, module)
