"""ATen RMSNorm boundaries for explicit CPU reproducible execution.

Imported only when the model requests the aten normalization backend. Operator
registrations belong to this module's lifetime, not an individual Trainer.
Only first-order gradients are supported.
"""

import torch
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.distributed.tensor.experimental import register_sharding


def _forward(x, w, eps):
    value = x.float()
    scale = torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + eps)
    return w * (value * scale).to(x.dtype)


@torch.library.custom_op("llmtuner::rms_norm_backward", mutates_args=())
def norm_backward(
    x: torch.Tensor, w: torch.Tensor, g: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    value = x.float()
    scale = torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + eps)
    norm = (value * scale).to(x.dtype)
    gw = g * norm
    if x.ndim > 1:
        gw = gw.sum(tuple(range(x.ndim - 1)))
    gw = gw.to(w.dtype)
    gn = (g * w).to(x.dtype).to(value.dtype)
    direct = gn * scale
    gs = (gn * value).sum(-1, keepdim=True)
    variance = (-0.5 * gs) * scale.pow(3)
    indirect = (variance.expand_as(value) / x.shape[-1]) * (2 * value)
    return (direct + indirect).to(x.dtype).contiguous(), gw.contiguous()


@norm_backward.register_fake
def _(x, w, g, eps):
    return (
        torch.empty_like(x, memory_format=torch.contiguous_format),
        torch.empty_like(w, memory_format=torch.contiguous_format),
    )


@torch.library.custom_op("llmtuner::rms_norm", mutates_args=())
def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return _forward(x, w, eps)


@rms_norm.register_fake
def _(x, w, eps):
    return torch.empty_like(x, dtype=torch.promote_types(x.dtype, w.dtype))


def _setup_context(ctx, inputs, output):
    x, w, ctx.eps = inputs
    ctx.save_for_backward(x, w)


def _backward(ctx, g):
    return *norm_backward(*ctx.saved_tensors, g, ctx.eps), None


rms_norm.register_autograd(_backward, setup_context=_setup_context)


def forward_aten_rms_norm(module, hidden_states):
    """Preserve HF Qwen's cast and reduction order without replacing parameters."""
    if hidden_states.device.type != "cpu":
        raise RuntimeError("model.norm_backend='aten' currently requires CPU tensors")
    if hidden_states.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("aten RMSNorm supports float16, bfloat16, and float32 inputs")
    return rms_norm(hidden_states, module.weight, module.variance_epsilon)


@register_sharding(torch.ops.llmtuner.rms_norm.default)
def norm_sharding(x, w, eps):
    return [([Replicate()], [Replicate(), Replicate(), None])] + [
        ([Shard(dim)], [Shard(dim), Replicate(), None]) for dim in range(x.ndim - 1)
    ]


@register_sharding(torch.ops.llmtuner.rms_norm_backward.default)
def backward_sharding(x, w, g, eps):
    return [
        (
            [Replicate(), Replicate()],
            [Replicate(), Replicate(), Replicate(), None],
        )
    ] + [
        ([Shard(dim), Partial()], [Shard(dim), Replicate(), Shard(dim), None])
        for dim in range(x.ndim - 1)
    ]
