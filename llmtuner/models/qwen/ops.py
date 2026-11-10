"""Qwen operator adapters shared only by the explicit Qwen model subclasses."""

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen3_5 import modeling_qwen3_5 as reference
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    causal_conv1d_fn,
    rotate_half,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    causal_conv1d_update as hf_causal_conv1d_update,
)


def rms_norm(hidden_states, weight, eps, use_eager=False):
    del use_eager
    return torch.nn.functional.rms_norm(hidden_states, (weight.numel(),), weight, eps)


def rotary_mul(query, cos, sin):
    return query * cos + rotate_half(query) * sin


def flash_attention_forward(*args, **kwargs):
    return sdpa_attention_forward(*args, **kwargs)


def causal_conv1d(inputs, weight, bias=None, activation=None, **kwargs):
    if activation not in (None, "silu", "swish"):
        raise ValueError("Qwen causal convolution supports silu/swish or no activation")
    kwargs.pop("head_num", None)
    kwargs.pop("implementation", None)
    if kwargs.get("cu_seqlens") is not None:
        raise ValueError(
            "Packed convolution requires a backend with explicit document boundaries"
        )
    kwargs.pop("cu_seqlens", None)
    if causal_conv1d_fn is not None:
        # HF and causal-conv1d accept [B, channels, T] and [channels, kernel];
        # this family interface uses [B, T, channels] and [kernel, channels].
        output = causal_conv1d_fn(
            inputs.transpose(1, 2),
            weight.transpose(0, 1),
            bias,
            activation=activation,
            **kwargs,
        )
        return output.transpose(1, 2), None
    sequence_length = inputs.shape[1]
    output = torch.nn.functional.conv1d(
        inputs.transpose(1, 2),
        weight.transpose(0, 1).unsqueeze(1),
        bias,
        padding=weight.shape[0] - 1,
        groups=inputs.shape[-1],
    )
    if activation in ("silu", "swish"):
        output = torch.nn.functional.silu(output)
    return output.transpose(1, 2)[:, :sequence_length], None


def causal_conv1d_update(
    inputs, conv_state, weight, bias=None, activation=None, **kwargs
):
    # Validate before invoking a backend that updates conv_state in place.
    if activation not in (None, "silu", "swish"):
        raise ValueError("Qwen causal convolution supports silu/swish or no activation")
    kwargs.pop("implementation", None)
    if kwargs:
        raise ValueError(
            "Unsupported incremental convolution options: " + ", ".join(sorted(kwargs))
        )
    torch_update = getattr(reference, "torch_causal_conv1d_update", None)
    if hf_causal_conv1d_update is None and torch_update is None:
        raise NotImplementedError(
            "Qwen recurrent convolution requires causal-conv1d; this Transformers "
            "version does not provide the torch update implementation"
        )
    return (
        (
            hf_causal_conv1d_update(
                inputs.transpose(1, 2), conv_state, weight, bias, activation
            ).transpose(1, 2)
            if hf_causal_conv1d_update is not None
            else torch_update(
                inputs.transpose(1, 2), conv_state, weight, bias, activation
            ).transpose(1, 2)
        ),
        None,
    )


def chunk_gdn(*args, implementation="eager", **kwargs):
    if implementation != "eager":
        raise NotImplementedError(
            "Portable Qwen chunk GDN supports implementation='eager' only"
        )
    return torch_chunk_gated_delta_rule(*args, **kwargs)


def fused_recurrent_gdn(*args, implementation="eager", **kwargs):
    if implementation != "eager":
        raise NotImplementedError(
            "Portable Qwen recurrent GDN supports implementation='eager' only"
        )
    return torch_recurrent_gated_delta_rule(*args, **kwargs)


def grouped_matmul(inputs, counts, weights, use_eager=False, weights_layout="linear"):
    del use_eager
    if weights_layout != "linear":
        raise ValueError("Explicit Qwen grouped GEMM expects linear weight layout")
    return torch._grouped_mm(
        inputs, weights.transpose(-2, -1), offs=counts.cumsum(0, dtype=torch.int32)
    )


def permute(tokens, indices, num_out_tokens=None, use_eager=False):
    del use_eager
    if num_out_tokens is not None and num_out_tokens != indices.numel():
        raise ValueError(
            "Portable Qwen permutation requires all routed rows; "
            "capacity truncation and padding are not implemented"
        )
    sorted_indices = torch.argsort(indices.reshape(-1), stable=True)
    top_k = indices.shape[-1]
    return tokens[sorted_indices // top_k], sorted_indices


def unpermute(
    permuted_tokens,
    sorted_indices,
    probs=None,
    restore_shape=None,
    use_eager=False,
):
    del use_eager
    if probs is None and restore_shape is None:
        raise ValueError("unpermute requires routing probabilities or restore_shape")
    if restore_shape is not None:
        output_shape = restore_shape
    else:
        output_shape = (probs.shape[0], permuted_tokens.shape[-1])
    output = permuted_tokens.new_zeros(output_shape)
    if probs is not None:
        top_k = probs.shape[-1]
    elif output_shape[0] == 0:
        if sorted_indices.numel():
            raise ValueError("Cannot restore nonempty routed rows to zero tokens")
        return output
    else:
        rows, remainder = divmod(sorted_indices.numel(), output_shape[0])
        if remainder or rows == 0:
            raise ValueError("Routed rows must contain the same top-k count per token")
        top_k = rows
    if probs is not None:
        # Rows are expert-sorted; scores must follow the same permutation.
        weights = probs.reshape(-1)[sorted_indices].unsqueeze(1)
        permuted_tokens = permuted_tokens * weights.to(permuted_tokens.dtype)
    token_indices = sorted_indices // top_k
    return output.index_add(0, token_indices, permuted_tokens)


def swiglu(inputs, dim=-1, fused=True):
    del fused
    gate, up = inputs.chunk(2, dim=dim)
    return torch.nn.functional.silu(gate) * up
