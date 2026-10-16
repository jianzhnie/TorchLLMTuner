# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0.
"""Kimi Delta Attention state context parallelism (KCP)."""

from __future__ import annotations

from functools import wraps

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ...accelerator.collectives import synchronize_local_errors
from ...accelerator.distributed import get_comm_device, get_world_size


def _kcp_runtime():
    """Load KCP kernels only when a KDA model actually requests them."""
    try:
        from fla.modules.conv.causal_conv1d import causal_conv1d
        from triton_ascend_kernels.attention.fla.kda import chunk_kda
        from triton_ascend_kernels.attention.fla.kda.cp import build_cp_context
    except ImportError as exc:
        raise ImportError(
            "KCP requires `fla` and `triton_ascend_kernels` with KDA context "
            "parallel support. Install the Ascend KCP runtime or use another "
            "context_parallel_strategy."
        ) from exc
    return causal_conv1d, chunk_kda, build_cp_context


def kcp_kda_forward(original_forward, group):
    """Wrap a Transformers KDA module with sequence-state parallel kernels."""

    @wraps(original_forward)
    def forward(
        module, hidden_states, attention_mask=None, cache_params=None, **kwargs
    ):
        with synchronize_local_errors("KCP KDA input validation", group=group):
            if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
                raise ValueError("KCP KDA requires [1, local_sequence, hidden] input")
            if hidden_states.shape[1] == 0:
                raise ValueError("KCP KDA requires a nonempty local sequence")
            if cache_params is not None or kwargs.get("cu_seqlens") is not None:
                raise ValueError("KCP KDA does not support cache or packed cu_seqlens")
            if attention_mask is not None and (
                attention_mask.shape != hidden_states.shape[:2]
                or attention_mask.device != hidden_states.device
            ):
                raise ValueError(
                    "KCP attention_mask must match the local [1, sequence] "
                    "shape and input device"
                )
            if attention_mask is not None and not torch.all(attention_mask == 1):
                raise ValueError(
                    "KCP KDA does not yet support padding: zeroing padded tokens "
                    "does not preserve compacted convolution/recurrent state"
                )
            kernel_size = module.q_conv1d.weight.size(-1)
            if kernel_size <= 1:
                raise ValueError(
                    f"KCP KDA convolution kernel size must be greater than 1; "
                    f"got {kernel_size}"
                )
            if module.head_k_dim > 256:
                raise ValueError(
                    f"KCP KDA key head dimension must be at most 256; "
                    f"got {module.head_k_dim}"
                )
            lower_bound = module.gate_lower_bound
            if lower_bound is not None and not -5 <= lower_bound < 0:
                raise ValueError(
                    f"KCP KDA gate lower bound must be in [-5, 0); got {lower_bound}"
                )
            if any(
                getattr(module, f"{name}_conv1d").weight.size(-1) != kernel_size
                for name in ("k", "v")
            ):
                raise ValueError("KCP KDA requires matching Q/K/V convolution kernels")
        batch, local_sequence, _ = hidden_states.shape
        if get_world_size(group) > 1:
            metadata = torch.tensor(
                [
                    local_sequence,
                    kernel_size,
                    module.num_heads,
                    module.head_k_dim,
                    module.head_dim,
                ],
                dtype=torch.int64,
                device=get_comm_device(group),
            )
            bounds = torch.stack((metadata, -metadata))
            dist.all_reduce(bounds, op=dist.ReduceOp.MIN, group=group)
            if bounds[0, 0].item() != -bounds[1, 0].item():
                raise ValueError(
                    "KCP KDA requires equal local sequence lengths on every CP "
                    "rank; repartition the sequence before state communication"
                )
            if not torch.equal(bounds[0, 1:], -bounds[1, 1:]):
                raise ValueError(
                    "KCP KDA requires matching convolution kernel size, head "
                    "count and head dimensions on every CP rank"
                )
        causal_conv1d, chunk_kda, build_cp_context = _kcp_runtime()
        context = build_cp_context(
            cu_seqlens=torch.tensor(
                [0, local_sequence * group.size()],
                dtype=torch.long,
                device=hidden_states.device,
            ),
            group=group,
            conv1d_kernel_size=kernel_size,
        )

        def project(name):
            projection = getattr(module, f"{name}_proj")(hidden_states)
            convolution = getattr(module, f"{name}_conv1d")
            result = causal_conv1d(
                x=projection,
                weight=convolution.weight.squeeze(1),
                bias=convolution.bias,
                activation=convolution.activation,
                initial_state=None,
                output_final_state=False,
                cp_context=context,
            )
            return result[0] if isinstance(result, tuple) else result

        query, key, value = (project(name) for name in ("q", "k", "v"))
        decay = module.f_b_proj(module.f_a_proj(hidden_states))
        beta = module.b_proj(hidden_states).float().sigmoid()
        query = query.view(batch, local_sequence, module.num_heads, module.head_k_dim)
        key = key.view(batch, local_sequence, module.num_heads, module.head_k_dim)
        value = value.view(batch, local_sequence, module.num_heads, module.head_dim)
        decay = decay.view(batch, local_sequence, module.num_heads, module.head_k_dim)
        beta = beta.view(batch, local_sequence, module.num_heads)
        output = chunk_kda(
            q=query,
            k=key,
            v=value,
            g=decay,
            beta=beta,
            A_log=module.A_log,
            dt_bias=module.dt_bias,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=False,
            safe_gate=lower_bound is not None,
            lower_bound=lower_bound,
            state_v_first=False,
            disable_recompute=getattr(module.config, "skip_kda_recompute", False),
            cp_context=context,
        )
        output = output[0] if isinstance(output, tuple) else output
        gate = (
            module.g_proj(hidden_states)
            if module.use_full_rank_gate
            else module.g_b_proj(module.g_a_proj(hidden_states))
        ).view(batch, local_sequence, module.num_heads, module.head_dim)
        output = module.o_norm(output, gate)
        output = module.o_proj(output.reshape(batch, local_sequence, -1))
        return output

    return forward


def kcp_mla_attention(group):
    """Ulysses for expanded Kimi MLA heads and contiguous causal token shards.

    Adapted from FSDPTurbo 0af4eca distributed/context_parallel/ulysses/
    ulysses_attention.py. Exchange autograd uses the existing CP primitives;
    output returns in the HF [batch, sequence, heads, value_dim] convention.
    """
    from .attention import HeadToSeq, SeqToHead

    def forward(
        module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
    ):
        with synchronize_local_errors("KCP MLA local mask validation", group=group):
            batch, heads, local_sequence, _ = query.shape
            if local_sequence == 0:
                raise ValueError("KCP MLA requires a nonempty local sequence")
            if heads % group.size():
                raise ValueError("KCP MLA heads must be divisible by the CP degree")
            valid = torch.ones(
                batch, local_sequence, device=query.device, dtype=torch.bool
            )
            # The decoder constructs a local causal mask. Recover padding from its
            # diagonal, then verify the whole mask before rebuilding global causality.
            # Arbitrary bias or document masks must never be silently discarded.
            if attention_mask is not None:
                if attention_mask.ndim != 4 or attention_mask.shape != (
                    batch,
                    1,
                    local_sequence,
                    local_sequence,
                ):
                    raise ValueError("KCP MLA requires a local square causal mask")
                allowed = (
                    attention_mask
                    if attention_mask.dtype == torch.bool
                    else attention_mask == 0
                )
                valid = allowed[:, 0].diagonal(dim1=-2, dim2=-1)
                causal = torch.ones(
                    local_sequence,
                    local_sequence,
                    device=query.device,
                    dtype=torch.bool,
                ).tril()
                expected = causal[None, None] & valid[:, None, None, :]
                if not torch.equal(allowed, expected):
                    raise ValueError("KCP MLA supports causal padding masks only")
                if attention_mask.dtype != torch.bool and not torch.all(
                    (attention_mask == 0) | torch.isneginf(attention_mask)
                ):
                    raise ValueError("KCP MLA cannot discard additive attention bias")
        masks = [torch.empty_like(valid) for _ in range(group.size())]
        dist.all_gather(masks, valid.contiguous(), group=group)
        valid = torch.cat(masks, dim=-1)
        query, key, value = (SeqToHead.apply(x, group) for x in (query, key, value))
        length = query.shape[-2]
        causal = torch.ones(
            length, length, device=query.device, dtype=torch.bool
        ).tril()
        mask = causal[None, None] & valid[:, None, None, :]
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=dropout, scale=scaling
        )
        return HeadToSeq.apply(output, group).transpose(1, 2).contiguous(), None

    return forward


def apply_kcp(model: nn.Module, cp_mesh) -> int:
    """Patch every structurally recognized KDA module and return its count."""
    from ...models.kimi.latent_attention import KimiLatentAttention

    required = ("q_conv1d", "k_conv1d", "v_conv1d", "A_log", "dt_bias")
    if any(
        (
            all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj"))
            or all(
                hasattr(module, name) for name in ("kv_a_proj_with_mqa", "kv_b_proj")
            )
        )
        and not all(hasattr(module, name) for name in required)
        and not isinstance(module, KimiLatentAttention)
        for module in model.modules()
    ):
        raise ValueError("KCP does not support hybrid KDA/softmax attention models")
    modules = [
        module
        for module in model.modules()
        if all(hasattr(module, name) for name in required)
    ]
    if not modules:
        raise ValueError("KCP selected but no Kimi Delta Attention module was found")
    _kcp_runtime()
    group = cp_mesh.get_group()
    mla_modules = [m for m in model.modules() if isinstance(m, KimiLatentAttention)]
    for module in mla_modules:
        if module.num_heads % group.size():
            raise ValueError("KCP MLA heads must be divisible by the CP degree")
    for module in mla_modules:
        module._kcp_attention = kcp_mla_attention(group)
    for module in modules:
        wrapped = kcp_kda_forward(type(module).forward, group)
        module.forward = wrapped.__get__(module, type(module))
    return len(modules)


__all__ = ["apply_kcp", "kcp_kda_forward"]
