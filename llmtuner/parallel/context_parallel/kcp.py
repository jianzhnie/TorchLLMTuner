"""Kimi Delta Attention state context parallelism (KCP)."""

from __future__ import annotations

from functools import wraps

import torch
import torch.nn as nn


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
        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
            raise ValueError("KCP KDA requires [1, local_sequence, hidden] input")
        if cache_params is not None or kwargs.get("cu_seqlens") is not None:
            raise ValueError("KCP KDA does not support cache or packed cu_seqlens")
        causal_conv1d, chunk_kda, build_cp_context = _kcp_runtime()
        batch, local_sequence, _ = hidden_states.shape
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[..., None].to(hidden_states)
        context = build_cp_context(
            cu_seqlens=torch.tensor(
                [0, local_sequence * group.size()],
                dtype=torch.long,
                device=hidden_states.device,
            ),
            group=group,
            conv1d_kernel_size=module.q_conv1d.weight.size(-1),
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
            safe_gate=module.gate_lower_bound is not None,
            lower_bound=module.gate_lower_bound,
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
        if attention_mask is not None:
            output = output * attention_mask[..., None].to(output)
        return output

    return forward


def apply_kcp(model: nn.Module, cp_mesh) -> int:
    """Patch every structurally recognized KDA module and return its count."""
    required = ("q_conv1d", "k_conv1d", "v_conv1d", "A_log", "dt_bias")
    if any(
        all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj"))
        and not all(hasattr(module, name) for name in required)
        for module in model.modules()
    ):
        raise ValueError("KCP does not support hybrid KDA/softmax attention models")
    _kcp_runtime()
    patched = 0
    for module in model.modules():
        if all(hasattr(module, name) for name in required):
            wrapped = kcp_kda_forward(type(module).forward, cp_mesh.get_group())
            module.forward = wrapped.__get__(module, type(module))
            patched += 1
    if not patched:
        raise ValueError("KCP selected but no Kimi Delta Attention module was found")
    return patched


__all__ = ["apply_kcp", "kcp_kda_forward"]
