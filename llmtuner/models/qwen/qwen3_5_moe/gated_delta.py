# Copyright 2026 The Qwen Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Transformers 5.11 modeling_qwen3_5_moe.
"""Model-owned GDN layout and execution; reference leaf kernels remain reusable."""

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as reference
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)


class Qwen3_5MoeRMSNormGated(reference.Qwen3_5MoeRMSNormGated):
    """Normalize before gating with the native mixed-precision cast order."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs: Any) -> None:
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        values = hidden_states.to(torch.float32)
        variance = values.pow(2).mean(-1, keepdim=True)
        normalized = values * torch.rsqrt(variance + self.variance_epsilon)
        # This cast precedes the affine multiply, unlike zero-centered RMSNorm.
        normalized = self.weight * normalized.to(input_dtype)
        return (normalized * F.silu(gate.to(torch.float32))).to(input_dtype)


class Qwen3_5MoeGatedDeltaNet(reference.Qwen3_5MoeGatedDeltaNet):
    """Own convolution/recurrence order and cached continuation state updates."""

    def __init__(self, config: Qwen3_5MoeTextConfig, layer_idx: int) -> None:
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = (
            Qwen3_5MoeRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
            if reference.FusedRMSNormGated is None or self.A_log.device.type != "cuda"
            else reference.FusedRMSNormGated(
                self.head_v_dim,
                eps=self.layer_norm_epsilon,
                activation=self.activation,
                device=self.A_log.device,
                dtype=config.dtype
                if config.dtype is not None
                else torch.get_default_dtype(),
            )
        )

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.causal_conv1d_fn = reference.causal_conv1d_fn
        self.causal_conv1d_update = (
            reference.causal_conv1d_update or reference.torch_causal_conv1d_update
        )
        self.chunk_gated_delta_rule = (
            reference.chunk_gated_delta_rule or reference.torch_chunk_gated_delta_rule
        )
        self.recurrent_gated_delta_rule = (
            reference.fused_recurrent_gated_delta_rule
            or reference.torch_recurrent_gated_delta_rule
        )

        if not reference.is_fast_path_available:
            reference.logger.warning_once(
                "The fast path is not available because one of the required "
                "library is not installed. Falling back to torch implementation. "
                "To install follow "
                "https://github.com/fla-org/flash-linear-attention#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )

        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states = reference.apply_mask_to_padding_states(
            hidden_states, attention_mask
        )
        # Installing CUDA-only extensions must not select them for CPU/NPU
        # tensors. Torch reference kernels run on the input device, without
        # transferring activations to CPU. Meta construction cannot establish
        # the eventual execution device, so make this choice at each forward.
        cuda_kernels = hidden_states.device.type == "cuda"
        conv_update = (
            self.causal_conv1d_update
            if cuda_kernels
            else reference.torch_causal_conv1d_update
        )
        chunk_rule = (
            self.chunk_gated_delta_rule
            if cuda_kernels
            else reference.torch_chunk_gated_delta_rule
        )
        recurrent_rule = (
            self.recurrent_gated_delta_rule
            if cuda_kernels
            else reference.torch_recurrent_gated_delta_rule
        )

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape

        # We have cached `conv_state` / `recurrent_state` to continue from. The two
        # cached modes
        # (single-token decode and chunk-tokens continuation) share the state read here;
        # they only
        # diverge in how the conv input is assembled and which kernel consumes the
        # states below,
        # which we gate locally on `seq_len`.
        use_precomputed_states = (
            cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        )

        # getting projected states from cache if it exists
        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states and seq_len == 1:
            # Single-token cached decode: the fused per-step kernel updates the conv
            # state in-place.
            mixed_qkv = conv_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            # Multi-token forward (prefill, or chunked-tokens decode when the cache has
            # prior state).
            if use_precomputed_states:
                # Cached chunked-tokens decode: prepend the cached conv context so the
                # causal conv
                # sees the correct left-context rather than zero-padding. Dropped from
                # the output
                # at the end of this branch.
                mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
            if cache_params is not None:
                new_conv_state = F.pad(
                    mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0)
                )
                cache_params.update_conv_state(new_conv_state, self.layer_idx)
            if cuda_kernels and self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=kwargs.get("seq_idx"),
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])
            if use_precomputed_states:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if use_precomputed_states and seq_len == 1:
            core_attn_out, last_recurrent_state = recurrent_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = chunk_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state if use_precomputed_states else None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                # The chunked FLA kernel takes a single `cu_seqlens` arg; for packed
                # self-attention this matches q-side lengths.
                cu_seqlens=kwargs.get("cu_seq_lens_q"),
            )

        # Update cache
        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        return output
