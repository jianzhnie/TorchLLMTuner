# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3.5 MoE model with portable core operators and the Turbo forward and loss."""

import inspect
from collections.abc import Callable
from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers import initialization as init
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeVisionConfig,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.vision_utils import get_vision_position_ids

try:
    from transformers.vision_utils import (
        get_vision_attention_seqlens,
        get_vision_interpolation_indices_and_weights,
    )
except ImportError:
    from ..context_parallel import unsupported_qwen_cp

    get_vision_attention_seqlens = unsupported_qwen_cp
    get_vision_interpolation_indices_and_weights = unsupported_qwen_cp

from ..context_parallel import (
    GatedDeltaNet,
    cal_split_sizes,
    cp_compute_lm_loss,
    gather_forward_split_backward,
    shard_fused_sequence_inputs,
    split_forward_gather_backward,
    split_tensor_to_group,
    ulysses_context_parallel_attention,
)
from ..ops import (
    causal_conv1d,
    causal_conv1d_update,
    chunk_gdn,
    flash_attention_forward,
    fused_recurrent_gdn,
    grouped_matmul,
    permute,
    rms_norm,
    rotary_mul,
    swiglu,
    unpermute,
)


def _attention_interface(implementation: str) -> Callable:
    if implementation == "flash_attention_2":
        return flash_attention_forward
    return ALL_ATTENTION_FUNCTIONS.get_interface(
        implementation, hf.eager_attention_forward
    )


class Qwen3_5MoeRMSNormGated(hf.Qwen3_5MoeRMSNormGated):
    """Apply the norm-before-gate convention used by the training kernels."""

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Normalize and scale hidden states before applying the SiLU gate."""
        normalized = rms_norm(hidden_states, self.weight, self.variance_epsilon)
        return normalized * F.silu(gate)


class Qwen3_5MoeRMSNorm(hf.Qwen3_5MoeRMSNorm):
    """Apply Qwen's 1 + weight scale through the common normalization operator."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize hidden states with the Qwen residual scale."""
        use_eager = hidden_states.device.type != "npu"
        weight = 1.0 + (self.weight.float() if use_eager else self.weight)
        return rms_norm(
            hidden_states,
            weight,
            self.eps,
            use_eager=use_eager,
        ).to(hidden_states.dtype)


class Qwen3_5MoeGatedDeltaNet(hf.Qwen3_5MoeGatedDeltaNet):
    """GDN computation with explicit convolution and recurrent cache state."""

    def __init__(self, config: Qwen3_5MoeTextConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.norm = Qwen3_5MoeRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.cp_group = None
        self.gdn_implementation = getattr(
            config,
            "gdn_implementation",
            "auto" if getattr(config, "use_fused_gdn", False) else "eager",
        )
        self.causal_conv1d_implementation = getattr(
            config,
            "causal_conv1d_implementation",
            "auto" if getattr(config, "use_fused_causal_conv1d", False) else "eager",
        )

    def causal_conv1d_fn(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Adapt channel-first Ulysses convolution inputs to the portable op."""
        output, _ = causal_conv1d(
            x.transpose(1, 2),
            weight.transpose(0, 1),
            bias=bias,
            activation=activation,
            head_num=(2 * self.num_k_heads + self.num_v_heads)
            // dist.get_world_size(self.cp_group),
            implementation=self.causal_conv1d_implementation,
        )
        return output.transpose(1, 2)

    def chunk_gated_delta_rule(
        self, *args: torch.Tensor, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Use the configured GDN backend for the Ulysses path."""
        return chunk_gdn(*args, implementation=self.gdn_implementation, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """Compute one GDN layer, retaining HF Cache continuation semantics."""
        if self.cp_group is not None:
            return GatedDeltaNet.forward(
                self,
                hidden_states,
                cache_params=cache_params,
                attention_mask=attention_mask,
                ulysses_group=self.cp_group,
                **kwargs,
            )
        hidden_states = hf.apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, sequence_length, _ = hidden_states.shape
        has_state = cache_params is not None and cache_params.has_previous_state(
            self.layer_idx, state_idx=0
        )
        decode = (
            has_state
            and sequence_length == 1
            and not cache_params.layers[self.layer_idx].record_past
        )
        recurrent_state = (
            cache_params.layers[self.layer_idx].recurrent_states[0]
            if has_state
            else None
        )
        mixed_qkv = self.in_proj_qkv(hidden_states)
        gate = self.in_proj_z(hidden_states).reshape(
            batch_size, sequence_length, -1, self.head_v_dim
        )
        beta = self.in_proj_b(hidden_states).sigmoid()
        decay = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a(hidden_states).float() + self.dt_bias
        )
        cu_seqlens = kwargs.get("cu_seqlens")
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.to(torch.int64)
        if decode:
            mixed_qkv, _ = causal_conv1d_update(
                mixed_qkv,
                cache_params.layers[self.layer_idx].conv_states[0],
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation=self.activation,
                implementation=self.causal_conv1d_implementation,
            )
        else:
            if cache_params is not None:
                mixed_qkv = cache_params.update_conv_state(
                    mixed_qkv.transpose(1, 2),
                    self.layer_idx,
                    conv_kernel_size=self.conv_kernel_size,
                ).transpose(1, 2)
            mixed_qkv, _ = causal_conv1d(
                mixed_qkv,
                self.conv1d.weight.squeeze(1).transpose(0, 1),
                self.conv1d.bias,
                activation=self.activation,
                cu_seqlens=cu_seqlens,
                head_num=2 * self.num_k_heads + self.num_v_heads,
                implementation=self.causal_conv1d_implementation,
            )
            mixed_qkv = mixed_qkv[:, -sequence_length:]
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(
            batch_size, sequence_length, self.num_k_heads, self.head_k_dim
        )
        key = key.reshape(
            batch_size, sequence_length, self.num_k_heads, self.head_k_dim
        )
        value = value.reshape(
            batch_size, sequence_length, self.num_v_heads, self.head_v_dim
        )
        if self.num_v_heads != self.num_k_heads:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        if decode:
            output, state = fused_recurrent_gdn(
                query,
                key,
                value,
                decay,
                beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                implementation="eager",
            )
        else:
            output, state = chunk_gdn(
                query,
                key,
                value,
                decay,
                beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
                implementation=self.gdn_implementation,
            )
        if cache_params is not None:
            cache_params.update_recurrent_state(state, self.layer_idx)
        output = self.norm(
            output.reshape(-1, self.head_v_dim), gate.reshape(-1, self.head_v_dim)
        )
        return self.out_proj(output.reshape(batch_size, sequence_length, -1))


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate the configured prefix of Q/K and preserve non-rotary channels."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = rotary_mul(q_rot, cos, sin)
    k_embed = rotary_mul(k_rot, cos, sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class Qwen3_5MoeAttention(hf.Qwen3_5MoeAttention):
    """Text attention with explicit Turbo backend selection and Qwen gating."""

    def __init__(self, config: Qwen3_5MoeTextConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.ulysses_group = None
        self.q_norm = Qwen3_5MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Project Q/K/V, update KV cache, and apply gated attention."""
        if kwargs.get("position_ids") is not None and kwargs["position_ids"].ndim == 3:
            kwargs["position_ids"] = kwargs["position_ids"][0]
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attention_interface = _attention_interface(self.config._attn_implementation)
        total_seq_len = query_states.shape[2]
        attention_kwargs = dict(
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            is_causal=True,
            **kwargs,
        )
        if self.ulysses_group is not None:
            cp_attention = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, hf.eager_attention_forward
            )
            attn_output, attn_weights = ulysses_context_parallel_attention(
                partial(cp_attention, self),
                query_states,
                key_states,
                value_states,
                ulysses_group=self.ulysses_group,
                flash_attention=True,
                **attention_kwargs,
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                total_seq_len=total_seq_len,
                seq_split_lens=None,
                **attention_kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3_5MoeExperts(hf.Qwen3_5MoeExperts):
    """Keep native HF expert weights with optional grouped matrix multiplication."""

    def __init__(self, config: Qwen3_5MoeTextConfig) -> None:
        """Reuse HF parameter shapes and read the existing compute option."""
        super().__init__(config)
        self.use_grouped_expert_matmul = getattr(
            config, "use_grouped_expert_matmul", False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """Route tokens with HF weights in [expert, output, input] layout."""
        if not self.use_grouped_expert_matmul:
            return super().forward(hidden_states, top_k_index, top_k_weights)
        use_eager = hidden_states.device.type != "npu"
        permuted, row_ids = permute(
            hidden_states, top_k_index.to(torch.int32), use_eager=use_eager
        )
        if use_eager:
            counts = torch.bincount(top_k_index.reshape(-1), minlength=self.num_experts)
        else:
            counts = torch.histc(
                top_k_index, bins=self.num_experts, min=0, max=self.num_experts
            )
        projected = grouped_matmul(
            permuted, counts, self.gate_up_proj, use_eager=use_eager
        )
        activated = swiglu(projected, fused=not use_eager)
        projected = grouped_matmul(
            activated, counts, self.down_proj, use_eager=use_eager
        )
        return unpermute(projected, row_ids, probs=top_k_weights, use_eager=use_eager)


class Qwen3_5MoeSparseMoeBlock(hf.Qwen3_5MoeSparseMoeBlock):
    """Evaluate routed experts before the shared expert."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply routed and shared experts and return their combined output."""
        batch_size, sequence_length, hidden_size = hidden_states.shape
        flattened_states = hidden_states.reshape(-1, hidden_size)
        _, routing_weights, selected_experts = self.gate(flattened_states)
        expert_output = self.experts(
            flattened_states, selected_experts, routing_weights
        )
        shared_output = self.shared_expert(flattened_states)
        shared_output = (
            torch.sigmoid(self.shared_expert_gate(flattened_states)) * shared_output
        )
        return (expert_output + shared_output).reshape(
            batch_size, sequence_length, hidden_size
        )


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute vision RoPE in FP32 before restoring Q/K dtypes."""
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = rotary_mul(q, cos, sin)
    k_embed = rotary_mul(k, cos, sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class Qwen3_5MoeVisionAttention(hf.Qwen3_5MoeVisionAttention):
    """Packed bidirectional attention for independent image and video frames."""

    def __init__(self, config: Qwen3_5MoeVisionConfig) -> None:
        super().__init__(config)
        self.cp_group = None

    def _forward_cp(
        self,
        hidden_states,
        cu_seqlens,
        rotary_pos_emb=None,
        position_embeddings=None,
        max_seqlen=None,
        **kwargs,
    ):
        seq_length = hidden_states.shape[0]
        # Explicit head_dim also permits an empty local sequence shard.
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin
        )

        raw_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, hf.eager_attention_forward
        )
        attention_interface = partial(raw_interface, self)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        if self.config._attn_implementation == "flash_attention_2":
            if max_seqlen is None:
                max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
            attention_kwargs = dict(
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
            if self.cp_group is None:
                attn_output, _ = attention_interface(
                    query_states, key_states, value_states, **attention_kwargs
                )
            else:
                attn_output, _ = ulysses_context_parallel_attention(
                    attention_interface,
                    query_states,
                    key_states,
                    value_states,
                    ulysses_group=self.cp_group,
                    total_seq_len=int(cu_seqlens[-1]),
                    flash_attention=True,
                    **attention_kwargs,
                )
        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2)
                for tensor in (query_states, key_states, value_states)
            ]
            attn_outputs = [
                attention_interface(
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits, strict=True)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """Attend within each sequence delimited by cumulative lengths."""
        if self.cp_group is not None:
            return self._forward_cp(
                hidden_states, cu_seqlens, rotary_pos_emb, position_embeddings, **kwargs
            )
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin
        )

        attention_interface = _attention_interface(self.config._attn_implementation)

        if self.config._attn_implementation == "flash_attention_2":
            query_states = query_states.unsqueeze(0)
            key_states = key_states.unsqueeze(0)
            value_states = value_states.unsqueeze(0)

            # Flash Attention 2: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                total_seq_len=seq_length,
                input_layout="1TND",
                **kwargs,
            )
        else:
            query_states = query_states.transpose(0, 1).unsqueeze(0)
            key_states = key_states.transpose(0, 1).unsqueeze(0)
            value_states = value_states.transpose(0, 1).unsqueeze(0)

            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2)
                for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits, strict=True)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3_5MoeVisionPatchMerger(hf.Qwen3_5MoeVisionPatchMerger):
    """Merge vision patches while retaining upstream Ulysses support."""

    def __init__(
        self, config: Qwen3_5MoeVisionConfig, use_postshuffle_norm: bool = False
    ) -> None:
        super().__init__(config, use_postshuffle_norm=use_postshuffle_norm)
        self.cp_group = None

    def forward(
        self, x: torch.Tensor, *, sequence_length: int | None = None
    ) -> torch.Tensor:
        """Normalize and merge patches, gathering across the CP group when set."""
        if self.cp_group is None:
            return super().forward(x)
        gather_sizes = cal_split_sizes(
            sequence_length, dist.get_world_size(self.cp_group)
        )

        if self.use_postshuffle_norm:
            x = gather_forward_split_backward(
                x, self.cp_group, dim=0, gather_sizes=gather_sizes
            )
            x = x.view(-1, self.hidden_size)
            # Split the merged tensor back for distributed processing.
            split_sizes = cal_split_sizes(x.size(0), dist.get_world_size(self.cp_group))
            x = split_forward_gather_backward(
                x, self.cp_group, dim=0, grad_scale="down", split_sizes=split_sizes
            )
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = gather_forward_split_backward(
                x, self.cp_group, dim=0, gather_sizes=gather_sizes
            )
            x = x.view(-1, self.hidden_size)
            # Split the merged tensor back for distributed processing.
            split_sizes = cal_split_sizes(x.size(0), dist.get_world_size(self.cp_group))
            x = split_forward_gather_backward(
                x, self.cp_group, dim=0, grad_scale="down", split_sizes=split_sizes
            )
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class Qwen3_5MoeVisionModel(hf.Qwen3_5MoeVisionModel):
    config: Qwen3_5MoeVisionConfig

    def __init__(self, config: Qwen3_5MoeVisionConfig) -> None:
        super().__init__(config)
        self._rotary_takes_hidden_states = (
            len(inspect.signature(self.rotary_pos_emb.forward).parameters) == 2
        )
        self.cp_group = None
        self.merger = Qwen3_5MoeVisionPatchMerger(config, use_postshuffle_norm=False)
        self.cp_group = None

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs
    ) -> hf.BaseModelOutputWithPooling:
        """Encode patches with upstream vision positions before CP sharding."""
        if self.cp_group is None:
            return super().forward(hidden_states, grid_thw, **kwargs)
        interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side=self.num_grid_per_side,
            mode=self.interpolation_mode,
            align_corners=self.interpolation_align_corners,
            spatial_merge_size=self.spatial_merge_size,
            kwargs=kwargs,
        )
        position_ids = get_vision_position_ids(
            grid_thw, self.spatial_merge_size, kwargs=kwargs
        )
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(
            grid_thw, self.config, kwargs=kwargs
        )
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = (self.pos_embed(interp_indices) * interp_weights[:, :, None]).sum(
            1
        )
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
        position_embeddings = self.rotary_pos_emb(hidden_states, position_ids)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)

        position_embeddings = tuple(
            split_tensor_to_group(embedding, self.cp_group, dim=0)
            for embedding in position_embeddings
        )
        split_sizes = cal_split_sizes(seq_len, dist.get_world_size(self.cp_group))
        hidden_states = split_forward_gather_backward(
            hidden_states, self.cp_group, dim=0, split_sizes=split_sizes
        )
        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        merged = self.merger(hidden_states, sequence_length=seq_len)
        gather_sizes = cal_split_sizes(
            seq_len // self.spatial_merge_size**2, dist.get_world_size(self.cp_group)
        )
        merged = gather_forward_split_backward(
            merged, self.cp_group, dim=0, grad_scale="up", gather_sizes=gather_sizes
        )
        return hf.BaseModelOutputWithPooling(
            last_hidden_state=hidden_states, pooler_output=merged
        )


class Qwen3_5MoeDecoderLayer(hf.Qwen3_5MoeDecoderLayer):
    """Decoder layer that assembles FSDPTurbo attention / GDN / MoE / norm sublayers."""

    def __init__(self, config, layer_idx):
        hf.GradientCheckpointingLayer.__init__(self)
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        self.block_type = self.layer_type
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5MoeGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5MoeAttention(config, layer_idx)
        self.mlp = Qwen3_5MoeSparseMoeBlock(config)
        self.mlp.experts = Qwen3_5MoeExperts(config)
        self.input_layernorm = Qwen3_5MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class Qwen3_5MoeTextModel(hf.Qwen3_5MoeTextModel):
    """Text model assembled from explicit Qwen component subclasses."""

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        return True

    def __init__(self, config):
        hf.Qwen3_5MoePreTrainedModel.__init__(self, config)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                Qwen3_5MoeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = hf.Qwen3_5MoeTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.cp_group = None
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> hf.BaseModelOutputWithPast:
        """Shard text inputs for CP and delegate decoder execution to Transformers."""
        if self.cp_group is not None:
            if past_key_values is not None or cache_position is not None or use_cache:
                raise NotImplementedError("Qwen3.5 CP does not support KV cache.")
            if any(
                kwargs.get(name) is not None
                for name in (
                    "cu_seqlens",
                    "cu_seq_lens_q",
                    "cu_seq_lens_k",
                    "seq_split_lens",
                )
            ):
                raise NotImplementedError(
                    "Qwen3.5 CP does not support packed sequences."
                )
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            if position_ids is None:
                position_ids = torch.arange(
                    inputs_embeds.size(1), device=inputs_embeds.device
                ).unsqueeze(0)
                position_ids = position_ids.expand(inputs_embeds.size(0), -1)
            inputs_embeds, position_ids, attention_mask = shard_fused_sequence_inputs(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                group=self.cp_group,
                gather_backward=True,
            )
            input_ids, use_cache = None, False
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        """Preserve HF initialization guards and Tensor/DTensor layouts for GDN."""
        if isinstance(module, hf.Qwen3_5MoeGatedDeltaNet):
            init.ones_(module.dt_bias)
            init.copy_(
                module.A_log,
                torch.empty_like(module.A_log, dtype=torch.float32)
                .uniform_(0.01, 16)
                .log_(),
            )
        else:
            super()._init_weights(module)


class Qwen3_5MoeModel(hf.Qwen3_5MoeModel):
    """Use HF's multimodal forward with the training text model and vision attention."""

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        return True

    def __init__(self, config: hf.Qwen3_5MoeConfig) -> None:
        # Initialize the HF base before assembling the custom text subtree.
        hf.Qwen3_5MoePreTrainedModel.__init__(self, config)
        self.visual = Qwen3_5MoeVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3_5MoeTextModel._from_config(config.text_config)
        self.rope_deltas = None
        for block in self.visual.blocks:
            block.attn = Qwen3_5MoeVisionAttention(config.vision_config)
        self.post_init()


class Qwen3_5MoeForConditionalGeneration(hf.Qwen3_5MoeForConditionalGeneration):
    """Qwen3.5 MoE with native HF expert weights and the Turbo forward/loss."""

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        return True

    def __init__(self, config: hf.Qwen3_5MoeConfig) -> None:
        """Assemble Turbo components using compute options from the HF text config."""
        hf.Qwen3_5MoePreTrainedModel.__init__(self, config)
        self.model = Qwen3_5MoeModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        self.cp_group = None
        self.post_init()

    def configure_context_parallel(
        self, *, cp_group: dist.ProcessGroup, ulysses_group: dist.ProcessGroup
    ) -> None:
        """Bind externally created groups to the model's existing CP-aware modules."""

        if cp_group is None or ulysses_group is None:
            raise ValueError(
                "Qwen3.5 CP requires explicit CP and Ulysses process groups."
            )
        if self.config.text_config._attn_implementation != "flash_attention_2":
            raise NotImplementedError("Qwen3.5 CP requires flash_attention_2.")
        if self.config.vision_config._attn_implementation != "flash_attention_2":
            raise NotImplementedError("Qwen3.5 vision CP requires flash_attention_2.")
        size = dist.get_world_size(cp_group)
        text_config = self.config.text_config
        if dist.get_world_size(ulysses_group) != size:
            raise ValueError("The Ulysses group and CP group must have the same size.")
        if (
            text_config.linear_num_key_heads % size
            or text_config.linear_num_value_heads % size
        ):
            raise ValueError(
                "GDN key/value heads must be divisible by the CP group size."
            )
        if (
            text_config.num_attention_heads % size
            or self.config.vision_config.num_heads % size
        ):
            raise ValueError(
                "Language and vision attention heads must be divisible by "
                "the Ulysses group size."
            )
        self.cp_group = cp_group
        self.model.language_model.cp_group = cp_group
        for index, layer in enumerate(self.model.language_model.layers):
            if text_config.layer_types[index] == "linear_attention":
                layer.linear_attn.cp_group = cp_group
            else:
                layer.self_attn.ulysses_group = ulysses_group
        self.model.visual.cp_group = ulysses_group
        self.model.visual.merger.cp_group = ulysses_group
        for block in self.model.visual.blocks:
            block.attn.cp_group = ulysses_group

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | hf.Qwen3_5MoeCausalLMOutputWithPast:
        """Compute multimodal outputs and the configured LM and router losses."""
        if self.cp_group is not None:
            if kwargs.get("output_router_logits", False):
                raise NotImplementedError(
                    "Qwen3.5 CP with router auxiliary loss is not supported."
                )
            if not isinstance(logits_to_keep, int) or logits_to_keep != 0:
                raise NotImplementedError(
                    "Qwen3.5 CP requires all local logits (logits_to_keep=0)."
                )
            if (
                past_key_values is not None
                or cache_position is not None
                or kwargs.get("use_cache", False)
            ):
                raise NotImplementedError("Qwen3.5 CP does not support KV cache.")
            kwargs["use_cache"] = False
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            mm_token_type_ids=mm_token_type_ids,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0]
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )

        loss = None
        logits = None
        chunk_loss_enabled = getattr(self, "enable_chunk_loss", False)
        if chunk_loss_enabled:
            # Training injects either legacy chunk loss or CCE through loss_function.
            # # Both consume the full sequence; logits_to_keep only selects returned
            # logits.
            loss = self.lm_head(hidden_states, self.loss_function, labels)
        else:
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            if labels is not None:
                if self.cp_group is not None:
                    loss = cp_compute_lm_loss(logits, labels, group=self.cp_group)
                else:
                    loss = self.loss_function(
                        logits=logits,
                        labels=labels,
                        vocab_size=self.config.text_config.vocab_size,
                    )

        aux_loss = None
        if kwargs.get("output_router_logits", False):
            aux_loss = hf.load_balancing_loss_func(
                outputs.router_logits,
                self.config.text_config.num_experts,
                self.config.text_config.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.config.text_config.router_aux_loss_coef * aux_loss.to(
                    loss.device
                )

        return hf.Qwen3_5MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            router_logits=outputs.router_logits,
        )


__all__ = ["Qwen3_5MoeForConditionalGeneration"]
