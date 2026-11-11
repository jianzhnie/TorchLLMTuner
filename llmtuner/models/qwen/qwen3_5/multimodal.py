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
"""Qwen3.5 model with portable core operators and the Turbo forward and loss."""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import initialization as init
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)

from ..ops import (
    causal_conv1d,
    causal_conv1d_update,
    chunk_gdn,
    flash_attention_forward,
    fused_recurrent_gdn,
    rms_norm,
    rotary_mul,
)


def _attention_interface(implementation: str) -> Callable:
    if implementation == "flash_attention_2":
        return flash_attention_forward
    return ALL_ATTENTION_FUNCTIONS.get_interface(
        implementation, hf.eager_attention_forward
    )


class Qwen3_5RMSNormGated(hf.Qwen3_5RMSNormGated):
    """Apply the norm-before-gate convention used by the training kernels."""

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Normalize and scale hidden states before applying the SiLU gate."""
        normalized = rms_norm(hidden_states, self.weight, self.variance_epsilon)
        return normalized * F.silu(gate)


class Qwen3_5RMSNorm(hf.Qwen3_5RMSNorm):
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


class Qwen3_5GatedDeltaNet(hf.Qwen3_5GatedDeltaNet):
    """GDN computation with explicit convolution and recurrent cache state."""

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute one GDN layer, retaining HF Cache continuation semantics."""
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


class Qwen3_5Attention(hf.Qwen3_5Attention):
    """Text attention with explicit Turbo backend selection and Qwen gating."""

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor = None,
        past_key_values: Cache = None,
        **kwargs,
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


class Qwen3_5VisionAttention(hf.Qwen3_5VisionAttention):
    """Packed bidirectional attention for independent image and video frames."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Attend within each sequence delimited by cumulative lengths."""
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


class Qwen3_5DecoderLayer(hf.Qwen3_5DecoderLayer):
    """Decoder layer that assembles FSDPTurbo attention / GDN / MLP / norm sublayers."""

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int) -> None:
        hf.GradientCheckpointingLayer.__init__(self)
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        self.block_type = self.layer_type
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer type: {self.layer_type!r}")
        self.mlp = hf.Qwen3_5MLP(config, config.intermediate_size)
        self.moe_block_name = None
        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class Qwen3_5TextModel(hf.Qwen3_5TextModel):
    """Text model assembled from explicit Qwen component subclasses."""

    def __init__(self, config: Qwen3_5TextConfig) -> None:
        hf.Qwen3_5PreTrainedModel.__init__(self, config)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            Qwen3_5DecoderLayer(config, index)
            for index in range(config.num_hidden_layers)
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = hf.Qwen3_5TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        """Preserve HF initialization guards and Tensor/DTensor layouts for GDN."""
        if isinstance(module, hf.Qwen3_5GatedDeltaNet):
            init.ones_(module.dt_bias)
            init.copy_(
                module.A_log,
                torch.empty_like(module.A_log, dtype=torch.float32)
                .uniform_(0.01, 16)
                .log_(),
            )
        else:
            super()._init_weights(module)


class Qwen3_5VisionBlock(hf.Qwen3_5VisionBlock):
    def __init__(self, config: Qwen3_5VisionConfig) -> None:
        hf.GradientCheckpointingLayer.__init__(self)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3_5VisionAttention(config)
        self.mlp = hf.Qwen3_5VisionMLP(config)


class Qwen3_5Model(hf.Qwen3_5Model):
    """Multimodal wrapper that assembles FSDPTurbo text and vision submodels."""

    def __init__(self, config: Qwen3_5Config) -> None:
        hf.Qwen3_5PreTrainedModel.__init__(self, config)
        self.visual = hf.Qwen3_5VisionModel._from_config(config.vision_config)
        self.visual.blocks = nn.ModuleList(
            [
                Qwen3_5VisionBlock(config.vision_config)
                for _ in range(config.vision_config.depth)
            ]
        )
        self.language_model = Qwen3_5TextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()


class Qwen3_5ForConditionalGeneration(hf.Qwen3_5ForConditionalGeneration):
    def __init__(self, config: Qwen3_5Config) -> None:
        hf.Qwen3_5PreTrainedModel.__init__(self, config)
        self.model = Qwen3_5Model(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        self.post_init()


__all__ = ["Qwen3_5ForConditionalGeneration"]
