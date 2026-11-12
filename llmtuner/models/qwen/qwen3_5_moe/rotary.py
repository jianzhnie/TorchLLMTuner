# Copyright 2026 The Qwen Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Transformers 5.11 modeling_qwen3_5_moe.
"""Qwen3.5 text rotary buffers, partial frequencies and interleaved mRoPE."""

import torch
from torch import Tensor, nn
from transformers import initialization as init
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as reference
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)


class Qwen3_5MoeTextRotaryEmbedding(reference.Qwen3_5MoeTextRotaryEmbedding):
    """Own frequency construction and axis mixing; reuse dynamic update policy."""

    inv_freq: Tensor

    def __init__(
        self, config: Qwen3_5MoeTextConfig, device: torch.device | str | None = None
    ) -> None:
        nn.Module.__init__(self)
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_type = config.rope_parameters["rope_type"]
        initializer = (
            self.compute_default_rope_parameters
            if self.rope_type == "default"
            else ROPE_INIT_FUNCTIONS[self.rope_type]
        )
        inv_freq, self.attention_scaling = initializer(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])

    @staticmethod
    def compute_default_rope_parameters(
        config: Qwen3_5MoeTextConfig,
        device: torch.device | str | None = None,
        seq_len: int | None = None,
    ) -> tuple[Tensor, float]:
        """Compute native partial inverse frequencies without consuming RNG."""
        base = config.rope_parameters["rope_theta"]
        fraction = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * fraction)
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )
        return inv_freq, 1.0

    @torch.no_grad()
    def initialize_frequencies(self) -> None:
        """Restore guarded frequency buffers using the established init policy.

        Native initialization computes these values on the default device;
        materialization and default-RoPE preservation retain separate owners.
        """
        initializer = (
            self.compute_default_rope_parameters
            if self.rope_type == "default"
            else ROPE_INIT_FUNCTIONS[self.rope_type]
        )
        frequencies, _ = initializer(self.config)
        init.copy_(self.inv_freq, frequencies)
        init.copy_(self.original_inv_freq, frequencies)

    @torch.no_grad()
    @dynamic_rope_update
    def forward(
        self, hidden_states: Tensor, position_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        frequencies = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(3, position_ids.shape[1], -1, 1)
            .to(hidden_states.device)
        )
        positions = position_ids[:, :, None, :].float()
        device_type = hidden_states.device.type
        if device_type == "mps":
            device_type = "cpu"
        with reference.maybe_autocast(device_type=device_type, enabled=False):
            phases = (frequencies.float() @ positions.float()).transpose(2, 3)
            phases = self.apply_interleaved_mrope(phases, self.mrope_section)
            embedding = torch.cat((phases, phases), dim=-1)
            cos = embedding.cos() * self.attention_scaling
            sin = embedding.sin() * self.attention_scaling
        return cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)

    def apply_interleaved_mrope(
        self, freqs: Tensor, mrope_section: list[int]
    ) -> Tensor:
        """Interleave height/width phases into the temporal frequency view."""
        temporal = freqs[0]
        for axis, offset in enumerate((1, 2), start=1):
            indices = slice(offset, mrope_section[axis] * 3, 3)
            temporal[..., indices] = freqs[axis, ..., indices]
        return temporal
