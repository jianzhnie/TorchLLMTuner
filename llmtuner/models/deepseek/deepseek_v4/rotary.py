# Copyright 2026 The DeepSeek-AI Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Transformers 5.11 modeling_deepseek_v4.
"""DeepSeek V4 separate main/compress frequencies for interleaved RoPE."""

import torch
from torch import nn
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as reference
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config


def apply_rotary_pos_emb(
    values: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """Rotate trailing interleaved pairs in FP32, preserving leading channels."""
    cosine = cosine.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    sine = sine.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    width = cosine.shape[-1]
    leading, rotary = values[..., :-width], values[..., -width:]
    rotated_pairs = torch.stack(
        (-rotary[..., 1::2], rotary[..., 0::2]), dim=-1
    ).flatten(-2)
    rotated = (rotary.float() * cosine + rotated_pairs.float() * sine).to(values.dtype)
    return torch.cat([leading, rotated], dim=-1)


class DeepseekV4RotaryEmbedding(reference.DeepseekV4RotaryEmbedding):
    """Own frequency buffers and forward; reuse explicit dynamic RoPE policy."""

    def __init__(self, config: DeepseekV4Config) -> None:
        nn.Module.__init__(self)
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.layer_types = [
            key
            for key, value in config.rope_parameters.items()
            if isinstance(value, dict)
        ]
        self.rope_type = {}
        for kind in self.layer_types:
            self.rope_type[kind] = config.rope_parameters[kind]["rope_type"]
            initializer = (
                self.compute_default_rope_parameters
                if self.rope_type[kind] == "default"
                else ROPE_INIT_FUNCTIONS[self.rope_type[kind]]
            )
            inv_freq, scaling = initializer(config, layer_type=kind)
            self.register_buffer(f"{kind}_inv_freq", inv_freq, persistent=False)
            self.register_buffer(
                f"{kind}_original_inv_freq", inv_freq.clone(), persistent=False
            )
            setattr(self, f"{kind}_attention_scaling", scaling)

    @staticmethod
    def compute_default_rope_parameters(
        config: DeepseekV4Config,
        device: torch.device | None = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ) -> tuple[torch.Tensor, float]:
        parameters = config.rope_parameters[layer_type]
        head_dim = config.head_dim or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * parameters.get("partial_rotary_factor", 1.0))
        exponent = (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float
            )
            / dim
        )
        return 1.0 / (parameters["rope_theta"] ** exponent), 1.0

    @torch.no_grad()
    @dynamic_rope_update
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        layer_type: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        scaling = getattr(self, f"{layer_type}_attention_scaling")
        frequencies = (
            inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        positions = position_ids[:, None, :].float()
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            angles = (frequencies.float() @ positions.float()).transpose(1, 2)
            # Interleaved pairs consume unique frequencies; no duplicated half.
            cosine = angles.cos() * scaling
            sine = angles.sin() * scaling
        return cosine.to(x.dtype), sine.to(x.dtype)
