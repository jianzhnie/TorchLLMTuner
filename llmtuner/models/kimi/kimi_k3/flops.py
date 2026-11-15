"""Model-owned training FLOPs estimates; no model construction or device work."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....utils.flops import (
    attention_param_term,
    int_attr,
    quadratic_attention_flops_per_token,
)

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig


def flops_per_token(arch: PretrainedConfig, *, seq_len: int) -> int:
    """Local Kimi model FLOPs, using recurrent-equivalent KDA state work."""
    names = (
        "hidden_size",
        "vocab_size",
        "num_hidden_layers",
        "num_attention_heads",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "kv_lora_rank",
        "intermediate_size",
    )
    values = [int_attr(arch, name) for name in names]
    if any(value is None or value <= 0 for value in values):
        return 0
    hidden, vocab, layers, heads, qk_nope, qk_rope, value_dim, _, dense = values
    settings = getattr(arch, "linear_attn_config", None)
    if not isinstance(settings, dict):
        return 0
    kda_layers = settings.get("kda_layers")
    full_layers = settings.get("full_attn_layers")
    if not isinstance(kda_layers, list) or not isinstance(full_layers, list):
        return 0
    schedule = kda_layers + full_layers
    if any(isinstance(index, bool) or not isinstance(index, int) for index in schedule):
        return 0
    if len(schedule) != layers or len(set(schedule)) != layers:
        return 0
    if set(kda_layers) & set(full_layers) or set(kda_layers) | set(full_layers) != set(
        range(1, layers + 1)
    ):
        return 0
    dimensions = [
        settings.get(name)
        for name in ("head_dim", "num_heads", "short_conv_kernel_size")
    ]
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in dimensions
    ):
        return 0
    head_dim, kda_heads, kernel = dimensions

    qk_dim = qk_nope + qk_rope
    mla = attention_param_term(
        arch,
        hidden=hidden,
        num_heads=heads,
        qk_head_dim=qk_dim,
        v_head_dim=value_dim,
    )
    if mla is None:
        return 0
    if getattr(arch, "mla_use_output_gate", False):
        mla += 2 * hidden * heads * value_dim
    mla *= 3
    mla += quadratic_attention_flops_per_token(
        num_heads=heads,
        qk_head_dim=qk_dim,
        v_head_dim=value_dim,
        seq_len=seq_len,
    )

    width = head_dim * kda_heads
    # Q/K/V/O, factorized decay gate and beta projection.
    kda_projection = 4 * hidden * width + hidden * head_dim + head_dim * width
    kda_projection += hidden * kda_heads
    if settings.get("use_full_rank_gate", False):
        kda_projection += hidden * width
    else:
        kda_projection += hidden * head_dim + head_dim * width
    kda = 6 * (kda_projection + 3 * width * kernel)
    kda += 18 * kda_heads * head_dim * head_dim
    total = 6 * hidden * vocab
    if getattr(arch, "attn_res_block_size", None) is not None:
        total += 6 * hidden * (2 * layers + 1)

    experts = int_attr(arch, "num_experts") or 0
    top_k = int_attr(arch, "num_experts_per_token") or 0
    expert_width = int_attr(arch, "moe_intermediate_size") or 0
    shared = int_attr(arch, "num_shared_experts") or 0
    latent = int_attr(arch, "routed_expert_hidden_size")
    first_dense = int_attr(arch, "first_k_dense_replace") or 0
    frequency = int_attr(arch, "moe_layer_freq") or 1
    for index in range(layers):
        total += kda if index + 1 in kda_layers else mla
        if index >= first_dense and index % frequency == 0 and experts:
            if not 1 <= top_k <= experts or expert_width <= 0:
                return 0
            routed_width = hidden if latent is None else latent
            moe = hidden * experts
            if latent is not None:
                moe += 2 * hidden * routed_width
            moe += 3 * top_k * routed_width * expert_width
            moe += 3 * shared * hidden * expert_width
            total += 6 * moe
        else:
            total += 18 * hidden * dense
    return total
