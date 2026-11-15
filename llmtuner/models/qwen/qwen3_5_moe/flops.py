"""Model-owned training FLOPs estimates; no model construction or device work."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....utils.flops import int_attr, quadratic_attention_flops_per_token

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig


def flops_estimation_note(arch: PretrainedConfig, *, seq_len: int) -> str | None:
    """Explain architecture-specific estimates alongside their arithmetic.

    The caller owns logging and rank selection. Returning no note preserves the
    existing output for architectures without an additional accounting caveat.
    """
    return (
        "Qwen3.5 model FLOPs: recurrent-equivalent GDN, gated "
        "projections, active experts and depthwise convolution; "
        f"full-square attention at sequence length {seq_len}. Excludes "
        "recomputation, padding overhead, optimizer and communication; "
        "this is model MFU, not measured kernel utilization."
    )


def flops_per_token(arch: PretrainedConfig, *, seq_len: int) -> int:
    """HF 5.13 native text MoE training model FLOPs per input token.

    Every decoder layer has a router, top-k routed experts, a shared expert
    and its scalar gate, irrespective of generic dense-prefix/sparse-step
    metadata. Count all linear projections, including the full-attention Q
    gate and GDN's z/a/b projections, plus lm_head (even when tied).

    Multiply-adds cost two FLOPs; training costs three times forward. GDN's
    memory read, rank-one state update and output read each cost 2*K*V per
    value head in forward, hence 18*H_v*K*V in training. This is the
    torchtitan recurrent-equivalent convention, NOT actual chunk/kernel FLOPs:
    chunk-local products, triangular solves and padding overhead are excluded.
    Depthwise convolution costs 6*channels*kernel_size per token, without
    discounting boundary zeros. Full attention uses full-square contractions,
    without discounting causal sparsity, at the supplied uniform seq_len.

    Exclude normalization, biases, nonlinearities, elementwise gates/decay,
    routing selection/combine, loss, optimizer, communication and all
    recomputation. All weights are assumed trainable. This covers text only;
    variable document lengths require a separate step-level attention count.
    Return zero for missing/invalid native geometry or unresolved layer types.
    """
    names = (
        "hidden_size",
        "vocab_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
    )
    sizes = [int_attr(arch, name) for name in names]
    if any(size is None or size <= 0 for size in sizes):
        return 0
    (
        hidden,
        vocab,
        layers,
        heads,
        kv_heads,
        head_dim,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        kernel,
        experts,
        top_k,
        expert_width,
        shared_width,
    ) = sizes
    if heads % kv_heads or value_heads % key_heads or top_k > experts:
        return 0
    layer_types = getattr(arch, "layer_types", None)
    if not isinstance(layer_types, list | tuple) or len(layer_types) != layers:
        return 0
    if any(kind not in ("linear_attention", "full_attention") for kind in layer_types):
        return 0

    # Native Qwen3.5 MoE has the same sparse FFN in every decoder layer.
    moe = hidden * (experts + 3 * top_k * expert_width + 3 * shared_width + 1)
    key_width = key_heads * key_dim
    value_width = value_heads * value_dim
    gdn_projection = hidden * (2 * key_width + 3 * value_width + 2 * value_heads)
    convolution = (2 * key_width + value_width) * kernel
    gdn = 6 * (gdn_projection + convolution) + 18 * value_heads * key_dim * value_dim
    full = 6 * hidden * head_dim * (3 * heads + 2 * kv_heads)
    full += quadratic_attention_flops_per_token(
        num_heads=heads,
        qk_head_dim=head_dim,
        v_head_dim=head_dim,
        seq_len=seq_len,
    )
    return 6 * (hidden * vocab + layers * moe) + sum(
        gdn if kind == "linear_attention" else full for kind in layer_types
    )
