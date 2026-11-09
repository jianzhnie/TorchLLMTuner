"""Training FLOPs per token: the model-work term in MFU's numerator.

The arithmetic is pure -- it reads an HF config and returns a number, independent
of model construction and trainer orchestration. Runtime assembly passes its resolved
text config to ``flops_per_token``; estimating throughput never rebuilds it.

The convention, matching torchtitan's (``models/utils.py`` plus the HF
backend's ``get_nparams_and_flops``), is documented on :func:`flops_per_token`.
"""

from __future__ import annotations

from typing import NamedTuple

from transformers.configuration_utils import PretrainedConfig

__all__ = [
    "flops_per_token",
    "quadratic_attention_flops_per_token",
]


def quadratic_attention_flops_per_token(
    *,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    seq_len: int,
    sliding_window_size: int | None = None,
) -> int:
    """Training FLOPs per token for quadratic or windowed attention.

    The two attention contractions (``q @ k`` and ``attn @ v``), each counted
    three times for forward and backward as ``6`` -- the convention upstream's
    ``models/utils.py`` uses. A windowed layer attends to at most
    ``sliding_window_size`` keys per query, so the sequence length is capped
    there; causal sparsity and the recomputed backward pass are not discounted,
    exactly as upstream.
    """
    if isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len < 0:
        raise ValueError(f"seq_len must be a non-negative integer, got {seq_len!r}")
    if sliding_window_size is not None and (
        isinstance(sliding_window_size, bool)
        or not isinstance(sliding_window_size, int)
        or sliding_window_size <= 0
    ):
        raise ValueError(
            "sliding_window_size must be a positive integer or None, "
            f"got {sliding_window_size!r}"
        )
    attended_tokens = (
        seq_len if sliding_window_size is None else min(seq_len, sliding_window_size)
    )
    return 6 * num_heads * (qk_head_dim + v_head_dim) * attended_tokens


class MoEGeometry(NamedTuple):
    """The MoE geometry of a config, and which layers route."""

    num_experts: int
    top_k: int
    expert_intermediate: int
    shared_intermediate: int
    layers: frozenset[int]


# ``moe_geometry`` has three outcomes: no MoE (``None``), a resolved geometry, or
# a declared MoE whose layers/widths the config does not determine -- which makes
# the whole FLOPs count unresolvable rather than approximate.
_UNRESOLVED = object()


def flops_per_token(arch: PretrainedConfig, *, seq_len: int) -> int:
    """Training FLOPs per token for the architecture ``arch`` describes.

    The convention, matching torchtitan's (``models/utils.py`` plus the HF
    backend's ``get_nparams_and_flops``):

    * Every matmul weight applied to every token costs ``2 * in * out`` FLOPs
      per token (one multiply and one add per weight), counted three times --
      once for the forward pass and twice for the backward. That ``3 * 2 = 6``
      is where the familiar ``6N`` comes from. Norms, biases and attention
      sinks are not matmuls and are not counted; upstream's parameter walk does
      include them, which overstates its own ``6N`` term by well under a
      percent.
    * Attention adds :func:`quadratic_attention_flops_per_token` per layer, the
      two attention contractions that the parameter term above does not cover.
    * A dense FFN is the three gated projections (gate, up, down). A MoE layer
      is its router plus ``top_k`` of its ``num_experts`` experts plus *all* of
      its shared experts -- the routed experts are the one place the parameter
      count and the compute diverge, which is why this cannot be read off
      ``intermediate_size`` alone.
    * MLA layers project through the low-rank ``q_lora``/``kv_lora`` path
      instead of a plain QKV, using ``qk_head_dim``/``v_head_dim`` rather than
      the ``head_dim`` field (which some MLA configs spell as the rope slice
      alone -- DeepSeek-V3's is 64 there against 192 for QK).
    * Embedding lookups are not matmuls and cost nothing here, but the output
      projection is a matmul and is counted, whether or not its weight is tied
      to the embedding table: tying is a storage decision, not a compute one.
    * ``qwen3_5_moe_text`` uses its native per-layer MoE and gated attention
      geometry, including depthwise convolution. GDN is charged as recurrent-
      equivalent model work, not actual chunk/kernel execution FLOPs; see
      :func:`qwen3_5_moe_text_flops_per_token` for the counting boundary.

    Returns 0 -- suppressing MFU and tflops rather than reporting a wrong
    magnitude -- when the geometry cannot be resolved exactly: a missing size,
    a MoE whose per-layer split or expert width the config does not state, a
    ``layer_types`` list shorter than the layer count, or a layer type whose
    parameter term is not derivable from the config (``linear_attention`` outside
    the dedicated Qwen3.5 MoE branch, and other unsupported layer families).
    """
    if isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len < 0:
        raise ValueError(f"seq_len must be a non-negative integer, got {seq_len!r}")

    if getattr(arch, "model_type", None) == "qwen3_5_moe_text":
        return qwen3_5_moe_text_flops_per_token(arch, seq_len=seq_len)
    if getattr(arch, "model_type", None) == "kimi_linear":
        return kimi_linear_flops_per_token(arch, seq_len=seq_len)

    hidden = int_attr(arch, "hidden_size")
    num_layers = int_attr(arch, "num_hidden_layers")
    vocab_size = int_attr(arch, "vocab_size")
    num_heads = int_attr(arch, "num_attention_heads")
    if any(
        value is None or value <= 0
        for value in (hidden, num_layers, vocab_size, num_heads)
    ):
        return 0

    head_dims = attention_head_dims(arch, hidden=hidden, num_heads=num_heads)
    if head_dims is None or any(value <= 0 for value in head_dims):
        return 0
    qk_head_dim, v_head_dim = head_dims

    moe = moe_geometry(arch, num_layers=num_layers)
    if moe is _UNRESOLVED:
        return 0

    attended = attention_windows(arch, num_layers=num_layers)
    if attended is None:
        return 0

    dense_intermediate = int_attr(arch, "intermediate_size")
    needs_dense_ffn = moe is None or len(moe.layers) != num_layers
    if needs_dense_ffn and (dense_intermediate is None or dense_intermediate <= 0):
        return 0

    attention_params = attention_param_term(
        arch,
        hidden=hidden,
        num_heads=num_heads,
        qk_head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
    )
    if attention_params is None:
        return 0

    matmul_flops = 0
    for layer in range(num_layers):
        matmul_flops += attention_params
        if moe is not None and layer in moe.layers:
            matmul_flops += moe_ffn_param_term(hidden=hidden, moe=moe)
        else:
            matmul_flops += 3 * 2 * hidden * dense_intermediate

    # The lm_head is the one parameter term outside the layer stack.
    matmul_flops += 2 * vocab_size * hidden
    attention_flops = sum(
        quadratic_attention_flops_per_token(
            num_heads=num_heads,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            seq_len=seq_len,
            sliding_window_size=window,
        )
        for window in attended
    )
    return 3 * matmul_flops + attention_flops


def qwen3_5_moe_text_flops_per_token(arch: PretrainedConfig, *, seq_len: int) -> int:
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


def int_attr(arch: PretrainedConfig, name: str) -> int | None:
    """An int attribute, or ``None`` when absent or not an int."""
    value = getattr(arch, name, None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def first_int_attr(arch: PretrainedConfig, *names: str) -> int | None:
    """The first of ``names`` the config spells as an int (families differ)."""
    for name in names:
        value = int_attr(arch, name)
        if value is not None:
            return value
    return None


def attention_head_dims(
    arch: PretrainedConfig, *, hidden: int, num_heads: int
) -> tuple[int, int] | None:
    """``(qk_head_dim, v_head_dim)``, or ``None`` when the config cannot say."""
    v_head_dim = first_int_attr(arch, "v_head_dim")
    qk_head_dim = first_int_attr(arch, "qk_head_dim")
    if qk_head_dim is None:
        nope = int_attr(arch, "qk_nope_head_dim")
        rope = int_attr(arch, "qk_rope_head_dim")
        qk_head_dim = None if nope is None or rope is None else nope + rope
    if qk_head_dim is not None or v_head_dim is not None:
        # MLA: half the pair is not enough to size the projections either.
        if qk_head_dim is None or v_head_dim is None:
            return None
        return qk_head_dim, v_head_dim

    head_dim = int_attr(arch, "head_dim")
    if head_dim is None:
        if hidden % num_heads:
            return None
        head_dim = hidden // num_heads
    return head_dim, head_dim


def attention_param_term(
    arch: PretrainedConfig,
    *,
    hidden: int,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
) -> int | None:
    """Per-layer attention projection cost, in units of ``2 * in * out``.

    ``None`` for an MLA config missing its KV rank or head geometry. Q can
    independently use a low-rank projection or a direct projection when
    ``q_lora_rank`` is None; K/V remain compressed in both cases.
    """
    num_kv_heads = int_attr(arch, "num_key_value_heads")
    if num_kv_heads is None:
        num_kv_heads = num_heads
    if num_kv_heads <= 0 or num_heads % num_kv_heads:
        return None
    q_lora_rank = int_attr(arch, "q_lora_rank")
    kv_lora_rank = int_attr(arch, "kv_lora_rank")
    if q_lora_rank is not None or kv_lora_rank is not None:
        qk_rope_head_dim = int_attr(arch, "qk_rope_head_dim")
        qk_nope_head_dim = int_attr(arch, "qk_nope_head_dim")
        if None in (kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim):
            return None
        # Q projects directly or through q_a/q_b; kv_b_proj splits its
        # output into the nope key and the value, so it is sized by their sum.
        q_proj = (
            hidden * num_heads * qk_head_dim
            if q_lora_rank is None
            else hidden * q_lora_rank + q_lora_rank * num_heads * qk_head_dim
        )
        kv_proj = hidden * (kv_lora_rank + qk_rope_head_dim) + kv_lora_rank * (
            num_heads * (qk_nope_head_dim + v_head_dim)
        )
    else:
        q_proj = hidden * num_heads * qk_head_dim
        kv_proj = hidden * num_kv_heads * qk_head_dim + hidden * num_kv_heads * (
            v_head_dim
        )
    o_proj = num_heads * v_head_dim * hidden
    return 2 * (q_proj + kv_proj + o_proj)


def moe_geometry(
    arch: PretrainedConfig, *, num_layers: int
) -> MoEGeometry | None | object:
    """The MoE geometry and its routed layers, ``None`` if there is no MoE.

    Returns :data:`_UNRESOLVED` for a declared MoE the config does not pin down
    -- that is the call sites' signal that the whole count must be suppressed.
    """
    num_experts = first_int_attr(
        arch, "num_experts", "n_routed_experts", "num_local_experts"
    )
    if num_experts is None:
        return None
    top_k = first_int_attr(arch, "num_experts_per_tok", "top_k")
    if num_experts <= 0 or top_k is None or not 1 <= top_k <= num_experts:
        return _UNRESOLVED

    # Dense layers: an explicit prefix (DeepSeek), an explicit index list
    # (Qwen3-MoE), or every-Nth (both, via ``moe_layer_freq``/``decoder_sparse_step``).
    first_dense = int_attr(arch, "first_k_dense_replace") or 0
    sparse_step = int_attr(arch, "decoder_sparse_step") or 1
    moe_freq = int_attr(arch, "moe_layer_freq") or 1
    mlp_only = getattr(arch, "mlp_only_layers", None) or ()
    layers = frozenset(
        layer
        for layer in range(num_layers)
        if layer not in mlp_only
        and layer >= first_dense
        and (layer + 1) % sparse_step == 0
        and layer % moe_freq == 0
    )
    if not layers:
        return _UNRESOLVED

    # Every expert's width: the MoE field, or the single width the config uses
    # for every FFN when no layer is dense (Mixtral, OLMoE, GPT-OSS).
    uniform_width = not first_dense and not mlp_only and sparse_step == 1
    expert_intermediate = int_attr(arch, "moe_intermediate_size")
    if expert_intermediate is None and uniform_width:
        expert_intermediate = int_attr(arch, "intermediate_size")
    if expert_intermediate is None or expert_intermediate <= 0:
        return _UNRESOLVED

    shared_intermediate = int_attr(arch, "shared_expert_intermediate_size")
    if shared_intermediate is None:
        n_shared = int_attr(arch, "n_shared_experts") or 0
        shared_intermediate = n_shared * expert_intermediate

    if shared_intermediate < 0:
        return _UNRESOLVED

    if len(layers) != num_layers and int_attr(arch, "intermediate_size") is None:
        return _UNRESOLVED
    return MoEGeometry(
        num_experts=num_experts,
        top_k=top_k,
        expert_intermediate=expert_intermediate,
        shared_intermediate=shared_intermediate,
        layers=layers,
    )


def moe_ffn_param_term(*, hidden: int, moe: MoEGeometry) -> int:
    """A MoE layer's matmul cost, in units of ``2 * in * out``.

    Router + ``top_k`` routed experts + *all* shared experts: the routed
    experts are stored ``num_experts`` times over but only ``top_k`` of them
    see each token, which is exactly the ``top_k / num_experts`` active-expert
    ratio upstream's parameter walk applies.
    """
    router = hidden * moe.num_experts
    routed = moe.top_k * 3 * hidden * moe.expert_intermediate
    shared = 3 * hidden * moe.shared_intermediate
    return 2 * (router + routed + shared)


def attention_windows(
    arch: PretrainedConfig, *, num_layers: int
) -> list[int | None] | None:
    """Per-layer attention window (``None`` = full), or ``None`` if unresolved."""
    layer_types = getattr(arch, "layer_types", None)
    if layer_types is None:
        sliding_window = int_attr(arch, "sliding_window")
        # ``use_sliding_window=False`` is the explicit opt-out; a window that is
        # declared and not opted out of is what the model's own attention reads.
        if sliding_window is not None and getattr(arch, "use_sliding_window", True):
            if sliding_window <= 0:
                return None
            return [sliding_window] * num_layers
        return [None] * num_layers

    layer_types = list(layer_types)
    if len(layer_types) < num_layers:
        # A config that declares fewer types than the layers it builds does not
        # say what the remaining layers do; upstream substitutes full attention,
        # which is only safe when that is really the default.
        return None
    windows: list[int | None] = []
    for layer_type in layer_types[:num_layers]:
        if layer_type in ("attention", "full_attention"):
            windows.append(None)
        elif layer_type == "sliding_attention":
            window = int_attr(arch, "sliding_window")
            if window is None or window <= 0:
                return None
            windows.append(window)
        elif layer_type == "chunked_attention":
            chunk = int_attr(arch, "attention_chunk_size")
            if chunk is None or chunk <= 0:
                return None
            windows.append(chunk)
        else:
            # ``linear_attention`` and any future spelling: its projection term
            # is not derivable from these generic fields.
            return None
    return windows


def kimi_linear_flops_per_token(arch: PretrainedConfig, *, seq_len: int) -> int:
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
