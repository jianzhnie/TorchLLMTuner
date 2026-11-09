"""Strict single-row GDN padding metadata checks; never alter activations.

HF attention masks use True for valid tokens; framework padding masks use True
for padding. This helper does not enable a backend or numerical policy.
"""

from __future__ import annotations

import torch

from ...errors import UnsupportedCombinationError

QWEN35_GDN_GEOMETRY = {
    "model_type": "qwen3_5_moe_text",
    "hidden_size": 2048,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
}


def validate_qwen35_gdn_geometry(config, *, backend: str) -> None:
    for name, value in QWEN35_GDN_GEOMETRY.items():
        if getattr(config, name, None) != value:
            raise UnsupportedCombinationError(
                f"{backend} requires verified Qwen3.5 MoE geometry: {name}={value}"
            )


def validate_gdn_sequence_metadata(
    inputs,
    positions,
    masks,
    padding,
    *,
    allow_right_padding: bool = False,
    validate_metadata: bool = True,
) -> None:
    if inputs.ndim != 1 or inputs.numel() <= 0 or inputs.numel() % 64:
        raise UnsupportedCombinationError(
            "GDN requires a flat, nonempty sequence divisible by 64"
        )
    if validate_metadata:
        validate_gdn_padding_metadata(
            batch_size=1,
            sequence_length=inputs.numel(),
            positions=positions,
            padding_mask=padding,
            attention_mask=masks,
            allow_right_padding=allow_right_padding,
        )


def validate_gdn_forward_metadata(module, args, kwargs):
    hidden = args[0] if args else kwargs["hidden_states"]
    allowed = {
        "hidden_states",
        "cache_params",
        "attention_mask",
        "use_cache",
        "position_ids",
        "cache_position",
        "output_router_logits",
        "output_attentions",
        "output_hidden_states",
        "is_causal",
    }
    for key in ("output_router_logits", "output_attentions", "output_hidden_states"):
        if key in kwargs and type(kwargs[key]) is not bool:
            raise UnsupportedCombinationError(f"GDN {key} must be bool")
    if "is_causal" in kwargs and kwargs["is_causal"] is not True:
        raise UnsupportedCombinationError("GDN requires is_causal=True")
    allow_padding = getattr(module, "_llmtuner_gdn_allow_right_padding", False)
    if kwargs.get("attention_mask") is not None and not allow_padding:
        raise UnsupportedCombinationError("GDN masks require right-padding opt-in")
    if (
        len(args) > 1
        or kwargs.keys() - allowed
        or kwargs.get("cache_params") is not None
        or kwargs.get("use_cache", False) not in (None, False)
    ):
        raise UnsupportedCombinationError(
            "GDN accepts unpacked training without cache or extra metadata"
        )
    if (
        hidden.ndim != 3
        or hidden.shape[0] != 1
        or hidden.shape[1] <= 0
        or hidden.shape[1] % 64
    ):
        raise UnsupportedCombinationError(
            "GDN requires one row and a nonempty sequence divisible by 64"
        )
    validate_gdn_padding_metadata(
        batch_size=1,
        sequence_length=hidden.shape[1],
        positions=kwargs.get("position_ids"),
        attention_mask=kwargs.get("attention_mask"),
        allow_right_padding=allow_padding,
    )
    position = kwargs.get("cache_position")
    if position is not None and (
        position.dtype not in (torch.int32, torch.int64)
        or position.numel() != hidden.shape[1]
        or not bool(
            (
                position.reshape(-1)
                == torch.arange(hidden.shape[1], device=position.device)
            ).all()
        )
    ):
        raise UnsupportedCombinationError(
            "GDN requires contiguous cache positions from zero"
        )
    return hidden


def validate_gdn_padding_config(config) -> None:
    allow = getattr(config, "gdn_allow_right_padding", False)
    if not isinstance(allow, bool) or (
        allow
        and (
            getattr(config, "linear_attention_backend", "native") != "npu_triton"
            or getattr(config, "gdn_numerical_policy", "native")
            != "split_qkv_fp32_norm"
        )
    ):
        raise UnsupportedCombinationError(
            "gdn_allow_right_padding requires boolean opt-in and "
            "npu_triton + split_qkv_fp32_norm"
        )


def validate_gdn_padding_metadata(
    *,
    batch_size: int,
    sequence_length: int,
    positions: torch.Tensor | None = None,
    padding_mask: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    allow_right_padding: bool = False,
) -> None:
    """Accept only B1 contiguous valid positions and a nonempty valid prefix.

    A framework padding mask may be flat or [1,L]; HF attention_mask must be
    [1,L], boolean or integer binary. Both provided masks must agree exactly.
    Missing positions mean the caller's ordinary contiguous uncached sequence.
    No masking, tensor casting in the model, reset, or cache behavior is added.
    """
    if batch_size != 1 or sequence_length <= 0:
        raise UnsupportedCombinationError("GDN requires one nonempty batch row")
    if not isinstance(allow_right_padding, bool):
        raise UnsupportedCombinationError("GDN allow_right_padding must be boolean")
    padding = None
    if padding_mask is not None:
        if (
            not isinstance(padding_mask, torch.Tensor)
            or padding_mask.dtype != torch.bool
            or tuple(padding_mask.shape)
            not in ((sequence_length,), (1, sequence_length))
        ):
            raise UnsupportedCombinationError(
                "GDN padding_mask must be boolean [L]/[1,L] with one entry per token"
            )
        padding = padding_mask.reshape(-1)
    if attention_mask is not None:
        if (
            not isinstance(attention_mask, torch.Tensor)
            or tuple(attention_mask.shape) != (1, sequence_length)
            or attention_mask.dtype
            not in (
                torch.bool,
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            )
            or not bool(((attention_mask == 0) | (attention_mask == 1)).all())
        ):
            raise UnsupportedCombinationError("GDN attention_mask must be binary [1,L]")
        if not allow_right_padding:
            raise UnsupportedCombinationError(
                "GDN attention masks require explicit opt-in"
            )
        from_attention = attention_mask.reshape(-1) == 0
        if padding is not None and not torch.equal(
            padding, from_attention.to(padding.device)
        ):
            raise UnsupportedCombinationError("GDN attention/padding masks disagree")
        padding = from_attention
    if padding is not None:
        if bool(padding[0]) or bool((padding[:-1] & ~padding[1:]).any()):
            raise UnsupportedCombinationError(
                "GDN requires nonempty valid prefix and suffix-only right padding"
            )
        if bool(padding.any()) and not allow_right_padding:
            raise UnsupportedCombinationError(
                "GDN right padding awaits full-module production validation; "
                "requires explicit opt-in"
            )

    if positions is not None:
        if (
            not isinstance(positions, torch.Tensor)
            or positions.dtype not in (torch.int32, torch.int64)
            or tuple(positions.shape) not in ((sequence_length,), (1, sequence_length))
        ):
            raise UnsupportedCombinationError(
                "GDN requires integer positions [L]/[1,L]"
            )
        # Only an explicitly verified padding suffix may restart positions.
        # All unmasked tokens remain a single document starting at zero.
        valid_length = sequence_length if padding is None else int((~padding).sum())
        prefix = positions.reshape(-1)[:valid_length]
        if not torch.equal(prefix, torch.arange(valid_length, device=prefix.device)):
            raise UnsupportedCombinationError(
                "GDN requires integer contiguous positions starting at zero "
                "on the valid prefix"
            )
        if bool((positions < 0).any()):
            raise UnsupportedCombinationError("GDN positions must be nonnegative")
