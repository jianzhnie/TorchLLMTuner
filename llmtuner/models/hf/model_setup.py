"""Construction-time validation and adaptation for Hugging Face models."""

from __future__ import annotations

import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig

from ...config.model_config import LM_HEAD_COMPUTE_DTYPES
from ...utils.dtypes import get_dtype
from ..common.cast_linear import build_cast_linear


def resolve_submodule_name(
    module: nn.Module, names: tuple[str, ...], *, component: str
) -> str:
    """Return the first candidate attribute holding an nn.Module.

    HF names the same submodule differently across model families
    (``embed_tokens``/``wte``, ``norm``/``final_layernorm``/``ln_f``), so the
    wrapper probes instead of hardcoding one family's spelling. Resolved once at
    construction: a missing part should fail at build time, not mid-forward.
    """
    for name in names:
        if isinstance(getattr(module, name, None), nn.Module):
            return name
    raise AttributeError(
        f"{type(module).__name__} has no {component} module under any of {names}. "
        "Add the model's spelling to the probe in HFTransformerModel.__init__."
    )


def validate_head_config(config: PretrainedConfig) -> None:
    """Reject GQA head counts that no attention kernel can honor."""
    num_heads = getattr(config, "num_attention_heads", None)
    num_kv_heads = getattr(config, "num_key_value_heads", None)
    num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    for name, value in (
        ("num_attention_heads", num_heads),
        ("num_key_value_heads", num_kv_heads),
    ):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if (
        num_heads is not None
        and num_kv_heads is not None
        and num_heads % num_kv_heads != 0
    ):
        raise ValueError(
            f"num_attention_heads ({num_heads}) must be divisible by "
            f"num_key_value_heads ({num_kv_heads})"
        )


def configure_experts_implementation(
    config: PretrainedConfig, *, model_cls: type
) -> None:
    """Set the requested HF experts kernel on the construction config.

    ``native`` preserves the model's built-in kernel. Other choices require
    Transformers' settable experts implementation API. Under EP, llmtuner
    replaces the MoE block after construction.
    """
    impl = getattr(config, "experts_implementation", "native")
    if impl == "native":
        return
    can_set = getattr(model_cls, "_can_set_experts_implementation", None)
    if impl not in ("grouped_mm", "batched_mm", "eager"):
        raise ValueError(
            f"experts_implementation must be one of 'native', "
            f"'grouped_mm', 'batched_mm', 'eager', got '{impl}'"
        )
    if can_set is None or not can_set():
        raise ValueError(
            f"{model_cls.__name__} does not support a settable experts "
            f"implementation, so experts_implementation='{impl}' cannot "
            "be honored. Set experts_implementation='native' to use "
            "the HF model's built-in experts kernel."
        )
    config._experts_implementation = impl


def apply_cast_lm_head(model: nn.Module, config: PretrainedConfig) -> None:
    """Optional fixed-dtype lm_head (torchtitan's CastLinear semantics):
    score the vocabulary logits in the requested dtype while the stored
    weight keeps its own. Swapped in place rather than wrapped so the
    state-dict FQNs -- and a weight tie with the embedding -- are
    untouched (see models/common/cast_linear.py for why subclassing
    beats wrapping)."""
    compute_dtype = getattr(config, "compute_dtype", None)
    if compute_dtype is None:
        return
    if compute_dtype not in LM_HEAD_COMPUTE_DTYPES:
        raise ValueError(
            f"compute_dtype must be one of {sorted(LM_HEAD_COMPUTE_DTYPES)}, "
            f"got {compute_dtype!r}"
        )
    if model.lm_head is not None:
        model.lm_head = build_cast_linear(
            model.lm_head, compute_dtype=get_dtype(compute_dtype)
        )
