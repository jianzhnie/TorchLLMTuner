"""Generic construction-time validation and adaptation for HF models."""

from __future__ import annotations

import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig

from ...config.model_config import LM_HEAD_COMPUTE_DTYPES
from ...utils.dtypes import get_dtype
from ...utils.logging import get_logger
from ..common.cast_linear import build_cast_linear

logger = get_logger(__name__)

MOE_CONFIG_ATTRIBUTES = (
    "num_local_experts",
    "num_experts",
    "n_routed_experts",
    "moe_intermediate_size",
)


def is_moe_config(config: PretrainedConfig) -> bool:
    """Whether ``config`` declares a mixture-of-experts model."""
    for name in MOE_CONFIG_ATTRIBUTES:
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return True
    return False


def resolve_submodule_name(
    module: nn.Module, names: tuple[str, ...], *, component: str
) -> str:
    """Return the first candidate attribute holding an ``nn.Module``."""
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
    """Set the requested HF experts kernel on the construction config."""
    impl = getattr(config, "experts_implementation", "native")
    if impl == "native":
        return
    if impl == "pre_down":
        if config.model_type != "qwen3_5_moe_text" or config.hidden_act != "silu":
            raise ValueError("pre_down experts require Qwen3.5 MoE with SiLU")
        config._experts_implementation = "eager"
        return
    if impl == "torchtitan":
        if config.model_type != "qwen3_moe":
            raise ValueError("experts_implementation='torchtitan' requires Qwen3-MoE")
        config._experts_implementation = "eager"
        return
    if impl not in ("grouped_mm", "batched_mm", "eager"):
        raise ValueError(
            f"experts_implementation must be one of 'native', "
            f"'grouped_mm', 'batched_mm', 'eager', 'torchtitan', got '{impl}'"
        )
    can_set = getattr(model_cls, "_can_set_experts_implementation", None)
    if can_set is None:
        if impl == "eager" and is_moe_config(config):
            logger.info(
                "%s has no settable experts implementation (transformers < 5); "
                "its built-in experts path is the per-expert loop, so "
                "experts_implementation='eager' is already in effect.",
                model_cls.__name__,
            )
            return
        raise ValueError(
            f"{model_cls.__name__} does not support a settable experts "
            f"implementation, so experts_implementation='{impl}' cannot be "
            "honored. Set experts_implementation='native' to use the HF "
            "model's built-in experts kernel. This transformers build has no "
            "_can_set_experts_implementation at all -- that API first ships "
            "with transformers 5, and below it a MoE block always runs the "
            "per-expert loop, which is what 'eager' names."
        )
    if not can_set():
        raise ValueError(
            f"{model_cls.__name__} does not support a settable experts "
            f"implementation, so experts_implementation='{impl}' cannot "
            "be honored. Set experts_implementation='native' to use the HF "
            "model's built-in experts kernel."
        )
    config._experts_implementation = impl


def apply_cast_lm_head(model: nn.Module, config: PretrainedConfig) -> None:
    """Apply the optional fixed-dtype lm-head without changing state keys."""
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
