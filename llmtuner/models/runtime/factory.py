"""Load model configurations and resolve exclusively registered local models.

External configuration formats remain supported. Loading a configuration does not
permit automatic external model construction; resolve_model_class enforces the
production registry boundary before any model allocation.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from transformers.configuration_utils import PretrainedConfig

from ..registry import get_model_spec
from .attention_backend import ATTENTION_BACKEND

_RUNTIME_CONFIG_FIELDS = (
    "attn_mask_type",
    "experts_implementation",
    "router_compute_dtype",
    "compute_dtype",
    "norm_backend",
    "linear_attention_backend",
    "gdn_numerical_policy",
    "gdn_allow_right_padding",
    "gdn_deterministic_reduction",
    "qkv_projection",
    "rotary_init_device",
    "rotary_compute_dtype",
    "rotary_cache_dtype",
    "attention_backend",
)

if TYPE_CHECKING:
    from torch import nn

    from ...config.llmtuner_config import LLMTunerConfig

__all__ = [
    "build_model_config",
    "build_model_config_for",
    "resolve_model_class",
    "unwrap_text_config",
]


def build_model_config(
    model_name_or_path: str,
    *,
    seq_len: int,
    arch_overrides: dict[str, Any] | None = None,
) -> PretrainedConfig:
    """Build the HF config the wrapper will instantiate.

    ``arch_overrides`` selects offline architecture construction, including an
    empty dict to use architecture defaults. ``None`` loads the config from a
    local checkpoint directory or Hub identifier.
    """
    architecture = get_model_spec(model_name_or_path) or get_model_spec(
        model_name_or_path, task="conditional_generation"
    )
    # Some HF composite constructors modify nested configuration dictionaries.
    overrides = deepcopy(arch_overrides) if arch_overrides is not None else {}
    if arch_overrides is not None:
        overrides.setdefault("max_position_embeddings", seq_len)
        if architecture is None:
            raise ValueError(
                f"No local configuration registered for {model_name_or_path!r}"
            )
        else:
            config = architecture.build_config(**overrides)
        if hasattr(config, "text_config"):
            config.text_config.max_position_embeddings = max(
                getattr(config.text_config, "max_position_embeddings", seq_len), seq_len
            )
        return config

    # Read external metadata, but let only the registered config class load it.
    # Do not mutate AutoConfig's process-global registry during a training run.
    metadata, _ = PretrainedConfig.get_config_dict(model_name_or_path)
    model_type = metadata.get("model_type", "")
    architecture = get_model_spec(model_type) or get_model_spec(
        model_type, task="conditional_generation"
    )
    if architecture is None:
        raise ValueError(f"No local configuration registered for {model_type!r}")
    config = architecture.config_type_loader().from_pretrained(model_name_or_path)
    text_config = getattr(config, "text_config", config)
    # Custom configs need not declare a positional limit. Match offline
    # construction, which supplies the requested length when it is absent.
    text_config.max_position_embeddings = max(
        getattr(text_config, "max_position_embeddings", seq_len), seq_len
    )
    return config


def apply_runtime_config(config: PretrainedConfig, cfg: LLMTunerConfig) -> None:
    config.attn_mask_type = cfg.dataloader.attention_mask_type()
    config.experts_implementation = getattr(cfg, "experts_implementation", "native")
    config.router_compute_dtype = getattr(cfg.model, "router_compute_dtype", None)
    config.compute_dtype = getattr(cfg, "compute_dtype", None)
    config.norm_backend = cfg.norm_backend
    config.linear_attention_backend = cfg.model.linear_attention_backend
    config.gdn_numerical_policy = getattr(cfg.model, "gdn_numerical_policy", "native")
    config.gdn_allow_right_padding = getattr(
        cfg.model, "gdn_allow_right_padding", False
    )
    config.gdn_deterministic_reduction = getattr(
        cfg.model, "gdn_deterministic_reduction", False
    )
    config.qkv_projection = cfg.qkv_projection
    config.rotary_init_device = cfg.rotary_init_device
    config.rotary_compute_dtype = cfg.model.rotary_compute_dtype
    config.rotary_cache_dtype = cfg.model.rotary_cache_dtype
    config.attention_backend = cfg.attention_backend


def build_model_config_for(cfg: LLMTunerConfig) -> PretrainedConfig:
    """Build architecture and runtime settings for a training run.

    Registered architecture names use offline construction with explicit model
    sizes and ``arch_overrides``. Hub IDs and existing checkpoint paths keep
    their saved architecture; only the supported sequence length is extended.

    Derive the attention mask from the dataloader's packing policy. Multiple
    documents require block-causal attention; synthetic data and one-document
    rows use ordinary causality. Forward the requested experts implementation
    and lm-head compute dtype for validation during model construction.
    """
    # Only registered architecture names select offline construction. Missing
    # paths and Hub IDs belong to from_pretrained, which owns their validation.
    offline = not os.path.exists(cfg.hf_model) and (
        get_model_spec(cfg.hf_model) is not None
        or get_model_spec(cfg.hf_model, task="conditional_generation") is not None
    )
    overrides = (
        {
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            # Last, so an architecture setting may correct one of the six --
            # an MoE's hidden width and its per-expert width are independent,
            # but a model that redefines ``intermediate_size`` can say so
            # rather than have the override silently lose.
            **cfg.arch_overrides,
        }
        if offline
        else None
    )
    if overrides is not None:
        architecture = get_model_spec(cfg.hf_model) or get_model_spec(
            cfg.hf_model, task="conditional_generation"
        )
        config_type = architecture.config_type_loader()
        # HF may assign extra kwargs after its canonical fields. A generic
        # default alias must not overwrite an explicit architecture dimension
        # (e.g. intermediate_size -> moe_intermediate_size in DeepSeek V4).
        for alias, canonical in config_type.attribute_map.items():
            if canonical in cfg.arch_overrides and alias not in cfg.arch_overrides:
                overrides.pop(alias, None)
    config = build_model_config(
        cfg.hf_model, seq_len=cfg.max_seq_len, arch_overrides=overrides
    )
    # A real corpus normally packs several documents and needs block-causal
    # masking. ``max_num_documents=1`` is the deliberate tensor-attention path:
    # one padded SFT document per row needs only ordinary causality, which lets
    # NPU/SDPA train without allowing attention across sample boundaries.
    apply_runtime_config(config, cfg)
    return config


def unwrap_text_config(config: PretrainedConfig) -> PretrainedConfig:
    """Return the text sub-config of a composite (vision-language) model.

    A VL checkpoint's top config describes the conditional model, not the text
    stack we train; its ``text_config`` does. Model lookup uses the registered
    model_type, never external class names in an architectures field.
    """
    if not hasattr(config, "text_config"):
        return config
    text_config = config.text_config
    text_config._attn_implementation = ATTENTION_BACKEND
    # Runtime settings originate on the composite config. The text model must
    # see the same mask policy and kernels after unwrapping.
    for name in _RUNTIME_CONFIG_FIELDS:
        if hasattr(config, name):
            setattr(text_config, name, getattr(config, name))
    return text_config


def resolve_model_class(config: PretrainedConfig) -> type[nn.Module]:
    """Resolve a registered local implementation without an external fallback."""
    specification = get_model_spec(config.model_type)
    if specification is None:
        raise ValueError(
            f"No local model registered for {config.model_type!r}; "
            "register an explicit implementation before training. "
            "Automatic Transformers model fallback is disabled."
        )
    return specification.resolve_model_class()
