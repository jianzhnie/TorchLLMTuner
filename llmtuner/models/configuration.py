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

from ..errors import UnsupportedCombinationError
from .registry import get_model_spec

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

    from ..config.llmtuner_config import LLMTunerConfig

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
) -> Any:
    """Build a registered model configuration without importing its implementation.

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
        config = architecture.build_config(**overrides)
    else:
        # Read external metadata, but only the registered config class loads it.
        # Do not mutate AutoConfig's process-global registry during training.
        metadata, _ = PretrainedConfig.get_config_dict(model_name_or_path)
        model_type = metadata.get("model_type", "")
        architecture = get_model_spec(model_type) or get_model_spec(
            model_type, task="conditional_generation"
        )
        if architecture is None:
            raise ValueError(f"No local configuration registered for {model_type!r}")
        config = architecture.config_type_loader().from_pretrained(model_name_or_path)
        if getattr(config, "model_type", None) != architecture.model_type:
            raise ValueError(
                f"Loaded configuration for {architecture.model_type!r} returned "
                f"model_type={getattr(config, 'model_type', None)!r}"
            )
    text_config = getattr(config, "text_config", config)
    # Apply the same context requirement to offline and saved configurations.
    # Only the text stack is extended; composite vision settings stay intact.
    text_config.max_position_embeddings = max(
        getattr(text_config, "max_position_embeddings", seq_len), seq_len
    )
    return config


def validate_native_execution_policy(
    config: PretrainedConfig, *, model_name: str
) -> None:
    """Validate a family's explicit choice to support native projection policies.

    The caller opts into this boundary; no model identification or adapter
    installation occurs here. Families implementing other policies validate
    those in their own components instead (for example explicit Qwen3).
    """
    for name, default in (
        ("norm_backend", "native"),
        ("qkv_projection", "separate"),
        ("rotary_init_device", "constructor"),
        ("rotary_compute_dtype", None),
        ("rotary_cache_dtype", None),
    ):
        requested = getattr(config, name, default)
        if requested != default:
            raise UnsupportedCombinationError(
                f"Explicit {model_name} {name}={requested!r} has no model-owned "
                f"implementation; supported value is {default!r}"
            )


def validate_attention_backend(config: PretrainedConfig) -> None:
    """Reject removed wrapper backends instead of silently executing another.

    Native attention selection remains owned by each explicit model and its
    reference components. The former wrapper-specific flex adapters must be
    migrated before their user-facing selectors can describe real execution.
    """
    validate_attention_backend_choice(getattr(config, "attention_backend", "auto"))


def validate_attention_backend_choice(requested: str) -> None:
    """Validate the requested selector without mutating a model configuration."""
    if requested != "auto":
        raise UnsupportedCombinationError(
            f"attention_backend={requested!r} has no explicit-model adapter yet; "
            "the former HF wrapper adapter was removed. Use 'auto' or implement "
            "and validate a model-owned attention backend before selecting it."
        )


def apply_runtime_config(config: Any, cfg: LLMTunerConfig) -> None:
    """Validate backend selection before applying runtime fields to external config.

    Rejected selectors must not leave caller-owned checkpoint metadata partially
    rewritten. Model-specific geometry remains validated by the model owner.
    """
    validate_attention_backend_choice(cfg.attention_backend)
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


def configure_reference_experts(config: PretrainedConfig) -> None:
    """Apply an explicit expert kernel choice before reference construction.

    The local model owns this call. Native leaves the reference's default or
    saved choice intact; supported overrides use its instance configuration,
    never its global backend registry. Custom expert math needs a model-owned
    implementation instead of silently inheriting another backend.
    """
    requested = getattr(config, "experts_implementation", "native")
    if requested == "native":
        return
    if requested not in ("eager", "grouped_mm", "batched_mm"):
        raise ValueError(
            "This explicit model supports experts_implementation native, eager, "
            "grouped_mm or batched_mm; "
            f"{requested!r} requires its own model-owned expert implementation."
        )
    config._experts_implementation = requested


def build_model_config_for(cfg: LLMTunerConfig) -> Any:
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
        # External configs may expose HF field aliases; plain model-owned
        # config classes need not implement Transformers' attribute_map.
        # HF may assign extra kwargs after its canonical fields. A generic
        # default alias must not overwrite an explicit architecture dimension
        # (e.g. intermediate_size -> moe_intermediate_size in DeepSeek V4).
        for alias, canonical in getattr(config_type, "attribute_map", {}).items():
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


def unwrap_text_config(config: Any) -> Any:
    """Return the text sub-config of a composite (vision-language) model.

    A VL checkpoint's top config describes the conditional model, not the text
    stack we train; its ``text_config`` does. Model lookup uses the registered
    model_type, never external class names in an architectures field.
    """
    if not hasattr(config, "text_config"):
        return config
    text_config = config.text_config
    # Runtime settings originate on the composite config. The text model must
    # see the same mask policy and kernels after unwrapping.
    for name in _RUNTIME_CONFIG_FIELDS:
        if hasattr(config, name):
            setattr(text_config, name, getattr(config, name))
    return text_config


def resolve_model_class(config: Any) -> type[nn.Module]:
    """Resolve a registered local implementation without an external fallback."""
    specification = get_model_spec(config.model_type)
    if specification is None:
        raise ValueError(
            f"No local model registered for {config.model_type!r}; "
            "register an explicit implementation before training. "
            "Automatic Transformers model fallback is disabled."
        )
    return specification.resolve_model_class()
