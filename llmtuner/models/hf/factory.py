"""Build Hugging Face model configs and resolve their model classes.

The helpers build the HF ``PretrainedConfig`` and resolve its ``ForCausalLM``
class. Meta storage allocation lives in ``materialization.py``; FLOPs estimation
lives in ``llmtuner.utils.flops``. The model body and five-part contract
live in ``model.py``, which imports these helpers; nothing here imports the model
body.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import transformers
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

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
    architecture = get_model_spec(model_name_or_path)
    # Some HF composite constructors modify nested configuration dictionaries.
    overrides = deepcopy(arch_overrides) if arch_overrides is not None else {}
    if arch_overrides is not None:
        overrides.setdefault("max_position_embeddings", seq_len)
        if architecture is None:
            config = AutoConfig.for_model(model_name_or_path, **overrides)
        else:
            config = architecture.build_config(**overrides)
        if hasattr(config, "text_config"):
            config.text_config.max_position_embeddings = max(
                getattr(config.text_config, "max_position_embeddings", seq_len), seq_len
            )
        return config

    # Resolve only the selected architecture's hooks before constructing any
    # config. HF may still cache an older class after an exist_ok replacement.
    metadata, _ = PretrainedConfig.get_config_dict(model_name_or_path)
    architecture = get_model_spec(metadata.get("model_type", ""))
    if architecture is not None:
        # Refresh even an existing HF entry: exist_ok registry replacement must
        # also take effect when loading a saved configuration.
        AutoConfig.register(
            architecture.model_type,
            architecture.config_type_loader(),
            exist_ok=True,
        )
    config = AutoConfig.from_pretrained(model_name_or_path)
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
        cfg.hf_model in CONFIG_MAPPING or get_model_spec(cfg.hf_model) is not None
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
        architecture = get_model_spec(cfg.hf_model)
        config_type = (
            architecture.config_type_loader()
            if architecture is not None
            else CONFIG_MAPPING[cfg.hf_model]
        )
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
    stack we train; its ``text_config`` does. Also ensure the sub-config carries
    ``architectures``, or the class lookup below has nothing to resolve.
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
    if not getattr(text_config, "architectures", None):
        cls_name = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(text_config.model_type, "")
        if cls_name:
            text_config.architectures = [cls_name]
    return text_config


def resolve_model_class(config: PretrainedConfig) -> type[nn.Module]:
    """Prefer a model-owned implementation, then resolve the HF model class.

    Registered implementations need not inherit PreTrainedModel. The HF
    fallback prefers the declared class name, then its model-type mapping.
    """
    architecture = get_model_spec(config.model_type)
    if architecture is not None:
        return architecture.resolve_model_class()

    for name in (config.architectures or []) + [
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(getattr(config, "model_type", ""), ""),
    ]:
        if not name:
            continue
        cls = getattr(transformers, name, None)
        if isinstance(cls, type) and issubclass(cls, transformers.PreTrainedModel):
            return cls

    raise ImportError(
        f"Could not resolve a ForCausalLM class for model_type "
        f"{getattr(config, 'model_type', None)!r} "
        f"(architectures={getattr(config, 'architectures', None)!r}). "
        "Check that the installed transformers version ships this model."
    )
