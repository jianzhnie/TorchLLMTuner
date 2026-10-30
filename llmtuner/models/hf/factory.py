"""Build Hugging Face model configs and resolve their model classes.

The helpers build the HF ``PretrainedConfig`` and resolve its ``ForCausalLM``
class. Meta storage allocation lives in ``materialization.py``; FLOPs estimation
lives in ``flops.py``. The model body and five-part contract live in ``model.py``,
which imports these helpers; nothing here imports the model body.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from .attention_backend import ATTENTION_BACKEND

if TYPE_CHECKING:
    from transformers import PreTrainedModel

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
    # Some HF composite constructors modify nested configuration dictionaries.
    overrides = deepcopy(arch_overrides) if arch_overrides is not None else {}
    if arch_overrides is not None:
        overrides.setdefault("max_position_embeddings", seq_len)
        if model_name_or_path == "kimi_linear":
            from ..kimi.kimi_linear_config import KimiLinearConfig

            config = KimiLinearConfig(**overrides)
        else:
            config = AutoConfig.for_model(model_name_or_path, **overrides)
        if hasattr(config, "text_config"):
            config.text_config.max_position_embeddings = max(
                config.text_config.max_position_embeddings, seq_len
            )
        return config

    # Register only the lightweight local config; model/kernel imports stay lazy.
    from ..kimi.kimi_linear_config import KimiLinearConfig

    AutoConfig.register("kimi_linear", KimiLinearConfig, exist_ok=True)
    config = AutoConfig.from_pretrained(model_name_or_path)
    text_config = getattr(config, "text_config", config)
    text_config.max_position_embeddings = max(
        text_config.max_position_embeddings, seq_len
    )
    return config


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
        cfg.hf_model in CONFIG_MAPPING or cfg.hf_model == "kimi_linear"
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
    config = build_model_config(
        cfg.hf_model, seq_len=cfg.max_seq_len, arch_overrides=overrides
    )
    # A real corpus normally packs several documents and needs block-causal
    # masking. ``max_num_documents=1`` is the deliberate tensor-attention path:
    # one padded SFT document per row needs only ordinary causality, which lets
    # NPU/SDPA train without allowing attention across sample boundaries.
    config.attn_mask_type = cfg.dataloader.attention_mask_type()
    # The requested HF experts kernel travels on the config; the wrapper
    # validates settable-ness against the resolved model class at build time.
    config.experts_implementation = getattr(cfg, "experts_implementation", "native")
    # Same ride for the lm_head compute-dtype cast: the wrapper applies it at
    # build time, and None (the default) means no cast.
    config.compute_dtype = getattr(cfg, "compute_dtype", None)
    config.norm_backend = cfg.norm_backend
    config.qkv_projection = cfg.qkv_projection
    config.rotary_init_device = cfg.rotary_init_device
    config.rotary_compute_dtype = cfg.model.rotary_compute_dtype
    config.rotary_cache_dtype = cfg.model.rotary_cache_dtype
    config.attention_backend = cfg.attention_backend
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
    for name in (
        "attn_mask_type",
        "experts_implementation",
        "compute_dtype",
        "norm_backend",
        "qkv_projection",
        "rotary_init_device",
        "rotary_compute_dtype",
        "rotary_cache_dtype",
        "attention_backend",
    ):
        if hasattr(config, name):
            setattr(text_config, name, getattr(config, name))
    if not getattr(text_config, "architectures", None):
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        )

        cls_name = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(text_config.model_type, "")
        if cls_name:
            text_config.architectures = [cls_name]
    return text_config


def resolve_model_class(config: PretrainedConfig) -> type[PreTrainedModel]:
    """Find the ``ForCausalLM`` class named by the config.

    Prefer the name the config declares; fall back to the ``model_type`` ->
    ``Auto`` mapping, which covers configs whose architecture name does not match
    the installed class name.
    """
    if config.model_type == "kimi_linear":
        from ..kimi.model import KimiLinearForCausalLM

        return KimiLinearForCausalLM

    import transformers
    from transformers import PreTrainedModel
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    )

    for name in (config.architectures or []) + [
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(getattr(config, "model_type", ""), ""),
    ]:
        if not name:
            continue
        cls = getattr(transformers, name, None)
        if isinstance(cls, type) and issubclass(cls, PreTrainedModel):
            return cls

    raise ImportError(
        f"Could not resolve a ForCausalLM class for model_type "
        f"{getattr(config, 'model_type', None)!r} "
        f"(architectures={getattr(config, 'architectures', None)!r}). "
        "Check that the installed transformers version ships this model."
    )
