"""Build Hugging Face model configs and resolve their model classes.

The helpers build the HF ``PretrainedConfig`` and resolve its ``ForCausalLM``
class. Meta storage allocation lives in ``materialization.py``; FLOPs estimation
lives in ``flops.py``. The model body and five-part contract live in ``model.py``,
which imports these helpers; nothing here imports the model body.
"""

from __future__ import annotations

import os
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
    overrides = dict(arch_overrides or {})
    if arch_overrides is not None:
        overrides.setdefault("max_position_embeddings", seq_len)
        return AutoConfig.for_model(model_name_or_path, **overrides)

    config = AutoConfig.from_pretrained(model_name_or_path)
    text_config = getattr(config, "text_config", config)
    text_config.max_position_embeddings = max(
        text_config.max_position_embeddings, seq_len
    )
    return config


def build_model_config_for(cfg: LLMTunerConfig) -> PretrainedConfig:
    """Build the model config for a training run.

    A thin adapter over :func:`build_model_config`: it maps the run's config onto
    the explicit ``(name, seq_len, arch_overrides)`` that function takes, and
    handles the offline case where the name is a bare architecture ("llama")
    rather than a hub id ("org/name"). Offline, the explicit sizes in ``cfg`` are
    authoritative, so they become the overrides; otherwise the Hub's own config
    wins and the overrides are empty.

    ``cfg.arch_overrides`` rides along on top of the explicit sizes, and exists
    because those sizes are only the dense-decoder six: an MoE or an MLA
    attention has fields -- ``n_routed_experts``, ``q_lora_rank`` -- that
    ``ModelConfig`` names nowhere, and a model built without them silently gets
    the architecture's own defaults (DeepSeek V3 asks for 256 experts). Same
    offline-only rule as the sizes: a hub id or a local checkpoint directory
    carries its own config, and overriding its architecture by hand is the one
    thing that would make the built model disagree with the weights it loads.

    It also sets ``attn_mask_type``, which is derived rather than configured. Any
    real corpus is packed -- ``datasets/loader_builder.py`` always runs the samples
    through ``ConcatThenSplitPackingConfig`` -- so a row holds several documents
    and attention must not cross a boundary between them. The synthetic random
    corpus is one document per row, where the document mask is a no-op.

    Left unset, ``attn_mask_type`` falls back to ``"causal"`` at the mask site.
    That is correct on a CPU/sdpa run only because the wrapper's packed-sequence
    guard raises first, and silently wrong on the flex path, where a causal-only
    mod attends across document boundaries without complaint. Deriving it here
    means the flag cannot disagree with the corpus the trainer loaded.
    """
    # Only registered architecture names select offline construction. Missing
    # paths and Hub IDs belong to from_pretrained, which owns their validation.
    offline = not os.path.exists(cfg.hf_model) and cfg.hf_model in CONFIG_MAPPING
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
    # ``build_model_config_for`` derives ``attn_mask_type`` and sets it on the
    # TOP config, but the model class is built from this sub-config and every
    # reader of that flag -- the packed guard, the mask builder, and
    # ``apply_cp``'s ulysses check -- reads ``self.model.config``. Left on the
    # top config it is invisible on a composite model: a packed corpus would
    # train with a causal-only mask (attention straight across document
    # boundaries), and the ulysses packed refusal would never fire. Carrying it
    # down here is what keeps "which corpus did the trainer load" answerable
    # from the object that actually runs.
    for name in ("attn_mask_type", "experts_implementation", "compute_dtype"):
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
    import transformers
    from transformers import PreTrainedModel
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    )

    for name in (config.architectures or []) + [
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(getattr(config, "model_type", ""), "")
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
