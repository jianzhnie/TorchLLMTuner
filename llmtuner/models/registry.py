"""Model-owned construction hooks, independent of any modeling backend."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    """Construction hooks for a model and its own configuration type.

    HF integration may require additional config methods, but registration and
    lookup do not require Transformers or a PreTrainedModel base class.
    """

    model_type: str
    config_factory: Callable[..., Any]
    config_type_loader: Callable[[], type]
    model_class_loader: Callable[[], type[nn.Module]]
    task: str = "causal_lm"
    explicit: bool = False

    def build_config(self, **kwargs: Any) -> Any:
        """Construct configuration without loading model or device code."""
        return self.config_factory(**kwargs)

    def resolve_model_class(self) -> type[nn.Module]:
        """Load only the selected implementation."""
        return self.model_class_loader()

    def build_model(self, config: Any) -> nn.Module:
        """Construct this registration without changing configuration ownership."""
        if config.model_type != self.model_type:
            raise ValueError(
                f"Model {self.model_type!r} cannot consume {config.model_type!r} config"
            )
        return self.resolve_model_class()(config)

    @classmethod
    def from_types(
        cls,
        model_type: str,
        *,
        config_type: type,
        model_class_loader: Callable[[], type[nn.Module]],
        task: str = "causal_lm",
        explicit: bool = False,
    ) -> ModelSpec:
        return cls(
            model_type,
            config_type,
            lambda: config_type,
            model_class_loader,
            task=task,
            explicit=explicit,
        )


def _load_class(module_name: str, class_name: str) -> type:
    return getattr(import_module(module_name, __package__), class_name)


def _load_kimi_model_class() -> type[nn.Module]:
    try:
        return _load_class(".kimi.kimi_k3.model", "KimiLinearForCausalLM")
    except ImportError as error:
        raise ImportError(
            f"Kimi Linear model dependencies are unavailable: {error}"
        ) from error


def _build_config(config_type_loader: Callable[[], type], **kwargs: Any) -> Any:
    return config_type_loader()(**kwargs)


_MODELS: dict[str, ModelSpec] = {}


def register_model(architecture: ModelSpec, *, exist_ok: bool = False) -> None:
    """Register construction hooks without importing model implementations."""
    if (
        not isinstance(architecture.model_type, str)
        or not architecture.model_type.strip()
    ):
        raise ValueError("model_type must be a non-empty string")
    if architecture.task not in {"causal_lm", "conditional_generation"}:
        raise ValueError(f"Unsupported model task {architecture.task!r}")
    if not isinstance(architecture.explicit, bool):
        raise TypeError("Model architecture explicit must be a bool")
    for name in ("config_factory", "config_type_loader", "model_class_loader"):
        if not callable(getattr(architecture, name)):
            raise TypeError(f"Model architecture {name} must be callable")
    previous = _MODELS.get(architecture.model_type)
    if previous is not None and not exist_ok:
        raise ValueError(
            f"Model architecture {architecture.model_type!r} is already registered"
        )
    _MODELS[architecture.model_type] = architecture


def get_model_spec(model_type: str, *, task: str = "causal_lm") -> ModelSpec | None:
    """Resolve an ID for a task without falling back to another task."""
    architecture = _MODELS.get(model_type)
    return (
        architecture if architecture is not None and architecture.task == task else None
    )


def registered_models() -> tuple[ModelSpec, ...]:
    """Return registered construction hooks without importing their models."""
    return tuple(_MODELS.values())


def build_registered_model(
    config: Any, *, task: str | None = None, explicit_only: bool = False
) -> nn.Module:
    """Construct from the sole registry, without backend or task fallback.

    None selects the registered task. Explicit task selection must match;
    conditional-generation models cannot silently resolve as text decoders.
    """
    specification = _MODELS.get(config.model_type)
    if (
        specification is None
        or (task is not None and specification.task != task)
        or (explicit_only and not specification.explicit)
    ):
        supported = ", ".join(
            item.model_type
            for item in registered_models()
            if (task is None or item.task == task)
            and (not explicit_only or item.explicit)
        )
        raise ValueError(
            f"No {'explicit ' if explicit_only else ''}model for "
            f"{config.model_type!r}; choose one of {supported}"
        )
    return specification.build_model(config)


for _model_type, _config_module, _config_name, _model_module, _model_name in (
    (
        "kimi_linear",
        ".kimi.kimi_k3.configuration",
        "KimiLinearConfig",
        ".kimi.kimi_k3.model",
        "KimiLinearForCausalLM",
    ),
    (
        "qwen3_moe",
        "transformers.models.qwen3_moe.configuration_qwen3_moe",
        "Qwen3MoeConfig",
        ".qwen.qwen3.model",
        "Qwen3MoeForCausalLM",
    ),
    (
        "deepseek_v4",
        "transformers.models.deepseek_v4.configuration_deepseek_v4",
        "DeepseekV4Config",
        ".deepseek.deepseek_v4.modeling",
        "DeepseekV4ForCausalLM",
    ),
    (
        "qwen3_5_moe_text",
        "transformers.models.qwen3_5_moe.configuration_qwen3_5_moe",
        "Qwen3_5MoeTextConfig",
        ".qwen.qwen3_5_moe.model",
        "Qwen3_5MoeForCausalLM",
    ),
):
    _config_loader = partial(_load_class, _config_module, _config_name)
    register_model(
        ModelSpec(
            _model_type,
            config_factory=partial(_build_config, _config_loader),
            config_type_loader=_config_loader,
            model_class_loader=(
                _load_kimi_model_class
                if _model_type == "kimi_linear"
                else partial(_load_class, _model_module, _model_name)
            ),
            explicit=_model_type == "deepseek_v4",
        )
    )


# Composite construction is explicit-only; causal-LM lookup must not select it.
for _model_type, _config_module, _config_name, _model_module, _model_name in (
    (
        "kimi_k3",
        ".kimi.kimi_k3.configuration",
        "KimiK3Config",
        ".kimi.kimi_k3.multimodal",
        "KimiK3ForConditionalGeneration",
    ),
    (
        "qwen3_5",
        "transformers.models.qwen3_5.configuration_qwen3_5",
        "Qwen3_5Config",
        ".qwen.qwen3_5.multimodal",
        "Qwen3_5ForConditionalGeneration",
    ),
    (
        "qwen3_5_moe",
        "transformers.models.qwen3_5_moe.configuration_qwen3_5_moe",
        "Qwen3_5MoeConfig",
        ".qwen.qwen3_5_moe.multimodal",
        "Qwen3_5MoeForConditionalGeneration",
    ),
):
    _config_loader = partial(_load_class, _config_module, _config_name)
    register_model(
        ModelSpec(
            _model_type,
            config_factory=partial(_build_config, _config_loader),
            config_type_loader=_config_loader,
            model_class_loader=partial(_load_class, _model_module, _model_name),
            task="conditional_generation",
            explicit=True,
        )
    )
