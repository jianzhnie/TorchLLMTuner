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
class ModelArchitecture:
    """Construction hooks for a model and its own configuration type.

    HF integration may require additional config methods, but registration and
    lookup do not require Transformers or a PreTrainedModel base class.
    """

    model_type: str
    config_factory: Callable[..., Any]
    config_type_loader: Callable[[], type]
    model_class_loader: Callable[[], type[nn.Module]]

    @classmethod
    def from_types(
        cls,
        model_type: str,
        *,
        config_type: type,
        model_class_loader: Callable[[], type[nn.Module]],
    ) -> ModelArchitecture:
        return cls(model_type, config_type, lambda: config_type, model_class_loader)


def _load_kimi_config_class() -> type:
    module = import_module(".kimi.kimi_k3.configuration", __package__)
    return module.KimiLinearConfig


def _build_kimi_config(**kwargs: Any) -> Any:
    return _load_kimi_config_class()(**kwargs)


def _load_kimi_model_class() -> type[nn.Module]:
    try:
        module = import_module(".kimi.kimi_k3.model", __package__)
    except ImportError as error:
        raise ImportError(
            f"Kimi Linear model dependencies are unavailable: {error}"
        ) from error
    return module.KimiLinearForCausalLM


def _load_qwen3_moe_config_class() -> type:
    module = import_module("transformers.models.qwen3_moe.configuration_qwen3_moe")
    return module.Qwen3MoeConfig


def _build_qwen3_moe_config(**kwargs: Any) -> Any:
    return _load_qwen3_moe_config_class()(**kwargs)


def _load_qwen3_moe_model_class() -> type[nn.Module]:
    module = import_module(".qwen.qwen3.model", __package__)
    return module.Qwen3MoeForCausalLM


def _load_class(module_name: str, class_name: str) -> type:
    return getattr(import_module(module_name, __package__), class_name)


def _build_config(config_type_loader: Callable[[], type], **kwargs: Any) -> Any:
    return config_type_loader()(**kwargs)


_ARCHITECTURES: dict[str, ModelArchitecture] = {}


def register_model_architecture(
    architecture: ModelArchitecture, *, exist_ok: bool = False
) -> None:
    """Register construction hooks without importing model implementations."""
    if (
        not isinstance(architecture.model_type, str)
        or not architecture.model_type.strip()
    ):
        raise ValueError("model_type must be a non-empty string")
    for name in ("config_factory", "config_type_loader", "model_class_loader"):
        if not callable(getattr(architecture, name)):
            raise TypeError(f"Model architecture {name} must be callable")
    previous = _ARCHITECTURES.get(architecture.model_type)
    if previous is not None and not exist_ok:
        raise ValueError(
            f"Model architecture {architecture.model_type!r} is already registered"
        )
    _ARCHITECTURES[architecture.model_type] = architecture


def get_model_architecture(model_type: str) -> ModelArchitecture | None:
    return _ARCHITECTURES.get(model_type)


def registered_model_architectures() -> tuple[ModelArchitecture, ...]:
    """Return registered construction hooks without importing their models."""
    return tuple(_ARCHITECTURES.values())


register_model_architecture(
    ModelArchitecture(
        "kimi_linear",
        config_factory=_build_kimi_config,
        config_type_loader=_load_kimi_config_class,
        model_class_loader=_load_kimi_model_class,
    )
)

register_model_architecture(
    ModelArchitecture(
        "qwen3_moe",
        config_factory=_build_qwen3_moe_config,
        config_type_loader=_load_qwen3_moe_config_class,
        model_class_loader=_load_qwen3_moe_model_class,
    )
)

for _model_type, _config_module, _config_name, _model_module, _model_name in (
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
    register_model_architecture(
        ModelArchitecture(
            _model_type,
            config_factory=partial(_build_config, _config_loader),
            config_type_loader=_config_loader,
            model_class_loader=partial(_load_class, _model_module, _model_name),
        )
    )
