"""Registration seam for model architectures not shipped by Transformers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .kimi.kimi_k3.kimi_linear_config import KimiLinearConfig

KIMI_MODEL_IMPORT_ERROR = None
try:
    from .kimi.kimi_k3.model import KimiLinearForCausalLM
except ImportError as error:
    KIMI_MODEL_IMPORT_ERROR = error

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel


@dataclass(frozen=True)
class ModelArchitecture:
    """Lightweight construction hooks for one custom HF architecture."""

    model_type: str
    config_factory: Callable[..., PretrainedConfig]
    config_type_loader: Callable[[], type[PretrainedConfig]]
    model_class_loader: Callable[[], type[PreTrainedModel]]

    @classmethod
    def from_types(
        cls,
        model_type: str,
        *,
        config_type: type[PretrainedConfig],
        model_class_loader: Callable[[], type[PreTrainedModel]],
    ) -> ModelArchitecture:
        return cls(model_type, config_type, lambda: config_type, model_class_loader)


def _load_kimi_model_class() -> type[PreTrainedModel]:
    if KIMI_MODEL_IMPORT_ERROR is not None:
        raise ImportError(
            f"Kimi Linear model dependencies are unavailable: {KIMI_MODEL_IMPORT_ERROR}"
        ) from KIMI_MODEL_IMPORT_ERROR
    return KimiLinearForCausalLM


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
    """Return the custom configs needed when loading saved HF configurations."""
    return tuple(_ARCHITECTURES.values())


register_model_architecture(
    ModelArchitecture.from_types(
        "kimi_linear",
        config_type=KimiLinearConfig,
        model_class_loader=_load_kimi_model_class,
    )
)
