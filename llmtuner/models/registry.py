"""Model-owned construction hooks, independent of any modeling backend."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from importlib import import_module
from typing import Any

from torch import nn

from .state import ModelStateFormat


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
    state_format_loader: Callable[[], type[ModelStateFormat]] | None = None
    flops_estimator: Callable[..., int] | None = None
    flops_note: Callable[..., str | None] | None = None

    def build_config(self, **kwargs: Any) -> Any:
        """Construct configuration without loading model or device code."""
        config = self.config_factory(**kwargs)
        if getattr(config, "model_type", None) != self.model_type:
            raise ValueError(
                f"Model {self.model_type!r} configuration factory returned "
                f"model_type={getattr(config, 'model_type', None)!r}"
            )
        return config

    def resolve_model_class(self) -> type[nn.Module]:
        """Validate the selected implementation without constructing a model."""
        model_class = self.model_class_loader()
        if not isinstance(model_class, type) or not issubclass(model_class, nn.Module):
            raise TypeError(
                f"Model {self.model_type!r} loader must return an nn.Module class"
            )
        return model_class

    def load_config(self, model_name_or_path: str) -> Any:
        """Load external metadata through this architecture's configuration class."""
        config = self.config_type_loader().from_pretrained(model_name_or_path)
        if getattr(config, "model_type", None) != self.model_type:
            raise ValueError(
                f"Loaded configuration for {self.model_type!r} returned "
                f"model_type={getattr(config, 'model_type', None)!r}"
            )
        return config

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
        state_format_loader: Callable[[], type[ModelStateFormat]] | None = None,
        flops_estimator: Callable[..., int] | None = None,
        flops_note: Callable[..., str | None] | None = None,
    ) -> ModelSpec:
        return cls(
            model_type,
            config_type,
            lambda: config_type,
            model_class_loader,
            task=task,
            explicit=explicit,
            state_format_loader=state_format_loader,
            flops_estimator=flops_estimator,
            flops_note=flops_note,
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
    for name in ("state_format_loader", "flops_estimator", "flops_note"):
        hook = getattr(architecture, name)
        if hook is not None and not callable(hook):
            raise TypeError(f"Model architecture {name} must be callable")
    previous = _MODELS.get(architecture.model_type)
    if previous is not None and not exist_ok:
        raise ValueError(
            f"Model architecture {architecture.model_type!r} is already registered"
        )
    _MODELS[architecture.model_type] = architecture


def get_model_spec(
    model_type: str, *, task: str | None = "causal_lm"
) -> ModelSpec | None:
    """Resolve an architecture, optionally requiring an exact task.

    Configuration discovery uses task=None; execution keeps task filtering.
    """
    architecture = _MODELS.get(model_type)
    return (
        architecture
        if architecture is not None and (task is None or architecture.task == task)
        else None
    )


def resolve_state_format(
    config: Any, *, expert_parallel: bool, sharded_expert_parallel: bool
) -> ModelStateFormat:
    """Resolve external state rules from the sole architecture registration."""
    specification = get_model_spec(getattr(config, "model_type", ""), task=None)
    if specification is None or specification.state_format_loader is None:
        return ModelStateFormat()
    format_class = specification.state_format_loader()
    if not isinstance(format_class, type) or not issubclass(
        format_class, ModelStateFormat
    ):
        raise TypeError(
            "Model state format loader must return a ModelStateFormat class"
        )
    return format_class(
        config,
        expert_parallel=expert_parallel,
        sharded_expert_parallel=sharded_expert_parallel,
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
    specification = get_model_spec(config.model_type, task=task)
    if specification is None or (explicit_only and not specification.explicit):
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


def _estimate_flops(module: str, config: Any, *, seq_len: int) -> int:
    return import_module(module, __package__).flops_per_token(config, seq_len=seq_len)


def _flops_note(module: str, config: Any, *, seq_len: int) -> str | None:
    return import_module(module, __package__).flops_estimation_note(
        config, seq_len=seq_len
    )


_FLOPS_MODULES = {
    "kimi_linear": ".kimi.kimi_k3.flops",
    "qwen3_5_moe_text": ".qwen.qwen3_5_moe.flops",
}


for (
    _model_type,
    _config_module,
    _config_name,
    _model_module,
    _model_name,
    _state_module,
    _state_name,
    _task,
) in (
    (
        "kimi_linear",
        ".kimi.kimi_k3.configuration",
        "KimiLinearConfig",
        ".kimi.kimi_k3.model",
        "KimiLinearForCausalLM",
        ".kimi.kimi_k3.checkpoint",
        "KimiStateFormat",
        "causal_lm",
    ),
    (
        "qwen3_moe",
        "transformers.models.qwen3_moe.configuration_qwen3_moe",
        "Qwen3MoeConfig",
        ".qwen.qwen3.model",
        "Qwen3MoeForCausalLM",
        ".qwen.qwen3.state",
        "Qwen3StateFormat",
        "causal_lm",
    ),
    (
        "deepseek_v4",
        "transformers.models.deepseek_v4.configuration_deepseek_v4",
        "DeepseekV4Config",
        ".deepseek.deepseek_v4.modeling",
        "DeepseekV4ForCausalLM",
        None,
        None,
        "causal_lm",
    ),
    (
        "qwen3_5_moe_text",
        "transformers.models.qwen3_5_moe.configuration_qwen3_5_moe",
        "Qwen3_5MoeTextConfig",
        ".qwen.qwen3_5_moe.model",
        "Qwen3_5MoeForCausalLM",
        None,
        None,
        "causal_lm",
    ),
    (
        "kimi_k3",
        ".kimi.kimi_k3.configuration",
        "KimiK3Config",
        ".kimi.kimi_k3.multimodal",
        "KimiK3ForConditionalGeneration",
        None,
        None,
        "conditional_generation",
    ),
    (
        "qwen3_5",
        "transformers.models.qwen3_5.configuration_qwen3_5",
        "Qwen3_5Config",
        ".qwen.qwen3_5.multimodal",
        "Qwen3_5ForConditionalGeneration",
        None,
        None,
        "conditional_generation",
    ),
    (
        "qwen3_5_moe",
        "transformers.models.qwen3_5_moe.configuration_qwen3_5_moe",
        "Qwen3_5MoeConfig",
        ".qwen.qwen3_5_moe.multimodal",
        "Qwen3_5MoeForConditionalGeneration",
        None,
        None,
        "conditional_generation",
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
            state_format_loader=(
                partial(_load_class, _state_module, _state_name)
                if _state_module is not None
                else None
            ),
            flops_estimator=(
                partial(_estimate_flops, _FLOPS_MODULES[_model_type])
                if _model_type in _FLOPS_MODULES
                else None
            ),
            flops_note=(
                partial(_flops_note, _FLOPS_MODULES[_model_type])
                if _model_type == "qwen3_5_moe_text"
                else None
            ),
            task=_task,
            explicit=True,
        )
    )
