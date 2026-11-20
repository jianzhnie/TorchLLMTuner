"""Shared configuration helpers without model or runtime dependencies."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, fields
from typing import Any, TypeVar, get_type_hints

from llmtuner.errors import ConfigError

_ConfigT = TypeVar("_ConfigT")


def rebuild_config_group(
    config_type: type[_ConfigT], source: object, **updates: Any
) -> _ConfigT:
    """Rebuild a CLI group as its target type, sharing nested values unchanged."""
    values = {
        item.name: getattr(source, item.name)
        for item in fields(config_type)
        if item.init
    }
    return config_type(**(values | updates))


def require_integer_value(value: int, *, name: str, minimum: int = 1) -> None:
    """Validate one integer bound without requiring a configuration object."""
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")


def require_integer_at_least(
    obj: object, *names: str, group: str, minimum: int = 1
) -> None:
    """Require integer fields at or above ``minimum``, rejecting booleans."""
    for name in names:
        require_integer_value(
            getattr(obj, name), name=f"{group}.{name}", minimum=minimum
        )


def require_bool(obj: object, *names: str, group: str) -> None:
    """Require actual booleans rather than truthy configuration values."""
    for name in names:
        value = getattr(obj, name)
        if not isinstance(value, bool):
            raise ConfigError(f"{group}.{name} must be bool, got {value!r}")


def require_string_patterns(value: list[str], *, field_name: str) -> None:
    """Require a list of nonblank patterns; an empty list selects nothing."""
    if not isinstance(value, list) or any(
        not isinstance(pattern, str) or not pattern.strip() for pattern in value
    ):
        raise ConfigError(f"{field_name} must be a list of nonempty strings")


def hide_cli_fields(cls: type, hidden: frozenset[str] | None) -> type:
    """Build a dataclass parser view without mutating its source fields."""
    if not hidden:
        return cls
    options = {option.name: option for option in fields(cls)}
    unknown = hidden - options.keys()
    if unknown:
        raise RuntimeError(
            f"{cls.__name__} has no fields {sorted(unknown)}: "
            "config.cli_config.PROGRAMMATIC_ONLY is stale"
        )
    hints = get_type_hints(cls)
    namespace: dict[str, Any] = {
        "__annotations__": {name: hints[name] for name in hidden}
    }
    for name in hidden:
        option = copy(options[name])
        option.init = False
        namespace[name] = option
    return dataclass(type(cls.__name__, (cls,), namespace))
