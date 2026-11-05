"""Shared configuration helpers without model or runtime dependencies."""

from __future__ import annotations

from llmtuner.errors import ConfigError


def require_integer_at_least(
    obj: object, *names: str, group: str, minimum: int = 1
) -> None:
    """Require integer fields at or above ``minimum``, rejecting booleans."""
    for name in names:
        value = getattr(obj, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ConfigError(f"{group}.{name} must be >= {minimum}, got {value}")


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
