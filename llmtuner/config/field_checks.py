"""The shared field-floor check behind the config groups' ``__post_init__``."""

from __future__ import annotations

from llmtuner.errors import ConfigError


def require_at_least(obj: object, *names: str, group: str, minimum: int = 1) -> None:
    """Raise ``ConfigError`` for any field of ``obj`` below ``minimum``.

    The message is the package-wide form ``<group>.<field> must be >= N, got
    <value>``, with ``group`` the config's short name (``model`` / ``parallel``
    / ...) matching the root attribute it is reached through.
    """
    for name in names:
        value = getattr(obj, name)
        if value < minimum:
            raise ConfigError(f"{group}.{name} must be >= {minimum}, got {value}")
