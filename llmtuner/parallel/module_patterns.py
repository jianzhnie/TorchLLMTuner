"""Extended module path matching shared by parallelization plans.

Besides ``*`` and ``?`` wildcards, ``{*}`` matches a numeric path segment and
``{start-end}`` matches an inclusive numeric range.  Keeping this in the
parallel package lets TP and FSDP use exactly the same selection semantics.
"""

from __future__ import annotations

import re
from functools import lru_cache

_NUMERIC_RANGE = re.compile(r"\{([0-9]+)-([0-9]+)\}")


@lru_cache(maxsize=256)
def compile_module_pattern(pattern: str) -> re.Pattern[str]:
    """Compile an FSDPTurbo compatible module path pattern."""
    parts: list[str] = []
    index = 0
    while index < len(pattern):
        if pattern.startswith("{*}", index):
            parts.append(r"[0-9]+")
            index += 3
            continue
        match = _NUMERIC_RANGE.match(pattern, index)
        if match:
            low, high = map(int, match.groups())
            if low > high:
                raise ValueError(f"invalid module pattern range {{{low}-{high}}}")
            parts.append("(?:" + "|".join(map(str, range(low, high + 1))) + ")")
            index = match.end()
            continue
        char = pattern[index]
        if char == "*":
            parts.append(".*")
        elif char == "?":
            parts.append(".")
        elif char in "{}":
            raise ValueError(f"invalid module pattern {pattern!r}")
        else:
            parts.append(re.escape(char))
        index += 1
    return re.compile("".join(parts))


def module_name_matches(module_name: str, *, pattern: str) -> bool:
    """Return whether ``module_name`` is selected by ``pattern``."""
    return compile_module_pattern(pattern).fullmatch(module_name) is not None


def resolve_module_patterns(
    module_names: list[str], *, patterns: list[str], plan_name: str
) -> set[str]:
    """Resolve patterns and reject typos that would otherwise be silent."""
    selected: set[str] = set()
    for pattern in patterns:
        compiled = compile_module_pattern(pattern)
        matches = {name for name in module_names if compiled.fullmatch(name)}
        if not matches:
            raise ValueError(f"{plan_name} pattern {pattern!r} matched no module")
        selected.update(matches)
    return selected


__all__ = ["compile_module_pattern", "module_name_matches", "resolve_module_patterns"]
