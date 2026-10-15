"""Resolve lazy package exports without importing optional training dependencies.

Each package owns its name-to-submodule mapping. PEP 562 __getattr__ delegates
here to import the defining submodule only when the attribute is requested.
This keeps lightweight package imports independent of the model/parallel stack.
Package indexes sort their mappings directly for __all__ and __dir__.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Any

__all__ = ["resolve_export"]


def resolve_export(module_name: str, sources: Mapping[str, str], name: str) -> Any:
    """Import ``name`` from the submodule ``sources`` maps it to.

    Args:
        module_name: the index's own ``__name__`` (the import is absolute
            against it, so callers pass the package they are).
        sources: name -> submodule, the index's table.
        name: the attribute being looked up.

    Raises:
        AttributeError: when the index does not export ``name``. A miss has to
            read as a miss -- a quiet ``None`` would turn a typo into a silently
            absent feature -- and the message names the module that was asked.
    """
    source = sources.get(name)
    if source is None:
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
    return getattr(import_module(f"{module_name}.{source}"), name)
