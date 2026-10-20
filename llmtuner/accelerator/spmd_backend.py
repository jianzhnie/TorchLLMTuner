"""Resolve the ``spmd_types`` SPMD backend, with a torch-2.9 fallback.

llmtuner builds its SPMD layer on the PyPI ``spmd_types`` package, but that
package imports ``torch.distributed._local_tensor`` at module scope -- a
private torch API that first ships with torch 2.12. Every released
``spmd_types`` (0.1.0 through 0.3.0) requires it, so on the torch 2.9
Ascend/``torch_npu`` stack the package cannot be imported at all. The usual
"optional dependency" answer (raise an ``ImportError`` carrying an install
hint) does not apply: no version exists that could be installed here.

This module therefore resolves the seven symbols llmtuner actually uses and
substitutes faithful no-ops when the real backend is missing:

* ``assert_type`` / ``assert_local_type_like`` -- *static* SPMD verification.
  The fallback disables the checks only; no tensor layout or collective
  changes, so numerics are untouched either way.
* ``S`` / ``R`` / ``PartitionSpec`` -- the placement specs those checks take.
* ``set_current_mesh`` -- enters spmd_types' local-tensor mode, which torch
  2.9 does not have; a null context is the exact equivalent.
* ``register_local_autograd_function`` -- makes a custom
  ``autograd.Function`` local-tensor aware; with no local-tensor mode the
  class is correctly returned unchanged.

The fallback is loud rather than silent: it warns once at import, naming the
disabled capability, so a layout mistake spmd_types would have caught is
attributed to the environment rather than hunted as a logic bug. Symbols the
package carries but llmtuner no longer uses (``redistribute``, ``local_map``,
``no_typecheck``) are deliberately not stubbed.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

import torch

from ..utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["SPMD_TYPES_AVAILABLE", "spmd"]


try:
    import spmd_types as spmd

    SPMD_TYPES_AVAILABLE = True
except ImportError:
    # Covers both "package absent" and "package present but its
    # ``torch.distributed._local_tensor`` import fails" -- the latter is the
    # torch 2.9/2.10 case, and raises ModuleNotFoundError, an ImportError.
    SPMD_TYPES_AVAILABLE = False

    class _FallbackSpec:
        """Placement spec stand-in; only ever passed to the no-op checks."""

        def __init__(self, *dims: Any) -> None:
            self.dims = dims

        def __repr__(self) -> str:
            return f"{type(self).__name__}({', '.join(map(repr, self.dims))})"

    class _FallbackReplicate(_FallbackSpec):
        def __init__(self) -> None:
            super().__init__()

    def _shard(*dims: Any) -> _FallbackSpec:
        return _FallbackSpec(*dims)

    def _assert_type(*args: Any, **kwargs: Any) -> None:
        return None

    def _assert_local_type_like(*args: Any, **kwargs: Any) -> None:
        return None

    @contextlib.contextmanager
    def _set_current_mesh(mesh: Any):
        yield

    def _register_local_autograd_function(fn: Any) -> Any:
        return fn

    spmd = SimpleNamespace(
        PartitionSpec=_FallbackSpec,
        S=_shard,
        R=_FallbackReplicate(),
        assert_type=_assert_type,
        assert_local_type_like=_assert_local_type_like,
        set_current_mesh=_set_current_mesh,
        register_local_autograd_function=_register_local_autograd_function,
    )

    logger.warning(
        "spmd_types is unavailable on torch %s (its module-scope import of "
        "torch.distributed._local_tensor needs torch>=2.12). SPMD static "
        "type-checking is DISABLED for this run; tensor layouts, collectives "
        "and numerics are unaffected.",
        torch.__version__,
    )
