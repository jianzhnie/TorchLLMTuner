"""Tensor parallelism: the declaration/realizer machinery (``tp``) and the
``apply_tp`` entry point (``apply``), re-exported at the package seam like
every other parallelism family."""

from .tensor_parallel import *  # noqa: F401,F403
from .tensor_parallel import __all__ as __all__
