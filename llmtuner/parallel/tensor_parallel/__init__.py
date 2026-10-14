"""Tensor parallelism public API.

``tensor_parallel.py`` is the entry and plan engine; ``linear_ops.py`` contains
the fused collective/GEMM primitives it realizes plans into.
"""

from .tensor_parallel import *  # noqa: F401,F403
from .tensor_parallel import __all__ as __all__
