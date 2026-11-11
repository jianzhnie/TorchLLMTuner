"""Semantic kernel entry points and CPU/NPU backend registration."""

from . import cpu as _cpu  # noqa: F401
from . import npu as _npu  # noqa: F401
from .registry import dispatch_kernel, get_kernel, register_kernel

__all__ = ["dispatch_kernel", "get_kernel", "register_kernel"]
