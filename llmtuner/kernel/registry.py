"""Device-dispatched kernel registration with explicit fallback semantics."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypeVar

import torch

from ..accelerator.device import device_type as default_device_type
from .logging import get_logger

logger = get_logger(__name__)

Kernel = Callable[..., Any]
KernelFunction = TypeVar("KernelFunction", bound=Kernel)

_KERNEL_REGISTRY: dict[str, dict[str, Kernel]] = {}
_warned_fallbacks: set[tuple[str, str]] = set()


def register_kernel(
    name: str, device_type: str
) -> Callable[[KernelFunction], KernelFunction]:
    """Register one backend implementation without silently replacing another."""

    if (
        not isinstance(name, str)
        or not name.strip()
        or not isinstance(device_type, str)
        or not device_type.strip()
    ):
        raise ValueError("Kernel name and device type must not be empty")

    def decorator(function: KernelFunction) -> KernelFunction:
        if not callable(function):
            raise TypeError("Kernel implementation must be callable")
        backends = _KERNEL_REGISTRY.setdefault(name, {})
        previous = backends.get(device_type)
        if previous is not None and previous is not function:
            raise ValueError(f"Kernel {name!r} already registered for {device_type!r}")
        backends[device_type] = function
        return function

    return decorator


def get_kernel(
    name: str, device_type: str | None = None, *, fallback_to_cpu: bool = False
) -> Kernel:
    """Resolve an implementation, optionally selecting the CPU/reference backend.

    Selection alone never transfers input tensors. Without a device argument,
    reuse the training environment selected by accelerator.device.
    """
    if device_type is None:
        device_type = default_device_type
    backends = _KERNEL_REGISTRY.get(name)
    if backends is None:
        raise KeyError(f"Kernel {name!r} is not registered")
    if device_type in backends:
        return backends[device_type]
    if fallback_to_cpu and "cpu" in backends:
        key = name, device_type
        if key not in _warned_fallbacks:
            logger.warning(
                "Kernel %s is unavailable on %s; using the torch reference "
                "implementation without moving tensors",
                name,
                device_type,
            )
            _warned_fallbacks.add(key)
        return backends["cpu"]
    raise KeyError(f"Kernel {name!r} is not available for device {device_type!r}")


def dispatch_kernel(
    name: str,
    *args: Any,
    device_type: str | None = None,
    fallback_to_cpu: bool = False,
    **kwargs: Any,
) -> Any:
    """Select the input tensor's device, then invoke the registered kernel.

    Explicit device selection overrides inference. The registry never moves
    tensors: an explicit CPU fallback executes reference torch operations on
    the supplied tensors. Hardware selection is used only for tensor-free calls.
    """
    if device_type is None:
        tensor = next(
            (
                value
                for value in (*args, *kwargs.values())
                if isinstance(value, torch.Tensor)
            ),
            None,
        )
        if tensor is not None:
            device_type = tensor.device.type
    return get_kernel(name, device_type, fallback_to_cpu=fallback_to_cpu)(
        *args, **kwargs
    )
