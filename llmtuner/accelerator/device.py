"""Device discovery and small backend-neutral runtime helpers.

NPU is preferred because it is TorchLLMTuner's primary accelerator, followed by
CUDA and other torch accelerators that provide distributed collectives. MPS is
intentionally excluded: it has no distributed backend and cannot host a
``DeviceMesh`` even when torch reports it as available.

``is_device_type_available`` and ``should_use_pin_memory`` derive from
OpenMMLab's ``mmengine.device`` conventions, merged here so nothing needs the
mmengine dependency. The rest of that file's surface was not carried over: its
import-time ``DEVICE`` constant and ``get_device()`` duplicate this module's
``device_type`` / ``get_device_type()``, its ``torch.npu.set_compile_mode``
call mutates global torch state at import time, and its per-vendor predicates
(``is_cuda_available`` / ``is_npu_available`` / ...), NPU full-precision probe
and peak-memory queries had no caller -- the `mmengine`-style surface is not
kept for its own sake, the same way the other vendored-but-unused files were
dropped.
"""

from __future__ import annotations

import importlib
import os

import torch

try:
    importlib.import_module("torch_npu")
except ImportError:
    pass

# Register the vendor extensions so ``torch.mlu`` / ``torch.musa`` exist for
# the availability probes below. Each import is a no-op when the vendor's
# torch build is not installed.
for _ext in ("torch_mlu", "torch_musa"):
    try:
        importlib.import_module(_ext)
    except ImportError:
        pass
del _ext

ACCELERATOR_TYPES = frozenset(("npu", "cuda", "xpu", "mlu", "musa"))
DEVICE_PRIORITY = ("npu", "cuda", "musa", "mlu", "xpu")
_BACKENDS = {
    "npu": "hccl",
    "cuda": "nccl",
    "xpu": "xccl",
    "mlu": "cncl",
    "musa": "mccl",
    "cpu": "gloo",
}


def is_device_type_available(kind: str) -> bool:
    """Return whether ``kind`` has an accessible torch device."""
    if kind == "cpu":
        return True
    module = getattr(torch, kind, None)
    if module is None:
        return False
    try:
        if not module.is_available():
            return False
        count = getattr(module, "device_count", None)
        return count is None or count() > 0
    except Exception:
        return False


def get_device_type() -> str:
    """Select the training device by priority: NPU first, CPU last.

    The result is frozen into the module-level ``device_type`` at import
    time, so device discovery happens exactly once per process.
    """
    return next(
        (kind for kind in DEVICE_PRIORITY if is_device_type_available(kind)), "cpu"
    )


def get_env_dist_info() -> tuple[int, int, int]:
    """Return ``(rank, world size, local rank)`` from torchrun's environment.

    Named ``get_env_dist_info`` to stay distinct from
    ``dist_utils.get_dist_info(group)``, which queries the live process
    group and returns only ``(rank, world size)``.
    """
    return (
        int(os.environ.get("RANK", 0)),
        int(os.environ.get("WORLD_SIZE", 1)),
        int(os.environ.get("LOCAL_RANK", 0)),
    )


def get_current_device(*, use_cpu: bool = False) -> torch.device:
    """Return this process's device using ``LOCAL_RANK``."""
    if use_cpu or device_type == "cpu":
        return torch.device("cpu")
    return torch.device(device_type, get_env_dist_info()[2])


def get_distributed_backend() -> str:
    """Return the backend derived from the active device type."""
    return _BACKENDS[device_type]


def set_device(device: torch.device) -> None:
    """Set the accelerator device, failing rather than misplacing ranks."""
    if device.type == "cpu":
        return
    module = getattr(torch, device.type, None)
    setter = getattr(module, "set_device", None)
    if setter is None:
        raise RuntimeError(f"Device type {device.type!r} has no set_device API")
    try:
        setter(device)
    except Exception as exc:
        raise RuntimeError(f"Failed to set device {device}: {exc}") from exc


def should_use_pin_memory(device: torch.device | None = None) -> bool:
    """Return whether asynchronous pinned-memory copies are useful."""
    device = get_current_device() if device is None else device
    return device.type in ACCELERATOR_TYPES


device_type = get_device_type()
device_module = torch if device_type == "cpu" else getattr(torch, device_type)
