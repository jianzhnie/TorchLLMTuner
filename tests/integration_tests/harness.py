"""Device and backend selection for the equivalence harnesses.

Every script in this directory is a two-arm comparison: one arm is the
unsharded single-rank reference, the other runs the real ``apply_*`` path, and
the two are compared numerically. That comparison is device-agnostic as long as
**all** tensors land on one device and the process group matches it -- which is
what this module provides:

* :func:`device` -- this rank's device (the accelerator when the host has one,
  CPU otherwise), matching ``llmtuner.accelerator.device.device_type``;
* :func:`init_distributed` -- start the process group for that device type and
  bind the rank to its device;
* :func:`moved` -- place a model or tensor on it.

``llmtuner`` itself picks the device this way (NPU first), so a harness that
hard-codes ``gloo`` while ``ParallelTopology`` builds an NPU mesh fails with
``No backend type associated with device type npu``. Import the helpers instead
of pinning a backend and the same script runs on CPU-only, NPU and CUDA hosts.

Scope of ``LLMTUNER_TEST_DEVICE=cpu``: it redirects **this module's** placement
and process group. It cannot redirect ``llmtuner``'s own: ``device_type`` is
resolved once at import from the hardware, and the paths that place tensors
themselves (``Trainer``, ``apply_fsdp``/``parallelize_hf_transformers``, the DCP
checkpointer) keep following it. So the switch gives the original CPU/gloo run
for the scripts whose tensors this module places -- the pure-redistribution
harnesses -- and a *starting* CPU for the rest, which then fail loudly on the
mixed devices rather than silently training on the accelerator. Running those
on CPU needs a CPU-only host: hiding the devices here is not an option, because
``torch_npu`` patches ``torch.optim`` unconditionally
(``torch_npu/__init__.py:178`` -> ``utils/_optim.py:patch_supported_devices``)
and the patch then queries ``torch_npu.npu.current_device()``, which fails with
``aclInit ... Invalid device ID`` once no device is visible.
"""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist

from llmtuner.accelerator.device import device_type, get_distributed_backend
from llmtuner.datasets.contracts import Batch

__all__ = ["backend", "device", "init_distributed", "moved", "moved_batch"]


def device() -> torch.device:
    """This rank's device, honouring ``LLMTUNER_TEST_DEVICE``."""
    if os.environ.get("LLMTUNER_TEST_DEVICE", "").lower() == "cpu":
        return torch.device("cpu")
    return torch.device(device_type, int(os.environ.get("LOCAL_RANK", 0)))


def backend() -> str:
    """The process-group backend matching :func:`device`."""
    return "gloo" if device().type == "cpu" else get_distributed_backend()


def init_distributed(*, timeout: timedelta | None = None) -> torch.device:
    """Start the matching process group, bind the device, return the device."""
    dev = device()
    if dev.type != "cpu":
        getattr(torch, dev.type).set_device(dev)
    dist.init_process_group(backend(), timeout=timeout)
    return dev


def moved(obj):
    """Place a tensor, ``nn.Module`` or ``nn.Parameter`` on this rank's device."""
    return obj.to(device())


def moved_batch(batch: Batch) -> Batch:
    """A ``Batch`` whose tensors live on this rank's device.

    The dataloader hands out CPU batches by design; the single-rank reference
    arm of a harness feeds one straight into a model that sits on the
    accelerator, so it needs the device copy.
    """
    return Batch(input_ids=moved(batch.input_ids), labels=moved(batch.labels))
