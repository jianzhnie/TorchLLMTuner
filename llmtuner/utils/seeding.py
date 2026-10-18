# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from __future__ import annotations

import os
import random
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..accelerator.device import device_module, device_type
from .logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


def initialize_dtensor_rng(mesh: DeviceMesh | None) -> None:
    """Initialize DTensor random-op tracking without a global seed broadcast.

    PyTorch 2.14 uses the device's default RNG directly. Disable first-use
    synchronization so PP stage seeds and checkpoint-restored streams survive.
    Constructing the tracker does not reseed or consume random numbers.
    """
    if mesh is None or mesh.device_type == "cpu":
        return
    from torch.distributed.tensor import _random

    if _random.is_rng_supported_mesh(mesh) and _random._rng_tracker is None:
        _random._rng_tracker = _random.OffsetBasedRNGTracker(mesh, run_state_sync=False)


@contextmanager
def preserve_rng_state(device: torch.device) -> Iterator[None]:
    """Isolate Python, NumPy, CPU and the selected device's default RNGs.

    This does not snapshot private generators owned by loaders. PyTorch 2.14
    regular DTensor random ops use the device default stream saved here.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(
            devices=[] if device.type == "cpu" else [device],
            device_type=device.type,
        ):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def set_deterministic_algorithms() -> None:
    """Enable deterministic algorithms and deterministic accelerator kernels."""
    os.environ["HCCL_DETERMINISTIC"] = "True"
    os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"
    # torchtitan's ``set_determinism``, minus the parts that only exist
    # for its own stack (the DTensor mesh-aware RNG tracker, the
    # flex-attention kernels) and plus this one spelled out:
    # ``use_deterministic_algorithms(True)`` turns on
    # ``fill_uninitialized_memory``, whose fill kernel races with side
    # streams and is what made HF's RoPE init observe NaN, so upstream
    # turns it back off for the same reason.
    #
    # ``warn_only=False`` is llmtuner's fixed, stricter setting;
    # upstream reads it from ``debug.deterministic_warn_only``.
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.utils.deterministic.fill_uninitialized_memory = False
    # Deterministic cuBLAS needs a workspace split, not the default one.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_everything(
    seed: int, *, deterministic: bool, detect_anomaly: bool = False
) -> None:
    """Seed every generator this run draws from.

    torchtitan's ``set_determinism`` semantics; ``Trainer.seed_everything``
    delegates here.
    """
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or not -(2**63) <= seed < 2**64
    ):
        raise ValueError(
            f"seed must be an integer in [-2**63, 2**64 - 1], got {seed!r}"
        )
    for name, value in (
        ("deterministic", deterministic),
        ("detect_anomaly", detect_anomaly),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be bool, got {value!r}")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    # Hash randomization is not observable in this process (PYTHONHASHSEED is
    # read at interpreter start), but dataloader workers are spawned later and
    # do read it, so upstream sets it here for them. Same spelling.
    os.environ["PYTHONHASHSEED"] = str(seed % 2**32)
    if device_type != "cpu":
        device_module.manual_seed_all(seed)
    if deterministic:
        set_deterministic_algorithms()
    if detect_anomaly:
        logger.warning(
            "Anomaly detection enabled. This incurs significant overhead and "
            "is for debugging only."
        )
        # ``check_nan=False``: the NaN/Inf gradient check calls
        # ``aten._is_any_true``, which has no DTensor sharding strategy and
        # would crash on sharded parameters. Stack-trace recording -- the
        # useful half -- stays on. Same setting as upstream's.
    torch.autograd.set_detect_anomaly(detect_anomaly, check_nan=False)


def derive_distinct_seed(seed: int, distinct_coords: Iterable[tuple[int, int]]) -> int:
    """Offset ``seed`` by this rank's coordinates along distinct mesh dims.

    Each ``(local_rank, dim_size)`` pair contributes ``local_rank`` times the
    product of all previous dimensions' sizes. The first dimension varies
    fastest, so every valid coordinate tuple maps to a unique offset. The
    result is reduced mod 2**64 to stay inside ``torch.manual_seed``'s range.

    Empty coordinates (or all-zero local ranks) add no offset. The seed is
    still normalized modulo 2**64, including negative seeds; PyTorch maps
    those to the same generator state as their normalized equivalents.
    """
    offset = 0
    cumulative_size = 1
    for local_rank, dim_size in distinct_coords:
        if (
            not isinstance(local_rank, int)
            or isinstance(local_rank, bool)
            or not isinstance(dim_size, int)
            or isinstance(dim_size, bool)
            or dim_size < 1
            or not 0 <= local_rank < dim_size
        ):
            raise ValueError(
                "each seed coordinate must contain an integer rank in "
                f"[0, dim_size), got ({local_rank!r}, {dim_size!r})"
            )
        offset += local_rank * cumulative_size
        cumulative_size *= dim_size
    return (seed + offset) % 2**64
