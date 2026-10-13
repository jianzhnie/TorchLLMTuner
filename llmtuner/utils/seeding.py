# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import os
import random

import numpy as np
import torch

from ..accelerator.device import device_module, device_type
from .logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int, set_deterministic: bool = False) -> None:
    """Seed the active device and host generators through the shared path."""
    seed_everything(seed, deterministic=set_deterministic)


def set_deterministic_algorithms():
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
        torch.autograd.set_detect_anomaly(True, check_nan=False)
