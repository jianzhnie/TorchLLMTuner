"""Mesh-aware seed derivation.

Pipeline stages must not share an RNG stream: stages hold *different* layers,
so seeding every stage identically would correlate their weight initialization
and dropout patterns. Ranks inside one SPMD group, by contrast, must draw the
same numbers so sharded parameters stay consistent. The derivation therefore
offsets the base seed only along mesh dimensions declared distinct (``pp``),
keeping all other coordinates' ranks on the base seed.
"""

import os
from collections.abc import Iterable

import torch

from ..accelerator.device import device_module, device_type
from ..utils.logger_utils import get_logger

logger = get_logger(__name__)


def derive_distinct_seed(seed: int, distinct_coords: Iterable[tuple[int, int]]) -> int:
    """Offset ``seed`` by this rank's coordinates along distinct mesh dims.

    Each ``(local_rank, dim_size)`` pair contributes ``local_rank`` times the
    product of all previous dimensions' sizes -- row-major indexing over the
    distinct sub-mesh, so every coordinate tuple maps to a unique offset. The
    result is reduced mod 2**64 to stay inside ``torch.manual_seed``'s range.

    An empty ``distinct_coords`` (or all-zero local ranks, e.g. a size-1 dim)
    returns ``seed`` unchanged, so single-stage runs are bit-identical to
    seeding without derivation.
    """
    offset = 0
    cumulative_size = 1
    for local_rank, dim_size in distinct_coords:
        offset += local_rank * cumulative_size
        cumulative_size *= dim_size
    return (seed + offset) % 2**64


def seed_everything(
    seed: int, *, deterministic: bool, detect_anomaly: bool = False
) -> None:
    """Seed every generator this run draws from.

    torchtitan's ``set_determinism`` semantics; ``Trainer.seed_everything``
    delegates here.
    """
    torch.manual_seed(seed)
    # Hash randomization is not observable in this process (PYTHONHASHSEED is
    # read at interpreter start), but dataloader workers are spawned later and
    # do read it, so upstream sets it here for them. Same spelling.
    os.environ["PYTHONHASHSEED"] = str(seed % 2**32)
    if device_type != "cpu":
        device_module.manual_seed_all(seed)
    if deterministic:
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
