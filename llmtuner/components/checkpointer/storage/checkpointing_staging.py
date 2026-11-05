"""Instance-scoped NPU event ordering for torch_checkpointing 0.1.0."""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import torch

from ....accelerator.device import device_type
from ....utils.logging import get_logger

try:
    from torch_checkpointing.staging import DefaultStager
except ImportError as error:
    DefaultStager = None
    BACKEND_IMPORT_ERROR = error

logger = get_logger(__name__)


def adapt_npu_staging(manager: Any) -> bool:
    """Wait for the producer event on this manager's staging worker.

    NPU elapsed_time can reject a producer event after the dependent stream's
    end event has been synchronized, even when both events query as complete.
    Waiting for the producer explicitly also makes its timing state ready.
    The async stage() submission and checkpoint upload remain asynchronous;
    this wait runs inside _stage, where copies already depend on the producer.
    """
    if device_type != "npu":
        return False
    saver = getattr(manager, "_saver", None)
    stager = getattr(saver, "_checkpoint_stager", None)
    if stager is None:
        return False
    try:
        release = version("torch_checkpointing")
    except PackageNotFoundError:
        # Source or editable installs may provide the module without metadata.
        return False
    if release != "0.1.0":
        return False
    if DefaultStager is None:
        raise ImportError("torch_checkpointing staging API is unavailable") from (
            BACKEND_IMPORT_ERROR
        )

    if type(stager) is not DefaultStager or not stager._accelerator_available:
        return False
    if getattr(stager, "_llmtuner_npu_event_wait", False):
        return False
    original_stage = stager._stage

    @wraps(original_stage)
    def stage_after_producer(
        state_dict: dict[str, Any],
        current_stream_ready: torch.Event | None,
        keys_not_requiring_copy: Sequence[str] = (),
    ) -> dict[str, Any]:
        if current_stream_ready is not None:
            # Match DefaultStager's failure contract even if the wait raises
            # before its original _stage can invalidate the previous snapshot.
            stager._staged_state_dict = None
            current_stream_ready.synchronize()
        return original_stage(state_dict, current_stream_ready, keys_not_requiring_copy)

    stager._stage = stage_after_producer
    stager._llmtuner_npu_event_wait = True
    logger.info("Applied instance-scoped torch_checkpointing 0.1.0 NPU staging wait")
    return True
