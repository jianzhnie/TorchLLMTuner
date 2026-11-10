"""Instance-scoped NPU staging compatibility for torch_checkpointing."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future
from contextlib import nullcontext
from contextvars import copy_context
from functools import wraps
from types import MethodType
from typing import Any

import torch

from ....accelerator.device import device_type
from ....utils.logging import get_logger

try:
    from torch_checkpointing.logging_utils import EventLogger, EventType
    from torch_checkpointing.staging import DefaultStager
    from torch_checkpointing.utils import set_thread_name_safe
except ImportError as error:
    DefaultStager = None
    BACKEND_IMPORT_ERROR = error

logger = get_logger(__name__)


def _npu_stage(
    self: Any,
    state_dict: dict[str, Any],
    keys_not_requiring_copy: Sequence[str] = (),
) -> dict[str, Any] | Future[dict[str, Any]]:
    """Submit a producer event without entering an upstream CUDA context."""
    with torch.npu.device(self._llmtuner_npu_device):
        ready = torch.Event(enable_timing=True)
        ready.record(torch.accelerator.current_stream())
    if self._config.use_async_staging:
        if self._staging_executor is None:
            raise RuntimeError("Async NPU staging requires its staging executor")
        return self._staging_executor.submit(
            copy_context().run,
            self._stage_on_worker,
            state_dict,
            ready,
            keys_not_requiring_copy,
        )
    return self._stage(state_dict, ready, keys_not_requiring_copy)


def _npu_init_worker(self: Any) -> None:
    torch.npu.set_device(self._llmtuner_npu_device)


def _npu_copy_stage(
    self: Any,
    state_dict: dict[str, Any],
    ready: torch.Event,
    keys_not_requiring_copy: Sequence[str] = (),
) -> dict[str, Any]:
    """Preserve upstream cache, exclusions, logging and event completion semantics."""
    event_logger = EventLogger()
    logger.info(
        "Initiating NPU checkpoint staging",
        extra=event_logger(event_type=EventType.CHECKPOINT_STAGING_START),
    )
    set_thread_name_safe(self._config.thread_name)
    self._staged_state_dict = None
    payload = state_dict
    if keys_not_requiring_copy:
        payload = state_dict.copy()
        for key in keys_not_requiring_copy:
            payload.pop(key, None)
    with torch.npu.device(self._llmtuner_npu_device):
        # NPU timing needs the producer event to be explicitly completed before
        # elapsed_time, even after the dependent stream's end event completes.
        ready.synchronize()
        with self._staging_stream or nullcontext():
            ready.wait()
            staged = self._state_dict_stager.stage(payload)
            end = torch.Event(enable_timing=True)
            end.record()
        end.synchronize()
        elapsed = ready.elapsed_time(end)
    for key in keys_not_requiring_copy:
        if key in state_dict:
            staged[key] = state_dict[key]
    logger.info(
        "NPU checkpoint staging completed",
        extra=event_logger(
            EventType.LOG_METRIC,
            metric_name=f"{self._metric_prefix}.staging.e2e.latency_ms",
            end_to_end=True,
            value=elapsed,
        ),
    )
    logger.info(
        "Staging copy done",
        extra=event_logger(
            EventType.CHECKPOINT_STAGING_END,
            metric_name=(
                f"{self._metric_prefix}.execute.subprocess_comm."
                "wait_for_tensor_copy.latency_ms"
            ),
        ),
    )
    self._staged_state_dict = staged
    return staged


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
    if DefaultStager is None:
        raise ImportError("torch_checkpointing staging API is unavailable") from (
            BACKEND_IMPORT_ERROR
        )

    if type(stager) is not DefaultStager or not stager._accelerator_available:
        return False
    if getattr(stager, "_llmtuner_npu_event_wait", False):
        return False
    if hasattr(stager, "_stage_on_worker"):
        stager._llmtuner_npu_device = torch.npu.current_device()
        stager.stage = MethodType(_npu_stage, stager)
        stager._init_worker_thread = MethodType(_npu_init_worker, stager)
        stager._stage = MethodType(_npu_copy_stage, stager)
        stager._llmtuner_npu_event_wait = True
        logger.info("Applied instance-scoped NPU worker staging")
        return True
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
