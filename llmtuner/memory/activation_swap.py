"""Opt-in asynchronous NPU activation exchange around decoder layers."""

from __future__ import annotations

import weakref
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from functools import update_wrapper
from threading import Lock
from time import perf_counter
from typing import Any

import torch
import torch.nn as nn

from ..accelerator.device import device_type, get_device_info
from ..errors import EnvironmentUnsupportedError
from ..models.contracts import DecoderStructure
from ..utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "ActivationSwapResources",
    "ActivationSwapMetrics",
    "apply_activation_swap",
    "drain_activation_swap",
]


def _stream_api():
    return get_device_info()[1]


def _new_event():
    event_type = getattr(_stream_api(), "Event", None)
    return event_type(enable_timing=True) if event_type is not None else None


def _record(event, stream) -> None:
    if event is not None:
        event.record(stream)


def _drain_streams(streams: Iterable[Any]) -> None:
    """Attempt every distinct stream wait before propagating the first failure."""
    first_error: BaseException | None = None
    for stream in {id(stream): stream for stream in streams}.values():
        try:
            stream.synchronize()
        except BaseException as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


_metrics_lock = Lock()


@dataclass
class ActivationSwapMetrics:
    """Instance-owned transfer counters and retryable timing windows."""

    enabled: bool = False
    _d2h_events: list[tuple[Any, Any]] = field(default_factory=list)
    _h2d_events: list[tuple[Any, Any]] = field(default_factory=list)
    _d2h_bytes: int = 0
    _h2d_bytes: int = 0
    _host_bytes_current: int = 0
    _host_bytes_peak: int = 0
    _copies_d2h: int = 0
    _copies_h2d: int = 0

    def snapshot(self) -> dict[str, float | int]:
        """Return cumulative transfer counters and consume the current timing window."""
        with _metrics_lock:
            d2h_events, h2d_events = self._d2h_events.copy(), self._h2d_events.copy()
            self._d2h_events.clear()
            self._h2d_events.clear()
            counters = {
                "d2h_bytes": self._d2h_bytes,
                "h2d_bytes": self._h2d_bytes,
                "host_bytes_current": self._host_bytes_current,
                "host_bytes_peak": self._host_bytes_peak,
                "copies_d2h": self._copies_d2h,
                "copies_h2d": self._copies_h2d,
            }
        started = perf_counter()
        try:
            d2h_seconds = _event_seconds(d2h_events)
            h2d_seconds = _event_seconds(h2d_events)
        except BaseException:
            # Preserve the entire unreported window, including events whose wait
            # already succeeded. New copies may have appended events while the
            # snapshot was being synchronized; keep those behind the older window.
            with _metrics_lock:
                self._d2h_events[:0] = d2h_events
                self._h2d_events[:0] = h2d_events
            raise
        return {
            **counters,
            "d2h_seconds": d2h_seconds,
            "h2d_seconds": h2d_seconds,
            "timing_collection_wait_seconds": perf_counter() - started,
        }

    def reset(self) -> None:
        """Enable isolated transfer measurements for a benchmark run."""
        self.snapshot()
        with _metrics_lock:
            self._d2h_events.clear()
            self._h2d_events.clear()
            self._d2h_bytes = 0
            self._h2d_bytes = 0
            self._copies_d2h = 0
            self._copies_h2d = 0
            self._host_bytes_peak = self._host_bytes_current
            self.enabled = True


@dataclass(eq=False)
class _SwappedActivation:
    device_tensor: torch.Tensor | None
    host_tensor: torch.Tensor | None
    d2h_stream: torch.Stream
    h2d_stream: torch.Stream
    device: torch.device
    budget: _HostBudget
    released: bool = False


def _should_swap(tensor: torch.Tensor) -> bool:
    return (
        tensor.device.type == "npu"
        and tensor.is_floating_point()
        and not isinstance(tensor, nn.Parameter)
        and tensor._base is None
        and tensor.numel() > 0
    )


@dataclass
class _HostBudget:
    metrics: ActivationSwapMetrics = field(default_factory=ActivationSwapMetrics)
    limit: int = 0
    current: int = 0
    pending: weakref.WeakSet[_SwappedActivation] = field(
        default_factory=weakref.WeakSet
    )
    failed_states: list[_ForwardSwapState] = field(default_factory=list)
    failed_transfers: dict[int, _SwappedActivation] = field(default_factory=dict)


def _release_host_bytes(num_bytes: int, budget: _HostBudget) -> None:
    with _metrics_lock:
        budget.metrics._host_bytes_current -= num_bytes
        budget.current -= num_bytes


class _ForwardSwapState:
    def __init__(
        self,
        protected: Any = (),
        *,
        sync_release: bool = False,
        budget: _HostBudget | None = None,
    ) -> None:
        self.sync_release = sync_release
        self.budget = budget if budget is not None else _HostBudget()
        self.d2h_stream = _stream_api().Stream()
        self.h2d_stream = _stream_api().Stream()
        self.packed: list[_SwappedActivation] = []
        self.protected_storages: set[int] = {
            value.untyped_storage()._cdata
            for value in _flatten(protected)
            if isinstance(value, torch.Tensor)
        }

    def pack(self, tensor: torch.Tensor) -> torch.Tensor | _SwappedActivation:
        metrics = self.budget.metrics
        storage_id = tensor.untyped_storage()._cdata
        if storage_id in self.protected_storages or not _should_swap(tensor):
            # A saved view can be unpacked before its swappable base. Keep
            # that shared storage alive rather than exposing a zero-sized base.
            self.protected_storages.add(storage_id)
            return tensor.detach()
        host_nbytes = tensor.numel() * tensor.element_size()
        # Reserve across all layers and outstanding forwards of this model
        # part, before allocating pinned memory. Exhaustion keeps the original
        # activation on device; it must never discard a saved tensor.
        with _metrics_lock:
            if (
                self.budget.limit
                and self.budget.current + host_nbytes > self.budget.limit
            ):
                return tensor.detach()
            self.budget.current += host_nbytes
            metrics._host_bytes_current += host_nbytes
            metrics._host_bytes_peak = max(
                metrics._host_bytes_peak, metrics._host_bytes_current
            )
        try:
            host_tensor = torch.empty_like(tensor, device="cpu", pin_memory=True)
        except BaseException:
            _release_host_bytes(host_nbytes, self.budget)
            raise
        weakref.finalize(host_tensor, _release_host_bytes, host_nbytes, self.budget)
        packed = _SwappedActivation(
            tensor.detach(),
            host_tensor,
            self.d2h_stream,
            self.h2d_stream,
            tensor.device,
            self.budget,
        )
        with _metrics_lock:
            self.budget.pending.add(packed)
        # Establish storage ownership before submitting copies: a copy/event
        # failure can itself be followed by a failed stream synchronization.
        self.packed.append(packed)
        start, end = (_new_event(), _new_event()) if metrics.enabled else (None, None)
        current = _stream_api().current_stream()
        with _stream_api().stream(self.d2h_stream):
            self.d2h_stream.wait_stream(current)
            _record(start, self.d2h_stream)
            host_tensor.copy_(tensor.detach(), non_blocking=True)
            _record(end, self.d2h_stream)
        copy_bytes = tensor.numel() * tensor.element_size()
        with _metrics_lock:
            metrics._d2h_bytes += copy_bytes
            metrics._copies_d2h += 1
            if start is not None:
                metrics._d2h_events.append((start, end))
        return packed

    def release(self, outputs: Any) -> None:
        output_storages = {
            value.untyped_storage()._cdata
            for value in _flatten(outputs)
            if isinstance(value, torch.Tensor)
        }
        if self.sync_release:
            _stream_api().current_stream().wait_stream(self.d2h_stream)
        for packed in self.packed:
            storage = packed.device_tensor.untyped_storage()
            if (
                storage._cdata in output_storages
                or storage._cdata in self.protected_storages
            ):
                continue
            # Release only our reference; autograd and callers may own aliases.
            # record_stream prevents allocator reuse until D2H has completed.
            packed.device_tensor.record_stream(self.d2h_stream)
            packed.device_tensor = None
            packed.released = True

    def abort(self) -> None:
        _drain_streams((self.d2h_stream, self.h2d_stream))


def drain_activation_swap(model: nn.Module) -> None:
    """Wait for outstanding model-owned transfers without discarding autograd state.

    Weak registrations leave saved tensor lifetime with autograd. Failed waits
    retain their stream owners so callers can retry shutdown.
    """
    streams: dict[int, Any] = {}
    budgets: dict[int, _HostBudget] = {}
    for module in model.modules():
        owner = _swap_forward(module)
        if owner is None:
            continue
        budget = owner.budget
        budgets[id(budget)] = budget
    # Keep strong references to every pending activation until all waits finish.
    # Retaining streams alone does not retain the buffers used by their copies.
    with _metrics_lock:
        snapshots = [
            (budget, tuple(budget.pending), tuple(budget.failed_states))
            for budget in budgets.values()
        ]
    for _, pending, failed_states in snapshots:
        for packed in (*pending, *failed_states):
            for stream in (packed.d2h_stream, packed.h2d_stream):
                streams[id(stream)] = stream
    try:
        _drain_streams(streams.values())
    except BaseException:
        # Weak autograd registrations alone cannot own a failed shutdown. Keep
        # transfer buffers alive even after the caller drops the exception.
        with _metrics_lock:
            for budget, pending, _ in snapshots:
                budget.failed_transfers.update(
                    (id(packed), packed) for packed in pending
                )
        raise
    # Failed forward graphs can release their Host storage only after both
    # streams drain. Leave new failures and external autograd owners untouched.
    for budget, pending, failed_states in snapshots:
        drained = {id(state) for state in failed_states}
        with _metrics_lock:
            for packed in pending:
                budget.failed_transfers.pop(id(packed), None)
            budget.failed_states[:] = [
                state for state in budget.failed_states if id(state) not in drained
            ]


@dataclass(slots=True)
class ActivationSwapResources:
    """Own transfer draining independently of execution engine construction."""

    model_parts: tuple[nn.Module, ...]
    _observers: tuple[ActivationSwapMetrics, ...] = field(init=False, repr=False)
    _pending_snapshots: dict[int, dict[str, float | int]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        observers = {}
        for part in self.model_parts:
            for module in part.modules():
                owner = _swap_forward(module)
                if owner is not None:
                    observers[id(owner.budget.metrics)] = owner.budget.metrics
        self._observers = tuple(observers.values())

    def metrics_snapshot(self) -> dict[str, float | int]:
        """Sum local part counters; peaks are sums of each part's own peak."""
        if not self._observers:
            return ActivationSwapMetrics().snapshot()
        # If one observer fails, retain snapshots already consumed from the
        # others so their timing windows are not lost on the next attempt.
        for observer in self._observers:
            if id(observer) not in self._pending_snapshots:
                self._pending_snapshots[id(observer)] = observer.snapshot()
        result: dict[str, float | int] = {}
        for snapshot in self._pending_snapshots.values():
            for key, value in snapshot.items():
                result[key] = result.get(key, 0) + value
        self._pending_snapshots.clear()
        return result

    @property
    def observation_metrics(self) -> dict[str, float]:
        return {
            f"swap/activation_local/{key}": float(value)
            for key, value in self.metrics_snapshot().items()
        }

    def reset_metrics(self) -> None:
        self.metrics_snapshot()
        for observer in self._observers:
            observer.reset()

    def close(self) -> None:
        first_error: BaseException | None = None
        failed: list[nn.Module] = []
        for part in reversed(self.model_parts):
            try:
                drain_activation_swap(part)
            except BaseException as error:
                failed.append(part)
                if first_error is None:
                    first_error = error
        # Retain only owners that still need draining, in their original order.
        self.model_parts = tuple(reversed(failed))
        if first_error is not None:
            raise first_error


def _flatten(value: Any) -> Iterator[Any]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _flatten(item)
    elif isinstance(value, tuple | list):
        for item in value:
            yield from _flatten(item)
    else:
        yield value


def _unpack(packed: torch.Tensor | _SwappedActivation) -> torch.Tensor:
    if not isinstance(packed, _SwappedActivation):
        return packed
    budget = packed.budget
    metrics = budget.metrics
    with _metrics_lock:
        if id(packed) in budget.failed_transfers:
            raise RuntimeError("Failed activation transfer must drain before reuse")
    if not packed.released:
        return packed.device_tensor
    try:
        current = _stream_api().current_stream()
        start, end = (_new_event(), _new_event()) if metrics.enabled else (None, None)
        host_nbytes = packed.host_tensor.numel() * packed.host_tensor.element_size()
        with _stream_api().stream(packed.h2d_stream):
            packed.h2d_stream.wait_stream(current)
            packed.h2d_stream.wait_stream(packed.d2h_stream)
            # Retain the destination before enqueueing any transfer: a later
            # copy/event failure must not leave an in-flight buffer unowned.
            device_tensor = torch.empty_like(packed.host_tensor, device=packed.device)
            packed.device_tensor = device_tensor
            _record(start, packed.h2d_stream)
            device_tensor.copy_(packed.host_tensor, non_blocking=True)
            _record(end, packed.h2d_stream)
        current.wait_stream(packed.h2d_stream)
        device_tensor.record_stream(current)
    except BaseException as error:
        try:
            _drain_streams((packed.d2h_stream, packed.h2d_stream))
        except BaseException as cleanup_error:
            with _metrics_lock:
                budget.failed_transfers[id(packed)] = packed
            raise error from cleanup_error
        packed.device_tensor = None
        raise
    packed.released = False
    with _metrics_lock:
        metrics._h2d_bytes += host_nbytes
        metrics._copies_h2d += 1
        if start is not None:
            metrics._h2d_events.append((start, end))
    packed.host_tensor = None
    return device_tensor


def _event_seconds(events: list[tuple[Any, Any]]) -> float:
    for _, end in events:
        if end is not None:
            end.synchronize()
    return (
        sum(start.elapsed_time(end) for start, end in events if start is not None)
        / 1000
    )


class _ActivationSwapForward:
    """Installed forward with explicit options and shared budget ownership."""

    def __init__(
        self, layer: nn.Module, budget: _HostBudget, sync_release: bool
    ) -> None:
        self.layer = layer
        self.budget = budget
        self.sync_release = sync_release
        update_wrapper(self, layer.forward, updated=())

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not torch.is_grad_enabled():
            return self.__wrapped__(*args, **kwargs)
        # Inputs and persistent state already remain on device. Avoid
        # allocating Host copies which release() would never consume.
        state = _ForwardSwapState(
            (args, kwargs, tuple(self.layer.parameters()), tuple(self.layer.buffers())),
            sync_release=self.sync_release,
            budget=self.budget,
        )
        try:
            with torch.autograd.graph.saved_tensors_hooks(state.pack, _unpack):
                outputs = self.__wrapped__(*args, **kwargs)
            # Inputs and persistent state are owned outside this invocation.
            # Saved views or detached aliases must not release their storage.
            state.release(
                (
                    outputs,
                    args,
                    kwargs,
                    tuple(self.layer.parameters()),
                    tuple(self.layer.buffers()),
                )
            )
        except BaseException:
            # Release can also fail after D2H copies have been submitted.
            try:
                state.abort()
            except BaseException:
                with _metrics_lock:
                    self.budget.failed_states.append(state)
                try:
                    logger.exception("Activation swap cleanup failed after forward")
                except BaseException:
                    pass  # Preserve the original forward/release failure.
            raise
        return outputs


def _swap_forward(module: nn.Module) -> _ActivationSwapForward | None:
    """Find our owner through wrappers preserving the standard __wrapped__ chain."""
    forward = module.forward
    seen = set()
    while forward is not None and id(forward) not in seen:
        if isinstance(forward, _ActivationSwapForward):
            return forward
        seen.add(id(forward))
        forward = getattr(forward, "__wrapped__", None)
    return None


def apply_activation_swap(
    model: nn.Module,
    *,
    sync_release: bool = False,
    max_host_bytes: int = 0,
    metrics: ActivationSwapMetrics | None = None,
) -> nn.Module:
    """Swap decoder activations within a per-model-part live Host byte budget.

    Zero means unlimited. At capacity, retain activations on device. This bounds
    live tensor payloads, not allocator caching, process RSS or virtual moments.
    Reapplication reuses the budget for newly added layers. Changing options or
    combining independently installed layer budgets is rejected before mutation.
    """
    if metrics is not None and not isinstance(metrics, ActivationSwapMetrics):
        raise TypeError("metrics must be ActivationSwapMetrics or None")
    if isinstance(max_host_bytes, bool) or not isinstance(max_host_bytes, int):
        raise TypeError("max_host_bytes must be a nonnegative integer")
    if max_host_bytes < 0:
        raise ValueError("max_host_bytes must be a nonnegative integer")
    if device_type != "npu":
        raise EnvironmentUnsupportedError(
            "training.activation_swap=True requires an Ascend NPU with "
            "pinned Host memory and asynchronous copy streams."
        )
    structure = getattr(model, "decoder_structure", None)
    if not isinstance(structure, DecoderStructure):
        raise TypeError("Activation swap requires a model-owned DecoderStructure")
    # Bind only local blocks: PP may have removed the embedding or head.
    layers = structure.bind_blocks(model)

    # Resolve existing ownership before touching any forward. A partial model
    # extension must share the original budget with outstanding saved tensors.
    budget = None
    targets = tuple(layers.children())
    for layer in targets:
        owner = _swap_forward(layer)
        if owner is None:
            continue
        existing_budget = owner.budget
        existing_sync = owner.sync_release
        if (
            not isinstance(existing_budget, _HostBudget)
            or existing_budget.limit != max_host_bytes
            or existing_sync != sync_release
            or (metrics is not None and existing_budget.metrics is not metrics)
        ):
            raise ValueError("Activation swap reapplication requires identical options")
        if budget is not None and budget is not existing_budget:
            raise ValueError("Activation swap layers have independent Host budgets")
        budget = existing_budget
    if budget is None:
        budget = _HostBudget(
            limit=max_host_bytes,
            metrics=ActivationSwapMetrics() if metrics is None else metrics,
        )
    layer_count = 0
    for layer in targets:
        if _swap_forward(layer) is not None:
            continue

        layer.forward = _ActivationSwapForward(layer, budget, sync_release)
        layer_count += 1
    logger.info("Applied NPU activation swap to %d layers", layer_count)
    return model
