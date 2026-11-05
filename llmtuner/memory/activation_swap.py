"""Opt-in asynchronous NPU activation exchange around decoder layers."""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from functools import wraps
from threading import Lock
from time import perf_counter
from typing import Any

import torch
import torch.nn as nn

from ..accelerator.device import device_type, get_device_info
from ..errors import EnvironmentUnsupportedError
from ..utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "activation_swap_metrics",
    "apply_activation_swap",
    "reset_activation_swap_metrics",
]


def _stream_api():
    return get_device_info()[1]


def _new_event():
    event_type = getattr(_stream_api(), "Event", None)
    return event_type(enable_timing=True) if event_type is not None else None


def _record(event, stream) -> None:
    if event is not None:
        event.record(stream)


@dataclass
class _SwappedActivation:
    device_tensor: torch.Tensor | None
    host_tensor: torch.Tensor | None
    d2h_stream: torch.Stream
    h2d_stream: torch.Stream
    device: torch.device
    released: bool = False


def _should_swap(tensor: torch.Tensor) -> bool:
    return (
        tensor.device.type == "npu"
        and tensor.is_floating_point()
        and not isinstance(tensor, nn.Parameter)
        and tensor._base is None
        and tensor.numel() > 0
    )


_metrics_enabled = False
_metrics_lock = Lock()
_d2h_events: list[tuple[Any, Any]] = []
_h2d_events: list[tuple[Any, Any]] = []
_d2h_bytes = 0
_h2d_bytes = 0
_host_bytes_current = 0
_host_bytes_peak = 0
_copies_d2h = 0
_copies_h2d = 0


@dataclass
class _HostBudget:
    limit: int = 0
    current: int = 0


def _release_host_bytes(num_bytes: int, budget: _HostBudget) -> None:
    global _host_bytes_current
    with _metrics_lock:
        _host_bytes_current -= num_bytes
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
        global _copies_d2h, _d2h_bytes, _host_bytes_current, _host_bytes_peak
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
            _host_bytes_current += host_nbytes
            _host_bytes_peak = max(_host_bytes_peak, _host_bytes_current)
        try:
            host_tensor = torch.empty_like(tensor, device="cpu", pin_memory=True)
        except BaseException:
            _release_host_bytes(host_nbytes, self.budget)
            raise
        weakref.finalize(host_tensor, _release_host_bytes, host_nbytes, self.budget)
        start, end = (_new_event(), _new_event()) if _metrics_enabled else (None, None)
        current = _stream_api().current_stream()
        with _stream_api().stream(self.d2h_stream):
            self.d2h_stream.wait_stream(current)
            _record(start, self.d2h_stream)
            host_tensor.copy_(tensor.detach(), non_blocking=True)
            _record(end, self.d2h_stream)
        packed = _SwappedActivation(
            tensor.detach(),
            host_tensor,
            self.d2h_stream,
            self.h2d_stream,
            tensor.device,
        )
        copy_bytes = tensor.numel() * tensor.element_size()
        with _metrics_lock:
            _d2h_bytes += copy_bytes
            _copies_d2h += 1
            if start is not None:
                _d2h_events.append((start, end))
        self.packed.append(packed)
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
        self.d2h_stream.synchronize()


def _flatten(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _flatten(item)
    elif isinstance(value, tuple | list):
        for item in value:
            yield from _flatten(item)
    else:
        yield value


def _unpack(packed: torch.Tensor | _SwappedActivation) -> torch.Tensor:
    global _copies_h2d, _h2d_bytes
    if not isinstance(packed, _SwappedActivation):
        return packed
    if not packed.released:
        return packed.device_tensor
    current = _stream_api().current_stream()
    start, end = (_new_event(), _new_event()) if _metrics_enabled else (None, None)
    host_nbytes = packed.host_tensor.numel() * packed.host_tensor.element_size()
    with _stream_api().stream(packed.h2d_stream):
        packed.h2d_stream.wait_stream(current)
        packed.h2d_stream.wait_stream(packed.d2h_stream)
        _record(start, packed.h2d_stream)
        device_tensor = packed.host_tensor.to(device=packed.device, non_blocking=True)
        _record(end, packed.h2d_stream)
    current.wait_stream(packed.h2d_stream)
    device_tensor.record_stream(current)
    packed.device_tensor = device_tensor
    packed.released = False
    with _metrics_lock:
        _h2d_bytes += host_nbytes
        _copies_h2d += 1
        if start is not None:
            _h2d_events.append((start, end))
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


def activation_swap_metrics() -> dict[str, float | int]:
    """Return cumulative transfer counters and consume the current timing window."""
    with _metrics_lock:
        d2h_events, h2d_events = _d2h_events.copy(), _h2d_events.copy()
        _d2h_events.clear()
        _h2d_events.clear()
        counters = {
            "d2h_bytes": _d2h_bytes,
            "h2d_bytes": _h2d_bytes,
            "host_bytes_current": _host_bytes_current,
            "host_bytes_peak": _host_bytes_peak,
            "copies_d2h": _copies_d2h,
            "copies_h2d": _copies_h2d,
        }
    started = perf_counter()
    d2h_seconds = _event_seconds(d2h_events)
    h2d_seconds = _event_seconds(h2d_events)
    return {
        **counters,
        "d2h_seconds": d2h_seconds,
        "h2d_seconds": h2d_seconds,
        "timing_collection_wait_seconds": perf_counter() - started,
    }


def reset_activation_swap_metrics() -> None:
    """Enable isolated transfer measurements for a benchmark run."""
    global _copies_d2h, _copies_h2d, _metrics_enabled, _d2h_bytes, _h2d_bytes
    global _host_bytes_current, _host_bytes_peak
    activation_swap_metrics()
    with _metrics_lock:
        _d2h_events.clear()
        _h2d_events.clear()
        _d2h_bytes = 0
        _h2d_bytes = 0
        _copies_d2h = 0
        _copies_h2d = 0
        _host_bytes_peak = _host_bytes_current
        _metrics_enabled = True


def apply_activation_swap(
    model: nn.Module, *, sync_release: bool = False, max_host_bytes: int = 0
) -> nn.Module:
    """Swap decoder activations within a per-model-part live Host byte budget.

    Zero means unlimited. At capacity, retain activations on device. This bounds
    live tensor payloads, not allocator caching, process RSS or virtual moments.
    """
    if isinstance(max_host_bytes, bool) or not isinstance(max_host_bytes, int):
        raise TypeError("max_host_bytes must be a nonnegative integer")
    if max_host_bytes < 0:
        raise ValueError("max_host_bytes must be a nonnegative integer")
    if device_type != "npu":
        raise EnvironmentUnsupportedError(
            "training.activation_swap=True requires an Ascend NPU with "
            "pinned Host memory and asynchronous copy streams."
        )
    layers = getattr(model, "layers", None)
    if not isinstance(layers, nn.Module):
        raise TypeError("activation swap requires a model with a .layers module")

    budget = _HostBudget(limit=max_host_bytes)
    layer_count = 0
    for layer in layers.children():
        forward = layer.forward
        if getattr(forward, "_activation_swap_enabled", False):
            continue

        @wraps(forward)
        def swapped_forward(*args, __forward=forward, __layer=layer, **kwargs):
            if not torch.is_grad_enabled():
                return __forward(*args, **kwargs)
            # Inputs and persistent state already remain on device. Avoid
            # allocating Host copies which release() would never consume.
            state = _ForwardSwapState(
                (args, kwargs, tuple(__layer.parameters()), tuple(__layer.buffers())),
                sync_release=sync_release,
                budget=budget,
            )
            try:
                with torch.autograd.graph.saved_tensors_hooks(state.pack, _unpack):
                    outputs = __forward(*args, **kwargs)
                # Inputs and persistent state are owned outside this invocation.
                # Saved views or detached aliases must not release their storage.
                state.release(
                    (
                        outputs,
                        args,
                        kwargs,
                        tuple(__layer.parameters()),
                        tuple(__layer.buffers()),
                    )
                )
            except BaseException:
                # Release can also fail after D2H copies have been submitted.
                try:
                    state.abort()
                except BaseException:
                    try:
                        logger.exception("Activation swap cleanup failed after forward")
                    except BaseException:
                        pass  # Preserve the original forward/release failure.
                raise
            return outputs

        swapped_forward._activation_swap_enabled = True
        layer.forward = swapped_forward
        layer_count += 1
    logger.info("Applied NPU activation swap to %d layers", layer_count)
    return model
