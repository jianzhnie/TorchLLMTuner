"""Explicit, synchronous AdamW moment exchange with bounded staging payload.

Host moments retain the parameter's DTensor ownership. Only local, contiguous
parameter/gradient slices are updated; no layout collective runs in step/load.
Optional lookahead overlaps H2D with the previous slice update. D2H completes
before releasing storage, so synchronous DCP observes ready Host state.
"""

from __future__ import annotations

import math
from collections import defaultdict
from copy import deepcopy
from numbers import Real
from typing import Any

import torch
from torch.distributed.tensor import DTensor, Partial
from torch.optim.adamw import adamw

from ...errors import UnsupportedCombinationError

__all__ = ["AdamWSwap"]


def _local(value: torch.Tensor) -> torch.Tensor:
    return value.to_local() if isinstance(value, DTensor) else value


def _host_moment(parameter: torch.Tensor, *, pin_memory: bool = False) -> torch.Tensor:
    local = _local(parameter)
    host = torch.zeros_like(
        local, device="cpu", pin_memory=pin_memory or local.device.type != "cpu"
    )
    if not isinstance(parameter, DTensor):
        return host
    # from_local moves CPU input onto the mesh device, even with run_check=False.
    # Optimizer state has no autograd graph: retain the existing ownership spec
    # while constructing directly around pinned CPU storage, as CPU offload needs.
    return DTensor(host, parameter._spec, requires_grad=False)


class AdamWSwap(torch.optim.AdamW):
    """AdamW with pinned Host moments and byte-bounded local slice staging.

    max_device_bytes bounds live moment staging across both lookahead slots,
    including parameter/gradient copies when CPU-offloaded. It excludes resident
    parameters/gradients, Adam temporaries, counters and allocator cache.
    max_host_bytes bounds owned moment payload (0 means unlimited). Blocking
    copies guarantee completion before buffers are released or state is saved.
    CPU tensors provide a mathematical fallback without pinned allocations.
    """

    def __init__(
        self,
        params,
        *,
        max_device_bytes: int = 64 * 1024**2,
        max_host_bytes: int = 0,
        prefetch: bool = False,
        memory_fraction: float | None = None,
        update_device: torch.device | str | None = None,
        **kwargs,
    ) -> None:
        for name, value, minimum in (
            ("max_device_bytes", max_device_bytes, 8),
            ("max_host_bytes", max_host_bytes, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not isinstance(prefetch, bool):
            raise ValueError("prefetch must be bool")
        if memory_fraction is not None and (
            isinstance(memory_fraction, bool)
            or not isinstance(memory_fraction, Real)
            or not math.isfinite(memory_fraction)
            or not 0 < memory_fraction <= 1
        ):
            raise ValueError("memory_fraction must be in (0, 1]")
        super().__init__(params, **kwargs)
        self.max_device_bytes = max_device_bytes
        self.max_host_bytes = max_host_bytes
        self.prefetch = prefetch
        self.memory_fraction = memory_fraction
        self.update_device = torch.device(update_device) if update_device else None
        self._copy_stream = None
        self.prefetch_count = 0
        self._failed = False
        self.staging_peak_payload_bytes = 0
        self.h2d_bytes = 0
        self.d2h_bytes = 0
        self.validate_param_groups(self.param_groups)
        for group in self.param_groups:
            for parameter in group["params"]:
                self._validate_parameter(parameter)

    @staticmethod
    def validate_param_groups(groups: list[dict[str, Any]]) -> None:
        """Reject unsupported update modes, including restored group options."""
        for group in groups:
            for name in ("capturable", "differentiable", "foreach"):
                if group.get(name, False):
                    raise UnsupportedCombinationError(
                        f"AdamWSwap requires {name}=False"
                    )
            for name in ("lr", "eps", "weight_decay"):
                value = group[name]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(value)
                    or value < 0
                    or (name == "eps" and value == 0)
                ):
                    raise ValueError(f"AdamWSwap {name} must be a finite valid scalar")
            if len(group["betas"]) != 2 or any(
                isinstance(beta, bool)
                or not isinstance(beta, Real)
                or not 0 <= beta < 1
                for beta in group["betas"]
            ):
                raise ValueError("AdamWSwap betas must be in [0, 1)")

    @staticmethod
    def _validate_parameter(parameter: torch.Tensor) -> None:
        local = _local(parameter)
        if local.dtype not in (torch.bfloat16, torch.float32, torch.float64):
            raise UnsupportedCombinationError(
                "AdamWSwap supports BF16/FP32/FP64 parameters"
            )
        if not local.is_contiguous():
            raise UnsupportedCombinationError(
                "AdamWSwap requires contiguous local parameters"
            )
        if isinstance(parameter, DTensor) and any(
            isinstance(placement, Partial) for placement in parameter.placements
        ):
            raise UnsupportedCombinationError(
                "AdamWSwap requires resolved DTensor ownership"
            )

    @staticmethod
    def _moment_names(group):
        return (
            ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
            if group["amsgrad"]
            else ("exp_avg", "exp_avg_sq")
        )

    def _device(self, parameter):
        return self.update_device or _local(parameter).device

    @property
    def host_state_bytes(self) -> int:
        """Current owned moment payload, excluding counters and allocator cache."""
        return sum(
            _local(state[name]).numel() * _local(state[name]).element_size()
            for state in self.state.values()
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
            if name in state
        )

    def _check_host_capacity(self, additional: int) -> None:
        if (
            self.max_host_bytes
            and self.host_state_bytes + additional > self.max_host_bytes
        ):
            raise MemoryError("AdamWSwap moment payload exceeds max_host_bytes")

    def initialize_state(self, parameters=None) -> None:
        """Create local Host destinations without a synthetic optimizer update."""
        if self._failed:
            raise RuntimeError(
                "AdamWSwap failed; restore a complete training checkpoint"
            )
        if parameters is None:
            parameters = [
                p for g in self.param_groups for p in g["params"] if p.requires_grad
            ]
        groups = {p: g for g in self.param_groups for p in g["params"]}
        missing = [p for p in parameters if not self.state.get(p)]
        self._check_host_capacity(
            sum(
                len(self._moment_names(groups[p]))
                * _local(p).numel()
                * p.element_size()
                for p in missing
            )
        )
        pending = {}
        for parameter in missing:
            self._validate_parameter(parameter)
            if (
                _local(parameter).numel()
                and len(self._moment_names(groups[parameter]))
                * parameter.element_size()
                > self.max_device_bytes
            ):
                raise ValueError(
                    "max_device_bytes cannot hold one element of each moment"
                )
            pending[parameter] = {
                "step": torch.tensor(
                    0.0,
                    dtype=(
                        torch.float64
                        if torch.get_default_dtype() == torch.float64
                        else torch.float32
                    ),
                ),
                **{
                    name: _host_moment(
                        parameter, pin_memory=self._device(parameter).type != "cpu"
                    )
                    for name in self._moment_names(groups[parameter])
                },
            }
        self.state.update(pending)

    @torch.no_grad()
    def step(self, closure=None):
        """Update only parameters with gradients, completing D2H before return."""
        if self._failed:
            raise RuntimeError(
                "AdamWSwap failed; restore a complete training checkpoint"
            )
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.validate_param_groups(self.param_groups)
        active = [
            p for g in self.param_groups for p in g["params"] if p.grad is not None
        ]
        for parameter in active:
            self._validate_parameter(parameter)
            gradient = parameter.grad
            if gradient.is_sparse or not _local(gradient).is_contiguous():
                raise UnsupportedCombinationError(
                    "AdamWSwap requires dense contiguous gradients"
                )
            if isinstance(parameter, DTensor) and (
                not isinstance(gradient, DTensor)
                or gradient.placements != parameter.placements
                or gradient.device_mesh != parameter.device_mesh
            ):
                raise UnsupportedCombinationError(
                    "AdamWSwap gradient ownership must match parameter"
                )
        self.initialize_state(active)
        jobs = iter(self._slice_jobs(active))
        staged = None
        try:
            job = next(jobs, None)
            if job is not None:
                staged = self._stage(job)
            while job is not None:
                following = next(jobs, None)
                # Allocation occurs before computing the current slice. Both
                # payloads count against the same budget, including CPU-offload
                # parameter/gradient buffers. No shared class-level stream/maps.
                next_staged = (
                    self._stage(following) if self.prefetch and following else None
                )
                self._update_slice(job, staged)
                if next_staged is not None:
                    self.staging_peak_payload_bytes = max(
                        self.staging_peak_payload_bytes, staged[2] + next_staged[2]
                    )
                staged = None
                job = following
                staged = (
                    next_staged
                    if next_staged is not None
                    else (self._stage(job) if job is not None else None)
                )
        except BaseException:
            self._failed = True
            raise
        finally:
            # In particular, a failed update must not free a pending H2D source
            # or destination while a side stream still uses it.
            if self._copy_stream is not None:
                self._copy_stream.synchronize()
        return loss

    def _slice_jobs(self, active):
        devices = {self._device(p) for p in active}
        if len(devices) > 1:
            raise UnsupportedCombinationError("AdamWSwap requires one update device")
        budget = self.max_device_bytes
        if active and self.memory_fraction is not None:
            device = self._device(active[0])
            if device.type == "cpu":
                raise UnsupportedCombinationError(
                    "memory_fraction requires an accelerator"
                )
            backend = getattr(torch, device.type)
            available = backend.get_device_properties(
                device
            ).total_memory - backend.memory_allocated(device)
            budget = min(budget, int(available * self.memory_fraction))
        slots = 2 if self.prefetch else 1
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                local = _local(parameter)
                # An empty ownership shard needs neither kernel support nor a
                # staging slot. Keep its logical Adam counter in sync even when
                # the configured budget cannot fit a nonempty lookahead pair.
                if not local.numel():
                    self.state[parameter]["step"].add_(1)
                    continue
                device = self._device(parameter)
                if (
                    group["fused"]
                    and device.type == "npu"
                    and local.dtype == torch.float64
                ):
                    raise UnsupportedCombinationError(
                        "NPU fused AdamWSwap requires BF16/FP32"
                    )
                names = self._moment_names(group)
                copies = len(names) + (2 if local.device != device else 0)
                chunk = budget // (slots * copies * local.element_size())
                if chunk < 1:
                    raise MemoryError(
                        "swap budget cannot hold one staged element per slot"
                    )
                for start in range(0, local.numel(), chunk):
                    yield (
                        group,
                        parameter,
                        start,
                        min(start + chunk, local.numel()),
                        names,
                    )

    def _stage(self, job):
        group, parameter, start, end, names = job
        device = self._device(parameter)
        value = _local(parameter).view(-1)[start:end]
        gradient = _local(parameter.grad).view(-1)[start:end]
        sources = [
            _local(self.state[parameter][name]).view(-1)[start:end] for name in names
        ]
        offload = value.device != device
        if offload:
            sources += [value, gradient]
        backend = getattr(torch, device.type) if device.type != "cpu" else None
        event = None
        if self.prefetch and backend is not None:
            if self._copy_stream is None:
                self._copy_stream = backend.Stream(device=device)
            # CPU-offload gradients may have just been produced by D2H.
            self._copy_stream.wait_stream(backend.current_stream(device))
            with backend.stream(self._copy_stream):
                buffers = [source.to(device, non_blocking=True) for source in sources]
                event = self._copy_stream.record_event()
            self.prefetch_count += 1
        else:
            buffers = [source.to(device) for source in sources]
        payload = sum(buffer.numel() * buffer.element_size() for buffer in buffers)
        self.staging_peak_payload_bytes = max(self.staging_peak_payload_bytes, payload)
        if device.type != "cpu":
            self.h2d_bytes += payload
        return buffers, event, payload, value, gradient, offload

    def _update_slice(self, job, staged):
        group, parameter, start, end, names = job
        buffers, event, _, value, gradient, offload = staged
        device = self._device(parameter)
        if event is not None:
            stream = getattr(torch, device.type).current_stream(device)
            stream.wait_event(event)
            for buffer in buffers:
                buffer.record_stream(stream)
        moments = buffers[: len(names)]
        if offload:
            value, gradient = buffers[-2:]
        state = self.state[parameter]
        counter = state["step"].to(device if group["fused"] else "cpu", copy=True)
        adamw(
            [value],
            [gradient],
            [moments[0]],
            [moments[1]],
            [moments[2]] if group["amsgrad"] else [],
            [counter],
            foreach=False,
            capturable=False,
            differentiable=False,
            fused=bool(group["fused"]),
            amsgrad=group["amsgrad"],
            beta1=group["betas"][0],
            beta2=group["betas"][1],
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            maximize=group["maximize"],
        )
        # Blocking copies complete before dropping the device buffers. Never
        # resize storage while a transfer still references it.
        for name, moment in zip(names, moments, strict=True):
            _local(state[name]).view(-1)[start:end].copy_(moment)
        payload = sum(moment.numel() * moment.element_size() for moment in moments)
        if offload:
            _local(parameter).view(-1)[start:end].copy_(value)
            payload += value.numel() * value.element_size()
        if device.type != "cpu":
            self.d2h_bytes += payload
        if end == _local(parameter).numel():
            state["step"].add_(1)

    def state_dict(self) -> dict[str, Any]:
        """Expose ordinary AdamW state with Host-owned moment tensors."""
        if self._failed:
            raise RuntimeError(
                "AdamWSwap failed; restore a complete training checkpoint"
            )
        return super().state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore into pinned destinations without materializing all moments on NPU.

        Mirrors the optimizer pre/post-hook and parameter-index mapping contract.
        Validation precedes copies; a copy failure marks state unusable until a
        successful restore. Existing destinations are reused on repeated loads.
        """
        state_dict = state_dict.copy()
        for hook in self._optimizer_load_state_dict_pre_hooks.values():
            result = hook(self, state_dict)
            if result is not None:
                state_dict = result
        groups = deepcopy(state_dict["param_groups"])
        self.validate_param_groups(groups)
        if len(groups) != len(self.param_groups) or any(
            len(saved["params"]) != len(current["params"])
            for saved, current in zip(groups, self.param_groups, strict=False)
        ):
            raise ValueError("AdamWSwap checkpoint parameter groups do not match")
        mapping = {
            key: parameter
            for saved, current in zip(groups, self.param_groups, strict=True)
            for key, parameter in zip(saved["params"], current["params"], strict=True)
        }
        saved_groups = {
            mapping[key]: group for group in groups for key in group["params"]
        }
        plans = []
        for key, incoming in state_dict["state"].items():
            if key not in mapping or set(incoming) != {
                "step",
                *self._moment_names(saved_groups[mapping[key]]),
            }:
                raise ValueError("AdamWSwap checkpoint has invalid Adam state keys")
            parameter = mapping[key]
            for name in self._moment_names(saved_groups[parameter]):
                moment = incoming[name]
                if (
                    not isinstance(moment, torch.Tensor)
                    or not moment.is_floating_point()
                    or moment.shape != parameter.shape
                    or _local(moment).shape != _local(parameter).shape
                    or isinstance(moment, DTensor) != isinstance(parameter, DTensor)
                ):
                    raise ValueError(
                        "AdamWSwap checkpoint moment shape/ownership mismatch"
                    )
                if isinstance(moment, DTensor) and (
                    moment.placements != parameter.placements
                    or moment.device_mesh != parameter.device_mesh
                ):
                    raise ValueError("AdamWSwap checkpoint moment ownership mismatch")
            step = incoming["step"]
            if isinstance(step, torch.Tensor):
                if step.numel() != 1 or step.is_complex():
                    raise ValueError("AdamWSwap checkpoint step must be a scalar")
                step = step.item()
            if (
                isinstance(step, bool)
                or not isinstance(step, Real)
                or not math.isfinite(step)
                or step < 0
            ):
                raise ValueError(
                    "AdamWSwap checkpoint step must be nonnegative and finite"
                )
            plans.append((parameter, incoming, step))
        additional = sum(
            _local(p).numel() * p.element_size()
            for p, _, _ in plans
            for name in self._moment_names(saved_groups[p])
            if name not in self.state.get(p, {})
        )
        self._check_host_capacity(additional)
        restored = defaultdict(dict)
        for parameter, _incoming, step in plans:
            old = self.state.get(parameter, {})
            restored[parameter] = {
                "step": torch.tensor(
                    step,
                    dtype=(
                        torch.float64
                        if isinstance(_incoming["step"], torch.Tensor)
                        and _incoming["step"].dtype == torch.float64
                        else torch.float32
                    ),
                ),
                **{
                    name: old[name]
                    if name in old
                    else _host_moment(
                        parameter, pin_memory=self._device(parameter).type != "cpu"
                    )
                    for name in self._moment_names(saved_groups[parameter])
                },
            }
        self._failed = True
        for parameter, incoming, _ in plans:
            for name in self._moment_names(saved_groups[parameter]):
                _local(restored[parameter][name]).copy_(_local(incoming[name]))
        for saved, current in zip(groups, self.param_groups, strict=True):
            saved["params"] = current["params"]
            if "param_names" not in saved and "param_names" in current:
                saved["param_names"] = current["param_names"]
        self.state = restored
        self.param_groups = groups
        self._failed = False
        try:
            for hook in self._optimizer_load_state_dict_post_hooks.values():
                hook(self)
        except BaseException:
            self._failed = True
            raise
