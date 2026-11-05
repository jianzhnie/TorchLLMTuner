"""Optimizer-state storage policies for supported accelerator backends."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from functools import wraps
from typing import Any

import torch
from torch.distributed.tensor import DTensor

from ...accelerator.device import device_type
from ...errors import EnvironmentUnsupportedError, UnsupportedCombinationError
from .state_dict import (
    validate_optimizer_state_tensor,
    validate_optimizer_step_counter,
)

try:
    import torch_npu
except ImportError as error:
    torch_npu = None
    TORCH_NPU_IMPORT_ERROR = error

__all__ = [
    "install_state_residency",
    "validate_checkpoint_configuration",
    "validate_virtual_optimizer_groups",
]


def validate_checkpoint_configuration(backend: str, async_mode: str) -> None:
    """Reject checkpoint modes that cannot preserve virtual optimizer state."""
    if backend == "dcp" and async_mode in (
        "disabled",
        "async",
        "async_with_pinned_mem",
    ):
        return
    raise UnsupportedCombinationError(
        "optimizer.state_residency='npu_virtual' requires "
        "checkpoint.backend='dcp' with async_mode='disabled', 'async' or "
        "'async_with_pinned_mem'; "
        f"got backend={backend!r}, async_mode={async_mode!r}. "
        "torch_checkpointing and HF checkpoint paths do not preserve "
        "swap-backed optimizer storage."
    )


def _empty_with_swapped_memory(
    shape: torch.Size, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if torch_npu is None:
        raise EnvironmentUnsupportedError(
            "Virtual optimizer allocation requires a working torch-npu installation"
        ) from TORCH_NPU_IMPORT_ERROR
    if not callable(getattr(torch_npu, "empty_with_swapped_memory", None)):
        raise EnvironmentUnsupportedError(
            "Virtual optimizer requires torch_npu.empty_with_swapped_memory; "
            "install a compatible torch-npu build or select state_residency='device'."
        )
    return torch_npu.empty_with_swapped_memory(shape, dtype=dtype, device=device)


def _allocate_moment(
    parameter: torch.Tensor, allocator: Callable[..., torch.Tensor]
) -> torch.Tensor:
    local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
    if local.numel() == 0:
        allocated = torch.empty_like(local)
    else:
        allocated = allocator(local.size(), dtype=local.dtype, device=local.device)
    allocated.zero_()
    if not isinstance(parameter, DTensor):
        return allocated
    return DTensor.from_local(
        allocated,
        parameter.device_mesh,
        parameter.placements,
        shape=parameter.size(),
        stride=parameter.stride(),
        run_check=False,
    )


def _initialize_moments(
    optimizer: torch.optim.Optimizer,
    allocator: Callable[..., torch.Tensor],
    *,
    parameters: Iterable[torch.Tensor] | None = None,
) -> None:
    """Initialize selected parameters, or live gradients for an ordinary step."""
    if parameters is None:
        parameters = (
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        )
    for parameter in parameters:
        state: dict[str, Any] = optimizer.state[parameter]
        # Publish only after every allocation succeeds. A failed allocation
        # must not leave a nonempty Adam state missing required fields.
        pending = {}
        if "step" not in state:
            # Match Adam's scalar dtype independently of model precision.
            # A BF16 step counter stops incrementing at 256.
            step_dtype = (
                torch.float64
                if torch.get_default_dtype() == torch.float64
                else torch.float32
            )
            pending["step"] = torch.tensor(
                0.0, dtype=step_dtype, device=parameter.device
            )
        for name in ("exp_avg", "exp_avg_sq"):
            if name not in state:
                pending[name] = _allocate_moment(parameter, allocator)
        state.update(pending)


def validate_virtual_optimizer_groups(groups: Iterable[Mapping[str, Any]]) -> None:
    """Validate both live groups and groups restored from a checkpoint."""
    for group in groups:
        for option in ("amsgrad", "differentiable", "capturable", "fused"):
            if group.get(option, False):
                raise UnsupportedCombinationError(
                    f"optimizer.state_residency='npu_virtual' does not support "
                    f"{option}=True; use the default device residency or disable "
                    "this optimizer option."
                )


def validate_virtual_optimizer_state(
    optimizer: torch.optim.Optimizer, state_dict: Mapping[str, Any]
) -> None:
    """Check restored Adam state before allocating or changing virtual storage."""
    # load_state_dict replaces group options as well as moments. Reject
    # unsupported checkpoint settings before allocating or mutating state.
    validate_virtual_optimizer_groups(state_dict["param_groups"])
    # PyTorch checks these after pre-hooks. Do it before materializing
    # potentially model-sized Host-backed destinations for an invalid load.
    saved_groups = state_dict["param_groups"]
    if len(saved_groups) != len(optimizer.param_groups):
        raise ValueError("loaded state dict has a different number of parameter groups")
    if any(
        len(saved["params"]) != len(current["params"])
        for saved, current in zip(saved_groups, optimizer.param_groups, strict=True)
    ):
        raise ValueError(
            "loaded state dict contains a parameter group "
            "that doesn't match the size of optimizer's group"
        )
    # copy_ accepts broadcasting, but Adam moments must match the parameter
    # exactly. Validate before allocating Host storage or replacing state.
    for saved, current in zip(saved_groups, optimizer.param_groups, strict=True):
        for key, parameter in zip(saved["params"], current["params"], strict=True):
            incoming = state_dict["state"].get(key, {})
            if not incoming:
                continue  # Adam legitimately has no state before its first step.
            for name in ("exp_avg", "exp_avg_sq"):
                try:
                    validate_optimizer_state_tensor(
                        incoming.get(name), parameter, key=f"state.{key}.{name}"
                    )
                except ValueError as error:
                    raise ValueError(
                        f"Invalid Adam state {name!r} for parameter {key!r}: "
                        f"Virtual Adam moment ownership or shape mismatch: {error}"
                    ) from error
            try:
                validate_optimizer_step_counter(
                    incoming.get("step"), key=f"state.{key}.step"
                )
            except ValueError as error:
                raise ValueError(
                    f"Invalid Adam step for parameter {key!r}: {error}"
                ) from error


def install_state_residency(
    optimizer: torch.optim.Optimizer,
    residency: str,
    *,
    allocator: Callable[..., torch.Tensor] = _empty_with_swapped_memory,
) -> None:
    """Install lazy Adam moment allocation for the selected storage policy."""
    if residency == "device":
        return
    if residency != "npu_virtual":
        raise ValueError(f"Unknown optimizer state residency {residency!r}")
    if device_type != "npu":
        raise EnvironmentUnsupportedError(
            "optimizer.state_residency='npu_virtual' requires Ascend NPU and "
            "torch_npu.empty_with_swapped_memory; use 'device' on this host."
        )
    validate_virtual_optimizer_groups(optimizer.param_groups)
    if not isinstance(optimizer, torch.optim.Adam | torch.optim.AdamW):
        raise UnsupportedCombinationError(
            "optimizer.state_residency='npu_virtual' supports Adam/AdamW only"
        )

    def initialize_virtual_moments(
        optim: torch.optim.Optimizer, args: tuple, kwargs: dict
    ) -> tuple[tuple, dict] | None:
        # Live groups can change after installation (including add_param_group).
        validate_virtual_optimizer_groups(optim.param_groups)
        _initialize_moments(optim, allocator)
        closure = (
            kwargs.get("closure")
            if "closure" in kwargs
            else (args[1] if len(args) > 1 else None)
        )
        if closure is None:
            return None

        @wraps(closure)
        def initialize_after_closure(*closure_args, **closure_kwargs):
            result = closure(*closure_args, **closure_kwargs)
            # Adam invokes closures after step pre-hooks. Gradients created
            # there must receive virtual moments before Adam's lazy init.
            validate_virtual_optimizer_groups(optim.param_groups)
            _initialize_moments(optim, allocator)
            return result

        if "closure" in kwargs:
            return args, {**kwargs, "closure": initialize_after_closure}
        return (args[0], initialize_after_closure, *args[2:]), kwargs

    optimizer.register_step_pre_hook(initialize_virtual_moments)

    destinations: list[tuple[torch.Tensor, str, torch.Tensor]] = []

    def prepare_virtual_load(
        optim: torch.optim.Optimizer, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        destinations.clear()
        validate_virtual_optimizer_state(optim, state_dict)
        # Older virtual checkpoints inherited the default model dtype for
        # counters. Preserve their saved value but prevent further rounding;
        # never modify the caller's checkpoint tensors or nested dictionaries.
        restored_states = dict(state_dict["state"])
        for key, state in restored_states.items():
            step = state.get("step")
            if isinstance(step, torch.Tensor) and step.dtype in (
                torch.float16,
                torch.bfloat16,
            ):
                restored_states[key] = {**state, "step": step.to(torch.float32)}
        restored_parameters = [
            parameter
            for saved, current in zip(
                state_dict["param_groups"], optim.param_groups, strict=True
            )
            for key, parameter in zip(saved["params"], current["params"], strict=True)
            if restored_states.get(key)
        ]
        _initialize_moments(optim, allocator, parameters=restored_parameters)
        for parameter in restored_parameters:
            for name in ("exp_avg", "exp_avg_sq"):
                destinations.append((parameter, name, optim.state[parameter][name]))
        return {**state_dict, "state": restored_states}

    def restore_virtual_destinations(optim: torch.optim.Optimizer) -> None:
        # PyTorch may replace moments while casting checkpoint tensors. Keep
        # the original swap-backed storage, including for repeated loads.
        try:
            pending = [
                (parameter, name, destination, optim.state[parameter].get(name))
                for parameter, name, destination in destinations
            ]
            # Reinstall all owned destinations before any device copy can fail.
            # Otherwise a retry may adopt an ordinary checkpoint tensor as its
            # new destination and silently abandon the virtual storage policy.
            for parameter, name, destination, incoming in pending:
                if incoming is not None:
                    optim.state[parameter][name] = destination
            with torch.no_grad():
                for _, _, destination, incoming in pending:
                    if incoming is None:
                        continue
                    destination.copy_(incoming)
        finally:
            destinations.clear()

    optimizer.register_load_state_dict_pre_hook(prepare_virtual_load)
    optimizer.register_load_state_dict_post_hook(restore_virtual_destinations)
