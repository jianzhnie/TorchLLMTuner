"""Initialize optimizer state and serialize it by parameter name.

Adapted from torchtitan ``components/optimizer/utils.py``. Native optimizer
state uses integer parameter indices; these helpers convert it to a flat,
fully qualified name (FQN) layout for distributed checkpointing.
``init_optim_state`` materializes lazy state while preserving gradients,
learning rates, existing state, and the first real Adam step.
"""

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch.distributed.tensor import DTensor

from .adamw_swap import AdamWSwap
from .flex_shard.dist_muon import DistMuon

__all__ = [
    "init_optim_state",
    "get_flat_optim_state_dict",
    "load_flat_optim_state_dict",
    "collect_local_optimizer_state",
]


def collect_local_optimizer_state(
    optimizers: Iterable[torch.optim.Optimizer],
) -> dict[str, Any]:
    """Materialize and merge local FQN state without layout communication.

    Callers coordinate local errors before entering TP/EP layout collectives.
    Optimizer initialization and state-dict hooks must not communicate here.
    """
    result: dict[str, Any] = {}
    for optimizer in optimizers:
        init_optim_state(optimizer)
        state = get_flat_optim_state_dict(optimizer)
        duplicated = result.keys() & state.keys()
        if duplicated:
            raise ValueError(
                "Optimizer checkpoint keys overlap across inner optimizers: "
                f"{sorted(duplicated)}. Model parts must use unique parameter FQNs."
            )
        result.update(state)
    return result


def init_optim_state(optim: torch.optim.Optimizer) -> None:
    """Materialize lazy state while preserving parameters and training state.

    Two callers need this, for the same underlying reason: PyTorch creates Adam's
    ``exp_avg``/``exp_avg_sq`` lazily, on the first ``step()``.

    * **Saving.** DCP reads whatever tensors the state dict reports. An
      optimizer that has not stepped reports none, so the checkpoint would carry
      model weights and no optimizer state -- silently, since the save succeeds.
    * **Loading.** DCP writes *into* the tensors a state dict reports rather
      than calling ``load_state_dict``. A fresh optimizer reports none, so there
      is nothing to write into and the restored run trains with a cold
      optimizer. ``OptimizersContainer.state_dict`` calls this for that case.

    The step runs with zero gradients and ``lr=0`` so parameters are untouched;
    the materialized state is then reset, so the first real update is still Adam
    step 1. Existing gradients and optimizer state are preserved.

    Native optimizer step hooks run during materialization, before checkpoint
    loading restores the trainer's step counter. Hooks must not assume every
    call is a completed training update or enter distributed collectives: this
    helper runs inside checkpoint's local error-coordination boundary.

    DistMuon explicitly initializes local momentum instead: a synthetic step
    would execute layout collectives inside checkpoint's local error boundary.

    No-op when every parameter that requires a gradient already has state.

    An ``OptimizersContainer`` is itself an ``Optimizer``, so wrapping this in a
    per-inner-optimizer loop -- which the container does -- is the whole of the
    multi-optimizer case: the reset branch below is type-gated on Adam, and
    would be skipped silently if a container were handed in whole.
    """
    if isinstance(optim, DistMuon | AdamWSwap):
        optim.initialize_state()
        return

    params = [
        param for param_group in optim.param_groups for param in param_group["params"]
    ]
    params_to_initialize = [
        param for param in params if param.requires_grad and not optim.state.get(param)
    ]
    if not params_to_initialize:
        return

    saved_grads = [param.grad for param in params]
    saved_empty_states = {
        param: optim.state[param]
        for param in params_to_initialize
        if param in optim.state
    }
    saved_lrs = []
    saved_adam_options = []
    try:
        for param in params:
            param.grad = None
        for param in params_to_initialize:
            param.grad = torch.zeros_like(param)

        # A zero learning rate keeps the materialization step from updating weights.
        for param_group in optim.param_groups:
            if "lr" in param_group:
                saved_lrs.append((param_group, param_group["lr"]))
                param_group["lr"] = (
                    torch.zeros_like(param_group["lr"])
                    if isinstance(param_group["lr"], torch.Tensor)
                    else 0.0
                )
        # Even at lr=0, Adam computes its normalized update: 0/0 becomes NaN
        # when eps is zero or underflows in FP16, and zero * NaN corrupts the
        # parameter. Use an exactly representable denominator floor only for
        # this state-allocation step; restore the user's epsilon afterward.
        if isinstance(optim, torch.optim.Adam | torch.optim.AdamW):
            for param_group in optim.param_groups:
                saved_adam_options.append(
                    (param_group, param_group["eps"], param_group["weight_decay"])
                )
                param_group["eps"] = 1.0
                # Coupled Adam decay can overflow even for finite weights.
                # State allocation must not depend on parameter magnitudes.
                param_group["weight_decay"] = 0.0
        optim.step(closure=None)

        # A zero learning rate keeps parameters unchanged, but Adam still advances
        # its step counter (and coupled weight decay can update its moments). Reset
        # the materialized state so the first real update remains Adam step 1.
        if isinstance(optim, torch.optim.Adam | torch.optim.AdamW):
            for param in params_to_initialize:
                state = optim.state[param]
                for name in ("step", "exp_avg", "exp_avg_sq"):
                    state[name].zero_()
                if "max_exp_avg_sq" in state:
                    state["max_exp_avg_sq"].zero_()

    except BaseException:
        # Adam may allocate one field or complete one parameter before an
        # allocation/hook fails. A nonempty partial entry would otherwise be
        # mistaken for initialized state on the next checkpoint attempt.
        for param in params_to_initialize:
            optim.state.pop(param, None)
            if param in saved_empty_states:
                original_state = saved_empty_states[param]
                original_state.clear()
                optim.state[param] = original_state
        raise
    finally:
        for param_group, epsilon, weight_decay in saved_adam_options:
            param_group["eps"] = epsilon
            param_group["weight_decay"] = weight_decay
        for param_group, lr in saved_lrs:
            param_group["lr"] = lr
        for param, grad in zip(params, saved_grads, strict=True):
            param.grad = grad


def get_flat_optim_state_dict(optim: torch.optim.Optimizer) -> dict[str, Any]:
    """Return a flat, FQN-keyed optimizer state dict ready for DCP.

    Output keys are ``state.{fqn}.{state_name}`` and ``param_groups.{fqn}.{key}``.
    The flat layout avoids the integer ``param_group`` index collisions that break
    pipeline-parallel checkpoints (multiple chunks reusing index 0).

    The optimizer state must already exist; call ``init_optim_state`` first.
    """
    fqn_sd = optim_state_dict_to_fqn_keys(optim.state_dict())

    flat: dict[str, Any] = {}
    for fqn, state in fqn_sd["state"].items():
        _flatten_state(state, prefix=f"state.{fqn}", out=flat)
    conflicts = _conflicting_state_paths(flat)
    if conflicts:
        raise ValueError(
            f"Conflicting nested optimizer state paths: {sorted(conflicts)}"
        )
    for param_group in fqn_sd["param_groups"]:
        for fqn in param_group["params"]:
            for key, value in param_group.items():
                if key != "params":
                    flat[f"param_groups.{fqn}.{key}"] = value
    return flat


def load_flat_optim_state_dict(
    optim: torch.optim.Optimizer, *, flat_state: dict[str, Any]
) -> None:
    """Load a flat, FQN-keyed optimizer state dict into ``optim``.

    Inverse of ``get_flat_optim_state_dict``. The optimizer state must already
    exist (it tells us which state tensors to expect); call ``init_optim_state``
    first. Keys in ``flat_state`` that this optimizer does not own are ignored, so a
    single flat dict covering several optimizers can be passed to each of them.
    """
    optim.load_state_dict(unflatten_optim_state_dict(optim, flat_state=flat_state))


def validate_optimizer_state_tensor(
    value: Any, expected: torch.Tensor, *, key: str
) -> None:
    """Validate destination storage locally, after TP/EP layout conversion.

    Loading may cast dtype or move devices, but must not silently change
    distributed ownership. This phase must not redistribute tensors.
    """
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Optimizer state {key!r} must be a tensor")
    if value.shape != expected.shape:
        raise ValueError(
            f"Optimizer state {key!r} has shape {tuple(value.shape)}, "
            f"expected {tuple(expected.shape)}"
        )
    if value.layout != expected.layout:
        raise ValueError(
            f"Optimizer state {key!r} has tensor layout {value.layout}, "
            f"expected {expected.layout}"
        )
    if isinstance(value, DTensor) != isinstance(expected, DTensor):
        raise ValueError(
            f"Optimizer state {key!r} has incompatible distributed tensor layout"
        )
    if isinstance(value, DTensor) and (
        value.device_mesh != expected.device_mesh
        or value.placements != expected.placements
        or value.to_local().shape != expected.to_local().shape
    ):
        raise ValueError(
            f"Optimizer state {key!r} must match the destination DTensor "
            "mesh, placements and local shape"
        )


def validate_optimizer_step_counter(value: Any, *, key: str) -> None:
    """Validate Adam's local scalar counter without DTensor scalar dispatch."""
    counter = value
    if isinstance(counter, DTensor):
        if any(not placement.is_replicate() for placement in counter.placements):
            raise ValueError(
                f"Optimizer state {key!r} counter DTensor must be fully replicated"
            )
        counter = counter.to_local()
    if isinstance(counter, torch.Tensor):
        if counter.numel() != 1:
            raise ValueError(
                f"Optimizer state {key!r} counter must contain exactly one value"
            )
        counter = counter.item()
    if (
        isinstance(counter, bool)
        or not isinstance(counter, int | float)
        or counter < 0
        or (
            isinstance(counter, float)
            and (not math.isfinite(counter) or not counter.is_integer())
        )
    ):
        raise ValueError(
            f"Optimizer state {key!r} counter must be a finite nonnegative integer"
        )


def optim_state_dict_to_fqn_keys(optim_sd: dict[str, Any]) -> dict[str, Any]:
    """Re-key an optimizer state dict from integer param ids to FQNs.

    Relies on ``param_names`` in each param group, which PyTorch populates when
    the optimizer is built with ``(name, param)`` tuples. ``param_names`` is
    dropped from the result.
    """
    id_to_fqn: dict[int, str] = {}
    new_param_groups: list[dict[str, Any]] = []
    seen_fqns: set[str] = set()
    for param_group in optim_sd["param_groups"]:
        if "param_names" not in param_group:
            raise ValueError(
                "Optimizer must be built with (name, param) tuples so that "
                "param_names is available for FQN-keyed state dicts."
            )
        fqns = param_group["param_names"]
        for param_id, fqn in zip(param_group["params"], fqns, strict=True):
            if param_id in id_to_fqn:
                raise ValueError(f"Duplicate optimizer parameter id: {param_id}")
            if fqn in seen_fqns:
                raise ValueError(f"Duplicate optimizer parameter FQN: {fqn!r}")
            id_to_fqn[param_id] = fqn
            seen_fqns.add(fqn)
        new_group = {k: v for k, v in param_group.items() if k != "param_names"}
        new_group["params"] = list(fqns)
        new_param_groups.append(new_group)

    new_state: dict[str, Any] = {}
    for param_id, state in optim_sd["state"].items():
        if param_id not in id_to_fqn:
            raise KeyError(
                f"Optimizer state has param id {param_id} that is not in any "
                f"param group. Known ids: {sorted(id_to_fqn)}"
            )
        new_state[id_to_fqn[param_id]] = state

    return {"state": new_state, "param_groups": new_param_groups}


def unflatten_optim_state_dict(
    optim: torch.optim.Optimizer, *, flat_state: dict[str, Any]
) -> dict[str, Any]:
    """Rebuild an integer-keyed optimizer state dict from a flat, FQN-keyed one.

    Walks the live optimizer's param groups to recover the FQN order and the set
    of state tensors to expect, then pulls matching values out of ``flat_state``.
    """
    state: dict[int, dict[str, Any]] = {}
    param_groups: list[dict[str, Any]] = []
    # A direct value and descendants cannot both describe the same state node.
    # Index conflicts once rather than scanning the full checkpoint per tensor.
    conflicting_paths = _conflicting_state_paths(flat_state)
    param_id = 0
    seen_fqns: set[str] = set()
    for param_group in optim.param_groups:
        fqns = param_group["param_names"]
        params = param_group["params"]
        restored_group = _restore_param_group(param_group, flat_state=flat_state)
        ids: list[int] = []
        for fqn, param in zip(fqns, params, strict=True):
            if fqn in seen_fqns:
                raise ValueError(f"Duplicate optimizer parameter FQN: {fqn!r}")
            seen_fqns.add(fqn)
            ids.append(param_id)
            if param in optim.state:
                param_state: dict[str, Any] = {}
                expected_state = dict(optim.state[param])
                if isinstance(optim, torch.optim.Adam | torch.optim.AdamW):
                    if restored_group["amsgrad"]:
                        expected_state["max_exp_avg_sq"] = expected_state["exp_avg_sq"]
                    else:
                        expected_state.pop("max_exp_avg_sq", None)
                for state_name in expected_state:
                    flat_key = f"state.{fqn}.{state_name}"
                    if flat_key in flat_state:
                        if flat_key in conflicting_paths:
                            raise ValueError(
                                "Conflicting nested optimizer state paths under "
                                f"{flat_key!r}"
                            )
                        value = flat_state[flat_key]
                        expected = expected_state[state_name]
                        if (
                            isinstance(optim, torch.optim.Adam | torch.optim.AdamW)
                            and state_name == "step"
                        ):
                            validate_optimizer_step_counter(value, key=flat_key)
                        # Adam moments must match the destination parameter layout.
                        # Optimizer.load_state_dict casts tensors but does not check
                        # shapes, so malformed moments otherwise fail on the first
                        # update after other optimizers have already been restored.
                        if isinstance(expected, torch.Tensor) and state_name != "step":
                            validate_optimizer_state_tensor(
                                value, expected, key=flat_key
                            )
                        param_state[state_name] = value
                    else:
                        # Only dictionary-valued state may use nested keys.
                        # A malformed subtree must not replace a required Adam
                        # tensor/counter and defer failure until the next update.
                        nested = (
                            _restore_nested_state(flat_state, prefix=flat_key)
                            if isinstance(expected_state[state_name], dict)
                            else None
                        )
                        if not nested:
                            raise KeyError(
                                f"Optimizer state key {flat_key!r} not found "
                                "in checkpoint."
                            )
                        param_state[state_name] = nested
                if param_state:
                    state[param_id] = param_state
            param_id += 1

        restored_group["params"] = ids
        param_groups.append(restored_group)

    return {"state": state, "param_groups": param_groups}


def _restore_param_group(
    param_group: dict[str, Any], *, flat_state: dict[str, Any]
) -> dict[str, Any]:
    """Restore compatible saved options for one current parameter group."""
    fqns = param_group["param_names"]
    if not fqns:
        # No FQN exists to key this group's options in the flat checkpoint.
        # Preserve the live options so even an empty group remains usable
        # by optimizer.step() and learning-rate schedulers after loading.
        return {**param_group, "params": [], "param_names": list(fqns)}
    new_group: dict[str, Any] = {"params": [], "param_names": list(fqns)}
    for key in param_group:
        if key in ("params", "param_names"):
            continue
        flat_key = f"param_groups.{fqns[0]}.{key}"
        if flat_key not in flat_state:
            raise KeyError(
                f"Optimizer param group key {key!r} not found in checkpoint "
                f"(looked up via param {fqns[0]!r})."
            )
        new_group[key] = flat_state[flat_key]
        for fqn in fqns[1:]:
            other_key = f"param_groups.{fqn}.{key}"
            if other_key not in flat_state:
                raise KeyError(
                    f"Optimizer param group key {other_key!r} not found in checkpoint."
                )
            expected, actual = new_group[key], flat_state[other_key]
            matches = (
                torch.equal(expected, actual)
                if isinstance(expected, torch.Tensor)
                and isinstance(actual, torch.Tensor)
                else expected == actual
            )
            if not matches:
                raise ValueError(
                    f"Cannot merge optimizer parameters {fqns[0]!r} and "
                    f"{fqn!r}: checkpoint group option {key!r} differs"
                )
    return new_group


def _conflicting_state_paths(flat_state: dict[str, Any]) -> set[str]:
    """Find values also used as dictionary parents in a dotted-key layout."""
    conflicts: set[str] = set()
    for key in flat_state:
        parent = key.rpartition(".")[0]
        while parent:
            if parent in flat_state:
                conflicts.add(parent)
            parent = parent.rpartition(".")[0]
    return conflicts


def _flatten_state(state: dict[str, Any], *, prefix: str, out: dict[str, Any]) -> None:
    """Flatten a (possibly nested) per-param state dict into dotted keys."""
    for key, value in state.items():
        flat_key = f"{prefix}.{key}"
        if isinstance(value, dict) and value:
            _flatten_state(value, prefix=flat_key, out=out)
        else:
            if flat_key in out:
                raise ValueError(
                    f"Optimizer state keys collide when flattened: {flat_key!r}"
                )
            out[flat_key] = value


def _restore_nested_state(flat_sd: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    """Rebuild the nested dict stored under ``prefix`` in a flat state dict."""
    result: dict[str, Any] = {}
    prefix_dot = prefix + "."
    for key, value in flat_sd.items():
        if not key.startswith(prefix_dot):
            continue
        current = result
        parts = key[len(prefix_dot) :].split(".")
        path = prefix
        for part in parts[:-1]:
            path = f"{path}.{part}"
            if path in flat_sd:
                raise ValueError(
                    f"Conflicting nested optimizer state paths under {prefix!r}: "
                    f"{key!r}"
                )
            current = current.setdefault(part, {})
        if parts[-1] in current:
            raise ValueError(
                f"Conflicting nested optimizer state paths under {prefix!r}: {key!r}"
            )
        current[parts[-1]] = value
    return result
