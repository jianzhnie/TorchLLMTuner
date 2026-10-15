"""Initialize optimizer state and serialize it by parameter name.

Adapted from torchtitan ``components/optimizer/utils.py``. Native optimizer
state uses integer parameter indices; these helpers convert it to a flat,
fully qualified name (FQN) layout for distributed checkpointing.
``init_optim_state`` materializes lazy state while preserving gradients,
learning rates, existing state, and the first real Adam step.
"""

import math
from typing import Any

import torch

__all__ = [
    "init_optim_state",
    "get_flat_optim_state_dict",
    "load_flat_optim_state_dict",
]


def init_optim_state(optim: torch.optim.Optimizer) -> None:
    """Materialize per-parameter optimizer state without changing anything.

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

    No-op when every parameter that requires a gradient already has state.

    An ``OptimizersContainer`` is itself an ``Optimizer``, so wrapping this in a
    per-inner-optimizer loop -- which the container does -- is the whole of the
    multi-optimizer case: the reset branch below is type-gated on Adam, and
    would be skipped silently if a container were handed in whole.
    """
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
                state["step"].zero_()
                state["exp_avg"].zero_()
                state["exp_avg_sq"].zero_()
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
    param_id = 0
    for param_group in optim.param_groups:
        fqns = param_group["param_names"]
        params = param_group["params"]
        restored_group = _restore_param_group(
            param_group, param_ids=[], flat_state=flat_state
        )
        ids: list[int] = []
        for fqn, param in zip(fqns, params, strict=True):
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
                        value = flat_state[flat_key]
                        expected = expected_state[state_name]
                        if (
                            isinstance(optim, torch.optim.Adam | torch.optim.AdamW)
                            and state_name == "step"
                        ):
                            counter = value
                            if isinstance(counter, torch.Tensor):
                                if counter.numel() != 1:
                                    raise ValueError(
                                        f"Optimizer state {flat_key!r} counter must "
                                        "contain exactly one value"
                                    )
                                counter = counter.item()
                            if (
                                isinstance(counter, bool)
                                or not isinstance(counter, int | float)
                                or counter < 0
                                or (
                                    isinstance(counter, float)
                                    and (
                                        not math.isfinite(counter)
                                        or not counter.is_integer()
                                    )
                                )
                            ):
                                raise ValueError(
                                    f"Optimizer state {flat_key!r} counter must be "
                                    "a finite nonnegative integer"
                                )
                        # Adam moments must match the destination parameter layout.
                        # Optimizer.load_state_dict casts tensors but does not check
                        # shapes, so malformed moments otherwise fail on the first
                        # update after other optimizers have already been restored.
                        if isinstance(expected, torch.Tensor) and state_name != "step":
                            if not isinstance(value, torch.Tensor):
                                raise ValueError(
                                    f"Optimizer state {flat_key!r} must be a tensor"
                                )
                            if value.shape != expected.shape:
                                raise ValueError(
                                    f"Optimizer state {flat_key!r} has shape "
                                    f"{tuple(value.shape)}, expected "
                                    f"{tuple(expected.shape)}"
                                )
                        param_state[state_name] = value
                    else:
                        # Only dictionary-valued state may use nested keys.
                        # A malformed subtree must not replace a required Adam
                        # tensor/counter and defer failure until the next update.
                        if not isinstance(expected_state[state_name], dict):
                            raise KeyError(
                                f"Optimizer state key {flat_key!r} not found "
                                "in checkpoint."
                            )
                        nested = _restore_nested_state(flat_state, prefix=flat_key)
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
    param_group: dict[str, Any], *, param_ids: list[int], flat_state: dict[str, Any]
) -> dict[str, Any]:
    """Restore compatible saved options for one current parameter group."""
    fqns = param_group["param_names"]
    if not fqns:
        # No FQN exists to key this group's options in the flat checkpoint.
        # Preserve the live options so even an empty group remains usable
        # by optimizer.step() and learning-rate schedulers after loading.
        return {**param_group, "params": param_ids, "param_names": list(fqns)}
    new_group: dict[str, Any] = {"params": param_ids, "param_names": list(fqns)}
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
