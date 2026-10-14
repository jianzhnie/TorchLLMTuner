"""Maintain EMA weights with the optimizer container's FQN checkpoint format.

Adapted from torchtitan ``components/optimizer/ema.py``. EMA is a sibling of the
training optimizer: the trainer calls ``step(current_step)`` explicitly, and
DCP saves the averages as ``state.{fqn}.ema_params``.

The tracked scope is fixed at construction: trainable parameters and selected
floating-point or complex buffers. Updates validate tensor identity before
changing any average. Dynamic decay derives its firing count from the trainer
step, so checkpoint resume preserves aging without a separate EMA counter.
Loading an empty state reseeds all averages from the restored model.

Configuration is passed as constructor arguments instead of upstream's
Configurable interface. Averages stay on their tensors' devices; the upstream
CUDA side-stream CPU offload path is not implemented.
"""

import re
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ...config.ema_config import validate_ema_schedule
from ...utils.logging import get_logger
from ...utils.module_names import canonical_fqn
from .optimizer import OptimizersContainer

__all__ = ["EMA"]

logger = get_logger(__name__)


class EMAStateOptimizer(Optimizer):
    """Holds ``state[t]["ema_params"]`` per tensor (parameter or buffer) for
    one model part.

    Never step()-ed; reuses ``Optimizer``'s per-tensor state dict plus the
    FQN-flattening DCP machinery in ``optimizer/state_dict.py`` instead of a
    bespoke state-dict format. Also used for buffer EMA (e.g. MoE's
    ``expert_bias_E``), which is why tensors need not be ``nn.Parameter``s.
    """

    def __init__(self, named_tensors: Sequence[tuple[str, torch.Tensor]]) -> None:
        tensors = [t for _, t in named_tensors]
        names = [canonical_fqn(name) for name, _ in named_tensors]
        if len(names) != len(set(names)):
            raise ValueError(
                "EMA tensor FQNs collide after wrapper removal: "
                f"{[name for name, _ in named_tensors]}"
            )
        super().__init__([{"params": tensors, "param_names": names}], {})
        for t in tensors:
            self.state[t]["ema_params"] = t.detach().clone()

    def step(self, closure=None) -> None:
        raise RuntimeError(
            "EMAStateOptimizer must not be step()-ed; call "
            "EMA.step(current_step) instead."
        )

    def tracked_averages(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        """Resolve fixed-identity EMA state without inserting missing entries."""
        # State is keyed by tensor identity and the tracked set is fixed at
        # construction, so a tensor that appeared or
        # was replaced since then has no entry. state is a defaultdict, so
        # indexing it here would insert an empty entry and fail later with a
        # bare KeyError("ema_params").
        untracked = [t for t in tensors if "ema_params" not in self.state.get(t, {})]
        if untracked:
            raise RuntimeError(
                f"EMA has no state for {len(untracked)} of {len(tensors)} "
                f"tensors (first: shape {tuple(untracked[0].shape)}, dtype "
                f"{untracked[0].dtype}). EMA tracks the tensors that existed "
                "when it was built, by identity, so unfreezing a parameter, "
                "replacing a parameter or buffer, or rebuilding part of the "
                "model after the EMA is constructed is not supported. Build "
                "the EMA after the model is final."
            )
        live_ids = {id(tensor) for tensor in tensors}
        missing = sum(
            id(tensor) not in live_ids
            for group in self.param_groups
            for tensor in group["params"]
        )
        if missing:
            raise RuntimeError(
                f"EMA tracked scope lost {missing} tensors. Removing or freezing "
                "a tracked parameter, removing a tracked buffer, or renaming it "
                "outside buffer_patterns after EMA construction is not supported. "
                "Build the EMA after the model is final."
            )
        return [self.state[t]["ema_params"] for t in tensors]


class EMA(OptimizersContainer):
    """Pseudo-optimizer maintaining an online EMA of model weights.

    Subclasses ``OptimizersContainer`` to reuse its FQN-flattened,
    resharding-safe ``state_dict()``/``load_state_dict()`` while overriding
    ``__init__``/``step()``/``zero_grad()`` -- this is never a real training
    optimizer. Never merged into ``Trainer.optimizer`` or the LR scheduler
    container -- it is a sibling object, only built when the run configures
    one (``training.ema``), and stepped explicitly from
    ``train_step()``.

    Args:
        model_parts: the model chunks to track, one per pipeline stage.
        decay: fixed decay per firing, or None (default) to compute it
            dynamically from ``half_life_fraction``.
        half_life_fraction: used when ``decay`` is None:
            ``decay = 2 ** (-1 / (half_life_fraction * num_updates))``.
        start_step: last trainer step before EMA tracking begins, so the first
            update fires at ``start_step + update_every_n_steps``.
        step_bias: offset added to the firing count when computing
            ``num_updates``, for renumbering a new training phase (e.g. a
            restart that resets the trainer step) without resetting EMA aging.
            Measured in EMA firings, not raw steps. A normal resume needs no
            bias -- the firing count already continues correctly on its own.
        update_every_n_steps: only fire the EMA update every N optimizer steps.
        buffer_patterns: regex patterns (``re.search``, matched against buffer
            FQNs from ``model.named_buffers()``) selecting which buffers also
            get an EMA tracked alongside trainable parameters. Empty (default):
            no buffers tracked.
    """

    def __init__(
        self,
        *,
        model_parts: list[nn.Module],
        decay: float | None = None,
        half_life_fraction: float = 0.05,
        start_step: int = 0,
        step_bias: int = 0,
        update_every_n_steps: int = 1,
        buffer_patterns: Sequence[str] = (),
    ) -> None:
        self.decay = decay
        self.half_life_fraction = half_life_fraction
        self.start_step = start_step
        self.step_bias = step_bias
        self.update_every_n_steps = update_every_n_steps
        validate_ema_schedule(self)
        self._buffer_patterns = [re.compile(pattern) for pattern in buffer_patterns]
        # The tracked scope is fixed; caller-owned list edits must not detach
        # model parts from the optimizer groups constructed below.
        self.model_parts = list(model_parts)

        buffer_groups = (
            [
                [
                    (name, buffer)
                    for name, buffer in model.named_buffers()
                    if any(pattern.search(name) for pattern in self._buffer_patterns)
                ]
                for model in self.model_parts
            ]
            if self._buffer_patterns
            else []
        )
        for named_buffers in buffer_groups:
            for name, buffer in named_buffers:
                if not (torch.is_floating_point(buffer) or torch.is_complex(buffer)):
                    raise ValueError(
                        f"EMA buffer_patterns matched buffer {name!r} of "
                        f"dtype {buffer.dtype}. Only floating-point and complex "
                        "buffers can be averaged: an integer or boolean "
                        "average has to be rounded back into the buffer's "
                        "dtype, which freezes it once the per-step increment "
                        "falls below one."
                    )

        parameter_groups = [
            [
                (name, parameter)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ]
            for model in self.model_parts
        ]
        all_params = [parameter for group in parameter_groups for _, parameter in group]
        self._validate_params(all_params)
        self._param_optimizers = [
            EMAStateOptimizer(group) for group in parameter_groups
        ]

        self._buffer_optimizers = [EMAStateOptimizer(group) for group in buffer_groups]
        if self._buffer_patterns and not any(buffer_groups):
            logger.warning(
                "EMA buffer_patterns=%s matched no buffers across any "
                "model part -- buffer EMA is configured but silently "
                "tracking nothing. Check the patterns for typos.",
                list(buffer_patterns),
            )

        # OptimizersContainer.state_dict()/load_state_dict() (reused as-is)
        # iterate self.optimizers and merge each one's FQN-keyed flat dict, so
        # folding _buffer_optimizers in here is what gives buffer EMA the same
        # "ema" checkpoint key as parameters.
        self.optimizers: list[EMAStateOptimizer] = (
            self._param_optimizers + self._buffer_optimizers
        )
        low_precision = sorted(
            {
                str(param_state["ema_params"].dtype)
                for ema_opt in self.optimizers
                for param_state in ema_opt.state.values()
                if param_state["ema_params"].dtype
                not in (torch.float32, torch.float64, torch.complex64, torch.complex128)
            }
        )
        if low_precision:
            logger.warning(
                "EMA is tracking %s tensors. Once the average and the live "
                "value are close, the per-firing increment rounds away at that "
                "precision and the EMA stops tracking altogether. Keep the "
                "tracked tensors in float32 (FSDP2's mixed_precision_param "
                "already does).",
                ", ".join(low_precision),
            )

        self._initialize_optimizer_interface(all_params)

    def zero_grad(self, *args, **kwargs) -> None:
        pass  # never called by the training loop; no-op for safety

    # Takes the step rather than an optimizer closure: the firing count has to
    # be derived from it, and this is never merged into Trainer.optimizer.
    @torch.no_grad()
    def step(self, current_step: int) -> None:
        """Call directly with the trainer's global step -- never merged into
        the optimizer container, so there is no closure/zero-arg step() to
        honor."""
        if (
            not isinstance(current_step, int)
            or isinstance(current_step, bool)
            or current_step < 0
        ):
            raise ValueError("EMA current_step must be a nonnegative integer")
        elapsed = current_step - self.start_step
        if elapsed <= 0 or elapsed % self.update_every_n_steps != 0:
            return
        # num_updates is the firing count (1, 2, 3, ...), derived from
        # current_step rather than kept as a counter so that it survives a
        # checkpoint resume: only ema_params are checkpointed, so a stored
        # counter would restart at 0 and the decay would collapse to a full
        # overwrite of the restored EMA. step_bias is added after the division
        # so its value is never truncated by update_every_n_steps.
        num_updates = elapsed // self.update_every_n_steps + self.step_bias
        decay = (
            self.decay
            if self.decay is not None
            else 2.0 ** (-1.0 / (self.half_life_fraction * num_updates))
        )
        for averages, tensors in self._tracked_groups():
            if tensors:
                torch._foreach_lerp_(averages, tensors, 1.0 - decay)

    def _tracked_groups(
        self,
    ) -> list[tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Validate all live tensor identities before an update or cold start."""
        groups: list[tuple[list[torch.Tensor], list[torch.Tensor]]] = []
        for ema_opt, model in zip(
            self._param_optimizers, self.model_parts, strict=True
        ):
            params: list[torch.Tensor] = [
                p for p in model.parameters() if p.requires_grad
            ]
            groups.append((ema_opt.tracked_averages(params), params))
        for ema_opt, model in zip(
            # Empty when buffer_patterns is unset: shorter than model_parts.
            self._buffer_optimizers,
            self.model_parts,
            strict=False,
        ):
            buffers: list[torch.Tensor] = [
                b
                for name, b in model.named_buffers()
                if any(p.search(name) for p in self._buffer_patterns)
            ]
            groups.append((ema_opt.tracked_averages(buffers), buffers))
        return groups

    # -- checkpointing ----------------------------------------------------------

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict:
            super().load_state_dict(state_dict)
            return
        # The checkpoint had no EMA data (excluded via exclude_from_loading,
        # or predates this feature) -- cold-start from the just-loaded model
        # weights (and buffers, if buffer_patterns is set).
        # Resolve every group before copying: a stale model identity must not
        # partially overwrite the restored averages or insert empty states.
        for averages, tensors in self._tracked_groups():
            for average, tensor in zip(averages, tensors, strict=True):
                average.copy_(tensor.detach())
        logger.warning(
            "EMA state was not restored; cold-starting it from the loaded "
            "model weights, which discards all EMA history. Expected the "
            "first time EMA is enabled against an older checkpoint. If "
            'checkpoint.exclude_from_loading still lists "ema", remove it, '
            "or every later resume will discard the EMA again."
        )
