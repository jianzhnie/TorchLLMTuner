"""Model-owned execution capabilities, independent of parallel backends."""

from dataclasses import dataclass
from math import prod
from typing import Any

import torch


def validate_training_output(
    output: torch.Tensor,
    labels: torch.Tensor,
    *,
    output_width: int | None = None,
) -> None:
    """Check local token/target geometry without touching tensor data.

    Flat token rows and batch/sequence rows may represent the same tokens.
    When both retain multiple token axes, those axes must match in order.
    Model-owned hidden/projection semantics remain outside this boundary.
    """
    if not isinstance(output, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("Training output and labels must be tensors")
    if output.ndim < 2 or output.shape[-1] <= 0:
        raise ValueError("Training output must have token rows and a nonempty width")
    if not output.is_floating_point():
        raise TypeError("Training output must be floating point")
    if labels.ndim not in (1, 2) or labels.dtype not in (torch.int32, torch.int64):
        raise TypeError("Training labels must contain one or two integer token axes")
    if output.device != labels.device:
        raise ValueError("Training output and labels must be on the same device")
    if prod(output.shape[:-1]) != labels.numel():
        raise ValueError("Training output must have one row per target token")
    if output.ndim > 2 and labels.ndim > 1:
        if tuple(output.shape[:-1]) != tuple(labels.shape):
            raise ValueError("Training output and labels disagree on token axis order")
    if output_width is not None:
        if type(output_width) is not int or output_width <= 0:
            raise ValueError("Declared training output width must be positive")
        if output.shape[-1] != output_width:
            raise ValueError("Training output disagrees with declared output width")


def validate_prepared_model_inputs(
    prepared: Any,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Validate the token/target boundary returned by model.prepare_inputs.

    Models own padding, document boundaries, multimodal fields and sequence
    sharding. This boundary only requires one aligned target per local token;
    kwargs may describe a gathered sequence and need not share token geometry.
    Flat and batched token representations may interoperate, but when both
    retain batch/sequence axes those axes must agree before model execution.
    """
    if not isinstance(prepared, tuple) or len(prepared) != 3:
        raise TypeError("prepare_inputs must return (tokens, labels, forward_kwargs)")
    tokens, labels, kwargs = prepared
    for name, value in (("tokens", tokens), ("labels", labels)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Prepared {name} must be a tensor")
        if value.ndim not in (1, 2):
            raise ValueError(f"Prepared {name} must have one or two dimensions")
        if value.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"Prepared {name} must contain integer indices")
    if tokens.numel() != labels.numel():
        raise ValueError("Prepared tokens and labels must have equal token counts")
    if tokens.ndim == labels.ndim == 2 and tokens.shape != labels.shape:
        raise ValueError("Prepared tokens and labels disagree on token axis order")
    if tokens.device != labels.device:
        raise ValueError("Prepared tokens and labels must be on the same device")
    if not isinstance(kwargs, dict) or any(not isinstance(key, str) for key in kwargs):
        raise TypeError("Prepared forward_kwargs must be a dictionary with string keys")
    return tokens, labels, kwargs


@dataclass(frozen=True, slots=True)
class ModelExecutionCapabilities:
    """Supported output requests; hidden output uses skip_lm_head=True."""

    logits: bool = True
    hidden_states: bool = False
    hidden_size: int | None = None

    def __post_init__(self) -> None:
        if type(self.logits) is not bool or type(self.hidden_states) is not bool:
            raise TypeError("Model execution capabilities must be boolean")
        if self.hidden_size is not None:
            if type(self.hidden_size) is not int or self.hidden_size <= 0:
                raise ValueError("Declared hidden size must be a positive integer")
            if not self.hidden_states:
                raise ValueError("Declared hidden size requires hidden output support")

    def output_tensor(
        self, output: Any, *, hidden_states: bool = False
    ) -> torch.Tensor:
        """Read the tensor corresponding to a supported model output request."""
        if not (self.hidden_states if hidden_states else self.logits):
            raise ValueError("Model does not support the requested output")
        field = "last_hidden_state" if hidden_states else "logits"
        if isinstance(output, torch.Tensor):
            return output
        value = getattr(output, field, None)
        if isinstance(value, torch.Tensor):
            return value
        raise TypeError(
            f"Model output must be a Tensor or expose tensor .{field}; got "
            f"{type(output).__name__}"
        )
