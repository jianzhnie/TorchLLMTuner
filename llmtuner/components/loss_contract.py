"""Loss service contracts and scalar validation for training execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypedDict

import torch

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


class VocabularyLossKwargs(TypedDict, total=False):
    """Loss keywords for an explicitly vocabulary-sharded output layout."""

    tp_group: ProcessGroup
    global_vocab_size: int


class LossSum(Protocol):
    """Return token-summed loss; the engine owns normalization and backward."""

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        /,
        *,
        tp_group: ProcessGroup | None = None,
        global_vocab_size: int | None = None,
    ) -> torch.Tensor: ...


class ChunkedLossBackward(Protocol):
    """Run scaled chunk backward and return detached, unnormalized loss."""

    def __call__(
        self,
        projection: torch.nn.Module,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        /,
        *,
        num_chunks: int,
        grad_scale: torch.Tensor | float,
        tp_group: ProcessGroup | None = None,
        global_vocab_size: int | None = None,
    ) -> torch.Tensor: ...


def validate_normalization_tokens(
    count: torch.Tensor, *, device: torch.device | None = None
) -> None:
    """Require detached real scalar counts before loss normalization."""
    if not isinstance(count, torch.Tensor) or count.ndim != 0:
        raise TypeError("Normalization tokens must be a scalar tensor")
    if count.requires_grad or count.is_complex() or count.dtype == torch.bool:
        raise TypeError("Normalization tokens must be detached real counts")
    if device is not None and count.device != device:
        raise ValueError("Normalization tokens must be on the execution device")


def validate_loss_sum(
    loss: torch.Tensor,
    *,
    device: torch.device,
    requires_grad: bool | None = None,
) -> None:
    """Validate a loss service's scalar metadata without reading its value.

    Loss implementations may communicate, so callers validate only after the
    implementation returns. Finite-value policy belongs to update coordination.
    """
    if not isinstance(loss, torch.Tensor):
        raise TypeError("Loss service must return a tensor")
    if loss.ndim != 0 or not loss.is_floating_point():
        raise ValueError("Loss service must return a floating-point scalar")
    if loss.device != device:
        raise ValueError("Loss scalar must be on the execution device")
    if requires_grad is not None and loss.requires_grad != requires_grad:
        requirement = "differentiable" if requires_grad else "detached"
        raise ValueError(f"Loss scalar must be {requirement}")


@dataclass(frozen=True, slots=True)
class LossFunctions:
    """Token-summed loss and optional head/chunk backward implementation.

    token_sum receives aligned [tokens, vocabulary] logits, [tokens] labels and
    vocabulary reduction keywords,
    returns an unnormalized scalar, and does not run backward. chunked_backward
    accepts the same arguments as chunked_lm_head_cross_entropy, performs scaled
    backward internally, and returns detached unnormalized loss. Custom losses
    must explicitly provide this function to support chunked execution.
    """

    token_sum: LossSum
    chunked_backward: ChunkedLossBackward | None = None

    def __post_init__(self) -> None:
        if not callable(self.token_sum):
            raise TypeError("Loss token_sum must be callable")
        if self.chunked_backward is not None and not callable(self.chunked_backward):
            raise TypeError("Loss chunked_backward must be callable")
