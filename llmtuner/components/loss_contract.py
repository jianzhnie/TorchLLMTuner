"""Loss execution interfaces shared by dense and pipeline assembly."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

LossSum = Callable[..., torch.Tensor]
ChunkedLossBackward = Callable[..., torch.Tensor]


@dataclass(frozen=True, slots=True)
class LossFunctions:
    """Token-summed loss and optional head/chunk backward implementation.

    token_sum receives aligned logits/labels and vocabulary reduction keywords,
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
