"""A fixed-dtype forward matmul for the decoder's output projection.

Semantics vendored from torchtitan's ``CastLinear``: the vocabulary logits are
scored in a configured dtype regardless of the dtype the rest of the model
runs in, because bf16 logits lose the precision the loss (and RL logprob/KL
math downstream of it) needs. Explicit models choose their output projection
at construction through ``build_lm_head``. The replacement builder remains
available for callers that already own a projection and must keep its parameters.

The class is an ``nn.Linear`` subclass rather than a wrapper around one on
purpose: a wrapper would register the original projection as a child and
prefix every state-dict key with the wrapper's name, breaking checkpoint
compatibility with an unwrapped run. Keeping ``weight``/``bias`` as direct
parameters of the replacement module is what leaves the FQNs -- and, because
the swap rebinds the same ``Parameter`` objects, an embedding/lm_head weight
tie -- exactly as they were.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CastLinear", "build_cast_linear", "build_lm_head"]


def build_lm_head(
    hidden_size: int, vocab_size: int, *, compute_dtype: str | None = None
) -> nn.Linear:
    """Build the declared output projection with one normal initialization.

    The default is an ordinary Linear. CastLinear uses the same constructor
    and parameter layout; model post_init and weight tying still own the
    initialization lifecycle. Reject unknown policies before allocating storage.
    """
    if compute_dtype is None:
        return nn.Linear(hidden_size, vocab_size, bias=False)
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if not isinstance(compute_dtype, str) or compute_dtype not in dtypes:
        raise ValueError(
            "lm_head compute_dtype must be float32, float16, bfloat16 or None, "
            f"got {compute_dtype!r}"
        )
    return CastLinear(
        hidden_size, vocab_size, bias=False, compute_dtype=dtypes[compute_dtype]
    )


class CastLinear(nn.Linear):
    """An ``nn.Linear`` whose forward matmul runs in ``compute_dtype``.

    Inputs, weight, and bias are cast to ``compute_dtype`` before ``F.linear``
    and the output is returned in that dtype. The stored parameters keep their
    original dtype -- autograd casts the incoming gradients back through the
    forward casts, so the weight gradient lands in the parameter's own dtype
    and the optimizer never sees an upcast copy. The cast cannot be cached
    across steps because the optimizer updates the weight every step.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        compute_dtype: torch.dtype,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        if compute_dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            raise ValueError(
                "compute_dtype must be float16, bfloat16, float32 or float64, "
                f"got {compute_dtype!r}"
            )
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.compute_dtype = compute_dtype

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # An outer autocast context must not override the configured matmul dtype.
        with torch.autocast(device_type=input.device.type, enabled=False):
            bias = None if self.bias is None else self.bias.to(self.compute_dtype)
            return F.linear(
                input.to(self.compute_dtype), self.weight.to(self.compute_dtype), bias
            )


def build_cast_linear(linear: nn.Module, *, compute_dtype: torch.dtype) -> CastLinear:
    """Build a ``CastLinear`` that reuses ``linear``'s parameter objects.

    Rebinding rather than copying is the point: a tied lm_head shares one
    ``Parameter`` with the embedding, and only reusing the object keeps the tie
    (and the optimizer's parameter identity) intact. The geometry is read off
    the module being replaced, so the swap cannot silently change shapes.
    """
    if not isinstance(linear, nn.Linear):
        raise TypeError(
            f"lm_head cast expects an nn.Linear to replace, got "
            f"{type(linear).__name__}. A model whose output projection is not "
            "a plain Linear needs its own cast path."
        )
    cast = CastLinear(
        linear.in_features,
        linear.out_features,
        compute_dtype=compute_dtype,
        bias=linear.bias is not None,
        # Parameters are rebound below; meta avoids allocating and randomly
        # initializing a second vocabulary-sized projection.
        device="meta",
        dtype=linear.weight.dtype,
    )
    cast.weight = linear.weight
    cast.bias = linear.bias
    cast.train(linear.training)
    return cast
