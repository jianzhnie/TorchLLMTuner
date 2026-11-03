# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muon update arithmetic and views over independent matrix batches."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class _MatrixBatchView:
    """A strided view into a contiguous compute tensor.

    Strides and offset are measured in elements, relative to the supplied tensor.
    """

    shape: torch.Size
    strides: tuple[int, ...]
    offset: int

    def view_as_matrix_batch(self, compute_tensor: Tensor) -> Tensor:
        """Return a zero-copy matrix-batch view of the compute tensor."""
        if not compute_tensor.is_contiguous():
            raise RuntimeError("matrix views require a contiguous compute tensor")
        end = self.offset
        if self.shape.numel():
            end += 1 + sum(
                (size - 1) * stride
                for size, stride in zip(self.shape, self.strides, strict=True)
            )
        if self.offset < 0 or end > compute_tensor.numel():
            raise RuntimeError(
                "compute tensor shape is inconsistent with its matrix view"
            )
        return compute_tensor.as_strided(
            self.shape,
            self.strides,
            storage_offset=compute_tensor.storage_offset() + self.offset,
        )


def _matrix_batch_views_from_shape(
    compute_shape: torch.Size,
    *,
    matrix_row_sizes: tuple[int, ...],
    logical_row_start: int,
) -> tuple[_MatrixBatchView, ...]:
    """Describe complete matrix batches in a logical parameter's row slice."""
    num_rows, matrix_columns = compute_shape
    matrix_row_stride = sum(matrix_row_sizes)
    position_row_offset = 0
    views = []
    for matrix_rows in matrix_row_sizes:
        first_local_row = (position_row_offset - logical_row_start) % matrix_row_stride
        num_matrices = max(
            0, 1 + (num_rows - first_local_row - matrix_rows) // matrix_row_stride
        )
        if num_matrices:
            views.append(
                _MatrixBatchView(
                    shape=torch.Size((num_matrices, matrix_rows, matrix_columns)),
                    strides=(matrix_row_stride * matrix_columns, matrix_columns, 1),
                    offset=first_local_row * matrix_columns,
                )
            )
        position_row_offset += matrix_rows
    assert sum(view.shape[0] * view.shape[1] for view in views) == num_rows, (
        "compute row boundaries must preserve complete matrices"
    )
    return tuple(sorted(views, key=lambda view: view.offset))


def _adjust_muon_learning_rate(
    lr: float,
    adjust_lr_fn: str | None,
    compute_matrix_shape: torch.Size | tuple[int, ...],
) -> float:
    """Adjust Muon's learning rate for the matrix aspect ratio."""
    rows, columns = compute_matrix_shape[-2:]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        ratio = math.sqrt(max(1.0, rows / columns))
    elif adjust_lr_fn == "match_rms_adamw":
        ratio = 0.2 * math.sqrt(max(rows, columns))
    elif adjust_lr_fn == "spectral_unclamped":
        ratio = math.sqrt(rows / columns)
    else:
        raise ValueError(f"unsupported adjust_lr_fn {adjust_lr_fn!r}")
    return lr * ratio


def _prepare_muon_input(
    gradient: Tensor,
    momentum_buffer: Tensor,
    *,
    momentum: float,
    nesterov: bool,
    out: Tensor,
) -> Tensor:
    """Update momentum and prepare the Tensor passed to Muon computation."""
    momentum_buffer.lerp_(gradient, 1 - momentum)
    if nesterov:
        torch.lerp(gradient, momentum_buffer, momentum, out=out)
    else:
        out.copy_(momentum_buffer)
    return out


def _compute_muon_direction(
    prepared: Tensor,
    *,
    matrix_views: Sequence[_MatrixBatchView],
    lr_reference_shape: torch.Size | tuple[int, ...] | None,
    adjust_lr_fn: str | None,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> Tensor:
    """Compute independent matrix directions with relative shape scaling."""
    reference_ratio = (
        None
        if lr_reference_shape is None
        else _adjust_muon_learning_rate(1.0, adjust_lr_fn, lr_reference_shape)
    )
    for view in matrix_views:
        matrices = view.view_as_matrix_batch(prepared)
        matrices.copy_(
            _zeropower_via_newtonschulz(
                matrices,
                ns_coefficients=ns_coefficients,
                ns_steps=ns_steps,
                eps=eps,
            )
        )
        if reference_ratio is None:
            continue
        ratio = _adjust_muon_learning_rate(1.0, adjust_lr_fn, matrices.shape[-2:])
        # Keep the original update arithmetic when shape factors already match.
        if ratio != reference_ratio:
            matrices.mul_(ratio / reference_ratio)
    return prepared


def _apply_muon_update(
    parameter: Tensor,
    direction: Tensor,
    *,
    lr: float,
    weight_decay: float,
    adjust_lr_fn: str | None,
    compute_matrix_shape: torch.Size | tuple[int, ...],
) -> Tensor:
    """Apply decoupled weight decay and a computed Muon direction."""
    adjusted_lr = _adjust_muon_learning_rate(lr, adjust_lr_fn, compute_matrix_shape)
    parameter.mul_(1 - lr * weight_decay)
    parameter.add_(direction, alpha=-adjusted_lr)
    return parameter


def _zeropower_via_newtonschulz(
    update: Tensor,
    *,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> Tensor:
    """Compute Muon's approximate polar factor without optimizer state."""
    a, b, c = ns_coefficients
    result = update.to(dtype=torch.bfloat16, copy=True)
    transposed = result.shape[-2] > result.shape[-1]
    if transposed:
        result = result.transpose(-2, -1)
    result.div_(result.norm(dim=(-2, -1), keepdim=True).clamp_min(eps))

    if result.ndim == 2:
        for _ in range(ns_steps):
            gram = result @ result.T
            gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
            result = torch.addmm(result, gram_update, result, beta=a)
    else:
        original_shape = result.shape
        matrices = result.reshape(-1, *original_shape[-2:])
        # Batched kernels and independent matrix calls can use different BF16
        # reduction orders.
        for _ in range(ns_steps):
            gram = matrices @ matrices.transpose(-2, -1)
            gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
            matrices = torch.baddbmm(matrices, gram_update, matrices, beta=a)
        result = matrices.reshape(original_shape)

    return result.transpose(-2, -1) if transposed else result
