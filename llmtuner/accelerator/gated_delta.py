# BSD 3-Clause License
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# (c) Meta Platforms, Inc. and affiliates.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,this list
# of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may
# be used to endorse or promote products derived from this software without specific
# prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
# SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

"""Opt-in native NPU BF16 GDN; no CPU execution or TorchTitan dependency.

Port of torchtitan-npu/torchtitan_npu/ops/triton/gdn/gated_delta.py.
Upstream commit: 4770484abd0f71b8ca98a8ad0af7af5e5aef4576 (clean tree).
Upstream source SHA256: 736c5811b9a3fa798f4b29761cbaee6d1330241fc6386d3627dc3218d8efe097
The source is self-contained: its Triton helpers are all defined in this file.
Default kernel arithmetic, FP32 normalization/BF16 roundtrip and handwritten
backward retain the upstream algorithm. Fake/first-order autograd contracts are
extended with a default-false deterministic_reduction flag. This opt-in replaces
dk/db cross-value-block atomic adds with FP32 per-block stores and host sums;
it deliberately changes reduction order and is a validation-stage prototype.
Other local changes: namespace, optional Triton import and strict metadata checks.
Higher-order derivatives are not implemented by the upstream backward operator.
Callers opt in by importing this module; torch_npu must already register NPU
before execution. No device is initialized here. PrivateUse1 dispatch spelling
allows import on CPU-only PyTorch without importing torch_npu.
"""

import math

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError as exc:
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = exc
else:
    _TRITON_IMPORT_ERROR = None

__all__ = ["gated_delta_rule"]

_CHUNK_SIZE, _VALUE_BLOCK_SIZE = 64, 16
_GatedDeltaInputs = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


def _require_triton() -> None:
    if triton is None:
        raise RuntimeError(
            "Native NPU GDN requires the environment's Ascend-compatible Triton"
        ) from _TRITON_IMPORT_ERROR


if triton is not None:

    @triton.jit
    def _update_state(state, key, value, decay, beta):
        projected = decay * state
        residual = value - tl.sum(projected * key[:, None], axis=0)
        return projected + key[:, None] * (beta * residual)[None, :], residual

    @triton.jit
    def _state_gradients(adjoint, key, projected, residual, beta):
        grad_update = tl.sum(adjoint * key[:, None], axis=0)
        grad_residual = beta * grad_update
        grad_key = tl.sum(adjoint * (beta * residual)[None, :], axis=1)
        grad_key -= tl.sum(projected * grad_residual[None, :], axis=1)
        grad_projected = adjoint - key[:, None] * grad_residual[None, :]
        grad_beta = tl.sum(grad_update * residual, axis=0)
        return grad_residual, grad_key, grad_projected, grad_beta

    @triton.jit
    def _kernel_context(tokens, heads, dim, chunk_size, value_block_size):
        blocks, program = dim // value_block_size, tl.program_id(0)
        value_block, head_batch = program % blocks, program // blocks
        head, batch = head_batch % heads, head_batch // heads
        d = tl.arange(0, dim)
        vd = value_block * value_block_size + tl.arange(0, value_block_size)
        matrix = d[:, None] * dim + vd[None, :]
        checkpoint_base = head_batch * (tokens // chunk_size) * dim * dim
        return (
            blocks,
            value_block,
            head_batch,
            head,
            batch,
            d,
            vd,
            matrix,
            checkpoint_base,
        )

    @triton.jit
    def _token_offset(batch, token, head, tokens, heads):
        return (batch * tokens + token) * heads + head, batch * tokens + token

    @triton.jit
    def _load_fp32_pair(first, second, first_offset, second_offset):
        return tl.load(first + first_offset).to(tl.float32), tl.load(
            second + second_offset
        ).to(tl.float32)

    @triton.jit
    def _load_gate(decay, step_size, row):
        return tl.exp(tl.load(decay + row).to(tl.float32)), tl.load(step_size + row).to(
            tl.float32
        )

    @triton.jit
    def _gated_delta_forward_kernel(
        q,
        k,
        v,
        g,
        beta,
        reset,
        output,
        checkpoints,
        tokens,
        heads,
        scale: tl.constexpr,
        dim: tl.constexpr,
        chunk_size: tl.constexpr,
        value_block_size: tl.constexpr,
        store_output: tl.constexpr,
        store_checkpoints: tl.constexpr,
    ):
        (
            _blocks,
            _value_block,
            _head_batch,
            head,
            batch,
            d,
            vd,
            matrix,
            checkpoint_base,
        ) = _kernel_context(tokens, heads, dim, chunk_size, value_block_size)
        state = tl.zeros((dim, value_block_size), tl.float32)
        if store_checkpoints:
            tl.store(checkpoints + checkpoint_base + matrix, state)
        for start in range(0, tokens, chunk_size):
            for offset in range(chunk_size):
                token = start + offset
                row, reset_index = _token_offset(batch, token, head, tokens, heads)
                vector = row * dim
                kt, vt = _load_fp32_pair(k, v, vector + d, vector + vd)
                alpha, bt = _load_gate(g, beta, row)
                state = tl.where(tl.load(reset + reset_index) != 0, 0.0, state)
                state = _update_state(state, kt, vt, alpha, bt)[0]
                if store_output:
                    qt = tl.load(q + vector + d).to(tl.float32)
                    tl.store(
                        output + vector + vd,
                        scale * tl.sum(state * qt[:, None], axis=0),
                    )
            if store_checkpoints:
                checkpoint = start // chunk_size + 1
                address = (
                    checkpoints + checkpoint_base + checkpoint * dim * dim + matrix
                )
                tl.store(address, state, mask=checkpoint < tokens // chunk_size)

    @triton.jit
    def _gated_delta_backward_kernel(
        q,
        k,
        v,
        g,
        beta,
        reset,
        dy,
        checkpoints,
        scratch,
        dqh,
        dkh,
        dv,
        da,
        db,
        tokens,
        heads,
        scale: tl.constexpr,
        dim: tl.constexpr,
        chunk_size: tl.constexpr,
        value_block_size: tl.constexpr,
        deterministic_reduction: tl.constexpr,
    ):
        blocks, value_block, head_batch, head, batch, d, vd, matrix, checkpoint_base = (
            _kernel_context(tokens, heads, dim, chunk_size, value_block_size)
        )
        scratch_base = head_batch * (chunk_size + 1) * dim * dim
        adjoint_state = tl.zeros((dim, value_block_size), tl.float32)
        chunk = tokens // chunk_size - 1
        while chunk >= 0:
            start = chunk * chunk_size
            state = tl.load(
                checkpoints + checkpoint_base + chunk * dim * dim + matrix
            ).to(tl.float32)
            tl.store(scratch + scratch_base + matrix, state)
            for offset in range(chunk_size):
                token = start + offset
                row, reset_index = _token_offset(batch, token, head, tokens, heads)
                vector = row * dim
                kt, vt = _load_fp32_pair(k, v, vector + d, vector + vd)
                alpha, bt = _load_gate(g, beta, row)
                state = tl.where(tl.load(reset + reset_index) != 0, 0.0, state)
                state = _update_state(state, kt, vt, alpha, bt)[0]
                tl.store(
                    scratch + scratch_base + (offset + 1) * dim * dim + matrix, state
                )
            for reverse in range(chunk_size):
                offset = chunk_size - reverse - 1
                token = start + offset
                row, reset_index = _token_offset(batch, token, head, tokens, heads)
                vector = row * dim
                previous, current = _load_fp32_pair(
                    scratch,
                    scratch,
                    scratch_base + offset * dim * dim + matrix,
                    scratch_base + (offset + 1) * dim * dim + matrix,
                )
                qt, dyt = _load_fp32_pair(q, dy, vector + d, vector + vd)
                kt, vt = _load_fp32_pair(k, v, vector + d, vector + vd)
                alpha, bt = _load_gate(g, beta, row)
                reset_state = tl.load(reset + reset_index) != 0
                previous = tl.where(reset_state, 0.0, previous)
                projected = alpha * previous
                residual = vt - tl.sum(projected * kt[:, None], axis=0)
                tl.store(
                    dqh + (row * blocks + value_block) * dim + d,
                    scale * tl.sum(current * dyt[None, :], axis=1),
                )
                adjoint = adjoint_state + scale * qt[:, None] * dyt[None, :]
                grad_residual, grad_k, grad_projected, grad_beta = _state_gradients(
                    adjoint, kt, projected, residual, bt
                )
                if deterministic_reduction:
                    tl.store(db + row * blocks + value_block, grad_beta)
                else:
                    tl.atomic_add(db + row, grad_beta)
                tl.store(dv + vector + vd, grad_residual)
                if deterministic_reduction:
                    tl.store(dkh + (row * blocks + value_block) * dim + d, grad_k)
                else:
                    tl.atomic_add(dkh + vector + d, grad_k)
                tl.store(
                    da + row * blocks + value_block, tl.sum(grad_projected * previous)
                )
                adjoint_state = tl.where(reset_state, 0.0, alpha * grad_projected)
            chunk -= 1


def _l2_normalize(x: Tensor) -> tuple[Tensor, Tensor]:
    value = x.float()
    inverse = torch.rsqrt(torch.sum(value * value, dim=-1, keepdim=True) + 1e-6)
    return (value * inverse).to(x.dtype), inverse


def _run_forward_kernel(inputs: _GatedDeltaInputs, scale: float, *, save: bool):
    q, k, v, g, beta, reset = inputs
    batch, tokens, heads, dim = q.shape
    output = v if save else torch.empty_like(v)
    checkpoints = output
    if save:
        checkpoints = torch.empty(
            (batch, heads, tokens // _CHUNK_SIZE, dim, dim),
            dtype=torch.float32,
            device=q.device,
        )
    _gated_delta_forward_kernel[(batch * heads * (dim // _VALUE_BLOCK_SIZE),)](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        reset=reset,
        output=output,
        checkpoints=checkpoints,
        tokens=tokens,
        heads=heads,
        scale=scale,
        dim=dim,
        chunk_size=_CHUNK_SIZE,
        value_block_size=_VALUE_BLOCK_SIZE,
        store_output=not save,
        store_checkpoints=save,
    )
    return output, checkpoints


def gated_delta_forward(inputs: _GatedDeltaInputs, scale: float):
    q, k, v, g, beta, reset = inputs
    _require_triton()
    qh, _ = _l2_normalize(q)
    kh, _ = _l2_normalize(k)
    return _run_forward_kernel((qh, kh, v, g, beta, reset), scale, save=False)[0]


def gated_delta_backward(
    inputs: _GatedDeltaInputs,
    dy: Tensor,
    scale: float,
    *,
    deterministic_reduction: bool = False,
):
    q, k, v, g, beta, reset = inputs
    _require_triton()
    batch, tokens, heads, dim = q.shape
    blocks = dim // _VALUE_BLOCK_SIZE
    qh, qi = _l2_normalize(q)
    kh, ki = _l2_normalize(k)
    _, checkpoints = _run_forward_kernel((qh, kh, v, g, beta, reset), scale, save=True)
    dqh = torch.empty((*q.shape[:3], blocks, dim), dtype=torch.float32, device=q.device)
    dkh = (
        torch.empty((*k.shape[:3], blocks, dim), dtype=torch.float32, device=k.device)
        if deterministic_reduction
        else torch.zeros_like(k, dtype=torch.float32)
    )
    dv = torch.empty_like(v, dtype=torch.float32)
    da = torch.empty((*g.shape, blocks), dtype=torch.float32, device=g.device)
    db = (
        torch.empty((*beta.shape, blocks), dtype=torch.float32, device=beta.device)
        if deterministic_reduction
        else torch.zeros_like(beta, dtype=torch.float32)
    )
    scratch = torch.empty(
        (batch, heads, _CHUNK_SIZE + 1, dim, dim),
        dtype=torch.float32,
        device=q.device,
    )
    _gated_delta_backward_kernel[(batch * heads * blocks,)](
        q=qh,
        k=kh,
        v=v,
        g=g,
        beta=beta,
        reset=reset,
        dy=dy,
        checkpoints=checkpoints,
        scratch=scratch,
        dqh=dqh,
        dkh=dkh,
        dv=dv,
        da=da,
        db=db,
        tokens=tokens,
        heads=heads,
        scale=scale,
        dim=dim,
        chunk_size=_CHUNK_SIZE,
        value_block_size=_VALUE_BLOCK_SIZE,
        deterministic_reduction=deterministic_reduction,
    )
    dqh = dqh.sum(3)
    if deterministic_reduction:
        dkh = dkh.sum(3)
        db = db.sum(-1)
    qf, kf = q.float(), k.float()
    dq = dqh * qi - qf * torch.sum(qf * dqh, -1, True) * qi * qi * qi
    dk = dkh * ki - kf * torch.sum(kf * dkh, -1, True) * ki * ki * ki
    dg = da.sum(-1) * g.float().exp()
    dv = dv.to(v.dtype)
    return dq.to(q.dtype), dk.to(k.dtype), dv, dg.to(g.dtype), db.to(beta.dtype)


def _validate_inputs(inputs: _GatedDeltaInputs, scale: float) -> None:
    q, k, v, g, beta, reset = inputs
    if q.device.type != "npu":
        raise ValueError(
            "gated_delta_rule requires native NPU tensors; no CPU fallback"
        )
    if any(not x.is_contiguous() for x in inputs):
        raise ValueError("custom-op inputs must be contiguous")
    if q.ndim != 4:
        raise ValueError("q must be a rank-4 tensor")
    if k.shape != q.shape:
        raise ValueError("k must match q in shape")
    if v.shape != q.shape:
        raise ValueError("v must match q in shape")
    if q.shape[-1] != 128 or q.shape[1] <= 0 or q.shape[1] % _CHUNK_SIZE:
        raise ValueError("head dim must be 128 and tokens a positive multiple of 64")
    if q.shape[0] <= 0 or q.shape[2] <= 0 or q.shape[0] * q.shape[2] > 8191:
        raise ValueError("batch and heads must be positive with product <= 8191")
    if q.dtype != torch.bfloat16 or not q.dtype == k.dtype == v.dtype:
        raise TypeError("q, k, and v must share the bfloat16 dtype")
    if any(x.dtype not in (q.dtype, torch.float32) for x in (g, beta)):
        raise TypeError("g and beta must match the input dtype or use float32")
    if g.shape != q.shape[:3] or beta.shape != q.shape[:3]:
        raise ValueError("g and beta must match the batch, token, and head dimensions")
    if any(x.device != q.device for x in (k, v, g, beta)):
        raise ValueError("all inputs must be on one device")
    if (
        reset.shape != q.shape[:2]
        or reset.dtype != torch.bool
        or reset.device != q.device
    ):
        raise ValueError(
            "reset must be a boolean [batch, tokens] tensor on the input device"
        )
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")


@torch.library.custom_op(
    "llmtuner::gated_delta_rule", mutates_args=(), device_types="privateuseone"
)
def gated_delta_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    reset: Tensor,
    *,
    scale: float,
    deterministic_reduction: bool = False,
) -> Tensor:
    inputs = q, k, v, g, beta, reset
    _validate_inputs(inputs, scale)
    return gated_delta_forward(inputs, scale)


@gated_delta_op.register_fake
def gated_delta_op_fake(
    q, k, v, g, beta, reset, *, scale, deterministic_reduction=False
):
    _validate_inputs((q, k, v, g, beta, reset), scale)
    return torch.empty_like(v, device=q.device)


@torch.library.custom_op(
    "llmtuner::gated_delta_rule_backward", mutates_args=(), device_types="privateuseone"
)
def gated_delta_backward_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    reset: Tensor,
    dy: Tensor,
    *,
    scale: float,
    deterministic_reduction: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    inputs = q, k, v, g, beta, reset
    _validate_inputs(inputs, scale)
    if not dy.is_contiguous():
        raise ValueError("dy must be contiguous for the native NPU kernel")
    if dy.shape != v.shape or dy.dtype != v.dtype or dy.device != v.device:
        raise ValueError("dy must match v in shape, dtype, and device")
    return gated_delta_backward(
        inputs, dy, scale, deterministic_reduction=deterministic_reduction
    )


@gated_delta_backward_op.register_fake
def gated_delta_backward_op_fake(
    q, k, v, g, beta, reset, dy, *, scale, deterministic_reduction=False
):
    _validate_inputs((q, k, v, g, beta, reset), scale)
    if not dy.is_contiguous():
        raise ValueError("dy must be contiguous for the native NPU kernel")
    if dy.shape != v.shape or dy.dtype != v.dtype or dy.device != v.device:
        raise ValueError("dy must match v in shape, dtype, and device")
    return tuple(torch.empty_like(x) for x in (q, k, v, g, beta))


def gated_delta_setup_context(ctx, inputs, keyword_only_inputs, output):
    del output
    ctx.save_for_backward(*inputs)
    ctx.scale = keyword_only_inputs["scale"]
    ctx.deterministic_reduction = keyword_only_inputs.get(
        "deterministic_reduction", False
    )


def gated_delta_autograd_backward(ctx, dy):
    q, k, v, g, beta, reset = ctx.saved_tensors
    # Avoid even a no-op contiguous() device guard for already contiguous
    # Fake NPU tensors in CPU-only PyTorch. Real strided cotangents still
    # require materialization before the kernel's contiguous pointer access.
    if not dy.is_contiguous():
        dy = dy.contiguous()
    grads = gated_delta_backward_op(
        q,
        k,
        v,
        g,
        beta,
        reset,
        dy,
        scale=ctx.scale,
        deterministic_reduction=ctx.deterministic_reduction,
    )
    return (*grads, None)


torch.library.register_autograd(
    gated_delta_op,
    gated_delta_autograd_backward,
    setup_context=gated_delta_setup_context,
)


def gated_delta_rule(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    reset: Tensor | None = None,
    scale: float | None = None,
    *,
    deterministic_reduction: bool = False,
) -> Tensor:
    """Native NPU GDN; optional fixed-order dk/db block reduction prototype.

    Default preserves upstream atomics. The opt-in uses FP32 block workspaces
    and sums them before normalization/casting; repeatability and full-model
    equivalence require hardware validation. Reference kernels are unchanged.
    """
    if not isinstance(deterministic_reduction, bool):
        raise TypeError("deterministic_reduction must be bool")
    # Validate metadata before head replication or contiguous materialization.
    if any(x.ndim != 4 for x in (q, k, v)):
        raise ValueError("q, k, v must be rank-4 [batch, tokens, heads, 128]")
    if q.shape != k.shape or q.shape[:2] != v.shape[:2]:
        raise ValueError("q/k shapes and q/k/v batch/token dimensions must match")
    if q.shape[-1] != 128:
        raise ValueError("q/k head dimension must be 128")
    if q.shape[2] <= 0 or v.shape[2] % q.shape[2]:
        raise ValueError("Value heads must be a positive multiple of Q/K heads")
    if any(x.device.type != "npu" for x in (q, k, v, g, beta)):
        raise ValueError(
            "gated_delta_rule requires native NPU tensors; no CPU fallback"
        )
    if any(x.device != v.device for x in (q, k, g, beta)):
        raise ValueError("all inputs must be on one NPU device")
    if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
        raise TypeError("q, k, v must use bfloat16")
    if v.shape[-1] != 128 or v.shape[1] <= 0 or v.shape[1] % _CHUNK_SIZE:
        raise ValueError("head dim must be 128 and tokens a positive multiple of 64")
    if v.shape[0] <= 0 or v.shape[2] <= 0 or v.shape[0] * v.shape[2] > 8191:
        raise ValueError("batch and value heads must be positive with product <= 8191")
    if g.shape != v.shape[:3] or beta.shape != v.shape[:3]:
        raise ValueError("g/beta must match value batch, token, and head dimensions")
    if any(x.dtype not in (torch.bfloat16, torch.float32) for x in (g, beta)):
        raise TypeError("g/beta must use bfloat16 or float32")
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    if reset is not None and (
        reset.shape != v.shape[:2]
        or reset.dtype != torch.bool
        or reset.device != v.device
    ):
        raise ValueError("reset must be boolean [batch, tokens] on the input NPU")
    if q.shape[2] != v.shape[2]:
        repeat = v.shape[2] // q.shape[2]
        q, k = (x.repeat_interleave(repeat, 2) for x in (q, k))
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    reset = (
        torch.zeros(q.shape[:2], dtype=torch.bool, device=q.device)
        if reset is None
        else reset.contiguous()
    )
    # Fake dispatch needs only metadata, even when Triton is absent. The real
    # forward/backward implementations check the dependency before launching.
    return gated_delta_op(
        q,
        k,
        v,
        g,
        beta,
        reset,
        scale=scale,
        deterministic_reduction=deterministic_reduction,
    )
