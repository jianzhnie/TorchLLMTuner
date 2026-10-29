"""CP on sdpa: a sharded-sequence attention must equal the full-sequence run.

Run under torchrun with 2 ranks:

    PYTHONPATH=. torchrun --nproc_per_node=2 \\
        tests/integration_tests/cp_sdpa_equivalence.py

``cp_equivalence.py`` pins the *exchange* (KV all-gather, ulysses round trip);
this pins the kernel that consumes it on a backend without flex: each rank holds
a contiguous query block and the matching key/value block, the kernel gathers
K/V and attends the local queries against the full sequence, and both the output
and the gradients must equal a single-rank attention over the whole sequence.

Three things are checked, and each one failed separately while this was written:

* the forward, against ``is_causal=True`` on the full sequence;
* the query gradient, which needs no cross-rank reduction;
* the key/value gradients, which DO: every rank's queries attend every rank's
  K/V rows, so the gradient of the block a rank owns is the sum over ranks.
  Getting that reduction order wrong (slice-then-reduce) produced sign-flipped
  values, so it is asserted here rather than left to a training run.

The kernel's gather is exercised through the kernel itself; ``--heads`` /
``--kv-heads`` cover the grouped-query case, whose head count differs between
query and key.
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

from llmtuner.parallel.context_parallel.sdpa_attention import CPSdpaKernel
from tests.integration_tests.harness import init_distributed

HEAD_DIM = 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--sliding-window", type=int, default=None)
    args = parser.parse_args()

    device = init_distributed()
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    tolerance = 1e-12 if device.type == "cpu" else 1e-5
    rank = dist.get_rank()
    world = dist.get_world_size()
    if args.seq <= 0 or args.seq % world:
        raise ValueError("sequence length must be positive and divisible by CP size")
    mesh = init_device_mesh(device.type, (world,), mesh_dim_names=("cp",))

    torch.manual_seed(0)
    q_full = torch.randn(1, args.heads, args.seq, HEAD_DIM, dtype=dtype, device=device)
    k_full = torch.randn(
        1, args.kv_heads, args.seq, HEAD_DIM, dtype=dtype, device=device
    )
    v_full = torch.randn(
        1, args.kv_heads, args.seq, HEAD_DIM, dtype=dtype, device=device
    )

    local = args.seq // world
    lo = rank * local
    q = q_full[:, :, lo : lo + local].clone().requires_grad_(True)
    k = k_full[:, :, lo : lo + local].clone().requires_grad_(True)
    v = v_full[:, :, lo : lo + local].clone().requires_grad_(True)

    gqa = args.heads != args.kv_heads
    kernel = CPSdpaKernel(cp_mesh=mesh)
    out = kernel(q, k, v, sliding_window=args.sliding_window)
    out.sum().backward()

    attention_mask = None
    if args.sliding_window is not None:
        indices = torch.arange(args.seq, device=device)
        rows = indices[:, None]
        cols = indices[None, :]
        attention_mask = ((cols <= rows) & (rows - cols < args.sliding_window))[
            None, None
        ]
    qr = q_full.clone().requires_grad_(True)
    kr = k_full.clone().requires_grad_(True)
    vr = v_full.clone().requires_grad_(True)
    with torch.no_grad():
        # The attention output is (batch, seq, heads, dim) after the transpose,
        # so this rank's query rows are dim 1. The gradients below stay
        # HF-shaped (batch, heads, seq, dim) and slice dim 2 instead.
        ref = F.scaled_dot_product_attention(
            qr,
            kr,
            vr,
            attn_mask=attention_mask,
            is_causal=args.sliding_window is None,
            enable_gqa=gqa,
        ).transpose(1, 2)[:, lo : lo + local]
    F.scaled_dot_product_attention(
        qr,
        kr,
        vr,
        attn_mask=attention_mask,
        is_causal=args.sliding_window is None,
        enable_gqa=gqa,
    ).sum().backward()

    with torch.no_grad():
        diffs = {
            "forward": (out - ref).abs().max().item(),
            "q grad": (q.grad - qr.grad[:, :, lo : lo + local]).abs().max().item(),
            "k grad": (k.grad - kr.grad[:, :, lo : lo + local]).abs().max().item(),
            "v grad": (v.grad - vr.grad[:, :, lo : lo + local]).abs().max().item(),
        }
    failed = [
        name
        for name, diff in diffs.items()
        if not math.isfinite(diff) or diff > tolerance
    ]
    bad = torch.tensor([1.0 if failed else 0.0], device=device)
    dist.all_reduce(bad, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(
            f"seq={args.seq} heads={args.heads} kv_heads={args.kv_heads} "
            f"cp={world} sliding_window={args.sliding_window} "
            f"tol={tolerance:g} device={device}"
        )
        for name, diff in diffs.items():
            print(f"  {name:<8} max abs diff = {diff:.3e}")
        print(f"failed ranks = {int(bad)}")
        print("all checks passed" if not failed and not bad else "FAILURES")
    dist.destroy_process_group()
    if float(bad) > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
