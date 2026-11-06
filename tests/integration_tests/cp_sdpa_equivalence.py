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

``--packed-rows`` additionally exercises real HF input preprocessing and CP
sharding for two independent rows with document resets and continuation RoPE
positions. Its expected mask is built independently from document identities.
"""

from __future__ import annotations

import argparse
import math
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

from llmtuner.models.hf.attention_backend import CP_SDPA_BACKEND
from llmtuner.models.hf.factory import build_model_config
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.parallel.context_parallel.sdpa_attention import CPSdpaKernel
from tests.integration_tests.harness import init_distributed

HEAD_DIM = 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--sliding-window", type=int, default=None)
    parser.add_argument("--packed-rows", action="store_true")
    args = parser.parse_args()

    device = init_distributed()
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    tolerance = 1e-12 if device.type == "cpu" else 1e-5
    rank = dist.get_rank()
    world = dist.get_world_size()
    if args.seq <= 0 or args.seq % world:
        raise ValueError("sequence length must be positive and divisible by CP size")
    mesh = init_device_mesh(device.type, (world,), mesh_dim_names=("cp",))

    packed_mask = reference_mask = None
    if args.packed_rows:
        if args.seq < 8 or args.seq % 4:
            raise ValueError("packed rows require seq >= 8 divisible by four")
        config = build_model_config(
            "qwen3",
            seq_len=args.seq,
            arch_overrides={
                "vocab_size": 32,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
            },
        )
        model = HFTransformerModel(config).to(device=device, dtype=dtype)
        model.model.config._attn_implementation = CP_SDPA_BACKEND
        model.model.config.attn_mask_type = "block_causal"
        indices = torch.arange(args.seq, device=device)
        positions = indices.remainder(args.seq // 4)
        # The second batch row starts with continuation positions. It must
        # remain independent even without a document reset at its first token.
        positions[args.seq // 2 : 3 * args.seq // 4] += args.seq
        tokens = indices.remainder(32).reshape(2, -1)
        topology = SimpleNamespace(
            get_optional_mesh=lambda name: mesh if name == "cp" else None
        )
        inputs, _, kwargs = model.preprocess_inputs(
            {"input": tokens, "labels": tokens, "positions": positions},
            parallel_dims=topology,
        )
        local_size = args.seq // world
        torch.testing.assert_close(
            inputs, tokens.flatten()[rank * local_size : (rank + 1) * local_size]
        )
        packed_mask = kwargs["attention_masks"]
        documents = indices // (args.seq // 4)
        reference_mask = (
            (indices[:, None] >= indices[None, :])
            & (documents[:, None] == documents[None, :])
        )[None, None]
        torch.testing.assert_close(packed_mask, reference_mask)

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
    out = kernel(q, k, v, block_mask=packed_mask, sliding_window=args.sliding_window)
    out.sum().backward()

    # A one-token window attends only to self, even when documents span ranks.
    if args.packed_rows and args.seq // 4 > local and args.sliding_window != 1:
        local_mask = reference_mask[..., lo : lo + local, lo : lo + local]
        if args.sliding_window is not None:
            indices = torch.arange(local, device=device)
            local_mask = local_mask & (
                indices[:, None] - indices[None, :] < args.sliding_window
            )
        with torch.no_grad():
            local_only = F.scaled_dot_product_attention(
                q, k, v, attn_mask=local_mask, enable_gqa=gqa
            ).transpose(1, 2)
            cross_rank_effect = (out - local_only).abs().max()
        dist.all_reduce(cross_rank_effect, op=dist.ReduceOp.MAX)
        if not torch.isfinite(cross_rank_effect) or cross_rank_effect <= 10 * tolerance:
            raise AssertionError("packed CP must attend effective remote K/V tokens")
        if rank == 0:
            print(f"local-only forward differs by {cross_rank_effect.item():.3e}")

    attention_mask = reference_mask
    if args.sliding_window is not None:
        indices = torch.arange(args.seq, device=device)
        rows = indices[:, None]
        cols = indices[None, :]
        attention_mask = ((cols <= rows) & (rows - cols < args.sliding_window))[
            None, None
        ]
        if reference_mask is not None:
            attention_mask = attention_mask & reference_mask
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
            is_causal=attention_mask is None,
            enable_gqa=gqa,
        ).transpose(1, 2)[:, lo : lo + local]
    F.scaled_dot_product_attention(
        qr,
        kr,
        vr,
        attn_mask=attention_mask,
        is_causal=attention_mask is None,
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
            f"packed_rows={args.packed_rows} "
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
