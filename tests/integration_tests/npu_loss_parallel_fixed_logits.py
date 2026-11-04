"""Diagnose loss-parallel backward with identical captured full-vocab logits.

Run with torchrun --nproc-per-node=2. This isolates loss arithmetic; it is not
whole-model acceptance and does not alter alignment tolerances.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.parallel import loss_parallel

from tests.integration_tests.harness import init_distributed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--batches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-assets", type=Path)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--weight-gradients", action="store_true")
    args = parser.parse_args()
    init_distributed()
    try:
        world, rank = dist.get_world_size(), dist.get_rank()
        for backend in (torch.npu.matmul, torch.npu.conv, torch.npu.aclnn):
            backend.allow_hf32 = False
        captured = torch.load(args.trace, map_location="cpu", weights_only=False)
        logits = captured["logits"].flatten(0, -2)
        hidden = (
            captured["final_norm"].flatten(0, -2) if args.weight_gradients else None
        )
        del captured
        batch = torch.load(args.batches, map_location="cpu", weights_only=False)[0]
        labels = batch["kwargs"]["labels"].flatten()
        normalizer = (
            torch.as_tensor(batch["kwargs"]["global_valid_tokens"]).item()
            if args.normalize
            else 1.0
        )
        if not torch.isfinite(torch.tensor(normalizer)) or normalizer <= 0:
            raise ValueError("Loss normalizer must be finite and positive")
        if logits.shape[0] != labels.numel() or logits.shape[1] % world:
            raise ValueError("Captured logits/labels do not fit the loss mesh")
        mesh = dist.device_mesh.init_device_mesh("npu", (world,))
        local = logits.chunk(world, dim=-1)[rank].contiguous().to("npu")
        sharded = DTensor.from_local(local, mesh, [Shard(-1)], run_check=False)
        sharded.requires_grad_()
        with loss_parallel():
            loss = torch.nn.functional.cross_entropy(
                sharded.float(), labels.to("npu"), reduction="sum"
            )
            (loss / normalizer).backward()
        # Compare only this rank's token partition and vocabulary partition.
        actual = sharded.grad.to_local().chunk(world, dim=0)[rank].float()
        native = logits.chunk(world, dim=0)[rank].contiguous().to("npu")
        native.requires_grad_()
        native_labels = labels.chunk(world)[rank].to("npu")
        native_loss = torch.nn.functional.cross_entropy(
            native.float(), native_labels, reduction="sum"
        )
        (native_loss / normalizer).backward()
        expected = native.grad.chunk(world, dim=-1)[rank].float()
        delta = (actual - expected).abs()
        result = {
            "rank": rank,
            "world_size": world,
            "logits_shape": list(logits.shape),
            "max_ae": delta.max().item(),
            "mean_ae": delta.mean().item(),
            "mismatches": int(torch.count_nonzero(delta)),
            "elements": delta.numel(),
            "loss_normalizer": normalizer,
            "scope": "fixed actual logits; isolated loss backward only",
        }
        if not torch.isfinite(delta).all() or not torch.count_nonzero(expected):
            raise ValueError("Loss backward diagnostic must be finite and nonzero")
        if args.hf_assets is not None:
            from safetensors import safe_open

            index = json.loads(
                (args.hf_assets / "model.safetensors.index.json").read_text()
            )["weight_map"]
            with safe_open(
                args.hf_assets / index["lm_head.weight"], framework="pt"
            ) as tensors:
                weight = tensors.get_tensor("lm_head.weight").to("npu")
            partial = sharded.grad.to_local() @ weight.chunk(world, dim=0)[rank]
            dist.all_reduce(partial)
            expected_hidden = native.grad @ weight
            actual_hidden = partial.chunk(world, dim=0)[rank]
            difference = (actual_hidden.float() - expected_hidden.float()).abs()
            result["hidden_gradient"] = {
                "max_ae": difference.max().item(),
                "mean_ae": difference.mean().item(),
                "mismatches": int(torch.count_nonzero(difference)),
                "elements": difference.numel(),
            }
        if hidden is not None:
            if hidden.shape[0] != logits.shape[0]:
                raise ValueError("Captured hidden states must cover every logit token")
            hidden = hidden.to("npu")
            baseline_weight = sharded.grad.to_local().transpose(0, 1) @ hidden
            candidate_weight = native.grad.transpose(0, 1) @ hidden.chunk(world)[rank]
            dist.all_reduce(candidate_weight)
            candidate_weight = candidate_weight.chunk(world)[rank]
            difference = (baseline_weight.float() - candidate_weight.float()).abs()
            result["weight_gradient"] = {
                "max_ae": difference.max().item(),
                "mean_ae": difference.mean().item(),
                "mismatches": int(torch.count_nonzero(difference)),
                "sign_changes": int(
                    torch.count_nonzero(
                        baseline_weight.sign() != candidate_weight.sign()
                    )
                ),
                "elements": difference.numel(),
            }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"rank{rank}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
