"""Explicit CPU Gloo2 diagnostic: real perexpert FSDP, not Pro/NPU acceptance."""

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate

from llmtuner.models.hf.deepseek_v4_expert_adapter import PackedExpertStateLayout
from llmtuner.models.hf.deepseek_v4_initialization_binding import (
    PackedInitializationBinding,
)
from llmtuner.models.hf.deepseek_v4_packed_initialization import PackedScratchReplay
from llmtuner.models.hf.deepseek_v4_streaming_checkpoint import (
    StreamingPackedCheckpoint,
)
from tests.unit_tests.cpu.models.test_deepseek_v4_initialization_binding import (
    setup_binding,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        if dist.get_world_size() != 2:
            raise ValueError("Two ranks required")
        mesh = init_device_mesh("cpu", (2,))
        _, _, runtime, replay, targets, transfers = setup_binding()
        # First initialize the unsharded reference, then replay the same stream
        # into the actual post-FSDP owned runtime below.
        torch.manual_seed(819)
        PackedInitializationBinding(replay, runtime, transfers, targets).run()
        adapter = runtime.model.layers[0].mlp.experts
        reference = copy.deepcopy(adapter)
        full_reference = copy.deepcopy(runtime)
        target_names = {id(p): n for n, p in runtime.named_parameters()}
        for path, module in list(runtime.named_modules()):
            if path.endswith(".mlp.experts"):
                for expert in module.experts:
                    fully_shard(expert, mesh=mesh)
        fully_shard(runtime, mesh=mesh)
        owned_targets = tuple(
            replace(t, tensor=runtime.get_parameter(target_names[id(t.tensor)]))
            for t in targets
        )
        owned_replay = PackedScratchReplay(
            replay.model,
            owned_targets,
            scratch_device=torch.device("cpu"),
            max_scratch_bytes=4096,
            copy_bytes=16,
        )
        owned_binding = PackedInitializationBinding(
            owned_replay, runtime, transfers, owned_targets
        )
        torch.manual_seed(819)
        owned_binding.run()
        for name, value in runtime.named_parameters():
            torch.testing.assert_close(
                value.full_tensor(),
                full_reference.get_parameter(name),
                rtol=0,
                atol=0,
            )
        for name, value in runtime.named_buffers():
            torch.testing.assert_close(
                value, full_reference.get_buffer(name), rtol=0, atol=0
            )
        # Exercise legal shard/unshard ownership via the real consumer model's
        # expert subtree. Parent owns nonexpert state; no parent params used here.
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.001)
        ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001)
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            ref_optimizer.zero_grad(set_to_none=True)
            x = (torch.arange(24).reshape(3, 8).float() / 40 + rank).requires_grad_()
            xr = x.detach().clone().requires_grad_()
            # Distinct rank routes, globally inactive expert 3; all experts
            # nevertheless participate as in original packed zero-grad semantics.
            route = torch.tensor([[rank, (rank + step + 1) % 3]] * 3)
            score = torch.full((3, 2), 0.5, requires_grad=True)
            sr = score.detach().clone().requires_grad_()
            actual, expected = adapter(x, route, score), reference(xr, route, sr)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual.sum().backward()
            expected.sum().backward()
            torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
            torch.testing.assert_close(score.grad, sr.grad, rtol=0, atol=0)
            for (name, p), (_, r) in zip(
                adapter.named_parameters(), reference.named_parameters(), strict=True
            ):
                if p.grad is None or r.grad is None:
                    raise ValueError(f"Missing gradient {name}")
                dist.all_reduce(r.grad)
                r.grad.div_(2)
                torch.testing.assert_close(
                    p.grad.full_tensor(), r.grad, rtol=1e-6, atol=1e-6
                )
            optimizer.step()
            ref_optimizer.step()
            for p, r in zip(adapter.parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(p.full_tensor(), r, rtol=1e-6, atol=1e-6)
                for field in ("step", "exp_avg", "exp_avg_sq"):
                    a, b = optimizer.state[p][field], ref_optimizer.state[r][field]
                    if hasattr(a, "full_tensor"):
                        a = a.full_tensor()
                    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
        layout = PackedExpertStateLayout(
            {
                name: (adapter.num_experts, *getattr(reference.experts[0], name).shape)
                for name in ("gate_up_proj", "down_proj")
            }
        )
        streaming = StreamingPackedCheckpoint(
            adapter, optimizer, (layout,), layout.projections
        )
        first = next(adapter.parameters())
        correct = optimizer.state[first]["exp_avg"]
        wrongs = {
            "replicate_same_global_shape": DTensor.from_local(
                torch.zeros(tuple(first.shape)),
                mesh,
                [Replicate()],
                run_check=False,
                shape=first.shape,
                stride=first.stride(),
            ),
            "wrong_local_shape_same_placement": DTensor.from_local(
                torch.zeros(tuple(first.shape)),
                mesh,
                first.placements,
                run_check=False,
                shape=first.shape,
                stride=first.stride(),
            ),
            "wrong_dtype": correct.to(dtype=torch.float64),
        }
        negative_reports = {}
        for label, wrong in wrongs.items():
            optimizer.state[first]["exp_avg"] = wrong
            try:
                streaming.state_dict()
            except ValueError as error:
                negative_reports[label] = str(error)
            else:
                raise AssertionError(f"Accepted incorrect ownership: {label}")
            finally:
                optimizer.state[first]["exp_avg"] = correct
        # Real replacement after export must fail even when values/layout match.
        streaming.state_dict()
        optimizer.state[first]["exp_avg"] = correct.clone()
        try:
            streaming.finish_load()
        except ValueError as error:
            negative_reports["moment_replaced_after_export"] = str(error)
        else:
            raise AssertionError("Accepted stale exported moment destination")
        finally:
            optimizer.state[first]["exp_avg"] = correct
        streaming.state_dict()
        # Test-only actual DTensor local storage rebinding, same Tensor identity.
        original_local = correct._local_tensor
        original_view = original_local.detach()
        with torch.no_grad():
            original_local.set_(original_local.clone())
        try:
            streaming.finish_load()
        except ValueError as error:
            negative_reports["moment_storage_rebound_after_export"] = str(error)
        else:
            raise AssertionError("Accepted stale moment storage")
        finally:
            with torch.no_grad():
                original_local.set_(original_view)
        exported = streaming.state_dict()
        snapshots = {
            name: value.to_local().detach().clone()
            for name, value in adapter.named_parameters()
        }
        moment_snapshots = {
            name: {
                k: (v.to_local() if hasattr(v, "to_local") else v).clone()
                for k, v in optimizer.state[p].items()
            }
            for name, p in adapter.named_parameters()
        }
        args.output.mkdir(parents=True, exist_ok=True)
        dcp.save(exported, checkpoint_id=args.output / "dcp")
        bad_model = dict(exported["model"])
        bad_model.pop("gate_up_proj")
        dcp.save(
            {"model": bad_model, "optimizer": exported["optimizer"]},
            checkpoint_id=args.output / "missing",
        )
        bad_groups = copy.deepcopy(exported["optimizer"]["param_groups"])
        bad_groups[0]["params"].reverse()
        dcp.save(
            {
                "model": exported["model"],
                "optimizer": {
                    "state": exported["optimizer"]["state"],
                    "param_groups": bad_groups,
                },
            },
            checkpoint_id=args.output / "wrong_groups",
        )
        for label in ("missing", "wrong_groups"):
            try:
                streaming.load(args.output / label)
            except (ValueError, RuntimeError) as error:
                negative_reports[label] = str(error)
            else:
                raise AssertionError(f"Accepted invalid checkpoint: {label}")
            for name, p in adapter.named_parameters():
                torch.testing.assert_close(
                    p.to_local(), snapshots[name], rtol=0, atol=0
                )
                for key, saved in moment_snapshots[name].items():
                    value = optimizer.state[p][key]
                    value = value.to_local() if hasattr(value, "to_local") else value
                    torch.testing.assert_close(value, saved, rtol=0, atol=0)
        with torch.no_grad():
            for p in adapter.parameters():
                p.zero_()
                for state in optimizer.state[p].values():
                    state.zero_()
        streaming.load(args.output / "dcp")
        for name, p in adapter.named_parameters():
            torch.testing.assert_close(p.to_local(), snapshots[name], rtol=0, atol=0)
            for key, saved in moment_snapshots[name].items():
                value = optimizer.state[p][key]
                if hasattr(value, "to_local"):
                    value = value.to_local()
                torch.testing.assert_close(value, saved, rtol=0, atol=0)
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / f"rank{rank}.json").open("x") as stream:
            json.dump(
                {
                    "passed": True,
                    "steps": 3,
                    "canonical_packed_distributed_dcp_roundtrip": True,
                    "negative_reports": negative_reports,
                    "rank": rank,
                    "scope": (
                        "CPU perexpert FSDP; post-shard owned replay; "
                        "no CPUOffload/NPU/Trainer acceptance"
                    ),
                },
                stream,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
