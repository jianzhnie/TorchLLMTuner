"""Coordinate Kimi mapping errors before tensor conversion or HF IO.

PYTHONPATH=. torchrun --nproc_per_node=2 \
    tests/integration_tests/kimi_adapter_coordination.py

Actual adapter and Gloo groups, plain CPU tensors; no Trainer or EP runtime.
"""

from datetime import timedelta
from itertools import product
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from llmtuner.models.hf.state_dict_adapter import HFTransformerStateDictAdapter


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=10))
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        prefix = "model.layers.0.block_sparse_moe."
        for to_hf, failing_rank in product((True, False), (0, 1)):
            adapter = HFTransformerStateDictAdapter(
                SimpleNamespace(model_type="kimi_linear", num_experts=2),
                None,
                kimi_expert_parallel=True,
            )
            grouped = {
                prefix + "routed_experts.inner_experts.w1_EFD": torch.arange(12)
                .reshape(2, 2, 3)
                .float()
            }
            individual = {
                f"{prefix}experts.{i}.w1.weight": value
                for i, value in enumerate(next(iter(grouped.values())).unbind(0))
            }
            healthy = grouped if to_hf else individual
            state = dict(healthy)
            if rank == failing_rank:
                state.pop(next(iter(state)))
                if to_hf:
                    state[next(iter(grouped))] = torch.ones(1, 2, 3)
            operation = adapter.to_hf if to_hf else adapter.from_hf
            caught = None
            try:
                operation(state)
            except BaseException as error:
                caught = error
            outcomes = [None, None]
            dist.all_gather_object(outcomes, type(caught).__name__)
            assert all(name != "NoneType" for name in outcomes), outcomes
            if rank == failing_rank:
                assert isinstance(caught, ValueError), caught
            else:
                assert isinstance(caught, RuntimeError), caught
                assert str(caught) == "Kimi checkpoint mapping failed on another rank"
            result = operation(healthy)
            expected = (
                {k.removeprefix("model."): v for k, v in individual.items()}
                if to_hf
                else {"model." + k: v for k, v in grouped.items()}
            )
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            dist.barrier()
            if rank == 0:
                print(f"PASS {to_hf=} {failing_rank=}", flush=True)
        for fault, failing_rank in product(("shape", "duplicate"), (0, 1)):
            adapter = HFTransformerStateDictAdapter(
                SimpleNamespace(model_type="kimi_linear", num_experts=2),
                None,
                kimi_expert_parallel=True,
            )
            healthy = {
                f"{prefix}experts.{index}.w1.weight": torch.ones(2, 3)
                for index in range(2)
            }
            state = dict(healthy)
            if rank == failing_rank:
                if fault == "shape":
                    state[prefix + "experts.1.w1.weight"] = torch.ones(4, 3)
                else:
                    state[prefix + "experts.00.w1.weight"] = torch.zeros(2, 3)
            caught = None
            try:
                adapter.from_hf(state)
            except BaseException as error:
                caught = error
            outcomes = [None, None]
            dist.all_gather_object(outcomes, type(caught).__name__)
            assert all(name != "NoneType" for name in outcomes), outcomes
            if rank == failing_rank:
                assert isinstance(caught, ValueError), caught
                assert (
                    "shapes disagree" if fault == "shape" else "duplicate expert index"
                ) in str(caught)
            else:
                assert isinstance(caught, RuntimeError), caught
                assert str(caught) == "Kimi checkpoint mapping failed on another rank"
            result = adapter.from_hf(healthy)
            expected = {
                "model." + prefix + "routed_experts.inner_experts.w1_EFD": torch.ones(
                    2, 2, 3
                )
            }
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            dist.barrier()
            if rank == 0:
                print(f"PASS {fault=} {failing_rank=}", flush=True)
        mesh = init_device_mesh("cpu", (2,))
        adapter = HFTransformerStateDictAdapter(
            SimpleNamespace(model_type="kimi_linear", num_experts=2),
            None,
            kimi_expert_parallel=True,
        )
        dense = torch.arange(12).reshape(2, 2, 3).float()
        for placement in (Replicate(), Shard(1)):
            grouped = {
                prefix + "routed_experts.inner_experts.w1_EFD": distribute_tensor(
                    dense, mesh, [placement]
                )
            }
            exported = adapter.to_hf(grouped)
            for index in range(2):
                value = exported[f"layers.0.block_sparse_moe.experts.{index}.w1.weight"]
                assert isinstance(value, DTensor)
                torch.testing.assert_close(
                    value.full_tensor(), dense[index], rtol=0, atol=0
                )
            restored = adapter.from_hf(exported)
            value = restored[next(iter(grouped))]
            assert isinstance(value, DTensor)
            torch.testing.assert_close(value.full_tensor(), dense, rtol=0, atol=0)
            if rank == 0:
                print(f"PASS DTensor {placement=}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
