"""Two CPU/Gloo ranks reject local HF key collisions before global key IO.

PYTHONPATH=. torchrun --nproc_per_node=2 \
    tests/integration_tests/hf_adapter_coordination.py

Real adapter conversion and collectives; ordinary tensors, no model/FSDP runtime.
Healthy disjoint key sets exercise the global PP coverage contract afterwards.
"""

from copy import deepcopy
from datetime import timedelta
from itertools import product
from types import SimpleNamespace

import torch
import torch.distributed as dist

from llmtuner.components.checkpointer.model_state import ModelStateDictAdapter


def main() -> None:
    dist.init_process_group("gloo", timeout=timedelta(seconds=10))
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        for indexed, failing_rank in product((False, True), (0, 1)):
            adapter = ModelStateDictAdapter(
                SimpleNamespace(model_type="qwen3_moe", tie_word_embeddings=False),
                None,
                expert_parallel=True,
            )
            first_key = "model.layers.0.mlp.gate.weight"
            second_key = "model.layers.1.mlp.gate.weight"
            if indexed:
                adapter._hf_keys = {first_key, second_key}
            state = {first_key if rank == 0 else second_key: torch.ones(2)}
            if rank == failing_rank:
                state = {
                    "model.layers.0.mlp.router.gate.weight": torch.ones(2),
                    first_key: torch.zeros(2),
                }
            before = deepcopy(state)
            caught = None
            try:
                adapter.to_hf(state)
            except BaseException as error:
                caught = error
            outcomes = [None, None]
            dist.all_gather_object(outcomes, type(caught).__name__)
            assert all(name != "NoneType" for name in outcomes), outcomes
            if rank == failing_rank:
                assert isinstance(caught, ValueError), caught
                assert "HF checkpoint key mapping collision" in str(caught)
            else:
                assert isinstance(caught, RuntimeError), caught
                assert str(caught) == "HF export key mapping failed on another rank"
            torch.testing.assert_close(state, before, rtol=0, atol=0)
            healthy = {first_key if rank == 0 else second_key: torch.ones(2)}
            result = adapter.to_hf(healthy)
            expected_key = first_key if rank == 0 else second_key
            assert set(result) == {expected_key}
            assert result[expected_key] is next(iter(healthy.values()))
            dist.barrier()
            if rank == 0:
                print(f"PASS {indexed=} {failing_rank=}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
