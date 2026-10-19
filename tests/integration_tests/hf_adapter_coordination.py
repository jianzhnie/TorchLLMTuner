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

from llmtuner.models.hf.state_dict_adapter import HFTransformerStateDictAdapter


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=10))
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        for indexed, failing_rank in product((False, True), (0, 1)):
            adapter = HFTransformerStateDictAdapter(
                SimpleNamespace(tie_word_embeddings=False), None
            )
            if indexed:
                adapter._hf_keys = {"weight", "other"}
            state = {"model.weight" if rank == 0 else "model.other": torch.ones(2)}
            if rank == failing_rank:
                state = {"model.weight": torch.ones(2), "weight": torch.zeros(2)}
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
            healthy = {"model.weight" if rank == 0 else "model.other": torch.ones(2)}
            result = adapter.to_hf(healthy)
            expected_key = "weight" if rank == 0 else "other"
            assert set(result) == {expected_key}
            assert result[expected_key] is next(iter(healthy.values()))
            dist.barrier()
            if rank == 0:
                print(f"PASS {indexed=} {failing_rank=}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
