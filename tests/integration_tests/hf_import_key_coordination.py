"""Two ranks: local HF import and explicit router-key failure coordination.

Run with torchrun --standalone --nproc_per_node=2. Uses NPU/HCCL by default;
LLMTUNER_TEST_DEVICE=cpu selects CPU/Gloo. This tests mapping, not training.
"""

from types import SimpleNamespace

import torch
import torch.distributed as dist

from llmtuner.accelerator.collectives import synchronize_local_errors
from llmtuner.components.checkpointer.model_state import ModelStateDictAdapter
from tests.integration_tests.harness import init_distributed


def main() -> None:
    device = init_distributed()
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        adapter = ModelStateDictAdapter(
            SimpleNamespace(model_type="qwen3_moe"),
            None,
            expert_parallel=True,
        )
        key = "model.layers.0.mlp.gate.weight"
        native_key = "model.layers.0.mlp.router.gate.weight"
        weight = torch.arange(4, device=device).reshape(2, 2) + rank
        for failing_rank in (0, 1):
            # Plain adapter mapping must remain callable on just one rank.
            # A hidden collective would mismatch the peer's barrier.
            if rank == failing_rank:
                local = adapter.from_hf({key: weight})
                assert local[native_key] is weight
            dist.barrier()
            source = {key: weight}
            if rank == failing_rank:
                source[native_key] = weight + 10
            try:
                with synchronize_local_errors("HF import key mapping"):
                    adapter.from_hf(source)
            except ValueError as error:
                assert rank == failing_rank and "key mapping collision" in str(error)
            except RuntimeError as error:
                assert rank != failing_rank
                assert "HF import key mapping failed on another rank" in str(error)
            else:
                raise AssertionError("ambiguous router keys were accepted")
            assert source[key] is weight
            restored = adapter.from_hf({key: weight})
            assert restored[native_key] is weight
            torch.testing.assert_close(restored[native_key], weight, rtol=0, atol=0)
            dist.barrier()
        print(
            f"PASS rank={rank}: HF import collision coordination and retry", flush=True
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
