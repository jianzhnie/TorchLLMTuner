"""Coordinate injected tied-head clone failures on real CPU/Gloo ranks.

PYTHONPATH=. torchrun --nproc_per_node=2 \
    tests/integration_tests/hf_clone_coordination.py

No actual allocation failure is induced. Actual DTensor clones are separately
checked for zero communication and independent local storage.
"""

from datetime import timedelta
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import (
    DTensor,
    Partial,
    Replicate,
    Shard,
    distribute_tensor,
)
from torch.distributed.tensor.debug import CommDebugMode

from llmtuner.models.hf.state_dict_adapter import HFTransformerStateDictAdapter


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=10))
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        mesh = init_device_mesh("cpu", (2,))
        dense = torch.arange(12).reshape(4, 3).float()
        for placement in (Replicate(), Shard(0), Partial()):
            tensor = (
                DTensor.from_local(dense / 2, mesh, [placement], run_check=False)
                if isinstance(placement, Partial)
                else distribute_tensor(dense, mesh, [placement])
            )
            with CommDebugMode() as communication:
                cloned = tensor.clone()
            assert communication.get_total_counts() == 0
            assert cloned.placements == tensor.placements
            assert cloned.to_local().data_ptr() != tensor.to_local().data_ptr()
            torch.testing.assert_close(cloned.full_tensor(), dense, rtol=0, atol=0)
            if rank == 0:
                print(f"PASS local clone {placement=}", flush=True)
        for disjoint, failing_rank, error_type in product(
            (False, True), (0, 1), (OSError, KeyboardInterrupt, SystemExit)
        ):
            adapter = HFTransformerStateDictAdapter(
                SimpleNamespace(tie_word_embeddings=True), None
            )
            weight = torch.arange(6).reshape(2, 3).float()
            owns_head = not disjoint or rank == failing_rank
            state = (
                {
                    "model.model.embed_tokens.weight": weight,
                    "model.lm_head.weight": weight,
                }
                if owns_head
                else {"model.other": weight}
            )
            failure = error_type("injected clone failure")
            original = torch.Tensor.clone

            def clone(
                value,
                *args,
                failing_rank=failing_rank,
                weight=weight,
                failure=failure,
                original=original,
                **kwargs,
            ):
                if rank == failing_rank and value is weight:
                    raise failure
                return original(value, *args, **kwargs)

            caught = None
            with patch.object(torch.Tensor, "clone", clone):
                try:
                    adapter.to_hf(state)
                except BaseException as error:
                    caught = error
            outcomes = [None, None]
            dist.all_gather_object(outcomes, type(caught).__name__)
            assert all(name != "NoneType" for name in outcomes), outcomes
            if rank == failing_rank:
                assert caught is failure
            else:
                assert isinstance(caught, RuntimeError), caught
                assert str(caught) == "HF export tied weights failed on another rank"
            torch.testing.assert_close(
                weight, torch.arange(6).reshape(2, 3).float(), rtol=0, atol=0
            )
            result = adapter.to_hf(state)
            if owns_head:
                assert result["model.embed_tokens.weight"] is weight
                assert result["lm_head.weight"].data_ptr() != weight.data_ptr()
                torch.testing.assert_close(
                    result["lm_head.weight"], weight, rtol=0, atol=0
                )
            else:
                assert result["other"] is weight
            dist.barrier()
            if rank == 0:
                print(
                    f"PASS {disjoint=} {failing_rank=} {error_type.__name__}",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
