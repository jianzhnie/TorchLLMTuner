"""Real two-rank DCP layout migration with a rank-local RNG setter failure.

Run with PYTHONPATH=. torchrun --nproc_per_node=2 on CPU/Gloo.
Uses small real Kimi MoE modules, not a full Trainer or device RNG setter.
"""

from contextlib import nullcontext
from itertools import product
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from llmtuner.components.checkpointer.kimi_layout import load_changed_kimi_layout
from llmtuner.components.checkpointer.model_state import ModelCheckpointState
from llmtuner.components.checkpointer.random_state import RNGCheckpointState
from llmtuner.models.hf.state_dict_adapter import HFTransformerStateDictAdapter
from llmtuner.parallel.expert_parallel.model_replacement import swap_hf_moe_blocks
from tests.unit_tests.cpu.parallel.test_kimi_conversion import make_model


def main():
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        assert dist.get_world_size() == 2
        context = TemporaryDirectory() if rank == 0 else nullcontext(None)
        with context as root:
            paths = [root]
            dist.broadcast_object_list(paths, src=0)
            for source_grouped, source_compiled in product((False, True), repeat=2):
                torch.manual_seed(11)
                source = torch.nn.Module()
                source.model = make_model(latent=True, shared=1)
                config = source.model.layers[0].block_sparse_moe.config
                if source_grouped:
                    swap_hf_moe_blocks(source.model)
                torch.manual_seed(100 + rank)
                rng = RNGCheckpointState(torch.device("cpu"))
                saved_model = (
                    torch.compile(source, backend="eager")
                    if source_compiled
                    else source
                )
                saved = dict(saved_model.state_dict())
                saved["rng_state"] = rng
                folder = f"{paths[0]}/{source_grouped}-{source_compiled}"
                dcp.save(saved, checkpoint_id=folder)
                expected_random = torch.rand(5)
                target = torch.nn.Module()
                target.model = make_model(latent=True, shared=1)
                if not source_grouped:
                    swap_hf_moe_blocks(target.model)
                states = {"model": ModelCheckpointState(target), "rng_state": rng}
                adapter = HFTransformerStateDictAdapter(
                    config, None, kimi_expert_parallel=not source_grouped
                )
                original = rng.load_state_dict
                failure = ValueError("injected Kimi RNG setter failure")

                def fail_local(state, original=original, failure=failure):
                    if rank == 0:
                        raise failure
                    original(state)

                with patch.object(rng, "load_state_dict", fail_local):
                    try:
                        load_changed_kimi_layout(folder, states=states, adapter=adapter)
                    except (ValueError, RuntimeError) as error:
                        if rank == 0:
                            assert error is failure
                        else:
                            assert str(error) == (
                                "Kimi checkpoint RNG restoration failed on another rank"
                            )
                    else:
                        raise AssertionError("local RNG failure did not propagate")
                dist.barrier()
                assert load_changed_kimi_layout(folder, states=states, adapter=adapter)
                torch.testing.assert_close(
                    torch.rand(5), expected_random, atol=0, rtol=0
                )
                inputs = torch.ones(1, 3, 8)
                torch.testing.assert_close(
                    target.model.layers[0].block_sparse_moe(inputs),
                    source.model.layers[0].block_sparse_moe(inputs),
                    atol=1e-6,
                    rtol=1e-6,
                )
                dist.barrier()
                if rank == 0:
                    print(
                        f"source_grouped={source_grouped}, "
                        f"source_compiled={source_compiled}: RNG failure coordinated"
                    )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
