"""Real DCP rejects damaged Kimi training state before cross-layout writes.

PYTHONPATH=. torchrun --nproc_per_node=2 \
    tests/integration_tests/kimi_layout_preflight.py

CPU/Gloo, ordinary tensors and actual MoE modules, no full Trainer or EP mesh.
Both ranks read the same damaged checkpoint; this is not asymmetric corruption.
"""

from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from itertools import product
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from llmtuner.components.checkpointer.kimi_layout import load_changed_kimi_layout
from llmtuner.components.checkpointer.model_state import ModelCheckpointState
from llmtuner.components.optimizer.ema import EMA
from llmtuner.models.hf.state_dict_adapter import HFTransformerStateDictAdapter
from llmtuner.parallel.expert_parallel.model_replacement import swap_hf_moe_blocks
from tests.unit_tests.cpu.models.test_kimi_dcp_layout import OptimizerState
from tests.unit_tests.cpu.parallel.test_kimi_conversion import make_model


def build():
    wrapper = torch.nn.Module()
    wrapper.model = make_model(latent=True, shared=1)
    return wrapper


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        context = TemporaryDirectory() if rank == 0 else nullcontext(None)
        with context as root:
            roots = [root]
            dist.broadcast_object_list(roots, src=0)
            torch.manual_seed(28)
            source = build()
            config = source.model.layers[0].block_sparse_moe.config
            source_optimizer = OptimizerState(source)
            inputs = torch.randn(1, 3, 8)
            source.model.layers[0].block_sparse_moe(inputs).square().sum().backward()
            source_optimizer.optimizer.step()
            source_optimizer.optimizer.zero_grad()
            source_ema = EMA(model_parts=[source], decay=0.5)
            source_ema.step(1)
            saved = dict(source.state_dict())
            saved["optimizer"] = source_optimizer.state_dict()
            saved["ema"] = source_ema.state_dict()
            healthy_path = f"{roots[0]}/healthy"
            dcp.save(saved, checkpoint_id=healthy_path)
            prefix = "model.layers.0.block_sparse_moe."
            for category, corruption in product(
                ("optimizer", "ema"), ("duplicate", "shape")
            ):
                damaged = deepcopy(saved)
                field = "exp_avg" if category == "optimizer" else "ema_params"
                key = f"state.{prefix}experts.0.w1.weight.{field}"
                if corruption == "duplicate":
                    alias = key.replace("experts.0.", "experts.00.")
                    damaged[category][alias] = torch.zeros_like(damaged[category][key])
                else:
                    damaged[category][key] = torch.zeros(1, 1)
                path = f"{roots[0]}/{category}-{corruption}"
                dcp.save(damaged, checkpoint_id=path)
                target = build()
                swap_hf_moe_blocks(target.model)
                optimizer = OptimizerState(target)
                ema = EMA(model_parts=[target], decay=0.5)
                states = {
                    "model": ModelCheckpointState(target),
                    "optimizer": optimizer,
                    "ema": ema,
                }
                before = {
                    name: deepcopy(state.state_dict()) for name, state in states.items()
                }
                rng_before = torch.get_rng_state().clone()
                adapter = HFTransformerStateDictAdapter(
                    config, None, kimi_expert_parallel=True
                )
                try:
                    load_changed_kimi_layout(path, states=states, adapter=adapter)
                except ValueError as error:
                    expected = (
                        "duplicate expert index"
                        if corruption == "duplicate"
                        else "moment tensors disagree"
                    )
                    assert expected in str(error), error
                else:
                    raise AssertionError("damaged training state was accepted")
                for name, state in states.items():
                    torch.testing.assert_close(
                        state.state_dict(), before[name], rtol=0, atol=0
                    )
                torch.testing.assert_close(
                    torch.get_rng_state(), rng_before, rtol=0, atol=0
                )
                dist.barrier()
                assert load_changed_kimi_layout(
                    healthy_path, states=states, adapter=adapter
                )
                torch.testing.assert_close(
                    target.model.layers[0].block_sparse_moe(inputs),
                    source.model.layers[0].block_sparse_moe(inputs),
                    rtol=1e-6,
                    atol=1e-6,
                )
                for name, source_state, suffix in (
                    ("optimizer", saved["optimizer"], "exp_avg"),
                    ("ema", saved["ema"], "ema_params"),
                ):
                    expected = torch.stack(
                        [
                            source_state[
                                f"state.{prefix}experts.{i}.w1.weight.{suffix}"
                            ]
                            for i in range(4)
                        ]
                    )
                    actual = states[name].state_dict()[
                        f"state.{prefix}routed_experts.inner_experts.w1_EFD.{suffix}"
                    ]
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                dist.barrier()
                if rank == 0:
                    print(f"PASS {category=} {corruption=}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
