"""Compare direct TP with pre-resolved TP using real distributed kernels.

Run explicitly with torchrun --standalone --nproc_per_node=2. The resolver
guard forbids a second search; it does not replace communication or compute.
"""

from unittest.mock import patch

import torch
import torch.distributed as dist

from experiments.qwen3_moe_framework.config import small_qwen3_moe_config
from experiments.qwen3_moe_framework.model import Qwen3MoeModel
from llmtuner.accelerator.collectives import synchronize_local_errors
from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.parallel.tensor_parallel.tensor_parallel import apply_tp, resolve_tp_plan
from llmtuner.parallel.topology import ParallelTopology, build_device_mesh
from tests.integration_tests.harness import init_distributed


def main() -> None:
    device = init_distributed()
    try:
        world = dist.get_world_size()
        cfg = ParallelConfig(tensor_parallel_size=world, data_parallel_shard_size=1)
        topology = ParallelTopology.from_config(cfg, world)
        mesh = build_device_mesh(topology)
        torch.manual_seed(1729)
        dtype = torch.bfloat16 if device.type == "npu" else torch.float32
        direct = Qwen3MoeModel(small_qwen3_moe_config()).to(device=device, dtype=dtype)
        planned = Qwen3MoeModel(small_qwen3_moe_config()).to(device=device, dtype=dtype)
        planned.load_state_dict(direct.state_dict(), strict=True)
        with synchronize_local_errors("TP preflight equivalence"):
            resolved = resolve_tp_plan(planned, cfg=cfg)
        direct = apply_tp(direct, mesh, cfg)
        replacements = []
        with (
            patch(
                "llmtuner.parallel.tensor_parallel.tensor_parallel.resolve_tp_plan",
                side_effect=AssertionError("Executor must consume the resolved plan"),
            ),
            patch(
                "llmtuner.parallel.tensor_parallel.tensor_parallel.looks_like_attention",
                side_effect=AssertionError("Executor must not inspect attention"),
            ),
            patch(
                "llmtuner.parallel.tensor_parallel.tensor_parallel.resolve_shared_expert",
                side_effect=AssertionError("Executor must use bound shared experts"),
            ),
        ):
            planned = apply_tp(
                planned,
                mesh,
                cfg,
                resolved_plan=resolved,
                on_replacement=lambda modules, parameters: replacements.append(
                    (modules, parameters)
                ),
            )
        tokens = torch.arange(8, device=device).reshape(1, -1)
        local = tokens.chunk(world, dim=1)[dist.get_rank()].contiguous()
        positions = torch.arange(8, device=device).reshape(1, -1)
        outputs = []
        for model in (direct, planned):
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            logits = model(local, position_ids=positions)
            logits.float().square().mean().backward()
            optimizer.step()
            outputs.append(logits.detach())
        with synchronize_local_errors("TP preflight result comparison"):
            assert len(replacements) == 1
            modules, parameters = replacements[0]
            assert modules and parameters
            current = {id(parameter) for parameter in planned.parameters()}
            for source, targets in parameters.items():
                assert id(source) not in current
                assert len(targets) == 1 and id(targets[0]) in current
            torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
            assert outputs[0].numel() > 0 and outputs[0].count_nonzero() > 0
            nonzero_gradients = 0
            for (name, expected), (other, actual) in zip(
                direct.named_parameters(), planned.named_parameters(), strict=True
            ):
                assert name == other
                torch.testing.assert_close(expected, actual, atol=0, rtol=0, msg=name)
                assert (expected.grad is None) == (actual.grad is None)
                if expected.grad is not None:
                    torch.testing.assert_close(
                        expected.grad, actual.grad, atol=0, rtol=0, msg=name
                    )
                    nonzero_gradients += int(bool(expected.grad.count_nonzero()))
            assert nonzero_gradients > 0
        if dist.get_rank() == 0:
            print("Pre-resolved TP: logits, gradients and updates exactly equal")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
