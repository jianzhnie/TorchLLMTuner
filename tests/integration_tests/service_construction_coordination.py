"""Keep optimizer service construction failures inside an explicit rank group.

Run with four ranks. Parameter planning or scheduler construction failure in
[0, 1] must reach its peer without involving [2, 3], whose real optimizer must
still update weights.
"""

from datetime import timedelta
from itertools import product
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn

from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config.llmtuner_config import LLMTunerConfig
from llmtuner.trainer.runtime import assembly
from tests.integration_tests.harness import init_distributed


def main() -> None:
    device = init_distributed(timeout=timedelta(seconds=45))
    try:
        if dist.get_world_size() != 4:
            raise ValueError("This test requires four ranks")
        rank = dist.get_rank()
        groups = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
        group = groups[rank // 2]
        for phase, failing_rank in product(("parameters", "scheduler"), (0, 1)):
            owner, attribute, error_phase = (
                (
                    OptimizersContainer,
                    "_build_impl_kwargs",
                    "Optimizer parameter planning",
                )
                if phase == "parameters"
                else (assembly, "build_lr_scheduler", "Optimizer services construction")
            )
            original = getattr(owner, attribute)
            model = nn.Linear(2, 2, bias=False, device=device)
            with torch.no_grad():
                model.weight.fill_(1)
            before = model.weight.detach().clone()
            failure = ValueError(f"injected {phase} construction failure")

            def build_scheduler(
                *args,
                failing_rank=failing_rank,
                failure=failure,
                original=original,
                **kwargs,
            ):
                if rank == failing_rank:
                    raise failure
                return original(*args, **kwargs)

            replacement = (
                staticmethod(build_scheduler)
                if phase == "parameters"
                else build_scheduler
            )
            with patch.object(owner, attribute, replacement):
                try:
                    stack = assembly.build_optimizer_stack(
                        LLMTunerConfig(),
                        model_parts=[model],
                        parallel_dims=None,
                        error_group=group,
                    )
                except ValueError as error:
                    assert rank == failing_rank and error is failure
                except RuntimeError as error:
                    assert rank < 2 and rank != failing_rank
                    assert str(error) == f"{error_phase} failed on another rank"
                else:
                    assert rank >= 2, "Failed construction was not coordinated"
                    model.weight.grad = torch.ones_like(model.weight)
                    stack.optimizer.step()
                    assert torch.isfinite(model.weight).all().item()
                    assert not torch.equal(model.weight, before)
            if rank < 2:
                torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
            dist.barrier()
            if rank == 0:
                print(
                    f"PASS service construction phase={phase} "
                    f"failing_rank={failing_rank}",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
