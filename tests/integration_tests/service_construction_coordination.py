"""Keep training service construction failures inside an explicit rank group.

Run with four ranks. Parameter planning or scheduler construction failure in
[0, 1] must reach its peer without involving [2, 3], whose real optimizer must
still update weights.
Loss factories additionally cover ordinary failures, interrupts and exits;
healthy subgroup peers run actual loss/backward/update after construction.
"""

from contextlib import ExitStack
from datetime import timedelta
from itertools import product
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn

from llmtuner.components.loss import cross_entropy_loss
from llmtuner.components.loss_contract import LossFunctions
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
        for error_type, failing_rank in product(
            (ValueError, KeyboardInterrupt, SystemExit), (0, 1)
        ):
            config = LLMTunerConfig()
            expected = LossFunctions(cross_entropy_loss)
            failure = error_type("injected loss construction failure")

            def loss_factory(
                actual,
                config=config,
                failure=failure,
                failing_rank=failing_rank,
                expected=expected,
            ):
                assert actual is config
                if rank == failing_rank:
                    raise failure
                return expected

            try:
                loss = assembly.build_loss_functions(
                    config, factory=loss_factory, error_group=group
                )
            except BaseException as error:
                if rank == failing_rank:
                    assert error is failure
                else:
                    assert rank < 2
                    assert isinstance(error, RuntimeError)
                    assert (
                        str(error)
                        == "Loss services construction failed on another rank"
                    )
            else:
                assert rank >= 2 and loss is expected
                logits = nn.Parameter(torch.tensor([[0.2, 0.3, -0.1]], device=device))
                before = logits.detach().clone()
                loss.token_sum(logits, torch.tensor([1], device=device)).backward()
                assert torch.isfinite(logits.grad).all().item()
                torch.optim.SGD([logits], lr=0.1, foreach=False, fused=False).step()
                assert not torch.equal(logits, before)
            dist.barrier()
            if rank == 0:
                print(
                    f"PASS loss construction {error_type.__name__} "
                    f"failing_rank={failing_rank}",
                    flush=True,
                )
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
            services = {}
            with ExitStack() as resources, patch.object(owner, attribute, replacement):

                def retain(name, service, services=services, resources=resources):
                    services[name] = service
                    close = getattr(service, "close", None)
                    if callable(close):
                        resources.callback(close)

                try:
                    stack = assembly.build_optimizer_stack(
                        LLMTunerConfig(),
                        model_parts=[model],
                        parallel_dims=None,
                        error_group=group,
                        on_service_acquired=retain,
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
            assert "optimizer" in services
            if phase == "scheduler" and rank != failing_rank:
                assert "lr_scheduler" in services
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
