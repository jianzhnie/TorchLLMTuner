"""Keep scheduler/EMA failures within an explicitly supplied update group.

Run with four processes using the device-neutral integration harness. Two
independent update groups deliberately take different paths: one sees a local
failure, while the other must complete its scheduler and EMA updates.

PYTHONPATH=. torchrun --standalone --nproc_per_node=4 \
    tests/integration_tests/engine_update_coordination.py
"""

from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn

from llmtuner.trainer.runtime.engine import TrainingEngine
from tests.integration_tests.harness import init_distributed


def run_case(
    device: torch.device,
    group: dist.ProcessGroup,
    *,
    rank: int,
    phase: str,
    failing_rank: int,
) -> None:
    """Verify committed weights and subsequent service ordering on each rank."""
    model = nn.Linear(1, 1, bias=False, device=device)
    with torch.no_grad():
        model.weight.fill_(1)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25, foreach=False)
    events = []
    failure = ValueError(f"injected {phase} failure on rank {rank}")

    def scheduler_step() -> None:
        events.append("scheduler")
        if phase == "scheduler" and rank == failing_rank:
            raise failure

    def ema_step(step: int) -> None:
        events.append(("ema", step))
        if phase == "ema" and rank == failing_rank:
            raise failure

    engine = TrainingEngine(
        model_parts=[model],
        parallel_dims=None,
        device=device,
        max_context_length=1,
        chunked_loss_num_chunks=1,
        param_context=nullcontext,
        optimizer=optimizer,
        lr_scheduler=SimpleNamespace(step=scheduler_step),
        ema=SimpleNamespace(step=ema_step),
        error_group=group,
    )
    try:
        engine.optimizer_step(7)
    except ValueError as error:
        assert rank == failing_rank and error is failure
    except RuntimeError as error:
        assert rank < 2 and rank != failing_rank
        assert str(error) == (
            f"Training {'EMA' if phase == 'ema' else 'scheduler'} "
            "update failed on another rank"
        )
    else:
        assert rank >= 2, "Failure did not reach the update peer"
    # Optimizer updates are committed even if later services fail.
    torch.testing.assert_close(
        model.weight, torch.full_like(model.weight, 0.75), rtol=0, atol=0
    )
    expected = ["scheduler"]
    if phase == "ema" or rank >= 2:
        expected.append(("ema", 7))
    assert events == expected, (rank, phase, events)
    dist.barrier()
    if rank == 0:
        print(
            f"{phase}, failing_rank={failing_rank}: "
            "group-local failure and independent updates verified",
            flush=True,
        )


def main() -> None:
    device = init_distributed(timeout=timedelta(seconds=45))
    try:
        if dist.get_world_size() != 4:
            raise ValueError("This test requires four ranks")
        rank = dist.get_rank()
        groups = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
        group = groups[rank // 2]
        for phase in ("scheduler", "ema"):
            for failing_rank in (0, 1):
                run_case(
                    device, group, rank=rank, phase=phase, failing_rank=failing_rank
                )
        dist.destroy_process_group(group)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
