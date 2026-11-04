"""Real rank-local duplicate optimizer FQN rejection before state mutation."""

import argparse
from copy import deepcopy

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config import OptimizerConfig
from llmtuner.config.optimizer_config import ParamGroupConfig
from tests.integration_tests.harness import init_distributed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-optimizers", action="store_true")
    args = parser.parse_args()
    device = init_distributed()
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        torch.manual_seed(31)
        model = nn.Linear(4, 2, device=device)
        optimizer = OptimizersContainer(
            OptimizerConfig(
                implementation="for-loop",
                param_groups=(
                    [
                        ParamGroupConfig(pattern="weight", optimizer_name="Adam"),
                        ParamGroupConfig(pattern="bias", optimizer_name="AdamW"),
                    ]
                    if args.split_optimizers
                    else []
                ),
            ),
            model_parts=[model],
        )
        inputs = torch.linspace(-1, 1, 16, device=device).reshape(4, 4) + rank
        model(inputs).square().sum().backward()
        optimizer.step()
        checkpoint = deepcopy(optimizer.state_dict())
        assert len(optimizer.optimizers) == (2 if args.split_optimizers else 1)
        originals = [
            (inner, p, state, deepcopy(state))
            for inner in optimizer.optimizers
            for p, state in inner.state.items()
        ]
        groups = [
            group for inner in optimizer.optimizers for group in inner.param_groups
        ]
        names = [list(group["param_names"]) for group in groups]
        assert sum(map(len, names)) == 2, names
        if rank == 0:
            groups[-1]["param_names"][-1] = names[0][0]
        error = None
        try:
            optimizer.load_state_dict(checkpoint)
        except (ValueError, RuntimeError) as caught:
            error = caught
        outcomes = [None, None]
        dist.all_gather_object(outcomes, (type(error).__name__, str(error)))
        assert outcomes[0][0] == "ValueError", outcomes
        message = (
            "overlap across inner optimizers"
            if args.split_optimizers
            else "Duplicate optimizer parameter FQN"
        )
        assert message in outcomes[0][1], outcomes
        assert outcomes[1][0] == "RuntimeError", outcomes
        assert "failed on another rank" in outcomes[1][1], outcomes
        for inner, parameter, original, values in originals:
            assert inner.state[parameter] is original
            torch.testing.assert_close(original, values, rtol=0, atol=0)
        for group, saved_names in zip(groups, names, strict=True):
            group["param_names"] = saved_names
        # The rejected attempt leaves the container usable for a valid retry.
        optimizer.load_state_dict(checkpoint)
        torch.testing.assert_close(optimizer.state_dict(), checkpoint, rtol=0, atol=0)
        optimizer.zero_grad(set_to_none=True)
        before = [p.detach().clone() for p in model.parameters()]
        model(inputs).square().sum().backward()
        optimizer.step()
        for parameter, old in zip(model.parameters(), before, strict=True):
            assert torch.isfinite(parameter).all().item()
            assert not torch.equal(parameter, old)
        print(f"PASS optimizer duplicate FQN rejection/retry/update rank={rank}")

        # Pre/post hooks can fail before or after an inner state was applied.
        for phase in ("pre", "post"):
            failing_rank = 0 if phase == "pre" else 1

            def fail_local_load(*args, failing_rank=failing_rank):
                if rank == failing_rank:
                    raise ValueError("injected optimizer load hook failure")

            register = getattr(
                optimizer.optimizers[-1], f"register_load_state_dict_{phase}_hook"
            )
            handle = register(fail_local_load)
            error = None
            try:
                optimizer.load_state_dict(checkpoint)
            except (ValueError, RuntimeError) as caught:
                error = caught
            finally:
                handle.remove()
            dist.all_gather_object(outcomes, (type(error).__name__, str(error)))
            assert outcomes[failing_rank][0] == "ValueError", outcomes
            assert "injected optimizer load hook failure" in outcomes[failing_rank][1]
            peer = outcomes[1 - failing_rank]
            assert peer[0] == "RuntimeError", outcomes
            assert "Optimizer checkpoint application failed on another rank" in peer[1]
            # Reload every state: coordinated application failure is not atomic.
            optimizer.load_state_dict(checkpoint)
            torch.testing.assert_close(
                optimizer.state_dict(), checkpoint, rtol=0, atol=0
            )
            print(
                f"PASS optimizer {phase}-load-hook coordinated failure/reload "
                f"rank={rank}"
            )
        # A parameterless pipeline stage must still enter every world boundary.
        empty_peer = OptimizersContainer(
            OptimizerConfig(implementation="for-loop"),
            model_parts=[model if rank == 0 else nn.Module()],
            pipeline_mesh=init_device_mesh(device.type, (2,), mesh_dim_names=("pp",)),
        )
        local_checkpoint = deepcopy(empty_peer.state_dict())
        assert len(empty_peer.optimizers) == (1 if rank == 0 else 0)
        assert bool(local_checkpoint) == (rank == 0)

        def fail_owned_load(*args):
            raise ValueError("injected owned optimizer load failure")

        handle = (
            empty_peer.optimizers[0].register_load_state_dict_post_hook(fail_owned_load)
            if rank == 0
            else None
        )
        error = None
        try:
            empty_peer.load_state_dict(local_checkpoint)
        except (ValueError, RuntimeError) as caught:
            error = caught
        finally:
            if handle is not None:
                handle.remove()
        dist.all_gather_object(outcomes, (type(error).__name__, str(error)))
        assert outcomes[0][0] == "ValueError", outcomes
        assert "injected owned optimizer load failure" in outcomes[0][1], outcomes
        assert outcomes[1][0] == "RuntimeError", outcomes
        assert (
            "Optimizer checkpoint application failed on another rank" in outcomes[1][1]
        )
        empty_peer.load_state_dict(local_checkpoint)
        torch.testing.assert_close(empty_peer.state_dict(), local_checkpoint)
        print(f"PASS optimizer empty-owner coordinated failure/reload rank={rank}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
