"""Check PP optimizer rules and asymmetric failures on two or four CPU ranks.

PYTHONPATH=. torchrun --nproc_per_node=2 tests/integration_tests/pp_optimizer_groups.py

Use --nproc_per_node=4 for PP2 x DP2 and world-scoped initialization errors.
"""

from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config.optimizer_config import OptimizerConfig, ParamGroupConfig


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        assert world_size in (2, 4)
        topology = init_device_mesh(
            "cpu", (world_size // 2, 2), mesh_dim_names=("dp", "pp")
        )
        mesh = topology["pp"]
        stage_rank = mesh.get_local_rank()
        initialization = (
            {"initialization_group": dist.group.WORLD} if world_size == 4 else {}
        )
        model = torch.nn.Module()
        parameter = torch.nn.Parameter(torch.tensor([float(rank + 1)]))
        model.register_parameter(f"stage_{stage_rank}_weight", parameter)
        rules = [
            ParamGroupConfig(
                pattern=f"^stage_{stage}_weight$",
                optimizer_name="AdamW",
                optimizer_kwargs={"lr": 0.01 * (stage + 1)},
            )
            for stage in range(2)
        ]
        optimizer = OptimizersContainer(
            OptimizerConfig(param_groups=rules, implementation="for-loop"),
            model_parts=[model],
            pipeline_mesh=mesh,
            **initialization,
        )
        assert len(optimizer) == 1
        assert next(iter(optimizer)).param_groups[0]["param_names"] == [
            f"stage_{stage_rank}_weight"
        ]
        reference = torch.nn.Parameter(parameter.detach().clone())
        reference_optimizer = torch.optim.AdamW([reference], lr=0.01 * (stage_rank + 1))
        parameter.grad = torch.ones_like(parameter)
        reference.grad = torch.ones_like(reference)
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
        for invalid in (
            rules + [ParamGroupConfig(pattern="never", optimizer_name="AdamW")],
            rules + [rules[0]],
        ):
            try:
                OptimizersContainer(
                    OptimizerConfig(param_groups=invalid, implementation="for-loop"),
                    model_parts=[model],
                    pipeline_mesh=mesh,
                    **initialization,
                )
            except ValueError as error:
                assert "matched no parameters" in str(error)
            else:
                raise AssertionError("globally unused rule was accepted")
            dist.barrier()
        for phase in ("planning", "initialization", "unknown_options"):
            invalid_model = torch.nn.Module()
            invalid_model.register_parameter(
                "weight", torch.nn.Parameter(torch.ones(1))
            )
            parts = [invalid_model]
            config = OptimizerConfig(implementation="for-loop")
            if rank == 0:
                if phase == "planning":
                    wrapped = torch.nn.Module()
                    wrapped.register_parameter(
                        "weight", torch.nn.Parameter(torch.zeros(1))
                    )
                    invalid_model.add_module("_orig_mod", wrapped)
                elif phase == "initialization":
                    parts.append(invalid_model)
                else:
                    config.param_groups[0].optimizer_kwargs["learning_rate"] = 0.25
            try:
                OptimizersContainer(
                    config,
                    model_parts=parts,
                    pipeline_mesh=mesh,
                    **initialization,
                )
            except ValueError as error:
                assert rank == 0
                expected = {
                    "planning": "collide after wrapper removal",
                    "initialization": "assigned 1 parameter(s) to more than one group",
                    "unknown_options": "AdamW optimizer_kwargs contain unknown options",
                }[phase]
                assert expected in str(error)
                if phase == "unknown_options":
                    assert "learning_rate" in str(error)
            except RuntimeError as error:
                assert rank != 0
                label = (
                    "parameter planning" if phase == "planning" else "initialization"
                )
                assert str(error) == f"Optimizer {label} failed on another rank"
            else:
                raise AssertionError("asymmetric optimizer failure was accepted")
            dist.barrier()
            print(f"rank={rank}: asymmetric optimizer {phase} failure coordinated")
        if world_size == 4:
            local_model = torch.nn.Module()
            name = "unmatched" if rank == 0 else f"stage_{stage_rank}_weight"
            local_model.register_parameter(name, torch.nn.Parameter(torch.ones(1)))
            try:
                OptimizersContainer(
                    OptimizerConfig(param_groups=rules, implementation="for-loop"),
                    model_parts=[local_model],
                    pipeline_mesh=mesh,
                    **initialization,
                )
            except ValueError as error:
                assert rank in (0, 1)
                assert "matched no parameters" in str(error)
            except RuntimeError as error:
                assert rank in (2, 3)
                assert str(error) == "Optimizer initialization failed on another rank"
            else:
                raise AssertionError(
                    "a failed PP subgroup left another replica running"
                )
            dist.barrier()
            print(f"rank={rank}: PP subgroup rule failure reached all replicas")
        print(f"rank={rank}: PP rules and AdamW update match, unused rules rejected")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
