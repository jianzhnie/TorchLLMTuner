"""Relocated FSDP hooks and multiple contexts against independent dense SGD.

Run: PYTHONPATH=. torchrun --nproc_per_node=2 \
    tests/integration_tests/fine_grained_fsdp_equivalence.py

CPU/Gloo by default; append --device cuda for CUDA/NCCL. Two managed units
per parent hook, two parent hooks, grad/no-grad inputs, explicit prefetch,
two accumulated microbatches, and three updates. Checks failed-forward reset.
CPU evidence does not establish target-device stream overlap or performance.
Append --outstanding-forwards to run two forwards before reversing their
backward order, including the fresh runtime restored from DCP.
Append --compile-backend aot_eager to compile before FSDP and retain the
independent eager dense reference. This does not validate Inductor kernels.
"""

from __future__ import annotations

import argparse
import os
from itertools import product
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint

from llmtuner.components.optimizer.state_dict import get_flat_optim_state_dict
from llmtuner.config.parallel_config import FSDPConfig, ParallelConfig
from llmtuner.parallel.fully_shard_parallel.fine_grained_param_group import (
    FineGrainedFSDPParamGroup,
)
from llmtuner.parallel.fully_shard_parallel.fine_grained_state import (
    FineGrainedFSDPState,
)
from llmtuner.parallel.fully_shard_parallel.fully_shard_parallel import (
    build_fsdp_placement_fn,
    force_sum_grad_reduction,
)
from llmtuner.parallel.fully_shard_parallel.plan import apply_fsdp_by_pattern
from llmtuner.utils.module_names import canonical_fqn


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 4)
        self.last = nn.Linear(4, 4)
        self.fail = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Inspect private lifecycle objects eagerly; Dynamo cannot safely trace
        # PyTorch's weak module-state registry. assert_clean runs after backward.
        if not torch.compiler.is_compiling() and hasattr(self.first, "_get_fsdp_state"):
            for context in self.first._get_fsdp_state().communication_contexts:
                if state := context.all_gather_state:
                    owner = state.hook_module()
                    owner = getattr(owner, "_checkpoint_wrapped_module", owner)
                    assert owner is self, "retained a previous hook buffer"
        hidden = torch.tanh(self.first(inputs))
        if self.fail:
            raise RuntimeError("injected forward failure")
        return self.last(hidden)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])
        self.head = nn.Linear(4, 2)
        self.checkpoint = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            inputs = (
                checkpoint(block, inputs, use_reentrant=False)
                if self.checkpoint
                else block(inputs)
            )
        return self.head(inputs)


def full(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def optimizer_state(optimizer: torch.optim.SGD) -> dict:
    return {
        name: full(value) if isinstance(value, torch.Tensor) else value
        for name, value in get_flat_optim_state_dict(optimizer).items()
    }


def assert_clean(model: Model, *, multiple_groups: bool) -> None:
    root = model._get_fsdp_state()
    assert isinstance(root, FineGrainedFSDPState)
    assert len(root.communication_contexts) == 2
    assert root._state_ctx.iter_forward_root is None
    assert not root._state_ctx.post_backward_final_callback_queued
    for context in root.communication_contexts:
        assert context.all_gather_state is None
        assert context.reduce_scatter_states == []
        assert context.post_forward_order == []
    for state in root._state_ctx.all_states:
        assert state._training_state == TrainingState.IDLE
        assert not state._modules_to_run_forward
        for group in state._fsdp_param_groups:
            assert isinstance(group, FineGrainedFSDPParamGroup)
            assert group.hook_module is state.hook_module
            assert group.communication_contexts is root.communication_contexts
            assert group._training_state == TrainingState.IDLE
    for block in model.blocks:
        first = block.first._get_fsdp_state()
        last = block.last._get_fsdp_state()
        assert first.hook_module() is block
        assert last.hook_module() is block
        assert first._comm_ctx is not last._comm_ctx
        assert first._comm_ctx.all_gather_stream is last._comm_ctx.all_gather_stream
        assert len(first._fsdp_param_groups) == (2 if multiple_groups else 1)
        assert len(last._fsdp_param_groups) == (2 if multiple_groups else 1)


def runtime(
    device: torch.device,
    mesh: DeviceMesh,
    replicated_mesh: DeviceMesh,
    *,
    reshard: bool,
    recompute: bool,
    multiple_groups: bool,
    seed: int = 42,
    learning_rate: float = 0.1,
    compile_backend: str | None = None,
) -> tuple[Model, torch.optim.SGD]:
    torch.manual_seed(seed)
    model = Model().to(device)
    model.checkpoint = recompute
    placement = {}
    if multiple_groups:
        from torch.distributed.tensor import Shard

        placement["shard_placement_fn"] = build_fsdp_placement_fn(
            {
                layer.bias: Shard(0)
                for block in model.blocks
                for layer in (block.first, block.last)
            },
            dp_mesh=mesh,
            edp_mesh=replicated_mesh,
        )
    cfg = ParallelConfig(
        data_parallel_shard_size=2,
        fsdp_config=FSDPConfig(
            fsdp_implementation="custom",
            fsdp_reshard_after_forward="always" if reshard else "never",
            fsdp_apply_modules={
                f"blocks.{index}.{name}": {"hook_module": block, **placement}
                for index, block in enumerate(model.blocks)
                for name in ("first", "last")
            },
            fsdp_forward_prefetch_count=2,
            fsdp_backward_prefetch_count=2,
        ),
    )
    if compile_backend is not None and recompute:
        from llmtuner.memory.activation_checkpoint import wrap_full

        # Match production AC -> compile -> FSDP, including the recompute context.
        model.checkpoint = False
        for index, block in enumerate(model.blocks):
            wrapped = wrap_full(block)
            model.blocks[index] = wrapped
            for name in ("first", "last"):
                cfg.fsdp_config.fsdp_apply_modules[f"blocks.{index}.{name}"][
                    "hook_module"
                ] = wrapped
    if compile_backend is not None:
        model = torch.compile(model, backend=compile_backend)
    apply_fsdp_by_pattern(
        model,
        dp_mesh=mesh,
        cfg=cfg,
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
    )
    if device.type == "cpu":
        force_sum_grad_reduction(model)
    optimizer = torch.optim.SGD(
        ((canonical_fqn(name), p) for name, p in model.named_parameters()),
        lr=learning_rate,
        momentum=0.9,
    )
    return model, optimizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--outstanding-forwards", action="store_true")
    parser.add_argument("--compile-backend", choices=("aot_eager",))
    args = parser.parse_args()
    if args.device == "cuda":
        if not torch.cuda.is_available():
            parser.error("--device cuda requires an available CUDA device")
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    dist.init_process_group("nccl" if args.device == "cuda" else "gloo")
    checkpoint_root = None
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        checkpoint_root = TemporaryDirectory() if rank == 0 else None
        paths = [checkpoint_root.name if checkpoint_root is not None else None]
        dist.broadcast_object_list(paths, src=0)
        mesh = init_device_mesh(device.type, (2,))
        replicated_mesh = init_device_mesh(
            device.type, (2, 1), mesh_dim_names=("dp_replicate", "dp_shard")
        )
        for input_grad, reshard, recompute, multiple_groups in product(
            (False, True), repeat=4
        ):
            if args.compile_backend is not None:
                from torch._dynamo.utils import counters

                torch._dynamo.reset()
                counters.clear()
            torch.manual_seed(42)
            reference = Model().to(device)
            options = dict(
                reshard=reshard,
                recompute=recompute,
                multiple_groups=multiple_groups,
                compile_backend=args.compile_backend,
            )
            model, optimizer = runtime(device, mesh, replicated_mesh, **options)
            reference_optimizer = torch.optim.SGD(
                reference.named_parameters(), lr=0.1, momentum=0.9
            )
            fault_block = model.blocks[1]
            if hasattr(fault_block, "_checkpoint_wrapped_module"):
                fault_block = fault_block._checkpoint_wrapped_module
            fault_block.fail = True
            try:
                model(torch.ones(2, 4, device=device))
            except RuntimeError as error:
                assert str(error) == "injected forward failure"
            else:
                raise AssertionError("injected forward failure was not raised")
            model.reset_iter_state()
            assert_clean(model, multiple_groups=multiple_groups)
            fault_block.fail = False
            failed_input = torch.ones(2, 4, device=device, requires_grad=True)

            def fail_backward(gradient):
                raise RuntimeError("injected backward failure")

            failure_handle = failed_input.register_hook(fail_backward)
            try:
                model(failed_input).square().sum().backward()
            except RuntimeError as error:
                assert str(error) == "injected backward failure"
            else:
                raise AssertionError("injected backward failure was not raised")
            finally:
                failure_handle.remove()
            model.reset_iter_state()
            assert_clean(model, multiple_groups=multiple_groups)
            assert all(parameter.grad is None for parameter in model.parameters())
            resumed = None
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                reference_optimizer.zero_grad(set_to_none=True)
                if resumed is not None:
                    resumed[1].zero_grad(set_to_none=True)
                pending = []
                for microbatch in range(2):
                    generator = torch.Generator().manual_seed(
                        100 + step * 2 + microbatch
                    )
                    inputs = torch.randn(4, 4, generator=generator).to(device)
                    labels = torch.randn(4, 2, generator=generator).to(device)
                    reference_input = inputs.clone().requires_grad_(input_grad)
                    local_input = (
                        inputs[rank * 2 : rank * 2 + 2]
                        .clone()
                        .requires_grad_(input_grad)
                    )
                    expected_output = reference(reference_input)
                    actual_output = model(local_input)
                    torch.testing.assert_close(
                        actual_output, expected_output[rank * 2 : rank * 2 + 2]
                    )
                    expected_loss = (expected_output - labels).square().sum() / 16
                    actual_loss = (
                        actual_output - labels[rank * 2 : rank * 2 + 2]
                    ).square().sum() / 16
                    if not args.outstanding_forwards:
                        expected_loss.backward()
                        actual_loss.backward()
                    if resumed is not None:
                        resumed_input = (
                            local_input.detach().clone().requires_grad_(input_grad)
                        )
                        resumed_output = resumed[0](resumed_input)
                        torch.testing.assert_close(
                            resumed_output, actual_output, rtol=0, atol=0
                        )
                        resumed_loss = (
                            resumed_output - labels[rank * 2 : rank * 2 + 2]
                        ).square().sum() / 16
                        if not args.outstanding_forwards:
                            resumed_loss.backward()
                        if input_grad and not args.outstanding_forwards:
                            torch.testing.assert_close(
                                resumed_input.grad, local_input.grad, rtol=0, atol=0
                            )
                        if not args.outstanding_forwards:
                            assert_clean(resumed[0], multiple_groups=multiple_groups)
                    if input_grad and not args.outstanding_forwards:
                        torch.testing.assert_close(
                            local_input.grad,
                            reference_input.grad[rank * 2 : rank * 2 + 2],
                        )
                    if args.outstanding_forwards:
                        pending.append(
                            (
                                expected_loss,
                                actual_loss,
                                reference_input,
                                local_input,
                                resumed_loss if resumed else None,
                                resumed_input if resumed else None,
                            )
                        )
                    else:
                        assert_clean(model, multiple_groups=multiple_groups)
                for (
                    expected_loss,
                    actual_loss,
                    reference_input,
                    local_input,
                    resumed_loss,
                    resumed_input,
                ) in reversed(pending):
                    expected_loss.backward()
                    actual_loss.backward()
                    if resumed_loss is not None:
                        resumed_loss.backward()
                    if input_grad:
                        torch.testing.assert_close(
                            local_input.grad,
                            reference_input.grad[rank * 2 : rank * 2 + 2],
                        )
                        if resumed_input is not None:
                            torch.testing.assert_close(
                                resumed_input.grad, local_input.grad, rtol=0, atol=0
                            )
                assert_clean(model, multiple_groups=multiple_groups)
                if resumed is not None:
                    assert_clean(resumed[0], multiple_groups=multiple_groups)
                reference_parameters = dict(reference.named_parameters())
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(
                        full(parameter.grad),
                        reference_parameters[canonical_fqn(name)].grad,
                        atol=2e-6,
                        rtol=2e-5,
                    )
                optimizer.step()
                reference_optimizer.step()
                if resumed is not None:
                    original_parameters = dict(model.named_parameters())
                    for name, parameter in resumed[0].named_parameters():
                        torch.testing.assert_close(
                            full(parameter.grad),
                            full(original_parameters[name].grad),
                            rtol=0,
                            atol=0,
                        )
                    resumed[1].step()
                    for name, parameter in resumed[0].named_parameters():
                        torch.testing.assert_close(
                            full(parameter),
                            full(original_parameters[name]),
                            rtol=0,
                            atol=0,
                        )
                    torch.testing.assert_close(
                        optimizer_state(resumed[1]),
                        optimizer_state(optimizer),
                        rtol=0,
                        atol=0,
                    )
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(
                        full(parameter),
                        reference_parameters[canonical_fqn(name)],
                        atol=2e-6,
                        rtol=2e-5,
                    )
                torch.testing.assert_close(
                    optimizer_state(optimizer),
                    optimizer_state(reference_optimizer),
                    atol=2e-6,
                    rtol=2e-5,
                )
                if step == 1:
                    folder = (
                        f"{paths[0]}/{input_grad}_{reshard}_"
                        f"{recompute}_{multiple_groups}"
                    )
                    model_state, optim_state = get_state_dict(model, optimizer)
                    dcp.save(
                        {"model": model_state, "optimizer": optim_state},
                        checkpoint_id=folder,
                    )
                    resumed = runtime(
                        device,
                        mesh,
                        replicated_mesh,
                        seed=999,
                        learning_rate=0.3,
                        **options,
                    )
                    assert not torch.equal(
                        full(model.head.weight), full(resumed[0].head.weight)
                    )
                    model_state, optim_state = get_state_dict(*resumed)
                    restored = {"model": model_state, "optimizer": optim_state}
                    dcp.load(restored, checkpoint_id=folder)
                    set_state_dict(
                        *resumed,
                        model_state_dict=restored["model"],
                        optim_state_dict=restored["optimizer"],
                    )
                    torch.testing.assert_close(
                        dict(
                            (name, full(p)) for name, p in resumed[0].named_parameters()
                        ),
                        dict((name, full(p)) for name, p in model.named_parameters()),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        optimizer_state(resumed[1]),
                        optimizer_state(optimizer),
                        rtol=0,
                        atol=0,
                    )
            if args.compile_backend is not None:
                assert counters["stats"]["unique_graphs"] > 0
            if rank == 0:
                print(
                    f"PASS custom hooks input_grad={input_grad} reshard={reshard}: "
                    f"recompute={recompute} multiple_groups={multiple_groups}; "
                    f"outstanding_forwards={args.outstanding_forwards}; "
                    f"compile_backend={args.compile_backend}; "
                    "reset, output, input/parameter gradients, momentum, DCP resume",
                    flush=True,
                )
        dist.barrier()
    finally:
        if checkpoint_root is not None:
            checkpoint_root.cleanup()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
