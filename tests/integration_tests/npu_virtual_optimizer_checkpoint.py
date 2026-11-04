"""Ascend NPU: virtual Adam moments match device state across checkpoint resume.

Run: python tests/integration_tests/npu_virtual_optimizer_checkpoint.py <output-dir>
Uses a real single-rank Trainer and synchronous native DCP, never a fake allocator.
"""

import os
import sys
from copy import deepcopy
from pathlib import Path

import torch

from llmtuner.components.optimizer.state_residency import install_state_residency
from llmtuner.config import (
    CheckpointConfig,
    LLMTunerConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)
from llmtuner.trainer.trainer import Trainer


def check_closure_moment_allocation() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -0.5], device="npu"))
    reference = torch.nn.Parameter(parameter.detach().clone())
    optimizer = torch.optim.AdamW([parameter], lr=0.01, foreach=False)
    expected = torch.optim.AdamW([reference], lr=0.01, foreach=False)
    install_state_residency(optimizer, "npu_virtual")

    def closure(optim, value):
        optim.zero_grad(set_to_none=True)
        loss = value.square().sum()
        loss.backward()
        return loss

    for _ in range(3):
        optimizer.step(closure=lambda: closure(optimizer, parameter))
        expected.step(lambda: closure(expected, reference))
        torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
        for name in ("exp_avg", "exp_avg_sq"):
            # Materialize on NPU: raw swapped-storage CPU copies do not
            # necessarily observe the values read by device kernels.
            torch.testing.assert_close(
                optimizer.state[parameter][name].clone(),
                expected.state[reference][name],
                rtol=0,
                atol=0,
            )
    torch.npu.synchronize()
    print("PASS NPU virtual closure: three updates and device moments exact")


def check_failed_moment_copy_retry() -> None:
    for failed_name in ("exp_avg", "exp_avg_sq"):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -0.5], device="npu"))
        optimizer = torch.optim.AdamW([parameter], lr=0.01, foreach=False)
        install_state_residency(optimizer, "npu_virtual")
        parameter.grad = torch.tensor([0.25, -0.5], device="npu")
        optimizer.step()
        snapshot = deepcopy(optimizer.state_dict())
        destinations = {
            name: optimizer.state[parameter][name] for name in ("exp_avg", "exp_avg_sq")
        }
        destination = destinations[failed_name]
        failure = OSError("injected NPU virtual moment restore failure")

        def fail_copy(incoming, failure=failure):
            raise failure

        destination.copy_ = fail_copy
        try:
            try:
                optimizer.load_state_dict(deepcopy(snapshot))
            except OSError as error:
                assert error is failure
            else:
                raise AssertionError("moment-copy failure was not propagated")
            for name, tensor in destinations.items():
                assert optimizer.state[parameter][name] is tensor
        finally:
            del destination.copy_

        optimizer.load_state_dict(deepcopy(snapshot))
        for name, tensor in destinations.items():
            assert optimizer.state[parameter][name] is tensor
            torch.testing.assert_close(
                tensor, snapshot["state"][0][name], rtol=0, atol=0
            )

        reference_parameter = torch.nn.Parameter(parameter.detach().clone())
        reference = torch.optim.AdamW([reference_parameter], lr=0.01, foreach=False)
        reference.load_state_dict(deepcopy(snapshot))
        parameter.grad = torch.tensor([-0.75, 0.125], device="npu")
        reference_parameter.grad = parameter.grad.clone()
        optimizer.step()
        reference.step()
        torch.testing.assert_close(parameter, reference_parameter, rtol=0, atol=0)
        torch.npu.synchronize()
        print(
            f"PASS NPU virtual {failed_name} copy failure, retry and next update exact",
            flush=True,
        )


def _config(directory: str, residency: str, steps: int) -> LLMTunerConfig:
    return LLMTunerConfig(
        model=ModelConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        ),
        optimizer=OptimizerConfig(implementation="for-loop", state_residency=residency),
        training=TrainingConfig(
            steps=steps,
            dump_folder=directory,
            checkpoint_config=CheckpointConfig(
                enable=True,
                interval=1,
                async_mode=os.environ.get("VIRTUAL_CHECKPOINT_ASYNC_MODE", "disabled"),
                last_save_model_only=False,
            ),
        ),
    )


def _run(directory: str, residency: str, steps: int, expected_start=None):
    trainer = Trainer(_config(directory, residency, steps))
    virtual_destinations = []
    if residency == "npu_virtual" and expected_start is not None:
        # Materialize the load targets before Trainer restores the checkpoint;
        # the same objects must remain installed after Optimizer.load_state_dict.
        trainer.optimizer.state_dict()
        for optimizer in trainer.optimizer.optimizers:
            for parameter, state in optimizer.state.items():
                for name in ("exp_avg", "exp_avg_sq"):
                    if name in state:
                        virtual_destinations.append(
                            (optimizer, parameter, name, state[name])
                        )
    torch.npu.reset_peak_memory_stats()
    try:
        trainer.train()
        for optimizer, parameter, name, destination in virtual_destinations:
            assert optimizer.state[parameter][name] is destination, (
                f"checkpoint restore replaced virtual optimizer state {name}"
            )
        if expected_start is not None:
            restored_model = {
                name: value.detach().cpu().clone()
                for name, value in trainer.example_model.state_dict().items()
            }
            restored_optimizer = {
                name: value.detach().cpu().clone()
                if isinstance(value, torch.Tensor)
                else value
                for name, value in trainer.optimizer.state_dict().items()
            }
            _compare(restored_model, expected_start[0])
            _compare(restored_optimizer, expected_start[1])
        model = {
            name: value.detach().cpu().clone()
            for name, value in trainer.example_model.state_dict().items()
        }
        optimizer = {
            name: value.detach().cpu().clone()
            if isinstance(value, torch.Tensor)
            else value
            for name, value in trainer.optimizer.state_dict().items()
        }
        return model, optimizer, torch.npu.max_memory_allocated()
    finally:
        trainer.close()


def _compare(actual, expected) -> None:
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        other = expected[key]
        if isinstance(value, torch.Tensor):
            difference = (value - other).abs().max().item()
            # Native NPU DCP BF16 serialization has a small baseline rounding
            # trajectory even without virtual optimizer state. Keep that platform
            # tolerance explicit rather than claiming a bitwise DCP contract.
            torch.testing.assert_close(
                value,
                other,
                rtol=5e-3,
                atol=5e-3,
                msg=f"{key}: maxdiff={difference}",
            )
        else:
            assert value == other, key


def main() -> None:
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    check_closure_moment_allocation()
    check_failed_moment_copy_retry()
    initial = _run(str(root / "initial"), "npu_virtual", 2)
    assert (root / "initial/checkpoint/step-2/.metadata").is_file()
    _run(str(root / "initial"), "npu_virtual", 2, expected_start=initial)
    virtual_model, virtual_optimizer, virtual_memory = _run(
        str(root / "initial"), "npu_virtual", 4
    )
    device_model, device_optimizer, device_memory = _run(
        str(root / "device"), "device", 4
    )
    _compare(virtual_model, device_model)
    _compare(virtual_optimizer, device_optimizer)
    print(
        "PASS npu_virtual: multi-step updates and resumed optimizer state match; "
        f"peak_allocated_virtual={virtual_memory}, device={device_memory}"
    )


if __name__ == "__main__":
    main()
