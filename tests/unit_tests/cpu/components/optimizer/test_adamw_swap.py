"""Real CPU mathematical fallback and bounded explicit AdamW state exchange."""

from copy import deepcopy

import pytest
import torch

from llmtuner.components.optimizer.adamw_swap import AdamWSwap
from llmtuner.components.optimizer.state_dict import init_optim_state
from llmtuner.config import (
    CheckpointConfig,
    LLMTunerConfig,
    OptimizerConfig,
    ParallelConfig,
    TrainingConfig,
)
from llmtuner.errors import ConfigError, UnsupportedCombinationError


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("maximize", [False, True])
def test_sliced_updates_counters_and_reload_match_native(dtype, maximize):
    torch.manual_seed(51)
    values = [
        torch.nn.Parameter(torch.randn(19, dtype=dtype)),
        torch.nn.Parameter(torch.randn(7, dtype=dtype)),
    ]
    references = [torch.nn.Parameter(value.detach().clone()) for value in values]
    options = dict(
        lr=0.03, weight_decay=0.1, betas=(0.8, 0.95), foreach=False, maximize=maximize
    )
    swap = AdamWSwap(values, max_device_bytes=32, **options)
    native = torch.optim.AdamW(references, **options)
    for step in range(5):
        for index, (value, reference) in enumerate(
            zip(values, references, strict=True)
        ):
            gradient = (
                None
                if (step + index) % 3 == 0
                else torch.full_like(value, (step + 1) * 0.125)
            )
            value.grad = gradient
            reference.grad = gradient.clone() if gradient is not None else None
        swap.step()
        native.step()
        for value, reference in zip(values, references, strict=True):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
            for name in ("step", "exp_avg", "exp_avg_sq"):
                if name in native.state.get(reference, {}):
                    torch.testing.assert_close(
                        swap.state[value][name],
                        native.state[reference][name],
                        rtol=0,
                        atol=0,
                    )
        snapshot = deepcopy(swap.state_dict())
        swap.load_state_dict(snapshot)
    assert swap.staging_peak_payload_bytes <= 32
    assert swap.host_state_bytes == 2 * sum(
        value.numel() * value.element_size() for value in values
    )


def test_cold_checkpoint_materialization_preserves_values_and_gradients():
    value = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = AdamWSwap([value], foreach=False)
    gradient = torch.ones_like(value)
    value.grad = gradient
    init_optim_state(optimizer)
    assert value.grad is gradient
    torch.testing.assert_close(value, torch.tensor([1.0, 2.0]), rtol=0, atol=0)
    assert optimizer.state[value]["step"].item() == 0
    assert not optimizer.state[value]["exp_avg"].count_nonzero().item()


def test_host_limit_fails_before_any_parameter_update():
    value = torch.nn.Parameter(torch.ones(16))
    optimizer = AdamWSwap([value], max_host_bytes=16, foreach=False)
    value.grad = torch.ones_like(value)
    with pytest.raises(MemoryError, match="max_host_bytes"):
        optimizer.step()
    assert not optimizer.state
    torch.testing.assert_close(value, torch.ones(16), rtol=0, atol=0)


@pytest.mark.parametrize("option", ["foreach", "capturable", "differentiable"])
def test_closure_cannot_enable_unsupported_update_mode(option):
    value = torch.nn.Parameter(torch.ones(3))
    optimizer = AdamWSwap([value], foreach=False)

    def closure():
        value.square().sum().backward()
        optimizer.param_groups[0][option] = True

    with pytest.raises(UnsupportedCombinationError, match=option):
        optimizer.step(closure)
    assert not optimizer.state
    torch.testing.assert_close(value, torch.ones(3), rtol=0, atol=0)


def test_invalid_restore_rejected_before_mutation():
    value = torch.nn.Parameter(torch.ones(3))
    optimizer = AdamWSwap([value], foreach=False)
    init_optim_state(optimizer)
    destination = optimizer.state[value]["exp_avg"]
    state = deepcopy(optimizer.state_dict())
    state["state"][0]["exp_avg"] = torch.ones(1)
    with pytest.raises(ValueError, match="shape/ownership"):
        optimizer.load_state_dict(state)
    assert optimizer.state[value]["exp_avg"] is destination
    optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    assert optimizer.state[value]["exp_avg"] is destination


def test_failed_update_blocks_step_and_save_until_restore(monkeypatch):
    value = torch.nn.Parameter(torch.ones(3))
    optimizer = AdamWSwap([value], foreach=False)
    init_optim_state(optimizer)
    state = deepcopy(optimizer.state_dict())
    value.grad = torch.ones_like(value)

    def fail(*args, **kwargs):
        raise RuntimeError("update failed")

    monkeypatch.setattr("llmtuner.components.optimizer.adamw_swap.adamw", fail)
    with pytest.raises(RuntimeError, match="update failed"):
        optimizer.step()
    with pytest.raises(RuntimeError, match="restore"):
        optimizer.state_dict()
    with pytest.raises(RuntimeError, match="restore"):
        optimizer.step()
    optimizer.load_state_dict(state)
    assert optimizer.state_dict()


def test_swap_config_requires_host_compatible_checkpoint_layout():
    with pytest.raises(ConfigError, match="independent"):
        OptimizerConfig(
            optimizer_type="AdamWSwap",
            implementation="for-loop",
            state_residency="npu_virtual",
        )
    for mode in ("disabled", "async", "async_with_pinned_mem"):
        LLMTunerConfig(
            optimizer=OptimizerConfig(
                optimizer_type="AdamWSwap", implementation="fused"
            ),
            training=TrainingConfig(
                checkpoint_config=CheckpointConfig(enable=True, async_mode=mode)
            ),
        )
    with pytest.raises(ConfigError, match="sharded_tensor_parallel"):
        LLMTunerConfig(
            optimizer=OptimizerConfig(
                optimizer_type="AdamWSwap", implementation="for-loop"
            ),
            parallel=ParallelConfig(tensor_parallel_size=2),
            training=TrainingConfig(checkpoint_config=CheckpointConfig(enable=True)),
        )
    LLMTunerConfig(
        optimizer=OptimizerConfig(optimizer_type="AdamWSwap", implementation="fused"),
        parallel=ParallelConfig(tensor_parallel_size=2),
        training=TrainingConfig(
            checkpoint_config=CheckpointConfig(
                enable=True, sharded_tensor_parallel=True
            )
        ),
    )


def test_empty_parameter_still_advances_counter_without_staging():
    value = torch.nn.Parameter(torch.empty(0))
    optimizer = AdamWSwap([value], foreach=False, max_device_bytes=8)
    value.grad = torch.empty_like(value)
    optimizer.step()
    assert optimizer.state[value]["step"].item() == 1
    assert optimizer.host_state_bytes == 0
    assert optimizer.staging_peak_payload_bytes == 0


def test_native_checkpoint_and_group_options_restore_before_next_update():
    value = torch.nn.Parameter(torch.ones(9))
    source = torch.optim.AdamW([value], lr=0.02, foreach=False)
    value.grad = torch.full_like(value, 0.25)
    source.step()
    target_value = torch.nn.Parameter(value.detach().clone())
    target = AdamWSwap([target_value], lr=0.5, foreach=False, max_device_bytes=16)
    checkpoint = deepcopy(source.state_dict())
    calls = []
    target.register_load_state_dict_pre_hook(lambda optim, state: calls.append("pre"))
    target.register_load_state_dict_post_hook(lambda optim: calls.append("post"))
    target.load_state_dict(checkpoint)
    assert calls == ["pre", "post"]
    assert target.param_groups[0]["lr"] == 0.02
    target_value.grad = value.grad.clone()
    target.step()
    source.step()
    torch.testing.assert_close(target_value, value, rtol=0, atol=0)
    torch.testing.assert_close(
        checkpoint["state"][0]["exp_avg"], torch.full_like(value, 0.025)
    )


def test_load_copy_failure_retains_owned_host_storage_and_can_retry(monkeypatch):
    value = torch.nn.Parameter(torch.ones(5))
    optimizer = AdamWSwap([value], foreach=False)
    init_optim_state(optimizer)
    saved = deepcopy(optimizer.state_dict())
    destination = optimizer.state[value]["exp_avg"]

    def fail_copy(*args, **kwargs):
        raise OSError("copy failed")

    monkeypatch.setattr(destination, "copy_", fail_copy)
    with pytest.raises(OSError, match="copy failed"):
        optimizer.load_state_dict(saved)
    assert optimizer.state[value]["exp_avg"] is destination
    with pytest.raises(RuntimeError, match="restore"):
        optimizer.step()
    monkeypatch.undo()
    optimizer.load_state_dict(saved)
    assert optimizer.state[value]["exp_avg"] is destination
    assert optimizer.state_dict()


@pytest.mark.parametrize("prefetch", [False, True])
def test_amsgrad_missing_gradient_and_reload(prefetch):
    value = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    reference = torch.nn.Parameter(value.detach().clone())
    swap = AdamWSwap([value], amsgrad=True, prefetch=prefetch, max_device_bytes=32)
    native = torch.optim.AdamW([reference], amsgrad=True, foreach=False)
    for index in range(4):
        value.grad = None if index == 1 else torch.full_like(value, index + 0.5)
        reference.grad = None if value.grad is None else value.grad.clone()
        swap.step()
        native.step()
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
        for name in ("step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            torch.testing.assert_close(
                swap.state[value][name], native.state[reference][name], rtol=0, atol=0
            )
        swap.load_state_dict(deepcopy(swap.state_dict()))
    assert swap.staging_peak_payload_bytes <= 32
    assert swap.host_state_bytes == 36


@pytest.mark.parametrize("fraction", [0, -1, 1.1, float("nan"), True])
def test_swap_memory_fraction_rejects_invalid_values(fraction):
    with pytest.raises(ValueError, match="memory_fraction"):
        AdamWSwap([torch.nn.Parameter(torch.ones(1))], memory_fraction=fraction)
    with pytest.raises(ConfigError, match="swap_memory_fraction"):
        OptimizerConfig(swap_memory_fraction=fraction)


def test_fused_capability_uses_optimizer_defining_module():
    from torch.optim.optimizer import _get_fused_kernels_supported_devices

    from llmtuner.components.optimizer.optimizer import fused_kernel_device_types

    assert fused_kernel_device_types() == frozenset(
        _get_fused_kernels_supported_devices()
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cpu_fused_slices_match_native_fused(dtype):
    value = torch.nn.Parameter(torch.linspace(0.2, 0.7, 13, dtype=dtype))
    reference = torch.nn.Parameter(value.detach().clone())
    swap = AdamWSwap([value], fused=True, prefetch=True, max_device_bytes=32)
    native = torch.optim.AdamW([reference], fused=True)
    for _ in range(3):
        value.grad = torch.ones_like(value)
        reference.grad = value.grad.clone()
        swap.step()
        native.step()
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
        for name in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                swap.state[value][name], native.state[reference][name], rtol=0, atol=0
            )


@pytest.mark.parametrize(
    "degree_name",
    [
        "pipeline_parallel_size",
        "context_parallel_ulysses_size",
        "tensor_parallel_size",
        "expert_parallel_size",
    ],
)
def test_swap_parallel_training_without_checkpoint_has_no_blanket_ban(degree_name):
    LLMTunerConfig(
        optimizer=OptimizerConfig(optimizer_type="AdamWSwap", implementation="fused"),
        parallel=ParallelConfig(**{degree_name: 2}),
        training=TrainingConfig(checkpoint_config=CheckpointConfig(enable=False)),
    )


def test_swap_expert_checkpoint_requires_chunks_and_accepts_async():
    optimizer = OptimizerConfig(optimizer_type="AdamWSwap", implementation="fused")
    with pytest.raises(ConfigError, match="sharded_expert_parallel"):
        LLMTunerConfig(
            optimizer=optimizer,
            parallel=ParallelConfig(expert_parallel_size=2),
            training=TrainingConfig(checkpoint_config=CheckpointConfig(enable=True)),
        )
    LLMTunerConfig(
        optimizer=optimizer,
        parallel=ParallelConfig(expert_parallel_size=2),
        training=TrainingConfig(
            checkpoint_config=CheckpointConfig(
                enable=True, sharded_expert_parallel=True, async_mode="async"
            )
        ),
    )


@pytest.mark.parametrize("amsgrad", [False, True])
def test_empty_shard_does_not_require_lookahead_capacity(amsgrad):
    value = torch.nn.Parameter(torch.empty(0, dtype=torch.float64))
    optimizer = AdamWSwap([value], prefetch=True, amsgrad=amsgrad, max_device_bytes=8)
    value.grad = torch.empty_like(value)
    for _ in range(2):
        optimizer.step()
    assert optimizer.state[value]["step"].item() == 2
    assert optimizer.staging_peak_payload_bytes == 0
    assert optimizer.host_state_bytes == 0
