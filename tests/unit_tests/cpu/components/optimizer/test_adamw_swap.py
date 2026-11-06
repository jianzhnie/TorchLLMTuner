"""Real CPU mathematical fallback and bounded explicit AdamW state exchange."""

import weakref
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


def test_loading_partial_checkpoint_preserves_lazy_state_access():
    values = [torch.nn.Parameter(torch.ones(3)) for _ in range(2)]
    references = [torch.nn.Parameter(value.detach().clone()) for value in values]
    optimizer = AdamWSwap(values, foreach=False)
    native = torch.optim.AdamW(references, foreach=False)
    values[0].grad = torch.ones_like(values[0])
    references[0].grad = torch.ones_like(references[0])
    optimizer.step()
    native.step()
    checkpoint = deepcopy(optimizer.state_dict())
    for _ in range(2):
        optimizer.load_state_dict(deepcopy(checkpoint))
        assert optimizer.state[values[1]] == native.state[references[1]] == {}
    values[0].grad = references[0].grad = None
    values[1].grad = torch.ones_like(values[1])
    references[1].grad = torch.ones_like(references[1])
    optimizer.step()
    native.step()
    for value, reference in zip(values, references, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
        for name in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                optimizer.state[value][name],
                native.state[reference][name],
                rtol=0,
                atol=0,
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


@pytest.mark.parametrize("invalid_field", ["moment_shape", "step"])
def test_later_parameter_validation_preserves_all_existing_state(invalid_field):
    values = [torch.nn.Parameter(torch.tensor([1.0, -2.0])) for _ in range(2)]
    optimizer = AdamWSwap(values, lr=0.02, foreach=False)
    for index, value in enumerate(values):
        value.grad = torch.full_like(value, index + 0.5)
    optimizer.step()
    before = deepcopy(optimizer.state_dict())
    destinations = [optimizer.state[value]["exp_avg"] for value in values]
    incoming = deepcopy(before)
    incoming["param_groups"][0]["lr"] = 0.9
    incoming["state"][0]["exp_avg"].fill_(123)
    if invalid_field == "moment_shape":
        incoming["state"][1]["exp_avg_sq"] = torch.ones(1)
    else:
        incoming["state"][1]["step"] = torch.tensor(float("inf"))

    with pytest.raises(ValueError):
        optimizer.load_state_dict(incoming)
    assert optimizer.param_groups[0]["lr"] == 0.02
    for index, value in enumerate(values):
        assert optimizer.state[value]["exp_avg"] is destinations[index]
        for name, expected in before["state"][index].items():
            torch.testing.assert_close(
                optimizer.state[value][name], expected, rtol=0, atol=0
            )
    # Validation failure leaves the optimizer usable without a repair load.
    optimizer.step()
    assert all(optimizer.state[value]["step"].item() == 2 for value in values)


def test_restore_preserves_fp64_counter_beyond_fp32_integer_precision():
    reference = torch.nn.Parameter(torch.tensor([0.5, -2.0]))
    native = torch.optim.AdamW([reference], foreach=False)
    reference.grad = torch.tensor([0.25, -0.75])
    native.step()
    native.state[reference]["step"] = torch.tensor(2**24 + 1, dtype=torch.float64)
    saved = deepcopy(native.state_dict())
    value = torch.nn.Parameter(reference.detach().clone())
    optimizer = AdamWSwap([value], foreach=False)
    optimizer.load_state_dict(saved)
    assert optimizer.state[value]["step"].dtype == torch.float64
    assert optimizer.state[value]["step"].item() == 2**24 + 1
    value.grad = reference.grad.clone()
    optimizer.step()
    native.step()
    assert optimizer.state[value]["step"].item() == 2**24 + 2
    torch.testing.assert_close(value, reference, rtol=0, atol=0)
    for name, expected in native.state[reference].items():
        torch.testing.assert_close(
            optimizer.state[value][name], expected, rtol=0, atol=0
        )


@pytest.mark.parametrize("counter", [0.5, torch.tensor(1.5)])
def test_fractional_checkpoint_counter_rejected_before_state_mutation(counter):
    value = torch.nn.Parameter(torch.ones(3))
    optimizer = AdamWSwap([value], foreach=False)
    value.grad = torch.ones_like(value)
    optimizer.step()
    before = deepcopy(optimizer.state_dict())
    incoming = deepcopy(before)
    incoming["state"][0]["step"] = counter
    incoming["state"][0]["exp_avg"].fill_(99)
    with pytest.raises(ValueError, match="nonnegative integer"):
        optimizer.load_state_dict(incoming)
    for name, expected in before["state"][0].items():
        torch.testing.assert_close(
            optimizer.state[value][name], expected, rtol=0, atol=0
        )
    optimizer.step()
    assert optimizer.state[value]["step"].item() == 2


@pytest.mark.parametrize("storage", ["meta", "sparse"])
def test_invalid_moment_storage_rejected_before_state_mutation(storage):
    value = torch.nn.Parameter(torch.ones(3))
    optimizer = AdamWSwap([value], foreach=False)
    value.grad = torch.ones_like(value)
    optimizer.step()
    before = deepcopy(optimizer.state_dict())
    incoming = deepcopy(before)
    incoming["state"][0]["exp_avg"].fill_(99)
    moment = incoming["state"][0]["exp_avg_sq"]
    incoming["state"][0]["exp_avg_sq"] = (
        moment.to("meta") if storage == "meta" else moment.to_sparse()
    )
    with pytest.raises(ValueError, match="moment shape/ownership mismatch"):
        optimizer.load_state_dict(incoming)
    for name, expected in before["state"][0].items():
        torch.testing.assert_close(
            optimizer.state[value][name], expected, rtol=0, atol=0
        )
    optimizer.step()
    assert optimizer.state[value]["step"].item() == 2


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


def test_nonprefetch_releases_previous_staging_before_allocating_next(monkeypatch):
    value = torch.nn.Parameter(torch.ones(8))
    value.grad = torch.full_like(value, 0.25)
    optimizer = AdamWSwap([value], max_device_bytes=16, prefetch=False)
    stage = optimizer._stage
    previous = []
    slices = []

    def checked_stage(job):
        assert all(reference() is None for reference in previous)
        result = stage(job)
        # CPU .to(cpu) aliases Host state; clone to model separately owned
        # device staging buffers whose lifetime is controlled by step().
        buffers = [buffer.clone() for buffer in result[0]]
        previous[:] = [weakref.ref(buffer) for buffer in buffers]
        slices.append(job[2:4])
        return (buffers, *result[1:])

    monkeypatch.setattr(optimizer, "_stage", checked_stage)
    optimizer.step()
    assert len(slices) > 1
    assert all(reference() is None for reference in previous)


def test_update_error_survives_copy_stream_cleanup_failure(monkeypatch):
    update_error = RuntimeError("update failed")
    cleanup_error = RuntimeError("copy stream failed")

    class FailedStream:
        def synchronize(self):
            raise cleanup_error

    def fail_update(*args, **kwargs):
        raise update_error

    value = torch.nn.Parameter(torch.ones(4))
    value.grad = torch.ones_like(value)
    optimizer = AdamWSwap([value])
    optimizer._copy_stream = FailedStream()
    monkeypatch.setattr(optimizer, "_update_slice", fail_update)
    with pytest.raises(RuntimeError) as caught:
        optimizer.step()
    assert caught.value is update_error
    assert caught.value.__cause__ is cleanup_error
    with pytest.raises(RuntimeError, match="restore a complete training checkpoint"):
        optimizer.state_dict()
    with pytest.raises(RuntimeError, match="restore a complete training checkpoint"):
        optimizer.step()


def test_copy_stream_synchronization_failure_invalidates_optimizer():
    class FailedStream:
        def synchronize(self):
            raise RuntimeError("copy stream failed")

    value = torch.nn.Parameter(torch.ones(4))
    value.grad = torch.ones_like(value)
    optimizer = AdamWSwap([value])
    optimizer.step()
    saved_parameter = value.detach().clone()
    saved_state = deepcopy(optimizer.state_dict())
    optimizer._copy_stream = FailedStream()
    with pytest.raises(RuntimeError, match="copy stream failed"):
        optimizer.step()
    with pytest.raises(RuntimeError, match="restore a complete training checkpoint"):
        optimizer.state_dict()
    with pytest.raises(RuntimeError, match="restore a complete training checkpoint"):
        optimizer.step()
    with pytest.raises(RuntimeError, match="restore a complete training checkpoint"):
        optimizer.initialize_state()

    # A real device error can require process restart. Here only the injected
    # stream fails; remove it and restore both halves of the training state.
    optimizer._copy_stream = None
    with torch.no_grad():
        value.copy_(saved_parameter)
    optimizer.load_state_dict(saved_state)
    reference = torch.nn.Parameter(saved_parameter.clone())
    native = torch.optim.AdamW([reference], foreach=False)
    native.load_state_dict(deepcopy(saved_state))
    reference.grad = value.grad.clone()
    optimizer.step()
    native.step()
    torch.testing.assert_close(value, reference, rtol=0, atol=0)
    for name, observed in optimizer.state[value].items():
        torch.testing.assert_close(
            observed, native.state[reference][name], rtol=0, atol=0
        )


@pytest.mark.parametrize("separate_groups", [False, True])
def test_duplicate_checkpoint_parameter_ids_rejected_before_mutation(separate_groups):
    values = [torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(3))]
    groups = [{"params": [p]} for p in values] if separate_groups else values
    optimizer = AdamWSwap(groups, lr=0.02, foreach=False)
    for index, value in enumerate(values):
        value.grad = torch.full_like(value, index + 0.25)
    optimizer.step()
    saved = deepcopy(optimizer.state_dict())
    malformed = deepcopy(saved)
    if separate_groups:
        malformed["param_groups"][1]["params"] = [0]
    else:
        malformed["param_groups"][0]["params"] = [0, 0]
    malformed["state"].pop(1)
    malformed["param_groups"][0]["lr"] = 0.5
    with pytest.raises(ValueError, match="duplicate parameter"):
        optimizer.load_state_dict(malformed)
    assert optimizer.param_groups[0]["lr"] == 0.02
    for index, value in enumerate(values):
        for name, expected in saved["state"][index].items():
            torch.testing.assert_close(
                optimizer.state[value][name], expected, rtol=0, atol=0
            )
    optimizer.load_state_dict(saved)
    optimizer.step()
    assert all(optimizer.state[p]["step"].item() == 2 for p in values)
    references = [torch.nn.Parameter(torch.ones(3)) for _ in values]
    reference_groups = (
        [{"params": [p]} for p in references] if separate_groups else references
    )
    native = torch.optim.AdamW(reference_groups, lr=0.02, foreach=False)
    for _ in range(2):
        for index, parameter in enumerate(references):
            parameter.grad = torch.full_like(parameter, index + 0.25)
        native.step()
    for value, reference in zip(values, references, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
        for name, expected in native.state[reference].items():
            torch.testing.assert_close(
                optimizer.state[value][name], expected, rtol=0, atol=0
            )


@pytest.mark.parametrize("source_amsgrad", [False, True])
@pytest.mark.parametrize("destination_amsgrad", [False, True])
def test_reload_amsgrad_policy_reuses_host_moments_and_matches_native(
    source_amsgrad, destination_amsgrad
):
    reference = torch.nn.Parameter(torch.tensor([0.5, -1.0, 2.0]))
    native = torch.optim.AdamW(
        [reference], lr=0.02, amsgrad=source_amsgrad, foreach=False
    )
    reference.grad = torch.tensor([0.25, -0.5, 1.0])
    native.step()
    saved = deepcopy(native.state_dict())
    value = torch.nn.Parameter(reference.detach().clone())
    optimizer = AdamWSwap(
        [value], amsgrad=destination_amsgrad, max_host_bytes=36, foreach=False
    )
    optimizer.initialize_state()
    old = dict(optimizer.state[value])
    optimizer.load_state_dict(saved)
    assert optimizer.param_groups[0]["amsgrad"] is source_amsgrad
    assert optimizer.host_state_bytes == (36 if source_amsgrad else 24)
    assert optimizer.state[value]["exp_avg"] is old["exp_avg"]
    assert optimizer.state[value]["exp_avg_sq"] is old["exp_avg_sq"]
    if source_amsgrad and destination_amsgrad:
        assert optimizer.state[value]["max_exp_avg_sq"] is old["max_exp_avg_sq"]
    elif not source_amsgrad:
        assert "max_exp_avg_sq" not in optimizer.state[value]
    owned = dict(optimizer.state[value])
    optimizer.load_state_dict(saved)
    for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        if name in owned:
            assert optimizer.state[value][name] is owned[name]
    value.grad = reference.grad.clone()
    optimizer.step()
    native.step()
    torch.testing.assert_close(value, reference, rtol=0, atol=0)
    for name, expected in native.state[reference].items():
        torch.testing.assert_close(
            optimizer.state[value][name], expected, rtol=0, atol=0
        )
