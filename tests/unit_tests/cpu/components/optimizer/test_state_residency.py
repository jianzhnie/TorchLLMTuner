"""Storage policies for Adam moments and their checkpoint boundaries."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.tensor import DTensor, Replicate

from llmtuner.components.checkpointer.backends.distributed_checkpoint import (
    DistributedCheckpointAsyncMode,
    DistributedCheckpointManager,
)
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.components.optimizer.state_dict import init_optim_state
from llmtuner.components.optimizer.state_residency import (
    _allocate_moment,
    install_state_residency,
    validate_checkpoint_configuration,
)
from llmtuner.config import (
    FSDPConfig,
    LLMTunerConfig,
    OptimizerConfig,
    ParallelConfig,
    ParamGroupConfig,
)
from llmtuner.errors import ConfigError, UnsupportedCombinationError
from llmtuner.trainer.runtime.assembly import _build_checkpointer
from tests.caps import require_env


def _allocator(calls):
    def allocate(shape, *, dtype, device):
        calls.append((tuple(shape), dtype, device))
        return torch.empty(shape, dtype=dtype, device=device)

    return allocate


def test_virtual_moments_match_device_updates_and_preserve_existing_state(monkeypatch):
    monkeypatch.setattr(
        "llmtuner.components.optimizer.state_residency.device_type", "npu"
    )
    calls = []
    model = torch.nn.Linear(3, 2, bias=False)
    reference = torch.nn.Linear(3, 2, bias=False)
    reference.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
    expected = torch.optim.AdamW(reference.parameters(), lr=0.01, foreach=False)
    install_state_residency(optimizer, "npu_virtual", allocator=_allocator(calls))

    for value in (0.25, -0.5):
        model.weight.grad = torch.full_like(model.weight, value)
        reference.weight.grad = model.weight.grad.clone()
        optimizer.step()
        expected.step()

    torch.testing.assert_close(model.weight, reference.weight, rtol=0, atol=0)
    state = optimizer.state[model.weight]
    torch.testing.assert_close(
        state["exp_avg"], expected.state[reference.weight]["exp_avg"]
    )
    assert len(calls) == 2


@pytest.mark.parametrize(
    "default_dtype", [torch.bfloat16, torch.float32, torch.float64]
)
def test_virtual_step_counter_matches_adam_beyond_bf16_integer_range(
    monkeypatch, default_dtype
):
    monkeypatch.setattr(
        "llmtuner.components.optimizer.state_residency.device_type", "npu"
    )
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
        reference = torch.nn.Parameter(parameter.detach().clone())
        optimizer = torch.optim.AdamW([parameter], lr=0.01, foreach=False)
        expected = torch.optim.AdamW([reference], lr=0.01, foreach=False)
        install_state_residency(optimizer, "npu_virtual", allocator=_allocator([]))
        for _ in range(300):
            parameter.grad = torch.full_like(parameter, 0.25)
            reference.grad = parameter.grad.clone()
            optimizer.step()
            expected.step()
        actual_step = optimizer.state[parameter]["step"]
        expected_step = expected.state[reference]["step"]
        assert actual_step.dtype == expected_step.dtype
        assert actual_step.item() == expected_step.item() == 300
        torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize("counter_dtype", [torch.bfloat16, torch.float16])
def test_virtual_load_promotes_legacy_step_counter_without_mutating_checkpoint(
    monkeypatch, counter_dtype
):
    monkeypatch.setattr(
        "llmtuner.components.optimizer.state_residency.device_type", "npu"
    )
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter], foreach=False)
    install_state_residency(optimizer, "npu_virtual", allocator=_allocator([]))
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    checkpoint = deepcopy(optimizer.state_dict())
    legacy_step = torch.tensor(256, dtype=counter_dtype)
    checkpoint["state"][0]["step"] = legacy_step

    optimizer.load_state_dict(checkpoint)
    assert optimizer.state[parameter]["step"].dtype == torch.float32
    for _ in range(44):
        optimizer.step()
    assert optimizer.state[parameter]["step"].item() == 300
    assert checkpoint["state"][0]["step"] is legacy_step
    assert legacy_step.item() == 256


def test_virtual_moments_keep_swap_destinations_after_load(monkeypatch):
    monkeypatch.setattr(
        "llmtuner.components.optimizer.state_residency.device_type", "npu"
    )
    source_calls = []
    source_parameter = torch.nn.Parameter(torch.ones(2))
    source = torch.optim.AdamW([source_parameter], lr=0.01, foreach=False)
    install_state_residency(source, "npu_virtual", allocator=_allocator(source_calls))
    source_parameter.grad = torch.full_like(source_parameter, 0.5)
    source.step()
    checkpoint = deepcopy(source.state_dict())

    target_calls = []
    target_parameter = torch.nn.Parameter(torch.ones(2))
    target = torch.optim.AdamW([target_parameter], lr=0.01, foreach=False)
    install_state_residency(target, "npu_virtual", allocator=_allocator(target_calls))
    target.load_state_dict(checkpoint)

    restored_exp_avg = target.state[target_parameter]["exp_avg"]
    restored_exp_avg_sq = target.state[target_parameter]["exp_avg_sq"]
    assert len(target_calls) == 2
    assert restored_exp_avg is not checkpoint["state"][0]["exp_avg"]
    torch.testing.assert_close(
        restored_exp_avg, checkpoint["state"][0]["exp_avg"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        restored_exp_avg_sq,
        checkpoint["state"][0]["exp_avg_sq"],
        rtol=0,
        atol=0,
    )

    target.load_state_dict(deepcopy(checkpoint))
    assert target.state[target_parameter]["exp_avg"] is restored_exp_avg
    assert target.state[target_parameter]["exp_avg_sq"] is restored_exp_avg_sq
    assert len(target_calls) == 2


@pytest.mark.parametrize("option", ["amsgrad", "differentiable", "capturable", "fused"])
def test_virtual_load_rejects_unsupported_checkpoint_options(monkeypatch, option):
    monkeypatch.setattr(
        "llmtuner.components.optimizer.state_residency.device_type", "npu"
    )
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter], foreach=False)
    calls = []
    install_state_residency(optimizer, "npu_virtual", allocator=_allocator(calls))
    checkpoint = deepcopy(optimizer.state_dict())
    checkpoint["param_groups"][0][option] = True

    with pytest.raises(UnsupportedCombinationError, match=option):
        optimizer.load_state_dict(checkpoint)

    assert not calls
    assert not optimizer.state
    assert not optimizer.param_groups[0].get(option, False)
    # A rejected load must leave the optimizer usable for a valid retry.
    optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert len(calls) == 4
    assert torch.isfinite(parameter).all()


def test_container_validates_virtual_options_before_loading_any_optimizer(monkeypatch):
    monkeypatch.setattr(
        "llmtuner.components.optimizer.state_residency.device_type", "npu"
    )
    model = torch.nn.Linear(2, 1)
    config = OptimizerConfig(
        implementation="for-loop",
        param_groups=[
            ParamGroupConfig(pattern="weight$", optimizer_name="Adam"),
            ParamGroupConfig(pattern="bias$", optimizer_name="AdamW"),
        ],
    )
    container = OptimizersContainer(config, model_parts=[model])
    container.state_residency = "npu_virtual"
    for optimizer in container.optimizers:
        install_state_residency(optimizer, "npu_virtual", allocator=_allocator([]))
    checkpoint = deepcopy(container.state_dict())
    first, second = container.optimizers
    first_lr = first.param_groups[0]["lr"]
    first_name = first.param_groups[0]["param_names"][0]
    second_name = second.param_groups[0]["param_names"][0]
    checkpoint[f"param_groups.{first_name}.lr"] = first_lr + 1
    checkpoint[f"param_groups.{second_name}.fused"] = True

    with pytest.raises(UnsupportedCombinationError, match="fused"):
        container.load_state_dict(checkpoint)

    assert first.param_groups[0]["lr"] == first_lr


def test_checkpoint_state_materialization_uses_virtual_allocator(monkeypatch):
    monkeypatch.setattr(
        "llmtuner.components.optimizer.state_residency.device_type", "npu"
    )
    calls = []
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter], lr=0.01, foreach=False)
    install_state_residency(optimizer, "npu_virtual", allocator=_allocator(calls))

    init_optim_state(optimizer)

    assert len(calls) == 2
    assert optimizer.state[parameter]["step"] == 0
    assert parameter.grad is None
    torch.testing.assert_close(parameter, torch.ones(2), rtol=0, atol=0)


def test_dtensor_uses_local_allocation_and_zero_shard_fallback():
    mesh = torch.distributed.device_mesh.DeviceMesh(
        "cpu", [0], mesh_dim_names=("dp",), _init_backend=False, _rank=0
    )
    parameter = DTensor.from_local(
        torch.ones(2, 3), mesh, [Replicate()], run_check=False
    )
    calls = []
    moment = _allocate_moment(parameter, _allocator(calls))
    assert isinstance(moment, DTensor)
    assert moment.device_mesh is mesh
    assert moment.placements == parameter.placements
    assert moment.shape == parameter.shape
    assert len(calls) == 1

    empty = DTensor.from_local(torch.empty(0, 3), mesh, [Replicate()], run_check=False)
    fallback = _allocate_moment(
        empty,
        lambda *args, **kwargs: pytest.fail("zero shard must not use swap allocator"),
    )
    assert isinstance(fallback, DTensor)
    assert fallback.to_local().shape == empty.to_local().shape


def test_virtual_residency_rejects_unsupported_environment_and_options(monkeypatch):
    monkeypatch.setattr(
        "llmtuner.components.optimizer.state_residency.device_type", "cpu"
    )
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    with pytest.raises(Exception, match="requires Ascend NPU"):
        install_state_residency(optimizer, "npu_virtual")

    monkeypatch.setattr(
        "llmtuner.components.optimizer.state_residency.device_type", "npu"
    )
    for option in ("amsgrad", "differentiable", "capturable", "fused"):
        optimizer = torch.optim.AdamW(
            [torch.nn.Parameter(torch.ones(1))], **{option: True}
        )
        with pytest.raises(UnsupportedCombinationError, match=option):
            install_state_residency(optimizer, "npu_virtual")


def test_virtual_residency_reaches_each_inner_optimizer(monkeypatch):
    require_env("torch_npu")
    monkeypatch.setattr(
        "llmtuner.components.optimizer.state_residency.device_type", "npu"
    )
    monkeypatch.setattr(
        "torch_npu.empty_with_swapped_memory", _allocator([]), raising=False
    )
    model = torch.nn.Linear(2, 1)
    config = OptimizerConfig(state_residency="npu_virtual", implementation="for-loop")
    container = OptimizersContainer(config, model_parts=[model])
    model(torch.ones(1, 2)).sum().backward()
    container.step()
    assert container.state_residency == "npu_virtual"
    assert container.optimizers[0].state


def test_state_residency_config_and_checkpoint_combinations():
    with pytest.raises(ConfigError, match="state_residency"):
        OptimizerConfig(state_residency="invalid")
    with pytest.raises(UnsupportedCombinationError, match="requires"):
        validate_checkpoint_configuration("torch_checkpointing", "disabled")
    with pytest.raises(UnsupportedCombinationError, match="async_mode='invalid'"):
        validate_checkpoint_configuration("dcp", "invalid")
    for async_mode in ("disabled", "async", "async_with_pinned_mem"):
        validate_checkpoint_configuration("dcp", async_mode)


def test_virtual_residency_rejects_fsdp_cpu_offload():
    config = OptimizerConfig(state_residency="npu_virtual")
    parallel = ParallelConfig(fsdp_config=FSDPConfig(fsdp_cpu_offload=True))

    with pytest.raises(ConfigError, match="incompatible with.*fsdp_cpu_offload"):
        LLMTunerConfig(optimizer=config, parallel=parallel)


def test_disabled_checkpoint_does_not_validate_virtual_checkpoint_options():
    trainer = SimpleNamespace(
        cfg=SimpleNamespace(
            optimizer=SimpleNamespace(state_residency="npu_virtual"),
            checkpoint=SimpleNamespace(
                enable=False, backend="torch_checkpointing", async_mode="async"
            ),
        ),
        dataloader=None,
    )
    with pytest.raises(RuntimeError, match="dataloader must be initialized"):
        _build_checkpointer(trainer, None)


def test_virtual_checkpoint_writer_disables_copy_ahead(monkeypatch, tmp_path):
    captured = {}

    def save(state_dict, *, storage_writer, checkpoint_id):
        captured["writer"] = storage_writer
        return "saved"

    monkeypatch.setattr(
        "llmtuner.components.checkpointer.backends.distributed_checkpoint.dcp.save",
        save,
    )
    manager = DistributedCheckpointManager.__new__(DistributedCheckpointManager)
    manager.states = {"optimizer": SimpleNamespace(state_residency="npu_virtual")}
    result = manager._save_checkpoint(
        {"weight": torch.ones(1)},
        str(tmp_path / "checkpoint"),
        async_mode=DistributedCheckpointAsyncMode.DISABLED,
    )
    assert result == "saved"
    assert isinstance(captured["writer"], FileSystemWriter)
    assert captured["writer"].per_thread_copy_ahead == 0


def test_failed_virtual_allocation_can_retry_without_partial_adam_state(monkeypatch):
    monkeypatch.setattr(
        "llmtuner.components.optimizer.state_residency.device_type", "npu"
    )
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter], foreach=False)
    calls = 0

    def allocate(shape, *, dtype, device):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("moment allocation failed")
        return torch.empty(shape, dtype=dtype, device=device)

    install_state_residency(optimizer, "npu_virtual", allocator=allocate)
    parameter.grad = torch.ones_like(parameter)
    with pytest.raises(RuntimeError, match="moment allocation failed"):
        optimizer.step()
    assert not optimizer.state[parameter]
    torch.testing.assert_close(parameter, torch.ones(2), rtol=0, atol=0)
    optimizer.step()
    assert optimizer.state[parameter]["step"].item() == 1
    assert calls == 4


@pytest.mark.parametrize("failed_moment", ["exp_avg", "exp_avg_sq"])
def test_failed_load_copy_keeps_virtual_destinations_for_retry(
    monkeypatch, failed_moment
):
    monkeypatch.setattr(
        "llmtuner.components.optimizer.state_residency.device_type", "npu"
    )
    parameters = [torch.nn.Parameter(torch.ones(2)) for _ in range(2)]
    optimizer = torch.optim.AdamW(
        [{"params": [parameter]} for parameter in parameters], lr=0.01, foreach=False
    )
    calls = []
    install_state_residency(optimizer, "npu_virtual", allocator=_allocator(calls))
    for index, parameter in enumerate(parameters):
        parameter.grad = torch.tensor([0.25 + index, -0.5 - index])
    optimizer.step()
    snapshot = deepcopy(optimizer.state_dict())
    destinations = [
        (index, parameter, name, optimizer.state[parameter][name])
        for index, parameter in enumerate(parameters)
        for name in ("exp_avg", "exp_avg_sq")
    ]
    failing = optimizer.state[parameters[0]][failed_moment]
    original_copy = failing.copy_
    failure = OSError("restore moment copy failed")

    def fail_copy(source):
        raise failure

    monkeypatch.setattr(failing, "copy_", fail_copy)
    with pytest.raises(OSError) as caught:
        optimizer.load_state_dict(deepcopy(snapshot))
    assert caught.value is failure
    for _, parameter, name, destination in destinations:
        assert optimizer.state[parameter][name] is destination

    monkeypatch.setattr(failing, "copy_", original_copy)
    optimizer.load_state_dict(deepcopy(snapshot))
    assert len(calls) == 4
    for index, parameter, name, destination in destinations:
        assert optimizer.state[parameter][name] is destination
        torch.testing.assert_close(
            destination, snapshot["state"][index][name], rtol=0, atol=0
        )

    reference_parameters = [
        torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters
    ]
    reference = torch.optim.AdamW(
        [{"params": [parameter]} for parameter in reference_parameters],
        lr=0.01,
        foreach=False,
    )
    reference.load_state_dict(deepcopy(snapshot))
    for index, (parameter, reference_parameter) in enumerate(
        zip(parameters, reference_parameters, strict=True)
    ):
        parameter.grad = torch.tensor([-0.75 - index, 0.125 + index])
        reference_parameter.grad = parameter.grad.clone()
    optimizer.step()
    reference.step()
    for parameter, reference_parameter in zip(
        parameters, reference_parameters, strict=True
    ):
        torch.testing.assert_close(parameter, reference_parameter, rtol=0, atol=0)


@pytest.mark.parametrize("keyword", [False, True])
def test_virtual_moments_created_by_closure_use_allocator(monkeypatch, keyword):
    monkeypatch.setattr(
        "llmtuner.components.optimizer.state_residency.device_type", "npu"
    )
    parameter = torch.nn.Parameter(torch.tensor([1.0, -0.5]))
    reference = torch.nn.Parameter(parameter.detach().clone())
    optimizer = torch.optim.AdamW([parameter], lr=0.01, foreach=False)
    expected = torch.optim.AdamW([reference], lr=0.01, foreach=False)
    calls = []
    install_state_residency(optimizer, "npu_virtual", allocator=_allocator(calls))

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = parameter.square().sum()
        loss.backward()
        return loss

    def reference_closure():
        expected.zero_grad(set_to_none=True)
        loss = reference.square().sum()
        loss.backward()
        return loss

    for _ in range(2):
        loss = optimizer.step(closure=closure) if keyword else optimizer.step(closure)
        expected_loss = expected.step(reference_closure)
        torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
        torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
    assert len(calls) == 2


@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize("option", ["amsgrad", "capturable", "differentiable", "fused"])
def test_virtual_revalidates_groups_after_closure(monkeypatch, keyword, option):
    monkeypatch.setattr(
        "llmtuner.components.optimizer.state_residency.device_type", "npu"
    )
    parameter = torch.nn.Parameter(torch.tensor([1.0, -0.5]))
    initial = parameter.detach().clone()
    optimizer = torch.optim.AdamW([parameter], foreach=False)
    calls = []
    install_state_residency(optimizer, "npu_virtual", allocator=_allocator(calls))

    def closure():
        parameter.square().sum().backward()
        optimizer.param_groups[0][option] = True

    with pytest.raises(UnsupportedCombinationError, match=option):
        if keyword:
            optimizer.step(closure=closure)
        else:
            optimizer.step(closure)
    assert not calls
    assert not optimizer.state
    torch.testing.assert_close(parameter, initial, rtol=0, atol=0)


def test_virtual_revalidates_groups_before_step(monkeypatch):
    monkeypatch.setattr(
        "llmtuner.components.optimizer.state_residency.device_type", "npu"
    )
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter])
    calls = []
    install_state_residency(optimizer, "npu_virtual", allocator=_allocator(calls))
    optimizer.param_groups[0]["amsgrad"] = True
    parameter.grad = torch.ones_like(parameter)
    with pytest.raises(UnsupportedCombinationError, match="amsgrad"):
        optimizer.step()
    assert not calls
    assert not optimizer.state


def test_virtual_rejects_non_adam_optimizer(monkeypatch):
    monkeypatch.setattr(
        "llmtuner.components.optimizer.state_residency.device_type", "npu"
    )
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(2))], lr=0.1)
    with pytest.raises(UnsupportedCombinationError, match="Adam/AdamW only"):
        install_state_residency(optimizer, "npu_virtual")
    assert not optimizer._optimizer_step_pre_hooks


@pytest.mark.parametrize("mismatch", ["group_count", "parameter_count"])
def test_invalid_virtual_load_rejects_layout_before_allocation(monkeypatch, mismatch):
    monkeypatch.setattr(
        "llmtuner.components.optimizer.state_residency.device_type", "npu"
    )
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter], foreach=False)
    calls = []
    install_state_residency(optimizer, "npu_virtual", allocator=_allocator(calls))
    checkpoint = deepcopy(optimizer.state_dict())
    if mismatch == "group_count":
        checkpoint["param_groups"].append(deepcopy(checkpoint["param_groups"][0]))
    else:
        checkpoint["param_groups"][0]["params"].append(1)

    with pytest.raises(ValueError, match="parameter group"):
        optimizer.load_state_dict(checkpoint)
    assert not calls
    assert not optimizer.state
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert len(calls) == 2
    assert optimizer.state[parameter]["step"].item() == 1


@pytest.mark.parametrize("invalid", ["broadcast", "missing_moment", "step"])
def test_virtual_load_rejects_malformed_state_before_host_allocation(
    monkeypatch, invalid
):
    monkeypatch.setattr(
        "llmtuner.components.optimizer.state_residency.device_type", "npu"
    )
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    source = torch.optim.AdamW([parameter], foreach=False)
    parameter.grad = torch.ones_like(parameter)
    source.step()
    checkpoint = deepcopy(source.state_dict())
    target_parameter = torch.nn.Parameter(parameter.detach().clone())
    target = torch.optim.AdamW([target_parameter], foreach=False)
    calls = []
    install_state_residency(target, "npu_virtual", allocator=_allocator(calls))
    malformed = deepcopy(checkpoint)
    if invalid == "broadcast":
        malformed["state"][0]["exp_avg"] = torch.ones(1, 3)
    elif invalid == "missing_moment":
        del malformed["state"][0]["exp_avg_sq"]
    else:
        malformed["state"][0]["step"] = torch.ones(2)
    with pytest.raises(ValueError, match="Invalid Adam"):
        target.load_state_dict(malformed)
    assert not calls
    assert not target.state
    target.load_state_dict(checkpoint)
    assert len(calls) == 2
    parameter.grad = torch.full_like(parameter, 0.25)
    target_parameter.grad = parameter.grad.clone()
    source.step()
    target.step()
    torch.testing.assert_close(target_parameter, parameter, rtol=0, atol=0)


@pytest.mark.parametrize("counter", [float("nan"), float("inf"), -1, True, "1"])
def test_virtual_load_rejects_invalid_step_before_host_allocation(monkeypatch, counter):
    monkeypatch.setattr(
        "llmtuner.components.optimizer.state_residency.device_type", "npu"
    )
    value = torch.nn.Parameter(torch.ones(2))
    source = torch.optim.AdamW([value], foreach=False)
    value.grad = torch.ones_like(value)
    source.step()
    checkpoint = deepcopy(source.state_dict())
    checkpoint["state"][0]["step"] = counter
    target_value = torch.nn.Parameter(torch.ones(2))
    target = torch.optim.AdamW([target_value], foreach=False)
    calls = []
    install_state_residency(target, "npu_virtual", allocator=_allocator(calls))
    with pytest.raises(ValueError, match="Invalid Adam step"):
        target.load_state_dict(checkpoint)
    assert not calls
    assert not target.state
