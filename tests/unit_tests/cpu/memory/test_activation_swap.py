import gc
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import llmtuner.memory.activation_swap as activation_swap
from llmtuner.errors import EnvironmentUnsupportedError


class _Stream:
    def __init__(self):
        self.synchronized = False

    def wait_stream(self, stream):
        pass

    def synchronize(self):
        self.synchronized = True


@pytest.fixture
def cpu_copies(monkeypatch):
    empty_like = torch.empty_like

    def cpu_empty_like(tensor, **kwargs):
        # Exercise ownership and autograd on CPU without a device allocator.
        # Real pinned allocations are covered by npu_activation_swap.py.
        kwargs.pop("pin_memory", None)
        return empty_like(tensor, **kwargs)

    current = _Stream()
    accelerator = SimpleNamespace(
        Stream=_Stream,
        current_stream=lambda: current,
        stream=lambda stream: nullcontext(),
    )
    monkeypatch.setattr(activation_swap, "_stream_api", lambda: accelerator)
    monkeypatch.setattr(activation_swap, "_should_swap", lambda tensor: True)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)
    monkeypatch.setattr(torch, "empty_like", cpu_empty_like)


def test_swap_releases_reference_without_mutating_storage(cpu_copies):
    state = activation_swap._ForwardSwapState()
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    expected = tensor.clone()

    packed = state.pack(tensor)
    state.release(outputs=())

    assert packed.released
    assert packed.device_tensor is None
    assert torch.equal(tensor, expected)
    restored = activation_swap._unpack(packed)
    assert torch.equal(restored, expected)
    assert tensor.untyped_storage().nbytes() == expected.untyped_storage().nbytes()
    assert torch.equal(tensor, expected)
    assert packed.host_tensor is None


def test_transfer_metrics_track_bytes_and_host_peak(cpu_copies):
    activation_swap.reset_activation_swap_metrics()
    state = activation_swap._ForwardSwapState()
    packed = state.pack(torch.ones(2, 3))
    state.release(outputs=())
    activation_swap._unpack(packed)
    del packed, state
    gc.collect()

    metrics = activation_swap.activation_swap_metrics()
    assert metrics["d2h_bytes"] == metrics["h2d_bytes"] == 24
    assert metrics["host_bytes_peak"] == 24
    assert metrics["host_bytes_current"] == 0


def test_dropping_forward_graph_releases_pending_host_bytes(cpu_copies):
    activation_swap.reset_activation_swap_metrics()
    baseline = activation_swap.activation_swap_metrics()["host_bytes_current"]
    state = activation_swap._ForwardSwapState()
    packed = state.pack(torch.ones(2, 3))
    state.release(outputs=())

    assert (
        activation_swap.activation_swap_metrics()["host_bytes_current"] == baseline + 24
    )

    del packed, state
    gc.collect()

    assert activation_swap.activation_swap_metrics()["host_bytes_current"] == baseline


def test_transfer_metrics_release_timing_events_after_read(cpu_copies):
    activation_swap.reset_activation_swap_metrics()
    activation_swap._d2h_events.append((None, None))
    activation_swap._h2d_events.append((None, None))

    first = activation_swap.activation_swap_metrics()
    second = activation_swap.activation_swap_metrics()

    assert first["copies_d2h"] == second["copies_d2h"] == 0
    assert first["d2h_seconds"] == first["h2d_seconds"] == 0
    assert activation_swap._d2h_events == activation_swap._h2d_events == []


def test_packed_copy_does_not_retain_autograd_graph(cpu_copies):
    state = activation_swap._ForwardSwapState()
    tensor = torch.ones(2, 3, requires_grad=True) * 2
    packed = state.pack(tensor)

    assert not packed.host_tensor.requires_grad
    assert packed.host_tensor.grad_fn is None
    assert packed.device_tensor.grad_fn is None


def test_forward_output_is_not_released(cpu_copies):
    state = activation_swap._ForwardSwapState()
    tensor = torch.ones(2, 3)
    packed = state.pack(tensor)

    state.release(outputs=(tensor,))

    assert not packed.released
    assert tensor.untyped_storage().nbytes() > 0
    assert torch.equal(activation_swap._unpack(packed), tensor)
    assert packed.device_tensor.grad_fn is None


def test_forward_output_view_preserves_shared_storage(cpu_copies):
    state = activation_swap._ForwardSwapState()
    tensor = torch.arange(6, dtype=torch.float32)
    output = tensor.view(2, 3)
    packed = state.pack(tensor)

    state.release(outputs={"output": output})

    assert not packed.released
    assert torch.equal(output, torch.arange(6, dtype=torch.float32).view(2, 3))


def test_repeated_unpack_preserves_restored_tensor(cpu_copies):
    state = activation_swap._ForwardSwapState()
    tensor = torch.arange(6, dtype=torch.float32)
    expected = tensor.clone()
    packed = state.pack(tensor)
    state.release(outputs=())

    first = activation_swap._unpack(packed)
    second = activation_swap._unpack(packed)

    assert first is second
    assert torch.equal(second, expected)


def test_failed_forward_synchronizes_pending_d2h(cpu_copies):
    state = activation_swap._ForwardSwapState()
    state.pack(torch.ones(2, 3))

    state.abort()

    assert state.d2h_stream.synchronized


def test_failed_release_synchronizes_pending_d2h(cpu_copies, monkeypatch):
    monkeypatch.setattr(activation_swap, "device_type", "npu")
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(
        [torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.SiLU())]
    )
    activation_swap.apply_activation_swap(model)
    states = []
    failure = RuntimeError("activation release failed")

    def fail_release(state, outputs):
        assert state.packed
        states.append(state)
        raise failure

    monkeypatch.setattr(activation_swap._ForwardSwapState, "release", fail_release)
    with pytest.raises(RuntimeError) as caught:
        model.layers[0](torch.ones(1, 2, requires_grad=True))

    assert caught.value is failure
    assert len(states) == 1
    assert states[0].d2h_stream.synchronized


def test_layer_preserves_caller_owned_input(cpu_copies, monkeypatch):
    class Square(torch.nn.Module):
        def forward(self, value):
            return value.square()

    monkeypatch.setattr(activation_swap, "device_type", "npu")
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([Square()])
    activation_swap.apply_activation_swap(model)
    value = torch.arange(6, dtype=torch.float32, requires_grad=True)
    expected = value.detach().clone()

    output = model.layers[0](value)

    assert value.untyped_storage().nbytes() > 0
    assert torch.equal(value, expected)
    output.sum().backward()
    assert torch.equal(value.grad, 2 * expected)


def test_activation_swap_rejects_cpu(monkeypatch):
    monkeypatch.setattr(activation_swap, "device_type", "cpu")
    with pytest.raises(EnvironmentUnsupportedError, match="requires an Ascend NPU"):
        activation_swap.apply_activation_swap(torch.nn.Linear(2, 2))


def test_retained_graph_supports_two_backwards(cpu_copies, monkeypatch):
    class Block(torch.nn.Module):
        def forward(self, value):
            hidden = value * 2
            return hidden.square()

    monkeypatch.setattr(activation_swap, "device_type", "npu")
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([Block()])
    activation_swap.apply_activation_swap(model)
    value = torch.arange(6, dtype=torch.float32, requires_grad=True)
    loss = model.layers[0](value).sum()

    loss.backward(retain_graph=True)
    first_gradient = value.grad.clone()
    value.grad = None
    loss.backward()

    torch.testing.assert_close(first_gradient, 8 * value.detach(), rtol=0, atol=0)
    torch.testing.assert_close(value.grad, first_gradient, rtol=0, atol=0)


def test_repeated_application_preserves_forward_identity(monkeypatch):
    monkeypatch.setattr(activation_swap, "device_type", "npu")
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
    activation_swap.apply_activation_swap(model)
    forward = model.layers[0].forward

    activation_swap.apply_activation_swap(model)

    assert model.layers[0].forward is forward


def test_no_grad_does_not_create_swap_streams(monkeypatch):
    monkeypatch.setattr(activation_swap, "device_type", "npu")
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
    value = torch.ones(1, 2)
    expected = model.layers[0](value).detach()
    activation_swap.apply_activation_swap(model)

    def reject_streams():
        raise AssertionError("no-grad forward must not allocate swap streams")

    monkeypatch.setattr(activation_swap, "_stream_api", reject_streams)
    with torch.no_grad():
        actual = model.layers[0](value)
    assert torch.equal(actual, expected)


def test_activation_swap_rejects_compile():
    from llmtuner.config import LLMTunerConfig, TrainingConfig
    from llmtuner.errors import ConfigError

    with pytest.raises(ConfigError, match="requires compile=False"):
        LLMTunerConfig(training=TrainingConfig(activation_swap=True, compile=True))


@pytest.mark.parametrize("ac_mode", ["selective", "region"])
def test_activation_swap_rejects_checkpoint_hook_modes(ac_mode):
    from llmtuner.config import LLMTunerConfig, TrainingConfig
    from llmtuner.errors import ConfigError

    with pytest.raises(ConfigError, match="incompatible with"):
        LLMTunerConfig(
            training=TrainingConfig(
                activation_swap=True, activation_checkpoint_mode=ac_mode
            )
        )


def test_activation_swap_must_be_boolean():
    from llmtuner.config import TrainingConfig
    from llmtuner.errors import ConfigError

    with pytest.raises(ConfigError, match="activation_swap must be bool"):
        TrainingConfig(activation_swap=1)


@pytest.mark.parametrize(
    "saved_source", ["input", "buffer", "detached_alias", "parameter_alias"]
)
def test_caller_owned_storage_does_not_create_unused_host_copies(
    cpu_copies, monkeypatch, saved_source
):
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.full((6,), 3.0))
            self.weight = torch.nn.Parameter(torch.full((6,), 5.0))

        def forward(self, value):
            if saved_source == "input":
                return value.square()
            if saved_source == "buffer":
                return value * self.scale
            if saved_source == "parameter_alias":
                return value * self.weight.detach()
            return value * value.detach()

    monkeypatch.setattr(activation_swap, "device_type", "npu")
    gc.collect()
    activation_swap.reset_activation_swap_metrics()
    baseline = activation_swap.activation_swap_metrics()["host_bytes_current"]
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([Block()])
    activation_swap.apply_activation_swap(model)
    value = torch.arange(6, dtype=torch.float32, requires_grad=True)
    model.layers[0](value).sum().backward()
    expected = (
        2 * value.detach()
        if saved_source == "input"
        else torch.full_like(value, 5.0)
        if saved_source == "parameter_alias"
        else torch.full_like(value, 3.0)
        if saved_source == "buffer"
        else value.detach()
    )
    torch.testing.assert_close(value.grad, expected, rtol=0, atol=0)
    metrics = activation_swap.activation_swap_metrics()
    assert metrics["copies_d2h"] == metrics["copies_h2d"] == 0, metrics
    assert metrics["host_bytes_current"] == baseline, metrics
