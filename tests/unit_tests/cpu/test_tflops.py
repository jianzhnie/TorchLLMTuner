from types import SimpleNamespace

import pytest

from examples import tflops


def test_checkpoint_replay_resumes_flop_counter_after_error(monkeypatch):
    calls = []
    counter = SimpleNamespace(
        pause=lambda: calls.append("pause"),
        resume=lambda: calls.append("resume"),
    )
    monkeypatch.setattr(tflops, "get_flops_counter", lambda: counter)

    def fail():
        raise RuntimeError("replay failed")

    context = SimpleNamespace(run_function=fail)
    original = context.run_function

    @tflops.checkpoint_function_backward_wrapper
    def backward(ctx):
        return ctx.run_function()

    with pytest.raises(RuntimeError, match="replay failed"):
        backward(context)

    assert context.run_function is original
    assert calls == ["pause", "resume"]


def test_failed_training_step_stops_flop_counter_without_recording(monkeypatch):
    calls = []
    counter = SimpleNamespace(
        start=lambda: calls.append("start"),
        stop=lambda: calls.append("stop"),
        get_flops=lambda: pytest.fail("failed step must not record FLOPs"),
    )
    monkeypatch.setattr(tflops, "get_flops_counter", lambda: counter)
    monkeypatch.setattr(tflops, "set_count", lambda counts: pytest.fail("recorded"))

    @tflops.train_step_wrapper
    def fail():
        raise RuntimeError("training failed")

    with pytest.raises(RuntimeError, match="training failed"):
        fail()

    assert calls == ["start", "stop"]
