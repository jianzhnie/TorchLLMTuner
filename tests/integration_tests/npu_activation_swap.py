"""Ascend NPU: activation exchange and AC numerical compatibility."""

import gc
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

from llmtuner.config import (
    CheckpointConfig,
    LLMTunerConfig,
    ModelConfig,
    ParallelConfig,
    TrainingConfig,
)
from llmtuner.memory import activation_swap
from llmtuner.parallel.tensor_parallel.linear import (
    ColumnParallelLinear,
    ColwiseLinearNoGather,
    RowParallelLinear,
)
from llmtuner.trainer.trainer import Trainer


def _run(
    directory: Path,
    swap: bool,
    ac_mode: str,
    sync_release: bool = False,
    max_host_bytes: int = 0,
) -> tuple[dict[str, torch.Tensor], int, int, dict[str, float | int]]:
    activation_swap.reset_activation_swap_metrics()
    tp_size = int(os.environ.get("NPU_SWAP_TP_SIZE", "1"))
    moe = os.environ.get("NPU_SWAP_MOE") == "1"
    config = LLMTunerConfig(
        parallel=ParallelConfig(tensor_parallel_size=tp_size),
        model=ModelConfig(
            model_name_or_path="qwen3_moe" if moe else "qwen3",
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            arch_overrides=(
                {
                    "num_experts": 4,
                    "num_experts_per_tok": 2,
                    "moe_intermediate_size": 16,
                }
                if moe
                else {}
            ),
        ),
        training=TrainingConfig(
            steps=2,
            dump_folder=str(directory),
            activation_checkpoint_mode=ac_mode,
            activation_swap=swap,
            activation_swap_sync_release=sync_release,
            activation_swap_max_host_bytes=max_host_bytes,
            checkpoint_config=CheckpointConfig(enable=False),
        ),
    )
    trainer = Trainer(config)
    if tp_size > 1:
        assert any(
            isinstance(
                module, ColumnParallelLinear | ColwiseLinearNoGather | RowParallelLinear
            )
            for part in trainer.model_parts
            for module in part.modules()
        ), "activation swap TP test has no TP projections"
    initial = {
        name: value.detach().cpu().clone()
        for name, value in trainer.example_model.state_dict().items()
    }
    torch.npu.reset_peak_memory_stats()
    released_count = 0
    original_release = activation_swap._ForwardSwapState.release

    def count_release(state, outputs):
        nonlocal released_count
        original_release(state, outputs)
        released_count += sum(packed.released for packed in state.packed)

    if swap:
        activation_swap._ForwardSwapState.release = count_release
    try:
        profiling = swap and os.environ.get("NPU_SWAP_PROFILE") == "1"
        if profiling:
            from torch_npu import profiler as npu_profiler

            context = npu_profiler.profile(
                activities=[
                    npu_profiler.ProfilerActivity.CPU,
                    npu_profiler.ProfilerActivity.NPU,
                ]
            )
        else:
            context = nullcontext()
        with context as profiler:
            trainer.train()
        if profiling:
            profiler.export_chrome_trace(str(directory / "trace.json"))
        metrics = activation_swap.activation_swap_metrics()
        updated = {
            name: value.detach().cpu().clone()
            for name, value in trainer.example_model.state_dict().items()
        }
        assert any(
            not torch.equal(value, initial[name]) for name, value in updated.items()
        )
        return (
            updated,
            torch.npu.max_memory_allocated(),
            released_count,
            metrics,
        )
    finally:
        activation_swap._ForwardSwapState.release = original_release
        trainer.close()


def verify_discarded_graph_cleanup() -> None:
    class SavedActivationLayer(torch.nn.Module):
        def __init__(self, fail):
            super().__init__()
            self.fail = fail

        def forward(self, value):
            hidden = value.sin()
            output = hidden.square()
            if self.fail:
                raise RuntimeError("intentional activation-swap forward failure")
            return output

    for fail in (False, True):
        gc.collect()
        activation_swap.reset_activation_swap_metrics()
        baseline = activation_swap.activation_swap_metrics()["host_bytes_current"]
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([SavedActivationLayer(fail)])
        activation_swap.apply_activation_swap(model)
        value = torch.linspace(-1, 1, 1024, device="npu").requires_grad_()
        if fail:
            try:
                model.layers[0](value)
            except RuntimeError as error:
                assert str(error) == "intentional activation-swap forward failure"
            else:
                raise AssertionError("the failing forward did not raise")
        else:
            output = model.layers[0](value)
            assert torch.isfinite(output).all().item()
            assert output.count_nonzero().item() > 0
            del output
        del model, value
        gc.collect()
        torch.npu.synchronize()
        metrics = activation_swap.activation_swap_metrics()
        assert metrics["copies_d2h"] > 0, metrics
        assert metrics["d2h_bytes"] > 0, metrics
        assert metrics["h2d_bytes"] == 0, metrics
        assert metrics["host_bytes_current"] == baseline, metrics
        print(f"PASS activation swap discarded graph fail={fail}: {metrics}")


def verify_retained_graph_reuse() -> None:
    class SavedActivationLayer(torch.nn.Module):
        def forward(self, value):
            return value.sin().square()

    gc.collect()
    activation_swap.reset_activation_swap_metrics()
    baseline = activation_swap.activation_swap_metrics()["host_bytes_current"]
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([SavedActivationLayer()])
    activation_swap.apply_activation_swap(model)
    value = torch.linspace(-1, 1, 1024, device="npu").requires_grad_()
    reference = value.detach().clone().requires_grad_()
    output = model.layers[0](value).sum()
    expected = reference.sin().square().sum()
    output.backward(retain_graph=True)
    expected.backward(retain_graph=True)
    torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)
    assert value.grad.count_nonzero().item() > 0
    first = activation_swap.activation_swap_metrics()
    assert first["copies_h2d"] > 0, first
    output.backward()
    expected.backward()
    torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)
    second = activation_swap.activation_swap_metrics()
    # Unpack may be called again, but the reloaded device tensor is reusable.
    assert second["copies_h2d"] == first["copies_h2d"], (first, second)
    assert second["h2d_bytes"] == first["h2d_bytes"], (first, second)
    del output, expected, value, reference, model
    gc.collect()
    torch.npu.synchronize()
    final = activation_swap.activation_swap_metrics()
    assert final["host_bytes_current"] == baseline, final
    print(f"PASS activation swap retained graph: {final}")


def verify_owned_storage_skips() -> None:
    class OwnedStorageLayer(torch.nn.Module):
        def __init__(self, source):
            super().__init__()
            self.source = source
            self.register_buffer("scale", torch.full((1024,), 3.0))
            self.weight = torch.nn.Parameter(torch.full((1024,), 5.0))

        def forward(self, value):
            if self.source == "input":
                return value.square()
            if self.source == "buffer":
                return value * self.scale
            if self.source == "parameter_alias":
                return value * self.weight.detach()
            return value * value.detach()

    for source in ("input", "buffer", "detached_alias", "parameter_alias"):
        gc.collect()
        activation_swap.reset_activation_swap_metrics()
        baseline = activation_swap.activation_swap_metrics()["host_bytes_current"]
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([OwnedStorageLayer(source)])
        model.to("npu")
        activation_swap.apply_activation_swap(model)
        value = torch.linspace(-1, 1, 1024, device="npu").requires_grad_()
        model.layers[0](value).sum().backward()
        expected = (
            2 * value.detach()
            if source == "input"
            else torch.full_like(value, 5.0)
            if source == "parameter_alias"
            else torch.full_like(value, 3.0)
            if source == "buffer"
            else value.detach()
        )
        torch.testing.assert_close(value.grad, expected, rtol=0, atol=0)
        assert value.grad.count_nonzero().item() > 0
        torch.npu.synchronize()
        metrics = activation_swap.activation_swap_metrics()
        assert metrics["copies_d2h"] == metrics["copies_h2d"] == 0, metrics
        assert metrics["host_bytes_current"] == baseline, metrics
        print(f"PASS activation swap owned storage {source}: {metrics}")


def verify_host_budget() -> None:
    class Layer(torch.nn.Module):
        def forward(self, value):
            return value.sin().square()

    gc.collect()
    activation_swap.reset_activation_swap_metrics()
    baseline = activation_swap.activation_swap_metrics()["host_bytes_current"]
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([Layer(), Layer()])
    activation_swap.apply_activation_swap(model, max_host_bytes=4096)
    values = [
        torch.linspace(-1, 1, 1024, device="npu").requires_grad_() for _ in range(2)
    ]
    references = [value.detach().clone().requires_grad_() for value in values]
    outputs = [layer(value) for layer, value in zip(model.layers, values, strict=True)]
    metrics = activation_swap.activation_swap_metrics()
    assert metrics["copies_d2h"] == 1, metrics
    assert metrics["host_bytes_current"] == baseline + 4096, metrics
    assert metrics["host_bytes_peak"] == baseline + 4096, metrics
    sum(output.sum() for output in outputs).backward()
    sum(value.sin().square().sum() for value in references).backward()
    for value, reference in zip(values, references, strict=True):
        torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)
        assert value.grad.count_nonzero().item() > 0
    del outputs, model
    gc.collect()
    torch.npu.synchronize()
    metrics = activation_swap.activation_swap_metrics()
    assert metrics["host_bytes_current"] == baseline, metrics
    assert metrics["copies_h2d"] == 1, metrics
    print(f"PASS activation swap shared Host budget: {metrics}")


def main() -> None:
    if not torch.npu.is_available():
        raise RuntimeError("activation swap integration requires a visible NPU")
    root = Path(sys.argv[1])
    # Every torchrun rank uses the same output root; creation is idempotent.
    root.mkdir(parents=True, exist_ok=True)
    for ac_mode in ("none", "full"):
        baseline, baseline_memory, _, _ = _run(
            root / ac_mode / "baseline", swap=False, ac_mode=ac_mode
        )
        swapped, swapped_memory, released_count, metrics = _run(
            root / ac_mode / "swapped", swap=True, ac_mode=ac_mode
        )
        if ac_mode == "none":
            assert released_count > 0, (
                "activation swap did not release any saved tensor: "
                f"released={released_count}, metrics={metrics}"
            )
            assert metrics["d2h_bytes"] > 0
            assert metrics["h2d_bytes"] > 0
            assert metrics["host_bytes_peak"] > 0
        else:
            assert metrics["d2h_bytes"] == metrics["h2d_bytes"] == 0
            assert metrics["host_bytes_peak"] == metrics["host_bytes_current"], metrics
        assert baseline.keys() == swapped.keys()
        for name, value in swapped.items():
            torch.testing.assert_close(value, baseline[name], rtol=0, atol=0, msg=name)
        print(
            f"PASS activation swap AC={ac_mode}: two-step updates match bitwise; "
            f"released={released_count}; "
            f"peak_allocated_baseline={baseline_memory}, swapped={swapped_memory}; "
            f"transfers={metrics}"
        )
    asynchronous, _, _, _ = _run(
        root / "none" / "asynchronous", swap=True, ac_mode="none"
    )
    synchronous, _, _, _ = _run(
        root / "none" / "synchronous",
        swap=True,
        ac_mode="none",
        sync_release=True,
    )
    for name, value in synchronous.items():
        torch.testing.assert_close(value, asynchronous[name], rtol=0, atol=0, msg=name)
    print("PASS activation_swap_sync_release: updates match asynchronous release")
    bounded, _, released, metrics = _run(
        root / "none" / "bounded", swap=True, ac_mode="none", max_host_bytes=1
    )
    for name, value in bounded.items():
        torch.testing.assert_close(value, asynchronous[name], rtol=0, atol=0, msg=name)
    assert released == 0, released
    assert metrics["copies_d2h"] == metrics["copies_h2d"] == 0, metrics
    print("PASS activation swap configuration budget: retained-device updates exact")
    verify_discarded_graph_cleanup()
    verify_retained_graph_reuse()
    verify_owned_storage_skips()
    verify_host_budget()


if __name__ == "__main__":
    main()
