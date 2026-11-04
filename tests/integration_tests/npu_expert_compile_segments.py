"""NPU grouped experts: aligned kernels and compiled unaligned fallback.

Run explicitly with PYTHONPATH=. and Python against this file.
Checks aot_eager/fullgraph, not Inductor or AutoFuse acceptance.
"""

import copy

import torch
from torch._dynamo.utils import counters

from llmtuner.accelerator.grouped_mm import aligned_npu_grouped_mm
from llmtuner.models.common.moe.experts import GroupedExperts
from tests.integration_tests.harness import device


def run_case(dev, *, width, hidden, compile_fallback, layout="contiguous"):
    torch.manual_seed(77)
    reference = GroupedExperts(width, hidden, 3, use_grouped_mm=False).to(
        device=dev, dtype=torch.bfloat16
    )
    for parameter in reference.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    actual = copy.deepcopy(reference)
    actual.use_grouped_mm = True
    if layout == "offset_weights":
        for parameter in actual.parameters():
            storage = parameter.new_empty(parameter.numel() + 1)
            view = storage[1:].view_as(parameter)
            view.copy_(parameter.detach())
            parameter.data = view
            assert parameter.storage_offset() == 1
    execute = (
        torch.compile(actual, backend="aot_eager", fullgraph=True)
        if compile_fallback
        else actual
    )
    tolerance = 0.0 if compile_fallback and layout == "contiguous" else 5e-3
    for counts in ((2, 0, 3), (0, 5, 0), (0, 0, 0)):
        reference.zero_grad(set_to_none=True)
        actual.zero_grad(set_to_none=True)
        inputs = torch.randn(
            sum(counts), width, device=dev, dtype=torch.bfloat16, requires_grad=True
        )
        if layout == "offset_inputs":
            storage = inputs.new_empty(sum(counts), width + 8)
            other = storage[:, 1 : width + 1]
            other.copy_(inputs.detach())
            other.requires_grad_()
            if sum(counts):
                assert other.storage_offset() == 1
        else:
            other = inputs.detach().clone().requires_grad_()
        tokens = torch.tensor(counts, device=dev)
        expected = reference(inputs, tokens)
        if compile_fallback:
            result = execute(other, tokens)
        else:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as profiler:
                result = execute(other, tokens)
            grouped_calls = sum(
                event.count
                for event in profiler.key_averages()
                if event.key == "aten::_grouped_mm"
            )
            expected_calls = {"contiguous": 3, "offset_inputs": 1, "offset_weights": 0}
            assert grouped_calls == (expected_calls[layout] if sum(counts) else 0), (
                grouped_calls
            )
        torch.testing.assert_close(result, expected, rtol=tolerance, atol=tolerance)
        upstream = torch.randn_like(expected)
        expected.backward(upstream)
        result.backward(upstream)
        torch.testing.assert_close(
            inputs.grad, other.grad, rtol=tolerance, atol=tolerance
        )
        for parameter, target in zip(
            actual.parameters(), reference.parameters(), strict=True
        ):
            assert parameter.grad is not None and target.grad is not None
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(
                parameter.grad, target.grad, rtol=tolerance, atol=tolerance
            )
            if sum(counts):
                assert parameter.grad.count_nonzero() > 0
            else:
                assert parameter.grad.count_nonzero() == 0
        print(
            f"PASS NPU D={width} F={hidden} compile_fallback={compile_fallback} "
            f"counts={counts} layout={layout}",
            flush=True,
        )


def verify_alignment_op_contract(dev):
    def project(left, right, offsets):
        return aligned_npu_grouped_mm(left, right, offsets)

    execute = torch.compile(project, backend="aot_eager", fullgraph=True)
    offsets = torch.tensor([2, 2, 5], device=dev, dtype=torch.int32)
    graphs_before = counters["stats"]["unique_graphs"]
    graph_counts = []
    for offset in (0, 1, 8, 1):
        left = (
            torch.randn(88, device=dev, dtype=torch.bfloat16)[offset : offset + 80]
            .view(5, 16)
            .detach()
            .requires_grad_()
        )
        right = (
            torch.randn(1160, device=dev, dtype=torch.bfloat16)[offset : offset + 1152]
            .view(3, 24, 16)
            .transpose(-2, -1)
            .detach()
            .requires_grad_()
        )
        other_left = left.detach().clone().requires_grad_()
        other_right = right.detach().clone().requires_grad_()
        expected = torch.cat(
            [
                other_left[:2] @ other_right[0],
                other_left[2:2] @ other_right[1],
                other_left[2:] @ other_right[2],
            ]
        )
        result = execute(left, right, offsets)
        graph_counts.append(counters["stats"]["unique_graphs"])
        torch.testing.assert_close(result, expected, rtol=5e-3, atol=5e-3)
        upstream = torch.randn_like(result)
        result.backward(upstream)
        expected.backward(upstream)
        torch.testing.assert_close(left.grad, other_left.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(right.grad, other_right.grad, rtol=1e-2, atol=1e-2)
        assert left.grad.count_nonzero() > 0 and right.grad.count_nonzero() > 0
        print(f"PASS compiled alignment offset transition={offset}", flush=True)
    assert graph_counts[0] > graphs_before, graph_counts
    assert len(set(graph_counts)) == 1, graph_counts
    print(f"PASS alignment graph reused across offsets: {graph_counts}", flush=True)
    # The upstream autograd_registration_check rejects NPU devices (NYI).
    # Check registration by actual gradients above, and run supported opchecks.
    result = torch.library.opcheck(
        aligned_npu_grouped_mm,
        (left, right, offsets),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )
    assert all(outcome == "SUCCESS" for outcome in result.values()), result
    print(f"PASS alignment opcheck: {result}", flush=True)


def verify_dynamic_empty_experts(dev):
    """One fullgraph receives nonempty/empty unbacked token dimensions."""
    torch.manual_seed(91)
    reference = GroupedExperts(16, 24, 3, use_grouped_mm=False).to(
        device=dev, dtype=torch.bfloat16
    )
    for parameter in reference.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    actual = copy.deepcopy(reference)
    actual.use_grouped_mm = True

    def project(values, mask):
        selected = values[mask]
        count = mask.sum()
        counts = torch.stack((count, count * 0, count * 0))
        return actual(selected, counts)

    graphs = []
    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        execute = torch.compile(project, backend="aot_eager", fullgraph=True)
        for selected in (3, 0, 1, 0):
            actual.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            values = torch.randn(
                3, 16, device=dev, dtype=torch.bfloat16, requires_grad=True
            )
            other = values.detach().clone().requires_grad_()
            mask = torch.arange(3, device=dev) < selected
            expected = reference(
                other[mask], torch.tensor([selected, 0, 0], device=dev)
            )
            result = execute(values, mask)
            graphs.append(counters["stats"]["unique_graphs"])
            torch.testing.assert_close(result, expected, rtol=5e-3, atol=5e-3)
            upstream = torch.randn_like(result)
            result.backward(upstream)
            expected.backward(upstream)
            torch.testing.assert_close(values.grad, other.grad, rtol=5e-3, atol=5e-3)
            for parameter, target in zip(
                actual.parameters(), reference.parameters(), strict=True
            ):
                assert parameter.grad is not None and target.grad is not None
                torch.testing.assert_close(
                    parameter.grad, target.grad, rtol=5e-3, atol=5e-3
                )
                assert bool(parameter.grad.count_nonzero()) == bool(selected)
            print(f"PASS unbacked expert token count={selected}", flush=True)
    assert len(set(graphs)) == 1, graphs
    print(f"PASS unbacked empty/nonempty graph reuse={graphs}", flush=True)


def verify_expanded_input_fallback(dev, *, compiled):
    """A broadcasted projection must use linear, not the NPU grouped kernel."""
    torch.manual_seed(113)
    reference = GroupedExperts(16, 24, 2, use_grouped_mm=False).to(
        device=dev, dtype=torch.bfloat16
    )
    for parameter in reference.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    actual = copy.deepcopy(reference)
    actual.use_grouped_mm = True

    def project(values, offsets):
        return actual._grouped_mm(values, actual.w1_EFD, offsets)

    execute = (
        torch.compile(project, backend="aot_eager", fullgraph=True)
        if compiled
        else project
    )
    reference_base = torch.randn(
        1, 16, device=dev, dtype=torch.bfloat16, requires_grad=True
    )
    actual_base = reference_base.detach().clone().requires_grad_()
    reference_inputs = reference_base.expand(4, 16)
    actual_inputs = actual_base.expand(4, 16)
    assert actual_inputs.stride(0) == 0
    offsets = torch.tensor([2, 4], device=dev, dtype=torch.int32)
    expected = torch.cat(
        [
            torch.nn.functional.linear(reference_inputs[:2], reference.w1_EFD[0]),
            torch.nn.functional.linear(reference_inputs[2:], reference.w1_EFD[1]),
        ]
    )
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profiler:
        result = execute(actual_inputs, offsets)
    assert not any("grouped_mm" in event.key for event in profiler.key_averages()), (
        "zero-stride input must take the per-expert linear fallback"
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    result.backward(upstream)
    torch.testing.assert_close(actual_base.grad, reference_base.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.w1_EFD.grad, reference.w1_EFD.grad, rtol=0, atol=0
    )
    print(f"PASS zero-stride NPU fallback compiled={compiled}", flush=True)


def main():
    dev = device()
    assert dev.type == "npu", "This regression requires real NPU execution"
    torch.npu.set_device(dev)
    verify_alignment_op_contract(dev)
    verify_dynamic_empty_experts(dev)
    verify_expanded_input_fallback(dev, compiled=False)
    verify_expanded_input_fallback(dev, compiled=True)
    run_case(dev, width=8, hidden=12, compile_fallback=True)
    run_case(dev, width=16, hidden=24, compile_fallback=False)
    for layout in ("offset_inputs", "offset_weights"):
        for compiled in (False, True):
            run_case(dev, width=16, hidden=24, compile_fallback=compiled, layout=layout)


if __name__ == "__main__":
    main()
