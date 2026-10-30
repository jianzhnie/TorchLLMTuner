"""Bounded, explicit layer selection for full-training backward diagnostics."""

import os
import re


def selected_trace_modules(trainer, *, baseline, selection):
    from tests.integration_tests.npu_model_forward_trace import trace_modules

    if selection == "final":
        return {"final_norm": trainer.model_parts[0].norm}
    if not re.fullmatch(r"\d+(,\d+)*", selection):
        raise ValueError("trace layers must be comma-separated nonnegative indices")
    indices = {int(value) for value in selection.split(",")}
    modules = trace_modules(trainer, baseline)
    available = {
        int(match.group(1))
        for name in modules
        if (match := re.fullmatch(r"layer(\d+)", name))
    }
    if not indices <= available:
        raise ValueError(
            f"trace layer indices unavailable: {sorted(indices - available)}"
        )
    # Exclude logits and embedding snapshots to bound host memory and transfers.
    return {
        name: module
        for name, module in modules.items()
        if name == "final_norm"
        or any(
            name == f"layer{i}"
            or (
                os.environ.get("LLMTUNER_ALIGNMENT_TRACE_BOUNDARIES_ONLY") != "1"
                and name.startswith(f"layer{i}.")
            )
            for i in indices
        )
    }
