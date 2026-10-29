"""Real NPU Trainer tracing preserves deterministic updates and exports kernels.

PYTHONPATH=. python tests/integration_tests/npu_profiler_equivalence.py
"""

import argparse
import json
from pathlib import Path

import torch

from examples.train_qwen3 import qwen3_config
from llmtuner.config.observability_config import ProfilerConfig
from llmtuner.trainer.trainer import Trainer


def train(root: Path, *, profiling: bool):
    config = qwen3_config()
    config.training.steps = 4
    config.training.global_batch_size = 1
    config.training.dump_folder = str(root)
    config.training.profiler_config = ProfilerConfig(
        enable_profiling=profiling,
        profile_freq=2,
        profiler_active=1,
        profiler_warmup=1,
        profiler_repeat=1,
    )
    trainer = Trainer(config)
    trainer.train()
    assert trainer.step == 4
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in trainer.model_parts[0].named_parameters()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/npu-profiler-equivalence")
    args = parser.parse_args()
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("This test requires an available NPU")
    root = Path(args.output)
    if root.exists():
        raise FileExistsError(f"Use a fresh output directory: {root}")
    reference = train(root / "reference", profiling=False)
    actual = train(root / "profiled", profiling=True)
    assert reference.keys() == actual.keys()
    for name in reference:
        torch.testing.assert_close(actual[name], reference[name], rtol=0, atol=0)
    traces = list((root / "profiled").rglob("rank0_trace.json"))
    assert traces, "No native NPU trace exported"
    raw_traces = [
        path for path in (root / "profiled").rglob("raw/**/*") if path.is_file()
    ]
    assert raw_traces, "No raw NPU TensorBoard trace exported"
    kernels = []
    for trace in traces:
        payload = json.loads(trace.read_text())
        events = payload if isinstance(payload, list) else payload["traceEvents"]
        hardware = {
            event["pid"]
            for event in events
            if event.get("name") == "process_name"
            and event.get("args", {}).get("name") == "Ascend Hardware"
        }
        kernels.extend(
            event
            for event in events
            if event.get("pid") in hardware
            and event.get("ph") == "X"
            and event.get("dur", 0) > 0
            and event.get("args", {}).get("Task Type", "").startswith("KERNEL_")
        )
    assert kernels, "Trace has no NPU kernel events"
    print(f"PASS: exact training updates; {len(kernels)} NPU kernels in {traces}")


if __name__ == "__main__":
    main()
