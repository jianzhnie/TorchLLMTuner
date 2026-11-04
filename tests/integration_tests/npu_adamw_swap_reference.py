"""Execute the actual FSDPTurbo AdamWSwap alongside native and local updates.

Use torchrun --standalone --nproc-per-node=1. Reference Python source is trusted
local repository code supplied explicitly with --reference-root, not a fixture.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from tests.caps import require_env

require_env("torch_npu")

from llmtuner.accelerator.device import get_current_device, set_device
from llmtuner.components.optimizer.adamw_swap import AdamWSwap


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def local(value):
    return value.to_local() if isinstance(value, DTensor) else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--amsgrad", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--trace-dir", type=Path)
    args = parser.parse_args()
    device = get_current_device()
    set_device(device)
    dist.init_process_group("hccl")
    profiler = None
    try:
        # Execute the reference startup compatibility code, without editing its
        # optimizer or replacing any of its math/stream methods.
        load_module(
            args.reference_root / "fsdp_turbo/utils/device.py", "reference_device"
        ).set_accelerator_compatible()
        reference_class = load_module(
            args.reference_root / "fsdp_turbo/memory/swap_optimizer/swap_optimizer.py",
            "reference_swap",
        ).AdamWSwap
        mesh = init_device_mesh(device.type, (dist.get_world_size(),))
        dtype = getattr(torch, args.dtype)
        values = [
            torch.linspace(0.2, 1.8, count, dtype=dtype, device=device)
            for count in (37, 19)
        ]

        def parameters():
            result = []
            for value in values:
                tensor = DTensor.from_local(
                    value.clone(), mesh, [Shard(0)], run_check=False
                )
                if args.cpu_offload:
                    host = local(tensor).cpu().pin_memory()
                    tensor = DTensor(host, tensor._spec, requires_grad=False)
                result.append(torch.nn.Parameter(tensor))
            return result

        actual, expected = parameters(), parameters()
        native_params = [torch.nn.Parameter(value.clone()) for value in values]
        options = dict(
            lr=0.003, betas=(0.8, 0.95), weight_decay=0.1, amsgrad=args.amsgrad
        )
        reference = reference_class(expected, param_prefetch=True, **options)
        swap = AdamWSwap(
            actual,
            fused=True,
            prefetch=True,
            max_device_bytes=256,
            memory_fraction=0.8,
            update_device=device,
            **options,
        )
        native = torch.optim.AdamW(native_params, fused=True, **options)
        names = (
            ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
            if args.amsgrad
            else ("exp_avg", "exp_avg_sq")
        )
        if args.trace_dir:
            from torch_npu import profiler as npu_profiler

            profiler = npu_profiler.profile(
                activities=[
                    npu_profiler.ProfilerActivity.CPU,
                    npu_profiler.ProfilerActivity.NPU,
                ],
                on_trace_ready=npu_profiler.tensorboard_trace_handler(
                    str(args.trace_dir)
                ),
                record_shapes=True,
            )
            profiler.start()
        for step in range(5):
            for index, (candidate, baseline, normal) in enumerate(
                zip(actual, expected, native_params, strict=True)
            ):
                gradient = torch.linspace(
                    -0.3, 0.7, normal.numel(), device=device, dtype=dtype
                ) * ((index + 1) * (step + 1))
                normal.grad = gradient
                for parameter in (candidate, baseline):
                    payload = (
                        gradient.cpu().pin_memory()
                        if args.cpu_offload
                        else gradient.clone()
                    )
                    parameter.grad = DTensor(
                        payload, parameter._spec, requires_grad=False
                    )
            with torch.profiler.record_function("reference_native_adamw"):
                native.step()
            with torch.profiler.record_function("reference_fsdpturbo_swap"):
                reference.step()
            with torch.profiler.record_function("candidate_adamw_swap"):
                swap.step()
            for candidate, baseline, normal in zip(
                actual, expected, native_params, strict=True
            ):
                for other in (local(baseline), normal):
                    torch.testing.assert_close(
                        local(candidate).cpu(), other.detach().cpu(), rtol=0, atol=0
                    )
                for name in names:
                    moment = local(swap.state[candidate][name])
                    assert moment.device.type == "cpu" and moment.is_pinned()
                    torch.testing.assert_close(
                        moment,
                        reference.param_to_cpu_states_map[baseline][name],
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        moment, native.state[normal][name].cpu(), rtol=0, atol=0
                    )
        # FSDPTurbo advances a group counter even for absent gradients. Preserve
        # native per-parameter semantics rather than reproducing that difference.
        actual[0].grad = None
        native_params[0].grad = None
        swap.step()
        native.step()
        for candidate, normal in zip(actual, native_params, strict=True):
            torch.testing.assert_close(
                local(candidate).cpu(), normal.detach().cpu(), rtol=0, atol=0
            )
            assert (
                swap.state[candidate]["step"].item()
                == native.state[normal]["step"].item()
            )
        assert swap.prefetch_count > 1
        assert 0 < swap.staging_peak_payload_bytes <= 256
        print(
            f"PASS actual FSDPTurbo/native/swap bitwise: {args.dtype}, "
            f"amsgrad={args.amsgrad}, cpu_offload={args.cpu_offload}",
            flush=True,
        )
    finally:
        if profiler is not None:
            profiler.stop()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
