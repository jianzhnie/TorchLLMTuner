"""Same-input raw depthwise-conv VJP diagnostic; no training acceptance.

Requires a real captured HF layer0 conv payload (never synthesizes activations).
``capture_hf_conv`` can surround ONE existing actual module/full-model forward
and backward; its tensor hook records the raw-conv output cotangent, including
all downstream SiLU/GDN effects. Observation hooks can affect scheduling: the
capture is provenance, not an uninstrumented whole-model acceptance result.

Replay compares original HF Conv1d (symmetric padding then causal crop) with
three original reference Conv1d modules on explicitly left-padded input. SiLU
is outside this raw-conv boundary and is already represented by captured dy.
Reference code, tolerances, parameters and production are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

SIZES = (2048, 2048, 4096)
ATOL, RTOL = 0.005, 0.02


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def conv_capture_path(base, *, phase, steps, rank):
    """PATH is a new directory containing one immutable payload per rank."""
    if base is None:
        return None
    if phase not in ("hf", "reference") or steps != 1:
        raise ValueError("--capture-first-conv requires single-step HF/reference phase")
    return Path(base) / f"rank-{rank:05d}.pt"


@contextmanager
def capture_hf_conv(
    gdn, destination: Path, *, provenance: dict, local_errors=nullcontext
):
    """Capture actual packed conv input/raw dy once; caller owns execution.

    destination must be new. Provenance must identify source artifacts/run;
    this function never launches a model or installs a production adapter.
    """
    captured, tensor_handles, observation_errors = {}, [], []
    handle = None

    def observe(module, inputs, output):
        # Never interrupt FSDP forward/backward for an observation-only error.
        # All ranks finish the original graph before the coordinated boundary.
        try:
            assert not captured, "Expected one GDN conv invocation, no AC recompute"
            assert len(inputs) == 1 and output.requires_grad
            captured["input"] = inputs[0].detach().cpu().clone()
            captured["weight"] = module.weight.detach().cpu().clone()
            captured["output"] = output.detach().cpu().clone()

            def cotangent(value):
                try:
                    assert "cotangent" not in captured
                    captured["cotangent"] = value.detach().cpu().clone()
                except Exception as error:
                    observation_errors.append(error)
                return None  # Preserve the original backward cotangent.

            tensor_handles.append(output.register_hook(cotangent))
        except Exception as error:
            observation_errors.append(error)

    try:
        with local_errors():
            assert provenance and not destination.exists()
            assert gdn.causal_conv1d_fn is None, "Fused conv is outside this probe"
            destination.parent.mkdir(parents=True, exist_ok=True)
            handle = gdn.conv1d.register_forward_hook(observe)
        yield
        with local_errors():
            if observation_errors:
                raise observation_errors[0]
            assert set(captured) == {"input", "weight", "output", "cotangent"}
            length = captured["input"].shape[-1]
            assert captured["cotangent"].shape == captured["output"].shape
            assert captured["output"].shape[-1] == length + 3
            assert not bool(captured["cotangent"][:, :, length:].any())
            with destination.open("xb") as stream:
                torch.save(
                    {
                        "schema": "qwen35-raw-conv-capture-v1",
                        "provenance": provenance,
                        "loss_scale": 1.0,
                        "normalization_multiplier": 1.0,
                        "causal_output": captured["output"][:, :, :length].clone(),
                        "normalized_cotangent": captured["cotangent"][
                            :, :, :length
                        ].clone(),
                        **captured,
                    },
                    stream,
                )
    finally:
        # Removal is local and must also run when setup coordination reports a
        # failure on another rank. Do not add a collective here: a primary
        # model/collective failure is outside this observer's recovery scope.
        if handle is not None:
            handle.remove()
        for tensor_handle in tensor_handles:
            tensor_handle.remove()


@contextmanager
def capture_reference_conv(
    gdn,
    destination: Path,
    *,
    provenance: dict,
    loss_scale: float,
    local_errors=nullcontext,
):
    """Observe original functional conv calls; never invoke a second forward.

    NPU reference reads conv.weight directly and bypasses module.forward.
    TorchFunctionMode observes torch.conv1d without replacing its arithmetic.
    raw cotangents remain scaled; normalized_cotangent is diagnostic-only FP32.
    """
    captured, errors, handles = {}, [], []
    modules = {}
    hook = None

    class ObserveConv(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            output = func(*args, **kwargs)
            if func is torch.conv1d:
                try:
                    weight = args[1] if len(args) > 1 else kwargs["weight"]
                    name = modules.get(weight.data_ptr())
                    if name is not None:
                        assert name not in captured, "Repeated conv/AC is unsupported"
                        x = args[0] if args else kwargs["input"]
                        assert weight.shape[-2:] == (1, 4) and output.requires_grad
                        assert x.shape[-1] == output.shape[-1] + 3
                        assert not bool(x[:, :, :3].any()), (
                            "Expected explicit left padding"
                        )
                        record = {
                            "input": x[:, :, 3:].detach().cpu().clone(),
                            "weight": weight.detach().cpu().clone(),
                            "output": output.detach().cpu().clone(),
                        }
                        captured[name] = record

                        def observe_dy(value):
                            try:
                                assert "cotangent" not in record
                                record["cotangent"] = value.detach().cpu().clone()
                            except Exception as error:
                                errors.append(error)
                            return None

                        handles.append(output.register_hook(observe_dy))
                except Exception as error:
                    errors.append(error)
            return output

    try:
        with local_errors():
            assert provenance and not destination.exists() and 0 < loss_scale <= 1
            destination.parent.mkdir(parents=True, exist_ok=True)
            for name in ("q", "k", "v"):
                weight = getattr(gdn, "conv_" + name).weight
                # FSDP supplies unsharded views only at forward time; avoid
                # reading a sharded DTensor pointer during setup.
                assert weight.ndim == 3 and weight.shape[-2:] == (1, 4)

        # Resolve current weight addresses inside the original GDN forward,
        # after its owning FSDP layer has all-gathered parameters.
        def before_gdn(module, args):
            try:
                for name in ("q", "k", "v"):
                    weight = getattr(module, "conv_" + name).weight
                    pointer = weight.data_ptr()
                    assert pointer not in modules
                    modules[pointer] = name
            except Exception as error:
                errors.append(error)

        with local_errors():
            hook = gdn.register_forward_pre_hook(before_gdn)
        try:
            with ObserveConv():
                yield
        finally:
            if hook is not None:
                hook.remove()
        with local_errors():
            if errors:
                raise errors[0]
            assert set(captured) == {"q", "k", "v"}
            for record in captured.values():
                assert set(record) == {"input", "weight", "output", "cotangent"}
                assert record["output"].shape == record["cotangent"].shape
            joined = {
                key: torch.cat(
                    [captured[name][key] for name in ("q", "k", "v")],
                    dim=1 if key != "weight" else 0,
                )
                for key in ("input", "weight", "output", "cotangent")
            }
            with destination.open("xb") as stream:
                torch.save(
                    {
                        "schema": "qwen35-reference-raw-conv-capture-v1",
                        "provenance": provenance,
                        "loss_scale": loss_scale,
                        "normalization_multiplier": 1.0 / loss_scale,
                        "branches": captured,
                        **joined,
                        "causal_output": joined["output"],
                        "normalized_cotangent": joined["cotangent"].float()
                        / loss_scale,
                    },
                    stream,
                )
    finally:
        if hook is not None:
            hook.remove()
        for handle in handles:
            handle.remove()


def compare(left, right):
    assert left.shape == right.shape and left.dtype == right.dtype
    a, b = left.float(), right.float()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    delta = (a - b).abs()
    failed = delta > ATOL + RTOL * a.abs()
    return {
        "exact": torch.equal(left, right),
        "finite": finite,
        "passed": finite and not bool(failed.any()),
        "failed_elements": int(failed.sum()),
        "elements": a.numel(),
        "max_abs": float(delta.max()) if finite else None,
        "rms": float(delta.square().mean().sqrt()) if finite else None,
    }


def compare_capture_boundaries(hf, reference):
    """Compare observed causal boundary only; never rescale real backward.

    Reference raw dy is a loss/8 VJP. Multiply in FP32 for comparison, retaining
    both original payloads and their loss scale. No reverse quantization claim.
    """
    assert hf["loss_scale"] == 1.0 and reference["loss_scale"] == 1.0 / 8
    for key in (
        "initial_manifest_sha256",
        "initial_layer0_sha256",
        "rank",
        "step",
        "tokens",
    ):
        assert hf["provenance"][key] == reference["provenance"][key], key
    length = hf["input"].shape[-1]
    assert hf["cotangent"].shape[-1] == length + 3
    assert not bool(hf["cotangent"][:, :, length:].any())
    assert reference["cotangent"].shape == reference["output"].shape
    pairs = {
        "input": (hf["input"], reference["input"]),
        "weight": (hf["weight"], reference["weight"]),
        "causal_output": (hf["output"][:, :, :length], reference["output"]),
        "normalized_cotangent": (
            hf["cotangent"][:, :, :length].float(),
            reference["cotangent"].float() * 8,
        ),
    }
    return {key: compare(left, right) for key, (left, right) in pairs.items()}


def raw_vjp(module, x, dy, *, packed, left_pad=False):
    """Use original modules unchanged; only assemble raw convolution boundary."""
    module.zero_grad(set_to_none=True)
    leaf = x.detach().clone().requires_grad_()
    length = leaf.shape[-1]
    if left_pad:
        assert packed, "Packed left-pad requires original HF module"
        output = F.conv1d(
            F.pad(leaf, (3, 0)),
            module.weight,
            module.bias,
            stride=module.stride,
            padding=0,
            dilation=module.dilation,
            groups=module.groups,
        )
    elif packed:
        raw = module(leaf)
        output = raw[:, :, :length]
    else:
        output = torch.cat(
            [
                conv(F.pad(part, (3, 0)))
                for conv, part in zip(module, leaf.split(SIZES, dim=1), strict=True)
            ],
            dim=1,
        )
    # Captured HF raw output has three unused right-tail positions. Their dy
    # must be zero; the cropped boundary consumes exactly the causal prefix.
    assert dy.shape[-1] == length + 3 and not bool(dy[:, :, length:].any())
    output.backward(dy[:, :, :length])
    weight = (
        module.weight.grad
        if packed
        else torch.cat([conv.weight.grad for conv in module], dim=0)
    )
    return {
        "output": output.detach().cpu(),
        "dx": leaf.grad.detach().cpu(),
        "dw": weight.detach().cpu(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    assert args.execute and args.repeats >= 2 and not args.output.exists()
    # Explicit opt-in only: importing this file for CPU contracts uses no NPU.
    actual = importlib.import_module(
        "tests.integration_tests.qwen35_gdn_module_alignment"
    )

    payload = torch.load(args.capture, map_location="cpu", weights_only=True, mmap=True)
    assert payload["schema"] == "qwen35-raw-conv-capture-v1" and payload["provenance"]
    source = torch.load(
        args.artifacts / "initial-0.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    weight = source["model.layers.0.linear_attn.conv1d.weight"]
    assert torch.equal(weight, payload["weight"]), (
        "Capture weight differs from shared initial"
    )
    config = actual.AutoConfig.from_pretrained(args.model, local_files_only=True)
    config = getattr(config, "text_config", config)
    spec = actual.model_registry("35B-A3B")
    with torch.device("meta"):
        hf = actual.Qwen3_5MoeGatedDeltaNet(config, 0).bfloat16().conv1d
        reference = spec.model.layers[0].delta_net.build()
    # Retain original reference convolution modules only, no full GDN allocation.
    separate = torch.nn.ModuleList(
        [reference.conv_q, reference.conv_k, reference.conv_v]
    )
    del reference
    device = torch.device("npu", args.device)
    torch.npu.set_device(device)
    hf.to_empty(device=device)
    separate.to(dtype=torch.bfloat16).to_empty(device=device)
    with torch.no_grad():
        hf.weight.copy_(weight)
        for conv, part in zip(separate, weight.split(SIZES), strict=True):
            assert conv.padding in (0, (0,)) and conv.stride in (1, (1,))
            conv.weight.copy_(part)
    x, dy = payload["input"], payload["cotangent"]
    assert x.shape[:2] == (1, 8192) and weight.shape == (8192, 1, 4)
    assert all(
        t.dtype == torch.bfloat16
        and bool(torch.isfinite(t).all())
        and bool(t.count_nonzero())
        for t in (x, dy, weight)
    )
    x, dy = x.to(device), dy.to(device)
    args.output.mkdir(parents=True)
    report = {
        "scope": "captured raw conv boundary, not full-model acceptance",
        "capture_sha256": file_hash(args.capture),
        "provenance": payload["provenance"],
        "initial_sha256": file_hash(args.artifacts / "initial-0.pt"),
        "tolerances": {"atol": ATOL, "rtol": RTOL},
        "repeats": {},
    }
    for repeat in range(args.repeats):
        values = {}
        for name, module in (
            ("packed", hf),
            ("separate", separate),
            ("packed_left_pad", hf),
        ):
            values[name] = raw_vjp(
                module,
                x,
                dy,
                packed=name != "separate",
                left_pad=name == "packed_left_pad",
            )
            path = args.output / f"{name}-{repeat}.pt"
            with path.open("xb") as stream:
                torch.save(values[name], stream)
        entry = {
            key: compare(values["packed"][key], values["separate"][key])
            for key in values["packed"]
        }
        for baseline in ("packed", "separate"):
            entry["packed_left_pad_vs_" + baseline] = {
                key: compare(values["packed_left_pad"][key], values[baseline][key])
                for key in values[baseline]
            }
        if repeat:
            for name in values:
                first = torch.load(
                    args.output / f"{name}-0.pt", map_location="cpu", weights_only=True
                )
                entry[name + "_repeat"] = {
                    key: compare(values[name][key], first[key]) for key in first
                }
        report["repeats"][repeat] = entry
    report["numerical_all_passed"] = all(
        item["passed"]
        for entry in report["repeats"].values()
        for key, item in entry.items()
        if key in ("output", "dx", "dw")
    )
    report["packed_left_pad_vs_separate_all_passed"] = all(
        item["passed"]
        for entry in report["repeats"].values()
        for item in entry["packed_left_pad_vs_separate"].values()
    )
    report["repeat_stability"] = {
        name: {
            "all_exact": all(
                item["exact"]
                for entry in report["repeats"].values()
                if name + "_repeat" in entry
                for item in entry[name + "_repeat"].values()
            ),
            "all_passed": all(
                item["passed"]
                for entry in report["repeats"].values()
                if name + "_repeat" in entry
                for item in entry[name + "_repeat"].values()
            ),
        }
        for name in ("packed", "separate", "packed_left_pad")
    }
    with (args.output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["numerical_all_passed"] else 1)


if __name__ == "__main__":
    main()
