"""Observation-only full-layer output cotangents; no retain_grad/recompute.

Hook failures are deferred until original backward finishes. This cannot
recover failures in model arithmetic/collectives. Captures are observed-graph
evidence, not an assertion that hooks leave BF16 accumulation scheduling exact.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch


def ordered_layers(model, *, reference):
    layers = model.layers if reference else model.model.layers
    if isinstance(layers, torch.nn.ModuleDict):
        assert list(layers) == [str(index) for index in range(40)]
        layers = list(layers.values())
    else:
        layers = list(layers)
    assert len(layers) == 40
    return layers


def backward_capture_path(base, *, phase, steps, rank):
    if base is None:
        return None
    if phase not in ("hf", "reference") or steps != 1:
        raise ValueError("Layer cotangent capture requires single-step HF/reference")
    return Path(base) / f"rank-{rank:05d}.pt"


@contextmanager
def capture_layer_cotangents(
    layers, destination: Path, *, provenance, loss_scale, local_errors=nullcontext
):
    """Observe each original layer output once, preserving its actual VJP."""
    forward_handles, tensor_handles, errors = [], [], []
    outputs, cotangents = {}, {}

    def observe(index):
        def forward(_module, _args, output):
            try:
                assert index not in outputs, "Repeated forward/AC is unsupported"
                assert isinstance(output, torch.Tensor) and output.requires_grad
                outputs[index] = {
                    "shape": list(output.shape),
                    "dtype": str(output.dtype),
                }

                def backward(value):
                    try:
                        assert index not in cotangents, "Repeated backward"
                        cotangents[index] = value.detach().cpu().clone()
                    except Exception as error:
                        errors.append(error)
                    return None

                tensor_handles.append(output.register_hook(backward))
            except Exception as error:
                errors.append(error)

        return forward

    try:
        with local_errors():
            assert len(layers) == 40 and provenance and 0 < loss_scale <= 1
            assert not destination.exists()
            destination.parent.mkdir(parents=True, exist_ok=True)
            for index, layer in enumerate(layers):
                forward_handles.append(layer.register_forward_hook(observe(index)))
        yield
        with local_errors():
            if errors:
                raise errors[0]
            assert set(outputs) == set(cotangents) == set(range(40))
            for index, value in cotangents.items():
                assert list(value.shape) == outputs[index]["shape"]
                assert str(value.dtype) == outputs[index]["dtype"]
                assert bool(torch.isfinite(value).all()), index
            with destination.open("xb") as stream:
                torch.save(
                    {
                        "schema": "qwen35-layer-output-cotangents-v1",
                        "provenance": provenance,
                        "loss_scale": loss_scale,
                        "normalization_multiplier": 1.0 / loss_scale,
                        "outputs": outputs,
                        "raw_cotangents": cotangents,
                        "normalized_cotangents": {
                            index: value.float() / loss_scale
                            for index, value in cotangents.items()
                        },
                    },
                    stream,
                )
    finally:
        for handle in forward_handles + tensor_handles:
            handle.remove()


def compare_layer_cotangents(hf, reference, *, atol=0.005, rtol=0.02):
    """Compare HF-relative tolerance in reverse layer order, no tolerance edits."""
    assert hf["loss_scale"] == 1 and reference["loss_scale"] == 1 / 8
    for key in (
        "initial_manifest_sha256",
        "initial_layer0_sha256",
        "rank",
        "step",
        "tokens",
    ):
        assert hf["provenance"][key] == reference["provenance"][key], key
    assert (
        set(hf["raw_cotangents"]) == set(reference["raw_cotangents"]) == set(range(40))
    )
    comparisons = {}
    for index in reversed(range(40)):
        left = hf["raw_cotangents"][index].float()
        right = reference["raw_cotangents"][index].float() * 8
        assert left.shape == right.shape
        assert hf["outputs"][index] == reference["outputs"][index]
        finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        delta = (left - right).abs()
        failed = delta > atol + rtol * left.abs()
        comparisons[index] = {
            "exact": torch.equal(left, right),
            "finite": finite,
            "passed": finite and not bool(failed.any()),
            "failed_elements": int(failed.sum()),
            "elements": left.numel(),
            "max_abs": float(delta.max()) if finite else None,
            "rms": float(delta.square().mean().sqrt()) if finite else None,
        }
    return {
        "layers_reverse_order": comparisons,
        "first_nonexact_from_output": next(
            (i for i, v in comparisons.items() if not v["exact"]), None
        ),
        "first_failed_from_output": next(
            (i for i, v in comparisons.items() if not v["passed"]), None
        ),
        "numerical_all_passed": all(v["passed"] for v in comparisons.values()),
    }
