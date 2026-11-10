"""Small explicit-NPU public Generator RNG falsification probe.

No production replay, no CPU RNG and no exactness extrapolation to Pro. Public
Generator set_offset is a stream offset; element-offset hypotheses are tested,
not asserted. --cpu-contract executes only nonrandom protocol guards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None


SHAPES = ((3, 14, 17), (2, 12, 16), (1, 1, 257), (1025,))
MAX_ELEMENTS = 4096


def checked_shape(shape: tuple[int, ...]) -> int:
    if not shape or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("Positive shape required")
    size = math.prod(shape)
    if size > MAX_ELEMENTS:
        raise ValueError("Small probe only; full Pro allocation forbidden")
    return size


def comparison(a: torch.Tensor, b: torch.Tensor) -> dict:
    if a.dtype != b.dtype or a.numel() != b.numel():
        raise ValueError("Comparison schema mismatch")
    aa, bb = a.flatten(), b.flatten()
    raw_a, raw_b = aa.view(torch.uint8), bb.view(torch.uint8)
    different = raw_a != raw_b
    return {
        "bitwise_exact": torch.equal(raw_a, raw_b),
        "different_bytes": int(different.sum().item()),
        "elements": aa.numel(),
        "finite": bool(torch.isfinite(aa).all() and torch.isfinite(bb).all()),
        "reference_nonzero": int(torch.count_nonzero(aa).item()),
        "candidate_nonzero": int(torch.count_nonzero(bb).item()),
        "reference_sha256": hashlib.sha256(raw_a.numpy().tobytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(raw_b.numpy().tobytes()).hexdigest(),
    }


def cpu_contract() -> dict:
    for shape in SHAPES:
        checked_shape(shape)
    for invalid in ((384, 6144, 7168), (0,), (-1,), (True,)):
        try:
            checked_shape(invalid)
        except ValueError:
            continue
        raise AssertionError("Invalid shape accepted")
    value = torch.tensor([0.0, 1.0, -2.0])
    assert comparison(value, value.clone())["bitwise_exact"]
    changed = value.clone()
    changed[0] = -0.0
    assert not comparison(value, changed)["bitwise_exact"]
    changed[1] = 2.0
    assert comparison(value, changed)["different_bytes"] > 0
    return {
        "cpu_contract_passed": True,
        "scope": "no random generation or NPU initialization",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-contract", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "execution_completed": False,
        "scope": "small NPU RNG probe; NOT Pro bounded replay or training acceptance",
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": [],
    }
    try:
        report["contract"] = cpu_contract()
        if args.cpu_contract:
            report["execution_completed"] = True
            return
        if torch_npu is None:
            raise ValueError("Actual torch_npu required")
        torch.npu.set_device(0)
        device = torch.device("npu:0")
        global_pre = torch.npu.get_rng_state().clone()
        report["versions"] = {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        }
        report["api"] = (
            "public torch.Generator(device=npu), manual_seed/get_offset/set_offset/"
            "get_state/set_state, Tensor.normal_"
        )
        payload = []

        def draw(shape, dtype, generator):
            checked_shape(tuple(shape))
            return torch.empty(shape, dtype=dtype, device=device).normal_(
                0.0, 0.02, generator=generator
            )

        for dtype in (torch.float32, torch.bfloat16):
            for shape in SHAPES:
                size = checked_shape(shape)
                for seed in (0, 281):
                    for initial_offset in (0, 12):
                        generator = torch.Generator(device=device).manual_seed(seed)
                        generator.set_offset(initial_offset)
                        pre = generator.get_state().clone()
                        baseline = draw(shape, dtype, generator).cpu().flatten()
                        post = generator.get_state().clone()
                        final_offset = generator.get_offset()
                        next_draw = draw((19,), dtype, generator).cpu()
                        for mode in (
                            "same_shape_restored",
                            "flat_same_numel",
                            "split_sequential",
                            "split_element_offset_hypothesis",
                            "split_aligned_offset_hypothesis",
                        ):
                            candidate_generator = torch.Generator(device=device)
                            candidate_generator.set_state(pre)
                            row = {
                                "dtype": str(dtype),
                                "shape": shape,
                                "seed": seed,
                                "initial_offset": initial_offset,
                                "reference_final_offset": final_offset,
                                "mode": mode,
                            }
                            try:
                                if mode == "same_shape_restored":
                                    result = (
                                        draw(shape, dtype, candidate_generator)
                                        .cpu()
                                        .flatten()
                                    )
                                elif mode == "flat_same_numel":
                                    result = draw(
                                        (size,), dtype, candidate_generator
                                    ).cpu()
                                else:
                                    # Odd boundaries and tails deliberately challenge
                                    # the unproven flat-element offset hypothesis.
                                    lengths = (
                                        (17, 240, size - 257)
                                        if size > 257
                                        else (17, size - 17)
                                    )
                                    parts, start = [], 0
                                    for length in lengths:
                                        if mode == "split_element_offset_hypothesis":
                                            candidate_generator.set_offset(
                                                initial_offset + start
                                            )
                                        prefix = 0
                                        if mode == "split_aligned_offset_hypothesis":
                                            prefix = start % 4
                                            candidate_generator.set_offset(
                                                initial_offset + start - prefix
                                            )
                                        part = draw(
                                            (length + prefix,),
                                            dtype,
                                            candidate_generator,
                                        ).cpu()
                                        parts.append(part[prefix:])
                                        start += length
                                    result = torch.cat(parts)
                                row["candidate_final_offset"] = (
                                    candidate_generator.get_offset()
                                )
                                candidate_post = candidate_generator.get_state().clone()
                                following = draw(
                                    (19,), dtype, candidate_generator
                                ).cpu()
                                row.update(comparison(baseline, result))
                                row["post_state_exact"] = torch.equal(
                                    post, candidate_post
                                )
                                row["next_draw_exact"] = comparison(
                                    next_draw, following
                                )["bitwise_exact"]
                                payload.append(
                                    {
                                        "row": row.copy(),
                                        "reference": baseline,
                                        "candidate": result,
                                        "pre": pre,
                                        "post": post,
                                        "candidate_post": candidate_post,
                                        "next": next_draw,
                                        "candidate_next": following,
                                    }
                                )
                                if mode == "same_shape_restored" and not (
                                    row["bitwise_exact"]
                                    and row["post_state_exact"]
                                    and row["next_draw_exact"]
                                ):
                                    raise AssertionError("Positive control failed")
                            except RuntimeError as error:
                                if mode != "split_element_offset_hypothesis":
                                    raise
                                row["hypothesis_error"] = str(error)
                            report["rows"].append(row)
        torch.npu.synchronize()
        if not torch.equal(global_pre, torch.npu.get_rng_state()):
            raise AssertionError("Explicit-generator experiment changed global RNG")
        report["global_generator_unchanged"] = True
        torch.save(payload, args.output / "payload.pt")
        report["payload_sha256"] = hashlib.sha256(
            (args.output / "payload.pt").read_bytes()
        ).hexdigest()
        report["counterexamples"] = sum(
            "hypothesis_error" in row
            or not (
                row["bitwise_exact"]
                and row["post_state_exact"]
                and row["next_draw_exact"]
            )
            for row in report["rows"]
        )
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    else:
        report["execution_completed"] = True
    finally:
        with (args.output / "report.json").open("x") as stream:
            json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
