"""Replay a pinned, trusted ASC graph on CPU and validate generated host fields.

This invokes the real installed AutoFuse scheduler/codegen. It does not import
Torch, initialize an NPU, run generated kernels, or prove model numerics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import traceback
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def duplicate_fields(header: str) -> list[dict]:
    """Check fields within each generated struct, never across distinct structs."""
    duplicates = []
    for match in re.finditer(
        r"BEGIN_TILING_DATA_DEF_T\((\w+)\)(.*?)END_TILING_DATA_DEF_T",
        header,
        re.DOTALL,
    ):
        seen: set[str] = set()
        for field in re.findall(
            r"TILING_DATA_FIELD_DEF_T(?:_STRUCT)?\([^,\n]+,\s*(\w+)\)",
            match[2],
        ):
            if field in seen:
                duplicates.append({"struct": match[1], "field": field})
            seen.add(field)
    return duplicates


def compiler_libraries() -> dict[str, str]:
    paths = {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if line.split()[-1].startswith("/")
        and any(
            name in line for name in ("libaihac", "libgraph_af", "libgraph_base_af")
        )
    }
    return {name: sha256(Path(name)) for name in sorted(paths)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--vector-cores", type=int, required=True)
    parser.add_argument("--ub-bytes", type=int, required=True)
    parser.add_argument("--cann", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-host", action="store_true")
    parser.add_argument("--soc-version")
    parser.add_argument("--graph-name")
    parser.add_argument("--cxx11-abi", choices=(0, 1), type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "CPU real scheduler/codegen and host header compile only",
        "passed": False,
        "model_numerics_proved": False,
        "kernel_execution_proved": False,
        "formal_repository_acceptance": False,
        "platform": {
            "arch": args.arch,
            "vector_cores": args.vector_cores,
            "ub_bytes": args.ub_bytes,
        },
    }
    try:
        if args.graph.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("Captured graph exceeds 4 MiB limit")
        source = args.graph.read_bytes()
        before = hashlib.sha256(source).hexdigest()
        report["source_pre"] = before
        if before != args.sha256:
            raise ValueError("Graph source SHA mismatch")
        if args.vector_cores <= 0 or args.ub_bytes <= 0:
            raise ValueError("Platform limits must be explicit positive values")
        # The replay environment supplies CANN; this import never imports Torch.
        from autofuse import pyautofuse

        host_compile = None
        if args.compile_host:
            if not args.soc_version or not args.graph_name:
                raise ValueError(
                    "Full host compile requires --soc-version/--graph-name"
                )
            from autofuse.compile_adapter import host_compile

        report["pyautofuse"] = {
            "path": pyautofuse.__file__,
            "sha256": sha256(Path(pyautofuse.__file__)),
        }
        report["compiler_libraries_pre"] = compiler_libraries()
        pyautofuse.ascir.utils.set_platform(args.arch, args.vector_cores, args.ub_bytes)
        scope: dict = {}
        exec(compile(source, str(args.graph), "exec"), scope)
        report["codegen_completed"] = True
        artifacts = {}
        for name in ("tiling_def", "host_impl", "device_impl"):
            value = scope[name]
            if not isinstance(value, str) or not value or len(value) > 32 * 1024**2:
                raise ValueError(f"Invalid generated artifact: {name}")
            path = args.output / f"{name}.txt"
            path.write_text(value)
            artifacts[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        report["artifacts"] = artifacts
        if not re.search(r"BEGIN_TILING_DATA_DEF_T\(\w+\)", scope["tiling_def"]):
            raise ValueError("Unsupported generated tiling struct schema")
        report["duplicate_fields"] = duplicate_fields(scope["tiling_def"])
        command = [
            "g++",
            "-std=c++17",
            "-fsyntax-only",
            "-x",
            "c++",
            "-I" + str(args.cann / "aarch64-linux/ascendc/include/highlevel_api"),
            str(args.output / "tiling_def.txt"),
        ]
        report["host_compile_command"] = command
        with (args.output / "host_compile.log").open("wb") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        report["host_compile_exit"] = result.returncode
        report["passed"] = not report["duplicate_fields"] and result.returncode == 0
        if report["passed"] and host_compile is not None:
            output_library = args.output / "tiling.so"
            options = [
                f"--output_file={output_library}",
                f"--graph_name={args.graph_name}",
                "--force_unknown=False",
                f"--soc_version={args.soc_version}",
                "--config_file=None",
                f"--compile_options=-D_GLIBCXX_USE_CXX11_ABI={args.cxx11_abi}",
            ]
            report["full_host_compile_options"] = options
            report["passed"] = False
            host_compile(scope["tiling_def"], scope["host_impl"], options)
            if not output_library.is_file() or output_library.stat().st_size == 0:
                raise ValueError("Host compiler returned without a nonempty library")
            report["full_host_library"] = {
                "bytes": output_library.stat().st_size,
                "sha256": sha256(output_library),
            }
            report["passed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        report["source_post"] = sha256(args.graph) if args.graph.is_file() else None
        report["source_prepost_equal"] = (
            report.get("source_pre") == report["source_post"]
        )
        report["compiler_libraries_post"] = compiler_libraries()
        report["compiler_libraries_prepost_equal"] = (
            report.get("compiler_libraries_pre") == report["compiler_libraries_post"]
        )
        report["passed"] &= report["source_prepost_equal"]
        report["passed"] &= report["compiler_libraries_prepost_equal"]
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
