"""Inspect an installed experimental AscendC runtime without importing Torch.

This source/capability diagnostic never initializes an NPU or proves lowering.
Use a separate actual-model run to validate compilation, kernels and numerics.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

EXPECTED_SOURCES = {
    "npu.py": "48d49fdfcc734b20f12997c08e0919dead241412694e4fb84392fd25bb2f1d7f",
    "asc_overrides.py": (
        "297e042eda2c4c5c16608c0d9b1d6e3e48d6cb95b4db75f796f06364c82a446e"
    ),
}


def inspect_site(site: Path, environment: dict[str, str]) -> dict:
    root = site / "torch_npu" / "_inductor"
    errors = []
    sources = {}
    for name, expected in EXPECTED_SOURCES.items():
        path = root / "ascendc" / name
        digest = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
        sources[name] = {"file": str(path), "sha256": digest, "expected": expected}
        if digest != expected:
            errors.append(f"Unsupported installed source: {name}")
    entry = root / "__init__.py"
    declares_ascendc = False
    if entry.is_file():
        tree = ast.parse(entry.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                declares_ascendc |= any(
                    isinstance(key, ast.Constant) and key.value == "ascendc"
                    for key in node.keys
                )
    if not declares_ascendc:
        errors.append("Installed backend loader does not declare ascendc")
    marker_path = root / "ascendc" / "_tlt_compat_build.json"
    marker = json.loads(marker_path.read_text()) if marker_path.is_file() else None
    if marker is None or marker.get("variant") != (
        "experimental-ascendc-bf16-transpose-mask-v1"
    ):
        errors.append(
            "Missing versioned fork build receipt (an overlay is insufficient)"
        )
    elif marker.get("formal_repository_acceptance") is not False:
        errors.append("Unexpected acceptance claim in compiler build receipt")
    for name, value in {
        "TORCHINDUCTOR_NPU_BACKEND": "ascendc",
        "TORCHINDUCTOR_COALESCE_TILING_ANALYSIS": "0",
    }.items():
        if environment.get(name) != value:
            errors.append(f"Set {name}={value} before importing Torch")
    return {
        "source_capability_ready": not errors,
        "errors": errors,
        "installed_sources": sources,
        "loader_declares_ascendc": declares_ascendc,
        "build_receipt": marker,
        "runtime_lowering_proved": False,
        "native_kernel_execution_proved": False,
        "formal_repository_acceptance": False,
        "scope": "Source and before-import configuration only; no NPU initialized",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        distribution = importlib.metadata.distribution("torch_npu")
        site = Path(distribution.locate_file(""))
        result = inspect_site(site, dict(os.environ))
        result["installed_version"] = distribution.version
    except (importlib.metadata.PackageNotFoundError, OSError, ValueError) as error:
        result = {
            "source_capability_ready": False,
            "errors": [str(error)],
            "runtime_lowering_proved": False,
            "native_kernel_execution_proved": False,
            "formal_repository_acceptance": False,
        }
    with args.report.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["source_capability_ready"] else 1)


if __name__ == "__main__":
    main()
