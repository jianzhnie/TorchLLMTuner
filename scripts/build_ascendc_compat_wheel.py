"""Build a versioned experimental TorchNPU wheel without installing packages.

Only the pinned official wheel and the reviewed patch series are accepted.
The result is an external compiler fork, not TorchLLMTuner backend acceptance.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

BASE_VERSION = "2.14.0.dev20260918"
FORK_VERSION = BASE_VERSION + "+tlt.ascendc1"
BASE_COMMIT = "ec693356f4644c5f50c990e78b8fd9ba7737a4d3"
BASE_WHEEL_SHA256 = "dabc2e33edfaada05e5edacc8bf8a6380d1543435af22a0d90d88ec4cb4affb6"
PATCHES = {
    "0001-bf16-transpose-prologue-staging.patch": (
        "b75310b59a9d97696a565d5d2c9e45570497bfb15aa1618bc1638a3f6b7224c6"
    ),
    "0003-bf16-bool-mask-not-eq.patch": (
        "8d5463e7788e70ee633e4b8a42030a140d8c58e6e53a4d60c4426ad9d55bd546"
    ),
}
SOURCES = {
    "npu.py": (
        "70b679b2420d13ac8123652579502ad0606e731a62fb1087e91f4c1deaa5fcc1",
        "48d49fdfcc734b20f12997c08e0919dead241412694e4fb84392fd25bb2f1d7f",
    ),
    "asc_overrides.py": (
        "8ee2f4c0e51fba719e0ec170d68b7d8cff289193d0930e2dd249838e0f2b24c2",
        "297e042eda2c4c5c16608c0d9b1d6e3e48d6cb95b4db75f796f06364c82a446e",
    ),
}
PREFIX = "torch_npu/_inductor/ascendc/"
MARKER = PREFIX + "_tlt_compat_build.json"


def file_sha256(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            state.update(chunk)
    return state.hexdigest()


def repack(
    source: Path, output: Path, replacements: dict[str, bytes], marker: dict
) -> None:
    """Stream unmodified wheel members, normalize timestamps and rebuild RECORD."""
    old_info = f"torch_npu-{BASE_VERSION}.dist-info/"
    new_info = f"torch_npu-{FORK_VERSION}.dist-info/"
    records = []
    with zipfile.ZipFile(source) as original:
        names = original.namelist()
        if len(set(names)) != len(names):
            raise ValueError("Duplicate wheel members")
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe wheel member")
        if any(name.endswith(("RECORD.jws", "RECORD.p7s")) for name in names):
            raise ValueError("Signed wheel requires an explicit re-signing workflow")
        edits = dict(replacements)
        metadata = original.read(old_info + "METADATA")
        expected = f"Version: {BASE_VERSION}\n".encode()
        if metadata.count(expected) != 1:
            raise ValueError("Unexpected official metadata version")
        edits[old_info + "METADATA"] = metadata.replace(
            expected, f"Version: {FORK_VERSION}\n".encode()
        )
        version = original.read("torch_npu/version.py")
        expected_version = f"__version__ = '{BASE_VERSION}'".encode()
        if version.count(expected_version) != 1:
            raise ValueError("Unexpected official Python version")
        edits["torch_npu/version.py"] = version.replace(
            expected_version, f"__version__ = '{FORK_VERSION}'".encode()
        )
        edits[MARKER] = json.dumps(marker, sort_keys=True, indent=2).encode()
        # Exclusive output prevents replacing a previously reviewed artifact.
        with output.open("xb") as raw_output:
            with zipfile.ZipFile(
                raw_output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
            ) as built:
                for name in sorted(set(names) | set(edits)):
                    if name.endswith("/") or name == old_info + "RECORD":
                        continue
                    target = name.replace(old_info, new_info, 1)
                    info = zipfile.ZipInfo(target, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = (
                        original.getinfo(name).external_attr
                        if name in names
                        else 0o644 << 16
                    )
                    state, size = hashlib.sha256(), 0
                    if name in edits:
                        reader = io.BytesIO(edits[name])
                    else:
                        reader = original.open(name)
                    with reader, built.open(info, "w", force_zip64=True) as writer:
                        while chunk := reader.read(1024 * 1024):
                            writer.write(chunk)
                            state.update(chunk)
                            size += len(chunk)
                    digest = base64.urlsafe_b64encode(state.digest()).rstrip(b"=")
                    records.append((target, "sha256=" + digest.decode(), size))
                record_path = new_info + "RECORD"
                text = io.StringIO(newline="")
                csv.writer(text, lineterminator="\n").writerows(
                    [*records, (record_path, "", "")]
                )
                info = zipfile.ZipInfo(record_path, (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                built.writestr(info, text.getvalue().encode())


def build(wheel: Path, destination: Path) -> Path:
    if file_sha256(wheel) != BASE_WHEEL_SHA256:
        raise ValueError("Official input wheel SHA256 mismatch")
    patch_root = Path(__file__).with_name("ascendc_patches")
    marker = {
        "variant": "experimental-ascendc-bf16-transpose-mask-v1",
        "official_wheel_sha256": BASE_WHEEL_SHA256,
        "embedded_base_commit": BASE_COMMIT,
        "upstream_fix_commit": None,
        "formal_repository_acceptance": False,
        "patches": PATCHES,
        "sources": {PREFIX + name: hashes[1] for name, hashes in SOURCES.items()},
        "required_environment": {
            "TORCHINDUCTOR_NPU_BACKEND": "ascendc",
            "TORCHINDUCTOR_COALESCE_TILING_ANALYSIS": "0",
        },
    }
    with tempfile.TemporaryDirectory(prefix="tlt_ascendc_wheel_") as temporary:
        tree = Path(temporary)
        with zipfile.ZipFile(wheel) as original:
            for name, (expected, _) in SOURCES.items():
                data = original.read(PREFIX + name)
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError(f"Unexpected official source: {name}")
                path = tree / PREFIX / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        for name, expected in PATCHES.items():
            patch = patch_root / name
            if file_sha256(patch) != expected:
                raise ValueError(f"Patch SHA256 mismatch: {name}")
            subprocess.run(
                ["git", "apply", "--check", str(patch)], cwd=tree, check=True
            )
            subprocess.run(["git", "apply", str(patch)], cwd=tree, check=True)
        replacements = {}
        for name, (_, expected) in SOURCES.items():
            data = (tree / PREFIX / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Patched source mismatch: {name}")
            replacements[PREFIX + name] = data
        destination.mkdir(parents=True, exist_ok=True)
        filename = wheel.name.replace(BASE_VERSION, FORK_VERSION, 1)
        if filename == wheel.name:
            raise ValueError("Unexpected wheel filename")
        output = destination / filename
        # A partial output is retained on failure; it must never be installed.
        repack(wheel, output, replacements, marker)
    report = {**marker, "output": str(output), "output_sha256": file_sha256(output)}
    with output.with_suffix(".json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if shutil.which("git") is None:
        parser.error("git is required for exact patch application")
    print(build(args.wheel, args.output_dir))


if __name__ == "__main__":
    main()
