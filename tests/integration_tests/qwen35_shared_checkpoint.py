"""Convert prepared full Qwen3.5 native text weights to an HF checkpoint on CPU.

Example in an existing environment with torch and safetensors installed::
    python tests/integration_tests/qwen35_shared_checkpoint.py \
      --artifacts /tmp/tlt_qwen35_cross_framework_v1 \
      --model /path/to/original/Qwen3.5-35B-A3B \
      --output /scratch/qwen35-shared-hf

Reads initial.json and initial-{outer,0..39}.pt without modifying them. Loads
one group at a time; never builds a model or imports an accelerator plugin.
The output is model-only initialization, not a resumable training checkpoint
or evidence of 100-step acceptance. Specify the original tokenizer directory
separately in the training recipe. No tokenizer assets are copied.

The index is written last as the completion marker. On failure the output may
contain partial files; a nonempty output is always rejected on the next run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, BinaryIO

import torch
from safetensors import safe_open
from safetensors.torch import save_file

GROUPS = ("outer", *(str(i) for i in range(40)))
OUTER_KEYS = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    value = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def stream_hash(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while block := stream.read(8 * 1024 * 1024):
        digest.update(block)
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return stream_hash(stream)


def require_hash(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"Invalid SHA256 for {name}")
    return value


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def validate_tensor(key: str, value: Any, group: str) -> None:
    expected_prefix = f"model.layers.{group}."
    if not isinstance(key, str) or not (
        key in OUTER_KEYS if group == "outer" else key.startswith(expected_prefix)
    ):
        raise ValueError(f"Non-native or incorrectly grouped key: {key!r} ({group})")
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Expected a tensor: {key}")
    if value.device.type != "cpu" or value.layout != torch.strided:
        raise ValueError(f"Expected a dense CPU tensor: {key}")
    if value.numel() == 0:
        raise ValueError(f"Empty tensor: {key}")
    # Bound temporary finite-check memory even for very large expert tensors.
    flat = value.reshape(-1)
    for chunk in flat.split(1024 * 1024):
        if not bool(torch.isfinite(chunk).all()):
            raise ValueError(f"Nonfinite initial tensor: {key}")


def convert(artifacts: Path, model: Path, output: Path) -> dict:
    artifacts, model = artifacts.resolve(), model.resolve()
    if output.is_symlink():
        raise ValueError("Output must not be a symlink")
    output = output.resolve()
    if (
        output == artifacts
        or artifacts in output.parents
        or output in artifacts.parents
    ):
        raise ValueError("Output must be separate from the immutable prepare artifacts")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Refusing stale/nonempty output: {output}")
    manifest_path = artifacts / "initial.json"
    manifest, manifest_hash = read_json(manifest_path)
    config_path = model / "config.json"
    original_config, config_hash = read_json(config_path)
    if config_hash != require_hash(manifest.get("config_sha256"), "config"):
        raise ValueError("Original config hash differs from prepare manifest")
    if manifest.get("resume") is not False or manifest.get("original_step") != 0:
        raise ValueError("Expected original step-zero initialization, not resume")
    if not isinstance(manifest.get("source"), str) or not manifest["source"]:
        raise ValueError("Missing initialization source")
    require_hash(manifest.get("initializer_source_sha256"), "initializer source")
    files = manifest.get("files")
    names = {f"initial-{group}.pt" for group in GROUPS}
    if not isinstance(files, dict) or set(files) != names:
        raise ValueError("Manifest must contain exactly outer and all 40 layer files")
    for name, digest in files.items():
        require_hash(digest, name)
    if {path.name for path in artifacts.glob("initial-*.pt")} != names:
        raise ValueError("Prepare directory has missing or stale initial groups")
    text_config = original_config.get("text_config", original_config)
    if not isinstance(text_config, dict):
        raise ValueError("Expected native text_config object")
    if (
        text_config.get("model_type") != "qwen3_5_moe_text"
        or text_config.get("num_hidden_layers") != 40
        or text_config.get("hidden_size") != 2048
        or text_config.get("num_experts") != 256
        or text_config.get("moe_intermediate_size") != 512
    ):
        raise ValueError("Expected complete native Qwen3.5-35B-A3B text geometry")
    output.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    tensor_metadata = {}
    shard_metadata = {}
    total_size = 0
    for number, group in enumerate(GROUPS, start=1):
        name = f"initial-{group}.pt"
        # Hash and deserialize the same open file; no accelerator deserialization.
        with (artifacts / name).open("rb") as stream:
            if stream_hash(stream) != files[name]:
                raise ValueError(f"Initial file hash differs: {name}")
            stream.seek(0)
            state = torch.load(stream, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state:
            raise ValueError(f"Expected nonempty state dict: {name}")
        if group == "outer" and set(state) != OUTER_KEYS:
            raise ValueError(
                "Outer state must include native embeddings, norm and head"
            )
        for key, value in state.items():
            if key in weight_map:
                raise ValueError(f"Duplicate tensor key across groups: {key}")
            validate_tensor(key, value, group)
        shard_name = f"model-{number:05d}-of-{len(GROUPS):05d}.safetensors"
        # Contiguity changes storage only, never dtype, logical shape or key/layout.
        tensors = {key: value.contiguous() for key, value in state.items()}
        save_file(tensors, str(output / shard_name), metadata={"format": "pt"})
        with safe_open(output / shard_name, framework="pt", device="cpu") as saved:
            if set(saved.keys()) != set(state):
                raise ValueError(f"Written shard keys differ: {shard_name}")
            for key, original in state.items():
                restored = saved.get_tensor(key)
                if (
                    restored.dtype != original.dtype
                    or restored.shape != original.shape
                    or not torch.equal(restored, original)
                ):
                    raise ValueError(f"Written tensor differs: {key}")
                weight_map[key] = shard_name
                total_size += original.numel() * original.element_size()
                tensor_metadata[key] = {
                    "dtype": str(original.dtype),
                    "shape": list(original.shape),
                }
                del restored
        shard_metadata[shard_name] = {
            "sha256": file_hash(output / shard_name),
            "source_file": name,
            "source_sha256": files[name],
        }
        # Loop variables also retain tensor storage unless explicitly cleared.
        del original, value, state, tensors
    if (
        file_hash(manifest_path) != manifest_hash
        or file_hash(config_path) != config_hash
    ):
        raise ValueError("Manifest or original config changed during conversion")
    write_json(output / "config.json", text_config)
    provenance = {
        "format_version": 1,
        "purpose": "Shared step-zero text initialization; not 100-step acceptance",
        "tokenizer": "Not copied; recipe must specify original tokenizer directory",
        "source": {
            "artifacts": str(artifacts),
            "model": str(model),
            "manifest_sha256": manifest_hash,
            "manifest": manifest,
        },
        "config": {
            "original_sha256": config_hash,
            "text_config_sha256": file_hash(output / "config.json"),
        },
        "shards": shard_metadata,
        "tensors": tensor_metadata,
        "converter_sha256": file_hash(Path(__file__)),
    }
    write_json(output / "initialization_provenance.json", provenance)
    index = {
        "metadata": {
            "total_size": total_size,
            "initialization_provenance_sha256": file_hash(
                output / "initialization_provenance.json"
            ),
        },
        "weight_map": weight_map,
    }
    write_json(output / "model.safetensors.index.json", index)
    return {
        "output": str(output),
        "shards": len(GROUPS),
        "tensors": len(weight_map),
        "total_size": total_size,
        "index_sha256": file_hash(output / "model.safetensors.index.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        required=True,
        help="Existing prepare directory; read-only",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Original HF directory whose config.json was hashed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New or empty HF text checkpoint directory",
    )
    args = parser.parse_args()
    print(json.dumps(convert(args.artifacts, args.model, args.output), indent=2))


if __name__ == "__main__":
    main()
