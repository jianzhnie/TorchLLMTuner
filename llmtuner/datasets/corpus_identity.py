"""Content and processing fingerprints for checkpointable corpus loaders."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from .sources import file_patterns_to_paths
from .text.recipes import DATASETS

if TYPE_CHECKING:
    from ..config.llmtuner_config import LLMTunerConfig


def corpus_data_identity(config: LLMTunerConfig, *, dataset: str, repeat: bool) -> str:
    """Fingerprint tokenizer assets, processing policy and source identity.

    Local text includes file contents. Other recipes require a caller-declared
    snapshot identifier; this cannot infer remote contents or callable semantics.
    This is a construction-time snapshot, not a file mutation lock.
    """
    data_settings = asdict(config.dataloader)
    if not data_settings["global_context_rows"]:
        data_settings.pop("global_context_rows")
    # Ranks may stage identical assets in different local directories. DCP
    # shares loader metadata, so paths cannot define content identity.
    for name in ("dataset_path", "tokenizer_path", "num_prefetch_batches"):
        data_settings.pop(name)
    if data_settings["data_identity"] is None:
        # Preserve existing local-text-v1 checkpoint identities.
        data_settings.pop("data_identity")
    settings = {
        "version": 1,
        "dataset": dataset,
        "seed": config.seed,
        "repeat": repeat,
        "data": data_settings,
    }
    if dataset not in {"local_jsonl", "local_jsonl_sft"} and dataset not in DATASETS:
        # Registering previously unregistered visual markers changes packed
        # samples with unchanged assets. Old media cursors cannot be reused.
        settings["multimodal_encoding"] = "atomic-markers-bounded-packing-v2"
    elif config.dataloader.packing == "first_fit":
        # Chunk boundaries now respect row capacity as well as context length.
        # Grain's pending chunk offsets from the old graph are not portable.
        settings["text_packing"] = "bounded-first-fit-v1"
    tokenizer_json = Path(config.dataloader.tokenizer_path) / "tokenizer.json"
    if tokenizer_json.is_file():
        with tokenizer_json.open(encoding="utf-8") as handle:
            tokenizer_settings = json.load(handle)
        if any(
            tokenizer_settings.get(name) is not None
            for name in ("padding", "truncation")
        ):
            # Older loaders inherited saved batch policies. Their cursor can
            # describe a different packed token stream even with identical files.
            settings["tokenizer_encoding"] = "unrestricted-samples-v1"
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    local_text = dataset in {"local_jsonl", "local_jsonl_sft"}
    if not local_text and config.dataloader.data_identity is None:
        raise ValueError("registered corpus identity requires dataloader.data_identity")
    files = (
        list(file_patterns_to_paths((config.dataloader.dataset_path,)))
        if local_text
        else []
    )
    tokenizer_path = Path(config.dataloader.tokenizer_path)
    for name in (
        "tokenizer.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        path = tokenizer_path / name
        digest.update(name.encode())
        if path.is_file():
            files.append(str(path))
        else:
            digest.update(b"absent")
    for filename in files:
        file_digest = hashlib.sha256()
        with open(filename, "rb") as handle:
            while chunk := handle.read(1024 * 1024):
                file_digest.update(chunk)
        digest.update(file_digest.digest())
    return (
        "local-text-v1:" if local_text else "declared-corpus-v1:"
    ) + digest.hexdigest()
