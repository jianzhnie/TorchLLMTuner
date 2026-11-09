"""Default HF tokenizer must retain pre-Kimi loader checkpoint identities."""

import hashlib
import json
from dataclasses import asdict

from llmtuner.config import DataloaderConfig, LLMTunerConfig
from llmtuner.datasets.corpus_identity import corpus_data_identity


def test_default_tokenizer_preserves_legacy_local_identity(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"text": "hello"}\n')
    cfg = LLMTunerConfig()
    cfg.training.dataloader_config = DataloaderConfig(
        dataset="local_jsonl", dataset_path=str(corpus), tokenizer_path=str(tmp_path)
    )
    data = asdict(cfg.dataloader)
    for name in (
        "tokenizer_backend",
        "chat_template_kwargs",
        "global_context_rows",
        "dataset_path",
        "tokenizer_path",
        "num_prefetch_batches",
        "data_identity",
    ):
        data.pop(name)
    settings = {
        "version": 1,
        "dataset": "local_jsonl",
        "seed": cfg.seed,
        "repeat": True,
        "data": data,
    }
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for name in (
        "tokenizer.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        digest.update(name.encode())
        digest.update(b"absent")
    digest.update(hashlib.sha256(corpus.read_bytes()).digest())
    expected = "local-text-v1:" + digest.hexdigest()
    assert corpus_data_identity(cfg, dataset="local_jsonl", repeat=True) == expected
    cfg.dataloader.tokenizer_backend = "kimi_k3"
    kimi = corpus_data_identity(cfg, dataset="local_jsonl", repeat=True)
    assert kimi != expected
    (tmp_path / "encoding_k3.py").write_text("version = 2\n")
    assert corpus_data_identity(cfg, dataset="local_jsonl", repeat=True) != kimi
