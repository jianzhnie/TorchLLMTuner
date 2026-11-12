"""Offline packed-corpus assets independent of model and parallel harnesses."""

import hashlib
import json
import os

from tests.data_fixtures import write_tokenizer


def write_packed_corpus(
    root: str,
    *,
    num_rows: int,
    limo_path: str | None = None,
    tokenizer_path: str | None = None,
    limo_rows: int = 2,
) -> tuple[str, str]:
    """A tiny tokenizer and a JSONL corpus of multi-document rows."""
    external_tokenizer = tokenizer_path
    tokenizer_path = os.path.join(root, "tokenizer")
    corpus_path = os.path.join(root, "rows.jsonl")
    if limo_path:
        if external_tokenizer is None:
            raise ValueError("LIMO corpus requires an external tokenizer")
        with open(limo_path, "rb") as source:
            original = source.read()
        rows = original.splitlines()[:limo_rows]
        assert len(rows) == limo_rows, "LIMO check requires the full selected prefix"
        with open(corpus_path, "w") as handle:
            for raw in rows:
                record = json.loads(raw)
                assert record["question"] and record["solution"]
                handle.write(
                    json.dumps({"text": record["question"] + "\n" + record["solution"]})
                    + "\n"
                )
        print(
            f"real_corpus=LIMO rows=0:{limo_rows} source_sha256="
            f"{hashlib.sha256(original).hexdigest()} tokenizer={external_tokenizer} "
            "objective=plain_text_causal auxiliary_small_model=True",
            flush=True,
        )
        return external_tokenizer, corpus_path
    write_tokenizer(tokenizer_path)
    with open(corpus_path, "w") as handle:
        for i in range(num_rows):
            handle.write(
                json.dumps({"text": f"w{i} " + "lorem ipsum " * (i % 5 + 1)}) + "\n"
            )
    return tokenizer_path, corpus_path
