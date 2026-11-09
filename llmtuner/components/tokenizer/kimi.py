"""Explicit official Kimi K3 tokenizer backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from transformers import AutoTokenizer

from .base import BaseTokenizer


class KimiK3Tokenizer(BaseTokenizer):
    """Adapt the official slow tokenizer while preserving special-token boundaries."""

    def __init__(self, *, tokenizer_path: str) -> None:
        super().__init__()
        self.tokenizer_path = tokenizer_path
        self.backend = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True, local_files_only=True
        )
        if type(self.backend).__name__ != "TikTokenTokenizer":
            raise ValueError("kimi_k3 tokenizer backend requires TikTokenTokenizer")
        self.bos_id = self.backend.bos_token_id
        self.eos_id = self.backend.eos_token_id

    def encode(
        self,
        text: str,
        *,
        add_bos: bool | None = None,
        add_eos: bool | None = None,
    ) -> list[int]:
        token_ids = self.backend.encode(text, allow_special_tokens=True)
        if add_bos:
            token_ids.insert(0, self.bos_id)
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        return self.backend.decode(token_ids, **kwargs)

    def get_vocab_size(self) -> int:
        return len(self.backend)

    def apply_chat_template(
        self, messages: Sequence[Mapping[str, Any]], **kwargs: Any
    ) -> str:
        kwargs.setdefault("add_generation_prompt", True)
        rendered = self.backend.apply_chat_template(messages, tokenize=False, **kwargs)
        expected = self.backend.apply_chat_template(messages, tokenize=True, **kwargs)
        actual = self.encode(rendered)
        if actual != expected:
            raise ValueError(
                "Kimi rendered-text encoding differs from segmented chat encoding; "
                "input text may contain special-token-like content"
            )
        return rendered
