"""Hugging Face tokenizer loading and special-token handling."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

from tokenizers import AddedToken, Tokenizer

from ...utils.logging import get_logger
from .base import BaseTokenizer

logger = get_logger(__name__)


class HuggingFaceTokenizer(BaseTokenizer):
    """A tokenizer wrapper that handles BOS/EOS token inference and encoding.

    This class loads tokenizer files and automatically infers BOS/EOS tokens from
    a configuration file (tokenizer_config.json) as well as specific formatting
    related to chat templates. Its encode method suppresses the tokenizer's own
    special tokens and adds BOS/EOS itself, controlled by the add_bos/add_eos
    arguments.

    Args:
        tokenizer_path: Path to directory containing tokenizer files.
    """

    # Following HF transformers convention (CHAT_TEMPLATE_FILE in
    # transformers/utils/hub.py), a standalone Jinja file at the model root
    # takes priority over an inline template in tokenizer_config.json. Models
    # like GPT-OSS use this pattern.
    CHAT_TEMPLATE_FILE = "chat_template.jinja"

    def __init__(self, *, tokenizer_path: str) -> None:
        super().__init__()
        self.tokenizer_path = tokenizer_path

        self.bos_id = None
        self.eos_id = None
        self.bos_token = None
        self.eos_token = None
        self.unk_token = None
        self.unk_id = None

        self.tokenizer = self._load_tokenizer_from_path(tokenizer_path)

        self._hf_config = self._load_config(
            os.path.join(tokenizer_path, "tokenizer_config.json")
        )
        if self._hf_config is None:
            logger.warning(
                "No tokenizer_config.json found at %s. "
                "Special token inference and chat template auto-loading disabled.",
                tokenizer_path,
            )

        # Infer special tokens from config (if available) and BOS/EOS behavior
        if self._hf_config is not None:
            self._infer_special_tokens()
        if self.unk_token is None:
            self.unk_token = getattr(self.tokenizer.model, "unk_token", None)
            if self.unk_token is not None:
                self.unk_id = self.tokenizer.token_to_id(self.unk_token)
        self._infer_should_add_bos_eos()

        # Auto-load chat template: standalone .jinja file takes priority
        # (e.g. GPT-OSS), then fall back to inline in tokenizer_config.json
        # (e.g. Llama3, Qwen3, DeepSeek V3).
        jinja_path = os.path.join(tokenizer_path, self.CHAT_TEMPLATE_FILE)
        if os.path.exists(jinja_path):
            with open(jinja_path, encoding="utf-8") as f:
                self.set_chat_template(f.read())
        elif self._hf_config is not None and "chat_template" in self._hf_config:
            template = self._select_chat_template(self._hf_config["chat_template"])
            if template is not None:
                self.set_chat_template(template)

    @staticmethod
    def _select_chat_template(template: Any) -> str | None:
        """Resolve an inline string or named template collection without IO."""
        if template is None:
            return None
        if isinstance(template, list):
            templates = {}
            for entry in template:
                if not isinstance(entry, dict) or not isinstance(
                    entry.get("name"), str
                ):
                    raise ValueError("chat_template entries must have a string name")
                name = entry["name"]
                if name in templates:
                    raise ValueError(f"duplicate chat_template name {name!r}")
                templates[name] = entry.get("template")
            template = templates
        if isinstance(template, dict):
            if "default" in template:
                template = template["default"]
            elif len(template) == 1:
                template = next(iter(template.values()))
            else:
                raise ValueError("chat_template collection requires a default entry")
        if not isinstance(template, str):
            raise ValueError("selected chat_template must be a string")
        return template

    @staticmethod
    def _load_config(config_path: str) -> dict[str, Any] | None:
        """Read an optional UTF-8 tokenizer configuration object."""
        if not os.path.exists(config_path):
            return None
        with open(config_path, encoding="utf-8") as file:
            config = json.load(file)
        if not isinstance(config, dict):
            raise ValueError(f"Tokenizer config {config_path!r} must be a JSON object")
        return config

    @staticmethod
    def _load_tokenizer_from_path(tokenizer_path: str) -> Tokenizer:
        """Load a serialized tokenizer or construct one from vocabulary files."""
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer path '{tokenizer_path}' does not exist")

        tokenizer_json_path = os.path.join(tokenizer_path, "tokenizer.json")
        vocab_txt_path = os.path.join(tokenizer_path, "vocab.txt")
        vocab_json_path = os.path.join(tokenizer_path, "vocab.json")
        merges_txt_path = os.path.join(tokenizer_path, "merges.txt")

        if os.path.exists(tokenizer_json_path):
            logger.info("Loading tokenizer from tokenizer.json")
            return Tokenizer.from_file(tokenizer_json_path)

        if not (os.path.exists(vocab_json_path) or os.path.exists(vocab_txt_path)):
            available_files = sorted(
                name
                for name in os.listdir(tokenizer_path)
                if os.path.isfile(os.path.join(tokenizer_path, name))
            )
            raise FileNotFoundError(
                f"No supported tokenizer files found in '{tokenizer_path}'. "
                f"Available files: {available_files}. "
                "Looking for: tokenizer.json, vocab.txt, or vocab.json "
                "(optionally with merges.txt)"
            )

        if os.path.exists(vocab_json_path):
            logger.info("Loading vocabulary from vocab.json")
            with open(vocab_json_path, encoding="utf-8") as file:
                vocab = json.load(file)
            vocab_source = "vocab.json"
        else:
            logger.info("Loading vocabulary from vocab.txt")
            vocab = {}
            with open(vocab_txt_path, encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    token = line.rstrip("\r\n")
                    # A blank line still occupies a vocabulary ID. Skipping
                    # it leaves holes and makes later IDs exceed vocab size.
                    if token in vocab:
                        raise ValueError(
                            f"Duplicate token {token!r} in {vocab_txt_path!r} "
                            f"at line {line_number}"
                        )
                    vocab[token] = line_number - 1
            vocab_source = "vocab.txt"

        from tokenizers import pre_tokenizers

        if not os.path.exists(merges_txt_path):
            from tokenizers.models import WordLevel

            logger.info("Loading WordLevel tokenizer from %s", vocab_source)
            tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
            return tokenizer

        from tokenizers import decoders, processors
        from tokenizers.models import BPE

        logger.info("Loading BPE tokenizer from %s + merges.txt", vocab_source)
        merges = []
        with open(merges_txt_path, encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(
                        f"BPE merge in {merges_txt_path!r} at line {line_number} "
                        "must contain exactly two tokens"
                    )
                merges.append((parts[0], parts[1]))

        tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        return tokenizer

    @staticmethod
    def _get_token_from_config(config: dict[str, Any], key: str) -> str | None:
        """
        Parse special tokens from config that can be either strings or dicts.
        HF tokens are stored as either ``{'bos_token': '<bos>'}`` or
        ``{'bos_token': {'content': '<bos>', ...}}``.
        """
        token = config.get(key)
        if isinstance(token, dict):
            if "content" not in token:
                raise ValueError(f"Could not parse {key} from config")
            token = token["content"]
            if not isinstance(token, str):
                raise ValueError(
                    f"Could not parse {key} from config - content must be a string"
                )
        if token is not None and not isinstance(token, str):
            raise ValueError(
                f"Could not parse {key} from config - expected string or dict"
            )
        return token

    @staticmethod
    def _build_added_token(token: str, config: str | dict[str, Any]) -> AddedToken:
        """Preserve the matching and decoding flags from HF token metadata."""
        if isinstance(config, dict):
            return AddedToken(
                content=token,
                single_word=config.get("single_word", False),
                lstrip=config.get("lstrip", False),
                rstrip=config.get("rstrip", False),
                normalized=config.get("normalized", True),
                special=config.get("special", True),
            )
        return AddedToken(content=token, special=True)

    def _infer_special_tokens(self) -> None:
        """
        Read special tokens from config and add them to the underlying tokenizer.
        Store BOS/EOS tokens as class attributes since they are frequently used.

        This method handles multiple token configuration formats:
        1. Standard top-level keys (bos_token, eos_token, etc.)
        2. added_tokens_decoder dictionary (used by models like Llama 3.1)
        """
        standard_keys = [
            "bos_token",
            "eos_token",
            "pad_token",
            "unk_token",
            "sep_token",
            "cls_token",
            "mask_token",
        ]

        if not self._hf_config:
            return

        added_tokens_to_add: list[AddedToken] = []
        for key in standard_keys:
            token_str = self._get_token_from_config(self._hf_config, key)
            if key in ("bos_token", "eos_token", "unk_token"):
                setattr(self, key, token_str)
            token_config = self._hf_config.get(key)
            if token_str is not None:
                added_tokens_to_add.append(
                    self._build_added_token(token_str, token_config)
                )

        # Process added_tokens_decoder (comprehensive special token definitions)
        added_tokens_decoder = self._hf_config.get("added_tokens_decoder", {})
        if not isinstance(added_tokens_decoder, dict):
            raise ValueError("tokenizer config added_tokens_decoder must be an object")
        explicit_tokens: dict[int, AddedToken] = {}
        explicit_ids: dict[str, int] = {}
        for token_id, token_config in added_tokens_decoder.items():
            if (
                not isinstance(token_id, str)
                or not token_id.isascii()
                or not token_id.isdecimal()
            ):
                raise ValueError(f"Invalid added_tokens_decoder ID {token_id!r}")
            expected_id = int(token_id)
            token_str = self._get_token_from_config(added_tokens_decoder, token_id)
            if token_str is None:
                raise ValueError(
                    f"added_tokens_decoder entry {token_id!r} must define token content"
                )
            if expected_id in explicit_tokens or token_str in explicit_ids:
                raise ValueError(
                    f"Duplicate added_tokens_decoder ID or token: {token_id!r}, "
                    f"{token_str!r}"
                )
            explicit_tokens[expected_id] = self._build_added_token(
                token_str, token_config
            )
            explicit_ids[token_str] = expected_id

        # Validate the complete ID plan before modifying the backend. Existing
        # tokens must match; missing tokens can only be appended consecutively.
        next_id = self.tokenizer.get_vocab_size()
        for expected_id, token in sorted(explicit_tokens.items()):
            actual_id = self.tokenizer.token_to_id(token.content)
            if actual_id is None:
                actual_id = next_id
                next_id += 1
            if actual_id != expected_id:
                raise ValueError(
                    f"Cannot restore added token {token.content!r}: configured ID "
                    f"{expected_id}, backend ID {actual_id}. Token IDs must match "
                    "existing vocabulary entries or append without gaps."
                )

        # Decoder metadata takes precedence for tokens also named by standard
        # keys. Add explicit IDs first so BOS/EOS cannot occupy their slots.
        added_tokens_to_add = [
            token for _, token in sorted(explicit_tokens.items())
        ] + [
            token for token in added_tokens_to_add if token.content not in explicit_ids
        ]

        if added_tokens_to_add:
            # add_special_tokens forces special=True even for decoder entries
            # explicitly marked as ordinary tokens. add_tokens respects each flag.
            self.tokenizer.add_tokens(added_tokens_to_add)
            for token, expected_id in explicit_ids.items():
                actual_id = self.tokenizer.token_to_id(token)
                if actual_id != expected_id:
                    raise ValueError(
                        f"Added token {token!r} has ID {actual_id}; "
                        f"expected {expected_id}"
                    )

            # Update BOS/EOS token IDs after adding to tokenizer (in case they changed)
            for kind in ("bos", "eos", "unk"):
                token = getattr(self, f"{kind}_token")
                if token is not None:
                    setattr(self, f"{kind}_id", self.tokenizer.token_to_id(token))

    def _infer_should_add_bos_eos(self) -> None:
        """
        Determine the default BOS/EOS behavior for ``encode``.

        ``encode`` suppresses the tokenizer's own special tokens, so these
        defaults are the only ones applied when the caller does not pass
        add_bos/add_eos explicitly.
        """
        self.default_add_bos: bool | None = None
        self.default_add_eos: bool | None = None
        self.hf_adds_bos = False
        self.hf_adds_eos = False

        # First, determine if underlying tokenizer auto-adds BOS/EOS tokens empirically
        encoded_empty_str = self.tokenizer.encode("").ids
        if self.bos_id is not None and self.bos_id in encoded_empty_str:
            self.hf_adds_bos = True
        if self.eos_id is not None and self.eos_id in encoded_empty_str:
            self.hf_adds_eos = True

        # Check tokenizer_config.json for explicit settings
        if self._hf_config:
            for kind in ("bos", "eos"):
                key = f"add_{kind}_token"
                enabled = self._hf_config.get(key)
                if enabled is not None and not isinstance(enabled, bool):
                    raise ValueError(f"tokenizer config {key} must be boolean or null")
                setattr(self, f"default_add_{kind}", enabled)

    def encode(
        self,
        text: str,
        *,
        add_bos: bool | None = None,
        add_eos: bool | None = None,
    ) -> list[int]:
        """Encode text, applying explicit or inferred BOS/EOS settings.

        Native post-processing is disabled so each special token is added once.
        ``None`` selects the defaults inferred from the tokenizer and config.
        """
        if add_bos is None:
            add_bos = (
                self.hf_adds_bos
                if self.default_add_bos is None
                else self.default_add_bos
            )
        if add_eos is None:
            add_eos = (
                self.hf_adds_eos
                if self.default_add_eos is None
                else self.default_add_eos
            )

        token_ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if add_bos and self.bos_id is not None:
            token_ids.insert(0, self.bos_id)
        if add_eos and self.eos_id is not None:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        """Decode IDs, forwarding backend options such as skip_special_tokens."""
        return self.tokenizer.decode(token_ids, **kwargs)

    def token_to_id(self, token: str) -> int | None:
        return self.tokenizer.token_to_id(token)

    def get_vocab_size(self) -> int:
        """Return the underlying tokenizer's vocabulary size."""
        return self.tokenizer.get_vocab_size()

    def apply_chat_template(
        self, messages: Sequence[Mapping[str, Any]], **kwargs: Any
    ) -> str:
        """Render messages, auto-injecting the special tokens templates expect.

        Some HF chat templates reference ``bos_token``/``eos_token`` as Jinja
        variables; rendering without them fails or substitutes an empty string
        silently. Inject them (and the generation-prompt default, matching
        torchtitan's ``HFBackendTokenizer``) as defaults only -- any caller
        that passes the kwargs explicitly wins. SFT full-conversation renders
        must pass ``add_generation_prompt=False`` explicitly, as
        ``datasets/text/processors.py`` does.
        """
        kwargs.setdefault("bos_token", self.bos_token or "")
        kwargs.setdefault("eos_token", self.eos_token or "")
        kwargs.setdefault("add_generation_prompt", True)
        return super().apply_chat_template(messages, **kwargs)
