"""Multimodal special-token validation and identifiers."""

from __future__ import annotations

from .huggingface import HuggingFaceTokenizer


class MultiModalTokenizer(HuggingFaceTokenizer):
    """Single source of truth for multimodal special tokens.

    Requires 5 token strings, validates them against the vocabulary at init, and
    exposes both string and ID attributes (e.g. ``image_token``, ``image_id``).
    The multimodal Grain processor reads these attributes directly; the collator
    packs the IDs into a plain ``dict[str, int]`` that travels through the batch
    to the model forward. Adding a new VLM means passing 5 strings -- no
    subclassing needed.

    # TODO: All 5 fields are currently required. If a future VLM doesn't need
    # some (e.g. no video, no vision_start/end markers), consider making fields
    # optional.
    """

    # Config field prefixes that follow the {name}_token pattern.
    TOKEN_FIELDS = ("image", "video", "vision_start", "vision_end", "pad")

    def __init__(
        self,
        *,
        tokenizer_path: str,
        image_token: str,
        video_token: str,
        vision_start_token: str,
        vision_end_token: str,
        pad_token: str,
    ):
        super().__init__(tokenizer_path=tokenizer_path)

        tokens = {
            "image": image_token,
            "video": video_token,
            "vision_start": vision_start_token,
            "vision_end": vision_end_token,
            "pad": pad_token,
        }

        added_tokens = self.tokenizer.get_added_tokens_decoder()
        token_to_id = {tok.content: tok_id for tok_id, tok in added_tokens.items()}

        for name in self.TOKEN_FIELDS:
            token_str = tokens[name]
            if token_str not in token_to_id:
                raise ValueError(
                    f"Special token '{token_str}' ({name}_token) "
                    f"not found in tokenizer at '{tokenizer_path}'. "
                    f"Available added tokens: {list(token_to_id.keys())}"
                )
            setattr(self, f"{name}_token", token_str)
            setattr(self, f"{name}_id", token_to_id[token_str])
