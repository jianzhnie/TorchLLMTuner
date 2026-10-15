"""Text sample processing: plain text and supervised chat.

Vendored from torchtitan ``hf_datasets/text_datasets.py``. A processor is
constructed with its build context and takes the few per-dataset knobs
(``text_fn``, ``messages_fn``) as keyword arguments.

``ChatProcessor`` is the one place in the data layer that can fail on a whole
dataset rather than a sample. Locating the prompt/response boundary by
re-rendering the prompt assumes the chat template emits it as a textual prefix
of the full render, which is a property of the template and the tokenizer
together. When that assumption breaks, every sample breaks with it, so this
raises instead of dropping -- dropping would silently train on a fraction of
the data while the loss curve looked healthy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from llmtuner.errors import ConfigError

from ...components.loss import IGNORE_INDEX
from ...utils.logging import get_logger
from ..contracts import DatasetBuildContext, SampleProcessor, TextSequence

logger = get_logger(__name__)

__all__ = [
    "ChatProcessor",
    "TextProcessor",
]


def read_text(sample: dict[str, Any]) -> str:
    """The default ``text_fn``: read the sample's ``"text"`` field."""
    return sample["text"]


class TextProcessor(SampleProcessor):
    """Tokenizes plain text into next-token input and label pairs."""

    def __init__(
        self,
        *,
        context: DatasetBuildContext,
        text_fn: Callable[[dict[str, Any]], str] = read_text,
    ) -> None:
        self._tokenizer = context.tokenizer
        self._text_fn = text_fn

    def __call__(
        self, sample: dict[str, Any], rng: np.random.Generator
    ) -> TextSequence | None:
        del rng
        input_ids = np.asarray(
            self._tokenizer.encode(self._text_fn(sample), add_bos=True, add_eos=True),
            dtype=np.int64,
        )
        if len(input_ids) < 2:
            return None
        return TextSequence(
            input_ids=input_ids[:-1],
            labels=input_ids[1:],
        )


def require_token_prefix(full_tokens: list[int], prompt_tokens: list[int]) -> None:
    """Raise if prompt_tokens is not an exact prefix of full_tokens.

    ChatProcessor locates the prompt/response boundary by re-rendering the
    prompt alone and requiring it to tokenize to a prefix of the full
    conversation. That holds only when rendering the prompt with
    ``add_generation_prompt=True`` produces a textual prefix of the full render
    and the tokenizer does not merge characters across that seam. Both are
    properties of the template and tokenizer together rather than of an
    individual sample.

    Raise instead of dropping the sample: a mismatch means the label boundary
    is unknown, and because the cause is systematic it would fire for most
    samples, so dropping would silently train on a fraction of the dataset.
    The overflow path drops because an oversized example really is per-sample.
    """
    if full_tokens[: len(prompt_tokens)] != prompt_tokens:
        raise ValueError(
            "Prompt tokens are not an exact prefix of the full conversation "
            "tokens, so the prompt/response boundary cannot be located. "
            "ChatProcessor requires that rendering the prompt with "
            "add_generation_prompt=True yields a textual prefix of the full "
            "render, and that the tokenizer does not merge characters across "
            "that seam. A template that rewrites earlier turns when later ones "
            "are present, or turn separators that only merge in context, break "
            "this assumption."
        )


class ChatProcessor(SampleProcessor):
    """Tokenizes chat samples and masks labels outside assistant responses.

    Two mutually exclusive paths, chosen at construction. ``renderer=None``
    (the default) is the single-turn chat-template path: the prompt/response
    boundary is located by prefix re-tokenization, and the tokenizer must
    have an EOS id. A ``renderers`` renderer switches to the multi-turn
    path: the renderer owns the token sequence and the per-token loss mask
    (every assistant turn supervised, prompts and non-content tokens
    masked), so no EOS inference or prefix property is needed. A renderer
    and the template path never combine -- the renderer replaces the
    template render entirely.
    """

    def __init__(
        self,
        *,
        context: DatasetBuildContext,
        messages_fn: Callable[[dict[str, Any]], list[dict[str, str]]],
        renderer: Any | None = None,
    ) -> None:
        if renderer is None and context.tokenizer.eos_id is None:
            raise ValueError(
                "Tokenizer does not have an eos_id set. "
                "ChatProcessor requires a tokenizer with a valid EOS token."
            )
        self._tokenizer = context.tokenizer
        self._eos_id = context.tokenizer.eos_id
        self._max_context_length = context.max_context_length
        self._messages_fn = messages_fn
        self._renderer = renderer
        self._logged_first_sample = False

    @staticmethod
    def _validate_messages(messages: list[dict[str, str]]) -> None:
        """Validate that messages are a single-turn [user, assistant] pair."""
        # Multi-turn conversations go through the renderer path instead: the
        # per-turn spans it computes survive templates that rewrite earlier
        # turns, which prefix re-rendering cannot.
        if len(messages) != 2:
            raise ValueError(
                f"Expected single-turn [user, assistant], got {len(messages)} messages"
            )
        if messages[0]["role"] != "user":
            raise ValueError(
                f"First message must be 'user', got '{messages[0]['role']}'"
            )
        if messages[1]["role"] != "assistant":
            raise ValueError(
                f"Second message must be 'assistant', got '{messages[1]['role']}'"
            )

    def _is_too_long(self, num_tokens: int) -> bool:
        """Whether a rendered sample has to be dropped, logging the reason.

        The count is the render's token count; the sequence keeps one token
        fewer as labels, hence the ``- 1``.

        TODO(data-sft-overflow): Consider truncating oversized examples
        instead. Causal loss remains valid for the retained response prefix.
        """
        if num_tokens - 1 <= self._max_context_length:
            return False
        logger.debug(
            "Dropping sample: token count exceeds max_context_length=%d",
            self._max_context_length,
        )
        return True

    def _log_first_sample(self, text: str) -> None:
        """Print the first full render once, so a template can be eyeballed.

        Nothing else in the loop shows what the tokenizer was actually fed.
        """
        if self._logged_first_sample:
            return
        logger.info(f"[ChatProcessor] First sample full:\n{text}")
        self._logged_first_sample = True

    @staticmethod
    def _shift_to_next_token(full_tokens) -> tuple[np.ndarray, np.ndarray]:
        """Split one token stream into next-token-aligned input and labels."""
        tokens = np.asarray(full_tokens, dtype=np.int64)
        return tokens[:-1], tokens[1:].copy()

    def _tokenize_single_turn(
        self, messages: list[dict[str, str]]
    ) -> TextSequence | None:
        """Tokenize a single-turn sample and mask prompt labels.

        Returns None if the rendered sample exceeds ``max_context_length``,
        avoiding training on truncated responses.

        Uses prefix re-tokenization to find the prompt/response token
        boundary, avoiding BPE merge errors.
        """
        self._validate_messages(messages)

        # The tokenizer defaults add_generation_prompt=True (torchtitan parity);
        # a full-conversation render must not grow a trailing generation prompt.
        full_text = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=False
        )
        # Strip extra newline and ensure the sequence ends with EOS without duplicates
        full_text = full_text.rstrip("\n")
        full_tokens = self._tokenizer.encode(full_text, add_bos=True, add_eos=False)
        if not full_tokens:
            raise ConfigError(
                "chat template rendered to zero tokens; check the template "
                "and the messages field"
            )
        if full_tokens[-1] != self._eos_id:
            full_tokens.append(self._eos_id)

        self._log_first_sample(full_text)
        if self._is_too_long(len(full_tokens)):
            return None

        # Find prompt/response boundary by tokenizing just the user message
        # with add_generation_prompt=True.
        prompt_text = self._tokenizer.apply_chat_template(
            messages[:1], add_generation_prompt=True
        )
        prompt_tokens = self._tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
        require_token_prefix(full_tokens, prompt_tokens)

        input_ids, labels = self._shift_to_next_token(full_tokens)
        labels[: max(len(prompt_tokens) - 1, 0)] = IGNORE_INDEX
        return TextSequence(
            input_ids=input_ids,
            labels=labels,
        )

    def _tokenize_with_renderer(
        self, messages: list[dict[str, str]]
    ) -> TextSequence | None:
        """Tokenize a multi-turn conversation through the renderer.

        Semantics (kept from upstream): ``build_training_sample`` with
        ``ensure_final_stop=True`` renders the whole conversation and
        guarantees a terminal stop token; the returned ``loss_mask`` marks
        the tokens the model is trained on (assistant content), and it is
        shifted with the labels because label ``j`` predicts token
        ``j + 1``.
        """
        from renderers import build_training_sample

        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError("Chat samples must end with an assistant message.")
        # TODO(data-sft-supervision): Support per-turn loss weighting.
        rendered = build_training_sample(
            self._renderer, messages, ensure_final_stop=True
        )
        if rendered.multi_modal_data is not None:
            raise ValueError("ChatProcessor supports text-only samples.")
        if len(rendered.token_ids) < 2:
            raise ConfigError(
                "chat renderer must produce at least two tokens for "
                "next-token supervision; check the renderer and messages"
            )

        # Decoded only when it will be printed: the render is per sample.
        if not self._logged_first_sample:
            self._log_first_sample(
                self._tokenizer.decode(
                    list(rendered.token_ids), skip_special_tokens=False
                )
            )
        if self._is_too_long(len(rendered.token_ids)):
            return None

        input_ids, labels = self._shift_to_next_token(rendered.token_ids)
        labels[~np.asarray(rendered.loss_mask[1:], dtype=bool)] = IGNORE_INDEX
        return TextSequence(
            input_ids=input_ids,
            labels=labels,
        )

    def __call__(
        self, sample: dict[str, Any], rng: np.random.Generator
    ) -> TextSequence | None:
        del rng
        messages = self._messages_fn(sample)
        if self._renderer is not None:
            return self._tokenize_with_renderer(messages)
        return self._tokenize_single_turn(messages)
