"""Tokenizer protocol and chat-template rendering."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any


class BaseTokenizer(ABC):
    """Interface every tokenizer in the data pipeline satisfies.

    ``eos_id`` is deliberately public and mutable rather than a property: the
    chat processor reads it at construction to decide whether it can locate a
    response boundary at all.
    """

    eos_id: int | None

    def __init__(self) -> None:
        self.eos_id = None
        self._chat_template = None

    @abstractmethod
    def encode(self, *args, **kwargs) -> list[int]: ...

    @abstractmethod
    def decode(self, *args, **kwargs) -> str: ...

    @abstractmethod
    def get_vocab_size(self) -> int: ...

    def set_chat_template(self, template: str) -> None:
        """Compile and store a Jinja chat template."""
        import jinja2
        import jinja2.ext
        import jinja2.sandbox

        def raise_exception(msg):
            raise jinja2.exceptions.TemplateError(msg)

        def tojson(
            x, ensure_ascii=False, indent=None, separators=None, sort_keys=False
        ):
            return json.dumps(
                x,
                ensure_ascii=ensure_ascii,
                indent=indent,
                separators=separators,
                sort_keys=sort_keys,
            )

        def strftime_now(fmt):
            from datetime import datetime

            return datetime.now().strftime(fmt)

        env = jinja2.sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=[jinja2.ext.loopcontrols],
        )
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = strftime_now
        env.filters["tojson"] = tojson
        self._chat_template = env.from_string(template)

    def apply_chat_template(
        self, messages: Sequence[Mapping[str, Any]], **kwargs
    ) -> str:
        """Render messages through the Jinja chat template. Returns formatted text.

        Messages should be a list of dicts with "role" and "content" keys, e.g.
        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}].

        Additional template variables (e.g. add_generation_prompt, tools) can be
        passed as kwargs.
        """
        if self._chat_template is None:
            raise ValueError("No chat template set. Call set_chat_template() first.")
        return self._chat_template.render(messages=messages, **kwargs)
