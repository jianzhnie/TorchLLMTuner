"""Model registration, family implementations, HF adaptation, and components.

* ``architecture.py`` registers model-owned config and construction hooks.
* ``deepseek/``, ``kimi/`` and ``qwen/`` contain independent model structures.
  ``explicit.py`` lazily constructs their opt-in HF subclasses without changing
  the default HF class resolution.
* ``hf/`` adapts HF decoders to the trainer's input, parallel, initialization,
  and checkpoint contracts. FLOPs estimation lives in ``llmtuner.utils.flops``.
* ``common/`` contains model-agnostic attention, MoE, feed-forward, embedding,
  RoPE, activation, and multimodal components.

Import implementations from their defining leaf modules. Package indexes do not
eagerly import model stacks or optional accelerator dependencies.
"""
