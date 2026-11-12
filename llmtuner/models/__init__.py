"""Model registration, explicit family implementations, and shared components.

* ``registry.py`` registers model-owned config and construction hooks.
* ``deepseek/``, ``kimi/`` and ``qwen/`` contain independent model structures.
  ``explicit.py`` exposes the same registry for callers that require an
  explicit architecture.
* ``common/`` contains model-agnostic attention, MoE, feed-forward, embedding,
  RoPE, activation, and multimodal components.

Import implementations from their defining leaf modules. Package indexes do not
eagerly import model stacks or optional accelerator dependencies. External model
formats are handled by model-specific state adapters; they do not define the
production construction path.
"""
