"""Model registration, family implementations, HF adaptation, and components.

* ``architecture.py`` registers custom HF config and model construction hooks.
* ``kimi/kimi_k3/`` and ``qwen/qwen3_5/`` contain model-specific behavior.
  Family directories group independent model structures; they do not re-export
  implementations or import sibling structures.
* ``hf/`` adapts HF decoders to the trainer's input, parallel, initialization,
  and checkpoint contracts. FLOPs estimation lives in ``llmtuner.utils.flops``.
* ``common/`` contains model-agnostic attention, MoE, feed-forward, embedding,
  RoPE, activation, and multimodal components.

Import implementations from their defining leaf modules. Package indexes do not
eagerly import model stacks or optional accelerator dependencies.
"""
