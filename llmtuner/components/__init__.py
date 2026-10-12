"""Trainer-side components: loss, metrics, tokenization, optimization, checkpoints.

The larger component families use subpackages: ``checkpointer/``, ``optimizer/``,
and ``tokenizer/``. Loss, metrics, and profiling remain cohesive modules.
Nothing here imports the trainer layer; the trainer composes these services.
"""
