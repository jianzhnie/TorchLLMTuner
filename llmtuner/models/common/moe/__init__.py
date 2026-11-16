"""Shared MoE routing, expert computation, layout contracts and balancing.

Dispatch observations live beside their forward boundary in ``experts.py``.
Token dispatch implementations belong to ``parallel/expert_parallel``.
Import public types from their defining modules; this package has no re-exports.
"""
