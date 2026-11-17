"""Shared MoE routing, expert computation and balancing.

Dispatch observations live beside their forward boundary in ``experts.py``.
Token dispatch implementations belong to ``parallel/expert_parallel``.
EP plan bindings and conversion validation belong to that layer's ``bindings``.
Import public types from their defining modules; this package has no re-exports.
"""
