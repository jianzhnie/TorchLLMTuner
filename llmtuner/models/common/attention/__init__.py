"""Shared attention primitives for explicit models and context parallelism.

Import the defining leaf directly: ``qkv`` owns packed QKV projection geometry,
``masks`` owns document/position masks, ``eager_flex`` exposes an explicit eager
Flex reference, and ``backend_names`` defines backend selection constants.
Model-specific attention math and device execution retain separate owners.
This package intentionally re-exports no symbols.
"""
