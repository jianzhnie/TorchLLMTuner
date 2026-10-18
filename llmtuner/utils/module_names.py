"""Canonical model names shared by optimizer and parallel checkpoint state.

This module has no torch dependency and does not define checkpoint format keys.
"""

_WRAPPER_SEGMENTS = ("_checkpoint_wrapped_module", "_orig_mod")


def canonical_fqn(
    name: str, *, wrapper_segments: tuple[str, ...] = _WRAPPER_SEGMENTS
) -> str:
    """Remove complete wrapper segments at any depth of a dotted model name.

    For example, ``layers.0._checkpoint_wrapped_module.weight`` becomes
    ``layers.0.weight``. This keeps serialized names stable when activation
    checkpointing is enabled or disabled.
    """
    return ".".join(
        segment for segment in name.split(".") if segment not in wrapper_segments
    )
