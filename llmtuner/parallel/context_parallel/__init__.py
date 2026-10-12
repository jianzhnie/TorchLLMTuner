"""Context parallelism: the CP flex kernel, input sharding, and wiring.

``context_parallel.py`` is the public entry and wires CP onto a model.
``attention.py`` holds attention redistribution, while
``input_sharding.py`` owns batch and mask sharding.
"""

from .attention import CPFlexKernel, DualCPFlexKernel
from .context_parallel import apply_cp
from .input_sharding import (
    shard_attention_mask_for_cp,
    shard_batch_for_cp,
    shard_batch_for_tp,
    shard_padding_mask_for_cp,
    shard_padding_mask_for_tp,
)

__all__ = [
    "CPFlexKernel",
    "DualCPFlexKernel",
    "apply_cp",
    "shard_attention_mask_for_cp",
    "shard_batch_for_cp",
    "shard_batch_for_tp",
    "shard_padding_mask_for_cp",
    "shard_padding_mask_for_tp",
]
