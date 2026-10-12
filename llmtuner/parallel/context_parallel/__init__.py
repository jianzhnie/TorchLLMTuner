"""Context parallelism: the CP flex kernel, input sharding, and wiring.

``context_parallel.py`` is the public entry and wires CP onto a model.
``cp_kernel.py`` holds the attention redistribution itself, while
``input_shard.py`` owns batch and mask sharding.
"""

from .context_parallel import apply_cp
from .cp_kernel import CPFlexKernel
from .input_shard import (
    shard_attention_mask_for_cp,
    shard_batch_for_cp,
    shard_batch_for_tp,
    shard_padding_mask_for_cp,
    shard_padding_mask_for_tp,
)

__all__ = [
    "CPFlexKernel",
    "apply_cp",
    "shard_attention_mask_for_cp",
    "shard_batch_for_cp",
    "shard_batch_for_tp",
    "shard_padding_mask_for_cp",
    "shard_padding_mask_for_tp",
]
