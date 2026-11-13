"""Explicit Qwen CP capability boundary.

The FSDPTurbo CP communication stack is not part of TorchLLMTuner's HF model
contract. Ordinary construction stays available; requesting these entry points
fails explicitly instead of silently running a different parallel algorithm.
"""

from llmtuner.errors import UnsupportedCombinationError


def unsupported_qwen_cp(*args, **kwargs):
    raise UnsupportedCombinationError(
        "Explicit Qwen CP requires the TorchLLMTuner CP installer"
    )


shard_fused_sequence_inputs = unsupported_qwen_cp
ulysses_context_parallel_attention = unsupported_qwen_cp
cp_compute_lm_loss = unsupported_qwen_cp
gather_forward_split_backward = unsupported_qwen_cp
split_forward_gather_backward = unsupported_qwen_cp
cal_split_sizes = unsupported_qwen_cp
split_tensor_to_group = unsupported_qwen_cp


class GatedDeltaNet:
    @staticmethod
    def forward(*args, **kwargs):
        unsupported_qwen_cp()
