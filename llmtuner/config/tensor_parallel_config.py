"""Tensor parallel configuration constants and validation."""

from llmtuner.errors import UnsupportedCombinationError


def validate_tensor_parallel_config(cfg) -> None:
    if not cfg.enable_sequence_parallel:
        raise UnsupportedCombinationError(
            "parallel.enable_sequence_parallel=False is not supported: "
            "llmtuner's tensor parallelism is sequence-parallel by "
            "construction. Leave it true, or set tensor_parallel_size=1."
        )


__all__ = ["validate_tensor_parallel_config"]
