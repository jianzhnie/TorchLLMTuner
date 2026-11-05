"""Pure token-throughput and model-compute metrics shared by logs and benchmarks.

FLOPs/token describes model forward/backward work, excluding AC recomputation.
Peak FLOPs is a per-device reference for the declared precision. Callers own
measurement synchronization and decide whether throughput is global or per-device;
``world_size`` must describe exactly the devices serving that throughput.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.training_config import TrainingConfig


def token_throughput(
    num_tokens: int, elapsed_seconds: float, *, replication_degree: int = 1
) -> float:
    """Normalize counted tokens by time and duplicated TP/CP/PP token streams.

    For rank-local loader counts, replication_degree=TP*CP*PP yields average
    per-device throughput. For unique global tokens use replication_degree=1.
    EP overlays the dense mesh and is never another multiplier or divisor.
    """
    return num_tokens / (elapsed_seconds * replication_degree)


def configured_tokens_per_step(config: TrainingConfig) -> int:
    """Global padded tokens consumed by one complete optimizer update.

    Each accumulation group reads a full global batch; PP microbatches only
    split that batch and TP/CP/EP do not introduce independent input tokens.
    """
    return (
        config.global_batch_size
        * config.max_seq_len
        * config.gradient_accumulation_steps
    )


def model_tflops(flops_per_token: int, tokens_per_second: float) -> float:
    """Model TFLOPS for the same global or per-device scope as tokens/s."""
    return flops_per_token * tokens_per_second / 1e12


def compute_utilization(
    *,
    tokens_per_second: float,
    flops_per_token: int,
    peak_flops_per_device: float,
    world_size: int,
) -> dict[str, float | None]:
    """Convert global throughput into TFLOPS and peak-compute utilization."""
    if flops_per_token <= 0 or peak_flops_per_device <= 0 or world_size <= 0:
        return {"tflops": None, "peak_tflops": None, "utilization_percent": None}
    tflops = model_tflops(flops_per_token, tokens_per_second)
    peak_tflops = peak_flops_per_device * world_size / 1e12
    return {
        "tflops": tflops,
        "peak_tflops": peak_tflops,
        "utilization_percent": 100 * tflops / peak_tflops,
    }
