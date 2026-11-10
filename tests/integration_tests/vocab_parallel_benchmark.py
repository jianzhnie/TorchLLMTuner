"""Forward benchmark and analytical communication comparison for VP and TP.

Environment: VP_BENCH_VOCAB, VP_BENCH_HIDDEN, VP_BENCH_TOKENS (global),
VP_BENCH_WARMUP, VP_BENCH_STEPS. Run with two ranks. HBM is measured per rank;
communication bytes are analytical off-rank send+receive totals, not a trace.
"""

import json
import os
import resource
import statistics
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh

from llmtuner.parallel.vocab_parallel import (
    VocabParallelEmbedding,
    VocabParallelPlan,
    vocab_fully_shard_modules,
    vocab_parallelize_modules,
)
from tests.integration_tests.harness import init_distributed


class TokenModel(nn.Module):
    def __init__(self, vocab: int, hidden: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden, dtype=dtype)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(ids)


def _synchronize(device: torch.device) -> None:
    if device.type != "cpu":
        torch.accelerator.synchronize()


def _measure(model: nn.Module, ids: torch.Tensor, device: torch.device):
    warmup = int(os.environ.get("VP_BENCH_WARMUP", "5"))
    steps = int(os.environ.get("VP_BENCH_STEPS", "30"))
    for _ in range(warmup):
        model(ids).sum()
    _synchronize(device)
    dist.barrier()
    backend = getattr(torch, device.type, None) if device.type != "cpu" else None
    if backend is not None:
        backend.reset_peak_memory_stats()
    timings = []
    for _ in range(steps):
        start = time.perf_counter()
        model(ids).sum()
        _synchronize(device)
        timings.append((time.perf_counter() - start) * 1000)
    samples = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(samples, timings)
    per_rank = [statistics.median(rank_times) for rank_times in samples]
    slowest_rank = max(range(len(per_rank)), key=per_rank.__getitem__)
    peak_hbm = (
        backend.max_memory_allocated()
        if backend is not None
        else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    peaks = [None for _ in samples]
    dist.all_gather_object(peaks, peak_hbm)
    return {
        "slowest_rank": slowest_rank,
        "slowest_rank_median_ms": per_rank[slowest_rank],
        "p50_ms": statistics.median(samples[slowest_rank]),
        "p90_ms": sorted(samples[slowest_rank])[int(0.9 * (len(timings) - 1))],
        "peak_memory_mib": max(peaks) / 2**20,
    }


def main() -> None:
    device = init_distributed()
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    vocab = int(os.environ.get("VP_BENCH_VOCAB", "32768"))
    hidden = int(os.environ.get("VP_BENCH_HIDDEN", "512"))
    tokens = int(os.environ.get("VP_BENCH_TOKENS", "4096"))
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    item_size = torch.tensor([], dtype=dtype).element_size()
    local_tokens = tokens // world
    mesh = init_device_mesh(device.type, (2, 1), mesh_dim_names=("vp", "vfsdp"))
    group = mesh["vp"].get_group()
    generator = torch.Generator(device="cpu").manual_seed(19)
    global_ids = torch.randint(0, vocab, (tokens,), generator=generator).to(device)
    local_ids = global_ids.chunk(world)[rank].contiguous()
    routed_ids = torch.randint(0, vocab, (local_tokens,), generator=generator).to(
        device
    )

    torch.manual_seed(7)
    model = TokenModel(vocab, hidden, dtype).to(device)
    plan = VocabParallelPlan(apply_modules=["embedding"], gradient_divide_factor=1.0)
    vocab_parallelize_modules(model, mesh["vp"], plan)
    vocab_fully_shard_modules(model, mesh["vfsdp"], plan)
    routed = _measure(model, routed_ids, device)
    routed_bytes = (
        2 * world * 8
        + 2 * local_tokens * 8 * (1 - 1 / world)
        + 2 * local_tokens * hidden * item_size * (1 - 1 / world)
    )
    routed.update(
        name="independent_vp_alltoallv",
        analytical_communication_mib=routed_bytes / 2**20,
    )

    source = TokenModel(vocab, hidden, dtype).to(device).embedding
    tp_embedding = VocabParallelEmbedding(
        source.weight.chunk(world)[rank].contiguous(),
        source=source,
        vocab_start_index=rank * (vocab // world),
        vocab_end_index=(rank + 1) * (vocab // world),
        tp_group=group,
    )
    tp = _measure(tp_embedding, local_ids, device)
    tp_bytes = tokens * 8 * (1 - 1 / world) + 2 * tokens * hidden * item_size * (
        1 - 1 / world
    )
    tp.update(
        name="tp_vocab_gather_allreduce",
        analytical_communication_mib=tp_bytes / 2**20,
    )
    if rank == 0:
        print(
            json.dumps(
                {
                    "device": device.type,
                    "world_size": world,
                    "vocab": vocab,
                    "hidden": hidden,
                    "global_tokens": tokens,
                    "dtype": str(dtype),
                    "results": [routed, tp],
                },
                indent=2,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
