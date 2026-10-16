"""Run with four CPU/Gloo ranks: TP2/EP2 packed checkpoint layout."""

from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn

from llmtuner.parallel.expert_parallel.checkpointing import (
    expert_shard_map,
    gather_expert_state,
    load_expert_state,
    mark_experts_ep_sharded,
)
from llmtuner.parallel.tensor_parallel.checkpointing import (
    gather_tensor_parallel_state,
    load_tensor_parallel_state,
)

dist.init_process_group("gloo", timeout=timedelta(seconds=30))
try:
    rank = dist.get_rank()
    groups = {
        tuple(ranks): dist.new_group(ranks)
        for ranks in ([0, 1], [2, 3], [0, 2], [1, 3])
    }
    tp = groups[(0, 1) if rank < 2 else (2, 3)]
    ep = groups[(0, 2) if rank % 2 == 0 else (1, 3)]
    model = nn.Module()
    full = torch.arange(4 * 8 * 3, dtype=torch.float32).reshape(4, 8, 3)
    expert = full.chunk(2, 0)[rank // 2]
    local = torch.cat([half.chunk(2, 1)[rank % 2] for half in expert.chunk(2, 1)], 1)
    model.register_parameter("weight", nn.Parameter(local.clone()))
    model._tp_checkpoint_layouts = {"weight": (1, True, tp)}
    mark_experts_ep_sharded(model, ep)
    shards = expert_shard_map([model])
    for key in ("weight", "state.weight.exp_avg", "state.weight.ema_params"):
        state = gather_expert_state(
            {key: local.clone()}, shard_map=shards, model_parts=[model]
        )
        state = gather_tensor_parallel_state(state, model_parts=[model])
        torch.testing.assert_close(state[key], full, rtol=0, atol=0)
        state = load_tensor_parallel_state(state, model_parts=[model])
        state = load_expert_state(state, shard_map=shards, model_parts=[model])
        torch.testing.assert_close(state[key], local, rtol=0, atol=0)
    print(f"rank {rank}: packed TP/EP global state and restore exact", flush=True)
    dist.barrier()
finally:
    dist.destroy_process_group()
