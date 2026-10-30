"""Four ranks on CPU/Gloo or accelerators: TP2/EP2 checkpoint layouts."""

import tempfile
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn

from llmtuner.components.checkpointer.model_state import ModelCheckpointState
from llmtuner.components.optimizer.ema import EMA
from llmtuner.components.optimizer.optimizer import OptimizersContainer
from llmtuner.config.optimizer_config import OptimizerConfig, ParamGroupConfig
from llmtuner.parallel.expert_parallel.checkpointing import (
    expert_shard_map,
    gather_expert_state,
    load_expert_state,
    mark_experts_ep_sharded,
)
from llmtuner.parallel.tensor_parallel.checkpointing import (
    gather_tensor_parallel_state,
    load_tensor_parallel_state,
    restore_disjoint_tp_ep_state,
    shard_disjoint_tp_ep_state,
    validate_disjoint_tp_ep_checkpoint_ownership,
)
from tests.integration_tests.harness import init_distributed

device = init_distributed(timeout=timedelta(seconds=90))
try:
    rank = dist.get_rank()
    groups = {
        tuple(ranks): dist.new_group(ranks)
        for ranks in ([0, 1], [2, 3], [0, 2], [1, 3])
    }
    tp = groups[(0, 1) if rank < 2 else (2, 3)]
    ep = groups[(0, 2) if rank % 2 == 0 else (1, 3)]
    model = nn.Module()
    full = torch.arange(4 * 8 * 3, dtype=torch.float32, device=device).reshape(4, 8, 3)
    expert = full.chunk(2, 0)[rank // 2]
    local = torch.cat([half.chunk(2, 1)[rank % 2] for half in expert.chunk(2, 1)], 1)
    model.register_parameter("weight", nn.Parameter(local.clone()))
    model._tp_checkpoint_layouts = {"weight": (1, True, tp)}
    mark_experts_ep_sharded(model, ep)
    shards = expert_shard_map([model])
    try:
        validate_disjoint_tp_ep_checkpoint_ownership([model])
    except ValueError as error:
        assert "disjoint TP and EP parameters" in str(error)
    else:
        raise AssertionError("Overlapping TP/EP ownership must be rejected")
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
    separate = nn.Module()
    dense_full = torch.arange(24.0, device=device).reshape(8, 3)
    separate.dense = nn.Parameter(dense_full.chunk(2, 0)[rank % 2].clone())
    separate.experts = nn.Module()
    expert_full = torch.arange(96.0, device=device).reshape(4, 8, 3)
    separate.experts.weight = nn.Parameter(expert_full.chunk(2, 0)[rank // 2].clone())
    separate._tp_checkpoint_layouts = {"dense": (0, False, tp)}
    mark_experts_ep_sharded(separate.experts, ep)
    chunks = shard_disjoint_tp_ep_state(separate.state_dict(), model_parts=[separate])
    assert chunks["dense"].shape == (8, 3)
    assert chunks["dense"].chunks[0][0] == (rank % 2 * 4, 0)
    assert chunks["experts.weight"].shape == (4, 8, 3)
    assert chunks["experts.weight"].chunks[0][0] == (rank // 2 * 2, 0, 0)
    paths = [tempfile.mkdtemp(prefix="tp-ep-disjoint-") if rank == 0 else None]
    dist.broadcast_object_list(paths)
    dcp.save(chunks, checkpoint_id=paths[0])
    complete = {
        "dense": torch.empty_like(dense_full),
        "experts.weight": torch.empty_like(expert_full),
    }
    dcp.load(complete, checkpoint_id=paths[0])
    torch.testing.assert_close(complete["dense"], dense_full, atol=0, rtol=0)
    torch.testing.assert_close(complete["experts.weight"], expert_full, atol=0, rtol=0)
    saved_chunks = {key: value.detach().clone() for key, value in chunks.items()}
    before = {key: value.clone() for key, value in separate.state_dict().items()}
    for failing_rank in range(dist.get_world_size()):
        invalid = dict(saved_chunks)
        if rank == failing_rank:
            invalid["experts.weight"] = saved_chunks["dense"]
        try:
            restore_disjoint_tp_ep_state(
                invalid, local_state=separate.state_dict(), model_parts=[separate]
            )
        except ValueError as error:
            assert rank == failing_rank
            assert "chunk layout differs" in str(error)
        except RuntimeError as error:
            assert rank != failing_rank
            assert "Joint TP/EP checkpoint restore failed on another rank" in str(error)
        else:
            raise AssertionError("Invalid expert layout must fail before copying")
        for key, value in separate.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        dist.barrier()
    missing = dict(saved_chunks)
    del missing["experts.weight"]
    try:
        restore_disjoint_tp_ep_state(
            missing, local_state=separate.state_dict(), model_parts=[separate]
        )
    except ValueError as error:
        assert "requires chunked source storage" in str(error)
    else:
        raise AssertionError("Missing expert source must fail before copying")
    for key, value in separate.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    with torch.no_grad():
        for parameter in separate.parameters():
            parameter.fill_(float("nan"))
        restored = restore_disjoint_tp_ep_state(
            saved_chunks, local_state=separate.state_dict(), model_parts=[separate]
        )
        separate.load_state_dict(restored)
    torch.testing.assert_close(
        separate.dense, dense_full.chunk(2, 0)[rank % 2], atol=0, rtol=0
    )
    torch.testing.assert_close(
        separate.experts.weight,
        expert_full.chunk(2, 0)[rank // 2],
        atol=0,
        rtol=0,
    )
    with torch.no_grad():
        for parameter in separate.parameters():
            parameter.fill_(float("nan"))
    dcp.load(chunks, checkpoint_id=paths[0])
    torch.testing.assert_close(
        separate.dense, dense_full.chunk(2, 0)[rank % 2], atol=0, rtol=0
    )
    torch.testing.assert_close(
        separate.experts.weight,
        expert_full.chunk(2, 0)[rank // 2],
        atol=0,
        rtol=0,
    )
    wrapper = ModelCheckpointState(
        separate, sharded_tensor_parallel=True, sharded_expert_parallel=True
    )
    dcp.save({"model": wrapper}, checkpoint_id=paths[0] + "-wrapper")
    with torch.no_grad():
        for parameter in separate.parameters():
            parameter.fill_(float("nan"))
    dcp.load({"model": wrapper}, checkpoint_id=paths[0] + "-wrapper")
    torch.testing.assert_close(
        separate.dense, dense_full.chunk(2, 0)[rank % 2], atol=0, rtol=0
    )
    torch.testing.assert_close(
        separate.experts.weight,
        expert_full.chunk(2, 0)[rank // 2],
        atol=0,
        rtol=0,
    )
    optimizer = OptimizersContainer(
        OptimizerConfig(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 0.01},
                )
            ],
        ),
        model_parts=[separate],
        sharded_tensor_parallel=True,
        sharded_expert_parallel=True,
    )
    for parameter in separate.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad()
    training_state = {"model": wrapper, "optimizer": optimizer}
    dcp.save(training_state, checkpoint_id=paths[0] + "-training")
    for parameter in separate.parameters():
        parameter.grad = torch.full_like(parameter, 0.25)
    optimizer.step()
    optimizer.zero_grad()
    expected = {key: value.clone() for key, value in separate.state_dict().items()}
    expected_optimizer = {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for key, value in optimizer.state_dict().items()
    }
    cold_optimizer = OptimizersContainer(
        OptimizerConfig(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 0.01},
                )
            ],
        ),
        model_parts=[separate],
        sharded_tensor_parallel=True,
        sharded_expert_parallel=True,
    )
    assert all(not inner.state for inner in cold_optimizer.optimizers)
    optimizer = cold_optimizer
    training_state["optimizer"] = optimizer
    dcp.load(training_state, checkpoint_id=paths[0] + "-training")
    for parameter in separate.parameters():
        parameter.grad = torch.full_like(parameter, 0.25)
    optimizer.step()
    optimizer.zero_grad()
    for key, value in separate.state_dict().items():
        torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
    for key, value in optimizer.state_dict().items():
        reference = expected_optimizer[key]
        if hasattr(value, "chunks"):
            for (_, actual), (_, saved) in zip(
                value.chunks, reference.chunks, strict=True
            ):
                torch.testing.assert_close(actual, saved, atol=0, rtol=0)
        elif isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, reference, atol=0, rtol=0)
        else:
            assert value == reference
    ema = EMA(
        model_parts=[separate],
        half_life_fraction=0.5,
        sharded_tensor_parallel=True,
        sharded_expert_parallel=True,
    )
    with torch.no_grad():
        for parameter in separate.parameters():
            parameter.add_(1)
    ema.step(1)
    dcp.save({"ema": ema}, checkpoint_id=paths[0] + "-ema")
    ema.step(2)
    expected_ema = {
        key: value.detach().clone() for key, value in ema.state_dict().items()
    }
    cold_ema = EMA(
        model_parts=[separate],
        half_life_fraction=0.5,
        sharded_tensor_parallel=True,
        sharded_expert_parallel=True,
    )
    dcp.load({"ema": cold_ema}, checkpoint_id=paths[0] + "-ema")
    cold_ema.step(2)
    for key, value in cold_ema.state_dict().items():
        reference = expected_ema[key]
        if hasattr(value, "chunks"):
            for (offset, actual), (saved_offset, saved) in zip(
                value.chunks, reference.chunks, strict=True
            ):
                assert offset == saved_offset
                torch.testing.assert_close(actual, saved, atol=0, rtol=0)
        elif isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, reference, atol=0, rtol=0)
        else:
            assert value == reference
    print(
        f"rank {rank}: joint TP/EP EMA cold restore and next update exact", flush=True
    )

    ep_only_wrapper = ModelCheckpointState(separate, sharded_expert_parallel=True)
    ep_only_optimizer = OptimizersContainer(
        OptimizerConfig(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 0.01},
                )
            ],
        ),
        model_parts=[separate],
        sharded_expert_parallel=True,
    )
    for parameter in separate.parameters():
        parameter.grad = torch.ones_like(parameter)
    ep_only_optimizer.step()
    ep_only_optimizer.zero_grad()
    ep_only_path = paths[0] + "-ep-only-training"
    dcp.save(
        {"model": ep_only_wrapper, "optimizer": ep_only_optimizer},
        checkpoint_id=ep_only_path,
    )
    for parameter in separate.parameters():
        parameter.grad = torch.full_like(parameter, 0.25)
    ep_only_optimizer.step()
    ep_only_optimizer.zero_grad()
    expected = {key: value.clone() for key, value in separate.state_dict().items()}
    expected_optimizer = {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for key, value in ep_only_optimizer.state_dict().items()
    }

    cold_separate = nn.Module()
    cold_separate.dense = nn.Parameter(torch.zeros_like(separate.dense))
    cold_separate.experts = nn.Module()
    cold_separate.experts.weight = nn.Parameter(
        torch.zeros_like(separate.experts.weight)
    )
    cold_separate._tp_checkpoint_layouts = {"dense": (0, False, tp)}
    mark_experts_ep_sharded(cold_separate.experts, ep)
    cold_wrapper = ModelCheckpointState(cold_separate, sharded_expert_parallel=True)
    cold_optimizer = OptimizersContainer(
        OptimizerConfig(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 0.9},
                )
            ],
        ),
        model_parts=[cold_separate],
        sharded_expert_parallel=True,
    )
    assert all(not inner.state for inner in cold_optimizer.optimizers)
    dcp.load(
        {"model": cold_wrapper, "optimizer": cold_optimizer},
        checkpoint_id=ep_only_path,
    )
    for parameter in cold_separate.parameters():
        parameter.grad = torch.full_like(parameter, 0.25)
    cold_optimizer.step()
    cold_optimizer.zero_grad()
    for key, value in cold_separate.state_dict().items():
        torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
    for key, value in cold_optimizer.state_dict().items():
        reference = expected_optimizer[key]
        if hasattr(value, "chunks"):
            for (offset, actual), (saved_offset, saved) in zip(
                value.chunks, reference.chunks, strict=True
            ):
                assert offset == saved_offset
                torch.testing.assert_close(actual, saved, atol=0, rtol=0)
        elif isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, reference, atol=0, rtol=0)
        else:
            assert value == reference

    ep_only_ema = EMA(
        model_parts=[separate],
        half_life_fraction=0.5,
        sharded_expert_parallel=True,
    )
    with torch.no_grad():
        for parameter in separate.parameters():
            parameter.add_(1)
    ep_only_ema.step(1)
    ep_only_ema_path = paths[0] + "-ep-only-ema"
    dcp.save({"ema": ep_only_ema}, checkpoint_id=ep_only_ema_path)
    ep_only_ema.step(2)
    expected_ema = {
        key: value.detach().clone() for key, value in ep_only_ema.state_dict().items()
    }
    cold_ema = EMA(
        model_parts=[cold_separate],
        half_life_fraction=0.5,
        sharded_expert_parallel=True,
    )
    dcp.load({"ema": cold_ema}, checkpoint_id=ep_only_ema_path)
    with torch.no_grad():
        for parameter in cold_separate.parameters():
            parameter.add_(1)
    cold_ema.step(2)
    for key, value in cold_ema.state_dict().items():
        reference = expected_ema[key]
        if hasattr(value, "chunks"):
            for (offset, actual), (saved_offset, saved) in zip(
                value.chunks, reference.chunks, strict=True
            ):
                assert offset == saved_offset
                torch.testing.assert_close(actual, saved, atol=0, rtol=0)
        elif isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, reference, atol=0, rtol=0)
        else:
            assert value == reference
    print(
        f"rank {rank}: EP-only model/optimizer/EMA cold restore exact",
        flush=True,
    )
    dist.barrier()
finally:
    dist.destroy_process_group()
