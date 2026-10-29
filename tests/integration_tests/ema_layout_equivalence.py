"""Two ranks: EMA updates fixed local layouts without communication.

Also verifies real DCP restore with one rank owning no tracked EMA tensors.
Run with PYTHONPATH=. torchrun --nproc_per_node=2.
"""

from copy import deepcopy
from pathlib import Path
from tempfile import mkdtemp

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import (
    DTensor,
    Partial,
    Replicate,
    Shard,
    distribute_tensor,
)
from torch.distributed.tensor.debug import CommDebugMode

from llmtuner.components.optimizer.ema import EMA
from tests.integration_tests.harness import device, init_distributed


def check_layout(mesh, placement, size, *, buffer_only=False):
    initial = torch.arange(size, dtype=torch.float32, device=device())
    if isinstance(placement, Partial):
        weight = DTensor.from_local(initial / 2, mesh, [placement], run_check=False)
    else:
        weight = distribute_tensor(initial, mesh, [placement])
    model = torch.nn.Module()
    if buffer_only:
        model.register_parameter(
            "frozen",
            torch.nn.Parameter(torch.ones(1, device=device()), requires_grad=False),
        )
        model.register_buffer("weight", weight)
    else:
        model.register_parameter("weight", torch.nn.Parameter(weight))
    ema = EMA(
        model_parts=[model],
        decay=0.75,
        buffer_patterns=["^weight$"] if buffer_only else (),
    )
    owner = ema._buffer_optimizers[0] if buffer_only else ema._param_optimizers[0]
    assert ema.param_groups[0]["params"][0] is model.weight
    average = owner.state[model.weight]["ema_params"]
    initial_average = average.full_tensor().clone()
    with torch.no_grad():
        model.weight.to_local().add_(
            dist.get_rank() + 1 if isinstance(placement, Partial) else 2
        )
    live = model.weight.full_tensor()
    print(
        f"rank={dist.get_rank()} size={size} buffer_only={buffer_only} "
        f"live={model.weight.placements}/{tuple(model.weight.to_local().shape)} "
        f"average={average.placements}/{tuple(average.to_local().shape)}",
        flush=True,
    )
    with CommDebugMode() as debug:
        ema.step(1)
    assert debug.get_total_counts() == 0
    torch.testing.assert_close(
        average.full_tensor(), initial_average * 0.75 + live * 0.25, rtol=0, atol=0
    )
    state = owner.state[model.weight]
    replacements = [average.full_tensor()]
    if not isinstance(placement, Replicate):
        replacements.append(average.redistribute(placements=[Replicate()]))
    for replacement in replacements:
        state["ema_params"] = replacement
        before = (
            replacement.to_local().clone()
            if isinstance(replacement, DTensor)
            else replacement.clone()
        )
        with CommDebugMode() as debug:
            try:
                ema.step(2)
            except RuntimeError as error:
                assert "EMA tracked tensor layout changed" in str(error)
            else:
                raise AssertionError("EMA accepted a changed tensor layout")
        assert debug.get_total_counts() == 0
        actual = (
            replacement.to_local() if isinstance(replacement, DTensor) else replacement
        )
        torch.testing.assert_close(actual, before, rtol=0, atol=0)
        state["ema_params"] = average


def check_empty_rank_checkpoint(empty_rank):
    model = (
        torch.nn.Linear(2, 2).to(device()).requires_grad_(dist.get_rank() != empty_rank)
    )
    error = None
    try:
        ema = EMA(model_parts=[model], decay=0.5)
    except Exception as failure:
        error = str(failure)
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    assert not any(errors), f"EMA construction failed: {errors}"
    expected = deepcopy(ema.state_dict())
    assert bool(expected) == (dist.get_rank() != empty_rank)
    folder = [mkdtemp(prefix="llmtuner-empty-ema-") if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(folder, src=0)
    root = Path(folder[0])
    dcp.save({"ema": ema}, checkpoint_id=root / "checkpoint")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(2)
    ema.step(1)
    dcp.load({"ema": ema}, checkpoint_id=root / "checkpoint")
    restored = ema.state_dict()
    assert restored.keys() == expected.keys()
    for name, tensor in expected.items():
        torch.testing.assert_close(restored[name], tensor, rtol=0, atol=0)
    ema.step(2)
    final = ema.state_dict()
    live = dict(model.named_parameters())
    for name, tensor in expected.items():
        fqn = name.removeprefix("state.").removesuffix(".ema_params")
        torch.testing.assert_close(
            final[name], tensor.lerp(live[fqn], 0.5), rtol=0, atol=0
        )
    if dist.get_rank() == 0:
        print(f"Empty EMA rank {empty_rank}: DCP boundary and update exact; {root}")


def main():
    selected_device = init_distributed()
    try:
        assert dist.get_world_size() == 2
        mesh = init_device_mesh(selected_device.type, (2,))
        for size in (1, 5, 8):
            for placement in (Shard(0), Replicate(), Partial()):
                for buffer_only in (False, True):
                    check_layout(mesh, placement, size, buffer_only=buffer_only)
        for empty_rank in (0, 1):
            check_empty_rank_checkpoint(empty_rank)
        dist.barrier()
        if dist.get_rank() == 0:
            print(
                "EMA local updates are exact; layout rejection needs no communication."
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
