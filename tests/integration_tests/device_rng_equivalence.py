"""Selected-device RNG checkpoint roundtrip and invalid-state rejection.

Run locally: PYTHONPATH=. python tests/integration_tests/device_rng_equivalence.py mps
Run CUDA: PYTHONPATH=. torchrun --nproc_per_node=2 \
    tests/integration_tests/device_rng_equivalence.py cuda

Checks device default generators, not DTensor random kernels or training.
The integration runner uses CPU by default; device evidence requires an
explicit device argument. CUDA uses LOCAL_RANK and independent rank seeds.
"""

from __future__ import annotations

import argparse
import os
import pickle
import random

import numpy as np
import torch
import torch.distributed as dist

from llmtuner.components.checkpointer.random_state import RNGCheckpointState
from llmtuner.utils.seeding import preserve_rng_state


def check_device_rng(device: torch.device) -> None:
    """Check the selected rank's streams without leaving RNG changes behind."""
    with preserve_rng_state(device):
        torch.manual_seed(101 + int(os.environ.get("RANK", "0")))
        checkpoint = RNGCheckpointState(device)
        saved = checkpoint.state_dict()
        if dist.is_initialized():
            assert checkpoint.key == f"rank_{dist.get_rank()}"
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, saved)
            assert (
                len({next(iter(state)) for state in gathered}) == dist.get_world_size()
            )
            assert len({next(iter(state.values())) for state in gathered}) == len(
                gathered
            )
        expected = torch.rand(17, device=device).cpu()
        checkpoint.load_state_dict(saved)
        torch.testing.assert_close(
            torch.rand(17, device=device).cpu(), expected, rtol=0, atol=0
        )
        check_invalid_state(checkpoint, saved, device)


def check_invalid_state(checkpoint, saved, device: torch.device) -> None:
    """Reject malformed device state without mutating any live stream."""
    module = torch.get_device_module(device)
    before_device = module.get_rng_state(device) if device.type != "cpu" else None
    before_cpu = torch.get_rng_state().clone()
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    payload = pickle.loads(saved[checkpoint.key])
    payload["cpu" if device.type == "cpu" else "device"] = torch.zeros(
        3, dtype=torch.uint8
    )
    try:
        checkpoint.load_state_dict({checkpoint.key: pickle.dumps(payload)})
    except RuntimeError:
        pass
    else:
        raise AssertionError("invalid RNG state was accepted")
    assert torch.equal(torch.get_rng_state(), before_cpu)
    assert random.getstate() == before_python
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == before_numpy[0]
    np.testing.assert_array_equal(actual_numpy[1], before_numpy[1])
    assert actual_numpy[2:] == before_numpy[2:]
    if before_device is not None:
        assert torch.equal(module.get_rng_state(device), before_device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "device", choices=("cpu", "mps", "cuda"), default="cpu", nargs="?"
    )
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable")
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device)
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        check_device_rng(device)
        print(f"PASS {device}: exact RNG continuation and rejected-state isolation")
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
