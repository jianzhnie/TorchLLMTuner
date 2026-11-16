"""Real distributed training and complete DCP continuation comparison.

Run with torchrun --nproc_per_node=2. Select --layout tp, dp, or ep.
No CPU override is used: the process group must match the topology device.
"""

import argparse
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist

from experiments.qwen3_moe_framework.config import ExperimentConfig, load_model_config
from experiments.qwen3_moe_framework.model import Qwen3MoeModel
from experiments.qwen3_moe_framework.trainer import ExperimentTrainer
from llmtuner.accelerator.collectives import synchronize_local_errors
from llmtuner.components.checkpointer.core.model_state import ModelCheckpointState
from llmtuner.config.parallel_config import ParallelConfig
from tests.integration_tests.harness import init_distributed


def snapshot(trainer):
    state = ModelCheckpointState(trainer.model).state_dict()
    return {
        "model": {
            name: value.full_tensor().detach().cpu().clone()
            if hasattr(value, "full_tensor")
            else value.detach().cpu().clone()
            for name, value in state.items()
        },
        "optimizer": deepcopy(trainer.optimizer.state_dict()),
    }


def assert_state_equal(expected, actual):
    assert expected.keys() == actual.keys()
    for name, value in expected.items():
        other = actual[name]
        if isinstance(value, dict):
            assert_state_equal(value, other)
        elif isinstance(value, torch.Tensor):
            if hasattr(value, "full_tensor"):
                value, other = value.full_tensor(), other.full_tensor()
            torch.testing.assert_close(value, other, atol=0, rtol=0, msg=name)
        else:
            assert value == other, name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", choices=("tp", "dp", "ep", "joint"), default="tp")
    args = parser.parse_args()
    device = init_distributed()
    path = None
    try:
        world = dist.get_world_size()
        parallel = ParallelConfig(
            tensor_parallel_size=world if args.layout in ("tp", "joint") else 1,
            data_parallel_shard_size=1 if args.layout in ("tp", "joint") else world,
            expert_parallel_size=world if args.layout in ("ep", "joint") else 1,
        )
        continuous = ExperimentTrainer(
            ExperimentConfig(parallel=parallel, batch_size=2, steps=3), device
        )
        if args.layout == "tp":
            torch.manual_seed(continuous.config.seed)
            unsharded = Qwen3MoeModel(load_model_config(None)).to(
                device=device, dtype=next(continuous.model.parameters()).dtype
            )
            ids, _, positions = continuous._batch(0)
            full_ids = torch.empty(
                (ids.shape[0], continuous.config.seq_len),
                dtype=ids.dtype,
                device=device,
            )
            pieces = [torch.empty_like(ids) for _ in range(world)]
            dist.all_gather(pieces, ids.contiguous())
            full_ids.copy_(torch.cat(pieces, dim=1))
            with torch.no_grad():
                expected_logits = unsharded(full_ids, position_ids=positions)
                local_logits = torch.cat(
                    [
                        continuous.model(
                            row.unsqueeze(0), position_ids=pos.unsqueeze(0)
                        )
                        for row, pos in zip(ids, positions, strict=True)
                    ],
                    dim=0,
                )
            start = dist.get_rank() * ids.shape[1]
            torch.testing.assert_close(
                local_logits,
                expected_logits[:, start : start + ids.shape[1]],
                atol=5e-3 if device.type == "npu" else 1e-6,
                rtol=5e-3 if device.type == "npu" else 1e-5,
            )
            del unsharded
        reference = continuous.train()
        expected = snapshot(continuous)
        del continuous
        paths = [
            tempfile.mkdtemp(prefix="qwen3_moe_framework_")
            if dist.get_rank() == 0
            else None
        ]
        dist.broadcast_object_list(paths)
        path = Path(paths[0])
        first = ExperimentTrainer(
            ExperimentConfig(parallel=parallel, batch_size=2, steps=2, checkpoint=path),
            device,
        )
        first.train()
        del first
        resumed = ExperimentTrainer(
            ExperimentConfig(parallel=parallel, batch_size=2, steps=3, resume=path),
            device,
        )
        result = resumed.train()
        actual = snapshot(resumed)
        assert_state_equal(expected, actual)
        with synchronize_local_errors("Experiment continuation metrics"):
            if dist.get_rank() == 0:
                assert result[0]["step"] == reference[2]["step"]
                assert result[0]["tokens"] == reference[2]["tokens"]
                # NPU FP32 loss sums may differ by one reduction rounding bit;
                # model parameters and moments above still require exactness.
                torch.testing.assert_close(
                    torch.tensor(result[0]["loss"]),
                    torch.tensor(reference[2]["loss"]),
                    atol=1e-6,
                    rtol=1e-6,
                )
                print(
                    f"Qwen3-MoE {args.layout}: "
                    "model/moments exact; next-step loss within 1e-6 passed",
                    flush=True,
                )
        dist.barrier()
    finally:
        if dist.get_rank() == 0 and path is not None:
            shutil.rmtree(path)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
