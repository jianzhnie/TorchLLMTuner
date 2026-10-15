"""Exercise validation's token/loss reductions with real four-rank Gloo meshes.

    PYTHONPATH=. torchrun --nproc_per_node=4 \
        tests/integration_tests/validation_reduction_equivalence.py

The model is a deterministic local projection, with explicit sequence slices.
This checks the production validation loop and collectives, not attention or
actual TP/FSDP model transformations.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

from llmtuner.components.loss import IGNORE_INDEX
from llmtuner.config import LLMTunerConfig, ValidationConfig
from llmtuner.trainer.runtime import validation
from llmtuner.trainer.trainer import Trainer


def batches(dp_rank):
    result = []
    for step, valid in enumerate((1 + dp_rank, 5 - dp_rank)):
        inputs = (torch.arange(6) + 2 * dp_rank + step).reshape(1, 6) % 7
        labels = (inputs + 1) % 7
        labels[:, valid:] = IGNORE_INDEX
        result.append({"input_ids": inputs, "labels": labels})
    return result


class Loader:
    def __init__(self, data):
        self.data = data
        self.closed = False

    def __iter__(self):
        return iter(self.data)

    def close(self):
        self.closed = True


class Projection(torch.nn.Module):
    def __init__(self, coordinate):
        super().__init__()
        self.coordinate = coordinate
        self.register_buffer("weight", torch.arange(49).reshape(7, 7).double() / 17)
        self.add_module("frozen", torch.nn.Dropout())
        self.frozen.eval()
        self.calls = 0

    def preprocess_inputs(self, batch, **kwargs):
        start = 3 * self.coordinate
        return (
            batch["input_ids"][:, start : start + 3].reshape(-1),
            batch["labels"][:, start : start + 3].reshape(-1),
            {},
        )

    def forward(self, inputs):
        assert not self.training and not self.frozen.training
        assert not torch.is_grad_enabled()
        self.calls += 1
        return self.weight[inputs]


def main():
    dist.init_process_group("gloo")
    original_builder = validation.build_validation_dataloader
    try:
        assert dist.get_world_size() == 4
        for axis in ("cp", "tp"):
            mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", axis))
            topology = SimpleNamespace(
                pp_enabled=False,
                spmd_dense_mesh=lambda mesh=mesh: mesh,
                spmd_sparse_mesh=lambda: None,
                dp_cp_enabled=True,
                tp_enabled=axis == "tp",
                get_optional_mesh=lambda name, mesh=mesh, axis=axis: (
                    mesh["dp", axis]._flatten(mesh_dim_name="loss")
                    if name == "loss"
                    else mesh[name]
                    if name in ("dp", axis)
                    else None
                ),
            )
            trainer = Trainer.__new__(Trainer)
            trainer.cfg = LLMTunerConfig()
            trainer.cfg.training.validation_config = ValidationConfig(steps=2)
            trainer.parallel_dims = topology
            trainer.device = torch.device("cpu")
            trainer.model = Projection(mesh[axis].get_local_rank())
            trainer.model_parts = [trainer.model]
            trainer.ntokens_seen = 123
            trainer.param_context = nullcontext
            trainer.to_device = lambda batch: batch
            logs = []
            trainer.metrics = SimpleNamespace(
                add_tokens=lambda count: None,
                log_validation=lambda logs=logs, **kw: logs.append(kw),
            )
            loader = Loader(batches(mesh["dp"].get_local_rank()))
            validation.build_validation_dataloader = lambda *args, loader=loader: loader
            trainer.validate(step=7)
            reference_sum = torch.zeros((), dtype=torch.float64)
            reference_count = 0
            for dp_rank in range(2):
                for batch in batches(dp_rank):
                    logits = trainer.model.weight[batch["input_ids"]]
                    labels = batch["labels"]
                    reference_sum += F.cross_entropy(
                        logits.flatten(0, 1),
                        labels.flatten(),
                        ignore_index=IGNORE_INDEX,
                        reduction="sum",
                    )
                    reference_count += int((labels != IGNORE_INDEX).sum())
            assert len(logs) == 1 and logs[0]["step"] == 7
            torch.testing.assert_close(
                torch.tensor(logs[0]["loss"], dtype=torch.float64),
                reference_sum / reference_count,
                rtol=1e-7,
                atol=1e-7,
            )
            assert loader.closed and trainer.model.training
            assert not trainer.model.frozen.training
            assert trainer.model.calls == 2 and trainer.ntokens_seen == 123
            if dist.get_rank() == 0:
                print(f"DP x {axis.upper()}: validation matches full-batch loss")
    finally:
        validation.build_validation_dataloader = original_builder
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
