"""Exercise validation's token/loss reductions with real four-rank Gloo meshes.

    PYTHONPATH=. torchrun --nproc_per_node=4 \
        tests/integration_tests/validation_reduction_equivalence.py

The model is a deterministic local projection, with explicit sequence slices.
This checks the production validation loop and collectives, not attention or
actual TP/FSDP model transformations.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

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
    original_builder = validation.create_validation_dataloader
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
            trainer.checkpointer = None
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
            validation.create_validation_dataloader = lambda *args, loader=loader: (
                loader
            )
            original_train = trainer.model.frozen.train
            for failing_rank in range(dist.get_world_size()):
                for error_type in (ValueError, KeyboardInterrupt, SystemExit):
                    failure = error_type("injected rank-local eval failure")

                    def fail_eval(
                        mode=True,
                        original_train=original_train,
                        failure=failure,
                        failing_rank=failing_rank,
                    ):
                        result = original_train(mode)
                        torch.rand(3)
                        if dist.get_rank() == failing_rank and not mode:
                            raise failure
                        return result

                    rng = torch.get_rng_state().clone()
                    modes = {
                        module: module.training for module in trainer.model.modules()
                    }
                    with patch.object(trainer.model.frozen, "train", fail_eval):
                        try:
                            trainer.validate(step=7)
                        except error_type as error:
                            assert dist.get_rank() == failing_rank and error is failure
                        except RuntimeError as error:
                            assert dist.get_rank() != failing_rank
                            assert str(error) == (
                                "Validation mode setup failed on another rank"
                            )
                        else:
                            raise AssertionError(
                                "Validation mode failure was not coordinated"
                            )
                    assert {
                        module: module.training for module in trainer.model.modules()
                    } == modes
                    assert trainer.model.calls == 0 and not loader.closed and not logs
                    torch.testing.assert_close(
                        torch.get_rng_state(), rng, atol=0, rtol=0
                    )
                    # Successful communication after every owner/exception case.
                    alive = torch.tensor(dist.get_rank() + 1)
                    dist.all_reduce(alive)
                    assert int(alive) == 10
                    dist.barrier()
            for failing_rank in range(dist.get_world_size()):
                for error_type in (ValueError, KeyboardInterrupt, SystemExit):
                    failed_loader = Loader(batches(mesh["dp"].get_local_rank()))
                    original_preprocess = trainer.model.preprocess_inputs
                    failure = error_type("injected validation preprocessing failure")

                    def fail_preprocess(
                        *args,
                        failure=failure,
                        original=original_preprocess,
                        failing_rank=failing_rank,
                        **kwargs,
                    ):
                        if dist.get_rank() == failing_rank:
                            raise failure
                        return original(*args, **kwargs)

                    rng = torch.get_rng_state().clone()
                    with (
                        patch.object(
                            trainer.model, "preprocess_inputs", fail_preprocess
                        ),
                        patch.object(
                            validation,
                            "create_validation_dataloader",
                            return_value=failed_loader,
                        ),
                    ):
                        try:
                            trainer.validate(step=7)
                        except error_type as error:
                            assert dist.get_rank() == failing_rank and error is failure
                        except RuntimeError as error:
                            assert dist.get_rank() != failing_rank
                            assert str(error) == (
                                "Training batch preprocessing failed on another rank"
                            )
                        else:
                            raise AssertionError(
                                "Preprocessing failure was not coordinated"
                            )
                    assert (
                        failed_loader.closed and trainer.model.calls == 0 and not logs
                    )
                    assert trainer.model.training and not trainer.model.frozen.training
                    assert trainer.ntokens_seen == 123
                    torch.testing.assert_close(
                        torch.get_rng_state(), rng, atol=0, rtol=0
                    )
                    dist.barrier()
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
        validation.create_validation_dataloader = original_builder
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
