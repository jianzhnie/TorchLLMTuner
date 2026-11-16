"""Restore FSDP bindings after successful and failed validation forwards.

    PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
        tests/integration_tests/validation_fsdp_bindings.py
"""

from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from llmtuner.components.loss import cross_entropy_loss
from llmtuner.datasets.contracts import Batch
from llmtuner.models.execution import ModelExecutionCapabilities
from llmtuner.parallel.assembly import ModelPartLayout
from llmtuner.trainer.runtime.engine import TrainingEngine
from tests.integration_tests.harness import init_distributed


class Projection(nn.Module):
    execution_capabilities = ModelExecutionCapabilities()

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 8)
        self.head = nn.Linear(8, 8)

    def prepare_inputs(
        self, batch: Batch, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        return batch.input_ids, batch.labels, {}

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(inputs))


def main() -> None:
    device = init_distributed(timeout=timedelta(seconds=30))
    try:
        assert dist.get_world_size() == 2
        torch.manual_seed(42)
        model = Projection().to(device)
        reference = deepcopy(model)
        mesh = init_device_mesh(device.type, (2,))
        fully_shard(model.embedding, mesh=mesh, reshard_after_forward=False)
        fully_shard(model, mesh=mesh, reshard_after_forward=False)
        layout = ModelPartLayout()
        layout.refresh(model)
        engine = TrainingEngine(
            model_parts=[model],
            parallel_dims=None,
            device=device,
            chunked_loss_num_chunks=1,
            param_context=nullcontext,
        )
        inputs = torch.arange(4, device=device)
        labels = (inputs + 1) % 8
        expected = cross_entropy_loss(reference(inputs), labels)
        with torch.no_grad():
            actual = engine.evaluate(Batch(input_ids=inputs, labels=labels))
        torch.testing.assert_close(actual, expected)
        layout.validate(model)

        def failing_loss(*args: object, **kwargs: object) -> torch.Tensor:
            raise ValueError("injected validation loss failure")

        engine.loss_sum = failing_loss
        try:
            with torch.no_grad():
                engine.evaluate(Batch(input_ids=inputs, labels=labels))
        except ValueError as error:
            assert str(error) == "injected validation loss failure"
        else:
            raise AssertionError("Validation unexpectedly succeeded")
        layout.validate(model)

        # Both exits must leave the original shards usable by the next backward.
        cross_entropy_loss(model(inputs), labels).backward()
        expected.backward()
        layout.validate(model)
        for parameter, wanted in zip(
            model.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(parameter.grad.full_tensor(), wanted.grad)
        dist.barrier()
        if dist.get_rank() == 0:
            print("Validation restores FSDP bindings on success and failure.")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
