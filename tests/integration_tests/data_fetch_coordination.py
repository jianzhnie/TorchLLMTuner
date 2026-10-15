"""Two-rank CPU/Gloo check of unequal exhaustion and loader exceptions.

PYTHONPATH=. torchrun --nproc_per_node=2 \
    tests/integration_tests/data_fetch_coordination.py
"""

from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from llmtuner.config.validation_config import ValidationConfig
from llmtuner.datasets.contracts import Batch, DataloaderExhaustedError
from llmtuner.trainer.runtime.batch import batch_generator
from llmtuner.trainer.runtime.validation import run_validation_loop


class Metrics:
    def __init__(self):
        self.tokens = 0

    def add_tokens(self, count):
        self.tokens += count

    def add_data_loading_time(self, duration):
        pass


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=10))
    try:
        rank = dist.get_rank()
        assert dist.get_world_size() == 2
        batch = Batch(input_ids=torch.ones(1, 4), labels=torch.ones(1, 4))
        for failure in ("exhaustion", "exception", "malformed"):
            trainer = SimpleNamespace(metrics=Metrics())

            def source(failure=failure):
                yield batch
                if rank == 0:
                    if failure == "exception":
                        raise ValueError("source decode failed")
                    if failure == "malformed":
                        yield {}
                    return
                yield batch

            iterator = batch_generator(trainer, source())
            assert next(iterator) is batch
            try:
                next(iterator)
            except DataloaderExhaustedError:
                assert failure == "exhaustion"
            except ValueError as error:
                assert rank == 0 and failure == "exception"
                assert str(error) == "source decode failed"
            except KeyError:
                assert rank == 0 and failure == "malformed"
            except RuntimeError as error:
                assert rank == 1 and failure != "exhaustion"
                assert "another rank" in str(error)
            else:
                raise AssertionError(f"rank {rank} yielded an unmatched batch")
            assert trainer.metrics.tokens == 4
            dist.barrier()
            print(f"rank={rank} {failure}: coordinated before yielding")
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp",))
        topology = SimpleNamespace(
            pp_enabled=False,
            dp_cp_enabled=True,
            tp_enabled=False,
            cp_enabled=False,
            spmd_dense_mesh=lambda: mesh,
            spmd_sparse_mesh=lambda: None,
            get_optional_mesh=lambda name: mesh,
        )
        for failure in ("exhaustion", "exception", "malformed"):
            logs = []

            class Loader:
                closed = False

                def __init__(self, failure):
                    self.failure = failure

                def __iter__(self):
                    yield batch
                    if rank == 0:
                        if self.failure == "exception":
                            raise ValueError("validation decode failed")
                        if self.failure == "malformed":
                            yield {}
                        return
                    yield batch

                def close(self):
                    self.closed = True

            loader = Loader(failure)
            metrics = Metrics()
            metrics.log_validation = lambda logs=logs, **values: logs.append(values)
            trainer = SimpleNamespace(
                parallel_dims=topology,
                device=torch.device("cpu"),
                metrics=metrics,
                count_valid_tokens=lambda value: 4,
                preprocess=lambda value: (torch.ones(4), value.labels, {}),
                param_context=nullcontext,
                example_model=lambda inputs: inputs * (rank + 1),
                loss_sum=lambda logits, labels: logits.sum(),
                loss_vocab_kwargs=lambda: {},
            )
            with patch(
                "llmtuner.trainer.runtime.validation.create_validation_dataloader",
                return_value=loader,
            ):
                try:
                    run_validation_loop(trainer, ValidationConfig(steps=3), step=7)
                except ValueError as error:
                    assert rank == 0 and failure == "exception"
                    assert str(error) == "validation decode failed"
                except KeyError:
                    assert rank == 0 and failure == "malformed"
                except RuntimeError as error:
                    assert rank == 1 and failure != "exhaustion"
                    assert "another rank" in str(error)
                else:
                    assert failure == "exhaustion"
            assert loader.closed
            assert metrics.tokens == 4
            if failure == "exhaustion":
                assert logs == [{"loss": 1.5, "step": 7}]
            else:
                assert logs == []
            dist.barrier()
            print(f"rank={rank} validation {failure}: closed and coordinated")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
