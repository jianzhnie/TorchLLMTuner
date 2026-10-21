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
from llmtuner.trainer.runtime.batch import batch_generator, fetch_batch
from llmtuner.trainer.runtime.validation import finalize_validation, run_validation_loop
from llmtuner.trainer.trainer import Trainer


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
        for failure in (
            "exhaustion",
            "exception",
            "malformed",
            "interrupt",
            "exit",
            "count",
        ):
            trainer = SimpleNamespace(metrics=Metrics())

            def source(failure=failure):
                yield batch
                if rank == 0:
                    if failure == "exception":
                        raise ValueError("source decode failed")
                    if failure == "interrupt":
                        raise KeyboardInterrupt("source interrupted")
                    if failure == "exit":
                        raise SystemExit("source exited")
                    if failure == "count":
                        yield {"labels": batch.labels, "num_valid_tokens": 1.5}
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
                assert rank == 0 and failure in ("exception", "count")
                if failure == "exception":
                    assert str(error) == "source decode failed"
                else:
                    assert "num_valid_tokens" in str(error)
            except (KeyboardInterrupt, SystemExit) as error:
                assert rank == 0
                assert failure == (
                    "interrupt" if isinstance(error, KeyboardInterrupt) else "exit"
                )
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
        for error_type in (KeyboardInterrupt, SystemExit):
            failure = error_type("fetch interrupted while peer exhausted")

            def interrupted_source(failure=failure):
                if rank == 0:
                    raise failure
                return
                yield

            try:
                fetch_batch(interrupted_source())
            except (KeyboardInterrupt, SystemExit) as error:
                assert rank == 0 and error is failure
            except RuntimeError as error:
                assert rank == 1 and "Data fetch failed" in str(error)
            else:
                raise AssertionError("fetch failure lost to exhaustion")
            dist.barrier()
            print(f"rank={rank} {error_type.__name__}: precedes exhaustion")
        # A malformed decoded batch must also beat peer exhaustion. Test both
        # owners so status precedence cannot accidentally depend on rank zero.
        for failing_rank in (0, 1):
            for invalid in ({}, {"labels": batch.labels, "num_valid_tokens": -1}):

                def malformed_source(invalid=invalid, failing_rank=failing_rank):
                    if rank == failing_rank:
                        yield invalid

                try:
                    fetch_batch(malformed_source())
                except (KeyError, ValueError):
                    assert rank == failing_rank
                except RuntimeError as error:
                    assert rank != failing_rank
                    assert str(error) == "Data fetch failed on another rank"
                else:
                    raise AssertionError("malformed batch lost to exhaustion")
                dist.barrier()
            print(f"rank={rank}: malformed rank {failing_rank} precedes exhaustion")
        for error_type in (ValueError, KeyboardInterrupt, SystemExit):
            failure = error_type("training iterator initialization failed")
            closed = []

            class InitializationLoader:
                def __iter__(self, failure=failure):
                    if rank == 0:
                        raise failure
                    return iter([batch])

                def close(self, closed=closed):
                    closed.append("loader")

            trainer = Trainer.__new__(Trainer)
            trainer.cfg = SimpleNamespace(checkpoint=SimpleNamespace(load_step=-1))
            trainer.step = 0
            trainer.dataloader = InitializationLoader()
            trainer.checkpointer = SimpleNamespace(
                load=lambda step: False,
                close=lambda closed=closed: closed.append("checkpoint"),
            )
            with patch("llmtuner.trainer.trainer.Profiler") as profiler:
                try:
                    trainer.train()
                except (ValueError, KeyboardInterrupt, SystemExit) as error:
                    assert rank == 0 and error is failure
                except RuntimeError as error:
                    assert rank == 1
                    assert str(error) == (
                        "Training input initialization failed on another rank"
                    )
                else:
                    raise AssertionError("iterator initialization was not coordinated")
                profiler.assert_not_called()
            assert closed == ["checkpoint", "loader"]
            dist.barrier()
            print(f"rank={rank} {error_type.__name__}: training iterator coordinated")
        from llmtuner.models.hf.factory import build_model_config
        from llmtuner.models.hf.model import HFTransformerModel

        model = HFTransformerModel(
            build_model_config(
                "qwen3",
                seq_len=4,
                arch_overrides={
                    "vocab_size": 32,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                },
            )
        )
        trainer = Trainer.__new__(Trainer)
        trainer.cfg = SimpleNamespace(max_seq_len=4)
        trainer.device = torch.device("cpu")
        trainer.parallel_dims = None
        trainer.model = model
        for failing_rank in (0, 1):
            value = {
                "input": torch.ones(4, dtype=torch.long),
                "labels": torch.ones(
                    3 if rank == failing_rank else 4, dtype=torch.long
                ),
            }
            try:
                trainer.preprocess(value)
            except ValueError as error:
                assert rank == failing_rank and "equal lengths" in str(error)
            except RuntimeError as error:
                assert rank != failing_rank
                assert (
                    str(error) == "Training batch preprocessing failed on another rank"
                )
            else:
                raise AssertionError("malformed preprocessing was not coordinated")
            dist.barrier()
        print(f"rank={rank}: HF preprocessing length errors coordinated")
        from llmtuner.trainer.runtime.pipeline import (
            prepare_pipeline_inputs,
            split_pipeline_microbatches,
        )

        trainer.pp_has_first_stage = rank == 0
        trainer.pp_has_last_stage = rank == 1
        trainer.pp_microbatches = lambda value: split_pipeline_microbatches(
            value, num_microbatches=2
        )
        for failing_rank in (0, 1):
            rows = 3 if rank == failing_rank else 4
            value = Batch(
                input_ids=torch.ones(rows, 4, dtype=torch.long),
                labels=torch.ones(rows, 4, dtype=torch.long),
            )
            with patch.object(trainer, "preprocess") as preprocess:
                try:
                    prepare_pipeline_inputs(trainer, value)
                except ValueError as error:
                    assert rank == failing_rank and "divisible" in str(error)
                except RuntimeError as error:
                    assert rank != failing_rank
                    assert str(error) == (
                        "Pipeline input splitting failed on another rank"
                    )
                else:
                    raise AssertionError("pipeline split failure was not coordinated")
                preprocess.assert_not_called()
            dist.barrier()
        print(f"rank={rank}: pipeline split errors coordinated before preprocessing")
        preprocessing_mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("sequence",))
        for axis in ("cp", "tp"):
            trainer.parallel_dims = SimpleNamespace(
                get_optional_mesh=lambda name, axis=axis: (
                    preprocessing_mesh if name == axis else None
                )
            )
            value = {
                "input": torch.arange(4),
                "labels": torch.arange(4),
            }
            inputs, labels, _ = trainer.preprocess(value)
            torch.testing.assert_close(inputs, value["input"].chunk(2)[rank])
            torch.testing.assert_close(labels, value["labels"].chunk(2)[rank])
            dist.barrier()
            print(f"rank={rank}: preprocessing {axis} local slicing passed")
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
        for failure in (
            "exhaustion",
            "exception",
            "malformed",
            "interrupt",
            "exit",
            "count",
        ):
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
                        if self.failure == "interrupt":
                            raise KeyboardInterrupt("validation interrupted")
                        if self.failure == "exit":
                            raise SystemExit("validation exited")
                        if self.failure == "count":
                            yield {"labels": batch.labels, "num_valid_tokens": 1.5}
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
                    assert rank == 0 and failure in ("exception", "count")
                    if failure == "exception":
                        assert str(error) == "validation decode failed"
                    else:
                        assert "num_valid_tokens" in str(error)
                except (KeyboardInterrupt, SystemExit) as error:
                    assert rank == 0
                    assert failure == (
                        "interrupt" if isinstance(error, KeyboardInterrupt) else "exit"
                    )
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
        for phase in ("construction", "iteration"):
            closed = []
            original = ValueError(f"validation {phase} failed")

            class InitializationLoader:
                def __iter__(self, phase=phase, original=original):
                    if rank == 0 and phase == "iteration":
                        raise original
                    return iter([batch])

                def close(self, closed=closed):
                    closed.append(True)

            def create(*args, phase=phase, original=original):
                if rank == 0 and phase == "construction":
                    raise original
                return InitializationLoader()

            with patch(
                "llmtuner.trainer.runtime.validation.create_validation_dataloader",
                side_effect=create,
            ):
                try:
                    run_validation_loop(trainer, ValidationConfig(steps=1), step=8)
                except ValueError as error:
                    assert rank == 0 and error is original
                except RuntimeError as error:
                    assert rank == 1
                    assert "Validation input initialization" in str(error)
                else:
                    raise AssertionError(
                        "validation initialization was not coordinated"
                    )
            assert closed == ([] if rank == 0 and phase == "construction" else [True])
            dist.barrier()
            print(f"rank={rank} validation {phase}: coordinated and cleaned")
        close_calls = []
        cleanup_error = OSError("validation loader close failed")

        class CleanupLoader:
            def __iter__(self):
                return iter([batch])

            def close(self):
                close_calls.append(True)
                if rank == 0:
                    raise cleanup_error

        with (
            patch(
                "llmtuner.trainer.runtime.validation.create_validation_dataloader",
                return_value=CleanupLoader(),
            ),
            patch(
                "llmtuner.trainer.runtime.validation.finalize_validation"
            ) as finalize,
        ):
            try:
                run_validation_loop(trainer, ValidationConfig(steps=1), step=9)
            except OSError as error:
                assert rank == 0 and error is cleanup_error
            except RuntimeError as error:
                assert rank == 1 and "Validation input cleanup" in str(error)
            else:
                raise AssertionError("validation cleanup was not coordinated")
            finalize.assert_not_called()
        assert close_calls == [True]
        dist.barrier()
        print(f"rank={rank} validation cleanup: coordinated before loss reduction")
        for failure in ("sink", "nonfinite", "preflight"):
            original = OSError("validation sink failed")

            def log_validation(*, loss, step, original=original):
                raise original

            report_trainer = SimpleNamespace(
                metrics=SimpleNamespace(log_validation=log_validation)
            )
            # Rank 0 represents a final PP stage; rank 1 holds no loss.
            loss = torch.tensor(float("nan") if failure == "nonfinite" else 4.0)
            try:
                finalize_validation(
                    report_trainer,
                    num_steps=0 if rank == 0 and failure == "preflight" else 1,
                    accumulated_loss=loss if rank == 0 else None,
                    total_global_valid_tokens=torch.tensor(4),
                    loss_mesh=None,
                    step=10,
                )
            except OSError as error:
                assert rank == 0 and failure == "sink" and error is original
            except FloatingPointError:
                assert rank == 0 and failure == "nonfinite"
            except ValueError:
                assert rank == 0 and failure == "preflight"
            except RuntimeError as error:
                assert rank == 1 and "Validation result" in str(error)
            else:
                raise AssertionError("validation result failure was not coordinated")
            dist.barrier()
            print(f"rank={rank} validation {failure}: result error coordinated")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
