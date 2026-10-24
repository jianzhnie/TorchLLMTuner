"""Two CPU ranks: tracing, GC and report failures stop the Trainer together.

Run with torchrun --nproc_per_node=2. Step computation and checkpoint IO are
substituted; CPU tracing, trace export, profiler cleanup, GC and train are real.
"""

import gc
from datetime import timedelta
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch.distributed as dist

from llmtuner.components.profiler import Profiler
from llmtuner.config.observability_config import ProfilerConfig
from llmtuner.datasets.contracts import DataloaderExhaustedError
from llmtuner.trainer.trainer import Trainer
from llmtuner.utils.garbage_collection import GarbageCollection


def check_failure(phase, failing_rank, error_type):
    with TemporaryDirectory() as root:
        config = ProfilerConfig(
            enable_profiling=True,
            profile_freq=1,
            profiler_active=1,
            profiler_warmup=0,
        )
        profiler = Profiler(config, base_folder=root)
        trainer = Trainer.__new__(Trainer)
        trainer.cfg = SimpleNamespace(
            checkpoint=SimpleNamespace(load_step=-1),
            profiler=config,
            dump_folder=root,
            gc_freq=1,
            steps=2,
        )
        exhausted = phase in ("exhaustion reporting", "exhausted checkpoint reporting")
        trainer.step = 1 if exhausted else 0
        trainer.ntokens_seen = 0
        trainer.parallel_dims = None
        trainer.data_iterator = lambda: iter(())
        trainer.batch_generator = lambda source: source

        def train_step(source):
            if exhausted:
                raise DataloaderExhaustedError()

        trainer.train_step = train_step
        trainer.should_validate = lambda step: False
        saves, closes = [], []

        def save(*args, **kwargs):
            saves.append(args)
            return phase in ("checkpoint reporting", "exhausted checkpoint reporting")

        trainer.checkpointer = SimpleNamespace(
            load=lambda step: phase == "resume reporting",
            save=save,
            close=lambda: closes.append("checkpointer"),
        )
        trainer.metrics = SimpleNamespace(
            close=lambda: closes.append("metrics"),
        )
        failure = error_type("injected real profiler failure")
        memory = profiler.build_memory_profiler

        def build_memory():
            if dist.get_rank() == failing_rank:
                raise failure
            return memory()

        def export(*args, **kwargs):
            if dist.get_rank() == failing_rank:
                raise failure
            return original_export(*args, **kwargs)

        import torch.profiler

        original_export = torch.profiler.profile.export_chrome_trace
        original_collect = GarbageCollection.collect
        from llmtuner.trainer.trainer import logger

        method = "warning" if phase == "exhaustion reporting" else "info"
        original_report = getattr(logger, method)

        def report(message, *args, **kwargs):
            prefix = (
                "Resuming from step"
                if phase == "resume reporting"
                else "Ran out of data"
                if phase == "exhaustion reporting"
                else "Saved checkpoint for step"
            )
            if str(message).startswith(prefix):
                if dist.get_rank() == failing_rank:
                    raise failure
            return original_report(message, *args, **kwargs)

        def collect(reason, **kwargs):
            target = (
                "Initial GC collection"
                if phase == "GC initialization"
                else "Performing periodic GC collection"
            )
            if reason == target and dist.get_rank() == failing_rank:
                raise failure
            return original_collect(reason, **kwargs)

        hook = (
            patch.object(logger, method, report)
            if phase.endswith("reporting")
            else patch.object(GarbageCollection, "collect", staticmethod(collect))
            if phase.startswith("GC ")
            else (
                patch.object(profiler, "build_memory_profiler", build_memory)
                if phase == "initialization"
                else patch.object(torch.profiler.profile, "export_chrome_trace", export)
            )
        )
        enabled = gc.isenabled()
        with (
            patch(
                "llmtuner.trainer.trainer.Profiler",
                return_value=profiler,
            ),
            hook,
        ):
            caught = None
            try:
                trainer.train()
            except BaseException as error:
                caught = error
        if dist.get_rank() == failing_rank:
            assert caught is failure
        else:
            assert isinstance(caught, RuntimeError), caught
            boundary = (
                phase
                if phase.startswith("GC ") or phase.endswith("reporting")
                else "profiler " + phase
            )
            assert str(caught) == (f"Training {boundary} failed on another rank"), str(
                caught
            )
        expected_step = (
            0
            if phase in ("initialization", "GC initialization", "resume reporting")
            else 2
            if phase == "GC step"
            else 1
        )
        assert trainer.step == expected_step
        expected_saves = (
            0
            if phase == "exhaustion reporting"
            else 1
            if phase == "GC step"
            else expected_step
        )
        assert len(saves) == expected_saves
        assert closes == ["checkpointer", "metrics"]
        assert not profiler._entered
        assert profiler.torch_profiler is None
        assert profiler.memory_profiler is None
        assert gc.isenabled() == enabled
        dist.barrier()


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=10))
    try:
        assert dist.get_world_size() == 2
        for phase in (
            "resume reporting",
            "exhaustion reporting",
            "exhausted checkpoint reporting",
            "checkpoint reporting",
            "initialization",
            "step",
            "GC initialization",
            "GC step",
        ):
            for failing_rank in (0, 1):
                for error_type in (OSError, KeyboardInterrupt, SystemExit):
                    check_failure(phase, failing_rank, error_type)
        if dist.get_rank() == 0:
            print("Profiler, GC and checkpoint report failures stop all ranks.")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
