"""Coordinate asymmetric HF model-build failures before parallel transforms.

PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
    tests/integration_tests/trainer_initialization_coordination.py

CPU/Gloo only. After each failed constructor, a barrier checks that peers have
not entered unmatched model collectives; each rank also checks host RNG and
execution-policy rollback. A fresh Trainer then performs an update.
"""

import os
import random
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist

from llmtuner.config.llmtuner_config import LLMTunerConfig
from llmtuner.config.model_config import ModelConfig
from llmtuner.config.observability_config import MetricsConfig
from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.config.training_config import TrainingConfig
from llmtuner.trainer.runtime import assembly
from llmtuner.trainer.trainer import Trainer


def config():
    return LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3",
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
        ),
        parallel=ParallelConfig(data_parallel_shard_size=-1),
        training=TrainingConfig(
            global_batch_size=2,
            max_seq_len=8,
            steps=1,
            metrics_config=MetricsConfig(log_freq=1),
        ),
    )


def failed_constructor(cfg):
    """Assert rollback independently on each rank, including the healthy peer."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    policy = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.is_anomaly_enabled(),
        torch.is_anomaly_check_nan_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.utils.deterministic.fill_uninitialized_memory,
    )
    environment = {
        key: os.environ.get(key)
        for key in (
            "PYTHONHASHSEED",
            "HCCL_DETERMINISTIC",
            "CLOSE_MATMUL_K_SHIFT",
            "CUBLAS_WORKSPACE_CONFIG",
        )
    }
    # Enable policies that differ from the caller's baseline, through public
    # config rather than calling seeding directly in the injected failure.
    cfg.training.deterministic = True
    cfg.training.detect_anomaly = True
    try:
        Trainer(cfg)
    except BaseException:
        assert random.getstate() == python_state
        actual_numpy = np.random.get_state()
        assert actual_numpy[0] == numpy_state[0]
        np.testing.assert_array_equal(actual_numpy[1], numpy_state[1])
        assert actual_numpy[2:] == numpy_state[2:]
        torch.testing.assert_close(torch.get_rng_state(), cpu_state, rtol=0, atol=0)
        assert policy == (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.is_anomaly_enabled(),
            torch.is_anomaly_check_nan_enabled(),
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
            torch.utils.deterministic.fill_uninitialized_memory,
        )
        assert environment == {key: os.environ.get(key) for key in environment}
        raise


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        for failing_rank, name in enumerate(
            ("build_model_config_for", "HFTransformerModel")
        ):
            original = getattr(assembly, name)
            failure = ValueError(f"injected {name} failure")

            def build(
                *args,
                failing_rank=failing_rank,
                failure=failure,
                original=original,
                **kwargs,
            ):
                if rank == failing_rank:
                    raise failure
                return original(*args, **kwargs)

            with (
                patch.object(assembly, name, build),
                patch.object(assembly, "_apply_parallelism") as transform,
            ):
                try:
                    failed_constructor(config())
                except ValueError as error:
                    assert rank == failing_rank and error is failure
                except RuntimeError as error:
                    assert rank != failing_rank
                    assert str(error) == "Model construction failed on another rank"
                else:
                    raise AssertionError(
                        "asymmetric constructor unexpectedly succeeded"
                    )
                transform.assert_not_called()
            dist.barrier()
            print(
                f"rank={rank} {name}: coordinated before parallel transforms",
                flush=True,
            )
        # No rank may enter checkpointer construction while another loader failed.
        original_loader = Trainer.build_dataloader
        closed = []
        failure = ValueError("injected dataloader construction failure")

        def build_loader(self):
            if rank == 0:
                raise failure
            return SimpleNamespace(close=lambda: closed.append(True))

        with (
            patch.object(Trainer, "build_dataloader", build_loader),
            patch.object(assembly, "_build_checkpointer") as checkpointer,
        ):
            try:
                failed_constructor(config())
            except ValueError as error:
                assert rank == 0 and error is failure
            except RuntimeError as error:
                assert rank == 1
                assert str(error) == "Dataloader construction failed on another rank"
            else:
                raise AssertionError("asymmetric loader unexpectedly succeeded")
            checkpointer.assert_not_called()
        assert closed == ([True] if rank == 1 else [])
        assert Trainer.build_dataloader is original_loader
        dist.barrier()
        print(
            f"rank={rank}: loader failure coordinated and peer loader closed",
            flush=True,
        )
        # Storage preflight must finish on every rank before model state or
        # the async checkpoint communication group is constructed.
        from llmtuner.components.checkpointer.distributed_checkpoint import (
            DistributedCheckpointManager,
        )
        from llmtuner.config.checkpoint_config import CheckpointConfig

        for failing_rank in (0, 1):
            with patch.object(dist, "new_group") as new_group:
                try:
                    DistributedCheckpointManager(
                        CheckpointConfig(
                            enable=True, async_mode="async", last_save_in_hf=True
                        ),
                        model_parts=[torch.nn.Linear(2, 2)],
                        optimizer=None,
                        lr_scheduler=None,
                        states={},
                        folder="/tmp/llmtuner-preflight-unused",
                        sd_adapter=None if rank == failing_rank else object(),
                    )
                except ValueError as error:
                    assert rank == failing_rank
                    assert "state_dict adapter" in str(error)
                except RuntimeError as error:
                    assert rank != failing_rank
                    assert str(error) == "Checkpoint preflight failed on another rank"
                else:
                    raise AssertionError("asymmetric checkpoint preflight succeeded")
                new_group.assert_not_called()
            dist.barrier()
        print(f"rank={rank}: checkpoint preflight coordinated", flush=True)
        import threading

        original_start = threading.Thread.start
        original_new_group = dist.new_group
        for failing_rank in (0, 1):
            created_groups = []
            started_threads = []
            failure = OSError("injected retention thread startup failure")

            def track_group(*args, created_groups=created_groups, **kwargs):
                group = original_new_group(*args, **kwargs)
                created_groups.append(group)
                return group

            def start_thread(
                thread,
                failing_rank=failing_rank,
                failure=failure,
                started_threads=started_threads,
            ):
                if rank == failing_rank:
                    raise failure
                original_start(thread)
                started_threads.append(thread)

            with (
                patch.object(dist, "new_group", track_group),
                patch.object(threading.Thread, "start", start_thread),
            ):
                try:
                    DistributedCheckpointManager(
                        CheckpointConfig(
                            enable=True, async_mode="async", keep_latest_k=2
                        ),
                        model_parts=[torch.nn.Linear(2, 2)],
                        optimizer=None,
                        lr_scheduler=None,
                        states={},
                        folder="/tmp/llmtuner-retention-unused",
                    )
                except OSError as error:
                    assert rank == failing_rank and error is failure
                except RuntimeError as error:
                    assert rank != failing_rank
                    assert str(error) == (
                        "Checkpoint services construction failed on another rank"
                    )
                else:
                    raise AssertionError("asymmetric retention startup succeeded")
            assert len(created_groups) == 1
            try:
                dist.get_process_group_ranks(created_groups[0])
            except KeyError:
                pass
            else:
                raise AssertionError("failed constructor leaked async group")
            assert len(started_threads) == (0 if rank == failing_rank else 1)
            assert all(not thread.is_alive() for thread in started_threads)
            dist.barrier()
        print(f"rank={rank}: retention failure coordinated and cleaned", flush=True)
        from llmtuner.parallel.expert_parallel.checkpointing import (
            expert_shard_map,
            gather_expert_state,
            mark_experts_ep_sharded,
        )

        expert = torch.nn.Module()
        expert.register_parameter("first", torch.nn.Parameter(torch.zeros(1, 2)))
        expert.register_parameter("second", torch.nn.Parameter(torch.zeros(1, 2)))
        mark_experts_ep_sharded(expert, dist.group.WORLD)
        # Same shapes make a swapped collective silently produce wrong values.
        items = [
            ("first", torch.full((1, 2), 10.0 + rank)),
            ("second", torch.full((1, 2), 20.0 + rank)),
        ]
        if rank == 1:
            items.reverse()
        gathered = gather_expert_state(
            dict(items), shard_map=expert_shard_map([expert]), model_parts=[expert]
        )
        for name, base in (("first", 10.0), ("second", 20.0)):
            torch.testing.assert_close(
                gathered[name],
                torch.tensor([[base, base], [base + 1, base + 1]]),
                rtol=0,
                atol=0,
            )
        dist.barrier()
        print(f"rank={rank}: EP checkpoint key ordering passed", flush=True)
        for case in (
            "invalid_type",
            "conflicting_ep_groups",
            "overlapping_keys",
            "hook_overlapping_keys",
        ):
            for failing_rank in (0, 1):
                part = torch.nn.Linear(2, 2)
                mark_experts_ep_sharded(part, dist.group.WORLD)
                model_parts = [part]
                if rank == failing_rank:
                    if case == "invalid_type":
                        model_parts.append(object())
                    elif case == "hook_overlapping_keys":
                        duplicate = torch.nn.Module()
                        duplicate.register_parameter(
                            "other", torch.nn.Parameter(torch.zeros(2, 2))
                        )

                        def export_logical_name(module, state, prefix, metadata):
                            state[f"{prefix}weight"] = state.pop(f"{prefix}other")

                        duplicate.register_state_dict_post_hook(export_logical_name)
                        model_parts.append(duplicate)
                    elif case == "overlapping_keys":
                        duplicate = torch.nn.Linear(2, 2)
                        mark_experts_ep_sharded(duplicate, dist.group.WORLD)
                        model_parts.append(duplicate)
                    else:
                        part.alias = torch.nn.Module()
                        part.alias.register_parameter("weight", part.weight)
                        mark_experts_ep_sharded(part, dist.group.WORLD)
                        mark_experts_ep_sharded(part.alias, object())
                with patch.object(dist, "new_group") as new_group:
                    try:
                        DistributedCheckpointManager(
                            CheckpointConfig(enable=True, async_mode="async"),
                            model_parts=model_parts,
                            optimizer=None,
                            lr_scheduler=None,
                            states={},
                            folder="/tmp/llmtuner-model-preparation-unused",
                        )
                    except (TypeError, ValueError) as error:
                        assert rank == failing_rank
                        if case == "invalid_type":
                            assert "must be an nn.Module" in str(error)
                        elif case in ("overlapping_keys", "hook_overlapping_keys"):
                            assert "overlap at part 1" in str(error)
                        else:
                            assert "different expert process groups" in str(error)
                    except RuntimeError as error:
                        assert rank != failing_rank
                        phase = (
                            "model state merge"
                            if case in ("overlapping_keys", "hook_overlapping_keys")
                            else "model preparation"
                        )
                        assert str(error) == (
                            f"Checkpoint {phase} failed on another rank"
                        )
                    else:
                        raise AssertionError("invalid model preparation succeeded")
                    new_group.assert_not_called()
                dist.barrier()
        print(f"rank={rank}: model checkpoint preparation coordinated", flush=True)
        original_metrics = assembly.MetricsProcessor
        metrics_closed = []
        failure = ValueError("injected metrics backend failure")

        def build_metrics(*args, **kwargs):
            if rank == 1:
                raise failure
            return SimpleNamespace(close=lambda: metrics_closed.append(True))

        with patch.object(assembly, "MetricsProcessor", build_metrics):
            try:
                failed_constructor(config())
            except ValueError as error:
                assert rank == 1 and error is failure
            except RuntimeError as error:
                assert rank == 0
                assert str(error) == "Metrics construction failed on another rank"
            else:
                raise AssertionError(
                    "asymmetric metrics backend unexpectedly succeeded"
                )
        assert metrics_closed == ([True] if rank == 0 else [])
        assert assembly.MetricsProcessor is original_metrics
        dist.barrier()
        print(
            f"rank={rank}: metrics failure coordinated and peer logger closed",
            flush=True,
        )
        trainer = Trainer(config())
        try:
            trainer.step = 1
            metrics = trainer.train_step(trainer.data_iterator())
            assert metrics is not None and torch.isfinite(torch.tensor(metrics["loss"]))
        finally:
            trainer.close()
        dist.barrier()
        print(f"rank={rank}: fresh Trainer update passed", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
