"""Real Trainer meta/FSDP materialization fails before optimizer.

Run with torchrun --nproc_per_node=2, or 4 for DP2 x Interleaved PP2.
Append compile to check aot_eager OptimizedModule setup before lazy tracing.
No checkpoint reader or training executes.
An optimizer sentinel detects peers crossing the failed materialization phase.
"""

import argparse
from datetime import timedelta
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch.distributed as dist
from torch._dynamo.eval_frame import OptimizedModule

from llmtuner.config.compile_config import CompileConfig
from llmtuner.trainer.cli import parse_config
from llmtuner.trainer.runtime import assembly
from llmtuner.trainer.trainer import Trainer


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=20))
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("mode", nargs="?", choices=("compile",))
        parser.add_argument(
            "--implementation", choices=("native", "custom", "both"), default="both"
        )
        args = parser.parse_args()
        compiled = args.mode == "compile"
        implementations = (
            ("native", "custom")
            if args.implementation == "both"
            else (args.implementation,)
        )
        world_size = dist.get_world_size()
        assert world_size in (2, 4)
        original = assembly.materialize_meta_model
        for implementation in implementations:
            for failing_rank in range(world_size):
                for error_type in (ValueError, KeyboardInterrupt, SystemExit):
                    with TemporaryDirectory() as root:
                        cfg = parse_config(
                            [
                                "--model_name_or_path",
                                "qwen3_moe",
                                "--vocab_size",
                                "32",
                                "--hidden_size",
                                "16",
                                "--intermediate_size",
                                "32",
                                "--num_hidden_layers",
                                "6" if world_size == 4 else "2",
                                "--num_attention_heads",
                                "2",
                                "--num_key_value_heads",
                                "2",
                                "--max_seq_len",
                                "8",
                                "--global_batch_size",
                                "8" if world_size == 4 else "2",
                                "--steps",
                                "1",
                                "--enable",
                                "true",
                                "--initial_load_in_hf",
                                "true",
                                "--initial_load_path",
                                root,
                                "--dump_folder",
                                root,
                                "--data_parallel_shard_size",
                                "2",
                                "--fsdp_implementation",
                                implementation,
                            ]
                        )
                        cfg.model.arch_overrides.update(
                            head_dim=8,
                            num_experts=4,
                            num_experts_per_tok=2,
                            moe_intermediate_size=8,
                        )
                        if world_size == 4:
                            cfg.parallel.pipeline_parallel_size = 2
                            cfg.parallel.pp_config.pipeline_parallel_schedule = (
                                "Interleaved1F1B"
                            )
                            cfg.parallel.pp_config.num_pp_microbatches = 4
                        if compiled:
                            cfg.training.compile = True
                            cfg.training.compile_config = CompileConfig(
                                backend="aot_eager"
                            )
                        failure = error_type("injected meta materialization failure")
                        entered = []
                        materialized = []

                        def materialize(
                            model,
                            device,
                            failing_rank=failing_rank,
                            failure=failure,
                            materialized=materialized,
                            **materialization_options,
                        ):
                            assert isinstance(model, OptimizedModule) is compiled
                            original(model, device, **materialization_options)
                            materialized.append(model)
                            expected_parts = 2 if world_size == 4 else 1
                            if (
                                dist.get_rank() == failing_rank
                                and len(materialized) == expected_parts
                            ):
                                raise failure

                        def optimizer(
                            cfg, *, model_parts, parallel_dims, entered=entered
                        ):
                            entered.append(True)
                            raise AssertionError("Optimizer entered after peer failed")

                        with (
                            patch.object(
                                assembly, "materialize_meta_model", materialize
                            ),
                            patch.object(assembly, "build_optimizer_stack", optimizer),
                        ):
                            caught = None
                            try:
                                Trainer(cfg)
                            except BaseException as error:
                                caught = error
                        if dist.get_rank() == failing_rank:
                            assert caught is failure, caught
                        else:
                            assert isinstance(caught, RuntimeError), caught
                            assert str(caught) == (
                                "Model materialization failed on another rank"
                            )
                        assert len(materialized) == (2 if world_size == 4 else 1)
                        assert not entered
                        dist.barrier()
        if dist.get_rank() == 0:
            print(
                f"{', '.join(implementations)} materialization failures "
                "stop before optimizer."
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
