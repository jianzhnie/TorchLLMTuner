"""Two-rank PP: reject rank-local global-plan errors before model splitting."""

from dataclasses import replace
from datetime import timedelta

import torch
import torch.distributed as dist
from transformers import Qwen3MoeConfig

from llmtuner.config.parallel_config import (
    FSDPConfig,
    ParallelConfig,
    PipelineParallelConfig,
)
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel.parallelize import parallelize_model
from llmtuner.parallel.topology import ParallelTopology
from tests.integration_tests.harness import init_distributed


def main() -> None:
    device = init_distributed(timeout=timedelta(seconds=60))
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        cfg = ParallelConfig(
            pipeline_parallel_size=2,
            pp_config=PipelineParallelConfig(num_pp_microbatches=2),
        )
        topology = ParallelTopology.from_config(cfg, world_size=2)
        topology.build_device_mesh()
        config = Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
            head_dim=8,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=12,
            mlp_only_layers=[0, 1],
            tie_word_embeddings=False,
        )
        config._attn_implementation = "eager"
        config._experts_implementation = "eager"
        torch.manual_seed(82)
        model = Qwen3MoeForCausalLM(config).to(device)
        parameters = list(model.parameters())
        layers = list(model.layers)
        for fault in ("missing_batch", "invalid_batch", "fsdp_binding"):
            batch = 4
            local_cfg = cfg
            if fault == "fsdp_binding":
                local_cfg = replace(
                    cfg,
                    fsdp_config=FSDPConfig(
                        fsdp_apply_modules={"missing" if rank == 0 else "": {}}
                    ),
                )
            elif rank == 0:
                batch = None if fault == "missing_batch" else 3
            caught = None
            try:
                parallelize_model(
                    model,
                    cfg=local_cfg,
                    mesh=None,
                    parallel_dims=topology,
                    device=device,
                    global_batch_size=batch,
                )
            except (ValueError, RuntimeError) as error:
                caught = error
            assert caught is not None
            if rank == 0:
                assert isinstance(caught, ValueError)
            else:
                phase = (
                    "Pipeline FSDP plan binding"
                    if fault == "fsdp_binding"
                    else "Pipeline global plan validation"
                )
                assert f"{phase} failed on another rank" in str(caught)
            assert all(
                a is b for a, b in zip(parameters, model.parameters(), strict=True)
            )
            assert all(a is b for a, b in zip(layers, model.layers, strict=True))
        # Real PP split and schedule construction after coordinated rejections.
        setup = parallelize_model(
            model,
            cfg=cfg,
            mesh=None,
            parallel_dims=topology,
            device=device,
            global_batch_size=4,
        )
        assert len(setup.model_parts) == len(setup.stages) == 1
        assert setup.schedule is not None
        dist.barrier()
        print(f"rank={rank}: PP plan rejection and assembly retry passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
