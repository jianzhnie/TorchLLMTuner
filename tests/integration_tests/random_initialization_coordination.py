"""CPU/Gloo: rank-local initializer failure stops runtime assembly on all ranks.

Run with torchrun --standalone --nproc_per_node=2. Uses real meta model
construction, parallel assembly, materialization and random initialization.
Only the local initializer failure is injected; no NPU acceptance claim.
"""

from datetime import timedelta
from types import MethodType, SimpleNamespace

import torch
import torch.distributed as dist

from llmtuner.config import LLMTunerConfig, ModelConfig
from llmtuner.models.configuration import build_model_config_for
from llmtuner.models.registry import build_registered_model
from llmtuner.trainer.runtime.assembly import _apply_parallelism


def main() -> None:
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    try:
        assert dist.get_world_size() == 2
        cfg = LLMTunerConfig(
            model=ModelConfig(
                model_name_or_path="kimi_linear",
                initialization_device="meta",
                arch_overrides={
                    "vocab_size": 16,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "linear_attn_config": {
                        "kda_layers": [1],
                        "full_attn_layers": [],
                        "head_dim": 4,
                        "num_heads": 2,
                        "short_conv_kernel_size": 3,
                    },
                },
            )
        )
        config = build_model_config_for(cfg)
        for failing_rank in (0, 1):
            # Failed assembly is not transactional: rebuild for each scenario.
            with torch.device("meta"):
                model = build_registered_model(config, explicit_only=True)
            initialize = model.initialize_parameters

            def fail_locally(self, initialize=initialize, failing_rank=failing_rank):
                initialize()
                if dist.get_rank() == failing_rank:
                    raise ValueError("injected rank-local initialization failure")

            model.initialize_parameters = MethodType(fail_locally, model)
            trainer = SimpleNamespace(
                cfg=cfg, device=torch.device("cpu"), mesh=None, parallel_dims=None
            )
            try:
                _apply_parallelism(trainer, model, load_initial_weights=False)
            except (ValueError, RuntimeError) as error:
                if dist.get_rank() == failing_rank:
                    assert "injected rank-local" in str(error)
                else:
                    assert "Random model initialization failed on another rank" in str(
                        error
                    )
            else:
                raise AssertionError("runtime continued after initialization failure")
            assert all(not parameter.is_meta for parameter in model.parameters())
            dist.barrier()
        dist.barrier()
        if dist.get_rank() == 0:
            print("PASS: random initialization failure coordinated before optimizer")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
