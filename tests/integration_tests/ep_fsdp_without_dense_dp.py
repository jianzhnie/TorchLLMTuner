"""EP can need expert FSDP even when dense DP and CP have degree one.

Run with ``PYTHONPATH=. torchrun --standalone --nproc_per_node=4
tests/integration_tests/ep_fsdp_without_dense_dp.py``.
"""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from llmtuner.config import LLMTunerConfig, ModelConfig, ParallelConfig, TrainingConfig
from llmtuner.models.hf.factory import build_model_config_for
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.parallel.parallel_dims import ParallelDims
from llmtuner.parallel.parallelize import parallelize_hf_transformers


def main() -> None:
    dist.init_process_group("gloo")
    assert dist.get_world_size() == 4
    cfg = LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="qwen3_moe",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=8,
            num_key_value_heads=8,
            arch_overrides={
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 48,
                "norm_topk_prob": True,
            },
        ),
        parallel=ParallelConfig(
            data_parallel_shard_size=1,
            tensor_parallel_size=4,
            expert_parallel_size=2,
        ),
        training=TrainingConfig(max_seq_len=32, steps=1),
    )
    dims = ParallelDims.from_config(cfg.parallel, 4)
    dims.build_mesh()
    torch.manual_seed(0)
    model = HFTransformerModel(build_model_config_for(cfg)).to(torch.float32)
    model = parallelize_hf_transformers(
        model,
        cfg=cfg.parallel,
        mesh=dims.spmd_dense_mesh(),
        parallel_dims=dims,
        device=torch.device("cpu"),
    )
    expert = model.layers[0].mlp.routed_experts.inner_experts.w1_EFD
    assert isinstance(expert, DTensor)
    assert expert.device_mesh.size() == 2
    if dist.get_rank() == 0:
        print("all checks passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
