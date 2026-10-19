"""Check fused EP references through actual EP2/FSDP assembly on four ranks.

PYTHONPATH=. torchrun --nproc_per_node=4 \
    tests/integration_tests/ep_fsdp_pattern_references.py

Append --ignore-modules to exclude the complete original expert subtree.
CPU/Gloo, custom FSDP. Verify copied values, exclusion ownership and relocated
hook identity. This does not validate training or automatic excluded-grad sync.
"""

import argparse
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoConfig

from llmtuner.config.parallel_config import FSDPConfig, ParallelConfig
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.parallel.parallelize import parallelize_hf_transformers
from llmtuner.parallel.topology import ParallelTopology


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ignore-modules", action="store_true")
    args = parser.parse_args()
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        assert dist.get_world_size() == 4
        torch.manual_seed(310)
        model = HFTransformerModel(
            AutoConfig.for_model(
                "qwen3_moe",
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                moe_intermediate_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                num_experts=4,
                num_experts_per_tok=2,
                experts_implementation="eager",
            )
        )
        source = model.layers[0].mlp.experts
        original = source.gate_up_proj
        expected = original.detach().clone()
        expected_down = source.down_proj.detach().clone()
        path = next(name for name, module in model.named_modules() if module is source)
        cfg = ParallelConfig(
            expert_parallel_size=2,
            fsdp_config=FSDPConfig(
                fsdp_implementation="custom",
                fsdp_apply_modules=(
                    {"": {}}
                    if args.ignore_modules
                    else {path: {"ignored_params": {original}, "hook_module": source}}
                ),
                fsdp_ignored_modules=[path] if args.ignore_modules else [],
            ),
        )
        dims = ParallelTopology.from_config(cfg, world_size=4)
        dims.build_device_mesh()
        ep_rank = dims.get_mesh("ep").get_local_rank()
        parallelize_hf_transformers(
            model, cfg=cfg, mesh=dims.spmd_dense_mesh(), parallel_dims=dims
        )
        grouped = model.layers[0].mlp.routed_experts.inner_experts
        assert not isinstance(grouped.w1_EFD, DTensor)
        assert not isinstance(grouped.w3_EFD, DTensor)
        if args.ignore_modules:
            assert not isinstance(grouped.w2_EDF, DTensor)
            torch.testing.assert_close(
                grouped.w2_EDF, expected_down.chunk(2, dim=0)[ep_rank], atol=0, rtol=0
            )
            assert cfg.fsdp_config.fsdp_ignored_modules == [path]
        else:
            assert isinstance(grouped.w2_EDF, DTensor)
            assert grouped._get_fsdp_state().hook_module() is grouped
            assert cfg.fsdp_config.fsdp_apply_modules[path]["ignored_params"] == {
                original
            }
            assert cfg.fsdp_config.fsdp_apply_modules[path]["hook_module"] is source
        gate, up = expected.chunk(2, dim=0)[ep_rank].chunk(2, dim=1)
        torch.testing.assert_close(grouped.w1_EFD, gate, atol=0, rtol=0)
        torch.testing.assert_close(grouped.w3_EFD, up, atol=0, rtol=0)
        print(
            f"rank={dist.get_rank()} EP references and FSDP ownership: passed "
            f"ignore_modules={args.ignore_modules}"
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
