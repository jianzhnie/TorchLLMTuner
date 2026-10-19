"""Check Kimi stacked exclusions and rejection on four CPU/Gloo ranks.

PYTHONPATH=. torchrun --nproc_per_node=4 \
    tests/integration_tests/kimi_fsdp_pattern_references.py

EP2/custom FSDP: complete projection and module exclusions retain exact copied
values. Partial stacks reject before installing blocks. No training-sync claim.
"""

from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from llmtuner.config.parallel_config import FSDPConfig, ParallelConfig
from llmtuner.errors import UnsupportedCombinationError
from llmtuner.models.kimi.kimi_linear_config import KimiLinearConfig
from llmtuner.models.kimi.moe import KimiSparseMoE
from llmtuner.parallel.parallelize import parallelize_hf_transformers
from llmtuner.parallel.topology import ParallelTopology


class ExpertModel(nn.Module):
    def __init__(self):
        super().__init__()
        config = KimiLinearConfig(
            hidden_size=8,
            num_experts=4,
            num_experts_per_token=2,
            moe_intermediate_size=12,
            num_shared_experts=0,
            num_expert_group=2,
            topk_group=1,
        )
        layer = nn.Module()
        layer.block_sparse_moe = KimiSparseMoE(config)
        self.layers = nn.ModuleList([layer])

    def forward(self, inputs):
        return self.layers[0].block_sparse_moe(inputs)


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        assert dist.get_world_size() == 4
        for mode in ("projection", "module", "gate", "partial"):
            torch.manual_seed(312)
            model = ExpertModel()
            source = model.layers[0].block_sparse_moe
            bias_requires_grad = source.gate.e_score_correction_bias.requires_grad
            originals = {expert.w1.weight for expert in source.experts}
            values = {
                name: torch.stack(
                    [
                        getattr(expert, name).weight.detach().clone()
                        for expert in source.experts
                    ]
                )
                for name in ("w1", "w2", "w3")
            }
            exclusions = (
                originals if mode == "projection" else {source.experts[0].w1.weight}
            )
            if mode in {"module", "gate"}:
                units = (
                    {"": {}}
                    if mode == "module"
                    else {"layers.0.block_sparse_moe.experts": {}}
                )
                ignored = [
                    "layers.0.block_sparse_moe."
                    + ("experts" if mode == "module" else "gate")
                ]
            else:
                units = {"": {"ignored_params": exclusions}}
                ignored = []
            cfg = ParallelConfig(
                expert_parallel_size=2,
                fsdp_config=FSDPConfig(
                    fsdp_implementation="custom",
                    fsdp_apply_modules=units,
                    fsdp_ignored_modules=ignored,
                ),
            )
            dims = ParallelTopology.from_config(cfg, world_size=4)
            dims.build_device_mesh()
            if mode == "partial":
                try:
                    parallelize_hf_transformers(
                        model, cfg=cfg, mesh=dims.spmd_dense_mesh(), parallel_dims=dims
                    )
                except UnsupportedCombinationError as error:
                    assert "part of a stacked" in str(error)
                else:
                    raise AssertionError("Partial stacked exclusion was accepted")
                assert model.layers[0].block_sparse_moe is source
                assert {expert.w1.weight for expert in source.experts} == originals
            else:
                parallelize_hf_transformers(
                    model, cfg=cfg, mesh=dims.spmd_dense_mesh(), parallel_dims=dims
                )
                grouped = model.layers[0].block_sparse_moe.routed_experts.inner_experts
                if mode == "gate":
                    converted = model.layers[0].block_sparse_moe
                    assert converted.router.gate.weight is source.gate.weight
                    assert (
                        converted.expert_bias_E is source.gate.e_score_correction_bias
                    )
                    assert not isinstance(converted.router.gate.weight, DTensor)
                    assert not isinstance(converted.expert_bias_E, DTensor)
                    assert converted.expert_bias_E.requires_grad == bias_requires_grad
                    assert "" not in cfg.fsdp_config.fsdp_apply_modules
                ep_rank = dims.get_mesh("ep").get_local_rank()
                for source_name, target_name in (
                    ("w1", "w1_EFD"),
                    ("w2", "w2_EDF"),
                    ("w3", "w3_EFD"),
                ):
                    actual = getattr(grouped, target_name)
                    sharded = mode == "gate" or (
                        mode == "projection" and source_name != "w1"
                    )
                    assert isinstance(actual, DTensor) == sharded
                    actual = actual.full_tensor() if sharded else actual
                    expected = values[source_name].chunk(2, dim=0)[ep_rank]
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            # Outside local rejection handling: all ranks can still rendezvous.
            dist.barrier()
            print(f"rank={dist.get_rank()} Kimi exclusions {mode}: passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
