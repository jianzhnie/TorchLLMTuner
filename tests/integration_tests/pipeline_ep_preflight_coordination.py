"""Reject rank-local EP semantic drift after real PP splitting, before TP/EP.

Four ranks execute production Trainer construction and PP stage binding. A
model stage callback deliberately changes routing on one rank; all peers must
reject the local plan without starting a parallel stage or pipeline schedule.
"""

from dataclasses import replace
from unittest.mock import patch

import torch.distributed as dist

from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel import parallelize
from llmtuner.trainer.trainer import Trainer
from tests.integration_tests.harness import init_distributed
from tests.integration_tests.pp_ep_equivalence import _cfg


def main() -> None:
    init_distributed()
    try:
        if dist.get_world_size() != 4:
            raise ValueError("This test requires four ranks")
        rank = dist.get_rank()
        original = Qwen3MoeForCausalLM.configure_pipeline_stage
        for failing_rank in (0, 3):

            def configure(model, *, failing_rank=failing_rank, **kwargs):
                original(model, **kwargs)
                if rank == failing_rank:
                    for layer in model.layers:
                        block = layer.mlp
                        block.moe_contract = replace(
                            block.moe_contract, route_scale=2.0
                        )

            with (
                patch.object(
                    Qwen3MoeForCausalLM, "configure_pipeline_stage", configure
                ),
                patch.object(parallelize, "apply_tp") as tp,
                patch.object(parallelize, "apply_ep") as ep,
                patch.object(parallelize, "build_pipeline_schedule") as schedule,
            ):
                try:
                    trainer = Trainer(_cfg())
                except ValueError as error:
                    assert rank == failing_rank
                    assert "Pipeline EP conversion semantics changed" in str(error)
                except RuntimeError as error:
                    assert rank != failing_rank
                    assert str(error) == (
                        "Pipeline local EP plan validation failed on another rank"
                    )
                else:
                    trainer.close()
                    raise AssertionError("Changed routing passed local plan validation")
                tp.assert_not_called()
                ep.assert_not_called()
                schedule.assert_not_called()
            dist.barrier()
            if rank == 0:
                print(f"PASS PP-local EP drift failing_rank={failing_rank}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
