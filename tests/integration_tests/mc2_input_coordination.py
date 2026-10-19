"""Four Gloo ranks: MC2 local preflight must span multiple EP subgroups.

Run with torchrun --nproc_per_node=4. No fused kernel is executed. The block
swap is substituted to isolate apply_ep's stage error-group wiring.
"""

from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist

from llmtuner.config.parallel_config import ExpertParallelConfig, ParallelConfig
from llmtuner.parallel.expert_parallel import expert_parallel, mc2_dispatch
from llmtuner.parallel.expert_parallel.token_dispatch import AllToAllTokenDispatcher


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=20))
    try:
        assert dist.get_world_size() == 4
        groups = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
        group = groups[dist.get_rank() // 2]
        dispatcher = AllToAllTokenDispatcher(4, 2)
        dispatcher.set_ep_group(ep_group=group)
        experts = mc2_dispatch.MC2GroupedExperts(4, 3, 2, use_grouped_mm=False).half()
        with patch.object(mc2_dispatch, "load_mc2_ops", return_value=object()):
            routed = mc2_dispatch.MC2RoutedExperts(experts, dispatcher)
        model = torch.nn.Sequential(routed)
        cfg = ParallelConfig(
            expert_parallel_size=2,
            ep_config=ExpertParallelConfig(ep_token_dispatcher="mc2"),
        )
        with patch.object(expert_parallel, "swap_hf_moe_blocks", return_value=1):
            expert_parallel.apply_ep(
                model, cfg, ep_group=group, error_group=dist.group.WORLD
            )
        for failing_rank in (0, 3):
            scores = torch.ones(1, 2, dtype=torch.float16)
            if dist.get_rank() == failing_rank:
                scores = scores[:, :1]
            with (
                patch.object(
                    mc2_dispatch,
                    "mc2_communication_context",
                    return_value=(object(), "test", 2),
                ),
                patch.object(
                    dispatcher,
                    "prepare_dispatch_counts",
                    side_effect=AssertionError("peer entered count exchange"),
                ),
            ):
                caught = None
                try:
                    routed(
                        torch.ones(1, 4, dtype=torch.float16),
                        scores,
                        torch.tensor([[0, 1]]),
                        torch.tensor([1, 1, 0, 0]),
                    )
                except (ValueError, RuntimeError) as error:
                    caught = error
                assert caught is not None
                if dist.get_rank() != failing_rank:
                    assert "MC2 input preparation" in str(caught)
            dist.barrier()
        if dist.get_rank() == 0:
            print("MC2 stage error group reaches both EP subgroups before exchange.")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
