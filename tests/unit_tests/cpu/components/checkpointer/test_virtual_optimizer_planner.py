"""Virtual planners preserve native DCP's CPU and partial-chunk contracts."""

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.staging import BlockingAsyncStager

from llmtuner.components.checkpointer.core.chunked_tensor import ChunkedCheckpointTensor
from llmtuner.components.checkpointer.storage.virtual_optimizer_planner import (
    VirtualOptimizerLoadPlanner,
    VirtualOptimizerSavePlanner,
    VirtualOptimizerStager,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_virtual_planners_restore_repartitioned_chunks_and_regular_state(
    tmp_path, dtype, asynchronous
):
    expected = torch.arange(1, 25, dtype=dtype).reshape(4, 2, 3)
    source = {
        "optimizer": {
            "moment": ChunkedCheckpointTensor(
                expected.shape,
                [((0, 0, 0), expected[:2]), ((2, 0, 0), expected[2:])],
            ),
            "step": 50,
        },
        "model": {"weight": torch.tensor([3.0, -2.0])},
    }
    if asynchronous:
        stager = VirtualOptimizerStager(BlockingAsyncStager())
        try:
            upload = dcp.async_save(
                source, checkpoint_id=tmp_path, async_stager=stager, no_dist=True
            )
            # Ensure the upload cannot observe subsequent live-state writes.
            expected.zero_()
            source["model"]["weight"].zero_()
            upload.result()
        finally:
            stager.close()
    else:
        dcp.save(
            source,
            checkpoint_id=tmp_path,
            planner=VirtualOptimizerSavePlanner(),
            no_dist=True,
        )
    local = torch.zeros_like(expected[1:3])
    restored = {
        "optimizer": {
            "moment": ChunkedCheckpointTensor(expected.shape, [((1, 0, 0), local)]),
            "step": 0,
        },
        "model": {"weight": torch.zeros(2)},
    }
    dcp.load(
        restored,
        checkpoint_id=tmp_path,
        planner=VirtualOptimizerLoadPlanner(),
        no_dist=True,
    )
    original = torch.arange(1, 25, dtype=dtype).reshape(4, 2, 3)
    torch.testing.assert_close(local, original[1:3], rtol=0, atol=0)
    torch.testing.assert_close(restored["model"]["weight"], torch.tensor([3.0, -2.0]))
    assert restored["optimizer"]["step"] == 50
    torch.testing.assert_close(
        expected, torch.zeros_like(original) if asynchronous else original
    )
