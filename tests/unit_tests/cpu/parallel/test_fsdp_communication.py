"""FSDP shutdown must drain distributed work even on synchronous CPU streams."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from llmtuner.parallel.fully_shard_parallel.communication import (
    drain_fsdp_communications,
)


def test_cpu_stream_shutdown_waits_for_collective_work():
    work = Mock()
    work.wait.return_value = True
    context = SimpleNamespace(all_gather_stream=torch.cpu.Stream())
    group = SimpleNamespace(
        fsdp_params=[],
        _all_gather_result=SimpleNamespace(all_gather_event=None, all_gather_work=work),
    )

    drain_fsdp_communications([context], [group])

    work.wait.assert_called_once_with()
    assert group._all_gather_result is None


def test_cpu_shutdown_retains_buffers_when_collective_work_fails():
    import pytest

    work = Mock()
    work.wait.return_value = False
    result = SimpleNamespace(all_gather_event=None, all_gather_work=work)
    group = SimpleNamespace(fsdp_params=[], _all_gather_result=result)

    with pytest.raises(RuntimeError, match="Work did not complete"):
        drain_fsdp_communications(
            [SimpleNamespace(all_gather_stream=torch.cpu.Stream())], [group]
        )

    assert group._all_gather_result is result
