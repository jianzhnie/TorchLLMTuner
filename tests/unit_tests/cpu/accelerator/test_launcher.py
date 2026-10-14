from types import SimpleNamespace

import pytest
import torch

from llmtuner.accelerator import distributed


@pytest.fixture(autouse=True)
def restore_launcher_environment(monkeypatch):
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def test_mpi_publishes_local_rank(monkeypatch):
    monkeypatch.setattr(distributed, "device_type", "cpu")
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "3")
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "7")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "8")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    calls = []
    monkeypatch.setattr(
        distributed.torch_dist, "init_process_group", lambda **kw: calls.append(kw)
    )
    distributed.init_dist_mpi("gloo")
    assert distributed.os.environ["LOCAL_RANK"] == "3"
    assert distributed.os.environ["RANK"] == "7"
    assert distributed.os.environ["WORLD_SIZE"] == "8"
    assert calls == [{"backend": "gloo"}]


def test_mpi_vendor_device_uses_its_backend_and_local_rank(monkeypatch):
    monkeypatch.setattr(distributed, "device_type", "xpu")
    monkeypatch.setattr(distributed, "get_distributed_backend", lambda: "xccl")
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "1")
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "5")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "8")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    devices = []
    groups = []
    monkeypatch.setattr(distributed, "set_device", devices.append)
    monkeypatch.setattr(
        distributed.torch_dist, "init_process_group", lambda **kw: groups.append(kw)
    )

    distributed.init_dist_mpi("nccl")

    assert devices == [torch.device("xpu:1")]
    assert groups == [{"backend": "xccl"}]


@pytest.mark.parametrize("existing_address", [None, "localhost"])
def test_slurm_resolves_address_only_when_needed(monkeypatch, existing_address):
    monkeypatch.setattr(distributed, "device_type", "cpu")
    for name, value in {
        "SLURM_PROCID": "1",
        "SLURM_NTASKS": "2",
        "SLURM_LOCALID": "1",
        "SLURM_NODELIST": "node[01-02]",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    if existing_address:
        monkeypatch.setenv("MASTER_ADDR", existing_address)
        monkeypatch.delenv("SLURM_NODELIST")
    commands = []

    def run(args, **kwargs):
        commands.append((args, kwargs))
        return SimpleNamespace(stdout="node01\nnode02\n")

    monkeypatch.setattr(distributed.subprocess, "run", run)
    monkeypatch.setattr(distributed.torch_dist, "init_process_group", lambda **kw: None)
    distributed.init_dist_slurm("gloo")
    assert distributed.os.environ["MASTER_ADDR"] == (existing_address or "node01")
    assert len(commands) == (0 if existing_address else 1)
    if commands:
        assert commands[0] == (
            ["scontrol", "show", "hostnames", "node[01-02]"],
            {"check": True, "capture_output": True, "text": True},
        )


@pytest.mark.parametrize("output", ["", "\n", "   \n"])
def test_slurm_empty_hostnames_do_not_initialize_group(monkeypatch, output):
    monkeypatch.setattr(distributed, "device_type", "cpu")
    for name, value in {
        "SLURM_PROCID": "0",
        "SLURM_NTASKS": "2",
        "SLURM_LOCALID": "0",
        "SLURM_NODELIST": "node[01-02]",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    monkeypatch.setattr(
        distributed.subprocess,
        "run",
        lambda *args, **kw: SimpleNamespace(stdout=output),
    )
    calls = []
    monkeypatch.setattr(
        distributed.torch_dist, "init_process_group", lambda **kw: calls.append(kw)
    )
    with pytest.raises(RuntimeError, match="no hostnames"):
        distributed.init_dist_slurm("gloo")
    assert calls == []
    assert "MASTER_ADDR" not in distributed.os.environ


@pytest.mark.parametrize("port, expected_port", [(None, "12345"), (23456, "23456")])
def test_slurm_cuda_bootstrap_preserves_rank_and_options(
    monkeypatch, port, expected_port
):
    monkeypatch.setattr(distributed, "device_type", "cuda")
    for name, value in {
        "SLURM_PROCID": "5",
        "SLURM_NTASKS": "8",
        "SLURM_LOCALID": "1",
        "MASTER_ADDR": "node01",
        "MASTER_PORT": "12345",
    }.items():
        monkeypatch.setenv(name, value)
    devices = []
    groups = []
    monkeypatch.setattr(torch.cuda, "set_device", devices.append)
    monkeypatch.setattr(
        distributed.torch_dist, "init_process_group", lambda **kw: groups.append(kw)
    )

    distributed.init_dist_slurm("nccl", port=port, timeout="sentinel")

    assert devices == [1]
    assert groups == [{"backend": "nccl", "timeout": "sentinel"}]
    assert distributed.os.environ["RANK"] == "5"
    assert distributed.os.environ["LOCAL_RANK"] == "1"
    assert distributed.os.environ["WORLD_SIZE"] == "8"
    assert distributed.os.environ["MASTER_PORT"] == expected_port
