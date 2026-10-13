from llmtuner.accelerator import distributed


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
