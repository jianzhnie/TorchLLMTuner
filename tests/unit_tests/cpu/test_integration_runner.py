"""Launch rank discovery must preserve scripts' documented world sizes."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "command",
    [
        "torchrun --nproc_per_node=4 script.py",
        "torchrun --standalone --nproc_per_node=4",
        "torchrun --nnodes=1 --nproc-per-node 4 script.py",
    ],
)
def test_rank_discovery_accepts_intervening_flags_and_both_spellings(tmp_path, command):
    runner = Path(__file__).parents[2] / "integration_tests" / "run_all.py"
    spec = importlib.util.spec_from_file_location("integration_runner", runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    script = tmp_path / "example.py"
    script.write_text(f'"""{command}"""\n')
    assert module._torchrun_args(script) == ["--nproc_per_node=4"]


@pytest.mark.parametrize(
    "script_name", ["pp_checkpoint_equivalence", "async_snapshot_equivalence"]
)
@pytest.mark.parametrize("full_returncode", [0, 1])
def test_checkpoint_runner_shares_folder_and_stops_on_full_failure(
    monkeypatch, full_returncode, script_name
):
    runner = Path(__file__).parents[2] / "integration_tests" / "run_all.py"
    spec = importlib.util.spec_from_file_location("integration_runner", runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [str(runner), "--master-port", "29897", script_name],
    )
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert "--nnodes=1" in command
        assert "--master-addr=127.0.0.1" in command
        assert "--master-port=29897" in command
        assert Path(command[-1]).is_dir()
        return SimpleNamespace(returncode=full_returncode if len(calls) == 1 else 0)

    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.main() == full_returncode
    assert calls[0][-2] == "full"
    if full_returncode == 0:
        assert len(calls) == 2
        assert calls[1][-2] == "resume"
        assert calls[0][-1] == calls[1][-1]
    else:
        assert len(calls) == 1
    assert not Path(calls[0][-1]).exists()


@pytest.mark.parametrize("port", ["0", "-1", "65536"])
def test_invalid_master_port_fails_before_launch(monkeypatch, port):
    runner = Path(__file__).parents[2] / "integration_tests" / "run_all.py"
    spec = importlib.util.spec_from_file_location("integration_runner", runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.sys, "argv", [str(runner), "--master-port", port])

    def run(*args, **kwargs):
        pytest.fail("Invalid rendezvous configuration must not launch workers")

    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2


@pytest.mark.parametrize(
    "name", ["cli_checkpoint_equivalence", "trainer_ema_process_resume"]
)
@pytest.mark.parametrize("list_only", [False, True])
def test_python_driver_runs_once_without_outer_distributed_launcher(
    monkeypatch, capsys, name, list_only
):
    runner = Path(__file__).parents[2] / "integration_tests" / "run_all.py"
    spec = importlib.util.spec_from_file_location("integration_runner", runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module.sys, "argv", [str(runner), name] + (["--list"] if list_only else [])
    )
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command == [
            module.sys.executable,
            f"tests/integration_tests/{name}.py",
        ]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.main() == 0
    assert len(calls) == (0 if list_only else 1)
    output = capsys.readouterr().out
    assert module.sys.executable in output
    assert "torchrun" not in output
