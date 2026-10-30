"""Run integration scripts with their documented Python or torchrun launcher.

Each ``*_equivalence.py`` (and sibling) script in this directory carries its
own launch command in its module docstring (``PYTHONPATH=. torchrun
--nproc_per_node=N <script>``). This runner reads that command rather than
duplicating it, so a script that needs a different rank count keeps a single
source of truth.

Usage (from the repo root):

    python tests/integration_tests/run_all.py --list          # just enumerate
    python tests/integration_tests/run_all.py                  # run everything
    python tests/integration_tests/run_all.py ac_equivalence   # one script
    python tests/integration_tests/run_all.py ac_equivalence.py tp_equivalence

Every script runs with ``PYTHONPATH=.``; Python drivers launch once, while
distributed scripts use torchrun; the
exit code is non-zero if any script fails, and a pass/fail table is printed
at the end either way.

Rendezvous uses static loopback on one node. For concurrent runners, supply
distinct ports with ``--master-port PORT`` (default: 29500).
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

_TORCHRUN_RE = re.compile(r"torchrun\b[^\n]*?--nproc[_-]per[_-]node(?:=|\s+)(\d+)")


def _torchrun_args(script: Path) -> list[str]:
    """The torchrun flags from the script's docstring command, or the default.
    A docstring line-continuation (backslash-newline) is gone by
    the time the docstring is parsed, so the separator between
    the flag and the script path may be plain whitespace, a
    literal backslash, or both.
    """
    docstring = ast.get_docstring(ast.parse(script.read_text())) or ""
    match = _TORCHRUN_RE.search(docstring)
    if match:
        return [f"--nproc_per_node={match.group(1)}"]
    return ["--nproc_per_node=2"]


def launch_command(script: Path, master_port: int) -> list[str]:
    """Keep self-launching Python drivers outside an outer rank launcher.

    Match the script's own documented command, not prose mentioning Python.
    Scripts with both device-local and distributed examples retain torchrun.
    """
    docstring = ast.get_docstring(ast.parse(script.read_text())) or ""
    python_driver = re.search(
        rf"\bpython(?:3)?\s+(?:[\w./-]+/)?{re.escape(script.name)}(?=\s|$)",
        docstring,
    )
    path = str(script.relative_to(REPO_ROOT))
    if python_driver and not _TORCHRUN_RE.search(docstring):
        return [sys.executable, path]
    return [
        "torchrun",
        "--nnodes=1",
        "--master-addr=127.0.0.1",
        f"--master-port={master_port}",
        *_torchrun_args(script),
        path,
    ]


def discover() -> list[Path]:
    """Every runnable script in this directory, sorted by name."""
    support_modules = {
        "__init__.py",
        "alignment_history.py",
        "alignment_module_trace.py",
        "harness.py",
        "npu_trainer_alignment_config.py",
        "run_all.py",
    }
    return sorted(p for p in SCRIPT_DIR.glob("*.py") if p.name not in support_modules)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "scripts",
        nargs="*",
        help="Script names (with or without .py) to run; default is all.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="Local rendezvous port; choose another port for concurrent runners.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the discovered scripts and launch commands, then exit.",
    )
    args = parser.parse_args()
    if not 1 <= args.master_port <= 65535:
        parser.error("--master-port must be between 1 and 65535")

    scripts = discover()
    if args.scripts:
        wanted = {s.removesuffix(".py") for s in args.scripts}
        scripts = [p for p in scripts if p.stem in wanted]
        missing = wanted - {p.stem for p in scripts}
        if missing:
            print(f"no such integration script(s): {sorted(missing)}")
            return 2

    if args.list:
        for script in scripts:
            command = " ".join(launch_command(script, args.master_port))
            print(f"{script.name:45s} {command}")
        print(f"{len(scripts)} scripts")
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    results: list[tuple[str, bool, float]] = []
    for script in scripts:
        cmd = launch_command(script, args.master_port)
        print(f"=== {' '.join(cmd)}", flush=True)
        start = time.monotonic()
        if script.stem in {"pp_checkpoint_equivalence", "async_snapshot_equivalence"}:
            # Separate processes must share the same checkpoint and trajectory.
            with tempfile.TemporaryDirectory(
                prefix="llmtuner-pp-checkpoint-"
            ) as folder:
                proc = subprocess.run([*cmd, "full", folder], cwd=REPO_ROOT, env=env)
                if proc.returncode == 0:
                    proc = subprocess.run(
                        [*cmd, "resume", folder], cwd=REPO_ROOT, env=env
                    )
        else:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
        elapsed = time.monotonic() - start
        results.append((script.name, proc.returncode == 0, elapsed))

    width = max(len(name) for name, _, _ in results)
    print("\n=== integration test summary ===")
    for name, ok, elapsed in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {elapsed:6.1f}s")
    failed = [name for name, ok, _ in results if not ok]
    print(f"{len(results) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
