"""The common package stays lightweight and exposes no implicit aliases."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import llmtuner
from llmtuner.models import common


def test_an_unknown_name_raises_an_attribute_error() -> None:
    """A miss must look like a miss, not like an empty value."""
    with pytest.raises(AttributeError, match="has no attribute 'Nope'"):
        _ = common.Nope


def test_importing_a_leaf_does_not_import_its_siblings() -> None:
    """Importing attention utilities must not load unrelated MoE modules.

    Checked in a subprocess because ``sys.modules`` is process-global and pytest
    shares one process across every test.
    """
    code = (
        "import sys;"
        "import llmtuner.models.common.attention;"
        "leaked = [m for m in sys.modules if m.startswith("
        "'llmtuner.models.common.moe')];"
        "assert not leaked, leaked;"
        "print('ok')"
    )
    # The package root, so the subprocess resolves ``llmtuner`` the same way
    # this one did (editable install or repo checkout, either is fine).
    repo_root = Path(llmtuner.__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
