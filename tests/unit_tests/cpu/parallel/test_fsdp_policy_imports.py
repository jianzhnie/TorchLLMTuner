"""Shared policies must not depend on either FSDP assembly path."""

import subprocess
import sys


def test_policy_import_does_not_load_decoder_or_pattern_assembly():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from llmtuner.parallel.fully_shard_parallel.policies "
            "import get_fsdp_reshard_after_forward_policy as resolve; "
            "assert resolve('default', False) is True; "
            "assert resolve('default', True) is False; "
            "assert 'llmtuner.parallel.fully_shard_parallel.plan' not in sys.modules; "
            "assert 'llmtuner.parallel.fully_shard_parallel.fully_shard_parallel' "
            "not in sys.modules; "
            "from llmtuner.parallel.fully_shard_parallel "
            "import get_fsdp_reshard_after_forward_policy; "
            "assert get_fsdp_reshard_after_forward_policy is resolve",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
