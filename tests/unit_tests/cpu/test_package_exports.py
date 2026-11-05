"""Public exports preserve lightweight imports without hiding configuration."""

import subprocess
import sys


def test_package_import_defers_configuration_and_training_engine():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import llmtuner; "
            "assert 'llmtuner.config' not in sys.modules; "
            "assert 'llmtuner.trainer.trainer' not in sys.modules; "
            "assert 'LLMTunerConfig' in dir(llmtuner); "
            "from llmtuner.config.llmtuner_config import LLMTunerConfig; "
            "assert llmtuner.LLMTunerConfig is LLMTunerConfig; "
            "assert 'llmtuner.trainer.trainer' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
