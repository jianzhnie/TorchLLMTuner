import subprocess
import sys


def test_layout_config_import_does_not_load_dist_muon_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import llmtuner.config.optimizer_config; "
            "import sys; "
            "assert 'llmtuner.components.optimizer.flex_shard.dist_muon' "
            "not in sys.modules; "
            "assert ('llmtuner.components.optimizer.flex_shard.' "
            "'_optimizer_reshard_runtime') "
            "not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_dist_muon_remains_available_from_package():
    from llmtuner.components.optimizer.flex_shard import DistMuon
    from llmtuner.components.optimizer.flex_shard.dist_muon import (
        DistMuon as LeafDistMuon,
    )
    from llmtuner.components.optimizer.flex_shard.optimizer_reshard import (
        DistMuon as LegacyDistMuon,
    )

    assert DistMuon is LeafDistMuon is LegacyDistMuon


def test_layout_types_keep_identity_across_old_and_config_import_paths():
    from llmtuner.components.optimizer.flex_shard import BucketConfig, ComputeLayout
    from llmtuner.components.optimizer.flex_shard.optimizer_reshard import (
        BucketConfig as LegacyBucketConfig,
    )
    from llmtuner.components.optimizer.flex_shard.optimizer_reshard import (
        ComputeLayout as LegacyComputeLayout,
    )
    from llmtuner.config import BucketConfig as ConfigBucketConfig
    from llmtuner.config import ComputeLayout as ConfigComputeLayout

    assert BucketConfig is ConfigBucketConfig
    assert ComputeLayout is ConfigComputeLayout
    assert LegacyBucketConfig is ConfigBucketConfig
    assert LegacyComputeLayout is ConfigComputeLayout
