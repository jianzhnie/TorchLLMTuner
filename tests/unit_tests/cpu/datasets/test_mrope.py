"""Standalone rotary coordinates must not require dataset or media backends."""

import subprocess
import sys
import textwrap


def test_dataset_lazy_exports_resolve_to_defining_modules():
    from importlib import import_module

    import llmtuner.datasets as data_api

    assert set(data_api.__all__) == set(data_api._EXPORT_SOURCES)
    for name, source in data_api._EXPORT_SOURCES.items():
        leaf = import_module(f"llmtuner.datasets.{source}")
        assert getattr(data_api, name) is getattr(leaf, name)


def test_mrope_import_and_text_coordinates_without_optional_backends():
    script = textwrap.dedent(
        """
        import sys
        import torch

        class BlockOptionalImports:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'grain', 'torchvision', 'datasets'}:
                    raise ImportError(f'Unexpected optional dependency: {fullname}')
                if fullname.startswith('torch.distributed.checkpoint'):
                    raise ImportError(f'Unexpected checkpoint dependency: {fullname}')

        sys.meta_path.insert(0, BlockOptionalImports())
        from llmtuner.datasets.multimodal.mrope import build_mrope_positions

        result = build_mrope_positions(
            torch.tensor([1, 1, 1]), None, None, torch.tensor([0, 1, 0]),
            spatial_merge_size=2, patch_order='block',
            image_token_id=2, video_token_id=3,
        )
        torch.testing.assert_close(result, torch.tensor([[0,0,0],[1,1,1],[0,0,0]]))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
