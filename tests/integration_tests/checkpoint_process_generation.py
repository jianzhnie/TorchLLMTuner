"""Real PROCESS uploads preserve generations across writer reuse and failure.

Run with torchrun --nproc-per-node=2. Host staging uses shared memory without
requesting pinning; this does not establish NPU pinned-memory support.
For two nodes with one rank each, set CHECKPOINT_TEST_ROOT to the same shared
directory on both nodes. Use a fresh directory for each run.
"""

import os
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType

from llmtuner.components.checkpointer.storage.checkpoint_writer import (
    GenerationFileSystemWriter,
)
from tests.integration_tests.harness import init_distributed


class PublicationFailureWriter(GenerationFileSystemWriter):
    """Inject a coordinator failure inside the actual background worker."""

    def finish(self, metadata, results):
        if (Path(self.path) / "inject-failure").exists():
            raise OSError("injected PROCESS metadata failure")
        return super().finish(metadata, results)


def main() -> None:
    device = init_distributed(timeout=timedelta(seconds=90))
    upload_group = dist.new_group(backend="gloo")
    options = StagingOptions(
        use_pinned_memory=False,
        use_shared_memory=True,
        use_async_staging=False,
        use_non_blocking_copy=False,
    )
    stager = None
    try:
        assert dist.get_world_size() == 2
        shared_root = os.environ.get("CHECKPOINT_TEST_ROOT")
        context = nullcontext(shared_root) if shared_root else TemporaryDirectory()
        with context as directory:
            if shared_root:
                Path(shared_root).mkdir(parents=True, exist_ok=True)
                dist.barrier()
            roots = [directory if dist.get_rank() == 0 else None]
            dist.broadcast_object_list(roots)
            root = Path(roots[0])
            writer = PublicationFailureWriter(root)
            previous = {}
            published = None
            for offset in (0, 3, 8, 13, 21):
                # Match the framework: PyTorch releases the stager's copy
                # dispatch after staging, so reconstruct it for the next save.
                if stager is not None:
                    stager.close()
                stager = DefaultStager(options)
                expected = torch.tensor([2.0, 7.0], device=device) + offset
                failing = offset == 13
                metadata_before = (root / ".metadata").read_bytes() if failing else None
                if failing and dist.get_rank() == 0:
                    (root / "inject-failure").touch()
                dist.barrier()
                response = dcp.async_save(
                    {"weight": expected},
                    storage_writer=writer,
                    process_group=upload_group,
                    async_checkpointer_type=AsyncCheckpointerType.PROCESS,
                    async_stager=stager,
                )
                try:
                    if hasattr(response, "staging_completion"):
                        response.staging_completion.result(timeout=90)
                        response.upload_completion.result(timeout=90)
                    else:
                        response.result(timeout=90)
                except BaseException as error:
                    assert failing and "injected PROCESS metadata failure" in str(error)
                else:
                    assert not failing, "PROCESS metadata failure was not propagated"
                dist.barrier()
                if failing:
                    assert (root / ".metadata").read_bytes() == metadata_before
                    if dist.get_rank() == 0:
                        (root / "inject-failure").unlink()
                    dist.barrier()
                else:
                    published = expected
                restored = {"weight": torch.zeros_like(expected)}
                dcp.load(restored, checkpoint_id=root)
                torch.testing.assert_close(
                    restored["weight"], published, rtol=0, atol=0
                )
                for filename, contents in previous.items():
                    assert (root / filename).read_bytes() == contents
                shards = {
                    path.name: path.read_bytes() for path in root.glob("*.distcp")
                }
                assert shards.keys() > previous.keys()
                previous = shards
            assert not list(root.glob("*.tmp"))
            dist.barrier()
            print(
                f"rank={dist.get_rank()}: PROCESS writer reuse, metadata failure, "
                "exact old loads and retry passed",
                flush=True,
            )
    finally:
        if stager is not None:
            stager.close()
        dist.destroy_process_group(upload_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
