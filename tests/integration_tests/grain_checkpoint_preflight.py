"""CPU/Gloo: malformed Grain cursors reject before live training-state writes.

Run with torchrun --nproc_per_node=2. Tests DDP/native/custom FSDP and
all-rank/rank-0/rank-1 invalid payloads. Append padding, truncation, atomic or
first_fit to check historical/current production data fingerprints instead of
malformed cursors. first_fit uses the production local-text loader with context16
and batch8; other modes use a pretokenized graph. Historical fingerprints are
emulated without executing historical tokenization or packing.
"""

import random
import sys
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import grain.python as grain
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from llmtuner.components.checkpointer.checkpoint_keys import DATALOADER
from llmtuner.config.data_config import DataloaderConfig
from llmtuner.config.llmtuner_config import LLMTunerConfig
from llmtuner.datasets.contracts import TextSequence
from llmtuner.datasets.dataloader import GrainDataLoader
from llmtuner.datasets.loader_builder import build_dataloader, corpus_data_identity
from tests.data_fixtures import write_tokenizer
from tests.integration_tests.cli_checkpoint_equivalence import assert_state_equal
from tests.integration_tests.rng_checkpoint_equivalence import manager, runtime, update


def loader(data_identity=None):
    graph = grain.MapDataset.source(
        [
            TextSequence(input_ids=np.array([1, 2]), labels=np.array([2, 3]))
            for _ in range(32)
        ]
    ).repeat()
    return GrainDataLoader(
        graph,
        dp_world_size=2,
        dp_rank=dist.get_rank(),
        tokenizer=None,
        max_context_length=2,
        num_tokens_per_batch=2,
        data_identity=data_identity,
    )


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        with TemporaryDirectory() if rank == 0 else nullcontext(None) as root:
            paths = [root]
            dist.broadcast_object_list(paths, src=0)
            policy = sys.argv[1] if len(sys.argv) > 1 else None
            if policy not in (None, "padding", "truncation", "atomic", "first_fit"):
                raise ValueError(
                    "Expected optional padding, truncation, atomic or first_fit"
                )
            legacy = current = None
            if policy is not None:
                from tokenizers import Tokenizer

                assets = f"{paths[0]}/assets"
                if rank == 0:
                    write_tokenizer(assets)
                    backend = Tokenizer.from_file(f"{assets}/tokenizer.json")
                    if policy == "padding":
                        backend.enable_padding(length=32)
                    elif policy == "truncation":
                        backend.enable_truncation(max_length=4)
                    backend.save(f"{assets}/tokenizer.json")
                    Path(f"{assets}/rows.jsonl").write_text(
                        ('{"text":"' + "lorem ipsum " * 10 + '"}\n') * 8
                    )
                dist.barrier()
                config = LLMTunerConfig()
                dataset = "obelics" if policy == "atomic" else "local_jsonl"
                config.training.dataloader_config = DataloaderConfig(
                    dataset=dataset,
                    dataset_path=f"{assets}/rows.jsonl",
                    tokenizer_path=assets,
                    data_identity="immutable-test-snapshot",
                    packing="first_fit"
                    if policy == "first_fit"
                    else "concat_then_split",
                )
                config.training.max_seq_len = 16
                current = corpus_data_identity(config, dataset=dataset, repeat=True)
                import json

                dumps = json.dumps

                def legacy_settings(settings, **kwargs):
                    settings = settings.copy()
                    settings.pop("tokenizer_encoding", None)
                    settings.pop("multimodal_encoding", None)
                    settings.pop("text_packing", None)
                    return dumps(settings, **kwargs)

                with patch(
                    "llmtuner.datasets.loader_builder.json.dumps", legacy_settings
                ):
                    legacy = corpus_data_identity(config, dataset=dataset, repeat=True)
                assert current != legacy

            def build_loader(identity):
                if policy != "first_fit":
                    return loader(identity)
                # Use the actual factory/processor/packing graph. Only emulate
                # the old identity; executing the old oversized graph would fail.
                with (
                    patch(
                        "llmtuner.datasets.loader_builder.json.dumps", legacy_settings
                    )
                    if identity == legacy
                    else nullcontext()
                ):
                    result = build_dataloader(
                        config,
                        dp_rank=rank,
                        dp_world_size=2,
                        num_tokens_per_batch=8,
                    )
                assert result.state_dict()["batch_layout"]["data_identity"] == identity
                return result

            mesh = init_device_mesh("cpu", (2,))
            for backend in ("ddp", "native", "custom"):
                selected_mesh = None if backend == "ddp" else mesh
                for failing_rank in (None, 0, 1):
                    folder = f"{paths[0]}/{backend}/{failing_rank}"
                    source = runtime(42, selected_mesh, custom=backend == "custom")
                    writer = manager(source, folder)
                    source_loader = build_loader(legacy)
                    try:
                        writer.states[DATALOADER] = source_loader
                        update(source)
                        update(source)
                        next(iter(source_loader))
                        saved = source_loader.state_dict()
                        if policy is None and (
                            failing_rank is None or rank == failing_rank
                        ):
                            saved[source_loader._rank_id]["next_index"] = -1
                        with patch.object(
                            source_loader, "state_dict", return_value=saved
                        ):
                            assert writer.save(2)
                    finally:
                        writer.close()
                        source_loader.close()
                    target = runtime(123, selected_mesh, custom=backend == "custom")
                    reader = manager(target, folder)
                    target_loader = build_loader(
                        current
                        if failing_rank is None or rank == failing_rank
                        else legacy
                    )
                    try:
                        reader.states[DATALOADER] = target_loader
                        update(target)
                        next(iter(target_loader))
                        states = (
                            target[0],
                            target[2],
                            target[3],
                            target[4],
                            target_loader,
                        )
                        before = deepcopy([state.state_dict() for state in states])
                        python_rng = random.getstate()
                        numpy_rng = np.random.get_state()
                        torch_rng = torch.get_rng_state().clone()
                        failing = failing_rank is None or rank == failing_rank
                        try:
                            reader.load(2)
                        except IndexError as error:
                            assert (
                                policy is None
                                and failing
                                and "next_index" in str(error)
                            )
                        except ValueError as error:
                            assert policy is not None and failing
                            assert "batch layout" in str(error)
                        except RuntimeError as error:
                            assert not failing
                            assert str(error) == (
                                "Checkpoint dataloader preflight validation "
                                "failed on another rank"
                            )
                        else:
                            raise AssertionError("invalid Grain checkpoint accepted")
                        assert_state_equal(
                            [state.state_dict() for state in states], before
                        )
                        assert random.getstate() == python_rng
                        actual_numpy = np.random.get_state()
                        assert actual_numpy[0] == numpy_rng[0]
                        np.testing.assert_array_equal(actual_numpy[1], numpy_rng[1])
                        assert actual_numpy[2:] == numpy_rng[2:]
                        assert torch.equal(torch.get_rng_state(), torch_rng)
                        next(iter(target_loader))
                        dist.barrier()
                    finally:
                        reader.close()
                        target_loader.close()
                if rank == 0:
                    print(
                        f"{backend}: Grain {policy or 'cursor'} rejection, "
                        "all/rank0/rank1 passed",
                        flush=True,
                    )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
