"""Round-trip indexed HF weights through two CPU pipeline stages.

Run from the repository root:
    PYTHONPATH=. torchrun --standalone --nproc-per-node=2 \
        tests/integration_tests/hf_checkpoint_pp.py

This checks distributed checkpoint key coverage and tensor I/O. It does not
check pipeline forward/backward scheduling or GPU collectives.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch.distributed.device_mesh import init_device_mesh
from transformers import Qwen3MoeConfig

from llmtuner.components.checkpointer.backends.distributed_checkpoint import (
    DistributedCheckpointManager,
)
from llmtuner.components.checkpointer.model_state import ModelStateDictAdapter
from llmtuner.config.checkpoint_config import CheckpointConfig
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel.pipeline_parallel.partition import (
    split_model_into_stages,
)


def main() -> None:
    dist.init_process_group("gloo")
    directory = None
    try:
        assert dist.get_world_size() == 2, "Run this check with two ranks"
        rank = dist.get_rank()
        locations = [tempfile.mkdtemp(prefix="llmtuner-hf-pp-") if rank == 0 else None]
        dist.broadcast_object_list(locations)
        directory = Path(locations[0])
        source = directory / "source"
        torch.manual_seed(17)
        config = Qwen3MoeConfig(
            vocab_size=16,
            hidden_size=16,
            head_dim=8,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            mlp_only_layers=[0, 1],
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            tie_word_embeddings=False,
        )
        whole_model = Qwen3MoeForCausalLM(config)
        reference = ModelStateDictAdapter(config, None).to_hf(whole_model.state_dict())
        if rank == 0:
            source.mkdir()
            keys = sorted(reference)
            shards = [keys[::2], keys[1::2]]
            weight_map = {}
            for index, shard in enumerate(shards, start=1):
                filename = f"model-{index:05d}-of-00002.safetensors"
                save_file(
                    {key: reference[key].contiguous() for key in shard},
                    str(source / filename),
                )
                weight_map.update({key: filename for key in shard})
            (source / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
            )
        dist.barrier()
        _, parts = split_model_into_stages(
            whole_model,
            init_device_mesh("cpu", (2,), mesh_dim_names=("pp",)),
            "1F1B",
            torch.device("cpu"),
            [
                ["model.embed_tokens", "model.layers.0", "model.rotary_emb"],
                ["model.layers.1", "model.norm", "lm_head", "model.rotary_emb"],
            ],
            replicated_module_names=whole_model.pipeline_runtime_modules,
        )
        assert len(parts) == 1
        local_keys = set(parts[0].state_dict())
        assert local_keys and local_keys < reference.keys()
        assert any(key.startswith(f"model.layers.{rank}.") for key in local_keys)
        ownership: list[set[str] | None] = [None, None]
        dist.all_gather_object(ownership, local_keys)
        first, last = ownership
        assert first is not None and last is not None
        assert not first.intersection(last), "PP state is duplicated across stages"
        assert first.union(last) == reference.keys(), "PP state coverage is incomplete"
        with torch.no_grad():
            for part in parts:
                for parameter in part.parameters():
                    parameter.fill_(float("nan"))
        optimizer = torch.optim.SGD(
            [p for part in parts for p in part.parameters()], lr=0.1, foreach=False
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        manager = DistributedCheckpointManager(
            CheckpointConfig(
                enable=True,
                keep_latest_k=0,
                initial_load_path=str(source),
                initial_load_in_hf=True,
                last_save_in_hf=True,
            ),
            model_parts=parts,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            states={},
            folder=str(directory),
            sd_adapter=ModelStateDictAdapter(config, str(source)),
        )
        try:
            assert manager.load()
            for part in parts:
                for key, value in part.state_dict().items():
                    torch.testing.assert_close(value, reference[key], atol=0, rtol=0)
            assert manager.save(1, last_step=True)
        finally:
            manager.close()
        dist.barrier()
        if rank == 0:
            exported = {}
            export_path = directory / "checkpoint" / "step-1"
            index = json.loads(
                (export_path / "model.safetensors.index.json").read_text(
                    encoding="utf-8"
                )
            )
            assert index["weight_map"].keys() == reference.keys()
            for filename in set(index["weight_map"].values()):
                shard = load_file(str(export_path / filename))
                for key, owner in index["weight_map"].items():
                    if owner == filename:
                        exported[key] = shard[key]
            assert exported.keys() == reference.keys()
            for key, value in reference.items():
                torch.testing.assert_close(exported[key], value, atol=0, rtol=0)
            print("PASS: indexed HF load and export across two CPU pipeline stages")
    finally:
        dist.destroy_process_group()
        if directory is not None and rank == 0:
            shutil.rmtree(directory)


if __name__ == "__main__":
    main()
