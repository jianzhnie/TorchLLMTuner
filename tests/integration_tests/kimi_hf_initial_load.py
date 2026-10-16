"""Real Trainer meta/FSDP/EP initialization from an original HF checkpoint.

Run with two CPU/Gloo ranks and a shared empty working directory:
LLMTUNER_TEST_KIMI=1 LLMTUNER_TEST_KIMI_MOE=1 LLMTUNER_TEST_KIMI_LATENT=1 \
LLMTUNER_TEST_EP_SIZE=2 LLMTUNER_TEST_PP_SIZE=1 \
LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom \
torchrun --nproc_per_node=2 tests/integration_tests/kimi_hf_initial_load.py WORKDIR
"""

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

from tests.caps import require_env

require_env("spmd_types", "flex_attention")

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from llmtuner.components.checkpointer.model_state import ModelCheckpointState
from llmtuner.models.hf.factory import build_model_config_for
from llmtuner.models.hf.model import HFTransformerModel
from llmtuner.trainer.trainer import Trainer
from tests.integration_tests.pp_checkpoint_equivalence import (
    _cfg,
    _compare_loaded_state,
    _runtime_state,
    _train_steps,
)


def run(workdir):
    dist.init_process_group("gloo")
    trainer = None
    try:
        cfg = _cfg(workdir)
        torch.manual_seed(174)
        source = HFTransformerModel(build_model_config_for(cfg)).eval()
        expected = {
            key: value.detach().clone()
            for key, value in source.model.state_dict().items()
        }
        path = Path(workdir) / "hf-source"
        if dist.get_rank() == 0:
            path.mkdir(parents=True, exist_ok=True)
            save_file(
                {key: value.contiguous() for key, value in expected.items()},
                path / "model.safetensors",
            )
        dist.barrier()
        cfg.checkpoint.initial_load_path = str(path)
        cfg.checkpoint.initial_load_in_hf = True
        cfg.checkpoint.initial_load_model_only = True
        cfg.checkpoint.last_save_model_only = True
        cfg.checkpoint.last_save_in_hf = True
        trainer = Trainer(cfg)
        assert trainer.checkpointer.load()
        state = ModelCheckpointState(trainer.model_parts).state_dict()
        exported = trainer.checkpointer.sd_adapter.to_hf(state)
        assert exported.keys() == expected.keys()
        for key, actual in exported.items():
            if hasattr(actual, "full_tensor"):
                actual = actual.full_tensor()
            torch.testing.assert_close(
                actual.cpu(), expected[key], atol=0, rtol=0, msg=key
            )
        ids = torch.tensor([1, 2, 3])
        trainer.model.eval()
        actual = trainer.model(ids)
        expected_logits = source(ids)
        torch.testing.assert_close(actual, expected_logits, atol=1e-6, rtol=1e-6)
        trainer.model.train()
        losses = _train_steps(trainer, iter(trainer.dataloader), 2)
        assert len(losses) == 2 and torch.isfinite(torch.tensor(losses)).all()
        before = trainer.checkpointer.sd_adapter.to_hf(
            ModelCheckpointState(trainer.model_parts).state_dict()
        )
        before = {
            key: (value.full_tensor() if hasattr(value, "full_tensor") else value)
            .detach()
            .cpu()
            .clone()
            for key, value in before.items()
        }
        assert trainer.checkpointer.save(trainer.step)
        trainer.checkpointer.maybe_wait_for_saving()
        checkpoint_path = Path(trainer.checkpointer._create_checkpoint_id(trainer.step))
        checkpoint_hashes = {
            file.name: hashlib.sha256(file.read_bytes()).digest()
            for file in checkpoint_path.iterdir()
            if file.is_file()
        }
        assert ".metadata" in checkpoint_hashes
        runtime_before = deepcopy(_runtime_state(trainer))
        assert trainer.checkpointer.save(trainer.step, last_step=True)
        for name, digest in checkpoint_hashes.items():
            assert (
                hashlib.sha256((checkpoint_path / name).read_bytes()).digest() == digest
            )
        failures = []
        _compare_loaded_state(
            _runtime_state(trainer), runtime_before, failures, "export"
        )
        assert not failures, failures
        export_path = Path(
            trainer.checkpointer._create_final_checkpoint_id(trainer.step)
        )
        assert export_path == checkpoint_path / "model-export"
        with (export_path / "model.safetensors.index.json").open() as stream:
            index = json.load(stream)
        on_disk = {}
        for filename in set(index["weight_map"].values()):
            on_disk.update(load_file(export_path / filename))
        assert on_disk.keys() == before.keys()
        for key in before:
            torch.testing.assert_close(
                on_disk[key], before[key], atol=0, rtol=0, msg=key
            )
        source.model.load_state_dict(on_disk, strict=True)
        trainer.model.eval()
        torch.testing.assert_close(
            trainer.model(ids), source(ids), atol=1e-6, rtol=1e-6
        )
        after = trainer.checkpointer.sd_adapter.to_hf(
            ModelCheckpointState(trainer.model_parts).state_dict()
        )
        for key, value in after.items():
            if hasattr(value, "full_tensor"):
                value = value.full_tensor()
            torch.testing.assert_close(
                value.cpu(), before[key], atol=0, rtol=0, msg=key
            )
        trainer.model.train()
        assert len(_train_steps(trainer, iter(trainer.dataloader), 1)) == 1
        if dist.get_rank() == 0:
            print(
                "Trainer HF load/export: exact weights, intact state "
                "and continued update passed",
                flush=True,
            )
    finally:
        if trainer is not None:
            trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    run(sys.argv[1])
