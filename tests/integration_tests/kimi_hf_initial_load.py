"""Real Trainer meta/FSDP/EP initialization from an original HF checkpoint.

Run with two ranks on the selected harness device (CPU/Gloo or NPU/HCCL)
and a shared empty working directory:
LLMTUNER_TEST_KIMI=1 LLMTUNER_TEST_KIMI_MOE=1 LLMTUNER_TEST_KIMI_LATENT=1 \
LLMTUNER_TEST_EP_SIZE=2 \
LLMTUNER_TEST_FSDP_IMPLEMENTATION=native \
torchrun --nproc_per_node=2 tests/integration_tests/kimi_hf_initial_load.py WORKDIR
"""

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from llmtuner.models.configuration import build_model_config_for, resolve_model_class
from llmtuner.parallel.fully_shard_parallel.policies import iter_fsdp_modules
from llmtuner.trainer.trainer import Trainer
from tests.integration_tests.checkpoint_scenarios import kimi_config, train_steps
from tests.integration_tests.cli_checkpoint_equivalence import assert_state_equal
from tests.integration_tests.harness import init_distributed
from tests.integration_tests.trainer_ema_process_resume import snapshot


@torch.no_grad()
def compare_logits(trainer, source, ids):
    try:
        torch.testing.assert_close(
            trainer.model(ids).logits, source(ids).logits, atol=1e-6, rtol=1e-6
        )
    finally:
        # Inference has no backward to reshard units retaining full parameters.
        for module in iter_fsdp_modules(trainer.model):
            module.reshard()


def run(workdir):
    device = init_distributed()
    trainer = None
    try:
        cfg = kimi_config(workdir)
        torch.manual_seed(174)
        model_config = build_model_config_for(cfg)
        source = resolve_model_class(model_config)(model_config).eval()
        expected = {
            key: value.detach().clone() for key, value in source.state_dict().items()
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
        state = trainer.checkpointer.states["model"].state_dict()
        exported = trainer.checkpointer.sd_adapter.to_hf(state)
        assert exported.keys() == expected.keys()
        for key, actual in exported.items():
            if hasattr(actual, "full_tensor"):
                actual = actual.full_tensor()
            torch.testing.assert_close(
                actual.cpu(), expected[key], atol=0, rtol=0, msg=key
            )
        source.to(device)
        ids = torch.tensor([1, 2, 3], device=device)
        trainer.model.eval()
        compare_logits(trainer, source, ids)
        trainer.model.train()
        losses = train_steps(trainer, iter(trainer.dataloader), 2)
        assert len(losses) == 2 and torch.isfinite(torch.tensor(losses)).all()
        before = trainer.checkpointer.sd_adapter.to_hf(
            trainer.checkpointer.states["model"].state_dict()
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
        runtime_before = deepcopy(snapshot(trainer))
        assert trainer.checkpointer.save(trainer.step, last_step=True)
        for name, digest in checkpoint_hashes.items():
            assert (
                hashlib.sha256((checkpoint_path / name).read_bytes()).digest() == digest
            )
        assert_state_equal(snapshot(trainer), runtime_before, "export")
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
        source.load_state_dict(on_disk, strict=True)
        trainer.model.eval()
        compare_logits(trainer, source, ids)
        after = trainer.checkpointer.sd_adapter.to_hf(
            trainer.checkpointer.states["model"].state_dict()
        )
        for key, value in after.items():
            if hasattr(value, "full_tensor"):
                value = value.full_tensor()
            torch.testing.assert_close(
                value.cpu(), before[key], atol=0, rtol=0, msg=key
            )
        trainer.model.train()
        assert len(train_steps(trainer, iter(trainer.dataloader), 1)) == 1
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
