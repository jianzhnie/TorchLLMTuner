"""Exact full Trainer resume with EMA in fresh CPU processes.

PYTHONPATH=. python tests/integration_tests/trainer_ema_process_resume.py

Add `corpus` for local JSONL/Grain/single-doc packing; append `async` for
asynchronous intermediate checkpoints (final saves remain synchronous).
Add `first_fit` for local JSONL with production FirstFit instead of concat packing.
Uses real constructor, loader, train loop and DCP. Compares the loaded
boundary before training and the final state with a continuous run. EMA is
programmatic-only, so workers use parse_config followed by the Trainer API.
Append `dp2` to launch each worker job with two ranks and native FSDP;
add `custom` to select the custom fine-grained FSDP engine.
Add `compile` for the production aot_eager whole-model compilation entry.
Add `ac` for production full activation checkpointing and `validate` to
run two validation batches after every training step.
Add `moe` for native Qwen3 MoE with fixed routing and auxiliary loss;
`ep2` additionally shards experts; `dp4` selects four worker ranks.
Add `tp2` with dp4/moe for DP2 x TP2 fixed routing at EP1.
Add `quantile` with moe/ep2 for DeepSeek-V3 sigmoid quantile routing.
Add `pp2` with dp4 for dense DP2 x PP2 Interleaved1F1B, six layers
and four microbatches; this mode excludes moe/tp2/ep2.
CPU only; no accelerator kernel coverage.
"""

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from tempfile import mkdtemp

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.debug import CommDebugMode

from llmtuner.accelerator.device import device_type
from llmtuner.config.ema_config import EMAConfig
from llmtuner.config.validation_config import ValidationConfig
from llmtuner.models.common.aux_loss import AuxLoss
from llmtuner.trainer.cli import parse_config
from llmtuner.trainer.trainer import Trainer
from tests.integration_tests.cli_checkpoint_equivalence import assert_state_equal


def snapshot(trainer):
    state = {
        key: value.state_dict() for key, value in trainer.checkpointer.states.items()
    }
    for key, encoded in state["rng_state"].items():
        rng = pickle.loads(encoded)
        numpy_state = rng["numpy"]
        rng["numpy"] = (
            numpy_state[0],
            torch.from_numpy(numpy_state[1].copy()),
            *numpy_state[2:],
        )
        state["rng_state"][key] = rng

    local_shards = {}

    def materialize(value, path="state"):
        if isinstance(value, DTensor):
            local_shards[path] = {
                "local_tensor": value.to_local().detach().clone(),
                "global_shape": tuple(value.shape),
                "global_stride": tuple(value.stride()),
                "placements": tuple(repr(item) for item in value.placements),
                "mesh": value.device_mesh.mesh.clone(),
                "mesh_dim_names": value.device_mesh.mesh_dim_names,
            }
            return value.full_tensor().detach().clone()
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, dict):
            return {
                key: materialize(item, f"{path}.{key}") for key, item in value.items()
            }
        if isinstance(value, list):
            return [materialize(item, f"{path}[{i}]") for i, item in enumerate(value)]
        if isinstance(value, tuple):
            return tuple(
                materialize(item, f"{path}[{i}]") for i, item in enumerate(value)
            )
        return value

    result = materialize(state)
    result["local_shards"] = local_shards
    return result


def worker(
    root,
    name,
    async_mode,
    implementation="native",
    compiled=False,
    recompute=False,
    validation_enabled=False,
    moe_enabled=False,
    ep_size=1,
    tp_size=1,
    quantile_enabled=False,
):
    folder = root / ("continuous" if name == "continuous" else "resumed")
    steps = (3 if async_mode == "async" else 2) if name == "initial" else 4
    config = parse_config(
        [
            "--model_name_or_path",
            "qwen3",
            "--vocab_size",
            "32",
            "--hidden_size",
            "16",
            "--intermediate_size",
            "32",
            "--num_hidden_layers",
            "2",
            "--num_attention_heads",
            "2",
            "--num_key_value_heads",
            "2",
            "--max_seq_len",
            "8",
            "--global_batch_size",
            str(max(2, int(os.environ.get("WORLD_SIZE", "1")))),
            "--seed",
            "42",
            "--deterministic",
            "true",
            "--enable",
            "true",
            "--last_save_model_only",
            "false",
            "--total_steps",
            "4",
            "--decay_ratio",
            "1.0",
            "--steps",
            str(steps),
            "--dump_folder",
            str(folder),
        ]
    )
    pp_size = 2 if (root / "pp2").exists() else 1
    if pp_size > 1:
        config.model.num_hidden_layers = 6
        config.training.global_batch_size = 8
        config.parallel.pipeline_parallel_size = pp_size
        config.parallel.pp_config.pipeline_parallel_schedule = "Interleaved1F1B"
        config.parallel.pp_config.num_pp_microbatches = 4
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if moe_enabled:
        config.model.model_name_or_path = "qwen3_moe"
        config.model.experts_implementation = "eager"
        config.model.arch_overrides.update(
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            router_aux_loss_coef=0.03,
        )
        config.parallel.ep_config.fixed_router = True
        config.parallel.ep_config.router_aux_loss_coef = 0.03
    if quantile_enabled:
        config.model.model_name_or_path = "deepseek_v3"
        config.model.arch_overrides = dict(
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            n_shared_experts=1,
            n_group=1,
            topk_group=1,
            first_k_dense_replace=0,
            scoring_func="sigmoid",
            norm_topk_prob=True,
            q_lora_rank=None,
            kv_lora_rank=8,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
        )
        config.parallel.ep_config.fixed_router = False
        config.parallel.ep_config.moe_quantile_balancing = True
    config.parallel.expert_parallel_size = ep_size
    config.parallel.tensor_parallel_size = tp_size
    config.parallel.data_parallel_shard_size = world_size // (tp_size * pp_size)
    config.parallel.fsdp_config.fsdp_implementation = implementation
    config.training.activation_checkpoint_mode = "full" if recompute else "none"
    if validation_enabled:
        config.training.validation_config = ValidationConfig(freq=1, steps=2)
    config.training.compile = compiled
    config.training.compile_config.backend = "aot_eager"
    config.checkpoint.async_mode = async_mode
    config.checkpoint.interval = 1
    if name == "resume":
        config.checkpoint.load_step = 2
    if (root / "corpus.jsonl").exists():
        config.dataloader.dataset = "local_jsonl"
        config.dataloader.dataset_path = str(root / "corpus.jsonl")
        config.dataloader.tokenizer_path = str(root / "tokenizer")
        config.dataloader.max_num_documents = 1
        if (root / "first_fit").exists():
            config.dataloader.packing = "first_fit"
    # A scheduled EMA, firing at steps 2 and 4, exercises both restored history
    # and the firing count derived from the restored trainer step.
    config.training.ema_config = EMAConfig(
        half_life_fraction=0.5, step_bias=1, update_every_n_steps=2
    )
    trainer = None
    try:
        trainer = Trainer(config)
        ema_steps = []
        original_ema_step = trainer.ema.step

        def update_ema(current_step):
            with CommDebugMode() as communication:
                original_ema_step(current_step)
            assert communication.get_total_counts() == 0
            ema_steps.append(current_step)

        trainer.ema.step = update_ema
        if quantile_enabled:
            from llmtuner.models.common.moe.layer import collect_moe_layers
            from llmtuner.models.common.moe.routing import QuantileBalancedTopKRouter

            quantile_layers = collect_moe_layers(trainer.model_parts[0])
            assert len(quantile_layers) == 2
            assert all(
                isinstance(block.router, QuantileBalancedTopKRouter)
                for block in quantile_layers
            )
        if pp_size > 1:
            assert trainer.parallel_dims.pp == 2
            assert len(trainer.model_parts) == 2
            assert trainer.pp_schedule.__class__.__name__ == "ScheduleInterleaved1F1B"
        # A pre-validation snapshot must own its storage: otherwise an in-place
        # model update could also mutate the expected value and hide a failure.
        probe_snapshot = snapshot(trainer)
        live_parameter = next(trainer.model_parts[0].parameters())
        original_parameter = live_parameter.detach().clone()
        with torch.no_grad():
            live_parameter.add_(1)
        try:
            changed_snapshot = snapshot(trainer)
            assert any(
                not torch.equal(value, changed_snapshot["model"][key])
                for key, value in probe_snapshot["model"].items()
            ), "Snapshot aliases live model storage"
        finally:
            with torch.no_grad():
                live_parameter.copy_(original_parameter)
        assert_state_equal(snapshot(trainer), probe_snapshot)
        assert (dist.get_world_size() if dist.is_initialized() else 1) == world_size
        if world_size > 1:
            assert any(
                isinstance(parameter, DTensor)
                for part in trainer.model_parts
                for parameter in part.parameters()
            ), "DP2 must exercise actual sharded parameters"
        if tp_size > 1:
            from llmtuner.parallel.expert_parallel.block_conversion import (
                FeatureParallelMoE,
            )

            blocks = [
                module
                for part in trainer.model_parts
                for module in part.modules()
                if isinstance(module, FeatureParallelMoE)
            ]
            assert len(blocks) == 2
            for block in blocks:
                assert block.router.fixed_router
                assert block._tp_seq_group.size() == tp_size
                assert block.routed_experts.inner_experts.w1_EFD.shape[1] == 8
        if ep_size > 1:
            from llmtuner.models.common.moe.layer import collect_moe_layers

            blocks = collect_moe_layers(trainer.model_parts[0])
            assert len(blocks) == 2
            for block in blocks:
                expert = block.routed_experts.inner_experts.w1_EFD
                assert expert.shape[0] == 2
                if world_size == 4:
                    assert isinstance(expert, DTensor)
                    assert expert.device_mesh.size() == 2
                    assert expert.to_local().numel() * 2 == expert.numel()
                assert block.routed_experts.token_dispatcher.ep_group.size() == ep_size
        if implementation == "custom":
            from llmtuner.parallel.fully_shard_parallel.fine_grained_state import (
                FineGrainedFSDPState,
            )

            assert any(
                isinstance(module._get_fsdp_state(), FineGrainedFSDPState)
                for part in trainer.model_parts
                for module in part.modules()
                if hasattr(module, "_get_fsdp_state")
            ), "Custom FSDP must reach its actual runtime state"
        if recompute:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                CheckpointWrapper,
            )

            assert (
                sum(
                    isinstance(module, CheckpointWrapper)
                    for part in trainer.model_parts
                    for module in part.modules()
                )
                == config.model.num_hidden_layers // pp_size
            )
        if name == "initial":
            torch.save(snapshot(trainer), root / f"initial_boundary{rank}.pt")
        if name == "resume":
            assert trainer.checkpointer.load(2)
            assert trainer.step == 2
            torch.save(snapshot(trainer), root / f"loaded_boundary{rank}.pt")
        save = trainer.checkpointer.save

        def save_and_snapshot(*args, **kwargs):
            saved = save(*args, **kwargs)
            if saved and not kwargs.get("last_step") and async_mode == "async":
                assert trainer.checkpointer.save_future is not None
            capture = (
                trainer.step == 2 if name == "initial" else kwargs.get("last_step")
            )
            if capture:
                assert saved
                torch.save(snapshot(trainer), root / f"{name}{rank}.pt")
            return saved

        trainer.checkpointer.save = save_and_snapshot
        validation_steps = []
        original_validate = trainer.validate

        def validate_and_record(step):
            before_validation = snapshot(trainer)

            def auxiliary_state():
                state = {
                    "buffers": {
                        (part_index, name): buffer.detach().clone()
                        for part_index, part in enumerate(trainer.model_parts)
                        for name, buffer in part.named_buffers()
                        if name.endswith(
                            (
                                "tokens_per_expert_E",
                                "instance_acc",
                                "required_bias_histogram_EB",
                            )
                        )
                    },
                    "instances": {
                        (part_index, name): {
                            attr: (
                                value.detach().clone() if value is not None else None
                            )
                            for attr in ("_runtime_denominator", "_step_metric")
                            for value in (getattr(module, attr),)
                        }
                        for part_index, part in enumerate(trainer.model_parts)
                        for name, module in part.named_modules()
                        if isinstance(module, AuxLoss)
                    },
                    "groups": {
                        key: value.detach().clone()
                        for key, value in AuxLoss.group_acc.items()
                    },
                    "counts": dict(AuxLoss._group_counts),
                    "denominator": (
                        AuxLoss._step_denominator.detach().clone()
                        if AuxLoss._step_denominator is not None
                        else None
                    ),
                }
                if moe_enabled:
                    assert len(state["buffers"]) == (6 if quantile_enabled else 4)
                    assert len(state["instances"]) == 2
                return state

            scratch = auxiliary_state()
            original_validate(step)
            assert_state_equal(auxiliary_state(), scratch)
            assert_state_equal(snapshot(trainer), before_validation)
            validation_steps.append(step)

        trainer.validate = validate_and_record
        initial_step = trainer.step
        trainer.train()
        assert ema_steps == list(range(initial_step + 1, steps + 1))
        if quantile_enabled:
            for block in quantile_layers:
                assert torch.count_nonzero(block.expert_bias_E) > 0
                assert not torch.count_nonzero(
                    block.router.quantile_balancer.required_bias_histogram_EB
                )
                assert not torch.count_nonzero(block.tokens_per_expert_E)
        if validation_enabled:
            assert validation_steps == list(range(initial_step + 1, steps + 1))
        else:
            assert not validation_steps
        if compiled:
            assert torch._dynamo.utils.counters["stats"]["unique_graphs"] > 0
    finally:
        try:
            if trainer is not None:
                trainer.close()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def main():
    if device_type != "cpu":
        raise RuntimeError("This check requires a CPU-only environment")
    if len(sys.argv) == 12 and sys.argv[1] == "worker":
        worker(
            Path(sys.argv[2]),
            sys.argv[3],
            sys.argv[4],
            sys.argv[5],
            compiled=sys.argv[6] == "compile",
            recompute=sys.argv[7] == "ac",
            validation_enabled=sys.argv[8] == "validate",
            moe_enabled=sys.argv[9] in {"moe", "quantile"},
            ep_size=int(sys.argv[10]),
            tp_size=int(sys.argv[11]),
            quantile_enabled=sys.argv[9] == "quantile",
        )
        return
    assert len(set(sys.argv[1:])) == len(sys.argv[1:])
    assert set(sys.argv[1:]) <= {
        "corpus",
        "first_fit",
        "async",
        "dp2",
        "dp4",
        "custom",
        "compile",
        "ac",
        "validate",
        "moe",
        "ep2",
        "tp2",
        "quantile",
        "pp2",
    }
    if {"dp2", "dp4"} <= set(sys.argv[1:]):
        raise ValueError("Choose only one data parallel size")
    world_size = 4 if "dp4" in sys.argv[1:] else 2 if "dp2" in sys.argv[1:] else 1
    if "ep2" in sys.argv[1:] and ("moe" not in sys.argv[1:] or world_size < 2):
        raise ValueError("ep2 requires moe and dp2 or dp4")
    pp_size = 2 if "pp2" in sys.argv[1:] else 1
    if pp_size > 1 and (
        world_size != 4 or {"moe", "tp2", "ep2", "quantile"} & set(sys.argv[1:])
    ):
        raise ValueError("pp2 requires dp4 and a dense model without tp2/ep2")
    ep_size = 2 if "ep2" in sys.argv[1:] else 1
    if "quantile" in sys.argv[1:] and (
        "moe" not in sys.argv[1:] or "tp2" in sys.argv[1:] or ep_size != 2
    ):
        raise ValueError("quantile requires moe/ep2 and currently excludes tp2")
    tp_size = 2 if "tp2" in sys.argv[1:] else 1
    if tp_size > 1 and (world_size != 4 or "moe" not in sys.argv[1:] or ep_size != 1):
        raise ValueError("tp2 requires dp4 and moe, with ep2 disabled")
    compiled = "compile" in sys.argv[1:]
    recompute = "ac" in sys.argv[1:]
    implementation = "custom" if "custom" in sys.argv[1:] else "native"
    if implementation == "custom" and world_size == 1:
        raise ValueError("custom requires dp2 or dp4")
    corpus = bool({"corpus", "first_fit"} & set(sys.argv[1:]))
    async_mode = "async" if "async" in sys.argv[1:] else "disabled"
    root = Path(mkdtemp(prefix="llmtuner-ema-process-resume-"))
    if pp_size > 1:
        (root / "pp2").touch()
    if "first_fit" in sys.argv[1:]:
        (root / "first_fit").touch()
    if corpus:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace

        tokenizer_path = root / "tokenizer"
        tokenizer_path.mkdir()
        vocab = {"[UNK]": 0, "[EOS]": 1, **{f"w{i}": i + 2 for i in range(30)}}
        tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.save(str(tokenizer_path / "tokenizer.json"))
        (tokenizer_path / "tokenizer_config.json").write_text(
            json.dumps({"unk_token": "[UNK]", "eos_token": "[EOS]"})
        )
        with (root / "corpus.jsonl").open("w") as handle:
            for row in range(64):
                # With no BOS and one EOS, 16 words yield 16 shifted targets,
                # filling context8 FirstFit chunks without padded token counts.
                words = 16 if "first_fit" in sys.argv[1:] else 12 + row % 7
                text = " ".join(f"w{(row + i) % 30}" for i in range(words))
                handle.write(json.dumps({"text": text}) + "\n")
    print(f"Artifacts: {root}", flush=True)
    for name in ("continuous", "initial", "resume"):
        with (root / f"{name}.log").open("w") as log:
            command = [sys.executable]
            if world_size > 1:
                command += [
                    "-m",
                    "torch.distributed.run",
                    "--rdzv_backend=c10d",
                    "--rdzv_endpoint=127.0.0.1:0",
                    "--rdzv_conf=is_host=true",
                    "--local_addr=127.0.0.1",
                    f"--nproc_per_node={world_size}",
                ]
            command += [
                __file__,
                "worker",
                str(root),
                name,
                async_mode,
                implementation,
                "compile" if compiled else "eager",
                "ac" if recompute else "noac",
                "validate" if "validate" in sys.argv[1:] else "novalidate",
                "quantile"
                if "quantile" in sys.argv[1:]
                else "moe"
                if "moe" in sys.argv[1:]
                else "dense",
                str(ep_size),
                str(tp_size),
            ]
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        assert result.returncode == 0, f"{name} failed: {root / f'{name}.log'}"
        assert "unwaited collective calls" not in (root / f"{name}.log").read_text(), (
            f"{name} left collective work pending: {root / f'{name}.log'}"
        )
    for rank in range(world_size):
        dp_rank = (rank // tp_size) % (world_size // (tp_size * pp_size))
        rank_id = f"dp_rank_{dp_rank}"
        initial = torch.load(root / f"initial{rank}.pt", weights_only=False)
        boundary = torch.load(root / f"loaded_boundary{rank}.pt", weights_only=False)
        if "quantile" in sys.argv[1:]:
            biases = [
                value
                for key, value in initial["model"].items()
                if key.endswith("expert_bias_E")
            ]
            assert len(biases) == 2 and all(torch.count_nonzero(b) for b in biases)
            assert not any("required_bias_histogram" in key for key in initial["model"])
        assert set(initial) == {
            "model",
            "optimizer",
            "lr_scheduler",
            "ema",
            "train_state",
            "dataloader",
            "rng_state",
            "local_shards",
        }
        if world_size > 1:
            for category in ("model", "optimizer", "ema"):
                assert any(
                    key.startswith(f"state.{category}.")
                    for key in initial["local_shards"]
                ), f"Missing local DTensor evidence for {category}"
        else:
            assert not initial["local_shards"]
        assert initial["ema"]
        if corpus:
            assert initial["dataloader"]["version"] == 1
            assert initial["dataloader"]["batch_layout"]["data_identity"].startswith(
                "local-text-v1:"
            )
            assert rank_id in initial["dataloader"]
        else:
            assert initial["dataloader"]["steps"] == 2
        assert_state_equal(boundary, initial)
        continuous = torch.load(root / f"continuous{rank}.pt", weights_only=False)
        resumed = torch.load(root / f"resume{rank}.pt", weights_only=False)
        assert continuous["train_state"] == {
            "step": 4,
            "ntokens_seen": (
                4
                * (8 if pp_size > 1 else max(2, world_size))
                * 8
                // (world_size // pp_size)
            ),
        }
        assert continuous["lr_scheduler"] == {"last_epoch": 4}
        if corpus:
            assert_state_equal(
                continuous["dataloader"]["batch_layout"],
                initial["dataloader"]["batch_layout"],
            )
            assert continuous["dataloader"][rank_id] != initial["dataloader"][rank_id]
        else:
            assert continuous["dataloader"]["steps"] == 4
        assert_state_equal(resumed, continuous)
        before = torch.load(root / f"initial_boundary{rank}.pt", weights_only=False)
        # Independent schedule and averaging formula. Compare against the actual
        # model at each firing; equality across two broken runs is insufficient.
        changed = False
        for key, start in before["ema"].items():
            fqn = key.removeprefix("state.").removesuffix(".ema_params")
            expected = start.lerp(initial["model"][fqn], 1 - 2 ** (-1 / (0.5 * 2)))
            torch.testing.assert_close(initial["ema"][key], expected, rtol=0, atol=0)
            expected = expected.lerp(
                continuous["model"][fqn], 1 - 2 ** (-1 / (0.5 * 3))
            )
            torch.testing.assert_close(continuous["ema"][key], expected, rtol=0, atol=0)
            changed |= not torch.equal(initial["ema"][key], start)
        assert changed, "EMA updates were not exercised"
    print("Trainer/EMA state exact at load boundary and after fresh-process resume.")


if __name__ == "__main__":
    main()
