"""The experiment's model contract, plan and single-process training loop."""

import torch

from experiments.qwen3_moe_framework import (
    ExperimentConfig,
    ExperimentTrainer,
    ModelContract,
    Qwen3MoeModel,
    small_qwen3_moe_config,
)
from llmtuner.config import ParallelConfig


def test_qwen3_moe_model_exposes_explicit_contract() -> None:
    model = Qwen3MoeModel(small_qwen3_moe_config())
    assert isinstance(model.contract, ModelContract)
    assert model.contract.architecture == "qwen3_moe"
    assert model.contract.expert_layout == "fused_gate_up"
    assert model.contract.tp_plan
    assert all(path.startswith("model.") for path in model.contract.tp_plan)


def test_single_process_training_and_resume(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    config = ExperimentConfig(
        parallel=ParallelConfig(data_parallel_shard_size=1),
        steps=1,
        checkpoint=checkpoint,
    )
    metrics = ExperimentTrainer(config, torch.device("cpu")).train()
    assert len(metrics) == 1
    assert metrics[0]["loss"] > 0
    resumed = ExperimentTrainer(
        ExperimentConfig(
            parallel=ParallelConfig(data_parallel_shard_size=1),
            steps=2,
            resume=checkpoint,
        ),
        torch.device("cpu"),
    )
    assert resumed.start_step == 1
    assert resumed.train()[0]["step"] == 1


def test_experiment_rejects_unimplemented_cp_pp() -> None:
    import pytest

    with pytest.raises(ValueError, match="TP/EP/FSDP only"):
        ExperimentConfig(
            parallel=ParallelConfig(
                data_parallel_shard_size=1, context_parallel_kv_allgather_size=2
            )
        )


def test_resume_matches_continuous_model_moments_and_rng(tmp_path):
    import random
    from copy import deepcopy

    parallel = ParallelConfig(data_parallel_shard_size=1)
    continuous = ExperimentTrainer(
        ExperimentConfig(parallel=parallel, steps=3), torch.device("cpu")
    )
    reference_metrics = continuous.train()
    expected = deepcopy(continuous.model.state_dict())
    expected_optim = deepcopy(continuous.optimizer.state_dict())
    expected_rng = torch.get_rng_state().clone()
    expected_python = random.getstate()
    path = tmp_path / "resume"
    ExperimentTrainer(
        ExperimentConfig(parallel=parallel, steps=2, checkpoint=path),
        torch.device("cpu"),
    ).train()
    resumed = ExperimentTrainer(
        ExperimentConfig(parallel=parallel, steps=3, resume=path), torch.device("cpu")
    )
    metrics = resumed.train()
    assert metrics == reference_metrics[2:]
    for name, tensor in expected.items():
        assert torch.equal(tensor, resumed.model.state_dict()[name]), name
    actual_optim = resumed.optimizer.state_dict()
    assert expected_optim.keys() == actual_optim.keys()
    for name, value in expected_optim.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, actual_optim[name]), name
        else:
            assert value == actual_optim[name], name
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert expected_python == random.getstate()


def test_plan_is_pure_and_stale_bindings_are_rejected():
    import pytest

    from experiments.qwen3_moe_framework.resolver import resolve_plan

    model = Qwen3MoeModel(small_qwen3_moe_config())
    parameters = dict(model.named_parameters())
    plan = resolve_plan(model, ParallelConfig(data_parallel_shard_size=1), world_size=1)
    assert plan.operations == ()
    plan.validate_binding(model)
    assert all(
        dict(model.named_parameters())[key] is parameter
        for key, parameter in parameters.items()
    )
    model.model.model.norm = torch.nn.Identity()
    with pytest.raises(ValueError, match="Stale plan binding"):
        plan.validate_binding(model)


def test_sequence_shard_divisibility_is_checked_before_training():
    import pytest

    with pytest.raises(ValueError, match="seq_len must be divisible"):
        ExperimentConfig(
            parallel=ParallelConfig(data_parallel_shard_size=1, tensor_parallel_size=2),
            seq_len=15,
        )


def test_checkpoint_rejects_changed_data_settings_before_loading_weights(tmp_path):
    import pytest

    parallel = ParallelConfig(data_parallel_shard_size=1)
    path = tmp_path / "checkpoint"
    trainer = ExperimentTrainer(
        ExperimentConfig(parallel=parallel, checkpoint=path), torch.device("cpu")
    )
    trainer.train()
    target = ExperimentTrainer(
        ExperimentConfig(parallel=parallel, seed=999), torch.device("cpu")
    )
    before = {name: value.clone() for name, value in target.model.state_dict().items()}
    with pytest.raises(ValueError, match="configuration/topology differs"):
        target.checkpoint.load(path)
    assert all(
        torch.equal(value, target.model.state_dict()[name])
        for name, value in before.items()
    )


def test_local_model_config_is_architecture_only(tmp_path):
    from experiments.qwen3_moe_framework.config import load_model_config

    config = small_qwen3_moe_config()
    config.save_pretrained(tmp_path)
    loaded = load_model_config(tmp_path)
    assert loaded.hidden_size == config.hidden_size
    assert loaded._experts_implementation == "eager"
    assert loaded.router_aux_loss_coef == 0
    assert not list(tmp_path.glob("*.safetensors"))


def test_cli_independent_process_checkpoint_continuation(tmp_path):
    import os
    import subprocess
    import sys

    environment = {**os.environ, "LLMTUNER_TEST_DEVICE": "cpu", "OMP_NUM_THREADS": "1"}
    continuous = tmp_path / "continuous"
    split = tmp_path / "split"
    restored = tmp_path / "restored"
    for arguments in (
        ["--steps", "3", "--checkpoint", str(continuous)],
        ["--steps", "2", "--checkpoint", str(split)],
        ["--steps", "3", "--resume", str(split), "--checkpoint", str(restored)],
    ):
        result = subprocess.run(
            [sys.executable, "-m", "experiments.qwen3_moe_framework.train", *arguments],
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    config = ExperimentConfig(
        parallel=ParallelConfig(data_parallel_shard_size=1), steps=3
    )
    expected = ExperimentTrainer(config, torch.device("cpu"))
    actual = ExperimentTrainer(config, torch.device("cpu"))
    assert expected.checkpoint.load(continuous) == actual.checkpoint.load(restored) == 3
    for name, tensor in expected.model.state_dict().items():
        assert torch.equal(tensor, actual.model.state_dict()[name]), name
    for name, value in expected.optimizer.state_dict().items():
        other = actual.optimizer.state_dict()[name]
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, other), name
        else:
            assert value == other, name
