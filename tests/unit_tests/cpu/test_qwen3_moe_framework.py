"""The experiment's model contract, plan and single-process training loop."""

from importlib import import_module

import pytest
import torch

experiment = import_module("experiments.mini-llmtuner")
ExperimentConfig = experiment.ExperimentConfig
ExperimentTrainer = experiment.ExperimentTrainer
ModelContract = experiment.ModelContract
Qwen3MoeModel = experiment.Qwen3MoeModel
small_qwen3_moe_config = experiment.small_qwen3_moe_config
ParallelConfig = experiment.ParallelConfig


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
    with pytest.raises(NotImplementedError, match="CP/PP"):
        ExperimentTrainer(
            ExperimentConfig(parallel=ParallelConfig(context_parallel_size=2)),
            torch.device("cpu"),
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
            assert_nested_equal(value, actual_optim[name])
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert expected_python == random.getstate()


def test_plan_is_pure_and_stale_bindings_are_rejected():
    resolve_plan = import_module("experiments.mini-llmtuner.resolver").resolve_plan

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
    with pytest.raises(ValueError, match="Stale plan"):
        plan.validate_binding(model)


def test_sequence_shard_divisibility_is_checked_before_training():
    with pytest.raises(ValueError, match="seq_len must be divisible"):
        ExperimentConfig(
            parallel=ParallelConfig(data_parallel_shard_size=1, tensor_parallel_size=2),
            seq_len=15,
        )


def test_checkpoint_rejects_changed_data_settings_before_loading_weights(tmp_path):
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
    load_model_config = import_module(
        "experiments.mini-llmtuner.config"
    ).load_model_config

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
            [
                sys.executable,
                "-m",
                "experiments.mini-llmtuner.train",
                "--device",
                "cpu",
                *arguments,
            ],
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
            assert_nested_equal(value, other)


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_nested_equal(a, b)
    else:
        assert left == right
