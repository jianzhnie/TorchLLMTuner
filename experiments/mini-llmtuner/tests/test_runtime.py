"""Regression checks against the reused Qwen3-MoE implementation."""

import importlib

import pytest
import torch

PACKAGE = "experiments.mini-llmtuner"
Config = importlib.import_module(f"{PACKAGE}.config").ExperimentConfig
Trainer = importlib.import_module(f"{PACKAGE}.trainer").ExperimentTrainer


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for x, y in zip(left, right, strict=True):
            assert_equal(x, y)
    else:
        assert left == right


def test_resume_matches_continuous_training(tmp_path):
    full = Trainer(Config(steps=3, gradient_accumulation_steps=2), torch.device("cpu"))
    full.train()
    split = Trainer(
        Config(steps=1, gradient_accumulation_steps=2, checkpoint=tmp_path),
        torch.device("cpu"),
    )
    split.train()
    resumed = Trainer(
        Config(steps=3, gradient_accumulation_steps=2, resume=tmp_path),
        torch.device("cpu"),
    )
    resumed.train()
    assert_equal(full.model.state_dict(), resumed.model.state_dict())
    assert_equal(full.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert full.data.cursor == resumed.data.cursor == 6
    assert resumed.checkpoint.step == 3
    with pytest.raises(ValueError, match="configuration/topology"):
        Trainer(Config(steps=3, resume=tmp_path, seq_len=8), torch.device("cpu"))


def test_accumulation_matches_combined_batch():
    accumulated = Trainer(Config(steps=1), torch.device("cpu"))
    combined = Trainer(Config(steps=1), torch.device("cpu"))
    batches = [accumulated.data.next_batch(), accumulated.data.next_batch()]
    batches[1]["labels"][:, 3:] = -100
    merged = {key: torch.cat([batch[key] for batch in batches]) for key in batches[0]}
    a = accumulated.engine.train_step(batches)
    b = combined.engine.train_step([merged])
    assert a.targets == b.targets == 18
    assert a.loss_sum == pytest.approx(b.loss_sum, rel=1e-6)
    for x, y in zip(
        accumulated.model.parameters(), combined.model.parameters(), strict=True
    ):
        torch.testing.assert_close(x, y, rtol=1e-4, atol=2e-7)


def test_zero_targets_rejected_before_update():
    trainer = Trainer(Config(steps=1), torch.device("cpu"))
    batch = trainer.data.next_batch()
    batch["labels"].fill_(-100)
    with pytest.raises(ValueError, match="valid targets"):
        trainer.engine.train_step([batch])
    assert not trainer.optimizer.state


def test_hf_llama_uses_existing_training_runtime():
    from transformers import LlamaConfig, LlamaForCausalLM

    adapter = importlib.import_module(f"{PACKAGE}.hf_adapter").LlamaModelAdapter
    model = adapter(
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
            )
        )
    )
    trainer = Trainer(
        Config(steps=1),
        torch.device("cpu"),
        model=model,
        runtime_signature="tiny-llama-v1",
    )
    trainer.assembly.execution.validate_binding(model)
    assert trainer.train()[0]["tokens"] == 15
    trainer.close()
    trainer.close()
    with pytest.raises(RuntimeError, match="closed"):
        trainer.train()


def test_components_are_injected_and_closed_on_failure():
    class Profiler:
        closed = False

        def step(self):
            raise RuntimeError("observer failed")

        def close(self):
            self.closed = True

    profiler = Profiler()
    trainer = Trainer(
        Config(steps=1),
        torch.device("cpu"),
        profiler=profiler,
        scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1
        ),
    )
    with pytest.raises(RuntimeError, match="observer failed"), trainer:
        trainer.train()
    assert trainer.checkpoint.step == 1
    assert profiler.closed
    assert trainer.engine.scheduler.last_epoch == 1


@pytest.mark.parametrize(
    "option,value",
    [("recompute", True), ("quantization", "int8"), ("context_parallel_size", 2)],
)
def test_unimplemented_plans_fail_before_training(option, value):
    parallel = importlib.import_module(f"{PACKAGE}.config").ParallelConfig
    with pytest.raises(NotImplementedError):
        Trainer(Config(parallel=parallel(**{option: value})), torch.device("cpu"))


def test_resolver_is_pure_and_rejects_replaced_parameter():
    model_type = importlib.import_module(f"{PACKAGE}.model").Qwen3MoeModel
    configuration = importlib.import_module(f"{PACKAGE}.config")
    resolve = importlib.import_module(f"{PACKAGE}.resolver").resolve_plan
    model = model_type(configuration.small_qwen3_moe_config())
    parameters = tuple(model.parameters())
    rng = torch.get_rng_state().clone()
    plan = resolve(
        model, configuration.ParallelConfig(tensor_parallel_size=2), world_size=2
    )
    plan.validate_binding(model)
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert torch.equal(rng, torch.get_rng_state())
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    with pytest.raises(ValueError, match="Stale plan parameter"):
        plan.validate_binding(model)


@pytest.mark.parametrize("degree", [0, -1, True, 1.5])
def test_parallel_degrees_require_positive_integers(degree):
    parallel = importlib.import_module(f"{PACKAGE}.config").ParallelConfig
    with pytest.raises(ValueError, match="positive integer"):
        parallel(expert_parallel_size=degree)
