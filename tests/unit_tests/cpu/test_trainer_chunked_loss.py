"""Trainer-level chunked loss: the loop must train identically, chunk or not.

The loss-function equivalence is pinned in
``components/test_chunked_loss.py``; this suite pins the *wiring*: the config
field reaches the body, the forward is entered with ``skip_lm_head=True``, and
the per-chunk backward composes with the loop's token normalization, clipping
and optimizer step. Two real (single-process) Trainers are built from the same
seed -- one plain, one chunked -- and must produce the same loss trajectory and
the same final parameters.
"""

from __future__ import annotations

from copy import deepcopy

from tests.caps import require_env

require_env("dtensor", "pipelining", "spmd_types")


import pytest
import torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)

from llmtuner.config import (
    LLMTunerConfig,
    MetricsConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)
from llmtuner.models.kimi.kimi_k3.configuration import KimiLinearConfig
from llmtuner.trainer.trainer import Trainer
from tests.unit_tests.cpu.models.test_owned_family_construction import (
    small_config,
    text_config,
)

STEPS = 3
# fp32 CPU; the only sanctioned divergence is the per-chunk summation order.
TOL = 1e-5


def _cfg(
    chunked_loss_num_chunks: int,
    dump_folder: str,
    architecture: str = "qwen3_moe",
) -> LLMTunerConfig:
    if architecture == "deepseek_v4":
        config = small_config()
    elif architecture == "qwen3_5_moe_text":
        config = text_config(Qwen3_5MoeTextConfig)
    elif architecture == "kimi_linear":
        config = KimiLinearConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            linear_attn_config={
                "kda_layers": [1],
                "full_attn_layers": [],
                "head_dim": 4,
                "num_heads": 2,
                "short_conv_kernel_size": 3,
            },
        )
    else:
        config = None
    if config is not None:
        model = ModelConfig(
            model_name_or_path=architecture, arch_overrides=config.to_dict()
        )
    else:
        model = ModelConfig(
            model_name_or_path="qwen3_moe",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            arch_overrides={
                "head_dim": 16,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 16,
            },
        )
    return LLMTunerConfig(
        model=model,
        optimizer=OptimizerConfig(learning_rate=3e-4, weight_decay=0.0),
        training=TrainingConfig(
            global_batch_size=8,
            max_seq_len=32,
            steps=STEPS,
            seed=42,
            deterministic=True,
            chunked_loss_num_chunks=chunked_loss_num_chunks,
            dump_folder=dump_folder,
            metrics_config=MetricsConfig(log_freq=1),
        ),
    )


def _run(cfg: LLMTunerConfig) -> tuple[list[float], dict[str, torch.Tensor]]:
    trainer = Trainer(cfg)
    try:
        data_iterator = trainer.data_iterator()
        losses = []
        for _ in range(STEPS):
            trainer.step += 1
            metrics = trainer.train_step(data_iterator)
            assert metrics is not None  # log_freq=1: every step reports
            losses.append(metrics["loss"])
        params = {
            name: p.detach().clone()
            for name, p in trainer.model_parts[0].named_parameters()
        }
        return losses, params
    finally:
        trainer.close()


@pytest.mark.parametrize(
    "architecture", ["qwen3_moe", "deepseek_v4", "qwen3_5_moe_text", "kimi_linear"]
)
@pytest.mark.parametrize("initialization_device", ["cpu", "meta"])
def test_chunked_trainer_matches_plain_trainer(
    tmp_path, architecture, initialization_device
) -> None:
    plain = _cfg(1, str(tmp_path / "plain"), architecture)
    chunked = _cfg(3, str(tmp_path / "chunked"), architecture)
    plain.model.initialization_device = initialization_device
    chunked.model.initialization_device = initialization_device
    plain_losses, plain_params = _run(plain)
    chunked_losses, chunked_params = _run(chunked)

    assert plain_params.keys() == chunked_params.keys()
    for step, (got, want) in enumerate(zip(chunked_losses, plain_losses, strict=True)):
        assert abs(got - want) <= TOL, (
            f"step {step + 1}: chunked loss {got:.6f} vs plain {want:.6f}"
        )
    for name, want in plain_params.items():
        torch.testing.assert_close(chunked_params[name], want, rtol=TOL, atol=TOL)


def test_chunked_loss_config_validates() -> None:
    with pytest.raises(ValueError, match="chunked_loss_num_chunks"):
        TrainingConfig(chunked_loss_num_chunks=0)


@pytest.mark.parametrize(
    "architecture", ["qwen3_moe", "deepseek_v4", "qwen3_5_moe_text", "kimi_linear"]
)
@pytest.mark.parametrize("initialization_device", ["cpu", "meta"])
def test_explicit_trainer_dcp_restores_next_update(
    tmp_path, architecture, initialization_device
):
    cfg = _cfg(3, str(tmp_path), architecture)
    cfg.model.initialization_device = initialization_device
    cfg.checkpoint.enable = True
    cfg.checkpoint.interval = 1

    def snapshot(trainer):
        return deepcopy(
            {
                "model": trainer.model.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "scheduler": trainer.lr_scheduler.state_dict(),
                "trainer": trainer.state_dict(),
                "dataloader": trainer.dataloader.state_dict(),
            }
        )

    original = Trainer(deepcopy(cfg))
    try:
        iterator = original.data_iterator()
        original.step = 1
        original.train_step(iterator)
        saved = snapshot(original)
        assert original.checkpointer.save(1)
        original.step = 2
        expected_metrics = original.train_step(iterator)
        expected = snapshot(original)
        expected_rng = torch.get_rng_state().clone()
    finally:
        original.close()

    restored = Trainer(deepcopy(cfg))
    try:
        assert restored.checkpointer.load(1)
        torch.testing.assert_close(snapshot(restored), saved, atol=0, rtol=0)
        restored.step += 1
        actual_metrics = restored.train_step(restored.data_iterator())
        assert expected_metrics is not None and actual_metrics is not None
        assert actual_metrics["loss"] == expected_metrics["loss"]
        torch.testing.assert_close(snapshot(restored), expected, atol=0, rtol=0)
        assert torch.equal(torch.get_rng_state(), expected_rng)
    finally:
        restored.close()


def test_the_reported_token_count_describes_the_data_not_the_split(tmp_path) -> None:
    """``n_tokens_seen`` must count the corpus read, not the rank's slice.

    It is the counterpart to the loss denominator: the loss is normalized by
    the tokens this step *trained on*, so the cumulative count logged beside it
    has to describe the same data -- if the two disagreed, the loss would not be
    reproducible from the token count. The counter is maintained from the
    unsharded batch (``microbatch``), so on one process it must therefore be
    exactly the batch size times the sequence length times the steps taken,
    whatever the loss body did with those tokens.

    Non-vacuity: the expected value is computed from the config rather than read
    off the trainer, so a counter that reported ``0`` -- which is what it would
    have to be to be absent from the payload -- fails rather than passes.
    """
    cfg = _cfg(1, str(tmp_path / "tokens"))
    expected_per_step = cfg.training.global_batch_size * cfg.training.max_seq_len

    trainer = Trainer(cfg)
    try:
        data_iterator = trainer.data_iterator()
        for step in range(STEPS):
            trainer.step += 1
            metrics = trainer.train_step(data_iterator)
            assert metrics is not None
            assert metrics["n_tokens_seen"] == expected_per_step * (step + 1)
    finally:
        trainer.close()
