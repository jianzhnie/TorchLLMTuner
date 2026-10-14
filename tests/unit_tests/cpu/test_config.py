"""Config validation: the rejections, since the acceptances are the default runs.

Every ``__post_init__`` in ``llmtuner/config/`` guards a combination that would
otherwise fail late -- inside a distributed launch, a checkpoint load, or a mesh
build -- or, worse, silently train something other than what was asked for. The
valid defaults are exercised by every other test in this suite (they construct a
``LLMTunerConfig``); these are the branches that only run when a user is wrong.

Each ``with pytest.raises`` is paired with a positive case where the guard has a
boundary worth pinning (``-1`` is allowed for dp_shard, ``0`` is not), so the
test cannot pass by the constructor rejecting everything.
"""

from __future__ import annotations

from tests.caps import require_env, skip_without

# ``ParallelConfig.__post_init__`` imports ``torch.distributed.pipelining`` to
# validate the schedule name, so *every* config construction needs it -- that,
# not DTensor, is this module's hard dependency. The two tests that also read
# the checkpointer's state keys carry their own ``dcp`` guard below.
require_env("pipelining")


import re
import subprocess
import sys

import pytest
import torch
from transformers import HfArgumentParser

from llmtuner.config import (
    CheckpointConfig,
    CompileConfig,
    DataloaderConfig,
    LLMTunerConfig,
    LRSchedulerConfig,
    MemoryBudgetACConfig,
    MetricsConfig,
    ModelConfig,
    OptimizerConfig,
    ParallelConfig,
    ProfilerConfig,
    RegionACConfig,
    SelectiveACConfig,
    TrainingConfig,
)
from llmtuner.config.cli_config import PARSER_GROUPS, PROGRAMMATIC_ONLY, cli_view
from llmtuner.config.parallel_config import (
    ContextParallelConfig,
    FSDPConfig,
    PipelineParallelConfig,
    TensorParallelConfig,
)


def test_cli_view_rejects_class_variables_in_hidden_fields(monkeypatch):
    from dataclasses import make_dataclass
    from typing import ClassVar

    config = make_dataclass("Config", [("shared", ClassVar[int], 1)])
    monkeypatch.setitem(PROGRAMMATIC_ONLY, config, frozenset({"shared"}))
    with pytest.raises(RuntimeError, match="PROGRAMMATIC_ONLY is stale"):
        cli_view(config)


def test_cli_views_preserve_hidden_field_properties():
    from dataclasses import fields

    for config, hidden in PROGRAMMATIC_ONLY.items():
        original_fields = {field.name: field for field in fields(config)}
        view_fields = {field.name: field for field in fields(cli_view(config))}
        for name in hidden:
            original = original_fields[name]
            view = view_fields[name]
            assert view is not original
            assert original.init
            assert not view.init
            for attribute in (
                "default",
                "default_factory",
                "metadata",
                "repr",
                "compare",
                "hash",
                "kw_only",
            ):
                assert getattr(view, attribute) == getattr(original, attribute)


from llmtuner.errors import ConfigError, UnsupportedCombinationError


def test_cli_config_parsing_does_not_require_training_engine():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "sys.modules['llmtuner.trainer.trainer'] = None; "
            "from llmtuner.trainer.cli import parse_config; "
            "assert parse_config(['--steps', '1']).steps == 1",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


# -- ParallelConfig ----------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "data_parallel_replicate_size",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "context_parallel_kv_allgather_size",
        "expert_parallel_size",
    ],
)
def test_a_size_below_one_is_rejected(field: str) -> None:
    """A zero size is a division by zero three layers down; catch it at parse."""
    with pytest.raises(ValueError, match=f"{field} must be >= 1"):
        ParallelConfig(**{field: 0})


def test_dp_shard_accepts_minus_one_as_derive_but_not_zero() -> None:
    """``-1`` means "derive it", ``0`` is a typo for neither."""
    assert ParallelConfig(data_parallel_shard_size=-1).data_parallel_shard_size == -1
    with pytest.raises(ValueError, match="must be >= 1 or -1"):
        ParallelConfig(data_parallel_shard_size=0)


def test_tp_config_rejects_conflicting_pattern_styles() -> None:
    with pytest.raises(ConfigError, match="conflicting styles"):
        ParallelConfig(
            tp_config=TensorParallelConfig(
                tp_colwise_modules=["layers.*"], tp_rowwise_modules=["layers.*"]
            )
        )


def test_tp_config_rejects_unsupported_sequence_module_plan() -> None:
    with pytest.raises(UnsupportedCombinationError, match="tp_sequence_modules"):
        ParallelConfig(
            tp_config=TensorParallelConfig(tp_sequence_modules=["layers.*.norm"])
        )


def test_tp_config_accepts_distinct_pattern_styles() -> None:
    config = ParallelConfig(
        tp_config=TensorParallelConfig(
            tp_colwise_modules=["*.up_proj"], tp_rowwise_modules=["*.down_proj"]
        )
    )
    assert config.tp_config.tp_colwise_modules == ["*.up_proj"]
    assert config.tp_config.tp_rowwise_modules == ["*.down_proj"]


@pytest.mark.parametrize("value", [True, 1.0, -1.0, float("nan"), "2", None])
def test_dp_shard_requires_an_integer(value) -> None:
    with pytest.raises(ConfigError, match="data_parallel_shard_size"):
        ParallelConfig(data_parallel_shard_size=value)


@pytest.mark.parametrize(
    "field",
    [
        "fsdp_forward_prefetch_count",
        "fsdp_backward_prefetch_count",
        "train_timeout_seconds",
    ],
)
@pytest.mark.parametrize("value", [True, 1.5, float("nan"), "2", None, -1])
def test_parallel_runtime_counts_require_valid_integers(field, value) -> None:
    with pytest.raises(ConfigError, match=field):
        (
            FSDPConfig(**{field: value})
            if field.startswith("fsdp_")
            else PipelineParallelConfig(**{field: value})
            if field == "num_pp_microbatches"
            else ParallelConfig(**{field: value})
        )


def test_fsdp_prefetch_allows_zero_and_explicit_windows() -> None:
    config = ParallelConfig(
        fsdp_config=FSDPConfig(
            fsdp_forward_prefetch_count=0, fsdp_backward_prefetch_count=3
        )
    )
    assert config.fsdp_config.fsdp_forward_prefetch_count == 0
    assert config.fsdp_config.fsdp_backward_prefetch_count == 3


@pytest.mark.parametrize("value", [True, 1.0, float("nan"), "2", None, 0, -1])
@pytest.mark.parametrize("shard_size", [-1, 1])
def test_dp_derivation_requires_positive_integer_world_size(value, shard_size) -> None:
    config = ParallelConfig(data_parallel_shard_size=shard_size)
    with pytest.raises(ConfigError, match="world_size must be a positive integer"):
        config.derive_dp(value)


def test_an_empty_load_balancer_is_rejected_rather_than_coerced() -> None:
    """``""`` is not ``None``: one disables, the other is a mistake."""
    with pytest.raises(ValueError, match="cannot be an empty string"):
        ParallelConfig(
            cp_config=ContextParallelConfig(context_parallel_load_balancer="")
        )
    assert (
        ParallelConfig(
            cp_config=ContextParallelConfig(context_parallel_load_balancer=None)
        ).cp_config.context_parallel_load_balancer
        is None
    )


def test_a_load_balancer_that_is_not_a_known_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        ParallelConfig(
            cp_config=ContextParallelConfig(context_parallel_load_balancer="roundrobin")
        )


def test_ptrr_is_rejected_at_config_time_not_at_the_first_forward() -> None:
    """It is a *recognized* name that llmtuner does not implement.

    Keeping it out of the message's "must be one of" list would report a
    deliberate, supported-in-upstream strategy as a typo, so it stays in the
    vocabulary and is refused explicitly instead. The failure used to surface as
    a NotImplementedError from the mesh builder, after process groups and model
    build -- this pins it to the config.
    """
    with pytest.raises(NotImplementedError, match="ptrr"):
        ParallelConfig(
            cp_config=ContextParallelConfig(context_parallel_load_balancer="ptrr")
        )


def test_sequence_parallel_cannot_be_turned_off() -> None:
    """llmtuner's TP is sequence-parallel by construction, so the flag has no
    "off" to select.

    It used to be read by no line of the package: ``enable_sequence_parallel
    =false`` was accepted and behaved exactly as ``true`` -- the silent no-op
    this project's own rule forbids. There is no replicated-activation TP
    realization to fall back to, so refusing is the only honest answer.
    """
    with pytest.raises(NotImplementedError, match="enable_sequence_parallel"):
        ParallelConfig(
            tensor_parallel_size=2,
            tp_config=TensorParallelConfig(enable_sequence_parallel=False),
        )
    # The default stays on, and saying so explicitly is not an error.
    assert ParallelConfig().tp_config.enable_sequence_parallel is True
    assert ParallelConfig(
        tp_config=TensorParallelConfig(enable_sequence_parallel=True)
    ).tp_config.enable_sequence_parallel
    assert not ParallelConfig(
        tensor_parallel_size=1,
        tp_config=TensorParallelConfig(enable_sequence_parallel=False),
    ).tp_config.enable_sequence_parallel


def test_an_unknown_context_parallel_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        ParallelConfig(
            cp_config=ContextParallelConfig(context_parallel_strategy="ring")
        )


def test_ulysses_cannot_share_a_load_balancer() -> None:
    """The permutation would land in the tokens but not in the mask.

    Ulysses hands flex the full sequence, and flex's only mask input is a mask
    over the order of the tensors it is given. The kernel rebuilds the
    full-length causal mask from the length alone -- correct for a contiguous
    split, where the causal mask is the same no matter how it was sharded. A
    load-balanced shard is a rearrangement of the sequence, though, and the
    rebuild cannot see which one, so attention runs over the permutation. It
    is bit-exactly the model trained on the reordered corpus, with nothing
    raised. kv_allgather has no such constraint: its pre-shard sequence -- and
    so the order the mask was built and sharded in -- is recoverable from the
    rearranged shards by the all-gather.

    The balancer defaults to ``None`` (contiguous sharding), so the illegal
    pairing has to be asked for: it is the *combination* that is refused, not
    ulysses on its own.
    """
    with pytest.raises(
        UnsupportedCombinationError, match="requires.*load_balancer=None"
    ):
        ParallelConfig(
            cp_config=ContextParallelConfig(
                context_parallel_strategy="ulysses",
                context_parallel_load_balancer="headtail",
            )
        )
    # Ullysses alone is legal, and so is the default balancer for the default
    # strategy -- the refusal is exactly the pair.
    assert (
        ParallelConfig(
            cp_config=ContextParallelConfig(context_parallel_strategy="ulysses")
        ).cp_config.context_parallel_strategy
        == "ulysses"
    )

    cfg = ParallelConfig(
        cp_config=ContextParallelConfig(
            context_parallel_strategy="ulysses", context_parallel_load_balancer=None
        )
    )
    assert cfg.cp_config.context_parallel_load_balancer is None
    # The default balancer is upstream's: None means contiguous sharding, and
    # the balanced split is opt-in. The pairing is only constrained the other
    # way round, so this is the one direction that must hold.
    assert ParallelConfig().cp_config.context_parallel_load_balancer is None


def _symm_mem_supported() -> bool:
    """Mirror of the guard in ``ParallelConfig.__post_init__``.

    Symmetric memory needs a device that can do it: any ROCm build, or an
    NVIDIA one at compute capability 9.0+. The test below pins the *rejection*,
    so it is only meaningful where this is False.
    """
    if not torch.cuda.is_available():
        return False
    return torch.version.hip is not None or torch.cuda.get_device_capability() >= (9, 0)


@pytest.mark.skipif(_symm_mem_supported(), reason="needs a device below CC 9.0")
def test_symmetric_memory_is_rejected_off_a_supported_device() -> None:
    """On a machine without the capability there is no silent fallback to catch."""
    with pytest.raises(ValueError, match="compute capability 9.0"):
        ParallelConfig(fsdp_config=FSDPConfig(enable_fsdp_symm_mem=True))


def test_an_unknown_pipeline_schedule_is_rejected() -> None:
    """``get_schedule_class`` is the authority; the error names the bad value."""
    with pytest.raises(ValueError, match="Invalid parallel.pipeline_parallel_schedule"):
        ParallelConfig(
            pp_config=PipelineParallelConfig(pipeline_parallel_schedule="NotASchedule")
        )


# -- LRSchedulerConfig -------------------------------------------------------


def test_a_negative_warmup_is_rejected() -> None:
    with pytest.raises(ValueError, match="warmup_steps must be >= 0"):
        LRSchedulerConfig(warmup_steps=-1)
    assert LRSchedulerConfig(warmup_steps=0).warmup_steps == 0


def test_total_steps_if_given_must_be_positive() -> None:
    """``None`` means "take it from training"; an explicit value must be real."""
    assert LRSchedulerConfig(total_steps=None).total_steps is None
    with pytest.raises(ValueError, match="total_steps must be >= 1"):
        LRSchedulerConfig(total_steps=0)


@pytest.mark.parametrize("field", ["decay_ratio", "min_lr_factor"])
@pytest.mark.parametrize(
    "value", [True, False, None, "0.5", float("nan"), float("inf")]
)
def test_scheduler_fractions_reject_non_numeric_and_nonfinite_values(field, value):
    from llmtuner.errors import ConfigError

    with pytest.raises(ConfigError, match=f"{field} must be in"):
        LRSchedulerConfig(**{field: value})


def test_decay_ratio_is_a_fraction() -> None:
    with pytest.raises(ValueError, match="decay_ratio must be in"):
        LRSchedulerConfig(decay_ratio=1.5)
    with pytest.raises(ValueError, match="decay_ratio must be in"):
        LRSchedulerConfig(decay_ratio=-0.1)
    assert LRSchedulerConfig(decay_ratio=1.0).decay_ratio == 1.0


def test_min_lr_factor_is_half_open_at_one() -> None:
    """``1.0`` would mean "decay to the base lr", i.e. not decay at all."""
    assert LRSchedulerConfig(min_lr_factor=0.0).min_lr_factor == 0.0
    with pytest.raises(ValueError, match=r"min_lr_factor must be in \[0, 1\)"):
        LRSchedulerConfig(min_lr_factor=1.0)


# -- CheckpointConfig --------------------------------------------------------


def test_a_whitespace_folder_is_not_a_folder() -> None:
    with pytest.raises(ValueError, match="checkpoint.folder must not be empty"):
        CheckpointConfig(folder="   ")


def test_load_step_is_either_derive_or_non_negative() -> None:
    assert CheckpointConfig(load_step=-1).load_step == -1
    with pytest.raises(ValueError, match="load_step must be >= -1"):
        CheckpointConfig(load_step=-2)


def test_negative_retention_is_rejected() -> None:
    with pytest.raises(ValueError, match="keep_latest_k must be >= 0"):
        CheckpointConfig(keep_latest_k=-1)


@skip_without("dcp")
def test_the_model_can_never_be_excluded_from_a_load() -> None:
    """Loading everything *except* the weights is never what a user means."""
    from llmtuner.components.checkpointer import MODEL

    with pytest.raises(ValueError, match="exclude_from_loading must not contain"):
        CheckpointConfig(exclude_from_loading=[MODEL])


@skip_without("dcp")
def test_excluding_the_optimizer_must_exclude_the_schedule_too() -> None:
    """``LRSchedulersContainer`` reads ``base_lrs`` off the optimizers it restores.

    A schedule without its optimizers would restore against a cold optimizer and
    silently restart the lr curve. The pairing is enforced rather than inferred.
    """
    from llmtuner.components.checkpointer import LR_SCHEDULER, OPTIMIZER

    with pytest.raises(ValueError, match=f"{LR_SCHEDULER} must be excluded"):
        CheckpointConfig(exclude_from_loading=[OPTIMIZER])
    # The paired form is accepted.
    cfg = CheckpointConfig(exclude_from_loading=[OPTIMIZER, LR_SCHEDULER])
    assert set(cfg.exclude_from_loading) == {OPTIMIZER, LR_SCHEDULER}


def test_a_relative_initial_load_path_is_rejected() -> None:
    """Resuming from a relative path resolves against the launch dir, not the cwd."""
    with pytest.raises(ValueError, match="must be an absolute path or a remote URI"):
        CheckpointConfig(initial_load_path="./weights")
    assert (
        CheckpointConfig(initial_load_path="/abs/weights").initial_load_path
        == "/abs/weights"
    )


def test_hf_load_modes_imply_each_other() -> None:
    """Each ``*_in_hf`` flag needs a partner that is not on by default.

    The ``*_model_only`` partners default to ``True``, so the Implies fire only
    when a caller turns the partner *off* -- which is the mistake worth pinning.
    """
    with pytest.raises(ValueError, match="requires initial_load_model_only"):
        CheckpointConfig(initial_load_in_hf=True, initial_load_model_only=False)
    with pytest.raises(ValueError, match="requires initial_load_in_hf"):
        CheckpointConfig(initial_load_in_hf_quantized=True)
    with pytest.raises(ValueError, match="requires last_save_model_only"):
        CheckpointConfig(last_save_in_hf=True, last_save_model_only=False)
    # The pairing the defaults describe is accepted without spelling it out.
    assert CheckpointConfig(initial_load_in_hf=True).initial_load_model_only is True


def test_an_unknown_async_mode_is_rejected_and_the_valid_one_is_lowered() -> None:
    with pytest.raises(ValueError, match="checkpoint.async_mode must be one of"):
        CheckpointConfig(async_mode="Threaded")
    # The field is normalized in place, so a mixed-case spelling still works.
    assert CheckpointConfig(async_mode="ASYNC").async_mode == "async"


# -- TrainingConfig ----------------------------------------------------------


@pytest.mark.parametrize(
    "field, bad",
    [
        ("global_batch_size", 0),
        ("max_seq_len", 0),
        ("steps", 0),
        ("gradient_accumulation_steps", 0),
    ],
)
def test_a_non_positive_loop_parameter_is_rejected(field: str, bad: int) -> None:
    """Each of these divides or iterates; zero is a hang or a ZeroDivision."""
    with pytest.raises(ValueError, match=f"{field} must be >= 1"):
        TrainingConfig(**{field: bad})


def test_activation_checkpoint_messages_name_the_training_field() -> None:
    """The messages spell the dotted path, like every other guard here.

    This is the pair of ``parallel/test_activation_checkpoint.py``'s
    ``match="requires training.compile"``: a message that spells ``self.compile``
    silently fails that test on any machine where its module is not env-skipped.
    """
    with pytest.raises(ValueError, match="training.activation_checkpoint_mode"):
        TrainingConfig(activation_checkpoint_mode="bogus")
    with pytest.raises(ValueError, match="requires training.compile"):
        TrainingConfig(activation_checkpoint_mode="memory_budget")


# -- CompileConfig / MemoryBudgetACConfig ------------------------------------


@pytest.mark.parametrize("field", ["per_block", "enable_async_tensor_parallel"])
@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_compile_switches_require_boolean_values(field, value):
    with pytest.raises(ConfigError, match=field):
        CompileConfig(**{field: value})


def test_an_empty_compile_backend_is_rejected() -> None:
    """``torch.compile(backend="")`` fails inside inductor, far from the flag."""
    with pytest.raises(ValueError, match="compile.backend.*non-empty string"):
        CompileConfig(backend="")
    assert CompileConfig().backend == "inductor"


def test_memory_budget_is_a_fraction() -> None:
    """The bounds are the partitioner's, so both endpoints are legal."""
    assert MemoryBudgetACConfig(memory_budget=0.0).memory_budget == 0.0
    assert MemoryBudgetACConfig(memory_budget=1.0).memory_budget == 1.0
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="memory_budget must be between"):
            MemoryBudgetACConfig(memory_budget=bad)


# -- MetricsConfig / ProfilerConfig ------------------------------------------


def test_a_non_positive_log_freq_is_rejected() -> None:
    """A zero-length window would divide by zero on the first log."""
    with pytest.raises(ValueError, match=r"metrics.log_freq must be >= 1"):
        MetricsConfig(log_freq=0)
    assert MetricsConfig(log_freq=1).log_freq == 1


@pytest.mark.parametrize("value", [True, 1.5, float("nan"), "2", None])
def test_observability_counts_require_integers(value) -> None:
    with pytest.raises(ConfigError, match="log_freq"):
        MetricsConfig(log_freq=value)
    with pytest.raises(ConfigError, match="profile_freq"):
        ProfilerConfig(profile_freq=value)
    with pytest.raises(ConfigError, match="interval"):
        CheckpointConfig(interval=value)


@pytest.mark.parametrize("field", ["profiler_warmup", "profiler_active"])
@pytest.mark.parametrize("value", ["2", None, True, 1.5])
@pytest.mark.parametrize("enabled", [False, True])
def test_profiler_validates_cycle_fields_before_arithmetic(field, value, enabled):
    with pytest.raises(ConfigError, match=field):
        ProfilerConfig(enable_profiling=enabled, **{field: value})


def test_profiling_must_fit_one_cycle_into_the_interval() -> None:
    """A cycle is ``profiler_warmup + profiler_active``; a shorter interval
    would never reach the active iterations it exists to capture."""
    assert ProfilerConfig(enable_profiling=True, profile_freq=4).profile_freq == 4
    with pytest.raises(ValueError, match="profiler.profile_freq must be >="):
        ProfilerConfig(enable_profiling=True, profile_freq=3)
    # Off is off: the interval is not read, so a small value is not an error.
    assert ProfilerConfig(profile_freq=1).profile_freq == 1


# -- the flat view the trainer reads -------------------------------------------
#
# ``LLMTunerConfig`` exposes the trainer's scalars as hand-written properties,
# so a new field on a group stays invisible to the trainer until its passthrough
# exists. The failure mode is an AttributeError on the first training step, not
# at parse time -- so the passthroughs are worth pinning explicitly.


def test_accumulation_and_gc_freq_reach_the_flat_view() -> None:
    """The trainer reads both off ``cfg``, not off ``cfg.training``."""
    cfg = LLMTunerConfig(
        training=TrainingConfig(gradient_accumulation_steps=3, gc_freq=7)
    )
    assert cfg.gradient_accumulation_steps == 3
    assert cfg.gc_freq == 7
    # The defaults the trainer runs with when nothing is passed.
    assert LLMTunerConfig().gradient_accumulation_steps == 1
    assert LLMTunerConfig().gc_freq == 50


def test_cp_must_divide_seq_len() -> None:
    """A ragged sequence split would give ranks unequal token counts."""
    with pytest.raises(ValueError):
        LLMTunerConfig(
            parallel=ParallelConfig(context_parallel_kv_allgather_size=3),
            training=TrainingConfig(max_seq_len=64),
        )


def test_tp_must_divide_seq_len() -> None:
    """TP splits the CP shard one level down; same ragged-split refusal."""
    with pytest.raises(ValueError):
        LLMTunerConfig(
            parallel=ParallelConfig(tensor_parallel_size=3),
            training=TrainingConfig(max_seq_len=64),
        )


def test_cp_tp_joint_seq_len_divisibility() -> None:
    """Each divisor passing alone is not enough: the CP shard must divide by tp.

    T=6, cp=2, tp=2 passes both single-axis checks but leaves a length-3
    shard for tp=2; the joint check catches it at config time.
    """
    with pytest.raises(ValueError, match=r"cp \* tp"):
        LLMTunerConfig(
            parallel=ParallelConfig(
                context_parallel_kv_allgather_size=2, tensor_parallel_size=2
            ),
            training=TrainingConfig(max_seq_len=6),
        )
    # And the valid case still passes.
    LLMTunerConfig(
        parallel=ParallelConfig(
            context_parallel_kv_allgather_size=2, tensor_parallel_size=2
        ),
        training=TrainingConfig(max_seq_len=8),
    )


def test_model_config_validates_architecture_numbers() -> None:
    """Illegal architecture values must die in the config, not in for_model."""
    for overrides in (
        {"vocab_size": 0},
        {"hidden_size": 0},
        {"num_attention_heads": 0},
        {"num_key_value_heads": 0},
        {"num_hidden_layers": -1},
        {"hidden_size": 65},  # 65 % 4 != 0
        {"num_attention_heads": 4, "num_key_value_heads": 8},
        {"compute_dtype": "float8"},
        {"experts_implementation": "bogus"},
    ):
        with pytest.raises(ConfigError, match="model"):
            ModelConfig(**overrides)
    # The valid GQA corner: kv == q heads.
    ModelConfig(num_attention_heads=4, num_key_value_heads=4)


def test_optimizer_scalar_ranges() -> None:
    with pytest.raises(ConfigError, match="learning_rate"):
        OptimizerConfig(learning_rate=-1.0)
    with pytest.raises(ConfigError, match="weight_decay"):
        OptimizerConfig(weight_decay=-0.1)
    with pytest.raises(ConfigError, match="eps"):
        OptimizerConfig(eps=0.0)
    OptimizerConfig(learning_rate=0.0)  # legal: a frozen run is a choice


def test_num_pp_microbatches_and_gc_freq_have_floors() -> None:
    with pytest.raises(ConfigError, match="num_pp_microbatches"):
        ParallelConfig(pp_config=PipelineParallelConfig(num_pp_microbatches=0))
    with pytest.raises(ConfigError, match="gc_freq"):
        TrainingConfig(gc_freq=0)


def test_profiler_and_dataloader_scalars_have_floors() -> None:
    with pytest.raises(ConfigError, match="profiler_active"):
        ProfilerConfig(profiler_active=0)
    with pytest.raises(ConfigError, match="profiler_warmup"):
        ProfilerConfig(profiler_warmup=-1)
    ProfilerConfig(profiler_warmup=0)  # zero warmup is a legal choice
    with pytest.raises(ConfigError, match="memory_snapshot_max_entries"):
        ProfilerConfig(memory_snapshot_max_entries=0)
    with pytest.raises(ConfigError, match="memory_snapshot_freq"):
        ProfilerConfig(memory_snapshot_freq=1.5)
    with pytest.raises(ConfigError, match="memory_snapshot_freq"):
        ProfilerConfig(memory_snapshot_freq=0)
    with pytest.raises(ConfigError, match="streaming_shuffle_buffer_size"):
        DataloaderConfig(streaming_shuffle_buffer_size=0)
    with pytest.raises(ConfigError, match="num_prefetch_batches"):
        DataloaderConfig(num_prefetch_batches=0)
    with pytest.raises(ConfigError, match="max_num_documents"):
        DataloaderConfig(max_num_documents=1.5)
    with pytest.raises(ConfigError, match="chunked_loss_num_chunks"):
        TrainingConfig(chunked_loss_num_chunks=1.5)
    with pytest.raises(ConfigError, match="load_step"):
        CheckpointConfig(load_step="latest")


@pytest.mark.parametrize("max_documents", [None, 1, 2])
@pytest.mark.parametrize("dataset", [None, "random", "local_jsonl"])
def test_attention_mask_follows_effective_corpus_and_document_limit(
    max_documents, dataset
):
    config = DataloaderConfig(max_num_documents=max_documents)
    expected = (
        "causal"
        if dataset in (None, "random") or max_documents == 1
        else "block_causal"
    )
    assert config.attention_mask_type(dataset=dataset) == expected
    assert config.dataset == "random"


def test_ac_determinism_check_is_an_enum() -> None:
    for cls in (SelectiveACConfig, RegionACConfig):
        with pytest.raises(ConfigError, match="determinism_check"):
            cls(determinism_check="bogus")
        cls(determinism_check="none")


def test_selective_ac_rejects_unsupported_debug_tracing() -> None:
    with pytest.raises(ConfigError, match="selective_ac.debug.*context_fn"):
        SelectiveACConfig(debug=True)
    assert SelectiveACConfig(debug=False).debug is False


def test_from_groups_does_not_mutate_the_parsed_groups() -> None:
    """Grafting goes through ``replace``: the caller's group instances keep
    their own values (configs are read-only once built)."""
    optimizer = OptimizerConfig()
    training = TrainingConfig()
    cfg = LLMTunerConfig.from_groups(
        model=ModelConfig(),
        parallel=ParallelConfig(),
        optimizer=optimizer,
        lr_scheduler=LRSchedulerConfig(warmup_steps=11),
        training=training,
        checkpoint=CheckpointConfig(enable=True),
        dataloader=DataloaderConfig(),
        metrics=MetricsConfig(),
        profiler=ProfilerConfig(),
    )
    assert not training.checkpoint_config.enable  # the caller's copy is intact
    assert cfg.training.checkpoint_config.enable  # the graft took
    assert cfg.optimizer.lr_scheduler_config.warmup_steps == 11


def test_every_training_scalar_reaches_the_flat_view_or_is_nested() -> None:
    """A new TrainingConfig scalar without a flat-view property dies at the
    first train step with AttributeError; pin the contract instead."""
    import dataclasses

    nested = {
        "checkpoint_config",
        "dataloader_config",
        "metrics_config",
        "profiler_config",
        "ema_config",
        "validation_config",
        "compile_config",
        "selective_ac",
        "memory_budget_ac",
        "region_ac",
        "compile",
        "activation_checkpoint_mode",
        "chunked_loss_num_chunks",
    }
    props = {
        name
        for name, value in vars(LLMTunerConfig).items()
        if isinstance(value, property)
    }
    missing = [
        f.name
        for f in dataclasses.fields(TrainingConfig)
        if f.name not in nested and f.name not in props
    ]
    assert not missing, f"TrainingConfig fields missing a flat view: {missing}"


# -- the CLI view -------------------------------------------------------------


def test_the_fields_the_cli_cannot_carry_are_not_offered_as_flags() -> None:
    """A flag that rejects every value is worse than no flag.

    ``arch_overrides`` (a dict), ``param_groups`` (a list of nested
    dataclasses) and ``purge_exempt`` (a callable) are programmatic-only, and
    ``HfArgumentParser`` rejects any value given to them -- but it still lists
    them in ``--help``, looking usable. ``cli_view`` hides them by handing the
    parser an ``init=False`` view of each group; this pins that, and pins the
    flags that must *not* disappear with them.

    Matched as option lines, not substrings: the prose of neighbouring help
    strings legitimately mentions these names (e.g. "``purge_exempt=None``").
    """
    parser = HfArgumentParser([cli_view(group) for group in PARSER_GROUPS])
    help_text = parser.format_help()

    for hidden in (
        "arch_overrides",
        "param_groups",
        "purge_exempt",
        "fsdp_apply_modules",
        "module_fqns_per_model_part",
        "lr_scheduler_config",
        # Nested dataclasses collapse to single-value flags that reject every
        # value; they are grafted from their own parser groups instead.
        "ema_config",
        "validation_config",
        "checkpoint_config",
        "dataloader_config",
        "metrics_config",
        "profiler_config",
        "compile_config",
        "selective_ac",
        "memory_budget_ac",
        "region_ac",
    ):
        assert not re.search(rf"^\s+--{hidden}\b", help_text, re.M), hidden
    for kept in (
        "tensor_parallel_size",
        "learning_rate",
        "steps",
        "tp_colwise_modules",
        "fsdp_ignored_modules",
        "warmup_steps",
    ):
        assert re.search(rf"^\s+--{kept}\b", help_text, re.M), kept


def test_the_cli_view_parses_into_the_real_groups() -> None:
    """The views are drop-in: defaults, ``__post_init__`` and ``isinstance``.

    ``parse_args_into_dataclasses`` returns the subclasses ``cli_view``
    generates, so every passthrough the graft table keys by group still works --
    and the hidden fields keep the group's own defaults rather than becoming
    missing attributes.
    """
    parser = HfArgumentParser([cli_view(group) for group in PARSER_GROUPS])
    parsed = parser.parse_args_into_dataclasses(
        ["--vocab_size", "64", "--tensor_parallel_size", "2", "--learning_rate", "1e-3"]
    )
    by_group = dict(zip(PARSER_GROUPS, parsed, strict=True))

    for group, instance in by_group.items():
        assert isinstance(instance, group), group.__name__

    cfg = LLMTunerConfig.from_groups(
        model=by_group[ModelConfig],
        parallel=by_group[ParallelConfig],
        optimizer=by_group[OptimizerConfig],
        lr_scheduler=by_group[LRSchedulerConfig],
        training=by_group[TrainingConfig],
        checkpoint=by_group[CheckpointConfig],
        dataloader=by_group[DataloaderConfig],
        metrics=by_group[MetricsConfig],
        profiler=by_group[ProfilerConfig],
    )
    assert cfg.model.vocab_size == 64
    assert cfg.parallel.tensor_parallel_size == 2
    assert cfg.optimizer.learning_rate == 1e-3
    # The hidden fields are the group defaults, not absent.
    assert cfg.model.arch_overrides == {}
    assert cfg.checkpoint.purge_exempt is None
    # ``__post_init__`` still ran: no explicit param_groups means the
    # catch-all group the optimizer builds for a run.
    assert len(cfg.optimizer.param_groups) == 1


def test_cli_hidden_configuration_defaults_are_not_shared():
    from llmtuner.config.cli_config import cli_view

    optimizer_view = cli_view(OptimizerConfig)
    first, second = optimizer_view(), optimizer_view()
    first.lr_scheduler_config.warmup_steps = 3
    assert second.lr_scheduler_config.warmup_steps == 0
    parallel_view = cli_view(ParallelConfig)
    first_parallel, second_parallel = parallel_view(), parallel_view()
    first_parallel.fsdp_config.fsdp_apply_modules["layers.*"] = {}
    assert second_parallel.fsdp_config.fsdp_apply_modules == {}


def test_unknown_packing_is_rejected():
    with pytest.raises(ConfigError, match="dataloader.packing"):
        DataloaderConfig(packing="typo")


def test_model_rejects_nonuniform_gqa_head_groups():
    with pytest.raises(ConfigError, match="divisible by num_key_value_heads"):
        ModelConfig(hidden_size=96, num_attention_heads=6, num_key_value_heads=4)


@pytest.mark.parametrize("kv_heads", [1, 2, 3, 6])
def test_model_accepts_uniform_attention_head_groups(kv_heads):
    config = ModelConfig(
        hidden_size=96, num_attention_heads=6, num_key_value_heads=kv_heads
    )
    assert config.num_key_value_heads == kv_heads


@pytest.mark.parametrize("relative", [False, True])
def test_auto_fill_reads_local_hf_architecture(tmp_path, monkeypatch, relative):
    from transformers import AutoConfig

    directory = tmp_path / "saved_model"
    saved = AutoConfig.for_model(
        "llama",
        vocab_size=257,
        hidden_size=96,
        intermediate_size=192,
        num_hidden_layers=3,
        num_attention_heads=6,
        num_key_value_heads=2,
    )
    saved.save_pretrained(directory)
    monkeypatch.chdir(tmp_path)
    cfg = LLMTunerConfig(
        model=ModelConfig(
            model_name_or_path="saved_model" if relative else str(directory)
        )
    )
    cfg.auto_fill_model()
    for field in (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
    ):
        assert getattr(cfg.model, field) == getattr(saved, field)


@pytest.mark.parametrize(
    "identifier", ["llama", "/missing/model/path", "local-checkpoint", "org/model"]
)
def test_auto_fill_only_skips_registered_offline_architectures(monkeypatch, identifier):
    from transformers import AutoConfig

    calls = []

    def load_config(name):
        calls.append(name)
        return AutoConfig.for_model("llama", vocab_size=257)

    monkeypatch.setattr(AutoConfig, "from_pretrained", load_config)
    cfg = LLMTunerConfig(model=ModelConfig(model_name_or_path=identifier))
    cfg.auto_fill_model()
    assert calls == ([] if identifier == "llama" else [identifier])
    if calls:
        assert cfg.vocab_size == 257


def test_auto_fill_uses_composite_hf_text_architecture(tmp_path):
    from transformers import AutoConfig

    saved = AutoConfig.for_model(
        "qwen2_5_vl",
        text_config={
            "vocab_size": 257,
            "hidden_size": 96,
            "intermediate_size": 192,
            "num_hidden_layers": 3,
            "num_attention_heads": 6,
            "num_key_value_heads": 2,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
    )
    saved.save_pretrained(tmp_path)
    cfg = LLMTunerConfig(model=ModelConfig(model_name_or_path=str(tmp_path)))
    cfg.auto_fill_model()
    for field in (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
    ):
        assert getattr(cfg.model, field) == getattr(saved.text_config, field)


@pytest.mark.parametrize("field", ["freq", "steps"])
@pytest.mark.parametrize("value", [True, 1.5, "2", None])
def test_validation_config_rejects_non_integer_step_controls(field, value):
    from llmtuner.config.validation_config import ValidationConfig

    with pytest.raises(ConfigError, match=f"validation.{field}"):
        ValidationConfig(**{field: value})


@pytest.mark.parametrize(
    "field",
    [
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "num_pp_microbatches",
        "context_parallel_kcp_size",
        "context_parallel_ulysses_size",
        "context_parallel_kv_allgather_size",
        "expert_parallel_size",
    ],
)
@pytest.mark.parametrize("value", [True, 1.5, "2", None])
def test_parallel_integer_degrees_reject_non_integer_values(field, value):
    with pytest.raises(ConfigError, match=f"parallel.{field}"):
        (
            FSDPConfig(**{field: value})
            if field.startswith("fsdp_")
            else PipelineParallelConfig(**{field: value})
            if field == "num_pp_microbatches"
            else ParallelConfig(**{field: value})
        )


@pytest.mark.parametrize(
    ("config_type", "field", "group"),
    [
        (ModelConfig, "hidden_size", "model"),
        (TrainingConfig, "steps", "training"),
        (DataloaderConfig, "num_packing_bins", "dataloader"),
        (ProfilerConfig, "memory_snapshot_max_entries", "profiler"),
    ],
)
@pytest.mark.parametrize("value", [True, 1.5, float("nan"), "2", None])
def test_integer_field_validation_covers_config_groups(
    config_type, field, group, value
):
    with pytest.raises(ConfigError, match=rf"{group}\.{field}"):
        config_type(**{field: value})


@pytest.mark.parametrize("field", ["warmup_steps", "total_steps"])
@pytest.mark.parametrize("value", [True, 1.5, float("nan"), "2"])
def test_lr_schedule_counts_require_integers(field, value):
    with pytest.raises(ConfigError, match=field):
        LRSchedulerConfig(**{field: value})


@pytest.mark.parametrize("value", [True, 1.5, float("nan"), "2", None])
def test_ema_update_interval_requires_integer(value):
    from llmtuner.config.ema_config import EMAConfig

    with pytest.raises(ConfigError, match="update_every_n_steps"):
        EMAConfig(update_every_n_steps=value)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), True, "1", None]
)
def test_training_rejects_invalid_gradient_clip_threshold(value):
    with pytest.raises(ConfigError, match="max_norm"):
        TrainingConfig(max_norm=value)


@pytest.mark.parametrize("value", [-1, 0, 1.5])
def test_training_accepts_finite_gradient_clip_threshold(value):
    assert TrainingConfig(max_norm=value).max_norm == value
