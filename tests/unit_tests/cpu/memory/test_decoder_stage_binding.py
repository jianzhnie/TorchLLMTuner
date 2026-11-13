"""Memory and compile stages bind declared PP-local blocks, not aliases."""

import pytest
import torch.nn as nn

from llmtuner.config.compile_config import CompileConfig
from llmtuner.config.parallel_config import ParallelConfig
from llmtuner.memory.activation_checkpoint import apply_ac
from llmtuner.models.contracts import DecoderStructure
from llmtuner.parallel.model_compile import apply_compile
from llmtuner.parallel.parallelize import parallelize_model


@pytest.fixture
def local_model():
    model = nn.Module()
    model.decoder = nn.Module()
    model.decoder.blocks = nn.ModuleList([nn.Linear(2, 2)])
    model.layers = nn.ModuleList([nn.Linear(2, 2)])
    model.decoder_structure = DecoderStructure(
        decoder="decoder",
        embedding="decoder.embedding",
        blocks="decoder.blocks",
        norm="decoder.norm",
        head="head",
    )
    return model


def test_ac_binds_declared_blocks_and_preserves_parameters(local_model):
    original = local_model.decoder.blocks[0]
    unrelated = local_model.layers[0]
    parameters = tuple(original.parameters())

    apply_ac(local_model, mode="full")

    assert local_model.decoder.blocks[0] is not original
    assert local_model.layers[0] is unrelated
    assert all(
        actual is expected
        for actual, expected in zip(
            local_model.decoder.blocks[0].parameters(), parameters, strict=True
        )
    )


def test_compile_binds_declared_blocks_without_replacing_them(local_model, monkeypatch):
    calls = []
    monkeypatch.setattr(
        nn.Module, "compile", lambda block, **kwargs: calls.append(block)
    )
    original = local_model.decoder.blocks[0]

    apply_compile(local_model, compile_config=CompileConfig(per_block=True))

    assert calls == [original]
    assert local_model.decoder.blocks[0] is original


@pytest.mark.parametrize("stage", ["ac", "compile"])
def test_stages_reject_invalid_declarations_without_fallback(local_model, stage):
    local_model.decoder_structure = "decoder.blocks"
    original = local_model.decoder.blocks[0]
    with pytest.raises(TypeError, match="DecoderStructure"):
        if stage == "ac":
            apply_ac(local_model, mode="full")
        else:
            apply_compile(local_model, compile_config=CompileConfig(per_block=True))
    assert local_model.decoder.blocks[0] is original


@pytest.mark.parametrize("stage", ["ac", "compile"])
def test_assembly_rejects_bad_targets_before_transforms(
    local_model, stage, monkeypatch
):
    local_model.decoder_structure = "decoder.blocks"
    original = local_model.decoder.blocks[0]
    reached = []
    monkeypatch.setattr(
        "llmtuner.parallel.parallelize.validate_fsdp_environment",
        lambda *args: reached.append("fsdp validation"),
    )
    options = (
        {"activation_checkpoint": "full"}
        if stage == "ac"
        else {"compile": True, "compile_config": CompileConfig(per_block=True)}
    )

    with pytest.raises(TypeError, match="DecoderStructure"):
        parallelize_model(
            local_model,
            cfg=ParallelConfig(),
            mesh=None,
            parallel_dims=None,
            **options,
        )

    assert not reached
    assert local_model.decoder.blocks[0] is original
