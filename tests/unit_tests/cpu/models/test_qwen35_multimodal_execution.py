"""Multimodal hidden-output contracts independent of vision and attention math."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from llmtuner.models.qwen.qwen3_5.multimodal import Qwen3_5ForConditionalGeneration
from llmtuner.models.qwen.qwen3_5_moe.multimodal import (
    Qwen3_5MoeForConditionalGeneration,
)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Parameter(torch.ones(1, 4, 8))
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return BaseModelOutputWithPast(last_hidden_state=self.hidden)


class ForbiddenHead(nn.Module):
    def forward(self, *args, **kwargs):
        pytest.fail("Hidden output must not execute the LM head")


@pytest.fixture(
    params=[Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration]
)
def model(request):
    # Only exercise the output adapter; no HF model construction or kernels.
    result = request.param.__new__(request.param)
    nn.Module.__init__(result)
    result.config = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=8, vocab_size=16), return_dict=False
    )
    result.model = Decoder()
    result.lm_head = ForbiddenHead()
    result.cp_group = None
    return result


def test_hidden_output_preserves_rows_media_and_gradients(model):
    tokens = torch.ones(1, 4, dtype=torch.long)
    pixels = torch.ones(3, 12)
    output = model(
        tokens,
        pixel_values=pixels,
        labels=tokens,
        logits_to_keep=1,
        return_dict=False,
        skip_lm_head=True,
    )
    assert output is model.model.hidden
    assert output.shape == (1, 4, model.execution_capabilities.hidden_size)
    assert model.vocab_size == 16
    call = model.model.calls[0]
    assert call["pixel_values"] is pixels
    assert call["use_cache"] is False and call["return_dict"] is True
    output.sum().backward()
    torch.testing.assert_close(model.model.hidden.grad, torch.ones_like(output))


@pytest.mark.parametrize(
    "cache", [{"use_cache": True}, {"past_key_values": object()}, {"cache_position": 0}]
)
def test_hidden_output_rejects_cache_before_decoder_execution(model, cache):
    with pytest.raises(ValueError, match="does not support KV cache"):
        model(torch.ones(1, 4, dtype=torch.long), skip_lm_head=True, **cache)
    assert model.model.calls == []


@pytest.mark.parametrize("field", ["image_token_id", "spatial_merge_size"])
def test_multimodal_checkpoint_rejects_media_geometry_changes(model, field):
    from llmtuner.components.checkpointer.core.format_state import CheckpointFormatState

    model.config.vision_config = SimpleNamespace(spatial_merge_size=2)
    model.config.image_token_id = 3
    saved = CheckpointFormatState.capture([model]).state_dict()
    if field == "image_token_id":
        model.config.image_token_id = 4
    else:
        model.config.vision_config.spatial_merge_size = 4
    target = CheckpointFormatState.capture([model])
    with pytest.raises(ValueError, match="model execution contracts differ"):
        target.validate_state_dict(saved)
