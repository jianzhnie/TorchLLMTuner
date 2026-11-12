"""Explicit Qwen3 compilation preserves training without the removed adapter."""

from copy import deepcopy

import torch
from torch._dynamo.utils import counters

from llmtuner.config.compile_config import CompileConfig
from llmtuner.models.qwen.qwen3.model import Qwen3MoeForCausalLM
from llmtuner.parallel.model_compile import apply_compile
from tests.unit_tests.cpu.models.test_qwen3_model_contract import model_config


def test_explicit_qwen3_aot_eager_matches_multistep_training():
    config = model_config()
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    torch.manual_seed(73)
    reference = Qwen3MoeForCausalLM(config)
    actual = deepcopy(reference)
    identities = {name: parameter for name, parameter in actual.named_parameters()}
    state_keys = tuple(actual.state_dict())
    optimizers = [
        torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=False)
        for model in (reference, actual)
    ]
    generator = torch.Generator().manual_seed(71)
    torch._dynamo.reset()
    counters.clear()
    try:
        compiled = apply_compile(
            actual, compile_config=CompileConfig(backend="aot_eager")
        )
        for _ in range(3):
            tokens = torch.randint(config.vocab_size, (2, 5), generator=generator)
            targets = torch.randint(config.vocab_size, (2, 5), generator=generator)
            outputs = []
            for model, optimizer in zip((reference, compiled), optimizers, strict=True):
                optimizer.zero_grad(set_to_none=True)
                logits = model(tokens, use_cache=False).logits
                outputs.append(logits)
                loss = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1), targets.flatten()
                )
                loss.backward()
            torch.testing.assert_close(outputs[1], outputs[0], atol=1e-6, rtol=1e-5)
            nonzero = 0
            for name, parameter in actual.named_parameters():
                baseline = dict(reference.named_parameters())[name]
                assert parameter is identities[name]
                torch.testing.assert_close(parameter.grad, baseline.grad)
                if parameter.grad is not None:
                    nonzero += int(bool(parameter.grad.count_nonzero()))
            assert nonzero > 0
            for optimizer in optimizers:
                optimizer.step()
            for name, parameter in actual.named_parameters():
                torch.testing.assert_close(
                    parameter, dict(reference.named_parameters())[name]
                )
            assert tuple(actual.state_dict()) == state_keys
        assert counters["stats"]["unique_graphs"] > 0
    finally:
        torch._dynamo.reset()
