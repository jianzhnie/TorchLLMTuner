"""PP2 × CP2 training against an independent dense model on CPU/Gloo.

Run with four ranks, optionally selecting a schedule and accumulation count:

    PYTHONPATH=. torchrun --nproc_per_node=4 \
        tests/integration_tests/pp_cp_equivalence.py 1F1B 2

Real Trainer, CP sharding/masks, KV collective/autograd and FSDP run unchanged.
Only attention computation uses dense-mask SDPA; production flex kernels and
target-device behavior are not validated. FSDP defaults to custom; set
LLMTUNER_TEST_FSDP_IMPLEMENTATION=native to exercise the native implementation.
For headtail numerical diagnosis, set LLMTUNER_TEST_CP_LOAD_BALANCER=headtail
and LLMTUNER_TEST_CP_FP64_DIAGNOSTIC=1. That explicitly substitutes FP64 model
and CE arithmetic, retaining FP32 scalar loss communication; it does not
validate the production FP32 trajectory.
LLMTUNER_TEST_CP_REORDER_REFERENCE=1 independently permutes dense reference
KV and mask columns into headtail order, isolating floating-point reduction
order. The original unpermuted reference remains the default.
"""

import os
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_mask

from llmtuner.accelerator.device import device_type
from llmtuner.models.hf.attention_backend import ATTENTION_BACKEND
from llmtuner.models.hf.model import HFTransformerModel


def _dense_attention(module, q, k, v, block_mask, kwargs):
    """Expand the actual sharded mask predicate at local-Q/full-KV lengths."""
    mask = create_mask(
        block_mask.mask_mod,
        B=1,
        H=1,
        Q_LEN=q.shape[2],
        KV_LEN=k.shape[2],
        device=str(q.device),
    )
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, scale=kwargs.get("scaling"), enable_gqa=True
    )


def _dense_adapter(module, q, k, v, mask, **kwargs):
    if os.environ.get("LLMTUNER_TEST_CP_REORDER_REFERENCE") == "1":
        # Independent arithmetic: pair chunk r with chunk 2*cp-1-r.
        # The dense oracle keeps original query/output order; only the KV
        # reduction order changes, with the same mask columns permuted.
        cp = int(os.environ["LLMTUNER_TEST_CP_SIZE"])
        chunks = torch.arange(k.shape[2], device=k.device).reshape(2 * cp, -1)
        order = torch.cat(
            [torch.cat((chunks[r], chunks[2 * cp - 1 - r])) for r in range(cp)]
        )
        dense_mask = create_mask(
            mask.mask_mod,
            B=1,
            H=1,
            Q_LEN=q.shape[2],
            KV_LEN=k.shape[2],
            device=str(q.device),
        ).index_select(-1, order)
        output = F.scaled_dot_product_attention(
            q,
            k.index_select(2, order),
            v.index_select(2, order),
            attn_mask=dense_mask,
            scale=kwargs.get("scaling"),
            enable_gqa=True,
        )
        return output.transpose(1, 2), None
    return _dense_attention(module, q, k, v, mask, kwargs).transpose(1, 2), None


def _validate_reordered_reference():
    """Check permutation invariance independently in FP64, including GQA grads."""
    generator = torch.Generator().manual_seed(123)
    mask = SimpleNamespace(mask_mod=lambda b, h, q, k: q >= k)
    for kv_heads in (4, 2):
        inputs = [
            torch.randn(
                1, heads, 16, 8, generator=generator, dtype=torch.float64
            ).requires_grad_()
            for heads in (4, kv_heads, kv_heads)
        ]
        expected = _dense_attention(None, *inputs, mask, {})
        actual = _dense_adapter(None, *inputs, mask)[0].transpose(1, 2)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        actual_grads = torch.autograd.grad(actual.square().sum(), inputs)
        expected_grads = torch.autograd.grad(expected.square().sum(), inputs)
        for got, want in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)


@contextmanager
def cpu_cp_attention():
    """Use one explicit CPU attention substitute across PP/CP harnesses."""
    if device_type != "cpu":
        raise RuntimeError("This explicit SDPA substitute requires CPU/Gloo")
    with (
        patch(
            "llmtuner.models.hf.model.available_attention_backend",
            return_value=ATTENTION_BACKEND,
        ),
        patch(
            "llmtuner.parallel.context_parallel.attention.run_flex", _dense_attention
        ),
        patch(
            "llmtuner.models.hf.attention_backend.flex_attention_forward",
            _dense_adapter,
        ),
    ):
        yield


if __name__ == "__main__":
    if device_type != "cpu":
        raise RuntimeError("This explicit SDPA substitute requires CPU/Gloo")
    # Set scenario defaults before importing the shared harness's constants.
    os.environ.setdefault("LLMTUNER_TEST_WORLD_SIZE", "4")
    os.environ.setdefault("LLMTUNER_TEST_CP_SIZE", "2")
    os.environ.setdefault("LLMTUNER_TEST_SEQ_LEN", "256")
    os.environ.setdefault("LLMTUNER_TEST_FSDP_IMPLEMENTATION", "custom")
    if os.environ.get("LLMTUNER_TEST_CP_REORDER_REFERENCE") == "1" and (
        os.environ.get("LLMTUNER_TEST_CP_LOAD_BALANCER") != "headtail"
    ):
        raise ValueError("The reordered reference requires headtail load balancing")
    if os.environ.get("LLMTUNER_TEST_CP_REORDER_REFERENCE") == "1":
        _validate_reordered_reference()
    from tests.integration_tests.pp_equivalence import main

    with ExitStack() as diagnostics, cpu_cp_attention():
        if os.environ.get("LLMTUNER_TEST_CP_FP64_DIAGNOSTIC") == "1":
            original_init = HFTransformerModel.__init__

            def initialize_double(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                self.double()

            def double_loss(pred, labels, **kwargs):
                # Scalar loss communication keeps Trainer's FP32 sentinel
                # contract; backward still evaluates CE on FP64 logits.
                return F.cross_entropy(
                    pred.reshape(-1, pred.shape[-1]),
                    labels.reshape(-1),
                    reduction="sum",
                    ignore_index=-100,
                ).float()

            diagnostics.enter_context(
                patch.object(HFTransformerModel, "__init__", initialize_double)
            )
            diagnostics.enter_context(
                patch(
                    "llmtuner.parallel.pipeline_parallel.pipeline_parallel."
                    "cross_entropy_loss",
                    double_loss,
                )
            )
            diagnostics.enter_context(
                patch(
                    "tests.integration_tests.pp_equivalence.cross_entropy_loss",
                    double_loss,
                )
            )
        main()
