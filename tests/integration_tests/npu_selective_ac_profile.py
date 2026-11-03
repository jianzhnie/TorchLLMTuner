"""Record non-vacuous selective-AC policy decisions on Qwen3."""

from collections import Counter

import torch

from examples.train_qwen3_30b_npu import qwen3_30b_npu_config
from llmtuner.memory import activation_checkpoint
from llmtuner.trainer.trainer import Trainer


def main() -> None:
    config = qwen3_30b_npu_config()
    config.training.activation_checkpoint_mode = "selective"
    counts = Counter()
    original = activation_checkpoint.selective_policy

    def counted_policy(*args, **kwargs):
        policy = original(*args, **kwargs)

        def wrapped(ctx, func, *op_args, **op_kwargs):
            decision = policy(ctx, func, *op_args, **op_kwargs)
            phase = "recompute" if ctx.is_recompute else "forward"
            counts[(phase, str(func), decision.name)] += 1
            return decision

        return wrapped

    activation_checkpoint.selective_policy = counted_policy
    try:
        torch.set_default_dtype(torch.bfloat16)
        Trainer(config).train()
    finally:
        activation_checkpoint.selective_policy = original
    assert any(key[2] == "MUST_SAVE" for key in counts)
    assert any(key[2] == "PREFER_RECOMPUTE" for key in counts)
    for (phase, op, decision), count in counts.most_common(40):
        print(f"SelectiveAC {phase} {decision} {op}: {count}", flush=True)


if __name__ == "__main__":
    main()
