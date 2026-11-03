"""Qwen3-30B: DistMuon attention updates, checkpoint and continued training."""

import os
import time
from collections import defaultdict

import torch

from examples.train_qwen3_30b_npu import qwen3_30b_npu_config
from llmtuner.components.optimizer.flex_shard import (
    BucketConfig,
    ComputeLayout,
    Owned,
    optimizer_reshard_runtime,
)
from llmtuner.components.optimizer.flex_shard.dist_muon import DistMuon
from llmtuner.config import OptimizerConfig, ParamGroupConfig
from llmtuner.trainer.runtime import assembly
from llmtuner.trainer.trainer import Trainer
from llmtuner.utils.module_names import canonical_fqn

PATTERN = r".*\.(q_proj|k_proj|v_proj|o_proj)\.weight$"


def install_profile():
    events = defaultdict(list)
    original_step = DistMuon.step
    original_compute = DistMuon._compute_update
    original_all_to_all = optimizer_reshard_runtime._execute_packed_all_to_all

    def timed(name, function, *args, **kwargs):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        result = function(*args, **kwargs)
        end.record()
        events[name].append((start, end))
        return result

    def compute(self, layout, tensor):
        return timed("newton_schulz", original_compute, self, layout, tensor)

    def all_to_all(*args, **kwargs):
        return timed("redistribution", original_all_to_all, *args, **kwargs)

    def step(self, closure=None):
        torch.npu.synchronize()
        before = torch.npu.memory_allocated()
        result = original_step(self, closure)
        torch.npu.synchronize()
        after = torch.npu.memory_allocated()
        peak = torch.npu.max_memory_allocated()
        print(
            f"DistMuon optimizer memory: before={before} after={after} peak={peak}",
            flush=True,
        )
        return result

    DistMuon.step = step
    DistMuon._compute_update = compute
    optimizer_reshard_runtime._execute_packed_all_to_all = all_to_all

    def restore():
        DistMuon.step = original_step
        DistMuon._compute_update = original_compute
        optimizer_reshard_runtime._execute_packed_all_to_all = original_all_to_all

    return events, restore


def print_profile(events, trainer):
    torch.npu.synchronize()
    for name, pairs in events.items():
        milliseconds = sum(start.elapsed_time(end) for start, end in pairs)
        print(f"DistMuon profile {name}: {milliseconds / 1000:.3f}s", flush=True)
    for optimizer in trainer.optimizer.optimizers:
        if not isinstance(optimizer, DistMuon):
            continue
        context = optimizer._redistribution_runtime._context
        buffers = [] if context is None else [slot.buffers for slot in context.slots]
        buffers.append(optimizer._redistribution_runtime._local_slot)
        nbytes = sum(
            tensor.numel() * tensor.element_size()
            for slot in buffers
            for reserved in slot.buffers.values()
            for tensor in (
                reserved.storage_exchange,
                reserved.compute_exchange,
                reserved.compute_scratch,
                reserved.storage_scratch,
            )
            if isinstance(tensor, torch.Tensor)
        )
        print(f"DistMuon reserved buffers: {nbytes} bytes/rank", flush=True)


def main():
    config = qwen3_30b_npu_config()
    if slots := int(os.environ.get("LLMTUNER_DIST_MUON_SLOTS", "0")):
        if slots < 1:
            raise ValueError("LLMTUNER_DIST_MUON_SLOTS must be positive")
        optimizer_reshard_runtime._NUM_PIPELINE_SLOTS = slots
    original = assembly._build_optimizer_stack
    counts = {}

    def build_optimizer_stack(trainer):
        layouts = {}
        for part in trainer.model_parts:
            for name, _parameter in part.named_parameters():
                if name.endswith(
                    ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight")
                ):
                    layouts[canonical_fqn(name)] = ComputeLayout(
                        {"dp_shard": Owned()}, compute_dtype=torch.bfloat16
                    )
        assert layouts
        counts["dist_muon_parameters"] = len(layouts)
        print(f"DistMuon layout examples: {list(layouts)[:2]}", flush=True)
        trainer.cfg.optimizer = OptimizerConfig(
            learning_rate=1e-5,
            weight_decay=0.1,
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=PATTERN,
                    optimizer_name="DistMuon",
                    optimizer_kwargs={"lr": 1e-5, "ns_device": "accelerator"},
                ),
                ParamGroupConfig(
                    pattern=".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 1e-5, "weight_decay": 0.1},
                ),
            ],
            compute_sharding_by_fqn=layouts,
            bucket_configs=[
                BucketConfig(
                    patterns=tuple(
                        f"model.model.layers.{index}.self_attn.{projection}.weight"
                        for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
                    ),
                    name=f"attention-layer-{index}",
                )
                for index in range(48)
            ],
        )
        original(trainer)

    profile = None
    if os.environ.get("LLMTUNER_DIST_MUON_PROFILE") == "1":
        profile = install_profile()
    assembly._build_optimizer_stack = build_optimizer_stack
    try:
        trainer = Trainer(config)
    except BaseException:
        if profile is not None:
            profile[1]()
        raise
    finally:
        assembly._build_optimizer_stack = original
    started = time.perf_counter()
    try:
        parameter_dtypes = {
            parameter.dtype
            for part in trainer.model_parts
            for parameter in part.parameters()
        }
        assert parameter_dtypes == {torch.bfloat16}, parameter_dtypes
        print(f"30B parameter dtypes: {parameter_dtypes}", flush=True)
        trainer.train()
        assert trainer.step == config.training.steps
        if profile is not None:
            print_profile(profile[0], trainer)
        for part in trainer.model_parts:
            for parameter in part.parameters():
                assert torch.isfinite(parameter).all().item()
        assert any(
            optimizer.__class__.__name__ == "DistMuon"
            for optimizer in trainer.optimizer.optimizers
        )
        print(
            f"PASS 30B DistMuon step={trainer.step} "
            f"parameters={counts['dist_muon_parameters']} "
            f"seconds={time.perf_counter() - started:.3f}",
            flush=True,
        )
    finally:
        if profile is not None:
            profile[1]()
        trainer.close()


if __name__ == "__main__":
    if os.environ.get("LLMTUNER_EP_TOKEN_DISPATCHER", "alltoall") != "alltoall":
        raise ValueError("DistMuon 30B validation uses the AllToAll baseline")
    # Match the production example rather than inheriting FP32 construction
    # from Python's default dtype. Muon still uses its explicit compute layout.
    torch.set_default_dtype(torch.bfloat16)
    main()
