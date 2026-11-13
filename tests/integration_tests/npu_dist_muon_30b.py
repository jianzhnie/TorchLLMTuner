"""Qwen3-30B DistMuon attention-update smoke test.

Checkpoint settings follow the example; this script does not assert recovery.
"""

import importlib
import os
import time
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from llmtuner.components.optimizer.flex_shard import (
    BucketConfig,
    ComputeLayout,
    Owned,
)
from llmtuner.components.optimizer.flex_shard.dist_muon import DistMuon
from llmtuner.config import OptimizerConfig, ParamGroupConfig
from llmtuner.trainer.runtime import assembly
from llmtuner.trainer.trainer import Trainer
from llmtuner.utils.module_names import canonical_fqn

PATTERN = r".*\.(q_proj|k_proj|v_proj|o_proj)\.weight$"


def install_profile():
    from llmtuner.components.optimizer.flex_shard import _optimizer_reshard_runtime

    events = defaultdict(list)
    original_step = DistMuon.step
    original_compute = DistMuon._compute_update
    original_all_to_all = _optimizer_reshard_runtime._execute_packed_all_to_all

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
    _optimizer_reshard_runtime._execute_packed_all_to_all = all_to_all

    def restore():
        DistMuon.step = original_step
        DistMuon._compute_update = original_compute
        _optimizer_reshard_runtime._execute_packed_all_to_all = original_all_to_all

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
    from llmtuner.components.optimizer.flex_shard import _optimizer_reshard_runtime

    recipe = importlib.import_module("examples.train_qwen3_30b_npu")
    config = recipe.qwen3_30b_npu_config()
    if slots := int(os.environ.get("LLMTUNER_DIST_MUON_SLOTS", "0")):
        if slots < 1:
            raise ValueError("LLMTUNER_DIST_MUON_SLOTS must be positive")
        _optimizer_reshard_runtime._NUM_PIPELINE_SLOTS = slots
    original = assembly.build_optimizer_stack
    counts = {}

    def build_optimizer_stack(cfg, *, model_parts, parallel_dims, error_group=None):
        layouts = {}
        for part in model_parts:
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
        cfg.optimizer = OptimizerConfig(
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
        return original(
            cfg,
            model_parts=model_parts,
            parallel_dims=parallel_dims,
            error_group=error_group,
        )

    profile = None
    if os.environ.get("LLMTUNER_DIST_MUON_PROFILE") == "1":
        profile = install_profile()
    assembly.build_optimizer_stack = build_optimizer_stack
    try:
        trainer = Trainer(config)
    except BaseException:
        if profile is not None:
            profile[1]()
        raise
    finally:
        assembly.build_optimizer_stack = original
    started = time.perf_counter()
    try:
        parameter_dtypes = {
            parameter.dtype
            for part in trainer.model_parts
            for parameter in part.parameters()
        }
        assert parameter_dtypes == {torch.bfloat16}, parameter_dtypes
        print(f"30B parameter dtypes: {parameter_dtypes}", flush=True)
        observed = {}
        for part in trainer.model_parts:
            for name, parameter in part.named_parameters():
                name = canonical_fqn(name)
                if ".layers.0.self_attn." in name and name.endswith(
                    ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight")
                ):
                    local = (
                        parameter.to_local()
                        if isinstance(parameter, DTensor)
                        else parameter
                    )
                    observed[name.rsplit(".", 2)[-2]] = (
                        parameter,
                        local.detach().cpu().clone(),
                    )
        assert set(observed) == {"q_proj", "k_proj", "v_proj", "o_proj"}, sorted(
            observed
        )
        trainer.train()
        assert trainer.step == config.training.steps
        if profile is not None:
            print_profile(profile[0], trainer)
        for part in trainer.model_parts:
            for parameter in part.parameters():
                assert torch.isfinite(parameter).all().item()
        muons = [
            optimizer
            for optimizer in trainer.optimizer.optimizers
            if isinstance(optimizer, DistMuon)
        ]
        assert muons
        changed = []
        for name, (parameter, initial) in sorted(observed.items()):
            local = (
                parameter.to_local() if isinstance(parameter, DTensor) else parameter
            )
            changed.append(int(not torch.equal(initial, local.detach().cpu())))
            owners = [optimizer for optimizer in muons if parameter in optimizer.state]
            assert len(owners) == 1, name
            momentum = owners[0].state[parameter]["momentum_buffer"]
            if isinstance(momentum, DTensor):
                momentum = momentum.to_local()
            assert torch.isfinite(momentum).all().item(), name
            assert momentum.count_nonzero().item() > 0, name
        changed = torch.tensor(changed, device=f"npu:{torch.npu.current_device()}")
        dist.all_reduce(changed)
        assert (changed > 0).all().item(), changed
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
