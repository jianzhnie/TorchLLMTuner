"""CPU/Gloo ranks export tied Llama weights for standard HF loading.

Run with torchrun --nproc_per_node=2. Uses the production pattern FSDP entry,
native/custom engines and one/two safetensors files. Config assets are supplied
by the caller; this does not test standalone config/tokenizer export.
For TP2 x FSDP2 use four ranks and append --tp-size 2.
"""

import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from llmtuner.components.checkpointer.distributed_checkpoint import (
    DistributedCheckpointManager,
)
from llmtuner.config.checkpoint_config import CheckpointConfig
from llmtuner.config.parallel_config import FSDPConfig, ParallelConfig
from llmtuner.models.hf.state_dict_adapter import HFTransformerStateDictAdapter
from llmtuner.parallel.fully_shard_parallel.fully_shard_parallel import apply_fsdp
from llmtuner.parallel.tensor_parallel.tensor_parallel import apply_tp
from llmtuner.parallel.topology import ParallelTopology


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    dist.init_process_group("gloo", timeout=timedelta(seconds=40))
    try:
        assert dist.get_world_size() == 2 * args.tp_size
        topology = ParallelTopology(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=args.tp_size,
            pp=1,
            ep=1,
            world_size=dist.get_world_size(),
        )
        topology.build_device_mesh()
        context = TemporaryDirectory() if dist.get_rank() == 0 else nullcontext(None)
        with context as root:
            paths = [root]
            dist.broadcast_object_list(paths, src=0)
            for engine in ("native", "custom"):
                for indexed in (False, True):
                    torch.manual_seed(184)
                    config = LlamaConfig(
                        vocab_size=16,
                        hidden_size=16,
                        intermediate_size=32,
                        num_hidden_layers=1,
                        num_attention_heads=4,
                        num_key_value_heads=2,
                        tie_word_embeddings=True,
                        attention_dropout=0.0,
                    )
                    config._attn_implementation = "eager"
                    model = LlamaForCausalLM(config).float().eval()
                    generation = deepcopy(model.generation_config)
                    inputs = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
                    with torch.no_grad():
                        expected_logits = model(inputs).logits.clone()
                    expected_state = {
                        name: value.clone()
                        for name, value in model.state_dict().items()
                    }
                    parallel = ParallelConfig(
                        tensor_parallel_size=args.tp_size,
                        fsdp_config=FSDPConfig(
                            fsdp_implementation=engine, fsdp_apply_modules={"": {}}
                        ),
                    )
                    if args.tp_size > 1:
                        apply_tp(model, topology.get_mesh("tp"), parallel)
                        assert model._tp_checkpoint_layouts
                        for name, (
                            dim,
                            packed,
                            group,
                        ) in model._tp_checkpoint_layouts.items():
                            assert not packed
                            chunks = expected_state[name].chunk(args.tp_size, dim=dim)
                            torch.testing.assert_close(
                                model.get_parameter(name),
                                chunks[dist.get_rank(group)],
                                rtol=0,
                                atol=0,
                            )
                    apply_fsdp(model, parallel, topology)
                    assert all(
                        isinstance(param, DTensor) for param in model.parameters()
                    )
                    assert all(
                        param.to_local().numel() * 2 == param.numel()
                        for param in model.parameters()
                    )
                    assert model.lm_head.weight is model.model.embed_tokens.weight
                    wrapper = torch.nn.ModuleDict({"model": model})
                    optimizer = torch.optim.AdamW(wrapper.parameters())
                    adapter = HFTransformerStateDictAdapter(config, None)
                    if indexed:
                        # Source-index coverage is separate from output shard
                        # placement. Exercise the real global key gather too.
                        adapter._hf_keys = set(expected_state)
                        adapter.fqn_to_index_mapping = {
                            name: index % 2 + 1
                            for index, name in enumerate(expected_state)
                        }
                    folder = f"{paths[0]}/{engine}-{indexed}"
                    manager = DistributedCheckpointManager(
                        CheckpointConfig(
                            enable=True, last_save_in_hf=True, keep_latest_k=0
                        ),
                        model_parts=[wrapper],
                        optimizer=optimizer,
                        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(
                            optimizer, lambda _: 1
                        ),
                        states={},
                        folder=folder,
                        sd_adapter=adapter,
                    )
                    try:
                        assert manager.save(1, last_step=True)
                        assert model.lm_head.weight is model.model.embed_tokens.weight
                        restored = AutoModelForCausalLM.from_pretrained(
                            f"{folder}/checkpoint/step-1",
                            config=deepcopy(config),
                            generation_config=generation,
                            local_files_only=True,
                            dtype=torch.float32,
                        ).eval()
                        torch.testing.assert_close(
                            restored.state_dict(), expected_state, rtol=0, atol=0
                        )
                        assert (
                            restored.lm_head.weight
                            is restored.model.embed_tokens.weight
                        )
                        with torch.no_grad():
                            actual_logits = restored(inputs).logits
                        torch.testing.assert_close(
                            actual_logits, expected_logits, rtol=0, atol=0
                        )
                        dist.barrier()
                        if dist.get_rank() == 0:
                            print(
                                f"{engine} indexed={indexed}: "
                                f"tp={args.tp_size}: tied HF state/logits exact",
                                flush=True,
                            )
                    finally:
                        manager.close()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
