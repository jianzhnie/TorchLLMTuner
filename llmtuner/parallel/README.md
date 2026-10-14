# Parallel package

`parallelize.py` is the composition entry. It applies the stages declared in
`stages.py` in this order:

```text
TP → EP → CP → activation checkpointing → compile → FSDP
```

Pipeline parallelism first splits the model, then applies the same sequence to
each local model part.

## Package layout

| Path | Responsibility |
| --- | --- |
| `topology.py` | Resolve degrees and construct all dense, sparse, loss, and data meshes. |
| `stages.py` | Define the shared transformation order for dense models and PP parts. |
| `attention_heads.py` | Validate attention head counts against TP and Ulysses degrees. |
| `model_compile.py` | Apply compilation and configure optional async TP support. |
| `module_patterns.py` | Shared extended module path matching. |
| `compatibility.py` | Central fail-fast checks for unsupported combinations. |
| `tensor_parallel/tensor_parallel.py` | TP public entry, model plan resolution, and module transformation. |
| `tensor_parallel/linear_ops.py` | TP collective and linear kernels. |
| `context_parallel/context_parallel.py` | CP public entry and attention attachment. |
| `context_parallel/attention.py` | KV AllGather, Ulysses, and dual-axis attention kernels. |
| `context_parallel/input_sharding.py` | CP/TP batch and mask sharding. |
| `context_parallel/kcp.py` | Kimi Delta Attention KCP adapter. |
| `expert_parallel/expert_parallel.py` | EP public entry. |
| `expert_parallel/hf_inspection.py` | Read supported Hugging Face MoE layouts. |
| `expert_parallel/block_conversion.py` | Convert one HF MoE block to the native MoE representation. |
| `expert_parallel/model_replacement.py` | Orchestrate conversion across model layers. |
| `expert_parallel/token_dispatch.py` | Dispatch tokens to experts and combine their outputs. |
| `expert_parallel/torchao_dispatch.py` | Adapt optional TorchAO padding and dispatch kernels. |
| `expert_parallel/checkpointing.py` | Gather and slice expert checkpoint state. |
| `fully_shard_parallel/fully_shard_parallel.py` | FSDP runtime entry, mesh policy, and decoder wrapping. |
| `fully_shard_parallel/plan.py` | Pattern-selected FSDP units and prefetch windows. |
| `fully_shard_parallel/fine_grained.py` | Native/custom PyTorch 2.14 strategy adapter. |
| `pipeline_parallel/pipeline_parallel.py` | PP public entry and schedule construction. |
| `pipeline_parallel/partition.py` | Stage partitioning and model splitting. |

Family packages expose public APIs lazily; implementation code imports symbols
from their defining modules. The same-named file is each family's core entry.
Supporting files implement its mechanisms.

Apply stages through `parallelize_hf_transformers` when composing parallelism.
FSDP can be required even with a singleton DP shard axis, so DP degree alone
does not determine whether to skip its stage.
