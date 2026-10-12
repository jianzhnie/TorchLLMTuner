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
| `module_patterns.py` | Shared extended module path matching. |
| `compatibility.py` | Central fail-fast checks for unsupported combinations. |
| `tensor_parallel/tensor_parallel.py` | TP public entry, model plan resolution, and module transformation. |
| `tensor_parallel/linear_ops.py` | TP collective and linear kernels. |
| `context_parallel/context_parallel.py` | CP public entry and attention attachment. |
| `context_parallel/attention.py` | KV AllGather, Ulysses, and dual-axis attention kernels. |
| `context_parallel/input_sharding.py` | CP/TP batch and mask sharding. |
| `context_parallel/kcp.py` | Kimi Delta Attention KCP adapter. |
| `expert_parallel/expert_parallel.py` | EP public entry. |
| `expert_parallel/hf_layout.py` | Read supported Hugging Face MoE layouts. |
| `expert_parallel/conversion.py` | Convert one HF MoE block to the native MoE representation. |
| `expert_parallel/model_conversion.py` | Orchestrate conversion across model layers. |
| `expert_parallel/checkpoint.py` | Gather and slice expert checkpoint state. |
| `fully_shard_parallel/fully_shard_parallel.py` | FSDP runtime entry, mesh policy, and decoder wrapping. |
| `fully_shard_parallel/plan.py` | Pattern-selected FSDP units and prefetch windows. |
| `fully_shard_parallel/fine_grained.py` | Native/custom PyTorch 2.14 strategy adapter. |
| `pipeline_parallel/pipeline_parallel.py` | PP public entry and schedule construction. |
| `pipeline_parallel/partition.py` | Stage partitioning and model splitting. |

Each family is imported through its package. The same-named file is its core
entry; supporting files do not orchestrate other parallel families.
