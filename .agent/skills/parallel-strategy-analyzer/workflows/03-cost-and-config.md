# 阶段 3 — 成本分析 + 配置输出

## 集合通信分解（按轴）

| 轴 | 集合通信 | 频率 | 放置规则 |
|----|----------|------|----------|
| TP | all-reduce / all-gather（sequence-parallel 融合引擎） | 每个线性层，前向+反向 | **仅节点内**：`tensor_parallel_size ≤ 每节点卡数` |
| CP kv_allgather | KV all-gather | 每个注意力层 | GQA（n_kv 小）可容忍跨节点 |
| CP ulysses | all-to-all（头数） | 每个注意力层 | 优先节点内 |
| EP | all-to-all token 分发/合并 | 每个 MoE 层，前向+反向 | 优先 `ep × tp ≤ 每节点卡数` |
| FSDP | 参数 all-gather + 梯度 reduce-scatter | 每层 / 每步 | 跨节点可接受 |
| PP | P2P send/recv | 每个 micro-batch 边界 | 跨节点可接受 |

标出任何违反放置规则的轴；给出带宽数据时估算暴露时间
`通信量 / 有效带宽`。

## 流水线 bubble（PP > 1）

```
bubble 占比 ≈ (pipeline_parallel_size - 1) / num_pp_microbatches
```

`num_pp_microbatches` 即 `pp_config.num_pp_microbatches`（按 DP rank 与
累积迭代计）。调度用 `pp_config.pipeline_parallel_schedule`（默认
`"1F1B"`；必须是合法的 `torch.distributed.pipelining` 调度名）。推荐足够
的 micro-batch 数把 bubble 压到约 10% 以内。

## 配置输出（只用真实字段）

输出如下形式的 `llmtuner.config` 片段：

```python
from llmtuner.config import (
    ContextParallelConfig, ExpertParallelConfig, FSDPConfig,
    ParallelConfig, PipelineParallelConfig, TensorParallelConfig, TrainingConfig,
)

parallel = ParallelConfig(
    tensor_parallel_size=...,            # TP 度（1 为关闭）
    pipeline_parallel_size=...,          # 物理 PP rank 数
    context_parallel_kv_allgather_size=...,  # 各 CP 轴（度数相乘）
    expert_parallel_size=...,            # EP 度（1 为关闭）
    data_parallel_shard_size=-1,         # -1：剩余 rank 派生为 FSDP
    data_parallel_replicate_size=1,      # >1 → DDP / HSDP
    tp_config=TensorParallelConfig(enable_sequence_parallel=True),
    fsdp_config=FSDPConfig(
        fsdp_reshard_after_forward="default",
        fsdp_reduce_dtype="float32",
        fsdp_cpu_offload=False,          # 最后的显存救济手段
    ),
    pp_config=PipelineParallelConfig(
        pipeline_parallel_schedule="1F1B",
        num_pp_microbatches=...,         # ≥ pp，压住 bubble
    ),
    cp_config=ContextParallelConfig(context_parallel_strategy="kv_allgather"),
    ep_config=ExpertParallelConfig(ep_token_dispatcher="alltoall"),
)
training = TrainingConfig(
    activation_checkpoint_mode="selective",  # none/full/selective/memory_budget/region
    ...
)
```

报告中需复述的校验点：

- `derive_dp` 要求 `world_size == dp_shard × dp_replicate × tp × pp × cp`
- 非 `alltoall` 调度器要求 `expert_parallel_size > 1`
- `memory_budget` 模式要求 `training.compile=True`
- `enable_fsdp_symm_mem` 仅限计算能力 ≥ 9.0 的 NVIDIA GPU

## 前 3 个备选

表格：配置 | 每卡显存 | bubble | 暴露通信 | 何时优先选它。
