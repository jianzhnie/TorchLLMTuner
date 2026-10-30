---
name: parallel-strategy-analyzer
description: >
  根据模型与硬件约束为 TorchLLMTuner 推荐 FSDP/TP/PP/CP/EP 策略组合，
  附显存/集合通信/bubble 估算与具体的 llmtuner.config 字段值。适用于策略
  规划、扩容、OOM 排查或配置对比。不用于实现并行代码（用各 *-dev-expert
  agent）或直接启动训练任务。
---

# 并行策略分析

为 `llmtuner` 训练推荐并行配置，并输出对应的
`llmtuner.config.ParallelConfig`（及相关的 training/memory 字段）。所有
并行度映射到 `llmtuner/config/parallel_config.py` 中的真实字段——绝不
编造配置键。

## 适用场景

- 选择或调整并行度；排查 OOM；对比两个配置；在启动前估算集合通信量或
  流水线 bubble。

## 不适用

- 实施配置改动或规划代码变更（改哪些文件、什么顺序）→ `planner` agent。
- 实现并行维度的代码 → 各 `*-dev-expert` agent。
- 启动或调参真实训练任务——本技能只产出建议与配置片段，不执行训练。
- 数值验收 → `hf-model-precision-validation` 技能。

## 分析流程（每个阶段前加载对应 workflow 文件）

```text
1 收集输入        → workflows/01-collect-inputs.md
2 基线 + 策略搜索 → workflows/02-baseline-and-search.md（先显存，后决策树）
3 成本 + 配置输出 → workflows/03-cost-and-config.md（通信/bubble + ParallelConfig 字段）
```

**先做显存基线再搜策略**——避免把"塞得进显存"置于"集合通信开销不可
接受"之上。

## 公式与契约（按需加载）

- [references/io-contract.md](references/io-contract.md)——必填输入、输出章节
- [references/memory-estimation.md](references/memory-estimation.md)——显存估算公式

## 输出（简版）

输出顺序：基线 → 推荐并行度 → 分片后每卡显存 → 集合通信分解 → bubble
估算 → `ParallelConfig`（+ 激活检查点/offload）配置片段 → 前 3 个备选
方案。完整格式见 `io-contract.md`。

## 硬规则

- 只引用 `llmtuner/config/` 中真实存在的字段（`parallel_config.py`、
  `training_config.py`、`activation_checkpoint_config.py`）。不确定的字段
  用泛化意图描述，不编造键名。
- 估算误差 ±10–20%；通信按理想带宽估算；MoE 负载不均衡不建模——在报告
  中声明这些前提。
- 按 `ParallelConfig.derive_dp` 的方式校验并行度可行性：
  `world_size == dp_shard × dp_replicate × tp × pp × cp`
  （`data_parallel_shard_size=-1` 表示"派生剩余 rank"）。
