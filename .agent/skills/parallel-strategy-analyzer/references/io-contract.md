# 输入 / 输出契约

## 调用示例

```bash
分析 Qwen3-30B-A3B（MoE）在 32 张 64GB NPU 上的策略
8B 稠密模型，16× A100 80GB，seq_len 8192，全局 batch 256
目前 13B 在 32 卡 GPU 上纯 FSDP，seq_len=32768 时 OOM
对比 70B 模型在 64 卡上 TP=8/PP=2 与 TP=4/PP=4
```

## 必填输入

| 参数 | 说明 | 示例 |
|------|------|------|
| 模型 | 参数量或 HF 架构名 | 30B、Qwen3-30B-A3B |
| 设备数 | 总卡数 | 32 |
| 单卡显存 | 每设备 HBM | 64GB、80GB |

## 可选输入（提高精度）

| 参数 | 示例 |
|------|------|
| 序列长度 | 4096、32768 |
| 全局 batch | 256 |
| hidden / 层数 / 头数 / KV 头数 / FFN 维度 | h=2048、L=48、n_h=32、n_kv=4 |
| MoE：专家数、top-k | 128 专家、top-8 |
| 每节点卡数 + 互联 | 每节点 8 卡，NVLink / HCCS |

## 输出章节（必需）

1. **全局基线**——无并行时的模型状态显存（BF16+Adam，16B/参数）与激活
   显存；指出主导项
2. **推荐方案**——并行度：`tensor_parallel_size`、`pipeline_parallel_size`、
   各 CP 轴大小、`expert_parallel_size`、派生的 `data_parallel_shard_size`
3. **分片后显存**——每卡参数/梯度/优化器/激活，对照 HBM（目标 ≤ 90%）
4. **集合通信分解**——各轴通信量与放置（节点内 vs 节点间）
5. **流水线 bubble**——PP > 1 时 `(pp_size - 1) / num_pp_microbatches`
6. **配置片段**——只用真实 `llmtuner.config` 字段
   （见 `workflows/03-cost-and-config.md`）
7. **前 3 个备选**——显存/bubble/通信取舍对比表

## 局限性

- 显存估算与实际运行 ±10–20% 误差
- 通信按理想带宽估算；计算-通信重叠为启发式
- MoE 专家负载不均衡不建模；`fixed_router` 仅为调试用途，绝不作为生产
  推荐
