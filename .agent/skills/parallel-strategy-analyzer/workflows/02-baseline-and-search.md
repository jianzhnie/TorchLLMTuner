# 阶段 2 — 基线 + 策略搜索

## 第 1 步：全局基线（无并行）

按整模型单卡放置估算每卡显存（公式见
`references/memory-estimation.md`）：模型状态 `P × 16B`、激活
`act_per_layer × L`，外加约 2GB 开销。指出主导项（模型状态 vs `s²`
注意力激活）。

## 第 2 步：决策树

```text
单卡放得下？  → 纯 DP（data_parallel_shard_size = world_size）
 │
短序列（s² 项不主导）：
 │  1. FSDP：data_parallel_shard_size = 剩余 rank        → 够用？完成
 │  2. + activation_checkpoint_mode = "selective" 或 "full" → 够用？完成
 │  3. + fsdp_config.fsdp_cpu_offload = True                → 够用？完成
 │  4. + tensor_parallel_size（≤ 每节点卡数）               → 够用？完成
 │  5. + pipeline_parallel_size（跨节点切层）
 │
长序列（s² 主导，s ≥ 8192）：
 │  GQA/MLA（n_kv << n_h）：CP 便宜 →
 │      FSDP → 激活检查点 → CP → offload → TP → PP
 │  MHA（n_kv == n_h）：CP 昂贵 →
 │      FSDP → 激活检查点 → TP → CP（小，2–4）→ PP
 │
MoE → 加 expert_parallel_size（需整除专家数）；ep × tp 尽量落在节点内
      （all-to-all 对带宽敏感）
```

## llmtuner 特有的放置规则

- **TP 构造上即 sequence-parallel**：`tensor_parallel_size > 1` 要求
  `tp_config.enable_sequence_parallel=True`（默认值；关闭会直接报配置
  错误）。TP 同时沿序列维切分激活——估算激活时计入这一点。
- **CP 有三个独立轴**（度数相乘）：
  `context_parallel_kv_allgather_size`、`context_parallel_ulysses_size`、
  `context_parallel_kcp_size`。策略字段
  `cp_config.context_parallel_strategy` 选择注意力重分布方式
  （`kv_allgather` / `ulysses` / `kcp`）；Ulysses 要求
  `context_parallel_load_balancer=None`；KCP 要求 `tensor_parallel_size=1`
  且与另两个轴互斥。不要组合校验会拒绝的轴。
- **EP 在稠密区域内铺 tile**：`expert_parallel_size` 从
  DP-shard × CP × TP 区域中切出专家；EP=1 时允许纯 TP 专家切分。
  `ep_config.ep_token_dispatcher`：`alltoall`（稳妥默认）、`torchao`、
  `domino`、`mc2`（NPU）；`deepep`/`hybridep` 是已注册缺口——绝不推荐。
- **DP replicate 与 shard**：`data_parallel_replicate_size > 1` 得 DDP；
  与 `data_parallel_shard_size > 1` 组合即 HSDP（常见布局为节点间
  replicate、节点内 shard）。
- **FSDP 引擎**：`fsdp_config.fsdp_implementation` 为 `native`（默认）或
  `custom`——除非用户已在跑 custom 引擎，否则推荐 `native`。
