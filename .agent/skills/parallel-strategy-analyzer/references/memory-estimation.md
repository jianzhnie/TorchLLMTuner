# 显存估算参考

## 模型状态（BF16 + Adam）

| 组成 | 字节/参数 |
|------|-----------|
| 参数（BF16） | 2 |
| 梯度（BF16；默认 `fsdp_reduce_dtype="float32"` 时为 FP32） | 2–4 |
| FP32 主权重 + 动量 + 方差 | 12 |
| **合计** | **约 16** |

```
每卡参数/梯度/优化器 ≈ 总参数字节 / tp / pp
FSDP（data_parallel_shard_size > 1）进一步把模型状态分片到 DP-shard 轴；
fsdp_config.fsdp_cpu_offload=True 将参数与梯度 offload 到主机内存。
```

## 每层 decoder 的激活显存（BF16）

记号：`s`=序列长度，`b`=micro-batch，`h`=hidden，`n_h`=头数，
`n_kv`=KV 头数，`d_ff`=FFN 维度。

```
act_per_layer ≈ 14·s·b·h + 5·s²·b·n_h + 6·s·b·d_ff   （字节）
```

长序列时 `s²` 项主导：当 `5·s²·n_h > 14·s·h + 6·s·d_ff` 时 CP 价值显著
（经验阈值：`s ≥ 8192`，MHA 模型更早）。

并行对激活的影响：

- TP 切分头数与 FFN 维度 → 注意力与 MLP 项 `/ tp`；llmtuner 的 TP 构造上
  即 sequence-parallel，激活还沿序列维切分
- CP 切分序列 → 每卡 `s → s/cp`（`kv_allgather` 策略下每卡保留完整 Q、
  聚合 KV；`ulysses` 重分头数）
- PP 切分层数 → 每卡 `层数 = L / pp`
- FSDP **不**减少激活显存

## 激活检查点（`training.activation_checkpoint_mode`）

模式定义见 `llmtuner/config/activation_checkpoint_config.py`
（`VALID_AC_MODES`），字段挂接在 `TrainingConfig`（`training_config.py`），
各模式的细配置分别在嵌套字段 `selective_ac` / `memory_budget_ac` /
`region_ac` 中：

| 模式 | 效果 | 代价 |
|------|------|------|
| `none` | 完整保存激活 | — |
| `full` | 逐层重计算；只保留层输入/输出（每层约 `4·s·b·h`） | 约 33% 额外计算 |
| `selective` | 逐算子保存/重计算策略（matmul/SDPA 保存集） | 约 15% 额外计算 |
| `region` | 块级重计算、保留声明的 region（需 `torch_remat`，torch ≥ 2.10） | 介于 full 与 selective 之间 |
| `memory_budget` | compile 分割器按 [0,1] 预算取舍；**要求 `training.compile=True`** | 可调 |

## 其他显存

集合通信缓冲 + 框架开销：每卡预留约 1.5–2.5 GB（FSDP all-gather 缓冲、
EP all-to-all 暂存、PP P2P 缓冲）。

## 升级顺序

显存一旦够用即停（每步都增加开销）：

1. FSDP（`data_parallel_shard_size`）——分片模型状态，每步一次通信
2. 激活检查点（`full` 或 `selective`）
3. `fsdp_config.fsdp_cpu_offload=True`——参数/梯度 offload 到 CPU
   （`fsdp_cpu_offload_pin_memory` 默认开启 pinned memory）
4. TP（`tensor_parallel_size`）——仅节点内，逐层集合通信
5. PP（`pipeline_parallel_size`）——引入 bubble
6. CP——用于长序列激活缓解，按注意力类型决定放置优先级
