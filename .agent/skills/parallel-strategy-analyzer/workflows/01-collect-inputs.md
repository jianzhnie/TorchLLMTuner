# 阶段 1 — 收集输入

目标：在任何计算之前，锁定模型、硬件与负载三类事实。

## 模型事实

- 参数量（或推导：稠密 ≈ `L·(4h² + 3·h·d_ff) + vocab·h`；MoE 每层另加
  专家参数 `专家数 · 3·h·d_ff_expert`）
- `hidden_size`（h）、`num_hidden_layers`（L）、`num_attention_heads`
  （n_h）、`num_key_value_heads`（n_kv < n_h 即 GQA）、FFN 维度
- MoE：专家数、top-k、专家 FFN 维度
- 这些与 `llmtuner/config/model_config.py` / HF `AutoConfig` 字段同构；
  对 hub 模型，读其 config.json 而非猜测。

## 硬件事实

- 总卡数、每节点卡数、单卡 HBM
- 互联：节点内带宽（NVLink/HCCS 级别）vs 节点间（RDMA）

## 负载事实

- `max_seq_len`（s）、`global_batch_size`、目标每卡 micro-batch
- 训练精度（默认：BF16 参数，经 `fsdp_config.fsdp_reduce_dtype="float32"`
  做 FP32 梯度归约）

## 前置可整除性检查

- 任何候选 TP 需满足 `n_h % tp == 0`
- `L % pp == 0` 为佳（llmtuner 也支持
  `pp_config.pipeline_parallel_first_stage_less_layers` /
  `pipeline_parallel_last_stage_less_layers` 来抵消 embedding/输出层开销）
- `专家数 % ep == 0`
- `global_batch_size % (dp × micro_batch × num_pp_microbatches（PP>1 时）) == 0`
- `world_size == dp_shard × dp_replicate × tp × pp × cp`——与
  `ParallelConfig.derive_dp` 强制的恒等式一致
