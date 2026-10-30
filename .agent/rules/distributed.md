---
name: distributed
description: llmtuner/parallel 分布式并行代码（FSDP/TP/PP/CP/EP）的硬规则
paths:
  - llmtuner/parallel/**
---

# 分布式系统

`llmtuner/parallel/` 负责应用 FSDP/TP/PP/CP/EP 五个并行维度。共享基础设施
位于包顶层（`topology.py`、`parallelize.py`、`stages.py`、
`attention_heads.py`、`module_patterns.py`、`compatibility.py`）。

## 公共入口约定

- 每个并行家族以**同名文件**作为公共入口：
  `fully_shard_parallel/fully_shard_parallel.py`、
  `tensor_parallel/tensor_parallel.py`、
  `pipeline_parallel/pipeline_parallel.py`、
  `context_parallel/context_parallel.py`、
  `expert_parallel/expert_parallel.py`。
- 公共导出集中在这几个文件（及家族 `__init__.py`）；实现细节留在同目录
  其他模块。内部改动不得破坏入口签名的一致性。

## FSDP（`fully_shard_parallel/`）

- 显存生命周期是首要正确性问题：all-gather/reduce-scatter 消费完成后及时
  释放分片存储（`untyped_storage().resize_(0)`）、清空通信缓冲、将已消费的
  梯度置空（`param.grad = None`）。
- 细粒度分片（`fine_grained*.py`、`fine_grained_param_group.py`）必须保持
  参数组状态（`fine_grained_state.py`）与 `plan.py` 的计划一致。
- CPU offload 路径必须在计算流访问前同步拷贝流。
- 用 `torchrun` 等价性脚本验证：`fsdp_mesh_equivalence.py`、
  `fsdp_checkpoint_equivalence.py`、`fine_grained_fsdp_equivalence.py` 等。

## 张量并行 TP（`tensor_parallel/`）

- 集合通信正确性：列并行与行并行线性层（`linear_ops.py`）的
  all-reduce/all-gather 位置不同——不要混淆 autograd 路径需要哪种集合通信。
- 检查点布局（`checkpointing.py`）必须能跨不同 TP 度往返恢复分片权重。
- 用 `tp_equivalence.py`、`tp_grad_norm_equivalence.py`、
  `tp_collective_dimensions.py` 验证。

## 流水线并行 PP（`pipeline_parallel/`）

- 阶段切分（`partition.py`）与 micro-batch 驱动（`microbatch_drivers.py`）
  必须在每个 micro-batch 后冲刷缓冲/缓存；跨 micro-batch 累积会导致显存
  持续增长。
- 跨阶段 P2P send/recv 顺序必须避免死锁——改动需用 `torchrun` 跑
  `pp_equivalence.py`、`pp_cp_equivalence.py`、`pp_ep_equivalence.py`。

## 上下文并行 CP（`context_parallel/`）

- 各注意力变体（`attention.py`、`sdpa_attention.py`、
  `compressor_attention.py`、`kcp.py`）必须在序列分片与输出布局上保持一致；
  输入分片在 `input_sharding.py`。
- 用 `cp_equivalence.py`、`cp_sdpa_equivalence.py`、
  `cp_ulysses_equivalence.py`、`cp_tp_equivalence.py` 验证。

## 专家并行 EP（`expert_parallel/`）

- token 分发（`token_dispatch.py`、`domino_dispatch.py`、`mc2_dispatch.py`）
  在梯度流经的路径上必须可微；权重布局转换（`block_conversion.py`、
  `kimi_conversion.py`、`model_replacement.py`）必须与检查点
  （`checkpointing.py`）一致。
- NPU 专用算子（`mc2_ops.py`、`npu_baseline_experts.py`）需硬件门控验证。
- 用 `ep_equivalence.py`、`ep_fsdp_equivalence.py`、`ep_wiring_equivalence.py`
  验证。

## 跨维度通用规则

- 集合通信语义的改动是跨切面的：共享辅助函数变化时，检查每个调用方获得
  相同的设备/进程组语义。
- 异步集合通信：读取输出前必须 `handle.wait()`；跨流依赖需要显式 event。
- 多轴组合（FSDP+TP+CP+EP）经由 `topology.py` 与 `parallelize.py` 交互——
  这两处改动影响面最大；硬件允许时运行组合等价性脚本
  （`trainer_tp_ep_cp_equivalence.py`、`tp_ep_cp_training_equivalence.py`）。
- 分布式行为不能仅靠 CPU 推理验证——集合通信正确性必须经 `torchrun`
  验证（硬件由 `tests.caps.require_env` 门控）。
