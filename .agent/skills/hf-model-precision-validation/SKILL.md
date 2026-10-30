---
name: hf-model-precision-validation
description: >
  用 tests/integration_tests/ 的 torchrun 等价性脚本验证 TorchLLMTuner 的
  HF 模型接入或并行配置的数值正确性：基线建立、单策略对齐、组合策略对齐、
  重计算覆盖。适用于数值验收与发散诊断。
---

# HF 模型精度验证

证明同一模型状态与同一全局 batch 在不同并行配置下产生对齐的 **loss** 与
**FP32 全局梯度范数**。仅 loss 一致不算 PASS。

## 硬规则

- `.agent/rules/testing.md`——集成脚本经 `torchrun` 运行，用
  `tests.caps.require_env` 门控；不得靠跳过伪装通过。
- 每个对齐 step 都要比较 **loss 和 pre-clip 全局梯度范数**；同时要求
  学习率与全局输入完全一致。
- 一两步的 smoke 不构成数值验收——需比较连续对齐的优化器 step 窗口
  （各等价性脚本的默认运行长度）。
- 确定性种子：固定 `training.seed` 并设 `deterministic=True`（模式见
  `examples/train_qwen3.py`）。
- 缺硬件/缺资源 → 报告 `BLOCKED`，不得悄悄缩小模型或序列长度。

## 流程

细节见 [references/equivalence-scripts.md](references/equivalence-scripts.md)
与 [references/diagnosis.md](references/diagnosis.md)。

### 1. 基线建立

在可用硬件上以最小并行度运行单策略参考，记录 loss + 梯度范数轨迹：

- FSDP 参考：`tests/integration_tests/fsdp_mesh_equivalence.py`
- TP 参考：`tests/integration_tests/tp_equivalence.py`、
  `tp_grad_norm_equivalence.py`
- CP 参考：`tests/integration_tests/cp_equivalence.py`、
  `cp_sdpa_equivalence.py`、`cp_ulysses_equivalence.py`
- PP 参考：`tests/integration_tests/pp_equivalence.py`
- EP 参考：`tests/integration_tests/ep_equivalence.py`、
  `ep_grad_norm_equivalence.py`、`ep_wiring_equivalence.py`
- 检查点往返：`fsdp_checkpoint_equivalence.py`、
  `pp_checkpoint_equivalence.py`、`hf_checkpoint_equivalence.py`

### 2. 单策略对齐

每种策略在度数 > 1 时必须复现参考轨迹：如
`fine_grained_fsdp_equivalence.py`（custom FSDP 引擎）、
`tp_collective_dimensions.py`、`cp_wiring_equivalence.py`、
`fixed_router_ep_equivalence.py`（固定路由仅是**诊断手段**，学习路由才是
正式用例）。

### 3. 组合策略对齐

多轴组合经由 `llmtuner/parallel/topology.py` 与 `parallelize.py` 交互——
必须显式验证组合 mesh：

- `tp_fsdp_equivalence.py`、`cp_tp_equivalence.py`、`ep_fsdp_equivalence.py`
- `tp_ep_cp_equivalence.py`、`tp_ep_cp_training_equivalence.py`
- `pp_cp_equivalence.py`、`pp_ep_equivalence.py`
- Trainer 级：`trainer_tp_ep_cp_equivalence.py`、
  `parallel_grad_norm_equivalence.py`、`reduce_equivalence.py`

### 4. 重计算 / 显存特性覆盖

- 激活检查点开关与各模式 parity：`ac_equivalence.py`、
  `ep_checkpoint_ac_equivalence.py`、`hf_fsdp_ac_prefetch_equivalence.py`
- CPU offload：`fsdp_cpu_offload_clipping.py`、`npu_meta_cpu_offload.py`（NPU）
- loss 路径：`chunked_loss_equivalence.py`、`vocab_parallel_loss_equivalence.py`

## 验收

一个用例仅当整个对比窗口内每步 loss 与 FP32 全局 pre-clip 梯度范数都在
脚本声明的容差内才算 PASS。否则按 `references/diagnosis.md` 诊断，报告
FAIL/BLOCKED 并给出首个发散 step 与张量。
