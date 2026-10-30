# 等价性脚本参考

所有脚本位于 `tests/integration_tests/`，用 `torchrun` 运行（典型用法
`torchrun --nproc_per_node=N tests/integration_tests/<script>.py`；
`run_all.py` 与 `harness.py` 提供共享的运行器/启动框架）。依赖硬件的脚本
通过 `tests.caps.require_env` 声明需求（如 `torch_npu`、`dtensor`、
`pipelining`、`transformers_qwen3`）。

## 脚本比较什么

等价性脚本让同一个小模型 + 固定数据在两种配置下运行（如度数 1 vs 度数 N，
或两种并行分解），断言 loss 与梯度轨迹在容差内对齐。共享 fixture 见
`tests/data_fixtures.py`。

## 脚本映射表

| 关注点 | 脚本 |
|--------|------|
| FSDP 核心 | `fsdp_mesh_equivalence.py`、`fine_grained_fsdp_equivalence.py`、`fsdp_excluded_gradient_equivalence.py`、`fsdp_pattern_references.py` |
| FSDP + 检查点/offload | `fsdp_checkpoint_equivalence.py`、`fsdp_cpu_offload_clipping.py` |
| TP | `tp_equivalence.py`、`tp_grad_norm_equivalence.py`、`tp_collective_dimensions.py`、`tp_replicated_grad_equivalence.py`、`fixed_router_tp_equivalence.py` |
| CP | `cp_equivalence.py`、`cp_grad_equivalence.py`、`cp_sdpa_equivalence.py`、`cp_ulysses_equivalence.py`、`cp_ulysses_varlen_equivalence.py`、`cp_wiring_equivalence.py`、`cp_packed_single_doc_equivalence.py` |
| PP | `pp_equivalence.py`、`pp_validation_equivalence.py`、`pp_frozen_stage_equivalence.py`、`pp_optimizer_groups.py`、`pp_real_corpus_equivalence.py` |
| EP | `ep_equivalence.py`、`ep_grad_norm_equivalence.py`、`ep_wiring_equivalence.py`、`ep_domino_equivalence.py`、`ep_mc2_equivalence.py`、`moe_aux_loss_grad_equivalence.py`、`shared_expert_tp_equivalence.py`、`tp_moe_equivalence.py` |
| 组合 mesh | `tp_fsdp_equivalence.py`、`cp_tp_equivalence.py`、`cp_tp_training_equivalence.py`、`ep_fsdp_equivalence.py`、`tp_ep_cp_equivalence.py`、`tp_ep_cp_training_equivalence.py`、`pp_cp_equivalence.py`、`pp_ep_equivalence.py`、`trainer_tp_ep_cp_equivalence.py` |
| 重计算 | `ac_equivalence.py`、`ep_checkpoint_ac_equivalence.py`、`hf_fsdp_ac_prefetch_equivalence.py`、`torch_checkpointing_cold_start.py` |
| loss / 归约 | `chunked_loss_equivalence.py`、`vocab_parallel_loss_equivalence.py`、`reduce_equivalence.py`、`parallel_grad_norm_equivalence.py`、`training_metric_reduction_equivalence.py`、`validation_reduction_equivalence.py` |
| 检查点 | `hf_checkpoint_equivalence.py`、`hf_checkpoint_pp.py`、`rng_checkpoint_equivalence.py`、`ema_layout_equivalence.py`、`tp_dp_checkpoint_layout.py`、`tp_ep_checkpoint_layout.py`、`ep_dp_checkpoint_layout.py` |
| NPU 专用 | `npu_*` 系列（如 `npu_moe_kernel_equivalence.py`、`npu_grad_norm_precision.py`）——由 `require_env("torch_npu")` 门控 |

## 新增等价用例

1. 复用 `harness.py` 或既有脚本结构，不新造框架。
2. 确定性输入（固定种子、固定数据）+ 显式声明的容差。
3. 硬件/后端需求在模块顶部用 `tests.caps.require_env` 门控。
4. 命名沿用既有约定：`<axis>_equivalence.py` 或
   `<axes>_<concern>_equivalence.py`。
