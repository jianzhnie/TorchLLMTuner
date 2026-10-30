# 数值发散诊断

等价用例 FAIL 时，定位**第一个**致因环节，而不是调容差。

## 排查顺序

1. **输入同一性**——种子、全局 batch、分片是否一致？数据分片依赖 DP
   world size；改变每卡输入流的配置变化在任何数值比较前就已让对比失效。
2. **状态同一性**——被比较的各次运行初始权重一致（共享检查点，或共享
   种子且 RNG 流与拓扑无关；`rng_checkpoint_equivalence.py`、
   `device_rng_equivalence.py` 覆盖此点）。
3. **首个发散 step 与张量**——定位 loss 或梯度范数首次超出容差的 step，
   再追踪到模块：`alignment_module_trace.py`、`alignment_gradient_trace.py`、
   `reconstruct_gradient_norms.py` 是现有的追踪/重构工具。
4. **集合通信语义**——共享集合通信辅助函数变化后，每个调用方必须保持
   相同的设备/进程组语义（`collective_contract_equivalence.py`）。
5. **精度路径**——梯度归约 dtype（`fsdp_config.fsdp_reduce_dtype`，默认
   FP32）、混合精度边界（`fine_grained_mixed_precision.py`）。

## 规则

- 靠后的传输/超时失败在清理证实之前都按级联处理。
- LR=0 或 `fixed_router=True` 可用来隔离原因，但仅为诊断手段，绝不替代
  正式用例。
- 不得为让某个用例通过而修改共享框架精度行为（归约 dtype/顺序、cast
  策略）——报告最早发散的张量并把决策上交给用户。
- FAIL 报告须含：首个发散 step、对比量（loss / 范数）、观测 vs 期望的
  差值、被波及的模块或集合通信。
