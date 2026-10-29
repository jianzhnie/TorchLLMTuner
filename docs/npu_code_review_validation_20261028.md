# 2026-10-28 NPU 改动验证

结论：本次小模型数值与 checkpoint 用例通过；共享设备上的计时未证明稳定提速。
本记录不代表完整 Trainer、多步优化器训练或四种生产拓扑的性能验收。

## 环境与版本

- 节点 `10.1.0.29`，容器 `hyper-parallel-env`，Ascend910_9392。
- PyTorch `2.12.0+cpu` 加 torch_npu；实际张量与算子在 NPU 上运行。
- 基础提交 `928c54c1b7dffe9936a5d97b72452c9f1bfe639b`，含当前未提交改动。
- 本机与远端关键源码 SHA256 已核对一致。
- 原容器 Transformers 4.56.2，不满足项目依赖。通过独立目录
  `outputs/npu_review_20261028/deps` 复用本机依赖，不修改全局安装。
- 8 张卡均有现有任务，每卡约占用 47 GB；未停止现有任务。

## 数值与恢复

以下最终执行均退出 0，日志与脚本保存在 `outputs/npu_review_20261028/`。

| 用例 | 结果与范围 | 日志 |
| --- | --- | --- |
| 4-rank CPU-offload clipping | 1、2、∞ 范数 × foreach 开关全部通过；容差 1e-6 | `offload_retry.log` |
| 4-rank joint TP/EP checkpoint | 各 rank 无效 chunk 协调、冷恢复、AdamW/EMA 下一次更新通过 | `joint_checkpoint.log` |
| 8-rank TP/EP/DP checkpoint | dense DP4/TP2、expert FSDP2/EP4 DCP 恢复与布局断言通过 | `checkpoint.log` |
| Qwen3 fused/separate QKV | FP32/BF16 输出最大差均为 0；最大梯度差分别 1.86e-9、6.10e-5 | `model_benchmark.log` |
| Flex/SDPA 前后向 | FP32 输出最大差 2.38e-7；BF16 0.015625，梯度最大差 1.53e-5；容差分别 3e-5、0.02 | `attention_numeric_cpu_reference.log` |
| FP32 RoPE/CPU FP64 参考 | FP32 Q 最大差 1.85e-6；BF16 Q 最大差 0；Q/K 均通过断言 | 同上 |

BF16 绝对容差 0.02 仅用于本次小模型探测，不作为生产训练误差标准。
Flex 使用未编译 eager 路径，不能代表 compiled fused kernel。
首次 RoPE 参考发现 NPU FP64 会降为 FP32，已使用 CPU FP64 参考重跑；
以 `attention_numeric_cpu_reference.log` 为准。

## 性能探测

Qwen3：2 层、hidden=128、intermediate=256、4 个 Q head、2 个 KV head、
head_dim=32、vocab=256、batch=1、sequence=128，SDPA、无 AC/compile，
分别以 FP32/BF16 执行。测量 zero_grad + forward + backward，前后 NPU 同步，
预热 5 次、记录 15 次；不包含 optimizer.step。

| 精度 | 首轮 separate/fused 中位数 ms | 复测 separate/fused 中位数 ms |
| --- | --- | --- |
| FP32 | 9.945 / 10.006 | 16.469 / 11.740 |
| BF16 | 11.396 / 11.509 | 17.863 / 18.895 |

两轮方向与幅度不稳定，存在约 173 ms 离群值，不报告加速比。
进程累计峰值分配约 113/136 MB，包含两模型及临时量，不能用来比较两条路径显存。
复测计时后额外采集一个 BF16 前后向 trace：`separate_trace.json`、
`fused_trace.json`。CPU aten 事件中 linear 由 15 降为 11，cat 由 6 增为 10，
符合两层分别减少 QKV linear、增加权重打包的行为；不据此推断端到端提速。
Profiler 提示停止时仍处于 RECORD，trace 仅用于定性诊断。

## 复现入口与未验证项

在远端容器中进入仓库，设置
`PYTHONPATH=outputs/npu_review_20261028/deps:.`，执行：

```bash
torchrun --standalone --nproc-per-node=4 tests/integration_tests/fsdp_cpu_offload_clipping.py
torchrun --standalone --nproc-per-node=4 outputs/npu_review_20261028/joint_checkpoint_npu.py
torchrun --standalone --nproc-per-node=8 outputs/npu_review_20261028/checkpoint_npu.py
python outputs/npu_review_20261028/model_benchmark.py
python outputs/npu_review_20261028/attention_numeric.py
```

两个 checkpoint 脚本由现有 CPU 用例仅调整 backend/device 得到；原始脚本未修改。
未验证：独占设备性能、compiled Flex、Qwen3-MoE torchtitan experts 数值对齐、
完整 Trainer 长程训练、TP/EP degree 变更恢复、四种生产并行配置吞吐、
CPU-offload 优化前后性能和所有异常路径。本轮未发现需要追加修改的源码 bug。
生成日志、依赖和 trace 不应提交到版本库。
