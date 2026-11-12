# mini-llmtuner：在原有实验代码上拆分职责

本目录复用已有 Qwen3-MoE 模型、HF adapter、并行计划和装配逻辑，不再维护另一套 `mini_llmtuner/` 实现，也不导入生产包 `llmtuner`。模型仍使用本目录的 `model.py` 和 `layers.py`，没有替换成额外的玩具网络。

## 代码如何协作

```text
ExperimentConfig
    ↓
ExperimentTrainer（兼容入口与默认装配）
    ├─ Qwen3MoeModel → resolve_plan → assemble_model
    ├─ optimizer / scheduler / loss → TrainingEngine
    ├─ SyntheticData
    └─ CheckpointManager
                ↓
BaseTrainer：取数据 → engine.train_step → metrics / profiler → 更新步数
```

| 模块 | 职责 |
| --- | --- |
| `model.py`、`layers.py` | Qwen3-MoE 数学实现、模型结构与显式并行边界 |
| `protocol.py`、`hf_adapter.py`、`hf_factory.py` | 模型接口与已有 HF 接入扩展点 |
| `config.py`、`resolver.py` | 声明配置、解析计划、检查模块绑定 |
| `assembly.py` | 在训练前应用并行计划，保留原有后端 |
| `engine.py` | 前向、反向、梯度裁剪、optimizer 和 scheduler 更新 |
| `data.py` | 可恢复的合成数据游标 |
| `trainer.py` | `BaseTrainer` 管生命周期；`ExperimentTrainer` 组装默认组件 |
| `components.py` | 可替换 loss、metrics、optimizer factory、scheduler、profiler |
| `checkpoint.py` | 模型状态适配、训练状态保存与恢复 |

自定义模型、数据或服务时，可以直接构造 `TrainingEngine`、`CheckpointManager` 和 `BaseTrainer`，无需修改默认 Qwen3-MoE 装配入口。训练循环不判断 TP、FSDP 等并行模式。

## 运行与恢复

在仓库根目录执行，环境需安装 PyTorch 与支持 Qwen3-MoE 配置的 Transformers：

```bash
python -m experiments.mini-llmtuner.train --steps 3 --gradient-accumulation-steps 2
python -m experiments.mini-llmtuner.train --steps 2 --checkpoint /tmp/mini-checkpoint
python -m experiments.mini-llmtuner.train --steps 3 --resume /tmp/mini-checkpoint
python -m pytest -q experiments/mini-llmtuner/tests
```

`--steps` 是目标 optimizer step 总数，恢复时从已完成的步数继续。loss 组件返回有效 token 的损失之和；引擎按整个累积窗口的有效 token 总数归一化，支持不同 microbatch 中有效 token 数不同的情况。默认梯度裁剪阈值为 1.0。

checkpoint 使用目录中的 `state.pt`，通过临时文件原子替换，包含模型、optimizer、scheduler、数据游标、随机状态与配置签名。恢复允许增加目标步数，但拒绝改变模型配置、拓扑、batch size、序列长度、累积步数、学习率或 seed。当前格式为版本 2，不兼容旧的无版本实验 checkpoint；只加载可信的本地 checkpoint。

## 当前边界

当前验证范围是 CPU 单进程 Qwen3-MoE 训练、梯度累积和精确恢复。CLI 默认使用 CPU 和合成数据；HF adapter 保留为接入扩展点，默认入口尚未提供 HF 权重选择。

原有 TP / FSDP2 装配代码继续保留，但并不意味着本次完成了分布式端到端验证：Qwen3-MoE 的 packed expert TP 仍显式报错，EP dispatcher、CP 和 PP schedule 尚未实现。当前 checkpoint 是单进程原型，不能用于多 rank 分片状态保存；跨拓扑恢复、HF 权重导出和真实分布式 token 归一化需要独立完善。重计算与量化配置也尚未接入执行。
