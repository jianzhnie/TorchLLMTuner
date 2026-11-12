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

## 复用默认装配，替换模型或组件

`ExperimentTrainer` 保留为兼容入口，支持通过关键字参数注入 `model`、`optimizer_factory`、`scheduler_factory`、`loss`、`data`、`metrics`、`profiler` 和 `checkpoint_adapter`。例如，在 Python 中通过 `importlib` 加载含连字符的实验包：

```python
from importlib import import_module

import torch
from transformers import LlamaConfig, LlamaForCausalLM

package = "experiments.mini-llmtuner"
Config = import_module(f"{package}.config").ExperimentConfig
Trainer = import_module(f"{package}.trainer").ExperimentTrainer
Adapter = import_module(f"{package}.hf_adapter").LlamaModelAdapter

model = Adapter(LlamaForCausalLM(LlamaConfig(
    vocab_size=64, hidden_size=32, intermediate_size=64,
    num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
)))
with Trainer(
    Config(steps=2), torch.device("cpu"), model=model,
    scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.9,
    ),
    runtime_signature="tiny-llama-step-lr-v1",
) as trainer:
    print(trainer.train())
```

自定义组件或预处理语义改变时，应更新 `runtime_signature`；签名会参与恢复校验。框架不会自动推断任意 callable 的配置。scheduler 和数据源需实现 `state_dict()` / `load_state_dict()`；数据源还需提供 `next_batch()`。loss 必须返回按 `labels != -100` 计算的损失总和。

`with` 退出时关闭 profiler 和数据源提供的 `close()`，异常退出同样执行，重复关闭不会重复释放。optimizer 更新成功后立即提交已完成步数，再调用 metrics 和 profiler，避免观察组件报错导致步数落后于参数更新。

HF Llama 的单进程训练已纳入回归测试。HF adapter 的 TP / EP 暂时明确拒绝执行：仅凭线性层名称推导切分策略不足以保证注意力及 DTensor 布局正确。重计算、量化和 CP / PP 等尚未实现的计划也会在训练前报错，避免配置被静默忽略。
