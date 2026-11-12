# Qwen3-MoE：模型契约 → 纯计划 → 直接执行

本实验是 `docs/model_distributed_architecture_guide.md` 第 7 部分的最小可运行实现。
它不是第二套生产训练框架，而是用一个 Qwen3-MoE 纵向闭环验证以下调用链：

```text
模型契约
  → 纯计划解析与预检
  → TP → EP → FSDP
  → 真实训练
  → DCP 保存与恢复
```

## 1. 模型拥有结构和数学契约

`model.py` 明确声明：

- decoder、MoE、embedding、norm、head 的准确 FQN；
- 模块角色和输入输出布局；
- TP 的 colwise、rowwise、packed-colwise 计划；
- 专家的 `fused_gate_up` 布局；
- attention head、专家宽度和专家数量的整除约束。

模型直接提供 TP 计划，不读取 HF 自动计划。`layers.py` 自定义 RMSNorm、default
RoPE、GQA attention、router、packed experts 和 decoder；`model.py` 自定义初始化、
层栈和输出 head。模型不导入 `llmtuner`，不继承或调用 Transformers modeling 类；
仅复用 `Qwen3MoeConfig`。包入口延迟加载，独立导入模型不加载训练服务。
模型支持 eager/SDPA、SiLU 与 default RoPE；其他 RoPE、sliding window、KV cache
和非零 router auxiliary loss 明确拒绝。训练与分布式服务由本实验的最小组件提供，
不导入 `llmtuner`。

## 2. 先解析和校验，再修改模型

`resolver.py` 在参数和 forward 被修改前完成：

- 将 contract 中的准确路径绑定到真实模块；
- 解析实际 DP 度数；
- 检查共享参数与 TP/EP 几何约束；
- 拒绝未支持的配置；
- 生成 frozen `ExecutionPlan`，只保存模块引用和元数据，不复制张量。

`assembly.py` 在执行前再次调用 `validate_binding()`，防止计划引用失效，并协调
rank 本地错误，避免其他 rank 进入不匹配的 collective。

预检可以减少半改模型，但不承诺执行阶段具有事务回滚能力。既有 TP/EP/FSDP
引擎仍保留自己的转换和必要校验；本实验不是完全由绑定引用驱动的全新执行器。

## 3. 计划与执行器分离

统一装配顺序为：

```text
resolve_plan
  → apply_tp
  → apply_ep
  → apply_fsdp
```

`assembly.py` 只负责顺序、依赖和失败协调。通用 collective、专家转换、FSDP 和
本地执行器当前只实现单进程安全路径；TP/EP/FSDP 可以通过同一 adapter 接口接入
真实后端，不改变模型和 trainer。

## 4. 训练语义明确且可复现

`trainer.py` 实现一个小而真实的训练闭环：

- 合成数据按 step 和 DP rank 确定生成；
- TP rank 共享同一文档并沿序列切分；
- 使用 next-token labels，末 token 目标为 `-100`；
- RoPE 使用完整文档位置；
- 按全局有效 target 数归一化 loss；
- 区分 TP 分片参数与复制参数的梯度处理；
- 执行真实 AdamW 更新。

默认 CPU FP32、NPU BF16；固定学习率，不使用 scheduler 或 EMA。关闭 router
auxiliary loss 和 attention dropout。TP 将每个文档作为独立微批执行，不在序列
边界直接拼接多个 `[B,T,H]` 输入；不支持的输入会在配置阶段拒绝。

## 5. Checkpoint 转换独立

`checkpoint.py` 使用独立的 `CheckpointAdapter` 保存：

- 模型及并行布局；
- FQN optimizer 状态与 Adam moments；
- 下一步游标；
- 每个 rank 的 Python、CPU 和设备 RNG。

加载前先比较不可变配置签名：模型、数据、精度、设备和拓扑必须一致。恢复从
下一步继续，不重复执行已完成的 step。Python RNG 的 pickle 仅用于可信的本地
checkpoint；本实验不支持跨拓扑恢复。

## 6. 明确能力边界，不做空壳支持

当前支持：

- 单进程；
- PyTorch 原生 TP（`torch.distributed.tensor.parallel`）；
- PyTorch FSDP2（`torch.distributed.fsdp.fully_shard`）；
- EP/CP/PP 的显式模型 hook 和能力检查；
- 同步 DCP 保存与恢复。

明确未接入：

- CP、PP 的具体 attention/schedule 实现；
- AC、compile；
- activation/optimizer swap；
- 真实数据 pipeline；
- 异步/原子 checkpoint 发布；
- 跨拓扑恢复、跨节点和长程验收；
- 完整模型 meta 初始化。

未实现能力没有虚假 CLI 开关。CP/PP 在配置入口拒绝；其他未实现功能不会以
空模块形式出现。实验阶段顺序只是支持子集，不改变生产 `parallel/stages.py`。

## 7. 文件职责

```text
config.py      实验配置与 Qwen3-MoE 配置读取，只读架构
protocol.py    ModelProtocol 与 checkpoint adapter 协议
components.py  可替换 loss、metrics、scheduler 和 profiler
model.py       显式构造、模块入口、准确 FQN、TP/专家布局与几何约束
resolver.py    不修改模型的目标绑定、ownership、拓扑和能力预检
assembly.py    rank 错误协调，然后执行 TP → EP → FSDP
trainer.py     可恢复合成数据、next-token loss、梯度同步和 AdamW 更新
checkpoint.py  模型/optimizer 转换、游标和 RNG
train.py       Python/torchrun 入口、设备绑定、rank0 摘要和资源关闭
```

## 8. 运行

## 8.1 接入 HuggingFace Llama 和 DeepSeek-V3

直接加载的 HuggingFace causal LM 通过 `HFModelAdapter` 接入，不需要修改
`BaseTrainer`。工厂会根据 `config.model_type` 选择 Llama 或 DeepSeek-V3 adapter，
并从实际模块树生成线性层 TP plan，避免依赖固定版本的通配符路径：

```python
from experiments.qwen3_moe_framework.hf_factory import load_hf_model

model = load_hf_model("meta-llama/Llama-3.2-1B", device=device)
assembly = model.parallelize(ParallelConfig(), world_size=1)
```

DeepSeek-V3 的 MoE 专家仍需要模型级 EP dispatcher；Llama 可直接使用通用的
TP/FSDP2 适配路径。CP 和 PP 仍需为具体 attention 和 pipeline schedule 提供
显式 hook。

在仓库根目录和已有依赖的 `hyper-parallel-env` 容器内运行。

单进程：

```bash
python -m experiments.qwen3_moe_framework.train --steps 3
```

TP2：

```bash
torchrun --standalone --nproc_per_node=2 \
  -m experiments.qwen3_moe_framework.train --tp-size 2 --steps 3
```

DP-shard2 × EP2：

```bash
torchrun --standalone --nproc_per_node=2 \
  -m experiments.qwen3_moe_framework.train \
  --dp-shard-size 2 --ep-size 2 --steps 3
```

外部配置与恢复（只读配置，参数仍随机初始化）：

```bash
python -m experiments.qwen3_moe_framework.train \
  --model-config /path/to/Qwen3-MoE/config.json --seq-len 128 --steps 2 \
  --checkpoint /tmp/qwen3-experiment

python -m experiments.qwen3_moe_framework.train \
  --model-config /path/to/Qwen3-MoE/config.json --seq-len 128 --steps 3 \
  --resume /tmp/qwen3-experiment
```

完整配置可能需要大量内存。本实验从 CPU 构造后搬至设备，不能替代生产 meta
路径；不会自动缩减模型或修改配置。合成数据仅用于验证，不代表真实语料训练质量。

`LLMTUNER_TEST_DEVICE=cpu` 可运行单进程 CPU CLI。分布式 mesh 必须与框架实际
设备一致；在可见 NPU 的环境中强制 CPU/Gloo 分布式会提前报错，不静默混用设备。
纯 CPU 分布式验证需使用 CPU 环境。

## 9. 验证

```bash
python -m pytest -q tests/unit_tests/cpu/test_qwen3_moe_framework.py

torchrun --standalone --nproc_per_node=2 \
  tests/integration_tests/qwen3_moe_framework_equivalence.py --layout tp
```

集成脚本支持 `--layout tp|dp|ep|joint`，比较连续三步与第二步保存后恢复的结果：
完整模型和 Adam moments 逐位一致，loss 容差 `atol=rtol=1e-6`。TP 另比较未分片
模型的初始 logits，NPU BF16 容差 `5e-3`；这不等同于跨框架长程精度验收。
CPU 测试另以三个独立 CLI 进程比较连续与恢复后的完整状态。

实际命令、退出码、日志和缺口见
[执行记录](../../docs/qwen3_moe_framework_experiment.md)。
