# TorchTitan distributed 设计及其与 TorchLLMTuner parallel 的差异

## 结论

TorchTitan 的 `distributed/` 更简洁，核心原因不是整个系统的代码量更小，而是
**模型层承担了更多并行职责**：模型组件声明初始化、分片、精度和 forward 契约，
`distributed/` 主要保留 mesh、FSDP、PP 和通用通信等基础设施。

TorchLLMTuner 的 `parallel/` 除了并行算法，还承担 HF 模型适配、模型转换、
设备兼容和 checkpoint 布局转换。两者是“原生可并行模型加通用分布式底座”与
“HF 模型适配层加完整分布式系统”的差别，不能只按目录行数评价设计质量。

## 代码规模对照

按当前工作区 Python 文件统计：

| 位置 | 文件数 | 约行数 | 说明 |
| --- | ---: | ---: | --- |
| TorchTitan `distributed/` | 20 | 11,000 | 其中 `flex_shard` 约 4,400 行 |
| TorchTitan `distributed/` 排除 `flex_shard` | 15 | 6,500 | 通用分布式基础设施 |
| TorchTitan `models/common/` | 43 | 13,900 | 模型组件及大量并行数学语义 |
| TorchTitan `models/base/` | 5 | 1,600 | 模型、初始化、SPMD 和配置契约 |
| TorchLLMTuner `parallel/` | 51 | 14,900 | HF 适配、并行变换、设备兼容和恢复 |

因此，TorchTitan 的目录较小并不代表相关功能不存在，而是许多功能位于模型层。

## 1. TorchTitan：模型原生支持并行

TorchTitan 的核心链路是：

```text
模型 Module / BaseModel
  → 声明参数初始化、sharding、dtype 和 forward 契约
  → SPMD 根据声明执行 TP/EP/CP 变换
  → distributed 提供 mesh、FSDP、PP 和通用通信
```

`torchtitan/models/base/module.py` 中的 `Module` 为模型组件提供统一契约，
包括：

- `_sharding_config`：参数分片声明；
- `_param_init`：参数初始化；
- `_preserve_dtype_params`：需要保留精度的参数；
- SPMD 初始化及 buffer 处理；
- remat 区域和重计算信息。

`torchtitan/models/base/model.py` 中的 `BaseModel.parallelize()` 组织模型级
并行生命周期。模型自身知道如何初始化、分片和应用 FSDP，不需要并行层根据
模型类型重建这些语义。

因此，TorchTitan 不需要独立的通用 TP 实现目录。TP 数学语义已经落在
Linear、Attention、MoE 等模型组件中。例如
`torchtitan/models/qwen3_5/model.py` 在 attention forward 内处理 TP 输入聚合。

## 2. TP、EP、CP 的实现不全在 distributed

TorchTitan 的主要职责分布如下：

| 能力 | 主要位置 |
| --- | --- |
| TP | 模型 Linear、Attention 和 `models/base/sharding.py` |
| EP | `models/common/moe` 和各模型目录 |
| CP 输入切分 | `distributed/context_parallel.py` |
| CP attention 通信 | `models/common/attention` |
| 初始化、参数精度和 SPMD buffer | `models/base/` |
| Checkpoint | 组件和 state adapter，而非 `distributed/` |

这是一种职责划分，而不是功能缺失。通用分布式代码保持较小，是因为模型组件
直接实现了并行所需的数学和状态契约。

## 3. TorchLLMTuner：并行层还要适配外部 HF 模型

TorchLLMTuner 的典型链路是：

```text
不同 HF 模型结构
  → 探测 router、experts、attention 和 decoder 接口
  → 转换或替换为本地可并行结构
  → 执行 TP/EP/CP/FSDP
  → 维护 HF checkpoint 与本地并行布局之间的映射
```

由此产生了 TorchTitan 原生模型路径通常不需要的工作：

- `parallel/expert_parallel/hf_inspection.py` 识别不同 HF 版本的 router、
  专家布局、score 位置和返回类型。
- `parallel/expert_parallel/model_replacement.py` 将 HF MoE 转换成本地 MoE，
  同时保留参数、共享关系和路由行为。
- `parallel/tensor_parallel/checkpointing.py` 处理物理 TP 切片、optimizer/EMA
  状态及恢复。
- `parallel/fully_shard_parallel/plan.py` 处理 HF pattern、PP ownership、
  共享参数和外部引用。

TorchTitan 的模型结构由自身定义，通常不需要这一层外部模型适配。

## 4. TorchLLMTuner 包含更多实现和兼容分支

当前 `parallel/` 还包括：

- TP fused GEMM 与 functional collective 两种实现；
- native/custom FSDP 及细粒度参数组；
- EP all-to-all、torchao、domino、MC2 等 dispatcher；
- KV all-gather、Ulysses、KCP 和 DeepSeek compressor CP；
- 独立 VP、TP vocab parallel 和 VFSDP；
- NPU/HCCL 的通信、排序、精度及资源生命周期处理；
- 旧 HF 布局、checkpoint、PP 路径和公开接口兼容。

这些功能带来了真实复杂度，不全是过度设计。

TorchTitan 可以通过统一模型协议，以及不承诺 Python API、配置和 native
checkpoint 向后兼容来删除旧路径。TorchLLMTuner 当前要求保留公开接口和恢复
语义，不能直接照搬这种删除方式。

## 5. TorchTitan 同样存在复杂基础设施

TorchTitan 并非所有实现都简单：

- `distributed/fsdp.py` 处理 EP placement、HSDP、预取和共享权重；
- `distributed/parallelism_context.py` 负责多种 mesh 视图；
- `models/base/module.py` 处理初始化、SPMD、dtype 和 remat；
- `models/common/` 包含大量模型及并行数学实现。

两边都需要处理 mesh、精度、生命周期和模型差异。TorchTitan 的优势是通过统一
模型契约，把这些复杂度放在更稳定、更接近其所有者的位置。

## 对 TorchLLMTuner 的启示

应借鉴职责边界，而不是简单删除代码：

1. 让模型逐步拥有初始化、专家布局、精度及拓扑约束。
2. 并行层只消费契约，不再反向理解每类 HF 模型。
3. 通用通信、mesh、FSDP 和 checkpoint 保留在并行基础设施。
4. 新模型优先走显式实现，避免继续扩张 HF inspection 和转换分支。
5. 老 HF 路径逐步收敛，在兼容要求未解决时不强行照搬 TorchTitan。

最终目标不是让 `parallel/` 的行数最小，而是形成稳定边界：

```text
模型提供结构和数学契约
并行层执行分片、通信和包装
训练层负责生命周期与执行编排
```

## 参考位置

- `torchtitan/models/base/module.py`
- `torchtitan/models/base/model.py`
- `torchtitan/models/base/sharding.py`
- `torchtitan/models/common/`
- `torchtitan/distributed/fsdp.py`
- `torchtitan/distributed/parallelism_context.py`
- `llmtuner/parallel/expert_parallel/hf_inspection.py`
- `llmtuner/parallel/expert_parallel/model_replacement.py`
- `llmtuner/parallel/fully_shard_parallel/plan.py`
- `llmtuner/parallel/tensor_parallel/checkpointing.py`
