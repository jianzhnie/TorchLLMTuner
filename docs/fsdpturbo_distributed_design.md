# FSDPTurbo 分布式设计与 TorchLLMTuner 装配优化

## 1. 结论与范围

FSDPTurbo 的主要模式是：**配置选择模块，模型提供约定，并行函数执行变换**。
它通过固定入口串联各项能力，因此装配代码较直接。TorchLLMTuner 还承担模型适配、
参数引用迁移、ownership、PP 分段和 AC/compile 生命周期协调，职责范围更广。
不能仅凭目录大小或代码行数判断设计优劣。

本文基于当前工作区的静态审查，记录日期为 2026-11-10。HEAD 分别为
TorchLLMTuner `fea434f`、FSDPTurbo `0dbd806`；未提交修改不由 HEAD 唯一描述。
本文不构成新的 NPU 正确性或性能验收。整体能力对比见
[FSDPTurbo 对比文档](fsdpturbo_comparison.md)。

## 2. FSDPTurbo 使用的设计模式

| 模式 | 实现入口 | 作用与代价 |
| --- | --- | --- |
| 门面与固定流水线 | [FSDPTurbo](../../FSDPTurbo/fsdp_turbo/fsdp_turbo.py) | 单一入口表达变换顺序；顺序本身成为隐含契约 |
| 配置驱动变换 | [配置](../../FSDPTurbo/fsdp_turbo/fsdp_turbo_config.py)与模块 pattern | 配置决定位置，函数执行操作；模型命名与配置必须匹配 |
| 策略模式 | [FSDPStrategy](../../FSDPTurbo/fsdp_turbo/distributed/fine_grained_fully_shard/strategy.py) | 选择 native/custom FSDP，隔离实现差异 |
| 单例上下文 | [ParallelState](../../FSDPTurbo/fsdp_turbo/distributed/parallel_state.py) | 统一访问 mesh、group、rank；同时引入隐式全局依赖 |
| 实例级行为替换 | [EP](../../FSDPTurbo/fsdp_turbo/distributed/expert_parallel/expert_parallel.py) | 绑定 expert forward 和 dispatcher；依赖专家容器的结构约定 |

当前主入口依次处理并行状态、量化、TP、TP 元信息、CP、模块补丁、EP、VP、
pre-FSDP hook、重计算、激活交换、Chunk Batch 和 FSDP。
部分专家 FSDP 在 EP 阶段完成，不能将其概括为所有参数最后统一包装。

## 3. 为什么装配更简单

### 3.1 TP 更接近 PyTorch 原语的直接组合

[TP 实现](../../FSDPTurbo/fsdp_turbo/distributed/tensor_parallel/tensor_parallel.py)
匹配模块，构造 `ColwiseParallel`、`RowwiseParallel`、`SequenceParallel`，然后调用
`parallelize_module`；额外 layout 参数由模型配置类提供。

TorchLLMTuner 还需处理融合投影的序列边界、MoE/shared expert 布局以及 checkpoint
转换语义。这些职责可以进一步归位，但不能通过删除检查消除。

### 3.2 EP 更依赖模型事先满足专家接口

FSDPTurbo 的 EP 根据选中模块建立本地专家信息、分片并绑定 dispatcher。
TorchLLMTuner 的[模型替换](../llmtuner/parallel/expert_parallel/model_replacement.py)
还处理专家结构转换、参数映射及别名保留。

值得借鉴的是让模型直接定义专家打包和数学顺序，使并行层消费明确契约；
不是让并行层默认所有模型都具有同一种专家结构。

### 3.3 FSDP 主要消费显式计划

[FSDP 实现](../../FSDPTurbo/fsdp_turbo/distributed/fully_shard_parallel/fully_shard_parallel.py)
根据配置选择单元、忽略项、策略、hook 和预取关系，再包装单元与根模块。
TorchLLMTuner 同时支持 decoder 自动装配与 pattern 模式，还需处理共享参数、
独立 dtype 和 dense/sparse mesh 的 ownership。

其中 [FSDPPatternReferences](../llmtuner/parallel/fully_shard_parallel/plan.py)
承担变换前后引用的维护。减少这种协调的方向，是让模型转换提供明确映射，
而不是在替换后继续猜测参数路径。

### 3.4 两者的装配范围不同

所审查的 FSDPTurbo 主入口与 parallel state 不承担 TorchLLMTuner 当前的 PP
分段运行时及统一 compile 阶段；这不代表仓库其他实验不存在相关能力。

TorchLLMTuner 的[装配入口](../llmtuner/parallel/parallelize.py)还需协调 PP 切分、
本地模型分段、引用重绑及失败传播。
[阶段定义](../llmtuner/parallel/stages.py)为：

```text
TP → EP → CP → AC → activation swap → compile → FSDP
```

dense 与 PP 本地分段已复用同一阶段执行逻辑，不应再创建第二套 runner。
阶段顺序也不表示任意特性组合都受支持。

### 3.5 简洁也来自复杂度的位置不同

FSDPTurbo 将部分复杂度放在模型、配置、dispatcher、patch 和全局并行状态中。
其 distributed 也包含细粒度 FSDP、CP 和专用通信实现。入口短不等于系统没有
复杂度；比较应沿调用链核查，而不是只统计装配文件。

## 4. 不能直接照搬的行为

- **冲突检测**：当前 FSDPTurbo `get_parallel_style` 使用 `if/elif` 匹配三类策略，
  随后的 `len(parallel_style) > 1` 无法检测跨类别重叠，实际选择首个类别。
  这是静态发现，本文未执行运行时复现；TorchLLMTuner 应保留显式冲突检查。
- **策略兼容**：FSDP adapter 对不支持的 `hook_module` 可以省略传参。
  借鉴接口统一时，需判断省略是否改变调用方语义，不能直接复制静默忽略行为。
- **单例状态**：便于调用，但不利于测试隔离和多拓扑共存；保留现有显式拓扑传递。
- **装配顺序**：CP/EP、AC、交换与 FSDP 的相对位置依赖模型和生命周期，不能机械迁移。
- **安全边界**：共享参数、collective 顺序、rank 失败协调和恢复检查应保留。
  静态代码差异不足以证明任一框架整体更正确或更快。

## 5. TorchLLMTuner 可执行改进方向

| 优先级 | 改进 | 验收边界 |
| --- | --- | --- |
| P0 | 模型拥有层栈、初始化、专家布局、score 乘法顺序与状态映射 | 并行层减少模型识别；参数身份、状态键与数学语义保持一致 |
| P0 | 先解析 TP/EP/FSDP 计划，再变换模型 | 修改前完成匹配、冲突和 ownership 校验；本地失败不导致 collective 不匹配 |
| P0 | 稳定模型接入契约 | 明确 decoder、embedding、head、共享参数与初始化接口 |
| P1 | 转换结果提供显式模块和参数映射 | 引用重绑不依赖模糊路径猜测，兼容 PP 与 checkpoint |
| P1 | 配置选择模块，设备实现留在 accelerator | 模型保留数学顺序，设备限制提前暴露，不静默回退 CPU |
| P1 | 保持窄策略接口与装配可观测性 | rank0 报告匹配数量、构造选择与回退原因；kernel 执行由 trace 证明 |
| P1 | 显式资源生命周期 | swap、预取、异步保存具有 owner、wait、release、close 与异常传播 |
| P2 | 兼容逻辑归属叶子模块 | 避免重复版本判断，不引入新的通用管理层 |

每批只处理一个职责，结构迁移与行为修复分别验证。先完成定向及全量 CPU、Ruff、
格式、compileall 和空白检查，再对受影响的真实 NPU 正反向、并行、AC/compile
及独立恢复进行验收。纯迁移检查确定性路径逐位一致；算子或归约变化预设容差。

性能改动须另做无诊断导出的重复基准，记录最慢 rank 时间、p50/p90、有效 TPS
与资源峰值。已有小模型结果、启动成功或静态结构相似，均不能替代目标模型验收。
