# 从模型定义到分布式执行：三种训练框架的架构比较与 TorchLLMTuner 重设计

## 引言：真正困难的不是“把参数切开”

在大模型训练中，TP、FSDP、EP、CP 和 PP 常被理解为几种独立的并行技术。
但在真实系统里，最困难的工作并不是调用一次 `Shard(0)` 或 all-reduce，
而是回答一组与模型语义紧密相关的问题：

- 这个参数代表什么，应该沿哪个轴切分？
- 切分之后，输入和输出分别是什么布局？
- 哪个模块负责发起通信，通信结果由谁消费？
- router、shared expert、recurrent state 和 mask 是否需要特殊处理？
- 参数被替换后，共享权重、optimizer 和 checkpoint 如何继续对应？

这些问题无法完全脱离模型定义。分布式框架设计的核心，也因此不是简单选择
一种并行算法，而是决定：**模型知识由谁表达，分布式系统如何消费这些知识。**

FSDPTurbo、HyperParallel 和 TorchTitan 给出了三种不同答案。本文比较它们的
设计边界，并在此基础上提出一套适合 TorchLLMTuner 的目标架构。

## 一、如何评价模型与分布式模块的耦合

只看 import 数量，很容易误判架构。模型与分布式之间至少存在四种耦合：

| 耦合层次 | 典型表现 | 风险 |
| --- | --- | --- |
| 代码依赖 | distributed 直接 import 具体模型类 | 循环依赖、启动成本高、模型无法独立演进 |
| 结构约定 | 依赖模块名、参数名、层数和 FQN | 命名不同就无法复用，自动匹配可能漏报 |
| 数学语义 | router、专家布局、score 乘法、mask 和状态语义不同 | 形状正确但结果错误，且难以发现 |
| 运行时生命周期 | forward 替换、hook、AC/compile、checkpoint 顺序 | 安装顺序改变行为，恢复后出现隐蔽错误 |

一个框架可以做到几乎没有模型 import，却仍然高度依赖模型的结构和数学约定。
因此，下面同时讨论代码依赖和语义边界，而不以目录大小或文件行数判断优劣。

## 二、FSDPTurbo：配置驱动的直接装配

### 2.1 基本设计

FSDPTurbo 的主要路径可以概括为：

```text
配置 + 模块 pattern
  → 选择 TP / EP / CP / FSDP 目标
  → 并行函数直接修改模块、参数或 forward
```

模型需要提前提供与并行实现匹配的层、专家容器和配置字段。TP 通过
`ColwiseParallel`、`RowwiseParallel`、`SequenceParallel` 和 `parallelize_module`
组合 PyTorch 原语；EP 为选中模块建立专家分片并绑定 dispatcher；FSDP 根据配置
确定包装单元、策略、hook 和预取关系。

它的优势是调用链直观：配置决定位置，固定入口决定顺序，并行函数完成变换。
阅读装配入口时，不需要先理解一套复杂的计划生成体系。

### 2.2 去耦合方式与局限

配置 pattern 和策略接口减少了模型类型分支，但模型仍然需要满足隐式的属性名、
专家布局和 forward 约定。实例级 forward 替换还会让行为依赖安装顺序。
`ParallelState` 单例简化了 mesh 获取，却引入全局状态和测试隔离成本。

所以，FSDPTurbo 的简洁主要来自**直接装配**，并不意味着模型与分布式之间没有
语义依赖。其部分复杂度位于模型、配置、dispatcher 和 patch 中，而非装配入口。

### 2.3 值得借鉴什么

- 用固定入口表达清晰的变换顺序；
- 用配置选择目标，避免装配层堆叠模型类型分支；
- 用窄策略接口隔离 native/custom 等实现差异。

不应照搬全局单例、静默兼容和隐式 forward 覆盖。共享参数、冲突检查、
collective 顺序和恢复语义不能为了入口简短而省略。

## 三、HyperParallel：adapter、plan 与 builder

### 3.1 基本设计

HyperParallel 将模型接入拆成三个阶段：

```text
模型 adapter
  → ModelAdapterSpec 声明能力
  → ShardingPlanner 生成 ShardingPlan
  → builder 执行参数分片、特殊处理和 forward 改写
```

`ModelAdapterSpec` 描述模型族的 replacement、attention、CP/EP provider、
sharding rules、FSDP 单元、执行顺序、recompute、loss 和 checkpoint。
registration/provider 延迟发现模型，避免加载 registry 时导入所有模型实现。

`ShardingPlan` 是解析与执行之间的中间表示，记录参数 placement、输入输出
layout、通信边界、local compute、inner wrapper、特殊 handler 和 tied weights。
builder 消费计划，而不是在每个执行函数里重新决定模型语义。

### 3.2 去耦合方式与局限

这套设计有三个明显收益：

1. **依赖方向清晰**：通用 distributed 很少直接导入具体模型类；
2. **解析与执行分离**：计划成为可以检查和验证的中间对象；
3. **专用能力有归属**：模型 adapter 提供 CP/EP/FSDP 策略和精确模块边界。

但它并没有消除模型知识。planner 仍会根据 FQN、attention/MLP/MoE 名称、
参数角色和配置推导结构；EP routing 保留部分模型族路由别名；KDA CP 还在通用
目录中处理多套 Kimi projection、recurrent state 和输出形式。

换句话说，**低 import 耦合与较高的结构、数学语义耦合可以同时存在。**
forward 改写也要求系统严格理解参数、mask/cache、返回结构和安装顺序。

### 3.3 值得借鉴什么

- 用 adapter 统一描述模型能力，并延迟加载模型实现；
- 用纯计划对象隔离“准备怎样切分”和“实际修改模型”；
- 特殊区域允许 `derive=False`，放弃自动推导，改为完全显式声明；
- 通用 EP 只提供 dispatch、collective 和执行骨架，模型 adapter 组合具体语义。

不应继续扩大自动 FQN 推断，也不能把 apply-time 属性存在性检查当成数值
正确性证明。

## 四、TorchTitan：模型原生并行契约

### 4.1 基本设计

TorchTitan 选择让模型承担更多并行职责。模型组件通过 `Module` / `BaseModel`
声明：

- 参数初始化与 buffer 重建；
- sharding config；
- 需要保留精度的参数；
- SPMD 输入输出和状态；
- remat 区域与模型级并行生命周期。

TP 数学位于 Linear、Attention 和 MoE 中，EP 位于模型的 MoE 组件，CP attention
也靠近模型 attention。`distributed/` 因而主要保留 mesh、FSDP、PP 和通用通信。

这不是功能减少，而是职责位置不同：模型原生定义并行所需的数学语义，通用层
不再需要反向识别每类模型。

### 4.2 去耦合方式与局限

模型初始化、dtype、buffer 和 forward 契约靠近其所有者，通用分布式只消费
稳定协议，checkpoint 通过 state adapter 隔离外部权重格式。这是三者中“模型
拥有数学语义”最彻底的路线。

代价是统一模型协议本身具有复杂度。`models/common/` 和基类仍包含大量初始化、
SPMD、attention、MoE 与并行数学。目录更小不等于系统更简单，也不意味着应该
把差异很大的外部模型强行塞进同一个 BaseModel。

### 4.3 值得借鉴什么

- 让模型拥有初始化、专家布局、精度和状态契约；
- 让并行层消费契约，而不是反向理解模型；
- 将外部 checkpoint 格式与训练模型的内部状态分离。

不必复制完整 BaseModel 体系，更不能为了统一抽象而合并 dense/MoE、文本/
多模态或不同 recurrent/attention 的近似语义。

## 五、三种设计的本质差异

| 维度 | FSDPTurbo | HyperParallel | TorchTitan |
| --- | --- | --- | --- |
| 模型知识主要位置 | 模型约定、配置和专用实现 | adapter 与 planner | 模型组件和模型契约 |
| 解析结果 | 配置与 pattern | 显式 ShardingPlan | 模型 sharding 声明 |
| 执行方式 | 固定入口直接变换 | builder 执行计划 | SPMD 与 distributed 执行 |
| 模型发现 | 配置和模块匹配 | 延迟注册 adapter | 统一模型体系 |
| 主要优点 | 装配直接、调用链短 | 解析与执行分离、专用能力可插拔 | 模型数学所有权清晰 |
| 主要风险 | 隐式约定、全局状态、forward 替换 | 自动推导与运行时语义耦合 | 协议较重、迁移外部模型成本高 |

对 TorchLLMTuner 最有价值的不是选择其中一条路线，而是组合三者优势：

- 用 TorchTitan 的**模型所有权**消除并行层对模型结构的反向理解；
- 用 HyperParallel 的**显式计划**隔离解析与修改；
- 用 FSDPTurbo 的**直接装配**保持执行流程简单；
- 保留 TorchLLMTuner 的显式拓扑、NPU 支持、失败协调和恢复语义。

## 六、TorchLLMTuner 需要解决的真实问题

当前 `parallel/` 同时承担 HF 模型识别与转换、TP/EP/CP/PP/FSDP 执行、参数引用
和 ownership、checkpoint 恢复、NPU/HCCL 适配，以及 AC/compile/swap 生命周期。
因此，不可能只靠删除文件获得 TorchTitan 式的简洁。

真正需要减少的是：

1. 对模型名称、类名和候选属性的猜测；
2. 模型转换后的路径重绑与隐式引用修复；
3. 同一职责在模型、parallel 和 Trainer 中的重复表达；
4. 新模型继续复用通用 HF inspection 的惯性。

与此同时，共享权重、collective 顺序、参数身份、RNG、checkpoint 和恢复检查
属于正确性契约，不能当成冗余代码删除。

## 七、目标架构：模型契约 + 纯计划 + 直接执行

### 7.1 总体原则

目标架构可以概括为一句话：

```text
模型知道自己在算什么
计划说明准备怎样切分
并行层知道如何执行
训练层只负责训练和恢复
```

具体遵循七条原则：

1. **模型拥有数学**：结构、初始化、router、专家布局、状态映射和特殊约束。
2. **计划与执行分离**：先形成计划并完成校验，再修改模型。
3. **依赖显式传递**：topology、mesh、资源和错误 group 不使用全局单例。
4. **契约保持最小**：从真实调用提炼，不预设庞大基类、模型 DSL 或策略管理器。
5. **设备与模型分离**：模型表达数学顺序，accelerator 提供设备实现。
6. **兼容有退出条件**：旧 HF 路径按模型迁移，不成为永久默认。
7. **正确性优先于抽象**：没有真实复用收益时不建立共享框架层。

### 7.2 目录职责

```text
llmtuner/
├── config/
│   ├── model.py                    # 用户模型与任务配置
│   └── parallel.py                 # 并行度、策略和静态校验
├── models/
│   ├── registry.py                 # 唯一架构元数据与延迟构造入口
│   ├── contracts.py                # 确有多个消费者时才建立共享类型
│   ├── common/                     # 数学和生命周期完全一致的共享组件
│   └── <family>/<structure>/
│       ├── configuration.py        # 本地配置与外部字段映射
│       ├── model.py                # 层栈、forward、初始化、buffer
│       ├── contract.py             # 结构、布局、并行语义和能力约束
│       └── state.py                # 外部状态键与张量布局映射
├── accelerator/
│   ├── capabilities.py             # 能力检测与明确支持边界
│   └── ops/                        # NPU/CUDA 通用设备算子
├── parallel/
│   ├── topology.py                 # 并行度和 mesh，无模型知识
│   ├── plan.py                     # 纯数据执行计划与诊断信息
│   ├── resolver.py                 # 目标、冲突、ownership、PP 绑定
│   ├── assemble.py                 # 唯一装配入口和阶段执行
│   ├── tensor_parallel/
│   ├── expert_parallel/            # 分片、dispatch 和 collective
│   ├── context_parallel/
│   ├── pipeline_parallel/
│   ├── fully_shard_parallel/
│   └── vocab_parallel/
├── memory/                         # AC、swap 与资源生命周期
├── components/
│   ├── optimizer/
│   └── checkpointer/               # DCP、异步发布与完整恢复
└── trainer/                        # 唯一训练编排和资源关闭入口
```

目录按职责形成，不要求每个模型拥有相同数量的文件。小型模型的 contract 可以
留在 `model.py`；`contracts.py` 只在确有多个消费者时创建，避免形成转发层。

### 7.3 最小模型契约

不需要庞大的 BaseModel。模型实例提供以下信息：

| 契约 | 模型需要回答的问题 |
| --- | --- |
| 构造与初始化 | 如何构造、物化和初始化？buffer 与 RNG 顺序是什么？ |
| 结构 | embedding、blocks、final norm、head 在哪里？稳定 block ID 和 FQN 是什么？ |
| 输入输出 | token、position、mask、padding、logits 和辅助 loss 如何表达？ |
| 专家数学 | score 如何计算？何时相乘？packed half 如何排列？shared expert 如何合并？ |
| 并行语义 | TP 输入输出是什么 layout？专家轴、CP 语义、FSDP 单元和 dtype 组是什么？ |
| 状态映射 | 外部键、转置、打包、共享别名和格式版本如何映射？ |
| 能力约束 | 支持哪些 dtype、拓扑、AC/compile、cache 和设备后端？ |

简单能力可以从属性开始，例如独立 dtype 参数名、CP backend、router 输出类型。
只有出现多个真实消费者后，才提升为共享 dataclass 或 Protocol。

### 7.4 ParallelPlan：可检查的中间表示

ParallelPlan 只保存模块引用和元数据，不复制张量，至少包含：

- `targets`：明确的模块、参数及规范 FQN；
- `layouts`：参数与激活的分片、复制、partial 和输出语义；
- `ownership`：rank、PP stage、共享参数与外部别名；
- `operations`：按顺序执行的 TP/EP/CP/FSDP 变换；
- `constraints`：dtype、设备、拓扑、AC/compile 和恢复限制；
- `diagnostics`：匹配数量、未命中规则、构造选择和拒绝原因。

模型声明和用户 override 可以共同生成计划，但 override 必须先解析为明确目标。
执行阶段不得重新搜索候选属性，也不得根据模型名称选择另一套数学。

### 7.5 统一装配流程

```text
配置与架构 ID
  → 静态配置、设备和组合校验
  → meta 构造及模型契约
  → 全局计划解析
  → 冲突、ownership、共享参数和能力校验
  → 协调相关 rank 的本地失败
  → PP 切分及本地计划重绑（如启用）
  → TP → EP → CP → AC → swap → compile → FSDP
  → 按既有生命周期物化、初始化或恢复
  → optimizer / scheduler / Trainer
```

整网与 PP 分段使用同一执行器。阶段顺序是生命周期契约，不表示任意组合自动
受支持。预校验可以减少半改模型，但不提供事务回滚；collective 失败按分布式
边界退出，并由完整 checkpoint 恢复。

参数替换需要显式旧→新映射，optimizer 在布局稳定后建立。模型 state adapter
只负责外部格式，native DCP 使用稳定 FQN 和显式布局。

## 八、一个 MoE 模型应该如何接入

假设接入一个新的 MoE 模型，理想流程不是先修改通用 EP inspection，而是：

1. 在模型目录定义层栈、router、专家参数和 forward；
2. 声明有序 blocks、专家轴、packed half 和 shared-expert 组合；
3. 声明 TP 输入输出 layout、EP ownership 和 CP 限制；
4. 声明独立 dtype 参数及状态键映射；
5. resolver 将声明转换为 ParallelPlan；
6. parallel 执行参数分片、dispatcher 和 collective；
7. 用单卡正反向、EP 等价、独立恢复和真实 NPU trace 验收。

在这个过程中：

- 模型不导入 Trainer；
- 通用 dispatcher 不根据模型名称决定 score 函数；
- checkpoint 不通过搜索参数形状恢复专家语义；
- parallel 不需要知道该类模型的名称。

这才是模型与分布式模块真正意义上的解耦：不是依赖消失，而是依赖被压缩为
一组明确、可验证的契约。

## 九、如何保持代码清晰、简洁和可读

### 9.1 控制抽象数量

一个文件只承担一个明确职责。一次性 helper 就近保留，不为两行代码新建工具
模块。只有数学语义和生命周期完全一致时才共享实现，不为“看起来统一”引入
策略管理器、注册表或模型 DSL。

### 9.2 保持依赖方向稳定

模型不依赖 Trainer；通用 distributed 不导入具体模型类；adapter 不复制通用
collective。通用计划由模型声明生成，实际执行仍归 parallel。配置和注册信息
各自只有一个事实来源。

### 9.3 让隐式行为变得可见

不使用全局 monkey patch，不隐式叠加 forward wrapper。重复安装要么幂等，
要么明确拒绝；冲突在安装前检查。rank0 报告请求值、构造选择和匹配数量，
实际 kernel 则由 trace 证明。

### 9.4 把生命周期写进接口

swap、预取和异步 checkpoint 明确资源 owner、wait、release、close 和异常传播。
版本兼容集中在所属叶子模块，优先能力检测，不把版本字符串判断扩散到 Trainer。

### 9.5 保持迁移纪律

每批只迁移一个完整职责，结构迁移与行为修复分开。新模型优先显式实现，旧 HF
路径按模型退出。不批量改名，不为了目录对齐移动同一套跨层依赖，未确认公开
入口和动态调用前不删除兼容代码。

## 十、验收：如何判断架构真的变好了

架构改进不能用文件数量或 import 成功证明，至少需要以下结果：

- 新模型接入不需要修改通用 parallel 的模型识别分支；
- 修改模型前已有明确计划，冲突和 ownership 可以测试；
- 参数身份、共享关系、状态键、RNG 和初始化顺序可以解释；
- 确定性纯迁移的 logits、loss 和梯度逐位一致；
- 适用的 TP/EP/PP/CP、AC/compile 和独立恢复通过；
- 真实 NPU 正反向、空专家、非对称路由和 rank 本地失败得到验证；
- 性能报告包含最慢 rank、p50/p90、有效 TPS、HBM/Host 峰值和通信；
- CPU、mock、skip、import 或启动成功不被当成 NPU 验收。

## 结语

FSDPTurbo 展示了直接装配的价值，HyperParallel 展示了 adapter 与计划中间层
的价值，TorchTitan 展示了模型原生契约的价值。更好的 TorchLLMTuner 不需要
复制其中任何一个框架，而应建立一条更短、更明确的因果链：

```text
模型定义数学
契约描述结构与约束
计划描述切分与通信
并行层执行计划
训练层管理训练与恢复
```

当模型知识只表达一次、计划在修改前可以检查、设备实现不再混入模型语义时，
代码的清晰、简洁和可读性才会成为自然结果。

## 参考文档

- [FSDPTurbo 分布式设计](fsdpturbo_distributed_design.md)
- [HyperParallel distributed 设计与耦合](hyper_parallel_distributed_design_review.md)
- [TorchTitan distributed 设计对比](torchtitan_distributed_design_comparison.md)
