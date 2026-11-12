# TorchLLMTuner 模型与分布式架构设计及重构路线图

本文合并模型与分布式架构指南、重构设计和实施路线，依次说明设计依据、目标边界、
当前差距、迁移批次与验收要求。核心问题是：**模型知识由谁表达，并行层如何执行，
训练与恢复如何保持一致？**

## 阅读导航

- [TorchLLMTuner 模型与分布式架构设计及重构路线图](#torchllmtuner-模型与分布式架构设计及重构路线图)
  - [阅读导航](#阅读导航)
  - [1. 设计依据：如何评价解耦](#1-设计依据如何评价解耦)
  - [2. 框架经验与设计取舍](#2-框架经验与设计取舍)
    - [TorchLLMTuner 需要解决的问题](#torchllmtuner-需要解决的问题)
  - [3. 目标与原则](#3-目标与原则)
  - [4. 当前状态与差距](#4-当前状态与差距)
  - [5. 代码职责与依赖](#5-代码职责与依赖)
  - [6. 接口与装配约束](#6-接口与装配约束)
    - [模型契约](#模型契约)
    - [ParallelPlan](#parallelplan)
    - [装配与生命周期](#装配与生命周期)
  - [7. 模型接入与工程约束](#7-模型接入与工程约束)
    - [一个 MoE 模型的接入流程](#一个-moe-模型的接入流程)
    - [控制复杂度的工程约束](#控制复杂度的工程约束)
      - [控制抽象数量](#控制抽象数量)
      - [保持依赖方向稳定](#保持依赖方向稳定)
      - [让隐式行为变得可见](#让隐式行为变得可见)
      - [把生命周期写进接口](#把生命周期写进接口)
      - [保持迁移纪律](#保持迁移纪律)
  - [8. 实施批次与出口门槛](#8-实施批次与出口门槛)
  - [9. 兼容与验收](#9-兼容与验收)
  - [10. 完成标准与交付记录](#10-完成标准与交付记录)

## 1. 设计依据：如何评价解耦


只看 import 数量，很容易误判架构。模型与分布式之间至少存在四种耦合：

| 耦合层次 | 典型表现 | 风险 |
| --- | --- | --- |
| 代码依赖 | distributed 直接 import 具体模型类 | 循环依赖、启动成本高、模型无法独立演进 |
| 结构约定 | 依赖模块名、参数名、层数和 FQN | 命名不同就无法复用，自动匹配可能漏报 |
| 数学语义 | router、专家布局、score 乘法、mask 和状态语义不同 | 形状正确但结果错误，且难以发现 |
| 运行时生命周期 | forward 替换、hook、AC/compile、checkpoint 顺序 | 安装顺序改变行为，恢复后出现隐蔽错误 |

一个框架可以做到几乎没有模型 import，却仍然高度依赖模型的结构和数学约定。
因此，下面同时讨论代码依赖和语义边界，而不以目录大小或文件行数判断优劣。

## 2. 框架经验与设计取舍

三种框架采用不同的复杂度分配方式，并不存在从一个框架向另一个框架的必然演进关系。

- **FSDPTurbo**：以配置和模块匹配驱动固定装配流水线，调用关系直接；模型需满足
  专家容器、布局和 forward 约定，单例并行状态也带来测试隔离成本。
- **HyperParallel**：通过模型 adapter、ShardingPlan 和 builder 分离声明、解析和执行；
  延迟注册降低直接依赖，但 planner 的名称推导、EP 路由和专用 CP 数学仍有语义耦合。
- **TorchTitan**：让模型组件拥有初始化、分片、精度和状态契约，通用分布式层消费这些
  声明；代价是模型协议和外部模型迁移成本。不能仅凭 distributed 目录大小评价复杂度。


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

### TorchLLMTuner 需要解决的问题


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

## 3. 目标与原则

生产调用链收敛为：

```text
显式模型契约 → 纯计划解析与校验 → 固定阶段执行 → 统一训练与恢复
```

优先完成 Qwen3-MoE 的生产闭环，再逐模型迁移 DeepSeek V4、Kimi K3、Qwen3.5。
复用现有拓扑、collective、内存、optimizer 和 checkpoint，不建立第二套训练栈。

不按行数或文件数评价进度，不因目录迁移重写已验证数学，不以实验模型代替
生产迁移，也不预设吞吐提升。

## 4. 当前状态与差距

分别记录实现与验证：实现分为已完成、部分完成、待实现；验证分为已验收指定
子集、待验证。明确拒绝属于支持范围，不属于验证结果。代码存在、配置允许和
目标拓扑通过验收是三件不同的事。

| 工作项        | 当前已实现                                                                  | 尚未完成                                    |
| ---------- | ---------------------------------------------------------------------- | --------------------------------------- |
| 注册与构造      | `registry.py` 唯一元数据、延迟构造、任务匹配；离线/外部配置返回架构校验；普通配置不要求 HF attribute_map | 按调用方退出旧兼容 API；未注册模型必须显式拒绝           |
| 模型结构       | `DecoderStructure`、`MoEBlockContract`；多个显式模型声明 decoder/MoE slot、实际共享权重 | 各模型完整数学、能力、初始化与状态格式契约                   |
| Qwen3      | 自有整网/decoder forward、初始化入口、TP 声明及权重/bias/norm/vocab 几何校验；PP 按阶段 ownership 校验 | 底层参考组件仍来自 Transformers；随机 meta TP/EP/PP/CP 初始化明确拒绝，目标 NPU 验收未完成 |
| Qwen3.5 | 自有整网 logits/loss/输出组装、GDN/full-attention 构造和 forward、routed/shared expert、native router 数学及 MoE 初始化规则；模型预检校验 attention/GDN/MoE/vocab 存储几何 | GDN kernel 分派和部分参考组件仍依赖 Transformers；无自有 PP stage I/O；通用 EP 明确拒绝 gated shared expert；完整状态映射及目标 NPU 验收未完成 |
| DeepSeek V4 | 自有 LM/decoder、attention、CSA/HCA/indexer、router、专家、共享 MLP、hyperconnection、norm/RoPE 和显式初始化；CSA/indexer 复用无状态 overlap 布局；meta/CP 绑定本地类型 | 仍继承参考类型并复用 cache/capture/kernel 基础设施；并行能力与状态映射契约、真实 NPU/collective 和独立恢复未完成；CPU 子集通过不等于目标验收 |
| Kimi EP 模型契约 | gate 自有 live routing 校验；SparseMoE 提供 grouped 激活校验/构造与数学快照；保留实际路由、SituGLU 和 latent norm 语义 | 完整状态格式、并行计划与目标 NPU/独立恢复验收；已通过 CPU 子集不可外推 |
| TP 计划      | `ParallelPlan`、目标绑定与失效检查；Qwen3/Qwen3.5 模型自有 TP 计划；attention/shared-expert 识别前移 | DeepSeek/Kimi 计划、完整跨阶段布局与 ownership |
| EP 计划 | `resolve_ep_plan` 绑定路由与 packed 专家；Kimi 后端、几何、存储及 alias 预检前移；模型声明实际激活与数学快照，执行拒绝过期语义 | 旧专家适配退出、完整跨阶段 ownership 和真实 EP/NPU 验收 |
| 参数映射       | TP/EP 替换回调，FSDP 引用修复                                                   | PP 及跨阶段共享参数、alias 和 ownership 的统一映射     |
| PP/CP/FSDP | 唯一 stage runner；PP 全局 TP 预检、局部重解析；Qwen3 自有 PP 阶段接口；CP 消费显式 attention 目标 | 完整全局→局部 ownership/alias 重绑、其他模型 PP 和组合验收 |
| 设备算子       | `kernel/ops`、CPU/NPU backend、共用校验，设备信息复用 accelerator                   | 其余模型直接 vendor 调用逐项迁移，compile/规模与性能矩阵    |
| 恢复         | 既有 DCP、FQN optimizer、EMA；Qwen3 小模型及 Trainer PP/PP+full AC 同步 DCP 独立恢复子集 | 真实数据游标、跨拓扑、多节点、异步失败和其余 PP 组合矩阵 |

状态基线记录中的工程门槛：全量 CPU 使用
`--continue-on-collection-errors` 实际执行得到 **7118 passed、52 skipped、
8 collection errors**，退出码 1；不能标记为全量通过。剩余错误来自旧 HF wrapper、
Flex、GDN 自定义策略、QKV、pre-down 专家、RoPE 精度和 residual norm 测试。
应迁移有效覆盖并补齐缺失的模型自有能力，不恢复通用 HF 探测，也不删除或 skip
旧测试来制造通过。日志与精确命令见迁移记录。

已有验证证据：

- [显式模型迁移记录](explicit_model_runtime_migration.md)：当前生产 Qwen3 的
  CPU/Gloo TP、EP、CP、TP+CP、PP 与独立恢复子集，含数值口径、命令和日志。
  TP+CP 已覆盖四 rank、两种 CP 策略和三步更新；不代表 NPU 或目标规模验收。
- [EP 计划验证记录](ep_plan_refactoring_validation.md)：定向测试、全量 CPU 和
  真实双 NPU 三步等价对照，适用范围及 fallback 警告见记录。
- 该记录对应当时的工作区快照，不自动覆盖后续能力预检改动。新增修改需重新
  验证；未附运行记录的支持项按待验证处理。


[Qwen3-MoE 实验](../experiments/qwen3_moe_framework/README.md)仅用于验证纵向设计，
不得把实验 Trainer、数据或 checkpoint owner 平移为第二套生产实现。

## 5. 代码职责与依赖

沿用现有目录和同名并行入口。下列模型文件按实际职责创建，不要求机械对齐。

```text
llmtuner/
├── config/                    用户配置、别名、静态组合校验
├── models/
│   ├── registry.py            唯一架构元数据与延迟构造
│   ├── contracts.py           多消费者共享的少量契约
│   ├── common/                数学和生命周期完全一致的组件
│   ├── <family>/<model>/
│   │   ├── model.py           结构、forward、初始化与 buffer
│   │   ├── contract.py        模型约束与并行语义（按需）
│   │   └── state.py           外部键、packed/alias 与格式版本（按需）
│   └── (no HF training adapter) 生产模型只从显式家族目录构造
├── accelerator/               设备发现、能力、通信与运行时工具
├── kernel/
│   ├── ops/                   语义算子 API
│   ├── cpu/、npu/             torch 参考与设备实现
│   ├── registry.py            实现注册与设备分派
│   └── validation.py          共用输入契约
├── parallel/
│   ├── topology.py            拓扑与 mesh
│   ├── plan.py                目标、绑定与计划数据
│   ├── stages.py              唯一阶段顺序
│   ├── parallelize.py         全局预检、错误协调与阶段执行
│   └── <family>/<family>.py   各并行家族公开入口
├── memory/                    AC、activation swap、资源生命周期
├── components/                optimizer、DCP、loss、指标等训练服务
├── datasets/                  数据、packing、DP 分流与可恢复游标
└── trainer/                   唯一训练、恢复与关闭入口
```

解析器留在现有叶子模块；多个消费者确有复用时再建立 `resolver.py`，不为目录
完整新增转发层。DeepSeek 现有 `modeling.py` 等公开路径不因图示批量重命名。

| 层             | 消费的接口                 | 必须守住的边界                            |
| ------------- | --------------------- | ---------------------------------- |
| models        | 配置、共享数学、语义算子          | 不解释用户 pattern，不依赖 Trainer 或隐式进程组   |
| resolver      | 模型契约、配置、拓扑规格          | 不替换参数、不安装 forward、不运行 collective   |
| parallel      | 明确目标、布局、mesh、错误 group | 不按模型名猜 score、packed half 或 mask 数学 |
| kernel        | 张量及算子参数、设备能力          | 不识别模型层栈，不决定训练策略，不静默搬至 CPU          |
| Trainer       | 模型 I/O、执行结果和训练服务      | 不解包专家或修补内部布局                       |
| state adapter | 外部格式与模型版本契约           | 不接管 optimizer/EMA/DCP 编排           |

环境设备信息复用 `accelerator/device.py`；算子有张量输入时以输入设备分派。
显式 torch reference 选择与数据搬至 CPU 是不同操作，日志不得混淆。

## 6. 接口与装配约束

### 模型契约

优先使用实例属性、方法或少量不可变数据类型，不要求统一 BaseModel。

| 契约    | 最少表达的信息                                                        |
| ----- | -------------------------------------------------------------- |
| 构造    | 架构 ID、任务、配置、meta 物化与初始化时点、RNG 顺序                               |
| 结构    | embedding、有序 block ID/FQN、norm、head、共享参数身份                     |
| I/O   | token、position、mask、padding、hidden/logits、aux loss、PP 阶段输出     |
| 专家数学  | router score、top-k、归一化、score 乘法位置、packed halves、shared experts |
| 并行与能力 | TP/EP 轴和输出语义、CP 历史需求、dtype、后端及不支持组合                            |
| 状态    | 外部键与布局映射、alias、格式版本、buffer/cache 和 AC 重放要求                     |

声明绑定具体实例。PP 切分后按稳定全局 block ID 重绑局部引用；不得继续使用
整网引用，也不得用候选属性、类名或源码 AST 补齐显式模型缺少的声明。

### ParallelPlan

状态基线中的生产计划类型已用于 TP 和 EP；实验目录后续可能独立演进，
不假定其类型或执行器与生产实现共享。TP、EP 已有阶段计划，并不意味着
完整全局计划已经落地；下表仍包含待补齐的字段和约束。
`PlanTarget.spec` 已承载部分阶段绑定，不应扩大为任意可执行闭包。

| 维度              | 后续必须补齐的约束                                  |
| --------------- | ------------------------------------------ |
| targets/layouts | 规范 FQN、角色、shard/replicate/partial、参数轴与激活归约 |
| ownership       | TP/EP/PP 归属、共享参数组、外部 alias、非本地规则           |
| operations      | 阶段、实现、依赖与前后布局，执行不重新选择模型数学                  |
| constraints     | dtype、设备、几何、AC/compile、物化与恢复条件             |
| diagnostics     | 匹配数量、覆盖来源、未命中、拒绝原因与构造选择                    |

目标解析顺序：模型声明 → 显式用户覆盖 → 目标和冲突校验。零匹配、跨实例目标和
含糊覆盖拒绝；PP 非本地规则保留全局 ownership，不能当作普通未命中。
计划只保存引用和元数据，不复制参数 storage；解析不得消耗 RNG、分配专家
权重或修改模型。现有转换前能力预检仍属于过渡实现，最终应由解析所需的显式
配置驱动，不把执行期才发现的错误算作计划校验已完成。frozen dataclass 不冻结被引用
的模型；配置、模块或参数变化后必须重新解析，并在执行前检查失效绑定。

### 装配与生命周期

```text
配置/架构 ID → 静态校验 → meta 模型及契约
 → 全局计划解析和 rank 本地错误协调
 → 沿既有顺序建立 mesh、PP 切分并重绑局部计划
 → TP → EP → CP → AC → activation_swap → compile → FSDP
 → 沿既有时点物化、初始化或 checkpoint load
 → optimizer → scheduler → Trainer
```

这是目标职责顺序，不授权移动现有初始化、collective 或 checkpoint 时点。
本地预检的错误协调块不得包含 collective；实际分布式操作按所属 group 的错误
边界执行。预检不提供事务回滚，失败后完整重建或恢复，不在半变换模型上切后端。

TP/EP/PP 的参数替换结果应提供旧→新映射和 alias 检查，optimizer 在布局稳定后
建立。swap、预取、异步保存各自明确 owner、`wait/release/close`、Host 上限和
异常传播；先完成在途操作再释放 storage。`npu_virtual`、AdamWSwap 与 FSDP
CPU offload 保持独立，不合并开关或生命周期。

## 7. 模型接入与工程约束

### 一个 MoE 模型的接入流程


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

### 控制复杂度的工程约束


#### 控制抽象数量

一个文件只承担一个明确职责。一次性 helper 就近保留，不为两行代码新建工具
模块。只有数学语义和生命周期完全一致时才共享实现，不为“看起来统一”引入
策略管理器、注册表或模型 DSL。

#### 保持依赖方向稳定

模型不依赖 Trainer；通用 distributed 不导入具体模型类；adapter 不复制通用
collective。通用计划由模型声明生成，实际执行仍归 parallel。配置和注册信息
各自只有一个事实来源。

#### 让隐式行为变得可见

不使用全局 monkey patch，不隐式叠加 forward wrapper。重复安装要么幂等，
要么明确拒绝；冲突在安装前检查。rank0 报告请求值、构造选择和匹配数量，
实际 kernel 则由 trace 证明。

#### 把生命周期写进接口

swap、预取和异步 checkpoint 明确资源 owner、wait、release、close 和异常传播。
版本兼容集中在所属叶子模块，优先能力检测，不把版本字符串判断扩散到 Trainer。

#### 保持迁移纪律

每批只迁移一个完整职责，结构迁移与行为修复分开。新模型优先显式实现，旧 HF
路径按模型退出。不批量改名，不为了目录对齐移动同一套跨层依赖，未确认公开
入口和动态调用前不删除兼容代码。

## 8. 实施批次与出口门槛

每批处理一个可独立验收的职责；达到门槛再迁移对应调用方，不以批量搬目录推进。

| 批次             | 状态    | 下一项交付                                                 | 出口门槛                                           |
| -------------- | ----- | ----------------------------------------------------- | ---------------------------------------------- |
| R0/P0 基线       | 部分完成  | 固定版本、API、配置、初始化/RNG、状态及支持矩阵                           | 相同配置可复跑，证据与未验证项可定位                             |
| R1/P0 Qwen3 模型 | 部分完成  | 收尾底层数学归属与 ownership-aware 并行 meta 初始化                                 | logits、梯度、更新、共享权重、buffer、状态键和 RNG 对照通过         |
| R2/P0 TP/EP 计划 | 部分完成  | 完成剩余专家适配与能力解析，执行不再识别布局                           | 解析无副作用；执行禁用再识别仍通过单卡、TP、EP、joint 对照             |
| R3/P0 跨阶段绑定    | 部分完成  | PP 全局→局部 ownership、alias 映射，CP 显式目标                   | FSDP/PP/CP 与适用 AC/swap/compile 组合及 rank 失败测试通过 |
| R4/P0 完整恢复     | 已验收子集 | 真实数据游标、冷/重复加载、异步失败、拓扑变化                               | 模型、moments、scheduler、RNG、游标、EMA 及下一步一致         |
| R5/P1 目标规模     | 待验收   | Qwen3-MoE 真实数据长训、跨框架与性能基准                             | 100 步、50→100 独立恢复；数值与资源通过后测稳态性能                |
| R6/P1 其他模型     | 部分完成  | DeepSeek/Kimi/Qwen3.5 各自契约及兼容退出                       | 对应模型分别满足 R1–R5 的适用门槛，不复用近似数学                   |
| R7/P1 跨层收敛     | 部分完成  | Trainer/data/memory/metrics 消费契约，剩余 vendor 调用归 kernel | 无重复实现；异常退出、资源上限和恢复无回退                          |
| R8/P2 兼容清理     | 待完成   | 删除无消费者入口，按 trace 优化瓶颈                                 | 调用方迁移、公开说明、旧格式策略及硬件回归齐全                        |

R4 可先验收已支持子集；高级组合依赖 R3。R6 可并行做模型契约整理，但该模型
正式退出旧路径前仍需完成对应验收。R7 只在已有真实消费者时引入共享契约。

近期按以下顺序推进：

1. **补齐 R0 证据**：固定生产 Qwen3-MoE 配置、版本和调用链，记录尚在使用的
   HF 构造、初始化、状态映射及公开兼容入口。
2. **收尾当前 R2 批次**：验证 packed 绑定与分配前能力预检；覆盖后续 block
   不支持时整批不修改模型、不消耗 RNG，再跑真实 EP 对照。
3. **推进 R1 纵向闭环**：Qwen3 已拥有整网 forward 与初始化入口；继续核实
   底层参考组件的归属、并行 meta 初始化和 buffer 生命周期，保持状态键与 RNG，
   不把已有 CPU/Gloo 子集外推为目标 NPU 支持。
4. **完成 R3，再扩展 R4**：先确定 PP ownership/alias 与 CP 显式语义，再验收
   受影响组合的保存、独立恢复和失败边界。
5. **进入 R5/R6**：生产闭环通过后开展目标规模测试和其他模型迁移；性能优化
   以 trace 为依据，不能与数学迁移混为同一批。

每批产出代码、定向回归和验证记录。未满足出口门槛时保留兼容路径及缺口，
不扩大支持声明。独立工作可以并行，存在生命周期依赖的步骤按顺序推进。

## 9. 兼容与验收

外部 Transformers 配置、参考实现和权重格式按模型列出调用方、支持范围、替代
入口和退出条件；它们不提供生产模型构造。未实现的显式架构或组合提前拒绝，不
静默回退。FQN、packed layout、alias 或格式改变时，提供
版本化映射和旧 fixture，或明确不兼容错误。native DCP 保留稳定 FQN；模型
adapter 只处理外部格式，模型与 optimizer 分别建立恢复目标。

| 验收层   | 必须验证                                                 |
| ----- | ---------------------------------------------------- |
| 契约/解析 | 无模型、参数、RNG 或 collective 副作用；空 ownership、meta、冲突及过期绑定 |
| 模型参考  | logits/loss、输入/参数/router 梯度、更新权重、初始化、buffer、共享身份     |
| 分布式   | 适用 DP/TP/EP/joint/PP/CP，非对称输入、空专家、rank 本地失败及通信顺序     |
| 生命周期  | AC、compile、swap、累积、重复安装、异常关闭，无重复归约和提前释放              |
| 恢复    | 旧格式、同步/异步、冷/重复 load、适用拓扑变化，完整状态与恢复后继续训练              |
| 长程/规模 | 目标模型和真实数据，100 步及 50→100 独立恢复，固定跨框架参考                 |
| 性能    | 同配置、多次稳态测量，最慢 rank、p50/p90、有效 TPS、HBM/Host 与通信 trace |
| 工程    | 定向及全量 CPU、Ruff、format、compileall、空白检查及可用 pre-commit  |

固定权重、数据、RNG、精度、batch、序列长度、optimizer 和拓扑。纯结构迁移
在确定性路径逐位一致；kernel、dtype 或归约变化预先声明容差，不降低阈值来
通过回归。所有测试包含非空、非零和非对称检查；空路径另做专门测试。

rank0 区分配置请求、构造选择和 trace 中实际 kernel。报告全局 loss、grad norm、
参数量 B、有效 target TPS、TFLOP/s 和 MFU；tokens 与 FLOPs 计数分别说明，
不能用有效 target 数替代实际计算量。若沿用既有 910C 基准中的每卡 376 TFLOP/s 峰值假设，
须核对设备规格、适用精度与计算口径；该数字不作为所有配置的通用峰值。关闭诊断导出，分离编译、预热和稳态。

## 10. 完成标准与交付记录

整体完成须同时满足：

- 目标模型拥有结构、数学、初始化和状态映射；训练不再通过 HF inspection
  推断其结构。保留的 Transformers 依赖有明确用途和调用方清单。
- TP/EP/PP/CP 消费已校验目标；执行不重新选择模型数学；跨阶段参数替换、
  共享身份和 ownership 可检查，未支持组合在变换前拒绝。
- 生产 Trainer 贯通训练、资源关闭和完整恢复；目标规模、独立恢复及适用
  分布式组合有真实 NPU 证据，异常不会留下未完成的异步操作。
- 兼容入口按模型满足退出条件后删除；文档、配置与支持矩阵一致，未验收项
  单列，不以实验实现或 CPU 测试替代生产验收。


维护收益用新增模型涉及的通用文件数、重复解析次数、错误暴露阶段和恢复覆盖率
评估；性能收益只引用目标硬件实测。

每批记录参考/工作仓版本、工作区差异、环境、配置、命令、退出码、日志、数值/
资源结果和剩余缺口。CPU、mock、skip、导入或启动成功不替代 NPU 验收。
保留用户修改；不自动提交或推送，不提交日志、权重、缓存等生成文件。

参考：

- [分布式训练框架架构比较](distributed_training_framework_comparison.md)
- [FSDPTurbo 分布式设计](fsdpturbo_distributed_design.md)
- [TorchTitan 分布式设计对比](torchtitan_distributed_design_comparison.md)
- [EP 计划验证记录](ep_plan_refactoring_validation.md)
