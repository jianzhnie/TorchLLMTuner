# TorchLLMTuner 重构设计与实施路线

本文定义生产框架的目标、当前差距和交付门槛，配合
[模型与分布式架构指南](model_distributed_architecture_guide.md)使用。
状态核对日期：2026-11-12。代码路径表示当前实现位置；验收只对记录的配置和
快照成立，工作区后续修改需要重新验证。

## 1. 目标与原则

生产调用链收敛为：

```text
显式模型契约 → 纯计划解析与校验 → 固定阶段执行 → 统一训练与恢复
```

优先完成 Qwen3-MoE 的生产闭环，再逐模型迁移 DeepSeek V4、Kimi K3、Qwen3.5。
复用现有拓扑、collective、内存、optimizer 和 checkpoint，不建立第二套训练栈。

不按行数或文件数评价进度，不因目录迁移重写已验证数学，不以实验模型代替
生产迁移，也不预设吞吐提升。

## 2. 当前状态与差距

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

当前工程门槛（本轮工作区）：全量 CPU 使用
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

## 3. 代码职责与依赖

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

## 4. 接口与装配约束

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

当前计划类型已用于 TP 和 EP，并与实验共享；不代表两个阶段已合并为完整
全局计划，也不代表下表字段都已实现。
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

## 5. 实施批次与出口门槛

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

## 6. 兼容与验收

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
不能用有效 target 数替代实际计算量。目标 910C 指定峰值为每卡 376 TFLOP/s，
注明适用精度、卡数及 FLOPs 口径。关闭诊断导出，分离编译、预热和稳态。

## 7. 完成标准与交付记录

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

- [模型与分布式架构指南](model_distributed_architecture_guide.md)
- [FSDPTurbo 分布式设计](fsdpturbo_distributed_design.md)
- [TorchTitan 分布式设计对比](torchtitan_distributed_design_comparison.md)
- [EP 计划验证记录](ep_plan_refactoring_validation.md)
