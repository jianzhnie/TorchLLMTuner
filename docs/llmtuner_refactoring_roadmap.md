# TorchLLMTuner 重构路线图：模型、并行与训练的职责分离

本文给出一套基于现有代码逐步落地的重构方案，目标是让新增模型、增加并行策略和调整训练流程分别发生在各自的责任范围内。

核心方向是：**模型定义数学与结构，计划描述布局与约束，并行层执行变换，训练引擎封装执行差异，Trainer 管理训练生命周期。** 现有拓扑、collective、优化器、DCP 和内存机制继续复用，通过迁移调用方逐步收敛接口。

本文是目标设计与实施建议。标为“现有”的内容依据本次源码阅读；标为“建议”的类型、文件和接口尚不代表已经实现。历史测试数字不作为当前状态，支持范围以具体配置的验证记录为准。

## 阅读导航

- [一、现状与重构重点](#一现状与重构重点)
- [二、目标架构与依赖方向](#二目标架构与依赖方向)
- [三、模型层：统一契约，保留数学差异](#三模型层统一契约保留数学差异)
- [四、并行层：先解析，再执行](#四并行层先解析再执行)
- [五、训练层：引擎与生命周期分离](#五训练层引擎与生命周期分离)
- [六、状态转换、恢复与资源管理](#六状态转换恢复与资源管理)
- [七、目标目录与现有代码迁移](#七目标目录与现有代码迁移)
- [八、分批实施路线](#八分批实施路线)
- [九、验收与完成标准](#九验收与完成标准)

## 一、现状与重构重点

### 1.1 已有基础

| 现有位置 | 已承担的职责 | 本轮设计中的处理方式 |
|---|---|---|
| `models/registry.py`、`models/configuration.py` | 架构注册、配置解析和显式模型构造 | 保留唯一构造入口，避免训练器再做模型识别 |
| `models/contracts.py` | `DecoderStructure`、`MoEBlockContract` 等结构契约 | 在真实消费者需要时补充布局、能力和状态声明 |
| `parallel/plan.py` | `PlanTarget`、`ParallelPlan` 及绑定校验 | 延续阶段计划，逐步补齐跨阶段 ownership 与映射 |
| `parallel/stages.py` | 唯一阶段顺序 | 保留为装配顺序的事实来源 |
| `parallel/parallelize.py` | 全局预检、PP 处理和阶段执行 | 保留装配入口，减少执行期模型语义推断 |
| `trainer/runtime/assembly.py` | 模型、并行、优化器、数据和 checkpoint 装配 | 将隐式写入 Trainer 属性改为返回明确结果 |
| `trainer/trainer.py` | 训练、前后向、loss、PP 分支、优化、恢复和关闭 | 抽取执行职责，保留生命周期入口 |
| `components/`、`memory/`、`accelerator/`、`kernel/` | 训练服务、内存、设备与算子 | 复用现有能力，明确输入输出及资源所有者 |

这些基础说明，重构不需要再建立一套通用注册器或装配管理器。主要问题是边界尚未收敛：已有契约和计划还没有覆盖全部消费者，Trainer 与装配代码仍相互了解较多细节。

### 1.2 优先解决四类耦合

1. **模型语义反向渗入通用层。** 通用执行代码不应通过模型名、候选属性或参数形状猜测 score、packed halves、mask 或历史状态需求。
2. **执行差异进入训练循环。** 当前 `forward_backward_step()` 选择 PP/非 PP 路径，普通前后向还处理 chunked loss。这些差异应归执行策略，Trainer 只组织训练步骤。
3. **装配依赖可变 Trainer。** helper 通过读取和写入 Trainer 的大量属性完成组装，使依赖、初始化顺序和可用状态难以判断。
4. **状态与参数变换缺少统一追踪。** TP/EP 替换、PP 切分、共享权重、优化器与外部格式必须沿同一身份和 ownership 规则关联。

解耦不能只检查 import。还需要检查结构约定、数学语义和生命周期：张量形状匹配并不能证明专家数学正确，函数移到另一个文件也不能自动缩小依赖。

## 二、目标架构与依赖方向

### 2.1 五项职责，三个主要边界

```text
模型边界                     并行边界                    训练边界
模型实现与能力声明 ──→ 计划解析与校验 ──→ 并行装配 ──→ 执行引擎 ──→ Trainer
       │                     │              │              │           │
  数学、I/O、初始化       目标、布局、约束   分片、通信绑定   前后向、更新   数据与生命周期
```

图中的箭头表示数据流。代码依赖应由装配入口连接各层，而不是让底层模块反向导入上层：

- 模型依赖配置、共享数学和语义算子，不依赖 Trainer 或全局进程组。
- 通用并行实现依赖契约、拓扑和已解析目标，不导入具体模型族。
- 训练引擎依赖模型执行接口、装配结果和训练组件，不遍历模型内部猜测结构。
- Trainer 依赖训练引擎、数据与生命周期服务，不直接安装 TP/EP/CP hook。
- 模型专用并行适配可以使用通用通信原语，但其数学规则归模型族所有。

### 2.2 明确组装与运行两个阶段

**组装阶段**决定模型来源、能力范围、参数布局、执行策略和状态管理者。**运行阶段**调用已选定策略，执行前后向、通信与优化。

“训练前应用计划”不表示训练中没有通信。PP schedule、EP dispatch、CP attention 与梯度同步仍在运行期执行；应消除的是 Trainer 中的模型类型判断与策略重新选择。

也不要求所有模型继承大型 BaseModel。优先扩展已有契约，仅在多个真实调用方共享接口时增加 Protocol 或数据类型。

## 三、模型层：统一契约，保留数学差异

### 3.1 模型应提供什么

| 契约 | 最少内容 | 消费方 |
|---|---|---|
| 构造与初始化 | 架构 ID、任务、配置、meta 物化、buffer 和 RNG 顺序 | 模型工厂、装配入口 |
| 结构 | embedding、稳定 block ID/FQN、norm、head、共享参数 | PP、FSDP、计划解析 |
| 输入输出 | token、position、mask、padding、logits/hidden、aux loss | 数据适配、loss、执行引擎 |
| 并行语义 | 参数轴、激活布局、专家路由、CP 历史需求、PP I/O | 各阶段解析器与后端 |
| 能力 | dtype、设备、几何条件、初始化方式、组合限制 | 预检与支持矩阵 |
| 状态格式 | 外部键、打包、转置、alias、格式版本 | 模型状态适配器 |

`DecoderStructure` 和 `MoEBlockContract` 应继续作为起点。复杂模型通过具体契约补充 recurrent state、共享专家或特殊参数，而不是把所有模型的可选字段塞进一个巨型结构。

### 3.2 训练接口与并行接口分别消费模型声明

建议模型训练接口只承担输入预处理和 forward 输出；并行声明由独立解析入口消费。若保留模型的 `parallelize()` 方法，应将其实现为薄委托，具体拓扑和通信逻辑仍归并行层。

统一 batch 不等于只允许 `input_ids` 与 `position_ids`。文本与 VLM 的输入可以是包含张量和元数据的映射，保留图像、视频、文档边界和 loss mask。模型输出通过明确字段描述 logits、hidden 或辅助目标，避免 Trainer 搜索不同返回形式。

对 HF 的处理也要区分：外部配置与权重格式可以保留；已注册生产模型优先走显式家族实现。若需要直接训练 HF 模型，使用模型侧的有边界适配器并列出支持组合，不把通用 HF inspection 重新作为所有模型的默认路径。

### 3.3 模型接入的预期流程

新增 MoE 模型时，应依次实现结构与 forward、路由和专家布局声明、TP/EP/CP/PP 能力、初始化与状态映射，再运行计划和数值验证。通用 dispatcher 不根据模型名称决定 score 函数，checkpoint 不根据形状猜专家含义。

Llama 之类的 dense 模型不必制造空 EP 契约；复杂 MoE 的 grouped 激活、score 乘法位置、shared expert 和专家打包则必须显式说明。共享实现的前提是数学与生命周期一致。

## 四、并行层：先解析，再执行

### 4.1 区分请求、计划和装配结果

| 对象 | 含义 | 不应承担的职责 |
|---|---|---|
| 用户配置 | 请求并行度、后端和覆盖规则 | 直接充当已经验证的计划 |
| 阶段/全局计划 | 明确目标、前后布局、ownership、约束与依赖 | 修改模型或执行 collective |
| 装配结果 | 本地模型分段、拓扑视图、参数映射和执行需求 | 隐式接管 Trainer 全部状态 |

现有 `ParallelPlan` 已提供阶段目标与绑定信息，不应宣传为完整跨阶段全局计划。建议先组合已有 TP/EP 计划和 PP ownership 元数据，再按需求补齐 CP/FSDP、重计算和量化的依赖；不强制所有阶段使用相同字段。

### 4.2 解析器必须守住的约束

解析顺序为模型声明、显式用户覆盖、精确目标绑定、几何/布局/能力/冲突校验。解析不得替换参数、安装 forward、消耗 RNG、分配专家权重或运行 collective。

计划可以引用模块和参数，但不复制其 storage；不可变 dataclass 不会冻结被引用对象。配置、绑定或参数布局改变后必须重新解析。PP 非本地规则应保留全局归属，不能作为普通零匹配处理。

本地检查结束后，由外层统一协调各 rank 的错误，再进入通信阶段。不要在本地校验函数内混入 collective，否则某个 rank 提前失败可能导致其他 rank 等待。

### 4.3 跨阶段身份与装配顺序

参数变换应返回模块映射、旧参数到新参数映射和 alias 关系。PP 使用稳定全局 block ID 关联局部对象；FQN、共享身份与 storage 关系分别记录，不能只用字符串替换完成重绑。

沿用 `parallel/stages.py` 的现有顺序：

```text
模型构造与能力预检
  → 全局目标解析与错误协调
  → 按既有流程建立 mesh、PP 切分及局部重绑
  → TP → EP → CP → AC → activation_swap → compile → FSDP
  → 按模型支持的时点物化、初始化或加载权重
  → 建立 optimizer、scheduler 与执行引擎
```

这是迁移时应保护的生命周期关系，不授权为符合图示而改变初始化或 checkpoint 时点。meta 初始化与外部权重加载分别验证；阶段顺序也不意味着所有功能组合自动受支持。

并行装配不提供天然事务回滚。预检应尽量避免半变换；执行失败后完整重建或恢复，不在半变换模型上静默切换后端。

## 五、训练层：引擎与生命周期分离

### 5.1 Trainer 保留什么

Trainer 管理数据迭代、步数、梯度累积组、验证周期、指标上报、保存恢复触发和关闭。训练状态保留明确的 owner，避免子组件各自维护不同的 step。

建议先保持公开 `Trainer` API，在内部抽取执行引擎；只有出现第二个真实训练应用时，再决定是否增加 `BaseTrainer`。建立空基类不能替代职责迁移。

### 5.2 训练引擎承担什么

建议在 `trainer/runtime/` 提炼最小执行接口，以下为设计示意，并非当前 API：

```python
class TrainingEngine(Protocol):
    def forward_backward(self, batch_group, *, normalization) -> StepResult: ...
    def optimizer_step(self) -> UpdateResult: ...
    def close(self) -> None: ...
```

`StepResult` 明确区分未归一化 loss sum、有效 target 数和辅助指标；`UpdateResult` 提供 grad norm、更新是否执行等信息。状态保存继续委托 checkpoint 服务，不把所有服务塞进引擎基类。

组装时选择普通、PP 或 chunked-loss 执行路径，训练循环调用同一接口。保留现有 `runtime/pipeline.py`，普通和 PP 路径共享 loss 与优化语义；只有确有多个路径需要时才拆分实现文件。

### 5.3 梯度累积和 loss 归一化必须统一

建议沿用现有全局有效 target 归一化思路：先获取一次 optimizer step 所需的 batch groups，计算该步有效 target 总数，并按既定数据/上下文分片语义归约；每组反向都使用同一个分母。

不能直接平均各 microbatch 的平均 loss，因为 padding、packing 或不同文档长度会导致权重不等。TP/PP 复制的数据也不能重复计入分母。FSDP 梯度平均、EP ownership 和复制参数同步必须与该规则共同验证，避免重复除以并行度。

一个 step 的责任顺序应可检查：

```text
准备 batch groups 与归一化上下文 → zero_grad
  → 多组 forward/backward
  → 必要的梯度同步 → grad norm / clipping
  → optimizer → scheduler / EMA / 模型更新 hook
  → 提交 step 与数据进度 → 指标与保存触发
```

溢出或跳过更新时，step、scheduler、EMA 与数据游标的推进规则必须显式定义。具体 hook 顺序保护现有行为，不为统一接口而擅自重排。

### 5.4 可替换组件通过构建接口接入

| 组件 | 接口重点 | 装配时必须确定的信息 |
|---|---|---|
| Loss | token sum、mask、辅助目标、vocab 布局 | 模型输出类型、loss group、归一化规则 |
| Optimizer | zero/update、状态导入导出 | 最终参数及分组、替换映射、独立 dtype |
| Scheduler / EMA | 更新与状态 | optimizer 更新口径、恢复游标 |
| Metrics | 接收 detached 指标与明确归约语义 | 指标 owner、记录 rank、重复计数规则 |
| Profiler | 上下文进入退出、step | 采集窗口和设备资源 |
| Checkpoint | save/load/wait/close | 完整 stateful 对象与格式适配器 |

配置在装配层解析为 builder 和实例；组件不接收整个 Trainer。scheduler 依赖 optimizer 时使用 builder 注入，不要求用户提前构造不完整对象。可替换不等于允许任意组合，兼容条件仍在构建阶段校验。

## 六、状态转换、恢复与资源管理

将状态相关职责拆成三部分：

1. **模型格式适配**：处理 HF 外部键、转置、专家打包与共享别名，由模型族说明。
2. **运行时布局适配**：依据 TP/EP/PP/FSDP 布局与参数映射建立模型及 optimizer 状态，保留规范 FQN。
3. **Checkpoint 编排**：保存模型、moments、scheduler、EMA、RNG、数据游标和训练进度，管理同步/异步持久化与关闭。

恢复先检查格式版本、模型能力、配置与拓扑兼容性，再修改目标状态。跨拓扑恢复必须具有对应重分片实现和验证，不能因使用 DCP 就默认成立。

optimizer 在参数布局稳定后创建；恢复目标的创建和 moments 初始化遵循现有 backend 契约。独立恢复不仅比较权重，还要比较下一步的 loss、更新、RNG 和数据位置。

swap、预取、profiler、异步保存分别持有自己的资源；装配入口记录创建顺序，异常退出时按依赖逆序关闭，先等待在途操作再释放 storage。`npu_virtual`、AdamWSwap 和 FSDP CPU offload 保留独立配置与生命周期。

## 七、目标目录与现有代码迁移

以下是职责图。`engine.py` 和部分数据类型为建议新增项，其余优先扩展现有叶子模块；不要求模型文件统一改名。

```text
llmtuner/
├── config/                       用户请求、静态校验与构建配置
├── models/
│   ├── registry.py               唯一架构注册
│   ├── configuration.py          显式模型配置与构造选择
│   ├── contracts.py              有真实复用的结构和能力契约
│   ├── common/                   数学与生命周期一致的组件
│   └── <family>/<model>/         模型数学、初始化、并行声明、格式映射
├── parallel/
│   ├── topology.py               mesh 与 rank/group 视图
│   ├── plan.py                   阶段计划、绑定与映射
│   ├── stages.py                 唯一阶段顺序
│   ├── parallelize.py            解析协调、PP 重绑与统一执行入口
│   └── <family>/<family>.py       各并行家族入口与所属解析器
├── trainer/
│   ├── trainer.py                公开生命周期入口
│   └── runtime/
│       ├── assembly.py           返回明确装配结果
│       ├── engine.py             建议：前后向与更新执行接口
│       ├── batch.py              数据到模型输入的连接
│       └── pipeline.py           既有 PP 执行实现
├── components/                   loss、optimizer、checkpoint、指标、profiler
├── datasets/                     数据读取、packing、分流及可恢复游标
├── memory/                       重计算、物化与 swap 资源
├── accelerator/                  设备、group 和运行时能力
└── kernel/                       语义算子、输入校验和设备实现
```

| 现有职责 | 迁移目标 | 保留要求 |
|---|---|---|
| assembly helper 读写 Trainer 属性 | 分别返回模型/并行装配结果和训练服务 | 分步迁移，公开 Trainer API 保持可用 |
| Trainer 的 PP/普通分支 | engine 构建时选择执行路径 | PP 与整网共用同一并行阶段 runner |
| Trainer 的 chunked loss 分支 | loss/执行策略 | 保留内部 backward 和归一化语义 |
| 模型类型相关组合判断 | 模型能力声明 + 所属阶段预检 | 失败发生在模型修改前，错误可定位 |
| 临时参数路径修补 | 显式替换映射与 ownership | 共享身份、optimizer 和 DCP 同步迁移 |
| 模型直接 vendor 调用 | 在语义一致时归 kernel | 模型保留数学顺序，不静默搬至 CPU |

建议装配结果只包含实际消费者需要的字段，例如 `model_parts`、拓扑视图、参数映射、输出布局和执行能力；训练服务单独返回。不要把它变成全局可变 context，也不要让 engine 接收 Trainer 实例。

## 八、分批实施路线

先用生产 Qwen3-MoE 做纵向迁移，再按各模型语义迁移 DeepSeek、Kimi 和 Qwen3.5。实验目录只验证设计，不直接复制为第二套生产 Trainer 或 checkpoint 实现。

| 批次 | 主要交付 | 出口门槛 |
|---|---|---|
| R0：固定基线 | 版本、配置、支持矩阵、公开 API、初始化及恢复参考 | 能复跑训练与独立恢复，记录现有失败与未验证项 |
| R1：装配结果显式化 | 将 assembly 对 Trainer 的隐式赋值分批改成明确返回值 | 创建顺序、RNG、状态键和公开入口行为保持一致 |
| R2：抽取执行引擎 | 迁移普通、PP、chunked-loss 前后向及更新职责 | Trainer 无 PP/TP/EP/CP 算法分支，累积与 loss 口径对照通过 |
| R3：补齐模型声明 | Qwen3-MoE 结构、初始化、专家数学、能力和状态契约 | 修改模型前可解释每个并行目标；缺失声明提前拒绝 |
| R4：跨阶段计划与映射 | 组合阶段计划、PP 全局到局部 ownership、alias 和替换映射 | 解析无副作用，过期绑定与失败协调测试通过 |
| R5：组件与完整恢复 | builder 注入、模型/布局/持久化分工、数据游标及关闭 | 连续训练与新进程恢复后下一步一致；异常资源清理通过 |
| R6：逐模型扩展 | DeepSeek/Kimi/Qwen3.5 的实际能力与约束 | 每个模型分别完成适用单卡、并行及恢复验证 |
| R7：兼容退出与规模验证 | 删除已迁移且无消费者入口，目标规模与性能基准 | 旧格式策略、调用方迁移、长程训练与硬件证据齐全 |

R1/R2 优先减少训练侧依赖；若其迁移暴露模型或布局语义缺口，先以最小契约补齐，再继续迁移，不等待一个庞大全局计划完成。R3/R4 逐阶段推进，R5 从已有组合开始，拓扑变化与高级内存组合单独验收。

### 下一批建议只做三件事

1. 为 `trainer/runtime/assembly.py` 列出输入、写入属性、创建资源与消费者，先提取模型/并行装配结果，保持调用顺序不变。
2. 将 `forward_backward_step()` 及普通/PP 执行依赖整理为引擎接口，`Trainer.train_step()` 保留数据、归一化上下文和生命周期触发；逐步迁出更新细节。
3. 用同一生产小配置对照迁移前后 loss sum、分母、梯度、optimizer、RNG 与独立恢复，补上不等长 microbatch 和 PP 场景。

每批只迁移一个可验收职责。结构变化与数学/归约修复分别提交验证结果，避免无法判断差异来源。

## 九、验收与完成标准

### 9.1 验收矩阵

| 层级 | 必须验证 |
|---|---|
| 契约与解析 | 精确绑定、零匹配/冲突、meta、ownership、alias、RNG 无副作用 |
| 模型数学 | 初始化、buffer、logits、loss、输入/参数/router 梯度与更新 |
| 并行执行 | 适用 TP/EP/CP/PP/FSDP 及组合，非对称数据、空专家、group 通信顺序 |
| 训练语义 | 有效 target 加权、梯度累积、重复归约、clip、更新跳过与 hook 顺序 |
| 生命周期 | AC/compile/swap、重复安装、构造失败、在途操作与异常关闭 |
| 恢复 | 权重、moments、scheduler、EMA、RNG、数据游标与下一步；冷/重复加载 |
| 长程与性能 | 真实数据、目标硬件，建议 100 步及 50→100 独立恢复；稳态吞吐与资源峰值 |
| 工程检查 | 定向/全量 CPU、Ruff、format、compileall、pre-commit 和差异空白检查 |

纯结构迁移在确定性路径下逐位对照；算子、dtype 或归约改变时先约定容差。CPU/Gloo 和小模型可以验证子集，不能替代目标 NPU、长序列、多节点或长程验收。

每次记录代码版本与工作区差异、环境、命令、配置、退出码、日志、数值、资源和剩余缺口。全量测试存在 collection error 时不能标为通过；架构文档不长期抄录容易失效的测试总数。

### 9.2 怎样判断真正完成了解耦

- 新模型主要修改模型族和注册入口，不扩张通用并行层的模型名判断。
- 新并行实现消费既有契约和明确布局，不要求修改 Trainer 主循环。
- 新数据或 loss 通过适配和组件接入，不反向理解专家内部存储。
- 参数变换、ownership、状态转换和恢复都有明确来源，失效时可提前定位。
- Trainer 保持唯一生命周期入口，运行策略和资源 owner 可单独验证。
- 现有公开路径和旧格式有明确迁移方案，未支持组合与未验证项分别列出。

维护收益用新增模型修改的通用文件数、重复解析次数、错误暴露阶段和恢复覆盖率衡量。性能收益只引用同配置的目标硬件实测。

## 参考位置

当前实现入口：`llmtuner/models/contracts.py`、`llmtuner/models/registry.py`、`llmtuner/parallel/plan.py`、`llmtuner/parallel/stages.py`、`llmtuner/parallel/parallelize.py`、`llmtuner/trainer/runtime/assembly.py`、`llmtuner/trainer/trainer.py`、`llmtuner/trainer/runtime/pipeline.py`。

设计依据见[分布式训练框架架构比较](distributed_training_framework_comparison.md)。实验原型只用于验证接口设计，其代码存在与单进程运行结果不作为生产支持声明。
