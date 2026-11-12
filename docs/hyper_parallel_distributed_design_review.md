# HyperParallel distributed 设计与模型架构耦合调研

## 结论

HyperParallel 的 `hyper_parallel/distributed/` 采用“**模型 adapter 提供能力，
通用 planner 生成 ShardingPlan，builder 执行参数分片与 forward 改写**”的设计。

它与模型架构代码的耦合需要分开评价：

- **直接 import 耦合较低**：通用 distributed 基本没有直接导入具体模型类；
  模型 adapter 延迟注册，依赖方向主要是 adapter → distributed。
- **结构与数学语义耦合中高**：planner 仍根据 FQN、属性名、配置和参数角色
  推导模型结构；EP routing 保留模型族别名；KDA CP 在通用目录中直接理解
  多套 Kimi 层 API；运行时还依赖 forward 包装和模块接口。
- **adapter 机制有效，但没有消除模型知识**。它主要解决了模型发现、按需加载和
  专用能力归属问题，尚未把整个 distributed 变成完全模型无关的执行层。

因此，不能简单评价为“耦合很严重”或“已经完全解耦”。更准确的判断是：
**代码依赖方向较好，运行时和数学契约耦合仍然较强，部分模块耦合偏高。**

## 1. 目录与代码规模

当前 `hyper_parallel/distributed/` 包含 56 个 Python 文件，约 24,160 行：

| 子目录 | 文件数 | 约行数 | 主要职责 |
| --- | ---: | ---: | --- |
| `_builder/` | 15 | 7,475 | 计划推导、参数分片、forward 改写、FSDP 适配 |
| `context_parallel/` | 13 | 8,187 | CP 通信、KDA、MLA、GDN 和序列处理 |
| `expert_parallel/` | 7 | 3,565 | 路由、专家执行、dispatch、模型 archetype |
| `encoder_parallel/` | 10 | 1,178 | 多模态 encoder 的独立并行与传输 |
| `tensor_parallel/` | 3 | 548 | 参数角色和 head 数推导 |
| 根目录其他文件 | 8 | 1,207 | 装配入口、plan、mesh、compile、AC 等 |

模型侧 `models/*/adapter/distributed/` 另有 28 个文件、约 4,805 行。
这说明专用并行代码并没有全部集中在通用 distributed 中，adapter 分流确实存在。

## 2. 总体架构

### 2.1 模型 adapter：能力声明与延迟发现

`models/adapter_spec.py` 中的 `ModelAdapterSpec` 是模型族的数据契约，
声明：

- 模块替换和 attention 适配；
- checkpoint、materialization 和外部状态；
- CP/EP provider；
- sharding rules；
- FSDP 包装单元、排除子树和执行顺序；
- recompute、loss、encoder parallel 和 validation。

`models/registry.py` 按 `models/<family>/adapter/registration.py` 的目录约定
发现模型族，首次查询时才导入 registration。registration 再延迟加载具体实现。

例如 `models/qwen3_moe/adapter/registration.py` 只注册 provider，不在发现阶段
导入模型、Trainer 或设备后端。

这层设计的收益是：

1. 新模型通常不需要修改通用 registry；
2. 通用层不直接持有模型类导入；
3. 模型专用 CP/EP/FSDP 策略有明确归属；
4. 重型模型和验证依赖不会随 registry import 被提前加载。

### 2.2 ShardingPlan：模型级执行契约

`distributed/plan.py` 中的 `ShardingPlan` 包含：

- 每个模块的参数 placement；
- 输入/输出 layout 和通信边界；
- `local_compute_fn`、`inner_wrapper` 等计算注入；
- 特殊参数 handler；
- tied weights；
- 启用的 mesh axes。

`explain()` 可以展示参数切分、输入输出通信和注入方式。它不是简单配置副本，
而是 planner 与 builder 之间的核心中间表示。

### 2.3 Planner：从模型树推导计划

`_builder/planner.py` 中的 `ShardingPlanner` 执行多阶段推导：

1. 参数角色分类；
2. 通信边界分组；
3. attention/MLP/MoE 语义角色推断；
4. 模板生成 `ModuleShardingSpec`；
5. 合并用户 `plan_overrides`；
6. 标记 terminal 和收集特殊参数 handler。

模型 adapter 的 `sharding_rules` 可以覆盖通用命名规则，但 planner 仍提供
自动推导。`derive=False` 可以关闭推导，改为完全显式声明，适用于自动推导
可能产生错误数学语义的区域。

### 2.4 Builder：参数分片与 forward 改写

`distributed/apply.py` 是薄入口，执行顺序为：

```text
mesh / 注入预检
  → Phase 0 配置规范化
  → Phase A 参数分片
  → Phase B 特殊参数处理
  → Phase C 构造 source layout、包装 forward
  → Phase D tied weights
```

生产模式和 validate 模式共享局部区域和通信重建逻辑，差别主要在边界：
生产路径在构造期解包为 local tensor，validate 路径保留 DTensor 传播检查。
设计目标是不让生产热路径继续承担通用 DTensor dispatch 成本。

## 3. 直接依赖耦合：总体较低

对 `distributed/**/*.py` 的 import 检查显示，通用代码直接导入
`hyper_parallel.models` 的位置主要只有：

- `activation_checkpoint.py`：`RecomputePolicy` 和 registry；
- `compile.py`、`mesh.py`：模型构建配置；
- `_builder/fsdp_adapter.py`：FSDP 配置和 adapter registry；
- `_builder/planner.py`：adapter registry。

这些导入针对配置和注册机制，而不是 Qwen、DeepSeek、Kimi 的具体模型类。
模型专用 EP 路径主要以字符串形式出现在 archetype 注册中。

因此：

- 通用 distributed → 具体模型实现的静态 import 很少；
- adapter → distributed 的复用关系是主要依赖方向；
- registry 和 provider 的延迟导入降低了循环依赖与启动成本。

这是架构中做得较好的部分。

## 4. 结构与数学语义耦合：仍然明显

### 4.1 Planner 仍包含模型结构知识

`_builder/planner.py` 不仅读取 adapter 规则，还包含：

- `attn`、`attention`、`mlp`、`ffn`、`moe` 等名称模式；
- q/k/v/o、gate/up/down、w1/w2/w3 等 leaf segment；
- 参数角色组合与 MoE 边界推断；
- `config.architectures` → `model_type` → 类名的架构回退识别；
- HF 模型模块命名和嵌套约定。

这不是直接 import 耦合，但属于**结构约定耦合**。如果模型数学相同但命名不同，
自动计划可能无法覆盖；如果命名相似但语义不同，则可能生成错误边界。
adapter 的 `sharding_rules` 和显式 `plan_overrides` 是必要补偿，而不是可选装饰。

### 4.2 EP 通用层仍有模型族路由知识

`expert_parallel/routing.py` 的 `MOE_ROUTER_ADAPTERS` 直接包含：

- `qwen2moe`、`qwen3moe`；
- `deepseekv3`、`deepseek_v3`；
- `glm4moe` 等模型族别名。

`expert_parallel/recipes.py` 还保留 Qwen2-MoE、DeepSeek-V3/GLM-4-MoE 的
完整 archetype 和模型名建议。新的 Qwen3-MoE 已迁移到模型 adapter，说明项目
正在改善归属，但旧实现尚未完全收敛。

较好的方向已经存在于
`models/qwen3_moe/adapter/distributed/expert_parallel.py`：模型 adapter 声明
router 和 shared-expert 组合，通用 `build_ep_compute` 提供 dispatch、专家执行和
collective skeleton。

### 4.3 KDA CP 是耦合最高的区域之一

`context_parallel/kimi_delta_attention.py` 约 1,500 行，直接处理：

- Kimi KDA 的状态递推和 local math；
- `f_proj` 与 `f_a_proj/f_b_proj` 两套 projection API；
- `num_v_heads/num_heads`、`head_v_dim/head_dim` 等字段差异；
- output gate、safe gate、short convolution 和输出 tuple 形式；
- 多种 kernel 与状态合并路径。

此外，`context_parallel/` 还包含 `kimi_delta_attention_rd.py`、
`kimi_delta_attention_state.py`、`kimi_delta_attention_rd_plan.py` 等专用实现。

这里已经不是单纯通用 CP 通信，而是“**模型专用并行数学实现放在通用目录**”。
它与 Kimi 模型结构、参数命名、forward 输出和 kernel 接口高度耦合，应评为高耦合。

### 4.4 FSDP 是声明与结构发现并存

`_builder/fsdp_adapter.py` 对 adapter 的支持较好：

- `fsdp_wrap_modules` 声明额外 FSDP 单元；
- `fsdp_excluded_subtrees` 排除非 decoder 分支；
- `fsdp_execution_order` 处理条件执行和预取顺序。

但它仍保留自动 decoder/expert 发现、类名匹配和 HF 配置识别。FSDP 的模型
适配成本因此没有消失，只是部分迁移到 adapter。

### 4.5 forward 改写带来运行时接口耦合

builder 会包装或替换 forward，并注入 local compute、inner wrapper 和 CP/EP
逻辑。即使静态 import 很少，运行正确性仍依赖：

- 参数名和模块嵌套；
- forward 参数与返回结构；
- mask/cache/padding 语义；
- router 输出、专家布局和 shared expert 调用方式；
- AC、compile 与 hook 生命周期。

这种耦合是 HF 模型适配本身难以完全消除的，但应由显式契约约束，不能只凭
目录和 import 关系判断系统已经解耦。

## 5. 分模块耦合评级

| 模块 | 直接依赖 | 语义耦合 | 评价 |
| --- | --- | --- | --- |
| registry / adapter_spec | 低 | 低 | 清晰的模型能力入口，建议保留 |
| mesh / ShardingPlan | 低 | 低到中 | 通用数据结构，边界较稳定 |
| `_builder/applier.py` | 低 | 中 | 执行编排通用，但依赖 forward 契约 |
| `_builder/planner.py` | 低 | 中高 | 自动推导包含大量模型命名和结构知识 |
| `_builder/fsdp_adapter.py` | 低 | 中 | adapter 声明有效，仍有自动结构发现 |
| EP collectives / dispatch | 低 | 低到中 | 通用通信可复用 |
| EP routing / legacy recipes | 低 | 中高 | 模型族别名和路由语义仍在通用层 |
| 通用 CP collectives | 低 | 中 | 依赖 attention 输入输出和 mask 契约 |
| KDA/Kimi CP | 中 | 高 | 模型专用数学、API 和状态语义集中在 distributed |
| encoder_parallel | 低 | 中 | 独立拓扑清晰，仍需模型提供执行边界 |

## 6. 优化建议

### P0：让显式契约成为正确性的主要来源

1. 新模型优先使用 adapter 声明和显式 `plan_overrides`，不要继续扩展 planner
   的模型名、类名和 FQN 猜测。
2. 对自动推导覆盖的模型保留非空覆盖、边界和 ownership 测试；推导失败必须
   明确报错，不能静默生成不完整计划。
3. 对 encoder、复杂 MoE、KDA 和特殊状态路径优先评估 `derive=False`，避免
   用后续补丁纠正错误的自动计划。
4. 将 forward 输入、输出、mask/cache、router 和共享专家调用形式纳入模型契约，
   不把 apply-time 属性存在性检查误认为数值正确性验证。

### P1：继续迁移模型专用实现

1. 按 Qwen3-MoE 的方式，将 Qwen2-MoE、DeepSeek-V3/GLM-4-MoE 的 archetype
   和 router alias 收敛到各自 adapter。
2. 通用 EP 层只保留 dispatch、collective、专家执行骨架及与模型无关的 router
   数学，不再维护模型族名称表。
3. 将 KDA 的 projection 适配、输出格式和模型数学移到 Kimi adapter；通用 CP
   只消费明确的状态、序列边界和通信契约。
4. FSDP 新模型优先使用精确 FQN 和 execution order 声明，逐步减少自动 decoder
   结构发现。

### P1：保持 adapter 轻量，避免反向膨胀

adapter 不应复制通用 collective、FSDP 或 dispatcher，也不应变成第二套 Trainer。
每个 provider 只负责模型特有语义，通用执行仍由 distributed 提供。共享代码
只有在数学语义和生命周期完全一致时才抽取，避免为了降低行数合并近似实现。

## 7. 对 TorchLLMTuner 的参考价值

最值得借鉴的是：

- `ModelAdapterSpec` 的模型能力声明；
- registration/provider 延迟发现；
- ShardingPlan 作为解析与执行之间的中间表示；
- 生产与 validate 共用显式局部区域，避免依赖运行时猜测；
- 模型专用 EP 实现迁移到 adapter 的方向。

不宜直接照搬的是：

- 继续扩大自动 FQN/属性推导；
- 用 forward 改写替代所有模型适配；
- 将模型专用数学留在通用 distributed；
- 把 registry 中不存在模型名误认为已经完全模型无关。

最终目标应是：**模型声明结构与数学契约，planner 只解析这些声明，builder
只执行已验证计划；adapter 机制用于隔离差异，而不是承载新的隐式推断。**

## 参考位置

- `hyper_parallel/models/adapter_spec.py`
- `hyper_parallel/models/registry.py`
- `hyper_parallel/models/qwen3_moe/adapter/registration.py`
- `hyper_parallel/models/qwen3_moe/adapter/distributed/expert_parallel.py`
- `hyper_parallel/distributed/apply.py`
- `hyper_parallel/distributed/plan.py`
- `hyper_parallel/distributed/_builder/planner.py`
- `hyper_parallel/distributed/_builder/fsdp_adapter.py`
- `hyper_parallel/distributed/expert_parallel/routing.py`
- `hyper_parallel/distributed/expert_parallel/recipes.py`
- `hyper_parallel/distributed/context_parallel/kimi_delta_attention.py`
