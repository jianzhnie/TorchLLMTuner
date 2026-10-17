---
name: dev-guide
description: 审查、修改和重构 TorchLLMTuner 的 llmtuner 代码时，指导模块归属、混合并行与训练正确性、上游实现对齐及验证。适用于该仓库的功能开发、缺陷修复和结构调整。
---

# TorchLLMTuner 开发指南

以仓库当前代码和根目录 `AGENTS.md` 为依据。开始任务时读取适用的
`AGENTS.md`，检查工作区状态和相关调用方；用户当前要求优先于本指南。
本文路径均相对仓库根目录，执行命令前确认位于 TorchLLMTuner。

## 1. 按职责放置代码

| 目录 | 职责与关键入口 |
| --- | --- |
| `llmtuner/config/` | 配置 dataclass、字段及组合校验；文件使用 `_config.py` 后缀 |
| `llmtuner/accelerator/` | 设备发现、进程组初始化、collective、SPMD 上下文、张量树搬运 |
| `llmtuner/datasets/` | 数据源、处理、组合、packing、collation 和 loader |
| `llmtuner/models/` | HF 适配、模型构件、attention、MoE routing 与专家计算 |
| `llmtuner/parallel/` | 拓扑、模型并行变换、通信机制和组合能力检查 |
| `llmtuner/memory/` | activation checkpoint 与重计算等内存优化 |
| `llmtuner/components/` | loss、优化器、checkpoint、tokenizer、metrics 和 profiler |
| `llmtuner/trainer/` | 训练编排、CLI，以及已有 `runtime/` 支持模块 |
| `llmtuner/utils/` | 日志、随机数、GC、dtype 和设备监测等通用工具 |

遵守已有边界：

- 保持 `trainer/trainer.py` 的训练与 optimizer step 实现完整，未经用户要求
  不进一步拆分；保持 `components/metrics.py` 单文件。
- 并行家族以同名文件为核心入口，例如
  `parallel/expert_parallel/expert_parallel.py`；支持模块承载具体机制。
- 数据共享类型定义在 `datasets/contracts.py`；配方组合在 `composition.py`，
  loader 装配在 `loader_builder.py`。
- MoE token dispatch 位于 `parallel/expert_parallel/token_dispatch.py`；
  TorchAO padding 适配位于同目录 `torchao_dispatch.py`，依赖在构造时导入；
  router、专家计算和 MoE layer 位于 `models/common/moe/`，FP32 router
  投影定义在 `models/common/moe/gate.py`。
- 优化器状态初始化与 FQN 序列化位于 `components/optimizer/state_dict.py`；
  TP 模型投影构件位于 `models/common/tp_projections.py`，通信原语位于
  `parallel/tensor_parallel/linear_ops.py`。
- HF 配置构建与类解析位于 `models/hf/factory.py`；并行变换后的 meta 存储
  分配、运行时 buffer 重建与共享引用恢复位于 `models/hf/materialization.py`。
- 优化器参数组、学习率调度和 EMA 配置分别位于 `config/optimizer_config.py`、
  `config/lr_scheduler_config.py` 和 `config/ema_config.py`；运行时直接导入
  定义模块，`config/__init__.py` 维护统一公共导出。

实现代码从符号的定义模块导入；包 `__init__.py` 只暴露有意维护的公共 API。
保留现有懒加载，避免配置或轻量入口引入训练引擎、媒体后端或可选内核。
本仓库允许有明确职责的跨层依赖，例如 TP 模型构件依赖并行通信原语；
按调用关系和循环导入风险判断模块归属，不采用文件夹必须独立复制的约束。

## 2. 配置、设备与日志

- 字段约束放在对应配置的 `__post_init__` 或其调用的 `validate_*_config` 中；
  并行组合限制集中复用 `parallel/compatibility.py`。
- 区分可在构造时判定的配置错误与依赖实际 world size、模型布局、设备能力的
  运行时检查。度数推导和 mesh 构造由 `parallel/topology.py` 负责，
  避免在 trainer、配置和并行入口复制同一规则。
- 使用 `errors.py` 的错误语义：`ConfigError` 表示配置非法，
  `UnsupportedCombinationError` 表示组合不支持，
  `EnvironmentUnsupportedError` 表示环境能力不足；可选包缺失使用带安装提示的
  `ImportError`。拒绝尚未实现的选项，避免静默忽略或宣称已支持。
- 通用设备操作复用 `accelerator/device.py` 的 `device_type`、`device_module`
  和设备查询函数。后端专用 API 放在负责该能力的适配模块，并检查依赖和能力；
  本仓库没有 FSDPTurbo 的 `torch.accelerator` 代理或 `ops` 注册体系。
- 嵌套张量搬运使用 `accelerator/tensor_tree.py` 的
  `move_tensor_tree`、`get_tensor_tree_device`，遵守其容器和输出结构契约。
- 日志使用 `utils/logging.py::get_logger`；通过现有 rank 过滤和
  `set_log_ranks` 控制输出。错误提示写实际配置字段及修复条件。

## 3. 审查训练与并行正确性

从 `trainer/trainer.py` 追踪相关调用链，按本次修改范围检查以下不变量：

- **装配顺序**：以 `parallel/stages.py` 为单一来源，保持
  `TP → EP → CP → activation checkpoint → compile → FSDP`。
  PP 先分割模型，再对本地 model parts 应用既定阶段顺序。
- **拓扑与归约**：检查 dense/sparse/loss/data mesh 的用途、rank ownership、
  collective group、形状和整除条件；EP 轴与其他轴的关系以 topology 实现为准，
  不凭并行度乘积猜测。检查复制参数梯度和 token 统计是否重复归约。
- **loss 与累积**：核对 summed loss、全局有效 token 分母、梯度累积、
  `IGNORE_INDEX`、TP vocab shard 和 PP microbatch 的缩放关系。
- **数据语义**：普通 `Batch` 的目标由模型预处理构造；packed 样本已对齐目标，
  不重复 shift。检查文档边界、positions、padding mask 和 CP 分片顺序。
  保持 map 数据的过滤、shuffle、DP 分片、repeat 顺序与 streaming 路径的区别。
- **模型变换**：保留参数身份或有意处理身份变化、冻结状态、训练模式、共享权重、
  buffer dtype 与 state-dict 键；重计算不得改变 token dispatch 的路由契约。
- **恢复与退出**：检查模型、优化器、scheduler、数据游标和 trainer 状态一致性；
  同时覆盖部分初始化失败、验证模式恢复和资源清理。

当参考 torchtitan 或 FSDPTurbo 时，先查看 `docs/llmtuner_upstream_map.md`、
`docs/torchllmtuner_design.md` 和相关源文件。用户指定本地上游路径时使用该路径；
否则先确认实际可用位置。对齐数值、布局、通信和生命周期语义，适配本仓库的
HF 与普通 dataclass 架构。记录来源路径、适配差异和未验证条件。

## 4. 重构与命名

先确定可证明的收益：职责错位、重复逻辑、模糊名称、循环依赖或无消费者包装。
合并只有单一调用方且不承载独立契约的薄封装；保留具有独立责任的构件。

- 使用四空格、Python 3.10 兼容语法、类型注解和 88 字符行宽。
- 文件及函数使用 `snake_case`，类使用 `PascalCase`，常量使用 `UPPER_CASE`。
  FSDP、TP、PP、CP、EP、FQN 和明确上下文中的张量维度缩写可保留。
- 名称表达职责，如 `loader_builder`、`state_dict`、`move_tensor_tree`；
  不为统一外观重命名已准确的公共 API。
- 重命名或搬运后同步更新实现导入、公共导出、测试、示例和文档，搜索旧引用。
  用户要求删除旧名时移除别名和兼容壳；上游来源记录中的原始名称保留。
- 全局审查更新 `docs/file_review.md`；移动、合并、删除文件后核对清单，
  将“已阅读”“静态检查通过”“硬件数值验证”分开记录。

## 5. 验证与交付

行为修改增加能暴露具体问题的测试，例如梯度等价、状态恢复、布局转换或失败路径。
纯搬运和命名调整优先复用已有回归测试。CPU 单测位于 `tests/unit_tests/cpu/`，
使用 `test_<behavior>.py` 和 `test_<expected_behavior>`；环境门控复用
`tests.caps.require_env`。

代码修改完成后执行仓库质量门禁：

```bash
pytest -q --disable-warnings
pre-commit run --all-files
python -m compileall -q llmtuner
git diff HEAD --check
```

局部迭代可用 `ruff check llmtuner tests`、`ruff format --check llmtuner tests`
和受影响测试。pre-commit 会自动修改文件，检查其 diff，修正后重新运行。
文档或技能修改按范围验证 frontmatter、引用和格式，无需运行训练测试。

collective、mesh 或并行数值修改还需选择 `tests/integration_tests/` 中相应的
`torchrun` 脚本，按脚本实际参数和目标硬件执行；这些脚本不由 pytest 自动收集。
缺少硬件时说明未运行的验证，不把 CPU 通过当作混合并行组合已验证。

交付说明实际改动、验证结果、剩余限制和公共名称迁移。只有用户已授权时才
commit/push；提交使用仓库的 Conventional Commit 风格，并确认分支与推送结果。
