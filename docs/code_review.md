# llmtuner 全局代码审查

本报告记录本轮与上一轮连续审查的结果。静态语法与 Ruff 检查覆盖
`llmtuner` 当前全部 151 个 Python 文件；逐文件阅读清单包含本次新增的配方模块，最新清单见
[file_review.md](file_review.md)，检查训练调用链、参数转换、恢复状态和异常清理路径。静态检查与 CPU 回归测试不能替代多设备数值等价测试，
本报告不将未运行的硬件组合记为已验证。

## 范围与职责

### Pattern 解析与匹配责任（2026-10-14）

- resolve_module_patterns 每个 pattern 先编译一次，再匹配全部模块，
  删除每个名字重复访问编译缓存；空 inventory 也检查 pattern 语法。
- 新增空 inventory 的非法范围回归，修正 attention head 整除说明。

### 验证指标非有限结果（2026-10-14）

- finalize_validation 在全局归约、token 归一化后拒绝 NaN/Inf loss，
  不向 metrics 发布无效结果；PP 最后阶段复用相同入口。
- 三项非有限回归确认日志未发布；验证专项 28 passed。多设备
  失败传播不由单进程回归证明。
- 完整 CPU 1743 passed、2 skipped；目标 pre-commit、compileall 与
  diff 检查通过。

### 验证预处理元数据边界（2026-10-14）

- 公共 preprocess 在调用模型前过滤 num_valid_tokens，使 PP 与非 PP
  验证入口均遵守同一边界；删除非 PP 验证中的重复过滤。
- 过滤创建字典，保留调用方批次和 tensor 身份；直接入口回归覆盖。

### 并行编排定义模块导入（2026-10-14）

- parallelize 的 AC/TP/EP/CP/PP/FSDP 实现引用直接指向定义文件，
  家族包 __init__ 保留外部公开 API，内部编排不依赖聚合导出。
- 源码清单核对为 151 文件、151 表项，无遗漏、失效路径和重复项；
  更新 contracts 表项包含 TrainerBatch。清单覆盖不等于硬件验证。
- parallel/Trainer 专项 370 passed、2 skipped；目标 pre-commit、
  compileall 和 diff 检查通过。

### FSDP prefetch 单元身份契约（2026-10-14）

- 直接调用拒绝重复 module 对象，避免自身被列入窗口且同一单元被
  后面的重复条目覆盖；检查先于任意窗口 setter。
- 回归确认重复单元失败前不配置任何窗口；正常窗口回归复用。
- FSDP 专项 37 passed；完整 CPU 1739 passed、2 skipped；目标
  pre-commit、compileall、diff 检查通过。硬件预取行为未验证。

### Loader 抽象批次类型（2026-10-14）

- BaseDataLoader 的迭代接口返回 Batch | TrainerBatch，覆盖 synthetic
  与 Grain 两种实现；具体子类保留精确返回类型。
- 批次元数据说明随 TrainerBatch 定义放在 contracts，删除 collators
  中迁移后留下的悬空注释。仅类型与说明变化，无运行逻辑变更。
- datasets/Trainer 专项 385 passed；目标 pre-commit、compileall、
  diff 检查通过。本轮未重复完整 CPU 套件。

### TrainerBatch 契约归属（2026-10-14）

- TrainerBatch 移至 datasets/contracts.py，Trainer、runtime、loader
  和多模态 collator 直接引用定义模块；datasets 公共导出保持。
- collators 不再声明该类型为自身公共 API，批次共享类型与数据
  处理实现职责分开；不增加兼容包装。

### 配置定义模块依赖归并（2026-10-14）

- Trainer、topology、parallelize、EP/PP/CP/FSDP 入口与 loader builder
  从配置定义模块导入，公共聚合导出留给外部 API；TYPE_CHECKING 保留。
- parallelize 说明修正 TP→EP→CP 次序、PP 完整阶段及权重迁移不重新
  初始化的语义；阶段表仍是执行顺序的单一来源。
- 完整 CPU 1738 passed、2 skipped；tracked-file pre-commit（排除用户
  示例）、compileall 和 diff 检查通过。

### FSDP pattern prefetch 顺序（2026-10-14）

- 复核纠正：prefetch 原实现已按 named_modules 注册顺序构造，上一轮
  仅提取变量，未修复顺序 bug；已撤去冗余变量，保留说明。
- 注册顺序独立于 pattern 插入顺序，但不能推断任意模型 forward 的
  实际执行顺序。FSDP contract 专项 36 passed 不证明该执行顺序。

### 并行阶段类型契约（2026-10-14）

- STAGE_ORDER 与 PP_STAGE_ORDER 现在显式使用 StageName 类型，阶段表、
  PP 子序列和执行器的参数契约一致；不再以任意字符串表示受限阶段名。
- 保留 stage_enabled 作为唯一 compile 条件入口，避免执行器复制条件。

### Loader 批次布局单次解析（2026-10-14）

- num_rows_per_batch 在 collator 构建后解析并校验一次，checkpoint
  布局与 Grain batching 使用同一值，删除重复调用。
- 修正 rank 注释：loader 记录恢复 ownership，不能推断输入图的
  分片策略。实际图回归确认 collator 方法仅调用一次。
- 数据 pipeline 专项 189 passed；完整 CPU 1738 passed、2 skipped；
  tracked-file pre-commit（排除用户示例）、compileall、diff 检查通过。

### Trainer 状态注册顺序（2026-10-14）

- step/ntokens_seen 在构建 checkpointer 前初始化，注册的 Stateful
  对象在装配期间也可提供完整 state_dict，不依赖后续赋值。
- 实际 Trainer 装配回归在 checkpointer 入口读取状态，确认 step-0
  初始值；未改变训练与恢复入口。
- 完整 CPU 1737 passed、2 skipped；tracked-file pre-commit（排除用户
  示例）、compileall 和 diff 检查通过。

### 模型配置离线入口契约（2026-10-14）

- build_model_config 明确以 arch_overrides 是否为 None 选择构建路径；
  空字典表示使用离线架构默认值，不再意外调用 from_pretrained。
- 更新函数说明，回归确认显式空字典不访问 Hub、不修改调用方字典。
- HF 模型专项 34 passed；完整 CPU 1736 passed、2 skipped；tracked-file
  pre-commit（排除用户示例）、compileall、diff 检查通过。

### Checkpoint 数据源状态契约（2026-10-14）

- `_build_checkpointer` 不再条件性注册 dataloader；缺失数据源在装配
  checkpointer 前明确失败，正常训练始终保存可恢复的数据游标。
- 新增缺失装配回归，避免 `data_iterator` 与 checkpoint states 对数据源
  的要求不一致。
- Trainer 专项 91 passed；完整 CPU 1735 passed、2 skipped；tracked-file
  pre-commit（排除用户示例）、compileall 和 diff 检查通过。

### Trainer 数据源入口归并（2026-10-14）

- data_iterator 仅返回已装配 dataloader 的迭代器，删除未注册到
  checkpoint 的随机源备用路径；装配缺失明确报错，不静默切换语料。
- 纯数据测试显式使用 build_dataloader 构建源；独立源确定性回归
  与缺失装配回归覆盖该入口，Trainer 保留 None 表示部分初始化。
- Trainer 专项 90 passed；完整 CPU 1734 passed、2 skipped；tracked-file
  pre-commit（排除用户示例）、compileall、diff 检查通过。

### CLI 与 Trainer 配置解析职责（2026-10-13）

- parse_config 仅解析/组合参数，删除启动前的 HF autofill 读取；
  模型架构由 Trainer 装配解析并同步尺寸，与程序化入口一致。
- 保留显式 auto_fill_model 供运行前预览；设计文档更新配置路径与
  解析顺序。本地 HF 实际启动回归确认 CLI 不读配置、装配只读一次，
  synthetic loader 使用解析后的词表。配置/Trainer 专项 272 passed。
- 完整 CPU 1733 passed、2 skipped；tracked-file pre-commit（排除用户
  示例）、compileall 和 diff 检查通过。

### 编译入口模型与 mesh 契约（2026-10-13）

- per_block 模式先确认模型持有 .layers 模块，再执行全局编译设置；
  非法输入不再在 async TP 设置后才泄漏 AttributeError。
- async TP 在注册 symmetric memory 前要求单轴且 size > 1 的 TP
  mesh，直接调用入口与配置的 TP 度数要求一致。
- CompileConfig 从定义模块导入；三项失败路径回归确认检查先于
  全局设置或 group 注册。编译专项 19 passed、2 skipped。
- 完整 CPU 1732 passed、2 skipped；tracked-file pre-commit（排除用户
  示例）、compileall、diff 检查通过；真实 async TP 硬件验证未执行。

### FLOPs 入口职责归并（2026-10-13）

- 删除无生产调用方的 factory.num_flops_per_token 与导出，统一为
  flops.flops_per_token(arch, seq_len=...)；不保留旧名或兼容包装。
- runtime assembly 已传入模型实际解析的 text config，统计路径不再
  提供重复解析模型配置的入口；更新包说明、metrics 引用与上游映射。
- 原转发测试改为直接验证运行序列长度影响 attention FLOPs；模型、
  FLOPs 与 Trainer 专项 141 passed。
- 完整 CPU 1729 passed、2 skipped；tracked-file pre-commit（排除用户
  示例）、compileall 和 diff 检查通过；旧实现入口引用搜索无残留。

### EMA 固定范围与更新入口（2026-10-13）

- 补齐固定追踪范围的反向检查：移除/冻结追踪参数或删除匹配 buffer
  时明确失败，更新和 cold start 均先校验所有组再修改平均值。
- 将仅由 step 调用的 _update 合并到 step，保留独立的 decay 公式
  和共享追踪范围检查；删去无消费者的私有更新包装。
- 六项回归覆盖三个范围缩减事件及两个入口，验证前组平均值不变。
- EMA 专项 25 passed；完整 CPU 1729 passed、2 skipped；tracked-file
  pre-commit（排除用户示例）、compileall 和 diff 检查通过。

### 单文件 HF checkpoint 适配（2026-10-13）

- 适配器读取 model.safetensors header 的 key 集合，与分片索引复用
  相同的覆盖校验及绑定 lm_head 省略规则；不加载完整权重到内存。
- 去除 model. 前缀时显式拒绝映射冲突，防止字典推导静默覆盖权重。
- 真实 safetensors/DCP 回归覆盖保存和省略 lm_head 的单文件 checkpoint；
  适配器专项 9 passed，多设备 PP 读取未在本轮验证。
- 累计完整 CPU 1723 passed、2 skipped；tracked-file pre-commit
  （排除用户示例）、compileall 和 diff 检查通过。

### Meta 物化共享参数回归（2026-10-13）

- 最小复现确认 to_empty 会拆开两个模块共享的 meta Parameter；
  物化前记录注册路径，物化后恢复参数和 buffer 的共享引用。
- 对比绑定模型的参数数量、checkpoint 加载输出与反向梯度；另测
  持久化 buffer 共享关系。优化器在物化后创建，捕获恢复后的参数。
- 分布式 DTensor 共享状态仍需硬件验证；只恢复输入模型已有的
  对象共享关系，不推断已经被其他变换拆开的绑定。
- 模型专项 33 passed；完整 CPU 1720 passed、2 skipped；tracked-file
  pre-commit（排除用户示例）、compileall 和 diff 检查通过。

### Meta 模型 buffer 物化职责（2026-10-13）

- 持久化 meta buffer 留给 checkpoint 填充；非持久化 buffer 才需要
  构造时重建，未知状态在 to_empty 前拒绝，避免失败后模型已被物化。
- 已有真实 buffer 及重复注册路径的值在 to_empty 后恢复；RoPE
  重建先准备后安装，原有权重加载数值回归继续覆盖。
- resolve_model_class 删除对同一 transformers 模块的重复 getattr。
  新增持久化加载、真实 buffer 保留与失败前状态回归。
- 累计完整 CPU 1718 passed、2 skipped；tracked-file pre-commit
  （排除用户示例）、compileall 和 diff 检查通过。分布式 meta/DTensor
  物化未在硬件环境验证。

### EP 布局预检与 shared expert 探测归并（2026-10-13）

- 替换入口先探测所有层，再转换合法候选；后续层专家布局非法时
  不再留下已替换的前层，也不提前注册辅助 loss。
- shared expert 不支持组合检查先于 router/aux loss 构造，拒绝路径
  不增加 AuxLoss 计数；TP/EP 共用 hf_inspection 的属性与路径探测。
- 回归核对原 block 身份、moe_enabled 与辅助 loss 计数；EP/TP
  专项 95 passed。其他构造失败尚不承诺完整事务回滚。
- 完整 CPU 1715 passed、2 skipped；tracked-file pre-commit（排除用户
  示例）、compileall、diff 检查通过；TP 的 EP 探测保留函数内导入。

### EP 转换状态归属复查（2026-10-13）

- FP32 buffer 恢复只处理 MoE 与 AuxLoss 自有统计状态，不再改写
  复用 shared expert 的模型 buffer dtype。
- shared expert 使用 None 判断，避免合法但真值为假的模块被丢弃；
  build_aux_loss 保留两种 router 共用的构造职责，补全返回类型。
- 新增 shared buffer dtype 和空 Sequential 身份保留回归；EP 模型
  替换专项 37 passed。模型替换失败的完整事务性仍需继续核对。
- 累计完整 CPU 1714 passed、2 skipped；tracked-file pre-commit
  （排除用户示例）、compileall 与 diff 检查通过。

### Vocab-parallel embedding 复查（2026-10-13）

- 将笼统的 Embedding 类名改为 VocabParallelEmbedding，删除旧名和
  未完成的命名 TODO；明确该独立组件尚未自动接入 HF TP 路径。
- 分片外 token 不再钳制为本地边界行，而是在 lookup 前排除，避免
  scale_grad_by_freq 错误计数和 max_norm 更新不相关行；空 shard
  返回零输出并保留空参数梯度。
- CPU 回归比对真实 nn.Embedding 的参数、梯度与输出；AC 配置
  导入同步改为定义模块。硬件数值与编译路径未验证。
- 本轮累计完整 CPU 1712 passed、2 skipped；tracked-file pre-commit
  （排除用户示例）、compileall 与 diff 检查通过。

| 层 | Python 文件数 | 审查关注点 |
| --- | ---: | --- |
| 包入口 | 3 | 公共导出、CLI 入口 |
| `accelerator` | 7 | 设备选择、collective、嵌套 tensor 转移 |
| `components` | 21 | loss、优化器、EMA、checkpoint、metrics、profiler |
| `config` | 19 | 字段校验、CLI 配置组合、并行能力限制 |
| `datasets` | 24 | DP ownership、packing、标签对齐、多模态处理 |
| `memory` | 3 | activation checkpoint 与重计算区域 |
| `models` | 28 | HF 适配、attention、投影、MoE routing 与 dispatch |
| `parallel` | 30 | topology、阶段顺序、TP/FSDP pattern、CP/PP/EP 转换 |
| `trainer` | 9 | token 归约、梯度累积、验证、退出与恢复 |
| `utils` | 7 | 随机数、GC、日志、设备监测 |

继续沿用现有职责边界：配置只描述策略，parallel 应用模型变换，trainer 编排训练。
`trainer.py` 的 optimizer step 保持完整实现；`metrics.py` 保持单文件；各并行家族
保留同名入口。注释中仍指向 `swap.py`、`probe.py`、`tp.py`、`validate.py`
等旧模块的引用已更新。

## 已修复的行为问题

| 问题 | 影响与修复 | 回归测试 |
| --- | --- | --- |
| scheduler 恢复仅更新报告值 | optimizer 可能使用初始 LR；同步恢复 parameter group LR | `test_lr_scheduler.py` |
| TP 替换参数默认可训练 | 冻结权重被重新启用；dense、routed/shared expert 保留 `requires_grad` | `test_tp.py`、`test_tp_moe.py` |
| EP 转换丢失冻结状态 | router 和 expert 的冻结策略被改变；逐个保留源参数状态 | `test_ep_model_replacement.py` |
| FSDP pattern 覆盖参数重复 | `mp_policy` 等覆盖值触发重复 keyword 错误；先合并默认与覆盖配置 | `test_fsdp_contract.py` |
| grouped router 缺少容量限制 | top-k 可能选到被屏蔽的专家；构造时验证组数、容量和 top-k | `test_moe_routing.py` |
| CastLinear 转换重新初始化大权重 | 增加构建期内存与 RNG 消耗；在 meta 上构造再复用原参数，并保留模式 | `test_cast_linear.py` |
| HF tied head 导出无条件删 key | 索引包含 head 时导出缺失权重；只在索引省略 head 时删除 | `test_state_dict_adapter.py` |
| HF import 修改输入字典 | 调用者映射产生隐式副作用；在浅拷贝上补齐 tied head | `test_state_dict_adapter.py` |
| 64 位 seed 无法进入 NumPy | PP 派生 seed 或大 seed 在初始化时失败；NumPy 使用 modulo 2^32 | `test_utils.py` |
| `set_seed` 调用不存在的 accelerator API | 当前 PyTorch 报 AttributeError；统一委托给活动设备的 seeding 路径 | `test_utils.py` |
| 标签 shift 对非连续 tensor 使用 view | 合法输入因 stride 不兼容报错；使用 reshape，显式拒绝非正 seq_len | `test_utils.py` |
| validation 覆盖子模块模式 | 冻结子模块原有 eval 状态丢失；逐模块保存并恢复 | `test_validate.py` |
| GC 未恢复、初始化失败泄漏状态 | 影响同进程后续工作；正常退出与初始化失败均恢复自动 GC 状态 | `test_utils.py` |
| Trainer cleanup 短路、覆盖训练错误 | 后续资源未释放或原始错误丢失；释放所有资源并保留原始训练错误 | `test_trainer.py` |
| CLI cleanup 失败跳过 PG 销毁 | process group 残留；PG 销毁置于 finally，保留 entry-point 错误 | `test_trainer.py` |
| profiler 部分启动后失败 | `__enter__` 失败不会自动调用 `__exit__`；显式停止已启动 trace | `test_profiler.py` |

## 性能优化

`compile_module_pattern` 增加容量为 256 的 LRU cache。TP/FSDP 在遍历模型时复用
相同 pattern 的编译结果，避免为每个模块重新解析范围并构造正则表达式。
该修改不改变匹配语义；本轮没有进行性能基准，因而不报告加速比例。

## 并行能力的实际边界

以下为当前实现明确声明的限制，不应被描述为已完整迁移或已在硬件验证：

- `custom` fine-grained FSDP 使用 PyTorch 公共 FSDP2 状态引擎；独立
  `hook_module` 被拒绝。它不等价于 FSDPTurbo 的全部私有状态实现。
- MC2、DeepEP、HybridEP 当前主动拒绝启用；配置枚举中的名称不代表后端可用。
- KCP 依赖 `fla` 和 Ascend KDA 内核，且不支持 cache 或 packed cu_seqlens。
- EP checkpoint 会物化并 all-gather 完整 expert state，可能提高保存峰值内存。
- 有限数据集在 DP > 1 时主动拒绝 `repeat=False`，以避免不同 rank 耗尽顺序
  不同而挂住 collective。
- TP/CP/PP/EP/FSDP 组合的实际 forward、backward、checkpoint reshard 正确性
  需要对应 `tests/integration_tests/` 脚本在目标设备执行。

## 验证

后续审查补充：

- FSDP ignored pattern 选中父模块时，排除其整个子树，避免忽略参数与
  分片单元选择的语义不一致；同名前缀的其他模块不受影响。
- EMA cold start 与普通更新共用 tensor identity 校验。cold start 在写入前
  检查所有 parameter/buffer，替换后的 tensor 会报明确错误，不插入空状态、
  不部分覆盖原平均值。
- 多模态 placeholder 数量必须与 token count 一一对应，且 count 不得为负。
  不一致时报告 ValueError，避免多余 count 被静默丢弃。
- 远程图像读取在解码前检查 HTTP 状态，并通过 context manager 关闭响应。
  resize 拒绝非正几何参数、反向像素预算和无法容纳最小 patch grid 的预算。
- 视频采样在打开文件前拒绝非有限或非正 FPS，以及无效的帧数上下界。
  参数错误不再进入解码失败的样本丢弃路径。

质量门禁：全量 CPU pytest、`pre-commit run --all-files`、Ruff lint/format、
`python -m compileall -q llmtuner`、`git diff --check`。

最新全量结果：`1119 passed, 10 skipped`。上述质量门禁均通过。

分布式测试需根据脚本要求选择 rank 数、设备和模型。全量 CPU 测试中的跳过项
以及尚未运行的 torchrun 脚本不能视为通过。本轮不发布或推送修改。

## 逐文件审查追加修复

- pipeline 能力探测容忍旧版缺失 eval；MPI 发布 LOCAL_RANK；硬件监控查询当前设备。
- ParallelConfig 统一委托家族校验，去除重复分支；数据配置拒绝未知 packing。
- TP 显式 pattern 先于默认 pattern 匹配；EP 完整校验 down projection 的 D/F 维度。
- 可选 torch_checkpointing 的 HF 最终导出修复 item_specs 函数被局部变量遮蔽。
- 优化器状态初始化失败时恢复原 LR 与梯度；合成数据接受完整 64 位种子。
- KCP 增加 context_parallel_kcp_size，修复合法配置无法启用 CP 的可达性问题。
  例如 strategy='kcp'、kcp_size=4、其余两个 CP 轴为 1；不支持 packed、
  headtail、TP 或 KDA/softmax 混合模型。Ascend 算子数值仍须目标设备验证。

## 日志生命周期审查

`utils/logging.py` 使用 handler 所有权标记代替名称初始化缓存，支持 handler 被移除后的重新初始化，保留应用添加的 handler。重复获取 logger 更新日志级别。自有 console handler 按 record 标记去重，保留向应用/root handler 的传播；应用 handler 的输出和过滤仍由应用控制。分布式 rank 查询失败直接回退环境变量，避免在查询异常中再次记录日志引起递归。回归覆盖层级去重、caplog、应用 handler、动态 rank、初始化恢复与异常回退。

日志后续优化：并发初始化加锁，fork 子进程重建锁；formatter 复制记录并复用过滤阶段查询的 rank。rank 回退仅解析 RANK，避免无关 WORLD_SIZE/LOCAL_RANK 配置错误阻塞日志，也不主动导入分布式/设备模块。set_log_ranks 显式校验整数与非负约束。新增并发安装、单次 rank 查询、原记录不变与环境回退回归。

日志 rank 获取调整为复用 accelerator.distributed.is_distributed/get_rank；启动前或查询失败保留 RANK 回退，不在异常路径记录日志。此前直接查询已加载 torch.distributed 的实现已替换。

日志新增文件输出：get_logger 保留位置参数 log_level，新增关键字 log_file/file_mode，默认 UTF-8 追加、无 ANSI 颜色，只有 rank 0 写入。RankFormatter 共享格式化职责；同一规范路径共享 FileHandler，过滤按文件目的地去重，覆盖模式只首次打开截断。回归覆盖中文、追加/覆盖、层级去重、并发初始化、rank 门控、traceback、模式冲突和关闭后重新配置。

日志 handler 所有权改用类型识别；记录状态集中为 RecordState，弱引用表随 LogRecord 生命周期释放，不修改应用收到的原始记录。控制台和文件 handler 分别装配自己的 formatter/filter，保留层级去重与 rank 策略。

日志进一步整理：锁、文件 handler 弱缓存、记录弱缓存与 rank 配置集中为 LoggingState 实例属性，通过延迟缓存访问；移除模块级 LOGGER_LOCK/FILE_HANDLERS/RECORD_STATES/LOG_FORMAT，格式字符串内联于 RankFormatter 构造，文件 handler 查找创建合并进 get_logger。全量 CPU 测试 1161 passed、10 skipped，pre-commit/compileall/diff 检查通过。

日志封装合并：纯文本和彩色 formatter 统一为 RankFormatter(color=...)；MainProcessFilter/FileLogFilter 合并为 DestinationFilter(path=...)，统一 rank 策略与目的地去重。get_logger 使用局部 config 复用状态访问；保留 handler 类型用于所有权识别，共享记录状态方法用于单次 rank 查询。

## CLI 调用链复查

新一轮全局审查进行中，尚未完成逐文件重新覆盖。入口修复 seed checkpoint 前置校验与进程组 teardown 异常遮蔽；Trainer 构造失败调用 close 释放部分装配资源，并保留原始失败。新增失败路径回归，未增加 Trainer 文件拆分。

运行时装配在已有进程组下查询实际 rank/world size，支持未设置 torchrun 环境的程序调用方；LOCAL_RANK 仍来自环境。microbatch 过滤统计字段使用浅拷贝，保留 loader 原始对象。新增两项回归，1169 passed、10 skipped。

数据与 EMA 复查：TextCollator 在输出存储上统一将 padding label 置为 IGNORE_INDEX，统计有效 token 前完成屏蔽，保留源数组；RandomTokenSource 在构造时校验 vocab_size/batch_size/seq_len 为正；EMA 复制调用方模型分块列表。新增 8 项回归，全量 1177 passed、10 skipped。

优化器复查：修复参数/FQN 元数据错配被宽松 zip 静默截断，保存和恢复均严格检查。容器保存模型分块列表副本。独立元数据测试 3 passed；named-parameter 容器测试受当前 PyTorch 能力限制跳过，不视为已验证。

模型构件复查：eager RoPE 显式 position 检查上下界，防止负位置读取缓存尾部；固定块 attention mask 构造拒绝非正块大小。新增 4 项回归。

HF/Tokenizer 继续复查：已阅读 models/hf/model_setup.py、factory.py、flops.py、attention_backend.py 和 components/tokenizer/base.py、huggingface.py。修复复合配置解包遗漏 compute_dtype/experts_implementation；独立 chat_template.jinja 无 tokenizer_config.json 时仍加载。新增 4 项配置回归和 1 项模板回归。本轮全局逐文件复查仍未完成，多设备数值验证未执行。

配置继续复查：parallel/module_patterns.py、config/field_checks_config.py、compile_config.py、validation_config.py、optimizer_config.py、observability_config.py、model_config.py。修复 optimizer learning_rate/weight_decay/eps 接受非有限值，LRSchedulerConfig 提前拒绝非法 decay_type。新增 10 项回归，1199 passed、10 skipped；pre-commit、compileall、diff 检查通过。全局逐文件复查仍在进行。


本轮继续复查 config/context_parallel_config.py、tensor_parallel_config.py、expert_parallel_config.py、pipeline_parallel_config.py、fsdp_config.py、parallel_config.py，以及 parallel/compatibility.py、stages.py、parallelize.py、topology.py，accelerator/tensor_tree.py、distributed.py、collectives.py。检查了 CP 轴乘积、EP 复用 dense 区域、loss/batch mesh 范围及 PP 每分块阶段顺序。上述阅读与 CPU 模拟不构成多设备组合数值验证。

继续复查 trainer/runtime/pipeline.py，并重查 validation.py 和 datasets/dataloader.py 的失败路径。修复恢复游标/验证异常被 loader close 异常掩盖；保留原始异常并记录清理失败，正常路径关闭失败仍抛出。PP 拆分新增非空行数及整除检查，避免字典丢弃尾行或 Batch 产生不完整 microbatch 集合。新增 9 项回归，相关数据与验证测试 100 passed。全局逐文件审查仍在进行，尚未覆盖的文件不得按历史清单计作本轮完成。

补充阅读 config/checkpoint_config.py、training_config.py、data_config.py、activation_checkpoint_config.py。新增两项 PP 验证清理失败回归，本次共新增 11 项测试。最终全量 CPU 测试 1210 passed、10 skipped；pre-commit（含两个未跟踪的新测试文件）、compileall、git diff HEAD --check 均通过。未执行多设备集成测试；本轮全局逐文件审查仍未完成。


本轮完整复查 trainer/trainer.py（保持单文件），components/loss.py、metrics.py、profiler.py、checkpointer/base.py、distributed_checkpoint.py、torch_checkpointing_backend.py；继续阅读 utils/monitoring.py、models/hf/state_dict_adapter.py、models/common/moe/layer.py、bias_update.py、accelerator/spmd.py、capabilities.py。核对了梯度累积 token 分母、TP 复制梯度归约、loss reporting、optimizer/scheduler/EMA 更新与 checkpoint 时序。

修复：普通 entropy 改用稳定 log_softmax 并屏蔽 0×(-inf)，保留全屏蔽分布的 NaN；checkpoint close 即使 staging 失败也等待保存并清理，保留第一个异常且关闭后幂等；staging future 在等待前取出，避免重复抛出失败；单进程 async checkpoint 不创建未初始化的 Gloo group，真实 CPU 异步保存/恢复通过；metrics 初始化失败释放已建 sink、关闭时释放全部 sink；profiler trace 退出失败仍尝试 OOM snapshot，保留原始训练异常并清空句柄。新增 11 项回归，组件定向测试 146 passed。

尚待追踪：HF adapter 的全索引 key 覆盖检查与 PP 本地子集的关系，需要核对 PP 的 HF load/export 调用路径；SPMD 单进程空上下文在嵌套分布式上下文内的语义需要继续检查。以上为待确认项，未作未经验证的放宽。全局逐文件复查尚未完成，未运行多设备数值验证或实际 torch_checkpointing 后端 round-trip。

本次最终验证：1221 passed、10 skipped；pre-commit（包括未跟踪的新测试文件）、compileall、diff 空白检查通过。新建的 examples/native_alltoall.py 不属于本次修改，未纳入审查或变更。源码本轮覆盖 85/149，剩余清单见 file_review.md。


继续复查 config/llmtuner_config.py、accelerator/device.py、parallel/attention_heads.py、pipeline_parallel/partition.py、pipeline_parallel/pipeline_parallel.py、models/hf/model.py。本轮累计阅读 91/149；上方待审清单中的这六个文件已完成阅读，余 58 个。HF adapter × PP 待确认项已修复：全索引覆盖校验使用所有 rank key 并集，输出仍只保留本地张量；本地及全局遗漏仍拒绝。新增三项 CPU 回归和 tests/integration_tests/hf_checkpoint_pp.py，两个 CPU/Gloo 进程真实 indexed HF 加载与导出逐参数比对通过。首次 standalone rendezvous 因本机地址解析连接失败，改为显式 127.0.0.1/lo0 成功；未验证 PP schedule 前向/反向或 GPU 通信。修复中间 PP chunk 的 embedding/head 同为 Identity 时 None is None 被误判为共享权重，新增回归。SPMD None 当前明确定义为 no-op，嵌套继承外层上下文是现有契约，未作改变。全局逐文件审查仍在进行。

本次最终验证：1225 passed、10 skipped；pre-commit（含新增 integration 文件与未跟踪测试）、compileall、diff 检查通过。两个 CPU/Gloo rank 的 HF checkpoint PP round-trip 实际通过；其余硬件组合未验证。


本轮继续完整阅读 models/common/moe/routing.py、experts.py、models/common/tp_projections.py，累计 94/149，剩余 55 个。修复 GroupedExperts FP16 输入被强制转 BF16 导致尾数精度损失：融合路径仅用于 BF16，FP16 保留逐专家原 dtype 计算；现有 loop/fused-request 精确等价测试增加 FP16。QuantileBalancer 拒绝 num_bins<=0，GroupedExperts 拒绝空 dim/hidden_dim/num_experts，新增五项输入回归。全局审查仍未完成，GPU grouped GEMM 性能和多设备 MoE 数值仍未验证。

补充完整阅读 components/tokenizer/multimodal.py、datasets/multimodal/placeholders.py，核对 added-token ID 校验、placeholder 数量与负计数拒绝；未发现本次需修改的具体问题。当前本轮阅读 96/149，待审 53 个。定向模型测试 41 passed；pre-commit、compileall、diff 检查通过。

本次全量 CPU 测试：1230 passed、10 skipped。GPU grouped GEMM 及多设备 MoE 组合尚未验证；全局逐文件审查继续进行。


## 剩余源码审查（CLI 起点，本轮覆盖 149/149）

- 多模态 collator 对输入 padding mask 屏蔽 labels，修正有效 token 分母；拼接后
  操作不会修改输入样本。处理器复制 texts，避免数据重试或重复遍历改变源行。
- TP 启用且无 mesh 时明确拒绝，mesh 大小与配置不一致时在参数变换前拒绝。
- pattern FSDP 复用既有 reshard 策略解析，并应用 symmetric memory scope，
  每个 unit 的显式 overrides 仍优先。参考 torchtitan/distributed/fsdp.py 的策略
  解析与通信设置；CPU 测试仅验证配置传递，不验证通信性能或设备内核。
- 更正包说明中的旧文件名，并修正 shared expert 参数名称集合的命名和注解。
- 所有源码均已逐文件阅读；覆盖率表示阅读范围，硬件验证限制见 file_review.md。

最终验证：全量 CPU 测试 1237 passed、10 skipped；pre-commit --all-files 与三个新增
文件的单独 hooks 均通过；compileall 与 git diff HEAD --check 通过。之前执行的
两进程 CPU/Gloo HF indexed checkpoint PP 加载/导出验证通过，本阶段未重复执行，
不将其作为 PP schedule 前向/反向验证。

提交前补充修正 tests/unit_tests/cpu/parallel/test_model_compile.py 的旧模块导入，
恢复此前被导入异常错误跳过的运行时测试。编译模块定向测试 12 passed、2 skipped；
最新全量测试 1244 passed、3 skipped。pre-commit、compileall 和 diff 检查通过。


## 跨模块复查：编译配置、视觉网格与真实通信

- 复现 regional inductor 进程全局开关泄漏：模型 A 选择 flex/aot_eager 后，
  模型 B 选择 inductor 仍读取开启状态。改为在各模型模块上保存设置，HF attention
  adapter 显式读取；CP 路径只标注 flex 计算，不把通信包入该区域。
  两模型隔离回归与 CP 标注回归通过；CPU aot_eager 的整模型和分块编译实际前向、
  全参数梯度均与 eager 一致。真实 CUDA regional inductor 尚未执行。
- 多模态 token 估计拒绝非完整 merge 网格；patch 提取检查空张量、维度、正大小
  与空间整除。raster 仍只要求 raw patch 整除，允许不按 block merge 对齐。
  构造阶段拒绝零 patch 大小；图像 resize 在除法前校验几何。
- 复现 smart_resize 在窄长图像上超过像素预算：32×6144、factor=32、预算
  65536 曾输出 32×3520=112640。上游算法的合法结果保持不变，仅越界时搜索
  可行整数网格；预算中没有任何 factor 对齐网格时明确拒绝。这是相对 torchtitan
  hf_datasets/multimodal/utils/image.py 的有意修正。多模态定向测试 98 passed。
- 真实 CPU/Gloo：TP、EP、TP×FSDP、EP×FSDP、PP、Ulysses CP 六个集成脚本通过。
  Ulysses 脚本旧断言期待 ValueError，更新为现有 UnsupportedCombinationError
  后重跑通过，未修改生产错误类型。CP 此处只验证完整模型前向与 all-to-all
  自身反向，不声称完整 flex attention backward 已通过。


补充 EP×pattern FSDP 数值回归：增加 ep_fsdp_equivalence.py 的
`--pattern-fsdp` 模式（实际路径模式 `model.model.layers.{*}`）。修复前专家权重
落在 size=4 dense mesh，loss 311.819031 对参考 311.874283，专家梯度出现
0.14～0.20 量级误差；修复后专家位于 size=2 sparse mesh，loss 311.874268，
所有路由及专家梯度通过 1e-4 容差。普通与 pattern 路径共用参数级 mesh 分配，
各 unit 的 mesh 和 shard_placement_fn 显式 override 仍优先；缺少 sparse mesh
在变换前拒绝。参考 torchtitan/distributed/fsdp.py 的专家 placement 语义。
复现日志为 /tmp/llmtuner-review-ep-pattern-fsdp-before.log，修复后日志为
/tmp/llmtuner-review-ep-pattern-fsdp-after.log；日志不是仓库提交文件。

普通 EP×FSDP 在复用 placement 逻辑后重跑通过。修正 tests/caps.py 的
optimizer param_names 能力探测，使用与生产代码一致的显式参数名 group；此前
错误要求 PyTorch 自动生成名称导致整组跳过，恢复后 20 项优化器测试通过。

本阶段最终全量测试：1311 passed、2 skipped；pre-commit --all-files、
compileall、git diff HEAD --check 均通过。六项基础集成及 EP×pattern FSDP
集成通过，普通 EP×FSDP 重跑通过；本阶段验证结果如上，提交与推送状态以 Git 历史为准。


补充 FSDP prefetch 重配修复：每次更新都写入 forward/backward 目标列表，
包括空列表，避免窗口设为零或执行顺序改变后残留旧目标。空 backward 列表
恢复 PyTorch 默认预取，不能视为禁用全部预取。窗口反转、边界清空与零窗口
回归通过；四进程 CPU/Gloo EP×pattern FSDP 重跑通过。最新全量 CPU 测试
1311 passed、2 skipped。GPU/NPU 内核与完整 CP backward 尚未验证。


### 2026-10-13：PP 恢复、验证及 CP 梯度的真实运行补证

本轮复查 trainer/runtime/validation.py、HF attention adapter 和 CP attention
接口，继续使用 torch 2.14 与 CPU/Gloo；以下脚本均退出 0：

| 脚本 | ranks | 实际验证范围 |
| --- | --- | --- |
| `pp_validation_equivalence.py` | 2 | 两步训练各触发一次 eval；末阶段 loss 有限且为正，验证后恢复训练模式 |
| `pp_checkpoint_equivalence.py full/resume WORKDIR` | 2 | 独立进程保存/恢复 step-2 DCP，恢复 Adam moments 与数据游标，续训两步 loss 逐位一致 |
| `pp_tp_trainer.py` | 4 | PP=2、TP=2 真实 trainer step；各 rank loss 一致，close 后 caller 进程组仍可归约 |
| `cp_grad_equivalence.py` | 2 | 无 attention 模型的纯 CP 输入分片、FSDP 参数梯度归约与四步 SGD 更新 |

PP checkpoint 原轨迹为 4.878459930419922、4.867841720581055、
4.858004093170166、4.86264705657959；恢复后的后两步完全一致，最大差异 0。
CP 最终参数最大误差 5.960e-08（容差 1e-4），禁用归约的参考副本梯度误差
分别为 6.229e-02、6.905e-02，确认测试能识别遗漏归约。

命令沿用上述显式 loopback torchrun 格式，端口依次 29731～29735；
checkpoint 的 full/resume 使用同一临时目录、不同 torchrun 进程。日志在
`/tmp/llmtuner-review-pp-validation.log`、`pp-checkpoint-full/resume.log`、
`pp-tp-trainer.log`、`cp-grad.log`（后三项也有相同 llmtuner-review 前缀）。
更新 PP validation 脚本中“等待多 rank 验证”的过时说明。

直接调用 CPU flex_attention backward 仍明确抛出 NotImplementedError。
因此本轮 CP 梯度结果只证明无 attention 的分片与归约，不能替代完整模型 CP
attention backward；该项、GPU/NPU 内核及可选后端仍待验证。本轮没有发现需
修改生产代码的新增缺陷，也未改变 trainer/trainer.py 或 components/metrics.py
的文件边界。此前 149/149 源码阅读覆盖保持不变。


### 2026-10-13：并行梯度 norm 契约复查

复查 accelerator/collectives.py 及 TP 复制参数梯度归约调用方。真实 CPU/Gloo
运行 tp_grad_norm_equivalence、ep_grad_norm_equivalence、
tp_replicated_grad_equivalence（各两 ranks）和 ep_fsdp_without_dense_dp
（四 ranks）均通过。最后一项仅检查 TP=4、EP=2、dense DP=1 时专家使用
size=2 sparse FSDP mesh，不能视为完整该组合训练等价。

发现并修复两项 clip_grad_norm_ 问题：

- PyTorch 接受文档中的 norm_type="inf"，本仓库分布式分支直接调用
  math.isinf 时抛 TypeError；入口统一转为 float，与 PyTorch 一致。
- error_if_nonfinite 仅检查局部 norm，漏掉归约溢出，并可能导致一个 rank
  提前抛错、其他 rank 等待 collective。改为完成 TP/EP/PP 及 DTensor
  norm 归约后检查全局值，异常时不修改梯度。旧实现的局部有限 norm 加
  归约溢出模拟复现为返回 tensor(inf)，没有按要求抛错。

扩展 TP 两进程集成：字符串 infinity norm、两个有限 2e38 的 SUM 溢出、
仅 rank 0 有 inf 三项均通过；后两项两个 rank 都抛出非有限值错误，梯度
未修改，后续 barrier 正常。新增两项单进程 inf/NaN 回归。
最新全量 CPU 测试 1313 passed、2 skipped。GPU/NPU 内核和完整 CP
attention backward 仍未验证。

修复后 EP norm 两进程与 PP×TP trainer 四进程重跑通过。
pre-commit --all-files、compileall 与 diff 检查通过；新增改动尚未提交。


### 2026-10-13：优化器状态初始化的 tensor LR 契约

复查 optimizer/state_dict.py、checkpoint base 与 torch_checkpointing backend
的恢复调用关系。复现 init_optim_state 临时用 torch.tensor(0.0) 替换 tensor
学习率：float64 CPU LR 被改为 float32，原 device 同样无法保留。改用
zeros_like(lr)，保留原 dtype/device/shape；原 LR 对象在 finally 中恢复，
梯度身份、参数值和首次 Adam step 计数保持不变。

新增 Adam/AdamW × 成功/step-pre-hook 异常四项回归，修复前四项均因 dtype
不一致失败，修复后通过。optimizer state 与 checkpointer 定向测试 40 passed；
最新全量 CPU 测试 1317 passed、2 skipped。pre-commit --all-files、compileall
及 diff 检查通过。上游当前路径为 torchtitan/components/optim/utils.py，
其 init_optim_state 仍使用 CPU 零 tensor；本次为本仓库有意修正，未修改上游。

CUDA capturable/fused 初始化尚未实机验证；当前 CUDA 不可用，且 importlib
确认 torch_checkpointing 未安装，因此不能将接口复查与模拟测试视为该后端
真实 round-trip。源码阅读覆盖仍为 149/149，新增改动尚未提交。


### 2026-10-13：复制 attention 的 TP 序列边界与组合数值验证

复查 TP 入口、trainer 梯度归约、chunked loss 及 MoE auxiliary loss 的调用契约。
shared_expert_tp_equivalence 首次运行因 ModuleList.index 不存在失败，改为
enumerate 保留层索引；随后发现脚本绕过 preprocess_inputs，未切分 TP token
输入。修正脚本后暴露生产问题：DeepSeek-V3 HF TP plan 只分片 MLP，MLA
attention 保持复制，局部长度 32 与完整 RoPE positions 长度 64 不匹配。

新增 ReplicatedAttentionSequenceBoundary：没有任何 TP 投影目标的 self_attn
先 gather 输入序列，执行原复制 attention，再切回本 rank token 输出。输出
切片不做 SUM；输入 gather backward 归约部分梯度，复制参数梯度仍由 trainer
求和。保留 module 路径、参数布局、tuple 辅助输出及原类行为。已有投影分片
attention 继续使用原路径。此修复适配 HF DeepSeek base_model_tp_plan 对复制
attention 的声明，不把其复制参数擅自改成 head shard。

两 CPU/Gloo ranks 的 DeepSeek/shared-expert TP 集成已扩展真实 backward：
logits 最大误差 0，全部复制 attention 参数及 shared gate/up/down shard 梯度
最大误差 2.043e-07。HF DeepseekV3RMSNorm 源码显式转 float32，因此梯度
容差用 1e-6，前向保留 1e-9；此前仅前向、未执行的注释已更新。

其他真实 CPU/Gloo 结果：

| 组合 | ranks | 结果及范围 |
| --- | --- | --- |
| PP=2 × EP=2 | 4 | 四步 trainer loss 与独立单进程参考通过 1e-4 容差 |
| MoE auxiliary loss × CP | 2 | router 梯度最大误差 1.192e-07，验证 forward SUM/identity backward |
| chunked loss × FSDP | 2 | 四步 loss 最大误差 4.768e-07，最终参数最大误差 1.490e-08 |
| chunked loss × TP | 2 | 四步 loss 最大误差 4.768e-07；此脚本不比较最终参数 |

命令使用前述显式 loopback torchrun，端口 29751、29771～29774；TP chunked
变体设置 CHUNKED_LOSS_EQ_TP=2。日志为 /tmp/llmtuner-review-pp_ep_equivalence.log
及 shared_expert_tp_equivalence、moe_aux_loss_grad_equivalence、
chunked_loss_equivalence[-tp] 的 llmtuner-review-*-passed.log。修复前失败日志
另行保留在 /tmp，不纳入仓库。

全量 CPU 测试 1317 passed、2 skipped；pre-commit --all-files、compileall 与
Git diff 检查通过。GPU/NPU 及复制 MLA attention × CP 等其他组合尚未验证；
源码阅读覆盖仍为 149/149。新增变更尚未提交或推送。


### 复制 MLA attention × TP × Ulysses CP 的校验补证

CP 原 head 校验无条件要求 global_heads % (tp*cp)==0，即使前一步 TP 已明确
保留 attention 复制。现在检查实际 attention 边界：所有层均为复制 attention
时只要求 heads % cp==0，其他布局保留 tp*cp 检查。新增 CPU 回归验证两
heads、TP=2、CP=2 的复制 attention 能附加 kernel；原 head 分片拒绝测试
继续通过。配置约束未放宽到未经标记的普通 attention。

新增 replicated_attention_cp_equivalence.py，四 CPU/Gloo ranks 实际运行
DeepSeek MLA + dense/shared/routed MLP，TP=2、Ulysses CP=2、两 attention
heads；通过真实 preprocess_inputs 检查每 rank 64/256 tokens，局部 logits
与完整 SDPA 参考对应切片最大差异 1.356e-08（容差 1e-7）。HF DeepSeek
RMSNorm 内部 float32 和 SDPA/flex 运算顺序差异使 float64 模型仍有 fp32
噪声。仅完整模型前向，不能作为 CP backward 验证。

测试装配初次采用全 dense DeepSeek，触发现有“MoE plan 无实际 MoE block”
拒绝；改为两层含实际 MoE。默认 grouped expert kernel 不支持 float64，测试
显式选择 eager，与此前 shared expert 测试一致。最终使用显式 loopback
四 ranks、端口 29785；日志 /tmp/llmtuner-review-replicated-attention-cp-passed.log。

最新全量 CPU 测试 1318 passed、2 skipped；pre-commit all-files 及新增未跟踪
集成脚本的单独 hooks、compileall、diff 检查通过。GPU/NPU 与完整 CP
backward 仍待验证；新增改动尚未提交。


### TP 复制 attention 的 head 整除条件

继 CP 校验修复后，复查 apply_tp 发现入口仍无条件要求 heads % tp==0，
错误拒绝只分片 MLP、保留复制 attention 的模型。现在根据解析后的 TP plan
是否实际匹配 self_attn 的 Linear 投影决定是否需要此条件；无 attention
结构信息时保留原检查，head 分片路径的检查顺序和拒绝测试保持有效。

shared_expert_tp_equivalence 改用三个 query/KV heads、TP=2，真实两进程
CPU/Gloo 前向与 backward 通过：logits 最大误差 0，复制 attention/shared
expert 梯度最大误差 3.262e-07（梯度容差 1e-6，原因仍为 HF RMSNorm FP32）。
这样该测试也能捕获对复制 attention 错误施加 head 整除条件的回归。
日志 /tmp/llmtuner-review-replicated-odd-heads.log，端口 29791。
pre-commit --all-files、compileall 与 diff 检查通过；新增改动尚未提交。

本次修改后全量 CPU 测试：1318 passed、2 skipped。


### Dense TP 目标的变换前校验

复查 apply_tp 的失败路径，发现 bias 和 weight 维度整除检查在逐个替换
模块时执行：后面的目标非法可能留下已经变换的前面模块。现在在 mesh
通信和 MoE/dense 参数变换之前校验全部 dense 目标，错误包含模块路径。
新增“第二个 Linear 有 bias”和“第二个 colwise 输出维度不整除”两项回归，
断言失败后所有 module/Parameter 对象身份不变。此保证覆盖可提前判定的
上述 dense 配置错误，不宣称自定义 realizer 任意运行时异常都可事务回滚。
定向 TP 测试 28 passed。新增改动尚未提交。

本次最终验证：1320 passed、2 skipped；pre-commit、compileall、diff 检查通过。


### CPU flex dropout 与 packed Ulysses 回归

复查 HF flex_attention_forward 明确拒绝非零 dropout；本仓库 CPU eager CP
fallback 原先静默忽略该 kwarg。现在同样拒绝非零 attention dropout，提示
设为零或 eval。新增拒绝回归，普通 dropout=0 路径保留，定向测试 8 passed。

四进程 CPU/Gloo cp_tp_equivalence 实际通过 KV-AllGather/Ulysses 两策略：
TP/CP 拼接 logits 和 reduced loss 误差均为 0，原 head 分片校验继续有效。
packed Ulysses 脚本首次在辅助 SDPA gradient 检查失败：mask 被扩展为全局
四 heads，all-to-all 后只有两个本地 heads。去掉不必要 expand，广播原
(1,1,S,S) mask，重跑通过。完整模型 packed forward logits 误差 0、loss
误差 2.274e-13；忽略文档边界的参考差异 0.6232，确认非空验证。SDPA
替代内核的 collective 梯度误差 0；不声称完整 flex backward 已验证。

端口 29801/29803；日志 /tmp/llmtuner-review-cp-tp-current.log、
/tmp/llmtuner-review-cp-varlen-fixed.log。源码覆盖仍为149/149，GPU/NPU
和完整 CP flex backward 仍待验证，新增改动未提交。

本次全量 CPU 测试1321 passed、2 skipped；pre-commit、compileall、diff 检查通过。


### Activation checkpoint × FSDP 补充数值验证

继续复查 memory/activation_checkpoint.py 的 full/selective policy、recompute
context 与 aux metric 抑制契约，核对已有 CPU MoE top-k/梯度回归。实际运行
两 CPU/Gloo ranks 的 ac_equivalence.py：full/selective 分别在每 rank 不同
数据上训练四步，与单进程合并批次、未启用 AC 的参考比较首步梯度、各步 loss
和最终参数。两个模式最终参数最大差异均为0，所有检查通过（容差1e-4）。
日志 /tmp/llmtuner-review-ac-current.log，显式 loopback 端口29811。

该脚本使用无 attention 的 decoder，证明的是 AC 重计算与 FSDP 复制梯度
归约组合；不能作为完整 HF attention/EP/设备内核数值验证。修正模块说明中
“仓库开发环境 torch2.2.2”的过时描述，改为具体能力缺失时的版本兼容语义。
本轮没有发现需修改生产行为的新增缺陷；前次全量测试1321 passed、2 skipped
仍为最新全量结果。源码逐文件覆盖149/149，GPU/NPU、完整 CP backward
及可选后端真实 round-trip 仍未完成；新增改动未提交。


### Loss reporting 与词表并行 loss 的真实通信补证

复查 trainer/runtime/reductions.py 的 token/loss mesh 选择和 loss.py 的
compute_logprobs/vocab-parallel 分支。真实两 CPU/Gloo ranks 运行
loss_report_rank_consistency：rank1 全部 labels=IGNORE_INDEX、rank0 正常，
两 rank 完成 train_step，所有 metrics 一致，loss/max_loss=4.862001895904541，
grad_norm=1.2443631887435913。此结果验证本地零有效 token 不会跳过 collective。

vocab_parallel_loss_equivalence 两 ranks 通过：V=8、7、5 的均匀/非均匀
词表分片、IGNORE_INDEX、reduction=none、带 group 的 full-vocab fallback
及 value/gradient 对比。未把此分支测试作为 trainer 自动分片 lm_head 的证明。
显式 loopback 端口29821/29822，日志 /tmp/llmtuner-review-
loss_report_rank_consistency-current.log 和 vocab_parallel_loss_equivalence-current.log。

本轮没有发现新增生产缺陷。最新全量 CPU 结果仍为1321 passed、2 skipped，
未因只增加审查记录重复执行全量测试。GPU/NPU、完整 CP flex backward 与
可选后端 round-trip 仍待验证；源码覆盖149/149，新增记录未提交。


### 集成 runner 的 world size 解析

复查 tests/integration_tests/run_all.py，发现仅匹配紧跟 torchrun 的
--nproc_per_node=N；带 --standalone 的 pp_tp_trainer 和
 ep_fsdp_without_dense_dp 被默认按两 ranks 启动，和实际要求四 ranks 冲突。
修正解析以允许中间选项，并兼容下划线/连字符及等号/空格参数形式。新增三项
回归3 passed，直接核对四个四-rank 脚本的解析均返回4。

此改动修正验证工具，不改变训练代码。runner 仍不能自动编排 PP checkpoint
的 full/resume 两阶段或所有脚本变体；不能以一次 run_all 证明全组合覆盖，
之前 checkpoint 已按两阶段单独实跑。新增记录与工具改动未提交。


### 集成 runner 自动编排 PP checkpoint 两阶段

修正 run_all 无参数启动 pp_checkpoint_equivalence 必定失败的问题：使用
共享 TemporaryDirectory 分别启动 full/resume torchrun；仅 full 成功后
运行 resume，两阶段任一失败返回失败，退出清理临时目录。成功/失败两项
runner 回归验证共用目录、full 失败不启动 resume 和清理行为；runner 定向
测试5 passed。

真实执行 GLOO_SOCKET_IFNAME=lo0 MASTER_ADDR=127.0.0.1 python
 tests/integration_tests/run_all.py pp_checkpoint_equivalence：两阶段完成，
checkpoint 数值比较通过，runner 报告1 passed、0 failed，耗时22.1s。
日志 /tmp/llmtuner-review-runner-checkpoint.log。其他可选脚本变体仍需单独
选择，未将 runner 默认调用视为所有并行矩阵覆盖。新增改动未提交。

本轮全量1324 passed、2 skipped；随后新增两项runner流程回归单独通过。
pre-commit all-files及新增文件hooks、compileall和diff检查通过。


### PP 多 stage 调度变体补证

复查 pipeline schedule 构建与模型 stage ownership，实际两 CPU/Gloo ranks
运行 pp_equivalence.py Interleaved1F1B 和 ZBVZeroBubble，各四步、四个
microbatches；均对独立单进程训练参考检查 loss 和全局 grad norm，并验证
stage 参数覆盖、ownership、首步梯度非空及累计 tokens。Interleaved1F1B
loss 最大差异0，ZBVZeroBubble loss最大差异4.768e-07，grad norm均通过
1e-5容差。未检查所有最终参数逐项数值，不能把该结果写成参数等价证明。

显式loopback端口29831/29832；日志 /tmp/llmtuner-review-pp-Interleaved1F1B.log
和 /tmp/llmtuner-review-pp-ZBVZeroBubble.log。此次没有发现新增生产缺陷。
最新全量仍为1324 passed、2 skipped，随后两项runner流程回归另已通过。
硬件与可选后端验证限制保持，新增记录未提交。


### PP packed corpus 集成的未完成项

实际两 CPU/Gloo ranks 运行 pp_real_corpus_equivalence.py，初始化被现有
UnsupportedCombinationError 拒绝：flat packed stream 的4 microbatches
会切断文档上下文。保留该生产保护，没有把测试失败记作通过。复查脚本发现
参考循环也假定行batch切分、使用每microbatch分母、报告未归一化loss，与
trainer 的完整step分母不一致。修正脚本顶部错误支持声明，保留明确待修状态。

要验证原范围需要 loader 先独立打包每个microbatch，并对齐positions/masks
和完整step分母；当前脚本不能作为真实packed PP训练的数值证明。日志
/tmp/llmtuner-review-pp-real-corpus.log，端口29841，退出1。此为新增具体未完成
验证项，源码149/149阅读与既有CPU测试通过不能证明它完成。未修改生产代码。


### Packed 数据调用链与 TextSequence 维度契约

复查 loader_builder、GrainDataLoader、TextCollator 和 pipeline runtime。
当前 packing context 使用完整 per-rank batch token预算，collator输出一维
stream，Grain按collator.num_rows_per_batch()==1合并，PP再切分。要实现
独立packed microbatches，必须将packing预算改为每microbatch并在loader
组织独立样本，不能仅去掉PP拒绝或在切片处重置positions；该支持尚未实现。

发现TextSequence只比较len，二维字段可能长度相等而通过，后续collator
向一维buffer拼接失败；零维字段则直接len TypeError。新增四个字段的ndim=1
校验，明确拒绝非token向量。八项回归覆盖input_ids/labels/positions/
padding_mask的零维和二维输入。全量CPU1334 passed、2 skipped；
pre-commit、compileall、diff检查通过。新增改动未提交，packed PP多
microbatch及既有硬件/可选后端验证项仍未完成。

## Packing 直接构造参数校验（2026-10-13）

- `datasets/packing/iterators.py`：直接构造 document splitter 时，零 context
  length 使 chunk end 等于起点，非空样本无法结束分块循环；负值同样不能保证前进。
- 复用 `contracts.require_positive_integer`，在 splitter、document packing dataset
  和 iterator 构造时校验 context length、每行 token 数及 document 容量。
  正常 builder 路径已有校验，本次覆盖直接调用路径。
- 回归新增 14 个零值/负值构造用例；不通过实际运行无限循环复现。
- 验证：数据管线单测 99 passed；完整 CPU 单测 1348 passed、2 skipped；
  pre-commit、compileall 和 diff whitespace 检查通过。

## Document packing 游标恢复验证（2026-10-13）

- 阅读安装的 Grain `FlatMapIterDataset` 和 `_FlatMapDatasetIterator`：当前
  first-fit 入口不使用 `max_fan_out` 限制 iterator 输出，不能据声明上限推断丢块。
- 新增 4 个 document packing 恢复用例，在消费 0/1/2/3 行后恢复：覆盖空样本、
  跨行文档余量、末行 padding；逐字段比较后续行，并核对真实 token 恰为 1..10。
- 同时从耗尽后的状态恢复，确认不会重放末行。该验证不涉及 PP microbatch
  的独立 packing，也不证明多进程 loader 的全部恢复组合。

## Packed PP 独立 microbatch 装配（2026-10-13）

- `datasets/loader_builder.py` 按 PP microbatch token 预算构造 text packing
  graph；`PipelineTextCollator` 将独立 packed rows 组成 `(M, T)` schedule
  batch。沿用现有 PP 按整行切分路径，不再切割整步平坦 packed stream。
- 有限尾批补 fully masked rows，保留固定 schedule 形状、不重放数据；合批
  输出沿用 pinned-memory 分配策略。文档容量约束作用于每个 packed row。
- 对外部平坦 packed batch 的拒绝检查保留，错误信息改为推荐配置好的 text loader。
- loader 回归覆盖 concat-then-split、first-fit 的独立 packing 等价及游标恢复，
  另验证有限尾批的 labels、padding、有效 token 计数和输入不被修改。
- 修复 `pp_real_corpus_equivalence.py`：全步有效 token 分母、microbatch
  行切分、PP stage 实际初始权重、探测后 loader 恢复及最终参数比较。
- 两个 CPU/Gloo ranks、4 microbatches、3 AdamW steps：loss 轨迹分别为
  4.877315、4.479658、4.298073，与单进程参考一致；最终参数最大误差
  7.451e-09。CPU harness 将生产 BlockMask 的 mask_mod 展开为稠密 SDPA
  mask；该结果证明 PP 装配/调度/缩放和优化更新，不证明 CUDA flex kernel。
- 原先“real-corpus harness 在 packed split 处失败”的历史记录由本节取代；
  GPU/NPU packed PP、PP×CP/TP packed 组合仍需进一步运行验证。
- 最终完整 CPU 单测：1355 passed、2 skipped；compileall 和 diff 检查通过。
  pre-commit 对除用户 `examples/native_alltoall.py` 外的全部 tracked files
  通过；all-files 曾被该 example 的既有 E501 注释阻止，未修改其内容。

## Packed PP 有限验证及单文档容量（2026-10-13）

- 扩展 real-corpus 两 rank PP 检查：训练后执行 `steps=-1` 的有限验证，
  与单进程逐有效 token 加权参考比较；明确断言语料触发全 padding
  microbatch，且验证后各 stage 恢复 train mode。
- CPU/Gloo 实际运行通过：finite validation loss 4.186483；训练 loss
  轨迹一致，最终参数最大误差仍为 7.451e-09。CPU attention 使用明确
  的 dense-mask SDPA 测试替代，未验证生产 CUDA flex。
- loader 等价/恢复用例补充 `max_num_documents=1` × 两种 text packing，
  确认容量作用于每个 microbatch 的真实文档，而非将 padding 计为文档。
- real-corpus 测试语料改用 TemporaryDirectory，在进程正常退出及 Python
  异常展开时清理；数据管线单测 108 passed。
- 完整 CPU 回归：1357 passed、2 skipped；compileall、diff 检查及
  排除用户 example 的全部 tracked-file pre-commit 通过。

## Packed PP DCP 恢复验证（2026-10-13）

- `pp_checkpoint_equivalence.py` 新增 `--packed`，full/resume 阶段使用同一
  checkpointable JSONL 图；CPU 明确使用 real-corpus harness 的稠密文档 mask
  SDPA 替代，未验证 CUDA flex。默认 synthetic 变体保留。
- 两阶段各两个 CPU/Gloo ranks 实际运行：step 2 保存 DCP，然后恢复继续
  step 3/4。不中断 loss 为 4.894712925、4.546628475、4.389602661、
  4.289792061；恢复 tail 与最后两步逐位一致，最大差异 0。
- 增加各 stage 最终参数逐位比较、累计 token 计数比较；恢复后的 Adam
  exp_avg 非空且非零。参数和计数均一致，证明 packed loader 游标与训练
  状态共同恢复；并未证明更换 packing 配置后的恢复语义。
- 修复测试资源管理：使用 `trainer.close()` 关闭 loader 等资源；resume
  不再删除调用者工作目录，由外层 TemporaryDirectory 管理清理。

## Loader batch layout 恢复校验（2026-10-13）

- Grain loader 检查点增加 `batch_layout`：每步 token 数、context 长度、
  每步 packed row 数、文档容量。恢复时在修改 iterator 前比较，拒绝
  配置变化后将旧 batch cursor 应用于新布局。
- 旧 version-1 检查点缺少该字段时保持兼容；无法从缺失元数据证明旧
  布局相同。该校验尚不涵盖数据内容、tokenizer、shuffle 或 packing recipe
  的变化，不宣称完整数据图 fingerprint 校验。
- 两种 packing × 文档容量 None/1 的恢复回归均加入不兼容 row 数断言；
  数据管线测试 108 passed。
- 带 layout 元数据的两 rank packed PP DCP full/resume 实际运行通过，
  loss tail、最终参数和 token 计数一致；完整 CPU 1357 passed、2 skipped。

## DCP layout 元数据验证及旧格式缺口（2026-10-13）

- 新增真实单进程 DCP 保存/加载回归：保存 token batch 32，恢复至 16，
  loader 拒绝 saved layout，iterator 状态保持不变且仍可正常取 batch；通过。
- 直接 `load_state_dict` 的旧格式兼容不能推广到 DCP：实测严格默认 planner
  对旧检查点缺少 `loader.batch_layout.*` 报 Missing key，在 loader 方法
  被调用前失败。此前旧格式兼容说明仅针对直接调用，该 DCP 迁移缺口
  尚待定向兼容修复，不能用全局 allow_partial_load 掩盖模型缺失权重。

## 旧 DCP layout 定向迁移（2026-10-13）

- 标准 DCP 加载增加 `CheckpointLoadPlanner`。仅当检查点含 loader
  version 且整组 batch_layout 字段缺失时，从请求的加载计划移除新增
  layout 字段并警告旧布局无法验证。其他字段仍由默认严格 planner 检查。
- 真实 DCP 回归：旧 loader cursor 和模型权重成功加载；缺失模型权重
  仍报错；保存端仅有部分 layout 字段也报错，避免掩盖不完整元数据。
- 修复前一节记录的旧 DCP 缺键迁移缺口；旧检查点缺少布局证据，仍不能
  宣称其布局与当前配置相同。checkpointer 单测 35 passed。

## DCP planner 多 rank 迁移回归（2026-10-13）

- 正常新格式 packed PP full/resume 在两 CPU/Gloo ranks 通过；新增 planner
  不改变恢复后的 loss、最终参数或 token 计数，loss 最大差异 0。
- checkpoint harness 增加 `LLMTUNER_TEST_LEGACY_LAYOUT=1` 保存端模拟：
  仅在保存时省略 loader layout 字段，恢复使用未修改的正常 manager。
  实际两阶段两 rank 运行通过并触发旧布局警告，恢复 tail 为
  4.389602661、4.289792061，逐位等于不中断训练；参数和 token 计数一致。
- 此回归验证缺失整组 layout 字段的迁移路径，无法校验真正旧运行的
  布局是否被用户改变；CPU attention 仍为明确标注的 SDPA 替代。

## CLI 隐式 args 文件审查（2026-10-13）

- HfArgumentParser 默认会探测 `sys.argv[0]` 同名 `.args`，与当前 CLI
  文档的 flags-only 输入契约冲突：旁置文件能悄悄改变未指定的训练参数。
- `parse_config` 显式设置 `look_for_args_file=False`。回归创建含 steps
  和 global_batch_size 的真实 `.args` 文件，确认仅显式 steps 生效，
  batch size 保持配置默认。不引入新的配置文件输入或优先级规则。
- 验证：新回归通过；完整 CPU 1361 passed、2 skipped；排除用户
  example 的 tracked-file pre-commit、compileall、diff 检查通过。

## 本地 HF 配置自动填充（2026-10-13）

- `auto_fill_model` 原仅按单个斜杠识别 Hub ID，绝对目录及无斜杠
  相对目录不填充；factory 实际按路径存在性读取本地 config，造成聚合
  config 的 vocab/layer/width 与实际模型分歧。
- 自动填充同样先识别现存路径；本地目录及 Hub ID 读取已保存的 HF
  架构，离线 architecture name 保持显式尺寸。同步修正文档与 CLI help。
- 两个真实本地 config.json 回归覆盖绝对路径及无斜杠相对目录，
  六个架构字段均匹配 saved config，验证无网络。
- 验证：完整 CPU 1363 passed、2 skipped；排除用户 example 的
  tracked-file pre-commit、compileall 和 diff 检查通过。

## 复合 HF text_config 自动填充（2026-10-13）

- Qwen2.5-VL 等复合配置的 decoder 尺寸在 `text_config`；model factory
  已 unwrap，但配置自动填充读取外层并退回默认，导致 run 架构字段不一致。
- 自动填充跟随 factory 读取 `text_config`，不引入 config 到模型层的
  import 依赖。真实本地 Qwen2.5-VL config.json 回归核对六个尺寸字段；
  不构建 vision encoder，不宣称多模态训练数值验证。
- 完整 CPU 回归 1364 passed、2 skipped；排除用户 example 的
  tracked-file pre-commit、compileall、diff 检查通过。

## 整数校验跨配置回归补齐（2026-10-13）

- 增加模型 hidden_size、训练 steps、数据 num_packing_bins、profiler
  memory_snapshot_max_entries 的非法类型回归，覆盖 bool/float/NaN/string/None。
- 更新共享校验函数说明，使其明确表达整数与非 bool 契约。
- 配置专项 92 passed；目标 pre-commit 与 diff 检查通过。本轮未修改运行行为。

## 复合 HF decoder context 扩展（2026-10-13）

- factory 加载预训练 config 后原直接访问外层 max_position_embeddings；
  Qwen2.5-VL 外层没有此字段，实际构造配置会 AttributeError。
- 读取 text_config（dense config 则自身）并扩展 decoder context。
  两个真实本地复合配置回归覆盖请求长度短于/长于 saved context，
  确认不缩短且 vision config 不改变；不涉及 vision 模型训练数值。
- 验证：factory config 6 passed；完整 CPU 1366 passed、2 skipped；
  排除用户 example 的 tracked-file pre-commit、compileall、diff 检查通过。

## 程序化 Trainer 架构同步（2026-10-13）

- 程序化 Trainer 不经过 CLI auto-fill；离线 arch_overrides 可覆盖
  vocab_size，但 synthetic loader 仍按 cfg 默认值生成 token，导致
  embedding 越界；saved HF 架构也有相同分歧风险。
- 在 `_build_model` 解析 HF config 后同步六个 decoder 尺寸到 cfg.model，
  随后的 parallel/loader 装配使用实际架构；不额外请求网络配置。
- 真实 Trainer 回归使用 vocab override 17，确认 cfg 与生成 token 范围
  一致，并成功完成一步 forward/backward/optimizer 更新。
- 验证：完整 CPU 1367 passed、2 skipped；排除用户 example 的
  tracked-file pre-commit、compileall、diff 检查通过。

## 装配配置复用与清单核对（2026-10-13）

- 实际源码 149 个 Python 文件与逐文件表 149 条一致，无遗漏或过期路径；
  此为阅读覆盖，不代表全部硬件组合验证。
- metrics FLOPs 改用 `_build_model` 已解析的 HF config，避免再加载本地
  文件或 Hub 配置；修正原“从参数测量”的错误注释，实际为架构公式。
- 新增真实 Trainer 装配回归，确认底层 HF config builder 只调用一次。
  既有 trainer 测试 71 passed；新增计数回归单独通过。

## FLOPs 配置复用完整回归（2026-10-13）

- 配置复用改动后的完整 CPU suite：1368 passed、2 skipped。
- 阅读 FLOPs 公式及 geometry helpers，确认仅读取配置；实际对 Llama 和
  Qwen3-MoE 比较调用前后完整 config.to_dict，均未改变。该证据覆盖
  指标计算的配置副作用，不证明 FLOPs 公式对所有模型家族均准确。

## Synthetic loader 游标类型（2026-10-13）

- synthetic loader 原接受 bool 游标；steps=True 等同 1，恢复会跳过
  首批数据。显式要求非负 int 并排除 bool，在 replay 之前报错。
- 五个回归覆盖 True/False/负值/浮点/字符串，确认状态未改变且下一批
  与新 loader 相同。正常整数及既有重复恢复语义保持。

## Synthetic 数据配置恢复校验（2026-10-13）

- synthetic 检查点新增 batch_layout，保存 seed/vocab_size/batch_size/seq_len；
  恢复时在 replay 前拒绝配置变化，避免同一个 cursor 恢复到另一条数据流。
- 旧 DCP planner 识别 historical synthetic steps 字段并定向兼容缺失
  整组 layout；其他键继续严格检查。旧格式无法证明数据配置相同。
- 四个回归分别改变四个配置字段，拒绝后状态未改变；数据管线 118 passed。

## Synthetic layout 两 rank 恢复（2026-10-13）

- 使用保存端 legacy-layout 模拟，默认 synthetic PP DCP 两阶段各两
  CPU/Gloo ranks 实际运行通过。不中断 loss 为 4.878459930、
  4.867841721、4.858004093、4.862647057；恢复最后两步逐位一致，
  最终 stage 参数和累计 token 计数一致。
- 该验证无 attention kernel 替代，覆盖 CPU synthetic 原生路径；
  真正旧格式缺少 seed/尺寸证据，仍不能校验用户是否改变旧数据配置。
- 新格式经 integration runner 的 full/resume 同样通过（20.8s），
  loss、参数和 token 计数一致；无额外 layout 缺失兼容。

## Scheduler 恢复 base learning rate（2026-10-13）

- 真实 AdamW 回归先恢复 optimizer，再恢复 scheduler，保存 lr=1.0、
  新建 lr=0.1。修复前 scheduler 使用构造时 base_lrs，恢复 lr 0.0625
  而参考 0.625，违反其“base lrs 由 optimizer 恢复”的契约。
- 恢复时从 optimizer param_groups.initial_lr 重建 base_lrs，再计算
  当前 lr；回归同时比较恢复时及下一步的学习率。scheduler 22 passed。

## Scheduler 修复完整回归（2026-10-13）

- 完整 CPU suite 1378 passed、2 skipped；scheduler 专项 22 passed。
- 两 rank PP DCP full/resume 经 runner 通过，恢复 loss tail 与不中断
  训练逐位一致，参数及 token 计数一致。此案例配置 lr 未变化；新建
  lr 与保存 lr 不同的场景由真实 optimizer/scheduler 专项回归验证。
- 排除用户 example 的 tracked-file pre-commit、compileall、diff
  检查通过；不将该 CPU 证据扩展为 GPU/NPU 恢复验证。

## DCP optimizer/scheduler 加载顺序（2026-10-13）

- 真实 Trainer DCP 回归发现 Stateful 回调顺序并不保证 optimizer 先于
  scheduler；保存 lr=0.01、新建 lr=0.001 时仍恢复错误 lr 0.0005，
  而正确值为 0.005。此前组件单测未覆盖实际 DCP 顺序。
- 标准 DCP 加载将 scheduler 状态作为普通 dict 读取，待 DCP 完成
  optimizer 恢复后再调用 scheduler.load_state_dict；HF model-only 不受影响。
- 真实 Trainer 保存/恢复回归修复前失败、修复后通过，验证 scheduler
  metrics 和 base_lrs 均使用保存值。

## DCP 不同新建学习率的两 rank 恢复（2026-10-13）

- checkpoint harness 增加 resume-only `LLMTUNER_TEST_RESUME_LR`，保持
  保存端配置不变。实际两 CPU/Gloo ranks 保存 lr=0.0003，恢复构造
  lr=0.003（10 倍），正常 DCP 加载后继续两步。
- 恢复 loss tail 与不中断训练逐位一致，最终参数及 token 计数一致；
  证明 optimizer 完成后应用 scheduler 的顺序在多 rank 实际路径生效。
- pre-commit（排除用户 example）与 diff 检查通过；CPU synthetic
  原生 attention，无 kernel 替代，GPU/NPU 尚未验证。

## Scheduler epoch 输入校验（2026-10-13）

- scheduler 恢复游标要求非负 int，排除 bool；LambdaLR 在本容器
  构造时已执行初始 step，所以合法保存状态从 epoch 0 开始。
- 五个回归覆盖负值、布尔、浮点和字符串，验证失败前不修改 epoch
  或当前学习率；scheduler 专项 27 passed。
- 完整 CPU 1384 passed、2 skipped；排除用户 example 的 tracked-file
  pre-commit、compileall、diff 检查通过。

## EMA 更新失败前身份校验（2026-10-13）

- EMA update 原逐 model part 校验后立即 lerp，后一个 part 参数被
  替换时先前 averages 已改变，失败不能保持原 EMA 状态。
- 更新先解析全部 parameter/buffer averages，身份校验完成后再 lerp；
  合并仅单调用方的 `_update_group`。多 part 回归替换后一个 part
  参数，确认报错前先前 averages 均保持不变；EMA 19 passed。

## EMA 修复完整回归（2026-10-13）

- 核对新增回归直接保存所有 EMA optimizer state 中的 live averages，
  第一 part live 参数加 1 后替换第二 part 参数，失败后逐个比较平均值；
  无依赖 flat state_dict 是否返回 Tensor 的空断言风险。
- 完整 CPU 1385 passed、2 skipped；排除用户 example 的 tracked-file
  pre-commit、compileall、diff 检查通过。身份预校验保证本次路径不部分
  更新，不宣称任意设备 kernel 异常下具有事务回滚。

## EMA 两 rank DCP 恢复验证（2026-10-13）

- 确认 trainer 在 optimizer update 后执行 EMA；实际 complex64 buffer
  更新 [1+2j] 至 live [3+4j]、decay 0.5 得 [2+3j]，声明 dtype 可用。
- checkpoint harness 新增 LLMTUNER_TEST_EMA=1（固定 decay 0.5），保存
  最终 EMA state 并逐字段比较恢复后的结果。两 CPU/Gloo ranks
  full/resume 经 runner 通过：EMA、最终模型参数、token 计数及 loss
  tail 逐位一致。该结果不涵盖 dynamic decay、buffer EMA 或 GPU/NPU。
- tracked-file pre-commit（排除用户 example）通过。

## Dynamic EMA 多 rank 恢复（2026-10-13）

- checkpoint harness 增加 LLMTUNER_TEST_EMA=dynamic，half_life_fraction
  0.5；两 CPU/Gloo ranks 的 full/resume 实际通过，EMA、模型参数、
  token 计数和 loss tail 逐位一致。覆盖从 trainer step 延续 EMA aging。
- 初始 runner rendezvous 地址被解析为非本机地址，TCPStore 持续重试，
  未进入训练；终止该 handle 后用显式 127.0.0.1 和独立端口通过。
  该启动失败不是 EMA 数值失败；runner 的本机 rendezvous 默认待改进。

## 集成 runner 本机 rendezvous 修复（2026-10-13）

- 默认 hostname 与 c10d 动态 rendezvous 均在本机网络环境中被解析为
  非本机地址，TCPStore 持续重试；失败发生在训练启动前。
- runner 使用单节点 static rendezvous，显式 master address 127.0.0.1；
  `--master-port` 默认 29500，可为并发任务指定独立端口，拒绝非法端口。
- 实际执行 `LLMTUNER_TEST_EMA=dynamic PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0
  python tests/integration_tests/run_all.py --master-port 29897
  pp_checkpoint_equivalence`：两 CPU/Gloo ranks full/resume 通过，22.7 秒，
  loss tail 逐位一致，最终模型、EMA、token 计数一致。
- 本次修复验证单机 runner，不涵盖多节点或 GPU/NPU 训练。

## Buffer EMA 实际 DCP 恢复契约（2026-10-13）

- 新增真实 DistributedCheckpointManager 保存/恢复回归，覆盖 float32 与
  complex64 buffer，各自验证正常恢复和 exclude_from_loading=["ema"]。
- 保存时刻的 buffer EMA 与 live buffer 明确不同，避免 round-trip 空验证；
  新模型从零 buffer 启动，恢复后精确比较所有 parameter/buffer EMA 键和值。
- 排除 EMA 时验证 averages 等于刚加载的模型状态，而非随机初始状态；
  随后改变 buffer 并执行下一 firing，核对 lerp 数值。四个场景均通过。
- 本项是 CPU 单进程真实 DCP 验证，未验证多 rank buffer EMA 或 GPU/NPU；
  本次未发现需修改 production 实现的问题。

- 本轮完整 CPU 回归：1392 passed、2 skipped；tracked-file pre-commit
  （排除未修改的 examples/native_alltoall.py）、compileall 与 diff 检查通过。

## 双 CP 轴 token 顺序与 mask 修复（2026-10-13）

- KV-outer/Ulysses-inner mesh 按全 CP rank 连续分片输入。原实现先 KV gather
  再 Ulysses exchange，形成不同的 KV token 排列，与 causal/document mask 不符。
  改为先 Ulysses exchange，再 KV gather；保留原反向 collective 链。
- 双轴 attention 查询长度为 S/kv，原模型按总 CP degree 分片 mask，
  查询长度不匹配。set_cp_mesh 新增 mask_mesh，双轴指定 KV mesh；
  普通和 packed mask 均仅按 KV 轴 Q-shard，输入仍按完整 CP mesh 分片。
- 新增 cp_dual_axis_equivalence.py，四 CPU/Gloo ranks、KV=2/Ulysses=2、
  GQA heads=4/KV heads=2；真实模型 causal/packed eager-flex 前向最大差异
  均为 0。生产 dual collectives 配合 dense SDPA 验证 Q/K/V 梯度与完整
  attention 参考一致且非零；额外旧通信顺序 oracle 要求数值明显不同。
- CPU flex backward 不可用，梯度验证仅替换 attention 内核；尚未证明
  CUDA/NPU flex backward 或双 CP×TP/PP 组合。CPU 单测1392 passed、2 skipped。

## 单轴 CP 回归与 KCP 输入约束（2026-10-13）

- 双轴修复后实际 runner 执行 cp_wiring_equivalence 与
  cp_ulysses_varlen_equivalence：两 CPU/Gloo ranks 分别10.4秒、5.7秒通过，
  覆盖 KV gather 前向/反向、headtail mask、packed Ulysses 输入与通信梯度。
- 对照 FSDPTurbo/fsdp_turbo/distributed/context_parallel/kcp/kda_state.py
  发现 llmtuner KCP 缺少卷积 kernel_size > 1 约束；现于 optional runtime
  加载前拒绝不满足条件的 KDA 输入，并复用已验证 kernel_size 构建 context。
- 新回归以未安装 KCP runtime 环境验证错误路径先于内核导入；CP 专项
  15 passed，修改文件 pre-commit 与 diff 检查通过。未证明 Ascend KCP
  内核数值正确性，fla/triton_ascend_kernels 在当前环境不可运行。

- 进一步对照 FSDPTurbo/fsdp_turbo/modules/kcp.py，补齐 head_k_dim<=256
  和 safe-gate lower_bound∈[-5,0) 约束；在 optional runtime 加载前拒绝
  超范围 head dimension、负界越界、零和 NaN。CP 专项19 passed。

## KCP 装配错误顺序（2026-10-13）

- apply_kcp 在确认模型存在 KDA 模块前加载 optional runtime，使普通模型
  在缺依赖环境中报 ImportError，掩盖实际不支持的模型结构。
- 先完成候选模块识别，无候选时报告 no Kimi Delta Attention module；
  通过后加载 runtime、取得一次 process group，再统一装配 forward。
- 回归验证不支持的模型在内核加载前失败且原 forward 不变；CP20 passed。

- KCP 本轮修改后的完整 CPU 回归1398 passed、2 skipped；tracked-file
  pre-commit（含新增双轴脚本，排除未修改的用户 example）、compileall 与
  git diff HEAD --check 通过。

## EP 空接收 expert 的梯度断链修复（2026-10-13）

- 新增两 rank CPU/Gloo EP 回归，token 数分别5/7，覆盖普通和两切片
  Domino；同时构造全部 token 只路由到 rank0 的零接收场景。
- 修复前正常收发场景通过，但零接收 rank 的 local_inputs.grad 为 None：
  GroupedExperts CPU fallback 没有任何 expert segment 时返回无图空张量，
  与 dispatch 输入和权重断开，导致通信 backward 不能完整执行。
- 空输入 fallback 使用 empty F.linear，保持输入和权重图连接，产生正确
  零权重梯度；正常 expert segment 的计算保持原实现。新增 CPU 回归
  验证空输出仍可 backward，输入梯度存在，所有 expert 权重梯度为零。
- 专项43 passed；两 rank 脚本逐个比较输出、输入/路由权重梯度以及
  本地 expert 权重梯度与全局完整参考。脚本使用float64、CPU fallback，
  尚未验证 CUDA grouped_mm 空接收场景或 accelerator stream overlap。

- 普通 RoutedExperts 与 DominoRoutedExperts 各自的正常/零接收四场景
  两 CPU/Gloo ranks 实际全部通过（4.1秒）；完整CPU1399 passed、2 skipped。
  本次修改文件 pre-commit、compileall 和 diff 检查通过。

## Grouped GEMM 空输入与混合精度（2026-10-13）

- 实际 bf16、use_grouped_mm=True 的空输入仍进入 fused op，报 Invalid
  strides/sizes；原空输入处理仅位于 CPU loop 前，未覆盖 fused 分支。
- 将 empty F.linear 放在 fused/loop 分支之前，空输入完全跳过 grouped
  GEMM，同时转换所选权重至输入 dtype，保持 fp32 权重/bf16 输入的
  零梯度连接。非空输入仍走既定 grouped/loop 路径。
- 四种 dtype×use_grouped_mm 回归，以 fp32 权重验证 float32/bf16 空输入
  输出dtype、输入梯度存在、全部权重梯度为零，并禁止调用 fused op。
  专项46 passed；此证据支持空输入分支，不证明非空 CUDA grouped_mm。

- 本轮完整CPU1402 passed、2 skipped；两 rank普通/Domino EP四场景
  复跑通过（3.6秒）；修改文件pre-commit、compileall、diff检查通过。

## EP×activation checkpoint FQN 修复（2026-10-13）

- 使用真实 torch checkpoint_wrapper 复现：model.state_dict 去掉
  _checkpoint_wrapped_module 段，但 expert_shard_map/named_params 保留该段，
  使模型及 canonical optimizer/EMA expert state 不匹配，跳过 EP gather。
- EP shard map 和参数映射统一复用 canonical_fqn，恢复与模型/optimizer
  state key 的一致性，避免同FQN各rank异质expert切片被当作复制状态。
- 新增真实 wrapper、模拟两rankcollective回归，覆盖模型键与optimizer
  exp_avg键；验证完整expert tensor被gather、rank1恢复正确切片。
  EP checkpoint/model state专项13 passed；尚未验证真实多rank DCP×AC。

## EP×AC 真实多 rank DCP 与 degree 恢复（2026-10-13）

- 新增 ep_checkpoint_ac_equivalence.py，两 CPU/Gloo ranks 使用真实
  checkpoint_wrapper、ModelCheckpointState、OptimizersContainer 和 EMA。
  各rank专家初值和Adam梯度不同，保存前更新optimizer/EMA，确保切片异质。
- 实际 DCP 保存后，依次恢复到 wrapped EP=2、unwrapped EP=2、
  unwrapped EP=1；模型、Adam状态（含metadata）、EMA完整键值精确一致。
  显式要求保存模型expert axis=4，防止未gather局部slice的空验证。
- runner 实际通过（4.5秒），脚本pre-commit与diff检查通过。此项证明
  wrapper键清理、真实EP collective和DCP恢复契约，不涵盖AC重计算
  backward、FSDP DTensor专家或GPU/NPU。

## EP×activation checkpoint 重计算反向（2026-10-13）

- ep_domino_equivalence 扩展真实checkpoint_wrapper；普通AllToAll与
  Domino×正常/零接收×checkpoint开关共八个两rank CPU/Gloo场景。
- checkpoint启用时计数forward调用，要求实际调用两次，避免只测试
  wrapper构造而未触发重计算；输出、输入/路由权重梯度、本地expert
  参数梯度均与完整expert参考一致，实际runner通过（4.0秒）。
- 本轮未发现新的production问题；脚本pre-commit和diff检查通过。
  结果不涵盖CUDA grouped_mm、编译或FSDP组合的重计算反向。

## EP combine 保留 float64 精度（2026-10-13）

- LocalTokenDispatcher.combine 无条件将expert输出转float32，在float64
  模型中损失精度；此前本地/分布式参考共用该combine，无法检出此问题。
- 改用promote_types(output.dtype,float32)，保持低精度输出的float32
  乘法，同时保留float64；返回仍遵守原expert输出dtype。
- 新测试采用float64精细值，直接与数学乘法比较输出、输入及score梯度，
  要求零误差；dispatcher15 passed，普通/Domino×零接收×AC八场景
  两CPU/Gloo ranks复跑通过（3.8秒）。修改文件质量检查通过。

## Router FP32 明确契约核查（2026-10-13）

- RouterGateLinear 明确规定FP32投影及反向GEMM；routing的softmax/sigmoid
  使用该FP32输出。该行为是router契约，区别于expert combine无意降精度。
- 补充float64输入/权重的实际投影回归，核对输出FP32、前向数学结果、
  以FP32计算后恢复至float64的输入和权重梯度。未修改生产契约。
- 模型构件/router专项70 passed，修改文件pre-commit通过。

## Quantile routing full AC 统计与更新（2026-10-13）

- 新增实际wrap_full重计算回归，使用quantile MoE、混合padding输入，
  比较开启/关闭AC的输出、输入及所有参数梯度。
- 确認初始histogram只含有效token，并要求backward后观测数增长，
  证明真实重放路由；均匀重复观测不改变本场景quantile bias更新。
- 两侧实际执行update_quantile_expert_bias后bias一致，重计算侧
  histogram/token counter清空；专项回归通过，pre-commit通过。
  不涵盖selective/region AC或不同微批重计算倍数的组合。

## Quantile selective AC 与梯度累积（2026-10-13）

- 原full AC回归参数化为full/selective，进一步覆盖1/3个不同微批。
  实际wrap_selective采用仓库生产save policy，不替换算子或重计算机制。
- 四场景比较输出、各微批输入梯度、累积参数梯度、有效token直方图
  计数和最终quantile bias；要求真实重放统计且更新后scratch清空。
- quantile routing专项28 passed；本轮未发现production缺陷，测试
  pre-commit通过。不涵盖region AC、跨rank或各微批不同AC策略。

## Quantile 全空观测窗口不漂移（2026-10-13）

- 实际零histogram连续100次estimate造成现有非零bias漂移2.38e-7：
  原实现虽然回填当前bin位置，但重新插值和mean-center仍产生roundoff。
- 全专家观测数为零时直接返回现有bias副本，无新观测不改变平衡状态。
- 回归要求100次空更新后torch.equal严格保持；quantile专项29 passed，
  修改文件pre-commit通过。部分专家空观测的原估计路径保持现有语义。

## Quantile 大计数插值精度（2026-10-13）

- 构造int64直方图复现float32 target/counts-before舍入问题：分位目标
  8388609.5落入单观测bin时丢失半个观测，bias为±0.25而非±0.125。
- 计数相关插值使用float64，最终结果恢复expert_bias dtype；新增独立
  数学参考回归要求精确±0.125。该场景含不同expert总计数，检查公开
  estimate接口的有效输入，不宣称常规observe会生成不同总计数。
- quantile专项30 passed；修改文件pre-commit通过。Ascend设备上的
  float64算子支持和成本尚未验证。

## Quantile 大计数采用整数分位秩（2026-10-13）

- 上轮float64插值方案替换为int64计数分位秩与残差：ceil(count*K/E)
  用整数除法，先计算count*K-before*E，再转float32形成bin内比例。
- 保留大计数半token目标的精度，避免训练设备float64支持要求；已有
  ±0.125精确参考、空窗口不漂移、full/selective AC累积回归均通过。
- quantile专项30 passed，pre-commit通过；Ascend实际运行仍待硬件验证。

## Quantile histogram 累积溢出（2026-10-13）

- 真实observe复现int32 bin达到2147483647后再观察一个token回绕
  为-2147483648；之后int64 sum无法修复，跨rank归约亦存在同类风险。
- non-persistent histogram改int64，使本地scatter_add与hook归约保持
  64位计数；内存由每bin4字节增至8字节，checkpoint格式不受影响。
- 边界回归实际调用observe，验证两个expert均为2147483648且总数
  4294967296；quantile专项31 passed，修改文件pre-commit通过。
  Ascend int64 scatter/collective实际内核仍待验证。

## Quantile DP×CP 实际归约（2026-10-13）

- 新增quantile_bias_equivalence.py，四CPU/Gloo ranks采用2×2独立
  DP/CP mesh，实际调用生产update_quantile_expert_bias完成两轴归约。
- rank各自填充不同bin，归约后每expert总观测数超过int32上限；
  以all_gather后int64求和构造独立完整histogram，核对最终bias精确
  一致，并验证各rank histogram与token counter清空。
- 实际runner通过（4.5秒），脚本pre-commit通过；不涵盖Ascend或
  真实router生成海量token的负载测试。

## Quantile TP 条件归约验证（2026-10-13）

- 四rank脚本扩展DP×TP、EP关闭/开启场景；EP关闭只参考同TP坐标
  的DP rank，EP开启参考全部DP×TP rank。输入刻意按rank设置不同bin。
- 除最终bias外，拦截估计器输入并精确比较完整histogram，防止均匀
  重复计数因quantile缩放不变而漏检；各场景scratch均清空。
- DP×CP与两种DP×TP共三场景实际通过（4.5秒），pre-commit通过。
  本项检查hook的mesh条件，未装配真实TP/EP模型或验证NPU通信。

## Profiler 初始化中断清理（2026-10-13）

- Profiler.__enter__在memory构建阶段只捕获Exception；KeyboardInterrupt
  不调用外层__exit__，使已启动trace留存。初始化失败清理改捕获
  BaseException，trace清理异常同样不覆盖原始初始化错误。
- 实际回归参数化ValueError/KeyboardInterrupt，验证trace恰好退出
  一次、收到原错误类型、handle清空且原始错误保留。
- profiler专项通过，修改文件pre-commit通过。

## Trainer/CLI 清理中断处理（2026-10-13）

- Trainer.close某资源close抛KeyboardInterrupt时原路径跳过后续资源。
  改记录BaseException，尝试释放全部资源后重新抛首个错误，维持幂等。
- train和CLI的清理异常处理同样覆盖BaseException，已有主异常时
  不用清理中断覆盖它；成功训练后发生清理中断仍向调用方抛出。
- 回归增加KeyboardInterrupt资源失败、主训练异常加清理中断，验证
  metrics/dataloader/GC均被释放、二次close无重复、主错误保留。

## Trainer 初始化失败主异常保留（2026-10-13）

- 构造器外层捕获BaseException，但partial-resource close只捕获
  Exception；清理KeyboardInterrupt仍会替换原assembly错误。
- 构造器清理异常处理统一BaseException；测试覆盖assembly与cleanup
  各自普通异常/KeyboardInterrupt四组合，原始assembly错误保留，
  checkpointer失败后metrics仍被释放。
- construction/cleanup/close专项通过，修改文件pre-commit通过。

## Allocator history 生命周期（2026-10-13）

- MemoryProfiler 启动 allocator history 后，Profiler 退出此前仅清空对象引用，
  同一进程的后续工作仍可能继续记录。新增幂等 close，调用设备 allocator
  的 _record_memory_history(enabled=None)；CPU 无 allocator 时跳过。
- 正常退出、普通异常和 OOM 均关闭记录；OOM snapshot 在关闭之前写出。
  snapshot 或 trace 失败后仍尝试关闭，清理错误不覆盖已有训练异常。
- CPU mock 回归验证关闭参数、顺序、幂等及关闭失败传播；没有 GPU/NPU，
  未验证设备实际停止记录。上游 torchtitan 当前也未显式关闭该记录。

## Validation 跨 rank 归约验证（2026-10-13）

- 从 Trainer.validate 追踪 mode restoration、temporary loader、loss/token
  mesh 和 finalize_validation，未发现本轮可复现的新生产缺陷。
- 新增 validation_reduction_equivalence.py，四 CPU/Gloo ranks 分别构造
  DP×CP、DP×TP 的 2×2 真实 mesh，运行生产验证循环和 collective。
  DP rank/step 有效 token 数不同，首步有一半 sequence shard 全部 masked。
- 用完整 DP batches 的独立 summed cross-entropy / valid token 参考核对
  所有 rank 的 loss，并检查 loader 关闭、混合 module modes 恢复、两次
  no-grad forward 和训练 ntokens_seen 不变。实际 runner 通过（5.6 秒）。
- 模型为 deterministic projection 和显式 sequence slices；topology 为
  mesh-backed test double，未覆盖实际 attention、TP/CP/FSDP 模型变换。
  validation CPU 专项 25 passed，新脚本 pre-commit 通过。

## Profiler schedule 配置提前校验（2026-10-13）

- ProfilerConfig 接受 profiler_repeat=-1/profiler_skip_first=-1，PyTorch
  schedule 在启动时抛 AssertionError；此前该输入错误会推迟到训练装配后。
- 配置层复用 require_integer_at_least 拒绝两字段负值，None、0 和正值仍有效，
  profiling 开启/关闭均校验，避免后续启用时暴露潜伏配置错误。
- 回归覆盖两个字段及 enable_profiling 开关；未改变 skip_first_wait，
  PyTorch 将它按布尔开关解释，并没有相同的非负计数约束。

## 异步 checkpoint 完成日志（2026-10-13）

- DCP async Future 的 done callback 不区分成功/异常/取消，总是打印
  Finished writing；保存失败虽在 wait 时传播，日志此前仍误报写入成功。
- 回调先检查 cancelled 和 exception，仅成功记录完成耗时；失败记录
  step 与实际异常，取消独立报告。没有在 callback 中消吞 wait 的错误。
- 回归注入真实 concurrent.futures.Future，覆盖成功、OSError(disk full)、
  cancelled，验证日志分类以及 wait 继续抛原始异常/CancelledError。
- checkpointer CPU 专项 36 passed，未改变保存格式、staging 或 collective。

## 异步 DCP 专用通信组释放（2026-10-13）

- DistributedCheckpointManager 为 async 保存创建专用 Gloo group，但此前
  close 未销毁该组。CLI 的全局 teardown 会清理它，programmatic 重复
  创建 manager 则会残留通信资源。
- close 在 staging/save futures 完成后清理 purge thread、stager 与自建
  group；某一清理失败仍尝试其他项，保留首个异常，清空已消费 handle。
  默认组已经全局销毁时跳过专用组销毁，避免重复退出错误。
- 实际单 rank Gloo 测试注入 stager.close OSError，验证专用组从注册表
  删除、默认组仍可 barrier、首个异常保留和二次 close 无重复。
- checkpointer CPU 专项 37 passed；多 rank/HCCL 环境未验证本项销毁。

## 实际双 rank async DCP 生命周期（2026-10-13）

- 新增 async_checkpoint_lifecycle_equivalence.py，两 CPU/Gloo ranks
  实际运行 async_save、等待、group teardown、DCP restore，重复两轮。
- 使用生产 OptimizersContainer 与 LRSchedulersContainer，恢复时故意
  改变初始 learning_rate，验证恢复后的下一步参数和完整 optimizer
  state 与连续训练逐元素一致（rtol=atol=0），scheduler state 相同。
- 每个 manager 自建组退出后从注册表消失；重复 close 无异常；默认组
  barrier/all_reduce 仍可用，save_future 清空。实际 runner 通过（3.7秒）。
- 初稿使用原生新 AdamW，DCP 不会恢复其未初始化 state slots；生产
  容器已有 init_optim_state，因此测试改用真实训练容器。不能据本项
  宣称裸 optimizer 具有相同恢复契约。
- 新脚本 pre-commit 通过；本项仅覆盖 replicated CPU parameters 与
  async thread 保存，未覆盖 pinned-memory/process staging 或 GPU/NPU。

## Scheduler restore 避免 get_lr 警告（2026-10-13）

- LambdaLR.get_lr 在 step 外调用会发 UserWarning，load_state_dict 原本
  用它恢复 _last_lr；warnings-as-errors 配置下会使正常 checkpoint
  restore 失败。改按恢复的 base_lrs 和对应 lr_lambdas(last_epoch) 重算，
  保留多 param-group 各自的 base LR、optimizer LR 和 epoch/count 同步。
- 已有恢复回归提升该警告为 error，验证 optimizer restored initial_lr
  与下一 scheduler step；scheduler 专项 27 passed。
- 两 rank async DCP 继续训练测试以相同警告为 error 实际重跑通过；
  模型/optimizer 与连续训练仍逐元素一致。

## DCP 构造失败释放自建通信组（2026-10-13）

- async manager 创建 Gloo group 后，retention thread.start 失败会发生在
  _initialized=True 之前；public close 跳过部分构造对象，造成自建组残留。
- 在 thread.start 失败处主动清理已获取资源；清理失败不会替换构造错误。
- 真实单 rank Gloo 回归注入线程启动 RuntimeError，验证原始异常保留、
  已创建专用组被释放、默认组仍可 barrier；checkpointer 专项 38 passed。

## 原生 DCP 最终模型导出与自动恢复（2026-10-13）

- 默认 final checkpoint 只保存模型，但同样含 .metadata；此前 automatic
  resume 只检查该文件，会选择较新的模型导出并因缺失训练状态而失败。
- DCP discovery 读取 metadata，非 step-0 必须包含 optimizer 和
  lr_scheduler keys 才参与 automatic resume；step-0 保留 model-only
  seed 语义。retention 的 valid 判断仍保留模型导出，不视为 abandoned。
- 真实 DCP 回归保存 full step-1 和 model-only step-2，验证自动选择
  step-1、恢复参数，以及 step-2 仍可通过 initial_load_model_only 导入。
- checkpointer/trainer 专项 121 passed；多 rank async 集成同步增加
  newer native model-only export 和 automatic resume 覆盖。
- 使用 DCP 自身 reader selection，兼容本地及 fsspec reader；本轮
  实际验证限 CPU/本地。可选 torch_checkpointing backend 未验证同项。

## Automatic resume 减少 metadata 读取（2026-10-13）

- 复查 requested state 加载：optimizer/scheduler 之外的 trainer、loader、
  EMA 仍由严格加载契约处理，没有新增静默补齐或绕过缺失状态的行为。
- discovery 原本探测全部 step 再取 max；metadata 内容检查使每个探测
  实际读取 metadata，旧 checkpoint 的 I/O 在找到最新完整项后仍发生。
- 改为按 step 递减探测，找到首个 resumable checkpoint 即返回；保留
  max_step、非规范目录过滤和无可恢复项返回 -1。
- 手工计数验证 step-5 model-only / step-3 full / step-1 full 时只探测
  5 和 3；max_step=2 只探测 1。checkpointer/trainer 专项 121 passed。

## 训练装配结构简化（2026-10-13）

- 删除 runtime/assembly.seed_everything 的无独立契约转发，Trainer 公共
  seed_everything 直接调用 utils/seeding 的定义；公开 Trainer API 保留。
- parallelize 的 shared apply_stages 通过 partial 和显式参数名绑定
  TP/EP/CP/AC/compile/FSDP，合并只有一处消费的 _apply_ac 包装。
  ep_group/tp_mesh 使用外层已解析值，不再在两个调用点重复传递。
- apply_stages 模型为唯一位置参数，mesh 和 stage_order 为 keyword-only，
  清晰区分变换对象、运行布局及顺序。保留 STAGES 和 PP_STAGE_ORDER 契约。
- 复用 assembly stages、PP distinct seed、trainer 回归：94 passed；
  未重命名已有准确入口，未拆分 trainer.py 或 metrics.py。

## Text processors 与 recipes 职责分离（2026-10-13）

- processors.py 保留 TextProcessor/ChatProcessor 的 tokenization、label
  对齐和 supervision；recipe factories 与 DATASETS 注册表移入 recipes.py，
  数据源/processor/filter 装配不再混入样本处理模块。
- 工厂统一 build_local_jsonl、build_local_jsonl_sft、
  build_local_jsonl_sft_multiturn，旧 make_* 名称移除；loader builder
  与测试使用新定义模块，config 提示和 upstream map 同步更新。
- recipes 复用 processors，没有循环依赖；text package 保持无 eager
  imports。文件阅读清单新增 recipes.py，原 processors 行同步职责。
- 本项为职责搬运和命名调整，复用现有数据/renderer/训练测试验证。

## Multimodal recipe 注册职责分离（2026-10-13）

- multimodal/processors.py 保留媒体/文本样本转换和长度过滤；
  MULTIMODAL_DATASETS 移入 multimodal/recipes.py，与文本 recipe 边界一致。
- loader 的 multimodal 分支仍按需导入 recipes，避免文本路径导入
  torchvision；配置提示、注册表测试和文件清单同步到新定义位置。
- 未保留旧 processors 注册表别名，调用方从 recipes 导入。

## Processor 工厂契约校正（2026-10-13）

- `SingleDataset.processor` 原声明只接受类，但多模态配方实际使用
  `partial(MultimodalProcessor, sample_processor=...)`。新增
  `SampleProcessorFactory` 协议，明确通过关键字 `context` 创建处理器；
  不改变现有运行时行为，也不引入兼容别名。
- 处理器类型与工厂契约统一放在 `datasets/contracts.py`，配方组合模块
  仅消费契约。纠正文档对 partial 的错误限制。
- 当前累计重组与契约修改验证：CPU 全量 1432 passed、2 skipped；
  pre-commit（排除未修改的用户示例 `examples/native_alltoall.py`）、
  compileall 和 diff 空白检查通过。源码清单为 151 个 Python 文件。
  本次未新增多设备数值验证。

## Loader 配方装配顺序统一（2026-10-13）

- 文本与多模态解析入口均返回 `(recipe, build_packing, collator)`，
  packer 通过 partial 绑定自身选项，删除额外的 packing_kwargs 中转字典。
- 修复多模态 loader 未转发 `num_packing_bins`、始终使用默认 8 的问题；
  配置帮助明确该选项同时适用于多模态 FirstFit。现有端到端媒体回归
  改用 3 bins，并检查该值实际到达 packer。
- 懒加载测试清单加入新增 recipes 与 packing 模块，文本模块探针
  改为真正的 text.recipes，避免重组后检查旧模块。
- 验证：数据专项 221 passed；CPU 全量 1432 passed、2 skipped；
  tracked-file pre-commit（排除未修改的用户示例 native_alltoall.py）、
  compileall、diff 空白检查通过。本项未修改分布式通信行为。

## 随机数工具职责归并（2026-10-13）

- `derive_distinct_seed` 从 trainer/runtime/assembly.py 移至现有
  utils/seeding.py；训练装配调用工具模块，旧装配模块不再导出该函数。
- 坐标偏移算法与参数顺序不变；修正文档中 row-major 的错误描述，
  明确首个坐标维度变化最快。PP seed 测试和上游映射同步引用定义模块。
- 移除 seed 单测对 DCP 的无关环境门控；专项 32 passed，完整 CPU
  1432 passed、2 skipped；tracked-file pre-commit（排除用户未修改示例）、
  compileall、diff 空白检查通过。未新增硬件数值验证。

## 训练装配依赖与顺序说明（2026-10-13）

- assembly 从 checkpointer、optimizer、parallel 的定义模块直接导入服务，
  移除通过包导出解析内部实现的绕行；外部公开导出保持不变。
- 删除已失效的步骤编号与重复并行顺序说明，明确模型构建时进行的组合
  检查和 PP 先切分、再应用本地阶段的职责。
- checkpoint 说明纠正 base_lrs 的归属：scheduler 保存其基准学习率
  和进度，不能仅靠 optimizer 状态恢复 schedule。
- 本项不改变训练流程、公共函数参数或模型变换顺序。
- 验证：trainer/PP seed 专项 87 passed；完整 CPU 1432 passed、
  2 skipped；tracked-file pre-commit（排除未修改用户示例）、compileall、
  diff 检查通过。151 个源码文件与审查清单逐路径一致，无遗漏或过期项；
  清单一致仅证明记录覆盖，不代表全部硬件组合已验证。

## 本地 SFT 配方去除动态子类（2026-10-13）

- 两种 SFT 工厂复用 ChatProcessor，通过 partial 绑定字段读取函数与
  renderer，移除仅负责传参的局部子类和嵌套 messages 闭包。
- 字段读取函数放在 recipe 模块，错误提示及样本转换语义不变；工厂
  参数使用 SampleProcessorFactory 契约。
- 新增普通 pickle 配方往返后 token/label 等价回归，覆盖单轮 SFT
  在进程 worker 间传递时的动态类不可序列化问题。多轮 renderer
  自身是否可序列化仍由对应后端决定。
- 验证：完整 CPU 1433 passed、2 skipped；tracked-file pre-commit
  （排除未修改用户示例）、compileall、diff 检查通过。新增回归
  使用显式 CHAT_TEMPLATE，避免默认模板与测试样本不匹配。

## 配方类型集中声明（2026-10-13）

- composition.py 定义 DatasetRecipe，组合节点与文本/媒体 packing
  入口复用同一个配方联合类型，避免六处重复维护节点列表。
- DatasetRecipe 表示未构建配方，GrainDataset 表示构建结果；公共数据
  包显式导出该类型，loader 内部从 packing.build 定义模块导入入口。
- 检查 get_type_hints 对递归组合节点与 packing 入口均能正常解析；
  不改变函数参数顺序或运行时构建行为。
- 验证：完整 CPU 1433 passed、2 skipped；tracked-file pre-commit
  （排除未修改用户示例）、compileall、diff 检查通过。

## FFN 维度计算契约补全（2026-10-13）

- compute_ffn_hidden_dim 对非正 dim/multiple_of、非有限或非正 multiplier
  及截断为零的 hidden dimension 给出明确 ValueError，避免除零或无效
  模型几何；基础维度使用整数 8 * dim // 3，避免不必要的浮点转换。
- 新增九项无效参数回归；正常尺寸、倍率和对齐逻辑继续使用现有测试。
- 复查 activation.py、linear.py 的调用与 router dtype 约束，本次未
  发现需重命名或搬运的具体问题。
- 验证：模型公共构件专项 55 passed；完整 CPU 1442 passed、
  2 skipped；tracked-file pre-commit（排除未修改用户示例）、compileall、
  diff 检查通过。未运行 CUDA BF16 router 路径或其他硬件数值验证。

## TP 模型投影与 HF 分片包装命名区分（2026-10-13）

- models/common/tp_projections.py 的 RowParallelLinear 改为
  ReduceScatterLinear，明确其融合 reduce-scatter/GEMM 的职责，避免
  与 tensor_parallel.py 中接收已分片权重的同名 HF 包装器混淆。
- 删除旧名，更新模块公开导出与模型构件测试。HF 分片包装器名称
  仍为 RowParallelLinear，二者参数、权重布局与执行行为不变。
- 验证：TP 模型构件/HF 分片专项 46 passed；完整 CPU 1442 passed、
  2 skipped；tracked-file pre-commit（排除未修改用户示例）、compileall、
  diff 检查通过。外部使用旧模型构件名的代码需迁移到 ReduceScatterLinear。

## QKV head 接口精简与几何校验（2026-10-13）

- local_head_split 删除未使用且无消费者的 dp_shard_dim 参数，仅保留
  张量与 head_dim；原有调用顺序不变。
- QKVLinear 构造前校验 head_dim/n_heads/n_kv_heads 正值，避免零 KV
  head 在取模时除零；head split 同样拒绝非正维度。
- 输出拆分复用 num_tokens，删除重复变量；不改变 KV group 布局、
  contiguous 输出或 checkpoint split/merge。新增八项非法输入回归。
- 验证：QKV/TP 投影专项 45 passed；完整 CPU 1450 passed、
  2 skipped；tracked-file pre-commit（排除未修改用户示例）、compileall、
  diff 检查通过。删除的参数属于公共签名变更，外部调用需移除该关键字；
  本仓库没有传入该参数的消费者。

## TP 投影初始化复用标准实现（2026-10-13）

- ReduceScatterLinear.reset_parameters 直接调用 nn.Linear 的初始化方法，
  删除复制的 Kaiming/bias 公式和私有 fan-in helper 依赖。
- 保留 nn.Module 继承关系，避免改变按 isinstance(nn.Linear) 选择模块
  的并行/重计算逻辑；参数布局、bias 默认值与 reset API 不变。
- 新增带/不带 bias 的固定 seed 初始化逐参数等价测试，并核对
  state_dict 键与标准 Linear 一致。
- 验证：TP 投影专项 20 passed；完整 CPU 1452 passed、2 skipped；
  tracked-file pre-commit（排除未修改用户示例）、compileall、diff 检查通过。

## 多模态文档数限制与数据契约说明（2026-10-13）

- 多模态 FirstFit packer 转发 context.max_num_documents 到
  max_sequences_per_bin，修复配置在媒体路径被忽略。端到端 loader
  回归检查自定义限制 2 实际到达 Grain packer。
- 精简 composition 和 datasets 包说明，明确 map filter 保留索引槽，
  DP 索引切片互斥不等于每 rank 有效样本数相同；修正 loader
  实际接收 LLMTunerConfig、文本 recipes 已被导入的描述。
- 验证：数据专项 222 passed；完整 CPU 1452 passed、2 skipped；
  tracked-file pre-commit（排除未修改用户示例）、compileall、diff 检查通过。

## 多模态 packing 标签输出契约（2026-10-13）

- packing_output_to_multimodal_sample 与文本转换统一：由 segment id
  识别 padding，在复制后的 labels 上写入 IGNORE_INDEX，确保转换输出
  自身满足 padding 不参与 loss 的约束。
- 复用媒体转换回归检查 padding 屏蔽、有效标签保留和输入数组不变。
  正常 Grain packer 已填充 IGNORE_INDEX 的输出数值不变。
- 验证：多模态数据专项 98 passed；完整 CPU 1452 passed、2 skipped；
  tracked-file pre-commit（排除未修改用户示例）、compileall、diff 检查通过。

## Hugging Face source 加载逻辑归并（2026-10-13）

- 两种 HF source 共用 load_dataset 调用，streaming 由具体 source 类型
  决定；随机访问返回类型校验与流式 cursor/DP 分片路径保留。
- 删除重复加载参数，简化 source 模块说明中的不准确表述。新增两项
  离线回归，检查所有加载选项转发和 DatasetDict 结果拒绝。
- 验证：完整 CPU 1454 passed、2 skipped；tracked-file pre-commit
  （排除未修改用户示例）、compileall、diff 检查通过。

## SFT 配置按实际处理分支校验（2026-10-13）

- 单轮 local_jsonl_sft 继续要求 prompt/response 字段非空；启用
  chat_renderer 的多轮路径只校验实际使用的 messages 字段，不再
  拒绝无消费者的空单轮字段。新增三项分支校验回归。
- dataset selector/path 的 CLI 帮助加入 local_jsonl_sft；shuffle
  帮助区分 map 全局 shuffle 与 stream rank 内 window shuffle。
- 验证：数据 pipeline 专项 124 passed；完整 CPU 1457 passed、
  2 skipped；tracked-file pre-commit（排除未修改用户示例）、compileall、
  diff 检查通过。

## CastLinear 固定 dtype 与 autocast（2026-10-13）

- 实际复现外层 CPU autocast 将 CastLinear(compute_dtype=float32) 输出
  改为 BF16；显式张量转换不足以保证固定 dtype matmul。
- forward 在输入设备的局部禁用 autocast，保留参数对象、冻结状态、
  tied weights 与 state_dict 布局。新增输出和 input/weight/bias
  梯度在有无 autocast 下完全等价回归。GPU/NPU 路径尚未实测。
- 验证：CastLinear 专项 13 passed；完整 CPU 1458 passed、2 skipped；
  tracked-file pre-commit（排除未修改用户示例）、compileall、diff 检查通过。

## Router FP32 GEMM 与 autocast 契约（2026-10-13）

- CPU 复现 RouterGateLinear 在外层 autocast 下返回 BF16，违反 FP32
  router score 契约。前向与反向 GEMM 局部禁用 autocast；保留 CUDA
  BF16 operands/FP32 output 优化和参数原 dtype 梯度转换。
- 新增输出与输入/权重梯度有无 autocast 完全等价回归，backward
  也在 autocast 上下文中执行以覆盖反向保护。CUDA/NPU 尚未实测。
- 验证：模型公共构件专项 56 passed；完整 CPU 1459 passed、
  2 skipped；tracked-file pre-commit（排除未修改用户示例）、compileall、
  diff 检查通过。

## Router 选择与权重提取职责统一（2026-10-13）

- debug 均衡选择函数改为 _select_balanced_experts，仅返回 expert IDs；
  正常、quantile 和 debug 选择之后统一执行无 bias score gather。
- 删除重复提取逻辑，保持归一化、缩放、aux loss 与 dispatch map
  路径不变；纠正 sorted=False 的排序说明。旧私有方法没有外部消费者。
- 验证：完整 CPU 1459 passed、2 skipped；tracked-file pre-commit
  （排除未修改用户示例）、compileall、diff 检查通过。复用现有
  debug/group/quantile/padding 回归，本项不改变路由数学或通信。

## MoE bias buffer 注册归并（2026-10-13）

- sign-based 与 quantile 方案共用 expert_bias_E 注册逻辑，保持互斥
  校验、persistent 状态和无 balancing 时的 None 契约。
- 拒绝非有限 load_balance_coeff，避免 NaN/Inf 通过正值校验后污染
  bias。新增 NaN/正负 Inf 三项构造回归。
- 验证：MoE bias/routing 专项 49 passed；完整 CPU 1462 passed、
  2 skipped；tracked-file pre-commit（排除未修改用户示例）、compileall、
  diff 检查通过。

## Bias 更新参数按消费者精简（2026-10-13）

- sign-based update_expert_bias 接收展平 MoE layer 列表，与 quantile
  更新接口一致，删除不使用的 model_part 参数载体及重复嵌套堆叠。
- 注册 hook 时收集一次 layers，计数写回使用 strict zip，删除手工
  row 索引。调用方和已有单/多 model-part 回归同步更新。
- 公共模块函数参数结构变更：外部直接调用者需传 list[MoE]。
- 验证：MoE bias 专项 18 passed；完整 CPU 1462 passed、2 skipped；
  tracked-file pre-commit（排除未修改用户示例）、compileall、diff 检查通过。

## MoE 统计归约规则归并（2026-10-13）

- sign-based counts 与 quantile histogram 共用 _reduce_token_statistics，
  将 DP/CP 和 EP 启用时 TP 的归约轴规则集中一处。
- 空 quantile layer 列表直接返回，与 sign-based 接口一致，避免
  torch.stack([]) 失败；新增不启动 collective 的空列表回归。
- 不改变归约轴顺序、group 或 tensor dtype；复用已有 hook/拓扑测试。
- 验证：MoE bias/routing 专项 50 passed；完整 CPU 1463 passed、
  2 skipped；tracked-file pre-commit（排除未修改用户示例）、compileall、
  diff 检查通过。未新增多设备数值验证。

## Optimizer FQN 映射与恢复清理（2026-10-13）

- FQN 序列化拒绝重复参数 ID 与重复名称，避免不同参数状态在 flat
  checkpoint 中静默覆盖；新增两项无歧义映射校验回归。
- 初始化状态时保存 (param_group, lr)，按直接对应关系恢复学习率，
  删除 pop(0) 顺序消费；梯度恢复使用 strict zip。
- 验证：optimizer state 专项 9 passed；完整 CPU 1465 passed、
  2 skipped；tracked-file pre-commit（排除未修改用户示例）、compileall、
  diff 检查通过。

## Optimizer 不完整状态恢复拒绝（2026-10-13）

- unflatten_optim_state_dict 对当前已初始化 optimizer 预期的状态字段，
  若 checkpoint 既无直接键也无嵌套值，则明确 KeyError，避免跳过
  Adam moment 后加载成不完整状态。
- 新增删除 exp_avg 的恢复回归，验证报错且 optimizer 原状态对象、
  moment 和 step 不变；其他 optimizer 不拥有的 flat keys 仍忽略。
- 验证：optimizer state 专项 10 passed；完整 CPU 1466 passed、
  2 skipped；tracked-file pre-commit（排除未修改用户示例）、compileall、
  diff 检查通过。

## Optimizer 容器跨实例 checkpoint 键冲突校验（2026-10-13）

- OptimizersContainer.state_dict 合并 inner optimizer 状态前检查 flat
  key 交集，拒绝跨 model part/optimizer 的同名 FQN 静默覆盖。
- 新增两个真实 AdamW 实例使用同名参数的回归；不依赖 named-param
  构造能力，确认容器层明确报错。合法互斥 FQN 保存路径不变。
- 验证：optimizer state 专项 11 passed；完整 CPU 1467 passed、
  2 skipped；tracked-file pre-commit（排除未修改用户示例）、compileall、
  diff 检查通过。

## 多 optimizer 恢复前完整预检（2026-10-13）

- 容器先初始化必要状态并重建所有 inner optimizer 的 checkpoint
  描述，再逐个 load_state_dict，避免后一个缺失字段时前一个已加载。
- 新增第二 optimizer 缺少 exp_avg 的回归，确认第一 optimizer 的
  lr 与状态对象未变化。该保证针对重建/元数据校验失败，不提供
  PyTorch load_state_dict 任意异常或 lazy state 初始化的事务回滚。
- 验证：optimizer state 专项 12 passed；完整 CPU 1468 passed、
  2 skipped；tracked-file pre-commit（排除未修改用户示例）、compileall、
  diff 检查通过。

## Scheduler 恢复预检与学习率归属说明（2026-10-13）

- load_state_dict 先计算全部 scheduler 的 base/last learning rates，再
  写入状态，避免后一个 group 缺少 initial_lr 时前一个已更新。
- 新增第二 optimizer 缺少 initial_lr 的失败回归，确认第一个
  scheduler 状态和学习率不变；纠正 base_lrs 来源的文档描述。
- 验证：scheduler 专项 28 passed；完整 CPU 1469 passed、2 skipped；
  tracked-file pre-commit（排除未修改用户示例）、compileall、diff 检查通过。

## EMA 跟踪分组校验归并（2026-10-13）

- 更新与冷启动恢复共用 _tracked_groups，统一参数/buffer 遍历和
  固定身份校验，删除重复逻辑及建议移除身份检查的过期 TODO。
- 两个写入路径均先校验全部组，保留替换模型参数时的失败语义。
- 验证：EMA 专项 19 passed；完整 CPU 1469 passed、2 skipped；
  tracked-file pre-commit（排除未修改用户示例）、compileall、diff 检查通过。

## Optimizer 恢复保留 FQN 元数据（2026-10-13）

- unflatten_optim_state_dict 将当前 param_names 带入重建组，保证
  普通 optimizer 加载后仍可再次按 FQN 保存。flat checkpoint 格式不变。
- 删除 EMA 单独快照/补回 param_names 的重复补丁，改为共享恢复契约。
- 新增加载后再次保存回归，复用 EMA roundtrip/冷启动测试。
- 验证：optimizer state/EMA 专项 32 passed；完整 CPU 1470 passed、
  2 skipped；pre-commit 自动格式化一处后复跑通过，compileall、
  diff 检查通过；全仓门禁排除未修改的用户示例。

## Trainer checkpoint 计数恢复校验（2026-10-13）

- step/ntokens_seen 先完整读取并校验为非负整数，再统一赋值，
  拒绝 bool、负值和浮点计数，避免部分恢复或错误训练终止条件。
- 新增六项非法计数回归，检查失败后两个旧计数均保留。
- 验证：trainer 专项 88 passed；完整 CPU 1476 passed、2 skipped；
  tracked-file pre-commit（排除未修改用户示例）、compileall、diff 检查通过。

## Synthetic 数据游标直接定位（2026-10-13）

- RandomTokenSource.iter_from(step) 按绝对 batch index 创建确定性流；
  loader 恢复重建对应迭代器，删除逐批重放历史数据。配置/layout
  校验、拒绝向后恢复与 DP 切片不变。
- 新增百万步游标回归，确认恢复期间不生成历史 batch，下一批与
  原 seed 公式完全一致，cursor 正确递增。
- 验证：数据 pipeline 专项 125 passed；完整 CPU 1477 passed、
  2 skipped；tracked-file pre-commit（排除未修改用户示例）、
  compileall、diff 检查通过。

## Synthetic 恢复更新活跃迭代器（2026-10-13）

- 审查发现直接定位恢复后，已经开始消费的 generator 内部 for 循环
  仍绑定旧 stream，导致读取的数据与恢复后的 checkpoint cursor 不一致。
- 每批从当前 self._iterator 取值，使恢复对训练循环持有的迭代器生效。
  保留原有游标校验、DP 切片与确定性 seed 公式。
- 新增 DP rank 1/2 回归：先消费，再恢复到第 10 批，使用同一个
  活跃迭代器连续读取两批，与新建恢复 loader 的数据和游标比较。
  测试在修复前失败，修复后通过。
- 验证：完整 CPU 1478 passed、2 skipped；双进程 CPU/Gloo
  async_checkpoint_lifecycle_equivalence 通过，两轮异步保存/恢复后
  optimizer、scheduler、模型继续训练状态完全一致，进程组正确释放。
- GPU/NPU staging、FSDP/TP/PP/CP/EP 硬件组合未由上述测试覆盖。
- 质量门禁：tracked-file pre-commit（排除未修改用户示例）、
  compileall、git diff HEAD --check 均通过。

## Model parts checkpoint 合并冲突校验（2026-10-13）

- ModelCheckpointState._get_state_dict 从覆盖式字典推导改为逐 part
  校验合并，拒绝同名 FQN，错误包含 part index 与冲突键。
  避免后一个 part 静默覆盖前一个 part 的参数、buffer 或 extra state。
- PP partition 已保留层原始索引并移除非 owner 模块，符合独立 FQN
  契约；不为重名状态自动添加前缀，保留既有 checkpoint 键布局。
- 新增参数重名、运行中新增 buffer 重名回归；后者确认失败前不修改
  cached_state_dict。无冲突合并与空 model parts 仍由已有测试覆盖。
- 验证：model state 专项 10 passed；完整 CPU 1480 passed、2 skipped；
  双进程 Gloo async_checkpoint_lifecycle_equivalence 通过，两轮恢复后
  继续训练的模型、optimizer 与 scheduler 状态一致，进程组释放正确。
- tracked-file pre-commit 通过（排除未修改用户示例）；GPU/NPU 及
  分片混合并行 checkpoint 仍需要目标硬件验证。

## Tokenizer 输入与选项契约简化（2026-10-13）

- BaseTokenizer 与 HuggingFaceTokenizer 使用明确 encode 签名：
  text 在首位，add_bos/add_eos 为 keyword-only，None 使用默认推断。
  删除手工 *args/**kwargs 拆包和无输入时静默编码空字符串的逻辑。
- decode 要求 token_ids，直接转发后端 kwargs，删除 positional/keyword
  两个重复分支；保留 skip_special_tokens 等后端选项。
- API 调整：未知 encode 选项、多余位置参数、缺失输入现在抛 TypeError；
  显式 None 从原先不添加特殊 token 改为默认推断，关闭应传 False。
  仓库 text/mm 处理与 renderer 调用已核对，无需迁移。
- 新增命名输入、None 默认、未知选项与 keyword-only 回归；复用原有
  native special token 禁用与 BOS/EOS 控制测试。
- 验证：data pipeline/chat renderer/mm 专项 245 passed；完整 CPU
  1484 passed、2 skipped；tracked-file pre-commit（排除未修改用户示例）、
  compileall、git diff HEAD --check 均通过。本轮未涉及 collective 实现。

## HF added token 标记保留与构造职责整理（2026-10-13）

- 实测 tokenizers.add_special_tokens 会把 AddedToken(special=False)
  强制变为 special=True，skip_special_tokens 解码将其删除。原加载代码
  对 added_tokens_decoder 全部走此路径，普通新增 token 语义丢失。
- 使用 add_tokens 保留各条目的 special、normalized、single_word、
  lstrip、rstrip；标准特殊 token 仍由 AddedToken(special=True) 表达。
- _process_special_token 替换为无副作用 _build_added_token；BOS/EOS
  名称在加载前各解析一次，ID 在注册后从实际 backend 查询。删除每个
  token 的重复 config 解析及随后会被覆盖的临时 config token_id 赋值。
- 新增真实本地 tokenizer 加载回归，确认 special=False 的 token 在
  skip_special_tokens=True 解码时保留，并保留 normalized/single_word。
- 验证：pipeline/chat renderer/mm 专项 246 passed；完整 CPU
  1485 passed、2 skipped；tracked-file pre-commit（排除未修改用户示例）、
  compileall、git diff HEAD --check 均通过。旧私有函数引用已清除。

## 特殊 token 配置 content 类型校验（2026-10-13）

- _get_token_from_config 原先仅校验字典有 content，未校验值类型；
  None 会被静默忽略，数字/容器可能进入 tokenizer backend 后报错。
- 字典 content 现在必须为字符串，加载时抛包含配置字段名的 ValueError；
  顶层 None 仍表示未配置。该解析函数无实例依赖，改为 staticmethod。
- 新增 None、整数、list、dict 四项真实配置加载回归。
- 复核 accelerator/tensor_tree.py 的递归搬运与 out 全树结构预校验，
  utils/dtypes.py 的显式 dtype 映射，utils/lazy_exports.py 的缺失导出语义；
  本轮未调整这些模块，未发现需要凭空改名或搬运的职责问题。
- 验证：完整 CPU 1489 passed、2 skipped；tracked-file pre-commit
  通过（排除未修改用户示例）。本轮不涉及硬件 collective 行为。

## AuxLoss hook 参数职责简化（2026-10-13）

- register_aux_loss_zero_hook 删除未读取的 parallel_dims 参数与过期 TODO；
  model_parts 改为 keyword-only，assembly 调用明确绑定该集合。
- 保留容器级 optimizer pre-hook，一步只归并一次；本地 instance_acc
  归并与 collect_aux_loss_metrics 的分布式归约边界不变。
- 此次属于无行为参数简化，复用已有 aux accumulator、trainer 装配
  回归，未增加镜像实现的测试。公开调用迁移为
  register_aux_loss_zero_hook(optimizer, model_parts=parts)。
- 验证：完整 CPU 1489 passed、2 skipped；tracked-file pre-commit
  （排除未修改用户示例）、compileall、git diff HEAD --check 均通过。

## MoE balancing hook 注册入口合并（2026-10-13）

- 合并 sign/quantile 两个注册函数为 register_moe_balancing_hook；
  model_parts、parallel_dims 使用 keyword-only，model parts 仅遍历一次。
- 完整策略校验后仅注册一个 container pre-hook：全 quantile 选择
  histogram 更新；非 quantile 且全层设置 coeff 选择 sign 更新；dense
  或全层未启用 balancing 不注册；混合策略/不一致 coeff 明确拒绝。
- 保留两个更新算法与既有 DP/CP/EP 下 TP 统计归约；assembly 从连续
  两次注册调用改为统一入口。删除旧名，同步 routing 文档及上游映射。
- 更新原 quantile-only no-op 测试为统一入口 sign 选择测试，验证
  optimizer step 更新 expert_bias 并清空计数；quantile、混合拒绝、
  dense/disabled 路径复用既有回归。
- 验证：MoE 专项 50 passed；完整 CPU 1489 passed、2 skipped；
  tracked-file pre-commit（排除未修改用户示例）、compileall、diff 通过。
  本轮未运行目标硬件 collective 数值验证。

## HF 输入预处理参数契约清理（2026-10-13）

- preprocess_inputs 删除完全未使用的 parallelism 参数及 del 语句；
  trainer runtime 的 batch/validation 不再每批传入配置，相关 CP
  integration 调用同步。CP load-balancer 仍由装配时的模型 setter 持有。
- input_dict 注解增加实际支持的 Batch，与 synthetic 和 packed dict
  两个已有实现分支一致；不修改 targets shift、CP/TP shard 与 mask 行为。
- 复用 trainer/HF/data/validation 全套测试；旧 parallelism= 调用搜索为空。
- 验证：完整 CPU 1489 passed、2 skipped；tracked-file pre-commit
  （排除未修改用户示例）、compileall、git diff HEAD --check 均通过。
  相关 CP integration 本轮仅同步接口，未在目标硬件执行。

## 数据尺寸正整数契约统一（2026-10-13）

- require_positive 重命名为 require_positive_integer，拒绝 bool、非整数
  与非正数；避免 True、浮点数和 NaN 通过尺寸入口后在 tensor/packing
  后端才失败。context/policy、collator、packing、synthetic 调用同步。
- 删除旧名，无兼容别名；新增六项 context 非整数尺寸回归。
- 验证：完整 CPU 1495 passed、2 skipped；pre-commit 首次修正两处
  import 排序后复跑通过（排除未修改用户示例），compileall 通过。

## 数据 DP 坐标校验归并（2026-10-13）

- contracts.validate_dp_coordinates 统一 rank/world size 契约，明确拒绝
  bool、非整数及越界 rank，并复用正整数 world size 校验。
- DatasetIterationPolicy 与 RandomTokenDataLoader 删除重复范围检查；
  GrainDataLoader 在图构造前调用同一校验，避免直接使用 loader 时
  非法 rank 进入状态键或 backend。
- 新增 policy/synthetic 的 bool、float rank 回归，以及 Grain loader
  的 bool、float、负值、越界 rank 提前拒绝回归。
- 验证：完整 CPU 1503 passed、2 skipped；tracked-file pre-commit
  （排除未修改用户示例）、compileall、git diff HEAD --check 通过。

## MemoryProfiler 构造参数显式绑定（2026-10-13）

- MemoryProfiler 六个构造参数改为 keyword-only，生产调用明确绑定
  step_num/freq/snapshot_dir/leaf_folder/rank/max_entries；删除难以区分
  含义的连续位置整数与空字符串传参，同步唯一位置式测试调用。
- 保留 leaf_folder 契约（测试覆盖 worker 子目录），未删除有实际
  用途的路径选项。快照频率、OOM 保存、history 清理行为不变。
- 验证：profiler 专项 44 passed；完整 CPU 1503 passed、2 skipped；
  tracked-file pre-commit（排除未修改用户示例）、compileall、diff 通过。
  allocator history 的 GPU/NPU 后端本轮未作硬件验证。

## Checkpoint 状态集合所有权与恢复顺序（2026-10-13）

- BaseCheckpointManager 持有独立状态映射，固定建立 model/optimizer/
  lr_scheduler（随后 EMA），再加入额外状态；不再修改调用方字典。
- 避免 dict.update 对已有 scheduler 键保留旧插入位置，破坏 optimizer
  先恢复、scheduler 后读取 initial_lr 的顺序。调用方后续清空字典
  不再移除 manager 状态；状态对象本身仍共享，无深拷贝。
- 新增乱序核心键、调用方字典保留与清空后的所有权回归；同步旧测试
  对构造失败时污染输入字典的断言。
- 验证：完整 CPU 1504 passed、2 skipped；双进程 Gloo 异步 checkpoint
  lifecycle 通过；tracked-file pre-commit（排除未修改用户示例）、
  compileall、diff 通过。GPU/NPU 分片恢复仍需硬件验证。

## Checkpoint 重构后测试与说明核对（2026-10-13）

- 修复上一轮所有权重构后弱化的 torch_checkpointing scheduler 注册测试：
  在共享 base 初始化后捕获 manager 状态，同时断言输入字典未污染与
  LR_SCHEDULER 仍指向传入对象。后端仍使用原有 fake 门控。
- 修正 base/DCP/package 说明：optimizer 恢复 group initial_lr，scheduler
  由其重建 base_lrs 与当前 lr；synthetic 与 Grain 都支持 cursor 恢复。
- 验证：checkpoint 专项 50 passed；tracked-file pre-commit 通过
  （排除未修改用户示例）。本轮只强化测试与说明，未重复全量训练测试。

## Loader checkpoint 元数据类型校验（2026-10-13）

- Grain 状态版本通过 get 读取并严格要求 int 1（拒绝 bool/float），
  缺失或非法版本统一抛描述明确的 ValueError。
- require_same_dp_degree 先要求保存的 degree 为正整数，再比较当前
  degree；synthetic/Grain 共用，避免 True/1.0 因 Python 相等规则通过。
- 元数据校验仍先于 iterator.set_state 或 synthetic 流重建；增加版本
  bool/float/None 与保存 degree bool/float/None/zero 回归。
- 验证：完整 CPU 1511 passed、2 skipped；pre-commit 自动格式化后
  复跑通过（排除未修改用户示例）；compileall、diff 检查通过。

## HF 数据源保留参数冲突校验补齐（2026-10-13）

- reject_duplicated_hf_fields 将 path 加入保留参数集合，避免 source.path
  与 load_dataset_kwargs 中 path 重复传参，直到 HF 后端才抛 TypeError。
- 新增两种 HF source × 五个保留字段的构造回归，确认显式字段冲突
  在数据源入口拒绝；允许的 data_files 等 kwargs 转发沿用已有测试。
- 验证：完整 CPU 1521 passed、2 skipped；ruff check/format、
  tracked-file pre-commit（排除未修改用户示例）、compileall 通过。

## 可变 HF source 构建前校验（2026-10-13）

- HF source 为可变 dataclass，load_dataset_kwargs 可在构造后修改；
  build_source 在调用 datasets.load_dataset 前复用保留字段冲突校验，
  避免构造后的编辑绕过契约而触发 backend 重复参数 TypeError。
- 保留 source 的可编辑语义，不引入冻结或深拷贝；两种 source 新增
  修改 streaming kwargs 后失败且 backend 不被调用的回归。
- 验证：完整 CPU 1523 passed、2 skipped；tracked-file pre-commit
  （排除未修改用户示例）、compileall、git diff HEAD --check 通过。

## Streaming cursor 恢复元数据校验（2026-10-13）

- HuggingFaceCursorIterator.set_state 先读取 epoch/hf，再校验 epoch 为
  非负整数（拒绝 bool/float/string），校验完成才触碰数据源。
- 本地 _epoch 在 HF 状态加载与 iterator 创建成功后更新；此调整不
  宣称第三方 backend 任意失败可回滚，仅保证元数据失败无副作用。
- 新增四项非法 epoch 回归：backend 无调用，epoch 与活跃迭代器
  保持原值，下一行仍正确。
- 验证：完整 CPU 1527 passed、2 skipped；ruff check/format、
  tracked-file pre-commit（排除未修改用户示例）、compileall 通过。

## JSONL 文件匹配契约明确（2026-10-13）

- file_patterns_to_paths 拒绝空 pattern 集合与目录等非文件匹配，
  在索引扫描前给出 source 层错误，避免空数据源或 open(directory) 异常。
- 修正说明为保留 pattern 输入顺序、每个 pattern 内排序；不改变现有
  样本顺序，新增顺序回归和两项非法输入回归。
- 验证：完整 CPU 1530 passed、2 skipped；tracked-file pre-commit
  （排除未修改用户示例）、compileall、git diff HEAD --check 通过。

## 当前结构与接口引用核对（2026-10-13）

- 实际 llmtuner Python 文件 151 个与逐文件表的 151 个路径完全一致，
  无未列出文件、无过期路径。此核对仅证明清单覆盖，不证明硬件正确性。
- 早期 packing 审查章节的 contracts.require_positive 引用同步为
  require_positive_integer；迁移历史中的旧名称保留以说明改名。
- 核对 MoE 注册旧名只剩上游来源记录；RowParallelLinear 仍是 HF
  权重分片包装器的有效名称，与模型构件 ReduceScatterLinear 区分。

## 配置模块清单与 CLI 导入归属核对（2026-10-13）

- config 下 18 个配置模块均使用 `_config.py` 后缀，并由 config/__init__.py
  按意图导出；区分 18 个模块与 17 个公开配置类，设计文档明确后者。
- cli_config.py 的 MetricsConfig、ProfilerConfig 改为从 observability_config
  定义模块导入，TrainingConfig 仅保留自身定义，消除经由 training_config 的
  重导出依赖；PARSER_GROUPS 顺序与 CLI graft 契约不变。
- config 专项测试与 ruff check/format 通过；本轮未改变字段或 CLI flag。

## 聚合配置直接导入定义模块（2026-10-13）

- llmtuner_config.py 的 MetricsConfig、ProfilerConfig、ValidationConfig
  改从各自定义模块导入，不再依赖 training_config 的重导出；cli_config
  的同类调整一并核对。
- TrainingConfig 只负责训练及其实际嵌套配置，聚合根仍保持原有字段、
  from_groups graft 顺序和 CLI 行为；未删除有意的 package-level 导出。
- 配置专项 52 passed，ruff/format 与目标文件 pre-commit 通过。

## Validation 离散步数配置类型校验（2026-10-13）

- ValidationConfig.freq/steps 先要求整数且非 bool，再执行原有范围
  检查；拒绝浮点数被当作步数、True 被当作一步及字符串/None 触发
  比较 TypeError。steps=-1 的有限数据集模式保留。
- 新增两个字段 × 四种非法类型回归，错误仍使用 ConfigError 与字段名。
- 验证：完整 CPU 1538 passed、2 skipped；pre-commit 自动格式化后
  复跑通过（排除未修改用户示例），compileall 通过。

## Tokenizer 定义模块导入边界（2026-10-13）

- datasets/dataloader.py、text/renderer.py、multimodal processors/collator
  及 loader_builder 的实现导入改为直接引用 tokenizer.base/huggingface/
  multimodal 定义模块，明确符号归属，消除对 package 重导出的依赖。
  Python 仍会先执行 package __init__，此调整不减少该包初始化导入。
- loader_builder 的 HuggingFace/Multimodal 运行时懒加载位置保持不变，
  可选 tokenizer 依赖不会被 synthetic 入口提前导入；package-level API
  继续供外部调用。
- 验证：数据与 renderer 专项 295 passed；完整 CPU 1538 passed、2 skipped；
  tracked-file pre-commit（排除未修改用户示例）、compileall、diff 通过。

## Tokenizer 包入口按需加载（2026-10-13）

- tokenizer/__init__.py 复用 lazy_exports，三个公共类按需从定义模块
  解析；BaseTokenizer 不再连带加载 HF/Multimodal 实现模块。
- 独立进程回归确认基类公开导出与定义一致，两个实现模块未加载。
- 禁用 tokenizers 的实验发现 llmtuner 配置初始化经 Transformers
  AutoConfig 仍需要该依赖；此更外层限制未由本次包入口调整解决。
- multimodal build 专项 6 passed；全量 CPU 结果另行记录。
- 完整 CPU 1539 passed、2 skipped；pre-commit、compileall、diff 检查通过。

## 配置 HF 架构推导延迟导入（2026-10-13）

- LLMTunerConfig 的 AutoConfig 导入移至 auto_fill_model 的实际 HF 分支，
  离线配置和基类接口导入不再提前加载 Transformers AutoConfig。
- 独立进程测试恢复禁用 tokenizers 的条件，BaseTokenizer 导入成功且
  HF/Multimodal 实现未加载，解决上一轮发现的外层配置依赖限制。
- HF 架构推导仍显式调用，现有本地 config 推导测试继续覆盖。
- 配置/导入专项 66 passed；tracked-file pre-commit（排除未修改用户
  示例）、compileall、diff 通过；完整 CPU 1539 passed、2 skipped。

## HF 架构字段填充重复赋值归并（2026-10-13）

- auto_fill_model 六个结构相同的 getattr/赋值合并为字段列表遍历，
  减少重复逻辑并直观列出会同步的架构字段；缺失属性仍保留当前值。
- 保留 composite text_config 与显式调用的 HF 加载顺序，不改变覆盖
  语义；复用本地/相对路径及 composite HF config 回归。
- 配置专项 60 passed；目标 pre-commit 自动格式化后复跑通过，diff 通过。

## 聚合配置 DP 推导薄封装删除（2026-10-13）

- 删除 LLMTunerConfig.derive_dp 的纯转发接口，将 DP 推导集中于
  ParallelConfig.derive_dp，与聚合配置不复制并行度 API 的边界一致。
- 生产无调用；六项 topology 测试使用聚合配置，迁移为
  cfg.parallel.derive_dp，保留 world size/EP 轴与组合约束断言。
  外部调用旧入口需迁移，无兼容别名。
- 配置/topology/CP 专项 112 passed；完整 CPU 1539 passed、2 skipped；
  目标 pre-commit、compileall、diff 检查通过。

## 配置共用下限校验严格整数化（2026-10-13）

- require_integer_at_least 用于并行 degree、模型几何、训练步数/批大小、数据
  buffer/bin 数及 profiler 计数；原实现仅比较大小，bool/浮点可能通过，
  字符串则在比较处暴露 TypeError。现统一要求非 bool 整数且不低于下限。
- 复用 field-specific 既有 ConfigError 消息格式；新增 TP、PP、microbatch
  字段对 bool/float/string/None 回归，既有配置字段原测试继续覆盖。
- 配置/parallel 专项 319 passed、2 skipped；完整 CPU 1551 passed、2 skipped。
  tracked-file pre-commit、compileall、diff 检查通过。

## DP shard 与 world size 输入契约（2026-10-13）

- ParallelConfig 拒绝非整数及 bool 的 DP shard degree，保留 -1 自动推导。
- derive_dp 在计算前拒绝零、负数及非整数 world size，避免返回零或浮点度数。
- 修正推导文档：EP 在既有 sparse region 内分配，不是额外 world 轴。
- 新增 20 项非法输入回归；完整 CPU 1591 passed、2 skipped。
  目标 pre-commit、compileall、diff 检查通过；未运行 GPU/NPU 数值验证。

## FSDP 预取窗口与训练超时校验归并（2026-10-13）

- fsdp_config 复用 require_integer_at_least 校验前后向预取计数，要求非 bool
  整数且至少为零，避免浮点数/NaN 进入窗口计算；零值语义保持。
- train_timeout_seconds 的重复比较合并到共享整数下限校验。
- 新增 19 项回归；完整 CPU 1610 passed、2 skipped；目标 pre-commit、
  compileall、diff 检查通过。GPU/NPU 预取数值与性能未验证。

## FSDP 预取运行时入口（2026-10-13）

- plan.py 的公开预取入口先验证两个计数均为非 bool 非负整数，再修改
  模块窗口，避免非法后向窗口导致前向配置已部分修改。
- ParallelConfig 导入改为定义模块；原窗口切片及反向顺序保持。
- 新增 12 项直接调用失败路径回归；完整 CPU 1622 passed、2 skipped。
  目标 pre-commit、compileall、diff 通过；GPU/NPU 验证仍未执行。

## FSDP 忽略 pattern 解析合并（2026-10-13）

- plan.py 移除重复的忽略 pattern 匹配与空结果检查，复用
  module_patterns.resolve_module_patterns；匹配结果仍递归排除子树。
- 补充无匹配及非法范围测试，复用既有子树边界测试。
- 完整 CPU 1624 passed、2 skipped；目标 pre-commit、compileall、diff
  通过。未执行 GPU/NPU FSDP 数值验证。

## TP pattern 配置校验提前（2026-10-13）

- tensor_parallel_config 在构造时拒绝同一 pattern 的 rowwise/colwise
  冲突以及未实现的 tp_sequence_modules，避免启动后才发现配置错误。
- 运行时检查保留以防配置构造后被修改；不同 pattern 的合法计划仍接受。
- 新增三项回归；完整 CPU 1627 passed、2 skipped；目标 pre-commit、
  compileall、diff 通过；未执行 GPU/NPU TP 数值验证。

## TP 目标解析与校验复用（2026-10-13）

- apply_tp 仅解析一次计划，attention 分片判断复用实际 dense targets；
  resolve_tp_targets 移除调用方未使用的计划和 EP 标志返回值。
- 计划校验现先于 head 校验；head 回归使用真实投影和合法计划，仍确认
  非整除 head 在访问 mesh 前拒绝。修正旧 MoE 实现函数名的文档引用。
- 完整 CPU 1627 passed、2 skipped；目标 pre-commit、compileall、diff
  通过；未执行 GPU/NPU TP 数值验证。

## TP 原始计划读取统一（2026-10-13）

- resolve_tp_targets 只读取一次模型原始计划，dense 解析和 MoE 声明
  检查使用同一份结果；补齐参数和目标列表返回类型。
- 动态属性回归拒绝第二次读取；完整 CPU 1628 passed、2 skipped。
  目标 pre-commit、compileall、diff 通过；GPU/NPU 验证未执行。

## TP 计划类型与参数契约（2026-10-13）

- 模型声明、计划规范化、目标解析和 apply_tp 统一使用只读 Mapping
  类型；MoE 目标列表补齐类型，配置导入使用定义模块。
- apply_tp 的可选覆盖参数仅允许 plan= 传入；仓库调用均已使用此形式。
  外部第四个位置参数需迁移。删除剩余过时 MoE 函数名注释。
- 完整 CPU 1628 passed、2 skipped；目标 pre-commit、compileall、diff
  通过；GPU/NPU 验证未执行。

## TP 根投影替换拒绝（2026-10-13）

- 根 nn.Linear 被计划选中时，原入口会把包装投影注册为空名称子模块，
  根权重未分片；现在在目标解析时拒绝并提示容器包装，早于 mesh 访问。
- 两项回归覆盖通配符及空路径，确认参数身份、形状和子模块均未修改。
- pattern 工具说明明确仅支持 * 和 ? 通配符，不暗示 shell 字符类支持。
- 完整 CPU 1630 passed、2 skipped；目标 pre-commit、compileall、diff
  通过；GPU/NPU 验证未执行。

## TP 共享权重身份保护（2026-10-13）

- dense 投影替换创建新 Parameter，默认模块遍历又去重，会破坏共享
  投影或 tied weight 的身份；现在替换前检查所有参数路径并明确拒绝。
- 两项回归覆盖共享模块及独立模块共享 Parameter，确认 mesh 访问前
  失败且模块、权重身份保留；共享目标应保持复制，尚无分片实现。
- 完整 CPU 1632 passed、2 skipped；目标 pre-commit、compileall、diff
  通过；GPU/NPU 验证未执行。

## MoE TP 全目标预校验（2026-10-13）

- 专家几何校验提取到 expert_tp_hidden_size，供直接分片与批量预校验
  复用；明确拒绝非三维权重。realize_moe_blocks_tp 先验证所有 block，
  再分片，避免后续非法 block 留下前面 block 的部分修改。
- shared expert 检查投影布局、bias 和分片维整除；bias 尚无支持实现。
- 四项回归覆盖第二个 block 的非法布局、专家维度、bias 与共享维度，
  确认各 block 参数身份和类型保留。
- 完整 CPU 1636 passed、2 skipped；目标 pre-commit、compileall、diff
  通过；GPU/NPU 数值验证未执行，运行时分配失败不保证事务回滚。

## Shared expert 属性解析统一（2026-10-13）

- resolve_shared_expert 合并预校验与分片的属性选择，按非 None 选择
  shared_expert/shared_experts 并保留实际路径。修复单数属性为 None、
  复数属性有效时预校验通过但分片断言失败的问题。
- 原双 rank CPU 部分和等价测试增加复数 fallback，精确检查排除 FQN。
- 完整 CPU 1637 passed、2 skipped；目标 pre-commit、compileall、diff
  通过；GPU/NPU 数值验证未执行。

## 验证预处理入口统一（2026-10-13）

- 非 PP 验证复用 Trainer.preprocess，与 PP 路径共享设备搬运、模型输入
  构造和额外 kwargs 处理，移除重复的 preprocess_inputs 调用。
- runtime 模块改为从定义模块导入 ValidationConfig 与 dataloader builder，
  batch 文档同步当前 synthetic 直接 seek 恢复语义。
- 完整 CPU 1637 passed、2 skipped；目标 pre-commit、compileall、diff
  通过；PP/GPU 验证未执行。

## 可观测性与 checkpoint 计数校验（2026-10-13）

- Metrics/Profiler 的频率、repeat/skip 及 memory snapshot 频率统一拒绝
  bool、浮点、NaN 和字符串；snapshot 为零仍保留 profiler 构造阶段的
  既有错误语义。
- Checkpoint interval/keep_latest_k/load_step 改用严格整数校验，避免
  配置通过后在调度或路径比较中出现隐式类型错误。
- 新增 5 项非法输入回归；完整 CPU 1642 passed、2 skipped；目标
  pre-commit、compileall、diff 通过。

## 数据与训练计数及恢复步数校验（2026-10-13）

- max_num_documents 与 chunked_loss_num_chunks 复用共享整数下限检查，
  避免浮点文档上限或 loss chunk 数进入运行时。
- checkpoint.load_step 类型检查移至大小比较之前，字符串输入明确
  报 ConfigError；保留 -1 及非负整数语义。
- 现有配置回归增加三项非法输入断言；完整 CPU 1642 passed、2 skipped。
  目标 pre-commit、compileall、diff 通过。

## Profiler 校验顺序与重复逻辑（2026-10-13）

- 周期运算移至字段类型与范围校验之后，warmup/active 非法输入统一
  报 ConfigError；可选 repeat/skip 字段合并遍历，减少重复检查。
- 新增 16 项启用/禁用 profiler 的非法周期输入回归；完整 CPU
  1658 passed、2 skipped；目标 pre-commit、compileall、diff 通过。

## Scheduler 与 EMA 整数调度契约（2026-10-13）

- warmup_steps、可选 total_steps 和 update_every_n_steps 复用整数
  下限校验；保持零 warmup、None total_steps 和合法更新周期。
- scheduler 类型导入改为定义模块，删除不符合现状的运行时导入说明。
- 新增 13 项非法输入回归；完整 CPU 1671 passed、2 skipped；目标
  pre-commit、compileall、diff 通过。

## 并行 CLI 复杂字段边界（2026-10-13）

- cli_config 将 fsdp_apply_modules 字典和 module_fqns_per_model_part
  嵌套列表加入 programmatic-only 清单，避免帮助列出无法正确构造的参数。
- Python 配置字段保留；TP/FSDP 一维 pattern 列表继续通过 CLI 暴露。
  帮助回归覆盖隐藏字段及可用参数，配置回归验证生成视图可正常装配。
- 完整 CPU 1671 passed、2 skipped；目标 pre-commit、compileall、diff
  通过。原复杂字段的 CLI 用户应迁移到 Python 配置。

## CLI scheduler 嵌套字段与装配（2026-10-13）

- 隐藏重复且不可解析的 lr_scheduler_config 参数，warmup_steps 等
  scheduler 顶层参数保留。from_groups 从基础 OptimizerConfig 字段重建，
  支持 CLI init=False 视图，避免 replace 覆盖隐藏字段时报错。
- 新增默认对象隔离回归，复用 CLI 训练与 seed checkpoint 子进程测试。
- 完整 CPU 1672 passed、2 skipped；目标 pre-commit、compileall、diff
  通过；旧嵌套 CLI 参数需迁移到 scheduler 顶层参数。

## Snapshot 零频率校验修正（2026-10-13）

- 修正先前为旧测试保留的延迟拒绝：memory_snapshot_freq=0 现在在
  配置构造时拒绝；None 仍回退到 profile_freq，正整数语义保持。
- 运行时零值检查继续覆盖构造后配置被修改的路径；ValidationConfig.freq
  合并到共享整数校验，错误格式不变。
- 完整 CPU 1672 passed、2 skipped；目标 pre-commit、compileall、diff 通过。

## 本地 GQA head 几何约束（2026-10-13）

- ModelConfig 补充 attention heads 对 KV heads 的整除校验，拒绝不能
  均匀重复 KV heads 的本地配置；HF 自动填充路径未新增通用架构限制。
- 新增非法 6/4 head 及合法 MQA/GQA/MHA 四种比例回归；完整 CPU
  1677 passed、2 skipped；目标 pre-commit、compileall、diff 通过。

## 验证 dataloader 清理归并（2026-10-13）

- PP 与非 PP 的临时 loader 关闭逻辑合并为 close_validation_dataloader，
  保留各循环 finally；验证失败时记录清理异常，正常结束时传播关闭错误。
- 复用两条路径的成功关闭、迭代/forward 失败及关闭失败回归；完整 CPU
  1677 passed、2 skipped；目标 pre-commit、compileall、diff 通过。

## Runtime loader 返回契约（2026-10-13）

- 训练 runtime 与 Trainer.build_dataloader 返回类型收紧为 BaseDataLoader，
  与所有 builder 分支一致；验证 loader 补齐返回类型，删除无作用中间变量。
- 检查多模态 collator 当前输出顶层张量和标量元数据；tensor_tree 仅接受
  张量叶子，未强行替换现有 batch 搬运，保留其 non_blocking 行为。
- 完整 CPU 1677 passed、2 skipped；目标 pre-commit、compileall、diff 通过。

## 梯度裁剪阈值有限性（2026-10-13）

- TrainingConfig 拒绝 NaN/Inf、bool 与非数值 max_norm，避免 NaN 在
  运行时 >0 判断中静默关闭裁剪；有限零/负数仍关闭裁剪并报告范数。
- 新增九项失败与边界回归；完整 CPU 1686 passed、2 skipped；目标
  pre-commit、compileall、diff 通过。

## Packing 文档上限传参简化（2026-10-13）

- 复查 DatasetBuildContext 与文本/多模态 first-fit 构造：文档上限
  已统一校验；文本直接传递可选上限，删除返回同一值的条件表达式。
- 数据专项 279 passed；目标 pre-commit、diff 通过；行为未改变。

## Streaming 重复读取失败状态（2026-10-13）

- cursor 在后端重置及迭代器创建成功后才更新本地 epoch，与恢复入口
  顺序一致；后端失败不再使本地 epoch 提前前进，不承诺后端事务回滚。
- 两项 shuffle 开关失败回归；数据专项 281 passed；目标 pre-commit、diff 通过。

## Map 数据构建薄封装合并（2026-10-13）

- 删除仅由 build_single_dataset 调用的 build_map_dataset，样本变换与迭代策略
  调用合并到 map 分支；共享变换、shuffle/shard/repeat 顺序保持。
- 同步测试注释，旧函数无别名；数据专项 281 passed；目标 pre-commit 通过。

## DatasetMix 权重输入契约（2026-10-13）

- 构建前明确要求非 bool 的有限正数权重，非法字符串/None 不再泄漏
  TypeError；五项回归确认非法输入早于子源构建失败。
- 数据专项 286 passed；目标 pre-commit、diff 通过。

## DatasetMix NumPy 实数兼容修正（2026-10-13）

- 上轮 int/float 类型判断误拒绝 NumPy float32/int64；改用 Real 契约，
  传入 Grain 前规范化为 float，仍拒绝 bool、非有限和非正数。
- 三项实际图构建回归覆盖 NumPy 实数；数据专项 289 passed；目标
  pre-commit、diff 通过。

## CLI 配置定义模块导入（2026-10-13）

- trainer/cli.py 直接从定义模块导入配置类，避免实现依赖公共导出层。
- 核对并行家族导出由入口 __all__ 限定；配置/Trainer 专项 271 passed，
  目标 pre-commit、diff 通过。

## GC 频率直接调用契约（2026-10-13）

- GarbageCollection 构造要求非 bool 正整数频率，检查先于全局 GC
  状态修改；五项非法输入回归确认不会调用 gc.disable。
- utils 专项 60 passed；目标 pre-commit、diff 通过。

## GC 初始化中断恢复（2026-10-13）

- 初始化捕获 BaseException 后恢复 GC，避免 KeyboardInterrupt/SystemExit
  在关闭自动回收后留下错误的全局状态；原异常继续传播。
- 回归覆盖三种异常与两种原始状态；utils 专项 65 passed；目标
  pre-commit、diff 通过。

## Profiler 初始化清理复用（2026-10-13）

- memory profiler 构造失败时复用 __exit__，删除重复 trace 关闭逻辑；
  保留原异常并清空资源引用，初始化和退出使用同一清理策略。
- Profiler 专项回归通过；目标 pre-commit、diff 通过。

## 组件配置类型导入统一（2026-10-13）

本轮追加审查：seeding.py 修正负数 seed 归一化说明并补全返回类型；
utils 与 PP seed 专项 70 passed。DatasetMix 在构建子源前按最小权重
规范化，避免 Grain 的缩放系数在极小权重时溢出；对无法表示的比例
提前报错。实际图覆盖最小正浮点数和极大权重，比例错误回归确认
不构建子源；数据专项 292 passed。
本轮累计完整 CPU 1709 passed、2 skipped；tracked-file pre-commit
（排除用户示例）、compileall 和 diff 检查通过。

- metrics、profiler、optimizer 与两个 checkpoint 后端的类型导入直接
  引用配置定义模块，保留 TYPE_CHECKING，避免引入运行时配置依赖。
- 累计修改完整 CPU 1706 passed、2 skipped；tracked-file pre-commit
  （排除用户示例）、compileall、diff 通过。硬件并行数值验证未执行。

## 批次元数据边界复查（2026-10-14）

训练 microbatch 只读取 token 统计并保留 loader 原始批次；删除其重复的
num_valid_tokens 过滤与字典复制。训练、验证和 PP 在共享 preprocess
入口过滤元数据，再调用模型；源字典及张量身份保持。
模块 pattern 每个表达式只解析一次，空模块清单也校验非法范围。

本轮验证：专项 127 passed；完整 CPU 1744 passed、2 skipped。
tracked-file pre-commit（排除未修改的 examples/native_alltoall.py）、
compileall 与 git diff HEAD --check 通过。未执行 GPU/NPU 分布式数值验证。

## EP 替换安装边界复查（2026-10-14）

model_replacement.py 先构建全部替换，再安装模块及 FSDP 标志；
后续 router 转换失败不再留下已替换的早期层。构建异常恢复 AuxLoss
实例计数，避免失败尝试影响后续 metric 分母。新增实际 HF 模型
后续层 router bias 拒绝回归，检查模块身份、标志及实例计数。
这不是任意副作用的完整事务：复用 shared expert 的 dtype/mode 操作
以及安装阶段异常仍需独立评估。暂存全部替换会增加装配峰值内存。

EP 本轮验证：专项 39 passed；完整 CPU 1745 passed、2 skipped。
tracked-file pre-commit（排除未修改的用户示例）、compileall 与
diff 检查通过；GPU/NPU 分布式路径未运行。

## Shared expert 复用边界（2026-10-14）

block_conversion.py 在新 MoE 子树完成 dtype/device、mode 和权重初始化
之后挂载 shared expert，避免递归 to/train 改变原 HF 模块。
成功转换与后续层拒绝均覆盖参数身份、独立 dtype、buffer dtype 和
混合子模块 training 状态。该修改解决上一轮记录的 shared expert
dtype/mode 副作用；安装阶段异常和装配峰值内存仍为独立限制。

Shared expert 本轮验证：EP 专项 41 passed；完整 CPU 1747 passed、
2 skipped。tracked-file pre-commit（排除未修改的用户示例）、compileall
及 diff 检查通过。151 个 Python 源文件与 151 个审查清单路径一致，
无遗漏、失效或重复路径；该清单一致性不代表硬件验证完成。

## EP 入口契约整理（2026-10-14）

model_replacement.py 为 process group、候选模块和替换模块补充明确
类型；无候选模块时在构建及 AuxLoss 状态快照前拒绝。修正异常类型
说明并补齐 fixed_router 参数文档。expert_parallel.py 直接读取
ParallelConfig 已声明的 fixed_router，删除掩盖缺失字段的 getattr 默认值。

EP 入口整理验证：EP/TP MoE 专项 65 passed；完整 CPU 1747 passed、
2 skipped。tracked-file pre-commit（排除未修改的用户示例）、compileall
与 diff 检查通过；未运行 GPU/NPU 分布式验证。

## Loader 装配入口复查（2026-10-14）

loader_builder.py 在任何 tokenizer、媒体依赖或数据源构建前复用
validate_dp_coordinates 校验 ownership；非法 rank/world size 不再
先触发资产加载错误。四项失败路径测试阻止 tokenizer 初始化，
覆盖 bool、越界、负 rank 和零 world size。text recipe resolver
补充 DataloaderConfig 与 BaseTokenizer 参数类型，保持延迟类型导入。

Loader 本轮验证：数据专项 193 passed；完整 CPU 1751 passed、
2 skipped。tracked-file pre-commit（排除未修改的用户示例）、compileall
及 diff 检查通过；未执行分布式硬件验证。

## PP packing 预算装配（2026-10-14）

loader_builder.py 在 tokenizer 构建前校验非 synthetic token 预算和
text PP microbatch 整除条件。直接按 microbatch 预算构建 packing
context，删除 dataclasses.replace 与重复 context；loader 保留完整
批次预算。新增无 tokenizer 资产的非法整除回归。

PP packing 本轮验证：数据及 trainer 专项 287 passed；完整 CPU
1752 passed、2 skipped。tracked-file pre-commit（排除未修改的用户
示例）、compileall 及 diff 检查通过；分布式硬件路径未验证。

## 单数据源构建流程合并（2026-10-14）

composition.py 将仅由 build_single_dataset 调用的 build_iter_dataset 合并
回 streaming 分支，删除旧函数名，无兼容转发。map 与 streaming 的
排序差异在同一构建入口可见；共享 apply_sample_transforms 保留。
streaming 仍先 window shuffle，再按 seed + dp_rank 处理样本，
map 仍先处理，再全局 shuffle、分片和 repeat。同步 fixture 说明。

单源构建合并验证：数据目录 298 passed；完整 CPU 1752 passed、
2 skipped。tracked-file pre-commit（排除未修改的用户示例）、compileall
和 diff 检查通过；llmtuner/tests 中旧函数名引用为零。

## CLI 配置责任说明复查（2026-10-14）

cli_config.py 原说明错误地称 EMA、validation、AC/compile nested
配置均通过独立 parser group graft；实际 PARSER_GROUPS 只有九组。
重写模块说明，明确仅 scheduler/checkpoint/data/metrics/profiler
独立解析，其余 nested 配置仅程序调用可配置，CLI 保留默认值。
清理 trainer/cli.py 的冗余标题。解析行为未修改。

## FFN 维度契约复查（2026-10-14）

feed_forward.py 合并 dim 与 multiple_of 校验，明确要求正整数并
拒绝 bool、浮点数、字符串和 None。此前正浮点输入可返回非整数
hidden dimension；bool 被当作 1 接受。十项非法类型回归覆盖两个
参数。校验保持模型层本地责任，不引入 datasets/config 依赖。

同步修正 tp_projections.py 构造参数说明，与实际继承及位置参数一致。
本轮模型专项 66 passed；完整 CPU 1762 passed、2 skipped。
compileall 与 diff 检查通过；未运行分布式硬件验证。

## FFN checkpoint 键冲突（2026-10-14）

feed_forward.py 拒绝同一 weight/bias 同时提供融合 w13 与完整逻辑
w1/w3 的 checkpoint，避免 merge hook 静默覆盖融合参数。两项
回归覆盖 weight/bias，检查拒绝后融合参数及调用方键保持。
这不保证任意模块加载异常的全局事务；仍保留标准 load_state_dict
的生命周期与已有单一布局加载路径。

FFN checkpoint 本轮验证：模型专项 68 passed；完整 CPU 1764 passed、
2 skipped。tracked-file pre-commit（排除未修改的用户示例）、compileall
和 diff 检查通过；未运行分布式硬件验证。

## QKV checkpoint 键冲突（2026-10-14）

attention/qkv.py 与 FFN 契约一致：完整逻辑 wq/wk/wv 与融合
wqkv 同时存在时明确拒绝，避免静默覆盖融合权重。复用 fused_key
变量，weight/bias 两项回归检查参数与调用方键保持。CPU 路径覆盖；
DTensor collective 数值路径仍未在多设备验证。

QKV 本轮验证：专项 29 passed；完整 CPU 1766 passed、2 skipped。
tracked-file pre-commit（排除未修改的用户示例）、compileall 与
diff 检查通过。

## QKV 尺寸契约（2026-10-14）

attention/qkv.py 构造 head_dim/n_heads/n_kv_heads 与 local_head_split
的 head_dim 明确要求正整数，拒绝 bool 和非整数类型。此前 2.0
可以通过构造检查，直到 reshape 才触发类型异常。二十项非法类型
回归覆盖四个入口参数，保留正数和 query/KV 整除校验。

QKV 尺寸验证：专项 49 passed；完整 CPU 1786 passed、2 skipped。
tracked-file pre-commit（排除未修改的用户示例）、compileall 与
diff 检查通过；未运行 GPU/NPU 分布式验证。

## PP schedule 输入准备统一（2026-10-14）

runtime/pipeline.py 新增 prepare_pipeline_inputs，合并训练和验证
重复的 microbatch 预处理与 arg/kwarg/target 列表构造。stage ownership
判断共用一个实现；训练私有驱动、验证 eval、loss 分母与统计保持
各自调用方责任。参数顺序统一 trainer、batch，返回列表标注明确类型。

PP 输入合并验证：trainer/validation 专项 121 passed；完整 CPU
1786 passed、2 skipped。tracked-file pre-commit（排除未修改的用户
示例）、compileall 和 diff 检查通过。PP schedule 验证包含 CPU
fake driver；未运行真实多设备 pipeline 数值验证。

## Validation runtime 契约说明（2026-10-14）

runtime/validation.py 删除过时的原样搬运说明，描述当前验证、归约
及临时 loader 生命周期职责。补齐 Trainer、Batch/TrainerBatch、
DeviceMesh 参数类型，使用 TYPE_CHECKING 避免 trainer 循环导入。
模式恢复说明明确逐模块恢复原状态，与混合 train/eval 实现一致。
执行行为未改变；按范围运行 validation 回归及格式、编译检查。

Validation 类型与说明整理：专项 28 passed，相关 pre-commit、
compileall 和 diff 检查通过；本轮不改运行行为，未重复全量测试。

## Optimizer/EMA 共享初始化命名（2026-10-14）

optimizer.py 的 _post_init 被普通 optimizer 和 EMA 共用，不能按
单调用方包装删除。重命名为 _initialize_optimizer_interface，明确
负责参数视图及 PyTorch hook 初始化，同步两个调用方并删除旧名。
精简模块历史说明，保留 torchtitan 来源、状态格式与未移植能力。

Optimizer 命名整理验证：optimizer 目录 106 passed；完整 CPU
1786 passed、2 skipped。tracked-file pre-commit（排除未修改的用户
示例）、compileall 与 diff 检查通过。

## Optimizer 参数归属保护（2026-10-14）

optimizer.py 构建 param group 时拒绝 optimizer_kwargs 中的 params
和 param_names；此前字典展开可覆盖内部匹配结果，破坏参数归属或
checkpoint FQN。两项回归覆盖保留字段。运行时校验也覆盖配置构造
后被修改的 kwargs；普通超参数覆盖优先级保持。

参数归属保护验证：optimizer 专项 108 passed；完整 CPU 1788 passed、
2 skipped。tracked-file pre-commit（排除未修改的用户示例）、compileall
与 diff 检查通过。

## Optimizer canonical FQN 冲突（2026-10-14）

optimizer.py 在构建参数组时检查 wrapper removal 后的规范名称
唯一性，明确列出冲突原名和目标名。避免两个不同参数共用 FQN，
等到 checkpoint 生成才暴露错误。新增 root weight 与 checkpoint
wrapper 子模块 weight 的实际模型回归；跨 model part 键检查仍由
现有 container state_dict 负责。

FQN 冲突验证：optimizer 专项 109 passed；完整 CPU 1789 passed、
2 skipped。tracked-file pre-commit（排除未修改的用户示例）、compileall
和 diff 检查通过。

## EMA FQN 唯一性（2026-10-14）

ema.py 的 EMAParamOptimizer 在分配平均状态前拒绝规范化后重复的
tensor FQN，参数与 buffer 路径遵守同一约束。两项回归覆盖 Parameter
和普通 Tensor；跨内层 optimizer 键冲突仍由 container state_dict
负责。

EMA 名称验证：optimizer 专项 111 passed；完整 CPU 1791 passed、
2 skipped。tracked-file pre-commit（排除未修改的用户示例）、compileall
与 diff 检查通过。

## EP checkpoint 参数索引（2026-10-14）

checkpointing.py named_params 以显式循环构造规范 FQN 索引，
不同参数共用规范名时拒绝，避免 dict comprehension 静默覆盖
live shape/placement 依据。同一对象重复引用允许。新增两项回归
覆盖 model part 冲突和重复身份；collective 硬件路径未验证。

EP checkpoint 索引验证：专项 7 passed；完整 CPU 1793 passed、
2 skipped。tracked-file pre-commit（排除未修改的用户示例）、compileall
与 diff 检查通过。

## EP checkpoint group 映射冲突（2026-10-14）

checkpointing.py expert_shard_map 拒绝同一规范 FQN 对应不同
process group，避免后续 model part 静默覆盖 collective 归属。
相同 group 对象重复引用允许；参数身份冲突由 named_params 检查。
两项纯逻辑回归覆盖拒绝和重复身份，不代表真实 collective 已验证。

EP group 映射验证：专项 9 passed；完整 CPU 1795 passed、2 skipped。
tracked-file pre-commit（排除未修改的用户示例）、compileall 和
diff 检查通过。

## EP checkpoint helper 命名（2026-10-14）

checkpointing.py named_params 重命名为 index_checkpoint_parameters，
matched_entries 重命名为 iter_expert_state_entries，明确索引与迭代
职责；删除旧名称并同步全部调用及测试。匹配方向标注 save/load
Literal，生成器标注 key/group/tensor 三元组 Iterator，执行逻辑保持。

EP helper 重命名验证：checkpoint/optimizer 专项 120 passed；相关
pre-commit、compileall 与 diff 检查通过。纯重命名及类型整理未重复
全量测试。

## Streaming 恢复结构校验（2026-10-14）

sources.py HuggingFaceCursorIterator.set_state 在调用数据集前校验
外层 dict、epoch 和 hf 字典；缺失字段明确报错。四项 malformed
state 回归以底层 load_state_dict 禁止调用确认先于源状态变化。
底层恢复失败的完整回滚仍不由该入口保证。

Streaming 结构校验验证：完整 CPU 1799 passed、2 skipped；相关
pre-commit、compileall 与 diff 检查通过。

## 数据源类型责任统一（2026-10-14）

sources.py 定义 DatasetSource 类型，集中列出 source 描述与已构建
source 的联合契约；build_source 和 SingleDataset 共用定义，删除
composition.py 重复联合类型及三个仅用于类型的 source 类导入。
执行逻辑不变，类型归 source 定义层。

数据源类型统一验证：数据目录 302 passed；相关 pre-commit、
compileall 与 diff 检查通过。类型去重未改变构建行为。

## JSONL 行类型契约（2026-10-14）

sources.py IndexedJsonlDataSource 读取后要求 JSON object，拒绝 list、
null、数字和字符串，错误带 row index 与路径。此前返回类型声称
dict，但合法非 object JSON 会进入 processor 才失败。四项回归
覆盖非法行类型，索引扫描及读取顺序保持。

JSONL 行类型验证：数据专项 202 passed；完整 CPU 1803 passed、
2 skipped。相关 pre-commit、compileall 和 diff 检查通过。

## Synthetic iterator 层级精简（2026-10-14）

synthetic.py 初次构建直接迭代无限 RandomTokenSource，与恢复路径
一致，删除不必要的 batch_iterator 重复层。有限源重复 helper 保留
其独立契约与测试。重写模块说明，区分生成、DP slicing 和有限源
repeat 三个责任。

Synthetic 迭代精简验证：数据/trainer 专项 399 passed；相关
pre-commit、compileall 与 diff 检查通过。复用已有确定性、DP slicing
及恢复回归；未重复全量测试。

## AC layer container 契约（2026-10-14）

activation_checkpoint.py 明确 layers 必须为 nn.Module，在修改
Dynamo 全局选项前拒绝 list/dict 等非法容器。此前仅检查 None，
会先修改全局设置再在 named_children 失败。三项失败路径回归
将全局 setup 替换为禁止调用，确认校验顺序。

AC container 验证：专项 38 passed；完整 CPU 1806 passed、2 skipped。
相关 pre-commit、compileall 与 diff 检查通过；GPU/NPU recompute
及混合并行数值未验证。

## AC 重计算上下文合并（2026-10-14）

activation_checkpoint.py 将仅由 marked_recompute 使用的
_recompute_aux_context 合并到该 context manager，删除单调用方
包装；ContextVar token 在 finally 恢复，范围继续包含 torch
recompute context 的进入和退出。共享 checkpoint_contexts 保留。

AC 上下文合并验证：专项 38 passed；相关 pre-commit、compileall
和 diff 检查通过。复用 loss 注入与重计算回归，未重复全量测试。

## Region AC 匹配函数命名（2026-10-14）

remat_regions.py unmatched_save_patterns 同时用于 save 与 recompute
模式，重命名为 unmatched_region_patterns，参数由 save_patterns
改为 patterns，明确中性职责。同步 AC 调用、导出及测试，删除
旧名；glob 匹配与 recompute 优先级不变。

Region 名称整理验证：memory 目录 54 passed；相关 pre-commit、
compileall 与 diff 检查通过。纯命名整理未重复全量测试。

## 上游结构映射同步（2026-10-14）

llmtuner_upstream_map.md 当前架构表移除不存在的 TP apply.py、
trainer builder.py/validate.py/batch.py 引用，改为实际 runtime
模块；配置路径统一 *_config.py 并列出总配置与解析视图，
optimizer protected hook 同步新名。历史审查记录保留原名作为
变更说明，不当作当前导入说明。

## Optimizer 参数扫描统一（2026-10-14）

optimizer.py _build_param_groups 单次读取 model.named_parameters 并
筛选 trainable 参数，各 pattern 匹配共用同一 inventory，删除重复
模型树遍历及 requires_grad 判断。first-match ownership、规范名称
冲突和未匹配错误保持。

参数扫描整理验证：optimizer 专项 111 passed；相关 pre-commit、
compileall 和 diff 检查通过。复用参数分组及冻结回归，未重复全量测试。

## Tokenizer BOS/EOS 优先级（2026-10-14）

huggingface.py 将 default_add_bos/eos 的未设置值保留为 None，
encode 仅在 None 时采用 native postprocessor 推断；显式 False
不再被 OR 运算重新开启。两项回归覆盖 BOS/EOS 配置关闭优先级，
caller encode 参数仍优先。

Tokenizer 优先级验证：数据专项 204 passed；完整 CPU 1808 passed、
2 skipped。相关 pre-commit、compileall 与 diff 检查通过。

## Tokenizer 配置读取契约（2026-10-14）

huggingface.py _load_config 改为 staticmethod，明确 UTF-8 编码与
dict[str, Any] 返回，拒绝非 object JSON。三项回归覆盖 list/null/
number；文件不存在仍返回 None，移除不使用 self 的实例方法契约。

Tokenizer 配置读取验证：数据专项 207 passed；相关 pre-commit、
compileall 和 diff 检查通过。

## Tokenizer 资产加载责任（2026-10-14）

huggingface.py _load_tokenizer_from_path 不读取实例状态，改为
staticmethod；vocab.json/vocab.txt/merges.txt 明确 UTF-8 读取，
与 tokenizer JSON 和 chat template 资产契约一致。文件选择及
BPE/WordLevel 构建顺序保持。

资产加载整理验证：数据专项 207 passed；相关 pre-commit、
compileall 和 diff 检查通过。

## Vocabulary-only WordLevel 切分（2026-10-14）

huggingface.py 为无 merges 的 WordLevel fallback 设置
WhitespaceSplit；此前无 pre-tokenizer 会把多词输入当作单个 token。
真实 vocab.txt/vocab.json 两项回归覆盖空格与 tab。该 fallback
仅定义 whitespace 词表语义，不宣称可还原任意 HF tokenizer；
复杂 normalization/pre-tokenization 仍应提供 tokenizer.json。

WordLevel fallback 验证：数据专项 209 passed；完整 CPU 1813 passed、
2 skipped。相关 pre-commit、compileall 与 diff 检查通过。

## vocab.txt token 内容保留（2026-10-14）

huggingface.py 读取词表仅去除换行符，避免 strip 改变前后空格
token；重复 token 明确拒绝并报告行号，避免覆盖旧 ID。两项
真实文件回归覆盖 token 内容与重复拒绝。空行仍跳过，其他 ID
继续按物理行号定义。

词表内容保留验证：数据专项 211 passed；相关 pre-commit、compileall
与 diff 检查通过。

## Document-aware packing 恢复校验（2026-10-14）

`datasets/packing/iterators.py` 恢复时验证布尔状态、余量偏移及状态一致性；
余量必须存在，且偏移位于样本内部。父数据源提前耗尽时明确报错，
避免旧逻辑静默结束并跳过 checkpoint 中的剩余 token。
七项回归覆盖负数、零、越界、布尔、非整数偏移，以及缺失样本。
现有四项不同消费位置的恢复测试继续验证无丢失、无重复。
恢复错误不保证回滚父数据源；失败后应中止恢复并重建 iterator。
数据专项 218 passed；相关代码 pre-commit 通过。

本轮完整 CPU 验证：1822 passed、2 skipped；tracked 文件 pre-commit
通过（保留用户文件 examples/native_alltoall.py，不纳入自动修改）；
compileall 与 git diff HEAD --check 通过。未执行多设备并行数值验证。

## Tokenizer 加载流程重组（2026-10-14）

`components/tokenizer/huggingface.py::_load_tokenizer_from_path` 改为顺序
决策和提前返回：serialized tokenizer → 词表读取 → WordLevel 或 BPE。
消除多层 elif/else，保留 tokenizer.json 优先、vocab.json 优先于
vocab.txt、物理行号 ID 和可选组件局部导入。错误提示补全无 merges
词表路径；文件列表排序，日志使用参数化消息。
新增真实文件回归：两种词表的 BPE merges 编码/解码，以及 serialized
文件优先于无效 fallback 词表。数据专项 221 passed；代码 pre-commit
通过。此调整未重命名公共 API 或移动文件。

本轮补全 tokenizer 方法返回类型与模板 kwargs 类型。完整 CPU suite
1825 passed、2 skipped；tracked pre-commit 通过（排除用户文件
examples/native_alltoall.py），compileall 与 diff 检查通过。

## EMA 更新流程整理（2026-10-14）

`components/optimizer/ema.py` 将单一调用方的 `_decay_at` 合并到
`EMA.step`，更新门控、step-derived firing count、动态/固定 decay
和平均更新在同一流程内可读。删除旧私有方法，无兼容壳。
精简模块说明，保留 FQN checkpoint、固定 tensor identity、空状态
cold start 和未移植 CPU offload 的契约；未改变公开 EMA API。
衰减测试改为观察实际平均权重，覆盖 firing 1/20/200 与固定 decay，
不再仅断言私有公式返回值。优化器专项 114 passed；相关 pre-commit
通过；代码、测试与文档无旧方法引用。

本轮完整 CPU suite：1828 passed、2 skipped；tracked pre-commit
通过（排除用户文件 examples/native_alltoall.py）；compileall 和
git diff HEAD --check 通过。未执行多设备数值验证。

## TP plan 查询接口与 FSDP pattern 校验（2026-10-14）

`tensor_parallel/tensor_parallel.py` 将泛名 `match` 改为
`match_tp_sharding_spec(module_path, *, plan=...)`，查询对象在前，策略使用
显式关键字；实现和测试同步迁移，未保留旧别名。保持插入顺序的首个
匹配规则，命中 None 的复制项仍阻止后续 sharding 规则覆盖。

`fully_shard_parallel/plan.py` 对每个 apply pattern 先编译，再筛选未忽略
模块，避免空候选集合掩盖非法 pattern，也避免每个候选重复查询缓存。
新增回归覆盖全部模块已忽略时的反向数字范围错误。
TP/FSDP/pattern 专项 77 passed；相关 pre-commit 通过。

本轮完整 CPU suite：1829 passed、2 skipped；tracked pre-commit
通过（排除用户文件 examples/native_alltoall.py）；compileall 与
diff 检查通过。未执行多设备通信或数值等价测试。

## Batch runtime 接口说明精简（2026-10-14）

`trainer/runtime/batch.py` 保持 loader 装配、fetch 计时、token 统计、
消费时搬运和模型预处理的现有边界，删除重复历史叙述，将说明集中到
DP ownership、synthetic/packed 目标差异、耗尽与 host accumulation 契约。
原有 Trainer 辅助函数补全 Trainer 参数类型，TYPE_CHECKING 导入避免
运行时循环依赖。未移动 trainer.py 的实现或改变函数参数调用方式。
训练与验证专项 121 passed；相关 pre-commit 通过。

本轮完整 CPU suite：1829 passed、2 skipped；tracked pre-commit
通过（排除用户文件 examples/native_alltoall.py）；compileall 与
diff 检查通过。未执行多设备数值验证。

## Trainer 装配参数收敛（2026-10-14）

`trainer/runtime/assembly.py` 的四个内部装配 helper 统一读取
`trainer.cfg`，删除与已绑定配置重复的 cfg 参数，避免 helper 接收不同
配置来源。自由函数使用 trainer 参数名并补全 Trainer/LLMTunerConfig/
PretrainedConfig 与模型返回类型；类型导入放在 TYPE_CHECKING。
`_apply_parallelism` 的 load_hf_weights 改为仅关键字参数，调用点显式传递。
删除未消费的模块 logger，checkpoint 相关测试调用和跟踪 hook 同步迁移。
装配阶段顺序与 checkpoint 状态注册顺序保持不变。

本轮完整 CPU suite：1829 passed、2 skipped；tracked pre-commit
通过（排除用户文件 examples/native_alltoall.py）；compileall 与
diff 检查通过。未执行多设备数值验证。

## CLI seed checkpoint world size 校验（2026-10-14）

`trainer/cli.py::main` 在创建 seed checkpoint 前优先读取已初始化
进程组的 world size；尚无进程组时才使用 WORLD_SIZE 环境变量。
避免程序先初始化多进程组、环境变量缺失或过期时绕过单设备限制。
新增回归覆盖环境值 1、实际进程组大小 2，确认构造 Trainer 前拒绝。
未修改普通训练路径、配置解析顺序或进程组清理语义。

本轮完整 CPU suite：1830 passed、2 skipped；tracked pre-commit
通过（排除用户文件 examples/native_alltoall.py）；compileall 与
diff 检查通过。进程组状态回归使用 mock，未执行多设备数值验证。

## Grain loader 构造边界整理（2026-10-14）

`datasets/dataloader.py` 精简构造说明，明确 graph 已由 builder 完成
shuffle/repeat/DP ownership；loader 的 repeat 仅影响 batch remainder 与
finite-DP 防护。context 服务于 collator/map reads，不能重新配置已构建
packing 图；max_num_documents 保留在 checkpoint 布局标识。
直接构造时校验 num_prefetch_batches 为正整数，与 DataloaderConfig
保持一致；零深度不再从直接构造路径绕过仓库配置约束。
五项回归覆盖 0、负数、布尔、非整数和字符串，在 graph/iterator 创建前报错。

本轮完整 CPU suite：1835 passed、2 skipped；tracked pre-commit
通过（排除用户文件 examples/native_alltoall.py）；compileall 与
diff 检查通过。未执行多设备数值验证。

## TP plan 解析命名与参数契约（2026-10-14）

`tensor_parallel/tensor_parallel.py` 将泛名 resolve_plan 改为
`resolve_tp_sharding_plan(model, *, plan=None)`，省略 plan 时读取模型声明。
`resolve_tp_targets(model, *, cfg, plan=None)` 同步将配置与策略设为关键字
参数。实现、dense/MoE 测试和当前上游映射引用已迁移，无旧名称兼容壳。
精简解析函数说明，保留插入顺序、复制项、MoE 结构路径及不支持布局的
拒绝契约；未改变 plan 内容或模型变换顺序。

本轮完整 CPU suite：1835 passed、2 skipped；tracked pre-commit
通过（排除用户文件 examples/native_alltoall.py）；compileall 与
diff 检查通过。未执行多设备数值验证。

## EP checkpoint 参数接口整理（2026-10-14）

`parallel/expert_parallel/checkpointing.py` 统一使用 state_dict 命名，
保存与恢复接口改为 `gather_expert_state(state_dict, *, shard_map, model_parts)`
和 `load_expert_state(state_dict, *, shard_map, model_parts)`，消除两个上下文
位置参数的歧义。模型 checkpoint、优化器（含继承路径 EMA）及 EP 回归
调用同步迁移。保持 FQN 匹配、标量跳过、DTensor materialization 和
expert slice 语义，不修改 checkpoint 文件格式。

本轮完整 CPU suite：1835 passed、2 skipped；tracked pre-commit
通过（排除用户文件 examples/native_alltoall.py）；compileall 与
diff 检查通过。未执行多设备 EP checkpoint 数值验证。

## EP expert 权重形状校验（2026-10-14）

`parallel/expert_parallel/checkpointing.py::iter_expert_state_entries` 对直接
模型权重键使用 shard_map 查表，避免每个已知 expert 权重扫描全部映射。
直接权重键在 save 时要求本地参数形状，load 时要求相同 ndim 和 trailing
维度；非法形状明确报错，不再被当作不相关状态静默跳过。load 的完整
expert 维度校验仍由 load_expert_state 负责。
优化器/EMA 前缀键继续使用原匹配规则，非参数形状状态及标量仍保留。
六项回归覆盖 save/load 下的 ndim/trailing 维度错误和独立统计状态保留。

本轮完整 CPU suite：1841 passed、2 skipped；tracked pre-commit
通过（排除用户文件 examples/native_alltoall.py）；compileall 与
diff 检查通过。未执行多设备 EP checkpoint 数值验证。

## MoE 层收集函数命名（2026-10-14）

`models/common/moe/layer.py` 的 MoE 收集函数改名为 collect_moe_layers，
明确其返回稳定列表，而非 iterator。保留 decoder 顺序、checkpoint
wrapper 解包和空列表返回；HF 模型、balancing hook、compile、测试与
文档引用同步迁移，无旧名别名。说明精简为实际查找与返回契约。

共享 MoE 回归在修复前复现 bias 更新幅度翻倍：同一对象被两层引用，
hook 从重复列表构造两份 counts，再对同一 bias 应用两次更新。
collect_moe_layers 现按对象身份去重，保留首次出现的层顺序。
新增测试通过实际 optimizer.step 验证一次系数更新，而非仅比较列表。

本轮完整 CPU suite：1842 passed、2 skipped；tracked pre-commit
通过（排除用户文件 examples/native_alltoall.py）；compileall 与
diff 检查通过。未执行多设备 MoE 数值验证。

## MoE balancing hook 跨 part 去重（2026-10-14）

`models/common/moe/bias_update.py` 注册 hook 时按对象身份合并各 part 的
MoE 列表，保留首次出现顺序，避免重复 model part 或跨 part 共享对象
再次引入重复 bias 更新。共享 MoE 回归扩展到重复 part 场景，使用
真实 optimizer.step 校验一次系数更新。
`update_expert_bias` 与 `update_quantile_expert_bias` 将 parallel_dims
改为关键字参数（默认 None），明确本地更新与并行上下文；hook、CPU
测试和 quantile 分布式脚本调用同步迁移。归约轴及策略选择不变。

本轮完整 CPU suite：1843 passed、2 skipped；tracked pre-commit
通过（排除用户文件 examples/native_alltoall.py）；compileall 与
diff 检查通过。quantile 分布式脚本仅更新调用，未在多设备执行。

## CastLinear replacement builder 命名（2026-10-14）

`models/common/cast_linear.py` 将转换 helper 改为
`build_cast_linear(linear, *, compute_dtype)`，明确构造新模块并复用原
Parameter 对象，而非就地改变输入模块。调用方仍负责安装替代模块。
HF model_setup、测试、公共导出和文档同步迁移，无旧名别名；模块
说明修正实际装配位置。保留 meta 构造、参数/bias 身份、training mode、
共享权重和 state-dict FQN，未合并用途不同的 export/compute dtype 表。

本轮完整 CPU suite：1843 passed、2 skipped；tracked pre-commit
通过（排除用户文件 examples/native_alltoall.py）；compileall 与
diff 检查通过。未执行多设备数值验证。

## HF attention 头数类型校验（2026-10-14）

`models/hf/model_setup.py::validate_head_config` 合并 attention/KV 头数
字段检查，要求非 bool 的正整数，再判断 GQA 整除关系。此前只检查
下限，布尔值和可整除浮点数可能通过，字符串会在比较时报 TypeError。
新增 12 项回归覆盖两字段的 0、负数、bool、整数值浮点数、分数值
浮点数与字符串。缺失 KV 头数时沿用 attention 头数，保持既有 fallback。

本轮完整 CPU suite：1855 passed、2 skipped；tracked pre-commit
通过（排除用户文件 examples/native_alltoall.py）；compileall 与
diff 检查通过。未执行多设备数值验证。

## HF 子模块属性名解析（2026-10-14）

`models/hf/model_setup.py` 将 first_present 改为
`resolve_submodule_name(module, names, *, component)`，明确返回属性名。
候选值须为 nn.Module；跳过 None 和非模块占位，避免有效后备子模块
存在时仍选中空属性。HF wrapper 调用、CP 说明及当前文件清单同步迁移，
不保留旧名。四项回归覆盖 None/数字/字符串占位后的有效 embedding，
以及所有候选均无有效 norm 时的构造期错误。

本轮完整 CPU suite：1859 passed、2 skipped；tracked pre-commit
通过（排除用户文件 examples/native_alltoall.py）；compileall 与
diff 检查通过。未执行多设备数值验证。

## Lazy package index 封装简化（2026-10-14）

删除 `utils/lazy_exports.py::export_names` 的 sorted 薄封装；accelerator、
memory、checkpointer、parallel、tokenizer 和 models/common 的索引
直接 sorted(_EXPORT_SOURCES)，保持 __all__/__dir__ 的排序与成员不变。
resolve_export 保留共享的延迟导入和 AttributeError 语义。精简工具说明，
去除已经过时的“五个包”数量和重复历史叙述；无旧函数别名。

本轮完整 CPU suite：1859 passed、2 skipped；tracked pre-commit
通过（排除用户文件 examples/native_alltoall.py）；compileall 与
diff 检查通过。独立进程验证六个包的索引排序，且 import 未加载
trainer.trainer 或 models.hf.model。未执行多设备数值验证。

## Dataset composition builder 命名（2026-10-14）

`datasets/composition.py` 将泛名 builder 改为
`build_single_dataset`、`build_dataset_mix`、`build_dataset_concat`，明确
它们构建的是 DatasetRecipe 节点而不是任意对象或底层 Grain 容器。
`build_dataset` dispatcher、测试 fixture 和文档历史引用同步更新，无旧名
兼容壳；参数顺序与 map/stream ordering、DP shard、repeat 语义不变。

本轮完整 CPU suite：1859 passed、2 skipped；tracked pre-commit 通过
（排除用户文件 examples/native_alltoall.py）；compileall 与 diff 检查
通过。未执行多设备数值验证。

## 配置整数下限校验命名（2026-10-14）

`config/field_checks_config.py` 将 `require_at_least` 改为
`require_integer_at_least`，名称明确表达其只接受非 bool 整数并检查
下限。所有配置模块、技能说明、审查记录引用同步更新，旧名称已删除；
错误类型和消息、minimum=0 的允许范围保持不变。

本轮完整 CPU suite：1859 passed、2 skipped；tracked pre-commit 通过
（排除用户文件 examples/native_alltoall.py）；compileall 与 diff 检查
通过。未执行多设备数值验证。

## DeviceMesh 构建入口命名（2026-10-14）

`parallel/topology.py` 将 `build_mesh` 改为 `build_device_mesh`，区分
ParallelTopology 解析和实际 DeviceMesh 构造。模块函数、ParallelTopology
方法、trainer 装配、集成脚本、测试、设计文档和上游映射全部同步更新，
旧名称已删除。mesh cache、world coverage、loss/batch/sparse view 和
单进程行为保持不变。

本轮完整 CPU suite：1859 passed、2 skipped；tracked pre-commit 通过
（排除用户文件 examples/native_alltoall.py）；compileall 与 diff 检查
通过。未执行多设备数值验证。

## 并行拓扑解析命名（2026-10-14）

`parallel/topology.py` 将 `build_parallel_topology` 改为
`resolve_parallel_topology(cfg, *, world_size)`，明确该入口解析配置与
实际 world size，返回 `ParallelTopology`，并不创建 DeviceMesh。
调用方、CPU/集成测试、设计文档、技能说明和上游映射同步迁移，旧名已
删除；world_size 改为关键字参数，单进程拒绝非 1 并行度和动态 dp_shard
推导行为保持不变。

本轮完整 CPU suite：1859 passed、2 skipped；tracked pre-commit 通过
（排除用户文件 examples/native_alltoall.py）；compileall 与 diff 检查
通过。重命名后的 world_size 关键字调用已覆盖 trainer 和 topology 测试。

## ParallelTopology 严格整数校验（2026-10-14）

`ParallelTopology._validate` 对 dp_replicate、dp_shard、cp、tp、pp、ep、
CP 子轴和 world_size 要求非 bool 整数；此前布尔值和浮点数可能参与乘法
及取模，导致错误位置不明确。dp_shard 继续允许 -1，动态推导路径不变。
错误消息保留既有 `>= 1`/`-1 or >= 1` 片段，同时补充正整数语义。
新增 28 项直接构造回归，覆盖各 degree 与 world_size 的 bool/float/string。

本轮完整 CPU suite：1887 passed、2 skipped；tracked pre-commit 通过
（排除用户文件 examples/native_alltoall.py）；compileall 与 diff 检查
通过。未执行多设备数值验证。

## 训练/验证归约 mesh 函数命名（2026-10-14）

`trainer/runtime/reductions.py` 将 `loss_reporting_meshes` 改为
`resolve_loss_reduction_meshes`，明确该函数解析 DP token-count mesh 和
loss-sum mesh，而不是执行日志 reporting。训练聚合和 validation 调用、
测试 monkeypatch、文件清单与文档同步迁移，无旧名兼容壳；
`training_reduction_meshes` 的四 mesh 返回结构和归约语义保持不变。

本轮完整 CPU suite：1887 passed、2 skipped；tracked pre-commit 通过
（排除用户文件 examples/native_alltoall.py）；compileall 与 diff 检查
通过。未执行多设备数值验证。

## Validation loader 构建命名（2026-10-14）

`trainer/runtime/validation.py` 将 `build_validation_dataloader` 改为
`create_validation_dataloader`，区分训练 loader 的 build_dataloader 与
validation 每次评估创建临时 loader 的生命周期语义。验证循环、集成
monkeypatch、测试和文档引用同步更新，无旧名兼容壳；repeat、DP ownership、
close-on-exit 和 checkpoint 不变。

本轮完整 CPU suite：1887 passed、2 skipped；tracked pre-commit 通过
（排除用户文件 examples/native_alltoall.py）；compileall 与 diff 检查
通过。未执行多设备验证。

## Compatibility matrix 行类型命名（2026-10-14）

`parallel/compatibility.py` 将泛名 `Row` 改为 `CompatibilityRule`，明确
ENTRIES 中的对象代表一条组合支持裁决规则，而不是数据表或训练 batch
行。`name`/`reason` 属性、ENTRIES 表及设计文档映射保持不变；旧名称已
删除，裁决函数和错误语义不变。

本轮完整 CPU suite：1887 passed、2 skipped；tracked pre-commit 通过
（排除用户文件 examples/native_alltoall.py）；compileall 与 diff 检查
通过。未执行多设备组合矩阵数值验证。

## XPU 峰值 FLOPs 设备探针（2026-10-14）

`utils/monitoring.py` 的 Intel Data Center GPU Max 1550 分支改用已解析的
`device_module` 和当前设备查询属性，避免绕过设备抽象直接访问 `torch.xpu`。
非 XPU 环境对该名称返回 0；新增测试验证当前设备索引和计算单元读取。

针对性监控与指标测试：69 passed；compileall 和 `git diff --check` 通过。

## 默认进程组接口（2026-10-14）

`accelerator/distributed.py` 的 `get_default_group` 改用公开的
`torch.distributed.group.WORLD`，并在进程组尚未初始化时明确返回 `None`，
去除对 `distributed_c10d._get_default_group` 私有实现的依赖。新增 CPU 单测
覆盖已初始化和未初始化两种状态；相关测试 5 passed，compileall 与 diff
检查通过。

## 并行子包显式 API（2026-10-14）

四个并行族的包索引移除 `import *`，改为从定义模块显式导出公共入口，保持
原有导入名称不变，同时让 API 来源、依赖关系和静态检查结果更直观。并行与
trainer 相关测试：415 passed、2 skipped；compileall 和 diff 检查通过。

## 基础探针返回契约（2026-10-14）

为 `accelerator.distributed.infer_launcher` 增加明确的字面量返回类型，限定
启动器结果为 `pytorch`、`slurm`、`mpi` 或 `none`；为监控内存模块和快照读取
函数补充返回类型，保持设备后端的动态接口兼容。监控、启动器和 profiler
测试：47 passed；compileall 与 diff 检查通过。

## 并行编排接口类型（2026-10-14）

`parallelize_hf_transformers` 为 `mesh` 和 `parallel_dims` 补充可空的
`DeviceMesh`/`ParallelTopology` 类型，准确表达单设备路径不创建并行拓扑的
实际契约，不改变运行时行为。激活检查点与并行测试：359 passed、2 skipped。

## Slurm 地址解析与失败路径（2026-10-14）

`accelerator/distributed.py` 仅在缺少 `MASTER_ADDR` 时调用 `scontrol`，采用
参数列表和 `check=True`，移除 shell 管道及对命令错误输出的静默接收。空主机
列表明确报错，不进入进程组初始化。测试覆盖已有地址且无 SLURM_NODELIST、
主机列表解析、空行和空白输出：6 passed。尚未验证真实 Slurm 集群。

## CUDA 内存统计设备一致性（2026-10-14）

`accelerator/device.py` 的 `get_max_cuda_memory` 查询指定设备后，现在重置
同一设备的峰值统计，修复指定非当前设备时错误清零另一设备的问题。文档将
单位准确标为 MiB；CPU 回归测试模拟 cuda:3 并检查查询与重置的设备一致性。

## MPI vendor backend 一致性（2026-10-14）

MPI 启动路径现在与 torchrun、Slurm 一样，通过设备层选择 vendor backend，
避免 XPU 等设备绑定后仍使用默认 NCCL。新增测试同时验证本地设备索引与
backend，accelerator CPU 测试 38 passed；真实 MPI/vendor 集群未验证。

## Slurm 与 torchrun 初始化复用（2026-10-14）

Slurm 发布标准 rank 环境后，其 CPU/CUDA PyTorch 路径复用
`init_dist_pytorch`，移除重复设备绑定和进程组初始化分支；vendor 和
ColossalAI 的专用路径保留。端口默认值使用 setdefault，显式 port 优先。
新增 CUDA 模拟测试验证本地 rank、全局 rank、world size、额外初始化参数
和端口优先级；accelerator 测试 40 passed。真实集群未验证。

## 配置与种子接口收敛（2026-10-14）

五个并行族验证函数声明 ParallelConfig 输入，使用 TYPE_CHECKING 保持运行时
导入方向不变。种子工具统一 deterministic 参数名；开关校验集中在
seed_everything，删除 set_seed 中的重复校验，错误说明具体字段和值。
配置测试 183 passed，种子相关测试 44 passed；Ruff 和格式检查通过。

## Checkpoint 删除错误一致性（2026-10-14）

filesystem.rmtree 合并本地与远端的错误处理，只忽略 FileNotFoundError，
本地权限或 IO 错误不再被 ignore_errors=True 吞掉，交给现有清理线程记录。
Checkpoint 测试 53 passed；Ruff、格式及 diff 检查通过。

补充完整清理调用链测试：本地删除出现 PermissionError 时，purge_thread
记录警告、继续删除下一目录并消费退出标记。Checkpoint 测试 54 passed。

## 四进程 validation 归约验证（2026-10-14）

执行 validation_reduction_equivalence.py，使用 CPU Gloo 四进程与显式
127.0.0.1 rendezvous。DP x CP、DP x TP 的 validation loss 均匹配完整
batch 参考结果。该脚本使用显式序列切片和确定性投影，验证生产 validation
循环与 collective，不代表实际 attention 或 TP/FSDP 模型变换已验证。
默认 standalone 主机解析在本机网络环境失败，显式回环地址运行退出码为 0。

## Collective 参数与均值契约（2026-10-14）

原地 all_reduce 拒绝整数 mean，防止浮点均值写回时被截断；操作名在通信前
校验。op 与 group 改为关键字参数，仓库现有调用均已符合此契约；下层 c10d
调用也使用明确参数名。两进程 Gloo reduce_equivalence 四项检查通过，覆盖
sum/max 与局部 token 分母保留；不覆盖 GPU/vendor backend。

扩展同一脚本验证 mean：float32/float64 的结果为 2.5，保留原 dtype
及张量存储地址。两进程 Gloo 五项检查通过，补齐均值分支的实际通信验证。
