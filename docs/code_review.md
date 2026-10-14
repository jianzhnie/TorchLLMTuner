# llmtuner 全局代码审查

本报告记录本轮与上一轮连续审查的结果。静态语法与 Ruff 检查覆盖
`llmtuner` 当前全部 149 个 Python 文件；本轮逐文件阅读覆盖全部 149 个文件，最新清单见
[file_review.md](file_review.md)，检查训练调用链、参数转换、恢复状态和异常清理路径。静态检查与 CPU 回归测试不能替代多设备数值等价测试，
本报告不将未运行的硬件组合记为已验证。

## 范围与职责

| 层 | Python 文件数 | 审查关注点 |
| --- | ---: | --- |
| 包入口 | 3 | 公共导出、CLI 入口 |
| `accelerator` | 7 | 设备选择、collective、嵌套 tensor 转移 |
| `components` | 21 | loss、优化器、EMA、checkpoint、metrics、profiler |
| `config` | 19 | 字段校验、CLI 配置组合、并行能力限制 |
| `datasets` | 22 | DP ownership、packing、标签对齐、多模态处理 |
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
