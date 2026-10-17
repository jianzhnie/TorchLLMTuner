# 逐文件代码审查

范围：`llmtuner/**/*.py`。状态仅代表实现阅读，不代表全部硬件组合已验证。静态检查不会自动将文件标为已审查。

2026-10-14 最新清单核对：当前源码 153 个 Python 文件，表格包含 153 个不同
路径，无遗漏、重复或已删除文件条目。此统计只证明清单覆盖，不代表全部
实现和并行组合已验证。以下为此前各轮审查记录。

此前复查 activation.py、linear.py、scatter_add.py：
激活接口、router FP32 计算与反向、TP partial bias、确定性开关恢复和自定义
autograd 注册；修正 linear.py 对旧 TP 文件名的说明。此记录不代表硬件数值验证。

复查 feed_forward.py 与 tp_projections.py：ReduceScatterLinear 的 nn.Module
基类保留，它在 TP/activation checkpoint 的 nn.Linear 类型匹配中有独立语义，
不能仅为初始化去重改成 nn.Linear。局部 53 项测试通过，但不证明改变基类安全。

sources.py 复查：拒绝字符串/bytes 被结构化随机访问协议误识别为数据源，
路径须使用明确的 source 描述；新增两项失败路径回归测试。

近期修改整体回归：1909 passed、2 skipped。tracked-file pre-commit 通过
（保护 examples/native_alltoall.py，排除该文件）；compileall 与 HEAD diff
检查通过，受保护示例无差异。composition.py 复查 map/stream 处理顺序、
混合权重和 concat 分片边界，本轮无行为修改。

lr_scheduler.py 复查恢复时先计算后提交的状态边界；get_metrics 合并重复
键名拼接，实例和参数组序号按条件追加到统一前缀，保留原有指标键名。

optimizer.py 复查：单 part 参数按 identity 去重，跨本地 parts 的重复参数
明确拒绝。修正重复参数提示与 PP optimizer 数量说明，避免将全局 PP degree
误当作当前 rank 的 model part 数量。

state_dict.py 修复 init_optim_state 的恢复范围：零梯度/零学习率分配也纳入
try/finally，分配失败仍恢复原梯度与学习率。新增分配失败回归测试。

补充多参数组失败路径：第一组已设 lr=0、第二组张量 lr 分配失败时，恢复
两组原学习率和梯度对象，且不创建 optimizer state。

module_patterns.py 复查：数字范围使用已编译正则从当前位置匹配，避免逐字符
复制剩余 pattern；通过 match.end 推进扫描，不改变 TP/FSDP pattern 语义。

fully_shard_parallel/plan.py 忽略参数改为递归收集匹配根的参数，删除对所有
后代再次递归的重复扫描。忽略名称仍覆盖整个子树，参数集合与共享参数语义不变。

pipeline_parallel/partition.py 合并首、中、末段重复的 decoder 分配循环，
先减去输入/输出模块的等效层配额，再统一分配 decoder；原模块顺序和所有权保留。

PP 分配函数在单 stage 返回前校验整数与范围，拒绝布尔度数、负层数以及
负数/非整数权重，保留零层数和零权重的合法表示。新增六项无效输入回归。

stage 索引 helper 合并到分割入口，V 型布局直接计算首尾配对编号；整除与
每 rank 两 stage 的校验改为 ValueError，避免 python -O 跳过。空分配和
无效 V 型布局在复制模型前失败。pipeline 测试 33 passed。

PipelineStage 的 get_mesh 能力只在分割入口探测一次；旧版 PyTorch 显式
传入不支持的回调时在模型复制前报错，避免静默忽略。pipeline 34 项测试通过。

prepend_first_stage_modules 拒绝存在但不是 nn.Module 的属性，避免把配置或
字符串加入模块所有权清单；失败时分配清单保持不变，缺失模块仍按变体跳过。

2026-10-14 复查 runtime/batch.py：删除 data_iterator 的随机备用源，
所有读取走装配及 checkpoint 注册的 dataloader；缺失装配明确失败。
assembly.py 的 checkpointer 构建也要求 dataloader 已初始化，始终注册
数据游标；缺失装配回归及完整 CPU 1735 passed、2 skipped。
factory.py 的 arch_overrides 区分 None 与空字典；显式空字典离线构建
架构默认配置，不调用 Hub，调用方字典保持不变。
assembly.py 在 checkpointer 注册前初始化 trainer 计数器，实际装配
回归确认注册入口能读取完整初始 state_dict。
dataloader.py 单次解析 collator batch 行数，checkpoint 布局与实际
batching 共用该值；修正图 rank 检查的过时注释。
fully_shard_parallel/plan.py 的 prefetch 原本已按注册顺序排列；上一轮
顺序修复结论撤回。注册顺序不能证明任意模型 forward 执行顺序。

已阅读文件仍可保留待核对项；其余文件明确列为待审查。

runtime/validation.py 全局结果归一化后拒绝非有限 loss，三项回归
确认 NaN/Inf 不写 metrics；验证专项 28 passed。

runtime/batch.py preprocess 统一过滤 num_valid_tokens；validation.py
非 PP 路径删除重复过滤，PP 路径共享该模型入口边界。

TrainerBatch 类型归 datasets/contracts.py，所有实现引用定义模块，
datasets 公共导出保持；不再从 loader 间接引入 collator 的共享类型。
BaseDataLoader 的迭代类型同时覆盖 Batch 与 TrainerBatch，符合两个
具体实现；批次元数据说明归 contracts。
FSDP prefetch 直接调用拒绝重复单元身份，避免自身预取和重复覆盖；
失败回归确认先于窗口 setter。

2026-10-14 依赖复查：Trainer、topology、parallelize、EP/PP/CP/FSDP
及 loader builder 的配置引用改为定义模块，类型延迟导入保持；
parallelize 说明与阶段表同步。

2026-10-13 复查：embedding.py 类名明确为 VocabParallelEmbedding；
修复分片外 token 对频率梯度和 max_norm 的影响，并支持空 shard。
新增参数/梯度等价回归，旧类名删除；activation_checkpoint.py
配置导入改为定义模块。硬件与编译路径未验证。
本轮累计完整 CPU 1712 passed、2 skipped；tracked-file pre-commit
（排除用户示例）、compileall 与 diff 检查通过。

| 文件 | 状态 | 检查内容与待办 |
| --- | --- | --- |
| `llmtuner/__init__.py` | 已阅读 | 公开配置与 Trainer 懒加载导出；检查导出路径及导入副作用。 |
| `llmtuner/__main__.py` | 已阅读 | CLI 入口转发；检查 main 调用。 |
| `llmtuner/accelerator/__init__.py` | 已阅读 | 包导出与懒加载。 |
| `llmtuner/accelerator/capabilities.py` | 已阅读 | 能力缓存、签名探测、错误提示；修复缺失 eval 时崩溃。 |
| `llmtuner/accelerator/collectives.py` | 已阅读 | TP/EP/PP 梯度范数归约、DTensor、空梯度、通信设备转换；PartialToInvariantAllReduce 的 forward SUM / backward identity CPU 契约回归；多设备等价性待验证。 |
| `llmtuner/accelerator/device.py` | 已阅读 | 设备发现、环境 rank、后端映射及设备设置。 |
| `llmtuner/accelerator/distributed.py` | 已阅读 | 三类 launcher、PG 查询和通信设备；修复 MPI LOCAL_RANK 缺失。 |
| `llmtuner/accelerator/spmd.py` | 已阅读 | 线程局部 mesh 栈、dense/sparse 注册、嵌套上下文恢复。 |
| `llmtuner/accelerator/tensor_tree.py` | 已阅读 | 统一张量树遍历、容器重建和 out 结构预校验；覆盖 namedtuple/defaultdict、键顺序、混合设备与失败前无写入。 |
| `llmtuner/components/__init__.py` | 已阅读 | 包入口及公开导出；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/checkpointer/__init__.py` | 已阅读 | __getattr__、__dir__；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/checkpointer/base.py` | 已阅读 | purge_thread、BaseCheckpointManager；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/checkpointer/checkpoint_keys.py` | 已阅读 | checkpoint 格式键与 HF 索引名；FQN 工具已迁到 utils/module_names.py。 |
| `llmtuner/utils/module_names.py` | 已阅读 | canonical_fqn；任意深度完整 wrapper 段移除，无 torch 依赖。 |
| `llmtuner/components/checkpointer/distributed_checkpoint.py` | 已阅读 | DistributedCheckpointAsyncMode、FilesystemCheckpointStorage、DistributedCheckpointManager；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/checkpointer/filesystem.py` | 已阅读 | is_remote、resolve、exists、isdir、isfile；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/checkpointer/model_state.py` | 已阅读 | ModelCheckpointState 聚合 PP/EP 状态并维护缓存；支持非张量 extra_state，按 shape/dtype/device 刷新张量缓存。 |
| `llmtuner/components/checkpointer/storage.py` | 已阅读 | CheckpointStorage 协议和 FilesystemCheckpointStorage；不依赖管理器或 DCP，保留本地及 fsspec 路径语义。 |
| `llmtuner/components/checkpointer/torch_checkpointing_backend.py` | 已阅读 | require_torch_checkpointing、BackendSymbols、BackendCheckpointStorage、init_subprocess_logging、item_specs；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/loss.py` | 已阅读 | vocab_shard_bounds、next_token_targets、cross_entropy_loss、shard_local_labels、LossParallelCrossEntropy；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/metrics.py` | 已阅读 | DeviceMemoryMonitor、build_device_memory_monitor、BaseLogger、TensorBoardLogger、WandBLogger；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/optimizer/__init__.py` | 已阅读 | 包入口及公开导出；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/optimizer/ema.py` | 已阅读 | EMAParamOptimizer、EMA；更新合并到 step，追踪范围双向校验；删除/冻结参数及删除 buffer 在更新和 cold start 前失败。 |
| `llmtuner/components/optimizer/lr_scheduler.py` | 已阅读 | wsd_factor、LRSchedulersContainer、build_lr_scheduler；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/optimizer/optimizer.py` | 已阅读 | fused_kernel_device_types、OptimizersContainer；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/optimizer/state_dict.py` | 已阅读 | init_optim_state、get_flat_optim_state_dict、load_flat_optim_state_dict、optim_state_dict_to_fqn_keys、unflatten_optim_state_dict；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/profiler.py` | 已阅读 | MemoryProfiler、Profiler、caused_by_oom；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/tokenizer/__init__.py` | 已阅读 | 包入口及公开导出；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/tokenizer/base.py` | 已阅读 | BaseTokenizer；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/tokenizer/huggingface.py` | 已阅读 | HuggingFaceTokenizer；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/tokenizer/multimodal.py` | 已阅读 | MultimodalTokenizer；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/config/__init__.py` | 已阅读 | 包入口及公开导出；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/config/activation_checkpoint_config.py` | 已阅读 | SelectiveACConfig、MemoryBudgetACConfig、RegionACConfig；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/config/checkpoint_config.py` | 已阅读 | CheckpointConfig；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/config/cli_config.py` | 已阅读 | cli_view、PARSER_GROUPS；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/config/compile_config.py` | 已阅读 | CompileConfig；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/config/data_config.py` | 已阅读 | DataloaderConfig；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/config/field_checks_config.py` | 已阅读 | require_integer_at_least；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/config/llmtuner_config.py` | 已阅读 | LLMTunerConfig；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/config/model_config.py` | 已阅读 | ModelConfig；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/config/observability_config.py` | 已阅读 | MetricsConfig、ProfilerConfig；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/config/optimizer_config.py` | 已阅读 | ParamGroupConfig、OptimizerConfig；优化器与参数组字段约束；组合独立的 LRSchedulerConfig。 |
| `llmtuner/config/lr_scheduler_config.py` | 已阅读 | LRSchedulerConfig；WSD 时长、decay 类型、比例字段的类型及范围校验。 |
| `llmtuner/config/ema_config.py` | 已阅读 | EMAConfig、validate_ema_schedule；配置与运行时共享更新计划校验。 |
| `llmtuner/config/parallel_config.py` | 已阅读 | ParallelConfig；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/config/training_config.py` | 已阅读 | TrainingConfig；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/config/validation_config.py` | 已阅读 | ValidationConfig；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/datasets/__init__.py` | 已阅读 | 包入口及公开导出；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/collators.py` | 已阅读 | Collator、TextCollator；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/contracts.py` | 已阅读 | BaseDataLoader、DataloaderExhaustedError、DP 恢复校验、build context、iteration policy、样本与 processor 工厂契约；检查对齐、形状、DP ownership。 |
| `llmtuner/datasets/dataloader.py` | 已阅读 | GrainDataLoader；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/loader_builder.py` | 已阅读 | _resolve_multimodal_recipe、_resolve_text_recipe、build_dataloader；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/composition.py` | 已阅读 | SingleDataset、DatasetMix、DatasetConcat、is_not_none、as_iter_dataset；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/multimodal/__init__.py` | 已阅读 | 包入口及公开导出；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/multimodal/collator.py` | 已阅读 | MultimodalCollator；文本/媒体拼接、padding 与 patch 装配，调用独立 MRoPE 构造函数。 |
| `llmtuner/datasets/multimodal/mrope.py` | 已阅读 | build_mrope_positions、text_positions、vision_grid_positions；文档重置、媒体/grid 对齐和三轴坐标，显式接收 merge factor 与 patch order。 |
| `llmtuner/datasets/multimodal/image.py` | 已阅读 | decode_image、smart_resize、resize_to_pixel_budget、resize_to_navit_patch_grid、process_image；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/multimodal/packing.py` | 已阅读 | build_multimodal_packing、multimodal_sample_to_packing_input、packing_output_to_multimodal_sample；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/multimodal/placeholders.py` | 已阅读 | insert_vision_placeholders；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/multimodal/processors.py` | 已阅读 | process_multimodal_sample、process_obelics_sample、process_cc12m_sample、MultimodalProcessor；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/multimodal/recipes.py` | 已阅读 | MULTIMODAL_DATASETS；流式数据源与 processor/filter 注册。 |
| `llmtuner/datasets/multimodal/video.py` | 已阅读 | load_video、process_video；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/packing/__init__.py` | 已阅读 | 包入口及公开导出；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/packing/build.py` | 已阅读 | row_lengths、build_concat_then_split_packing、build_first_fit_packing；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/packing/conversions.py` | 已阅读 | packing_output_is_full、text_sequence_to_packing_input、packing_output_to_text_sequence；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/packing/iterators.py` | 已阅读 | DocumentAwareConcatThenSplitIterDataset、DocumentAwareConcatThenSplitIterator、SplitTextSequenceDocuments、next_document_chunk_end；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/sources.py` | 已阅读 | RandomAccessDataSource、IndexedJsonlSource、IndexedJsonlDataSource、reject_duplicated_hf_fields、HuggingFaceRandomAccessSource；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/synthetic.py` | 已阅读 | RandomTokenSource、RandomTokenDataLoader；检查样本/标签对齐、DP 分片、绝对游标恢复；无生产消费者的 batch_iterator 已删除。 |
| `llmtuner/datasets/text/__init__.py` | 已阅读 | 包入口及公开导出；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/text/processors.py` | 已阅读 | read_text、TextProcessor、require_token_prefix、ChatProcessor；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/text/recipes.py` | 已阅读 | DATASETS、build_local_jsonl、build_local_jsonl_sft、build_local_jsonl_sft_multiturn；数据源、processor 和过滤器装配。 |
| `llmtuner/datasets/text/renderer.py` | 已阅读 | RendererTokenizerAdapter、build_chat_renderer；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/errors.py` | 已阅读 | 异常层次及 ValueError/NotImplementedError 兼容。 |
| `llmtuner/memory/__init__.py` | 已阅读 | AC 公开入口懒加载。 |
| `llmtuner/memory/activation_checkpoint.py` | 已阅读 | 全量/选择性/区域/预算 AC、重算辅助损失标记；可选 remat 运行待验证。 |
| `llmtuner/memory/remat_regions.py` | 已阅读 | 线性区域选择、保存/重算优先级与未匹配模式。 |
| `llmtuner/models/__init__.py` | 已阅读 | 包入口及公开导出；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/__init__.py` | 已阅读 | __getattr__、__dir__；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/activation.py` | 已阅读 | ActivationFn、SwiGLU；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/tp_projections.py` | 已阅读 | warn_once_no_tp_overlap、AllGatherFusedQKVLinear、ReduceScatterLinear、TensorParallelFeedForward；无消费者的重复前置校验已删除；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/attention/__init__.py` | 已阅读 | 包入口及公开导出；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/attention/masks.py` | 已阅读 | VarlenMetadata、round_up、get_causal_mask_mod、get_document_mask_mod、get_efficient_causal_mask_mod_for_packed_document；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/attention/qkv.py` | 已阅读 | local_head_split、QKVLinear；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/aux_loss.py` | 已阅读 | AuxLossInjection、AuxLoss、zero_aux_losses、collect_aux_loss_metrics、register_aux_loss_zero_hook；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/cast_linear.py` | 已阅读 | CastLinear、build_cast_linear；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/embedding.py` | 已阅读 | VocabParallelEmbedding；检查 padding 映射、频率梯度、max_norm 与空 shard。 |
| `llmtuner/models/common/feed_forward.py` | 已阅读 | compute_ffn_hidden_dim、FeedForward、SigmoidGatedFeedForward；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/moe/gate.py` | 已阅读 | RouterGateLinearFunction、RouterGateLinear；FP32 router GEMM 与 backward；从原 common/linear.py 搬入，未接入生产路径的 PartialBiasRowwiseLinear 已删除。 |
| `llmtuner/models/common/moe/__init__.py` | 已阅读 | 包入口及公开导出；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/moe/auxiliary_loss.py` | 已阅读 | MicrobatchWiseLoadBalanceLoss；PartialToInvariantAllReduce 从 accelerator/collectives.py 导入；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/moe/bias_update.py` | 已阅读 | update_expert_bias、update_quantile_expert_bias、register_moe_balancing_hook；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/moe/experts.py` | 已阅读 | GroupedExperts、RoutedExperts、DominoRoutedExperts；直接复用 capability probe；检查分段 counts 契约、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/moe/layer.py` | 已阅读 | MoE、collect_moe_layers；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/moe/routing.py` | 已阅读 | TokenChoiceTopKRouter、QuantileBalancedTopKRouter、QuantileBalancer；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/parallel/expert_parallel/token_dispatch.py` | 已阅读 | materialize、LocalDispatchMetadata、AllToAllDispatchMetadata、LocalTokenDispatcher、BaseEPTokenDispatcher；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/parallel/expert_parallel/torchao_dispatch.py` | 已阅读 | TorchAOTokenDispatcher、可选包安装提示、padded permutation 和 EP=1 combine；真实 torchao/CUDA 数值验证待执行。 |
| `llmtuner/models/common/multimodal.py` | 已阅读 | get_vision_positions、build_vision_bank_indices、gather_vision_embeds、scatter_vision_embeds；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/rope.py` | 已阅读 | reshape_for_broadcast、yarn_inv_freq、RoPE、ComplexRoPE、CosSinRoPE；配置归 config/rope_config.py；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/config/rope_config.py` | 已阅读 | RoPEConfig；偶数 dim、缓存长度、theta 与按模式执行的 Llama/YaRN 缩放参数校验。 |
| `llmtuner/models/common/scatter_add.py` | 已阅读 | deterministic_scatter_add、deterministic_scatter_add_fake、_backward、_setup_context；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/hf/__init__.py` | 已阅读 | 包入口及公开导出；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/hf/attention_backend.py` | 已阅读 | available_attention_backend、flex_attention_adapter、uses_deepseek_sparse_attention、register_attention_backend；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/hf/factory.py` | 已阅读 | HF 配置构建与模型类解析；FLOPs 入口归 flops.py，meta 存储物化归 materialization.py。 |
| `llmtuner/models/hf/materialization.py` | 已阅读 | materialize_meta_model；从 factory.py 搬入，负责 meta 存储分配、RoPE buffer 重建和共享参数/buffer 引用恢复；相关回归覆盖。 |
| `llmtuner/models/hf/flops.py` | 已阅读 | quadratic_attention_flops_per_token、MoEGeometry、flops_per_token、int_attr、first_int_attr；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/hf/model.py` | 已阅读 | HFTransformerModel；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/hf/model_setup.py` | 已阅读 | resolve_submodule_name、validate_head_config、configure_experts_implementation、apply_cast_lm_head；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/hf/state_dict_adapter.py` | 已阅读 | 单文件/index 共用 key 覆盖校验；真实 DCP 单文件绑定 head 加载回归，FQN 映射冲突明确拒绝；PP 硬件读取待验证。 |
| `llmtuner/parallel/__init__.py` | 已阅读 | __getattr__、__dir__；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/attention_heads.py` | 已阅读 | head_counts、require_heads_divisible_by；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/compatibility.py` | 已阅读 | CompatibilityRule、validation_once_requires_dp1、validation_once_requires_finite_corpus、chunked_loss_pp、chunked_loss_validation；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/context_parallel/__init__.py` | 已阅读 | 包入口及公开导出；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/context_parallel/attention.py` | 已阅读 | cp_all_to_all、reject_unrepresentable_attention_kwargs、run_flex、SeqToHead、HeadToSeq；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/context_parallel/context_parallel.py` | 已阅读 | require_ulysses_compatible、apply_cp；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/context_parallel/input_sharding.py` | 已阅读 | require_torch_cp、resolve_load_balancer、shard_batch_for_cp、shard_padding_mask_for_cp、shard_padding_mask_for_tp；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/context_parallel/kcp.py` | 已阅读 | _kcp_runtime、kcp_kda_forward、apply_kcp；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/expert_parallel/__init__.py` | 已阅读 | 包入口及公开导出；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/expert_parallel/block_conversion.py` | 已阅读 | restore_fp32_state_buffers、build_aux_loss、convert_block；FP32 恢复限定为 MoE/AuxLoss 自有 buffer，shared expert 不使用真值判断；多设备数值待验证。 |
| `llmtuner/parallel/expert_parallel/checkpointing.py` | 已阅读 | mark_experts_ep_sharded、expert_shard_map、named_params、matched_entries、gather_expert_state；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/expert_parallel/expert_parallel.py` | 已阅读 | apply_ep；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/expert_parallel/hf_inspection.py` | 已阅读 | router_of、FusedExperts、fused_experts_of、has_router_weight、is_hf_moe_block、resolve_shared_expert；TP/EP 复用结构探测，后层布局失败在替换前拒绝；多设备数值待验证。 |
| `llmtuner/parallel/expert_parallel/model_replacement.py` | 已阅读 | swap_hf_moe_blocks；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/fully_shard_parallel/__init__.py` | 已阅读 | 包入口及公开导出；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/fully_shard_parallel/fine_grained.py` | 已阅读 | fully_shard_fine_grained、NativeFSDPStrategy、CustomFSDPStrategy、get_fsdp_strategy；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/fully_shard_parallel/fully_shard_parallel.py` | 已阅读 | iter_transformer_layers、resolve_fsdp_mesh、resolve_sparse_fsdp_mesh、iter_fsdp_modules、disable_fsdp_gradient_division；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/fully_shard_parallel/plan.py` | 已阅读 | set_fsdp_prefetch_windows、_resolve_pattern_units、apply_fsdp_by_pattern；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/model_compile.py` | 已阅读 | per_block 的 layers 结构先检验，async TP 在注册前检查单轴/非 singleton mesh；配置定义模块导入；真实 symmetric-memory 硬件验证待执行。 |
| `llmtuner/parallel/module_patterns.py` | 已阅读 | compile_module_pattern、module_name_matches、resolve_module_patterns；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/parallelize.py` | 已阅读 | parallelize_hf_transformers；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/pipeline_parallel/__init__.py` | 已阅读 | 包入口及公开导出；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/pipeline_parallel/partition.py` | 已阅读 | generate_llm_fqn_per_model_part、split_model_into_stages；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/pipeline_parallel/pipeline_parallel.py` | 已阅读 | PipelineParallelSetup、scalar_loss_fn、make_schedule_loss_fn、get_pipeline_metadata、prepend_first_stage_modules；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/stages.py` | 已阅读 | Stage、StageName、stage_enabled 与两条阶段顺序；顺序常量使用受限类型，PP 子序列与执行器契约一致；多设备数值待验证。 |
| `llmtuner/parallel/tensor_parallel/__init__.py` | 已阅读 | 包入口及公开导出；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/tensor_parallel/linear_ops.py` | 已阅读 | ensure_symm_mem_ops、functional_collectives、all_gather_linear、linear_reduce_scatter、all_gather_along；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/tensor_parallel/tensor_parallel.py` | 已阅读 | shard_weight、ColumnParallelLinear、RowParallelLinear、ColwiseLinearNoGather、GatherSequenceFirst；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/topology.py` | 已阅读 | ParallelTopology、resolve_parallel_topology、build_device_mesh；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/trainer/__init__.py` | 已阅读 | 包入口及公开导出；检查训练编排、分母/梯度归约、checkpoint、验证和清理流程。 |
| `llmtuner/trainer/cli.py` | 已阅读 | parse_config 仅解析组合参数；HF 架构解析归 Trainer 装配，真实本地配置启动只读一次回归；main 清理与 seed checkpoint 检查。 |
| `llmtuner/trainer/runtime/__init__.py` | 已阅读 | 包入口及公开导出；检查训练编排、分母/梯度归约、checkpoint、验证和清理流程。 |
| `llmtuner/trainer/runtime/assembly.py` | 已阅读 | _build_model、_apply_parallelism、_build_optimizer_stack、_build_checkpointer；检查训练编排、分母/梯度归约、checkpoint、验证和清理流程。 |
| `llmtuner/trainer/runtime/batch.py` | 已阅读 | dp_rank_world_size、batch_size_per_rank、build_dataloader、data_iterator、batch_generator；检查训练编排、分母/梯度归约、checkpoint、验证和清理流程。 |
| `llmtuner/trainer/runtime/pipeline.py` | 已阅读 | split_pipeline_microbatches、run_pipeline_step；检查训练编排、分母/梯度归约、checkpoint、验证和清理流程。 |
| `llmtuner/trainer/runtime/reductions.py` | 已阅读 | TrainingReductionMeshes、resolve_loss_reduction_meshes、resolve_training_reduction_meshes；检查训练编排、分母/梯度归约、checkpoint、验证和清理流程。 |
| `llmtuner/trainer/runtime/validation.py` | 已阅读 | create_validation_dataloader、count_validation_batch_tokens、finalize_validation、check_validation_feasibility、should_validate；检查训练编排、分母/梯度归约、checkpoint、验证和清理流程。 |
| `llmtuner/trainer/trainer.py` | 已阅读 | Trainer；检查训练编排、分母/梯度归约、checkpoint、验证和清理流程。 |
| `llmtuner/utils/__init__.py` | 已阅读 | 空包入口，无运行逻辑。 |
| `llmtuner/utils/dtypes.py` | 已阅读 | dtype 映射及未知类型拒绝。 |
| `llmtuner/utils/garbage_collection.py` | 已阅读 | GC 启停、异常回滚、重复 close；多管理器并存未承诺。 |
| `llmtuner/utils/lazy_exports.py` | 已阅读 | 导出排序、叶模块解析及未知属性拒绝。 |
| `llmtuner/utils/logging.py` | 已阅读 | rank 过滤与层级去重；保留应用 handler，按 handler 实例判断初始化，修复 rank 查询失败递归。 |
| `llmtuner/utils/monitoring.py` | 已阅读 | 设备容量、FLOPS、颜色、内存快照；修复设备 0 硬编码。 |
| `llmtuner/utils/seeding.py` | 已阅读 | Python/NumPy/Torch 种子及确定性设置。 |

## 已确认并修复

- 旧版 pipeline 缺少 eval 时能力探测崩溃；MPI LOCAL_RANK 未发布；设备监控固定查询设备 0。
- ParallelConfig 重复校验清理；未知 packing 名称拒绝。
- TP 显式 pattern 优先于默认 pattern；EP expert down projection 完整尺寸校验。
- torch_checkpointing HF 最后一次导出的 item_specs 名称遮蔽；优化器初始化异常时恢复 lr 和 grad。
- 合成数据种子归一化为 64 位；KCP 增加独立 degree，拒绝 packed、headtail 和混合 softmax 模型。

## 后续验证与限制

- 已逐个阅读 149/149 个 Python 文件；这是实现审查覆盖，不等于所有组合均已运行通过。
- accelerator/tensor_tree.py 已明确支持容器范围，out 结构在复制前递归校验；训练归约调用只传 Tensor。
- 日志祖先传播、allocator memory history 退出关闭、指标构建失败清理仍需专项优化。
- KV AllGather × Ulysses 的 token 重排和 mask 对齐须多设备等价性验证；不能由 CPU 测试推定正确。
- KCP 依赖 Ascend FLA，新增 degree 和拒绝路径可在 CPU 验证，KDA 算子数值尚未实测。
- custom FSDP 为原生 FSDP2 适配；独立 hook 模块未实现。MC2/DeepEP/HybridEP 当前明确拒绝。
- EP checkpoint 会 materialize 全量 expert，峰值内存需硬件测量；可选 torch_checkpointing 通过 stub 验证项目侧调用。

## 验证结果

- `pytest -q --disable-warnings`：1119 passed、10 skipped。
- `pre-commit run --all-files`：全部通过（包含 Ruff lint/format）。
- `python -m compileall -q llmtuner`、`git diff HEAD --check`：通过。
- 清单与当前源文件集合一致：149 条，无缺失、无重复。

## 本轮结构调整

- EP dispatch 实现整体迁移至 `parallel/expert_parallel/token_dispatch.py`。
- 优化器状态管理集中于 `components/optimizer/state_dict.py`。
- TP 模型构件改名为 `models/common/tp_projections.py`；前馈类使用
  `TensorParallelFeedForward`，前置检查使用 `validate_tp_projection_preconditions`。
- 删除 HF `input_processing.py`：reshape 直接放在唯一调用方
  `preprocess_inputs`，目标构造直接复用 `components.loss.next_token_targets`。
- 合并 HF 预处理时源文件数为 147；合并后的预处理逻辑继续在 `models/hf/model.py` 中审查。

- 设备树模块改为 `accelerator/tensor_tree.py`，公开函数改为
  `get_tensor_tree_device` 和 `move_tensor_tree`；旧名称删除并更新全部调用方。

- 数据契约集中在 `datasets/contracts.py`：`TextSequence` 和 `SampleProcessor`
  从构建模块迁入；配方组合与 Grain 构建改为 `datasets/composition.py`，
  dataloader 装配改为 `datasets/loader_builder.py`。旧模块删除，消费者直接引用定义模块。

- 数据组合统一 `_apply_sample_transforms`，保留 map 全局种子和 streaming
  rank 偏移种子，以及两条路径不同的 shuffle/分片顺序。
- checkpoint 的延迟导入符号容器改名 `BackendSymbols`，区分运行配置和后端 API。

- checkpoint 路径协议和文件系统适配器集中于 `storage.py`；base 保留共享策略，DCP 后端保留保存/加载实现。当前源文件数为 148。

- 模型 checkpoint 状态适配独立为 `model_state.py::ModelCheckpointState`；新增模块后当前文件数为 149。

- 模型状态缓存刷新前显式校验 key 集合；检测包装后新增/删除参数或 buffer，避免保存陈旧或混合结构状态。模型状态适配回归集中于 `test_model_state.py`。

- ModelCheckpointState 构造时复制模型分块集合，避免调用方列表修改导致 checkpoint 范围与 EP shard map 不一致；保留空集合兼容性并拒绝非模块成员，覆盖两分块恢复。

- 日志初始化移除全局名称缓存，直接识别自有 handler；重复获取更新级别，保留应用 handler。祖先自有 handler 跳过已输出记录，同时保留 root 传播与 caplog 捕获。rank 查询失败直接回退，避免 DEBUG 日志递归。

本轮日志优化验证：CPU 全量测试 1138 passed、10 skipped；pre-commit、compileall 与 diff 空白检查通过。新增 5 项日志回归测试，源文件总数仍为 149。本轮未运行多设备训练数值验证。

- 日志第二轮：初始化 check/add 加锁并在 fork 后重建锁；过滤和格式化共享单次 rank 查询，formatter 复制 record 避免影响应用 handler；rank 查询不触发设备导入，仅依赖已加载的 torch.distributed 或 RANK；log_ranks 拒绝字符串、非整数及负数。

日志 rank 获取调整为复用 accelerator.distributed.is_distributed/get_rank；启动前或查询失败保留 RANK 回退，不在异常路径记录日志。此前直接查询已加载 torch.distributed 的实现已替换。

日志新增文件输出：get_logger 保留位置参数 log_level，新增关键字 log_file/file_mode，默认 UTF-8 追加、无 ANSI 颜色，只有 rank 0 写入。RankFormatter 共享格式化职责；同一规范路径共享 FileHandler，过滤按文件目的地去重，覆盖模式只首次打开截断。回归覆盖中文、追加/覆盖、层级去重、并发初始化、rank 门控、traceback、模式冲突和关闭后重新配置。

文件日志验证：CPU 全量测试 1161 passed、10 skipped；pre-commit（含新增日志测试文件）、compileall 和 diff 空白检查通过。未运行多进程硬件训练；rank 策略由 CPU 模拟验证。

日志可读性重构：删除 HANDLER_MARKER 及所有 llmtuner 动态 record 标记；ConsoleHandler/FileHandler 显式表达所有权，RecordState 通过弱引用表保存 rank 和目的地去重状态。统一 LOGGER_LOCK/LOG_RANKS 引用，修复部分改名后的未定义引用。

日志进一步整理：锁、文件 handler 弱缓存、记录弱缓存与 rank 配置集中为 LoggingState 实例属性，通过延迟缓存访问；移除模块级 LOGGER_LOCK/FILE_HANDLERS/RECORD_STATES/LOG_FORMAT，格式字符串内联于 RankFormatter 构造，文件 handler 查找创建合并进 get_logger。全量 CPU 测试 1161 passed、10 skipped，pre-commit/compileall/diff 检查通过。

日志封装合并：纯文本和彩色 formatter 统一为 RankFormatter(color=...)；MainProcessFilter/FileLogFilter 合并为 DestinationFilter(path=...)，统一 rank 策略与目的地去重。get_logger 使用局部 config 复用状态访问；保留 handler 类型用于所有权识别，共享记录状态方法用于单次 rank 查询。

## 从 CLI 开始的新一轮审查（进行中）

已复查入口 __main__.py、trainer/cli.py、config/cli_config.py、errors.py，以及 trainer 构造和退出路径、runtime/validation.py。本轮尚未重新逐文件覆盖全部源码；历史清单不代表本轮已完成。

修复：seed checkpoint 条件在 Trainer 创建前检查；进程组销毁异常不覆盖训练或资源清理的原始异常；Trainer 部分装配失败主动释放已获取资源。验证模式恢复、CLI 配置分组顺序和普通/PP 验证流程已阅读，未运行多设备数值验证。

本轮继续复查 runtime/assembly.py、runtime/batch.py、runtime/reductions.py、datasets/synthetic.py、datasets/dataloader.py、utils/seeding.py、utils/garbage_collection.py、utils/dtypes.py、utils/lazy_exports.py。修复已初始化进程组时使用环境 rank/world size 导致拓扑错误，以及 microbatch 原地删除 loader-owned 字典字段。全量 CPU 测试 1169 passed、10 skipped；pre-commit、compileall、diff 检查通过。其余文件本轮仍待复查，不将历史阅读状态算作本轮覆盖。

本轮继续阅读：datasets/loader_builder.py、contracts.py、collators.py、composition.py、sources.py、packing/build.py、packing/conversions.py、packing/iterators.py；components/optimizer/ema.py、lr_scheduler.py。修复 collator 未屏蔽输入 padding label、synthetic source 接受空形状，以及 EMA 模型列表由外部修改导致固定跟踪范围失效。全量测试 1177 passed、10 skipped；pre-commit、compileall 与 diff 检查通过。本轮逐文件复查尚未覆盖其余模型、并行和组件，不将静态检查视为全部人工审查完成。

本轮复查 optimizer/optimizer.py、optimizer/state_dict.py、checkpointer/checkpoint_keys.py、checkpointer/storage.py、checkpointer/model_state.py、checkpointer/filesystem.py、models/common/activation.py、models/common/feed_forward.py、models/common/linear.py。优化器容器复制模型列表；FQN 保存/恢复改用严格 zip，拒绝参数名数量错配。3 项元数据回归在当前环境实际通过；容器列表回归所在模块因 torch_param_names 能力门控跳过。尚未完成其余文件复查或多设备数值验证。

继续阅读 models/common/rope.py、embedding.py、cast_linear.py、attention/qkv.py、attention/masks.py、scatter_add.py、aux_loss.py、moe/auxiliary_loss.py。修复显式 RoPE 负位置按负索引读取缓存尾部，以及 fixed_block_size 非正值被接受。新增两种 RoPE 负位置与两项 block mask 配置回归；compile 分支位置检查仍按既定语义跳过，未进行多设备数值验证。

HF/Tokenizer 继续复查：已阅读 models/hf/model_setup.py、factory.py、flops.py、attention_backend.py 和 components/tokenizer/base.py、huggingface.py。修复复合配置解包遗漏 compute_dtype/experts_implementation；独立 chat_template.jinja 无 tokenizer_config.json 时仍加载。新增 4 项配置回归和 1 项模板回归。本轮全局逐文件复查仍未完成，多设备数值验证未执行。

配置继续复查：parallel/module_patterns.py、config/field_checks_config.py、compile_config.py、validation_config.py、optimizer_config.py、observability_config.py、model_config.py。修复 optimizer learning_rate/weight_decay/eps 接受非有限值，LRSchedulerConfig 提前拒绝非法 decay_type。新增 10 项回归，1199 passed、10 skipped；pre-commit、compileall、diff 检查通过。全局逐文件复查仍在进行。


本轮继续复查 config/context_parallel_config.py、tensor_parallel_config.py、expert_parallel_config.py、pipeline_parallel_config.py、fsdp_config.py、parallel_config.py，以及 parallel/compatibility.py、stages.py、parallelize.py、topology.py，accelerator/tensor_tree.py、distributed.py、collectives.py。检查了 CP 轴乘积、EP 复用 dense 区域、loss/batch mesh 范围及 PP 每分块阶段顺序。上述阅读与 CPU 模拟不构成多设备组合数值验证。

继续复查 trainer/runtime/pipeline.py，并重查 validation.py 和 datasets/dataloader.py 的失败路径。修复恢复游标/验证异常被 loader close 异常掩盖；保留原始异常并记录清理失败，正常路径关闭失败仍抛出。PP 拆分新增非空行数及整除检查，避免字典丢弃尾行或 Batch 产生不完整 microbatch 集合。新增 9 项回归，相关数据与验证测试 100 passed。全局逐文件审查仍在进行，尚未覆盖的文件不得按历史清单计作本轮完成。

补充阅读 config/checkpoint_config.py、training_config.py、data_config.py、activation_checkpoint_config.py。新增两项 PP 验证清理失败回归，本次共新增 11 项测试。最终全量 CPU 测试 1210 passed、10 skipped；pre-commit（含两个未跟踪的新测试文件）、compileall、git diff HEAD --check 均通过。未执行多设备集成测试；本轮全局逐文件审查仍未完成。


本轮完整复查 trainer/trainer.py（保持单文件），components/loss.py、metrics.py、profiler.py、checkpointer/base.py、distributed_checkpoint.py、torch_checkpointing_backend.py；继续阅读 utils/monitoring.py、models/hf/state_dict_adapter.py、models/common/moe/layer.py、bias_update.py、accelerator/spmd.py、capabilities.py。核对了梯度累积 token 分母、TP 复制梯度归约、loss reporting、optimizer/scheduler/EMA 更新与 checkpoint 时序。

修复：普通 entropy 改用稳定 log_softmax 并屏蔽 0×(-inf)，保留全屏蔽分布的 NaN；checkpoint close 即使 staging 失败也等待保存并清理，保留第一个异常且关闭后幂等；staging future 在等待前取出，避免重复抛出失败；单进程 async checkpoint 不创建未初始化的 Gloo group，真实 CPU 异步保存/恢复通过；metrics 初始化失败释放已建 sink、关闭时释放全部 sink；profiler trace 退出失败仍尝试 OOM snapshot，保留原始训练异常并清空句柄。新增 11 项回归，组件定向测试 146 passed。

尚待追踪：HF adapter 的全索引 key 覆盖检查与 PP 本地子集的关系，需要核对 PP 的 HF load/export 调用路径；SPMD 单进程空上下文在嵌套分布式上下文内的语义需要继续检查。以上为待确认项，未作未经验证的放宽。全局逐文件复查尚未完成，未运行多设备数值验证或实际 torch_checkpointing 后端 round-trip。

### CLI 起点复查：逐文件覆盖状态

当前 `llmtuner/` 共 149 个 Python 文件，本轮已完整阅读 149/149，待阅读 0 个。
历史阶段计数保留在后续记录中；本段为最新覆盖状态。已阅读不代表所有硬件组合
已通过数值验证，也不保证不存在尚未暴露的缺陷。

本次最终验证：1221 passed、10 skipped；pre-commit（包括未跟踪的新测试文件）、compileall、diff 空白检查通过。新建的 examples/native_alltoall.py 不属于本次修改，未纳入审查或变更。源码本轮覆盖 85/149，剩余清单见 file_review.md。


继续复查 config/llmtuner_config.py、accelerator/device.py、parallel/attention_heads.py、pipeline_parallel/partition.py、pipeline_parallel/pipeline_parallel.py、models/hf/model.py。本轮累计阅读 91/149；上方待审清单中的这六个文件已完成阅读，余 58 个。HF adapter × PP 待确认项已修复：全索引覆盖校验使用所有 rank key 并集，输出仍只保留本地张量；本地及全局遗漏仍拒绝。新增三项 CPU 回归和 tests/integration_tests/hf_checkpoint_pp.py，两个 CPU/Gloo 进程真实 indexed HF 加载与导出逐参数比对通过。首次 standalone rendezvous 因本机地址解析连接失败，改为显式 127.0.0.1/lo0 成功；未验证 PP schedule 前向/反向或 GPU 通信。修复中间 PP chunk 的 embedding/head 同为 Identity 时 None is None 被误判为共享权重，新增回归。SPMD None 当前明确定义为 no-op，嵌套继承外层上下文是现有契约，未作改变。全局逐文件审查仍在进行。

本次最终验证：1225 passed、10 skipped；pre-commit（含新增 integration 文件与未跟踪测试）、compileall、diff 检查通过。两个 CPU/Gloo rank 的 HF checkpoint PP round-trip 实际通过；其余硬件组合未验证。


本轮继续完整阅读 models/common/moe/routing.py、experts.py、models/common/tp_projections.py，累计 94/149，剩余 55 个。修复 GroupedExperts FP16 输入被强制转 BF16 导致尾数精度损失：融合路径仅用于 BF16，FP16 保留逐专家原 dtype 计算；现有 loop/fused-request 精确等价测试增加 FP16。QuantileBalancer 拒绝 num_bins<=0，GroupedExperts 拒绝空 dim/hidden_dim/num_experts，新增五项输入回归。全局审查仍未完成，GPU grouped GEMM 性能和多设备 MoE 数值仍未验证。

补充完整阅读 components/tokenizer/multimodal.py、datasets/multimodal/placeholders.py，核对 added-token ID 校验、placeholder 数量与负计数拒绝；未发现本次需修改的具体问题。当前本轮阅读 96/149，待审 53 个。定向模型测试 41 passed；pre-commit、compileall、diff 检查通过。

本次全量 CPU 测试：1230 passed、10 skipped。GPU grouped GEMM 及多设备 MoE 组合尚未验证；全局逐文件审查继续进行。


### 剩余文件复查完成

补充阅读 datasets/multimodal 的 image、video、packing、collator、processors；
datasets/text 的 processors、renderer；memory 的 activation_checkpoint、remat_regions；
models/common/multimodal；utils/logging；parallel/model_compile；所有 26 个包 __init__.py；
CP 的 attention、context_parallel、input_sharding、kcp；EP 的 block_conversion、
checkpointing、expert_parallel、hf_inspection、model_replacement、token_dispatch；
FSDP 的 fine_grained、fully_shard_parallel、plan；TP 的 linear_ops、tensor_parallel。
本轮逐文件阅读覆盖 149/149。

修复多模态已有 padding label 未屏蔽与处理器修改输入文本列表；TP 缺少 mesh
静默跳过及 mesh 度数不一致；pattern FSDP 遗漏 reshard 与 symmetric memory 策略。
清理包说明中的过期路径，并将 shared expert 的参数名称集合命名为 shared_names。
不拆分 trainer.py 或 metrics.py，不调整准确且有消费者的公共 API。

剩余验证限制：GPU/NPU 混合并行数值、KCP 内核、torchao padded dispatch、
symmetric memory 与真实 torch_checkpointing round-trip 未执行。

最终验证：全量 CPU 测试 1237 passed、10 skipped；pre-commit --all-files 与三个新增
文件的单独 hooks 均通过；compileall 与 git diff HEAD --check 通过。之前执行的
两进程 CPU/Gloo HF indexed checkpoint PP 加载/导出验证通过，本阶段未重复执行，
不将其作为 PP schedule 前向/反向验证。

提交前补充修正 tests/unit_tests/cpu/parallel/test_model_compile.py 的旧模块导入，
恢复此前被导入异常错误跳过的运行时测试。编译模块定向测试 12 passed、2 skipped；
最新全量测试 1244 passed、3 skipped。pre-commit、compileall 和 diff 检查通过。


### 逐文件阅读后的跨模块复查

本轮源码仍为 149 个文件，已阅读 149/149。继续修正 model_compile 的全局
regional inductor 状态与 CP 标注遗漏；核对 image token 数、patch 提取与 resize
的几何和预算契约。新增不同模型编译配置隔离、真实 CPU 编译前向/梯度、完整
视觉网格与紧预算回归。详情及上游差异见 code_review.md 最新章节。

已执行六个真实 CPU/Gloo 集成脚本：tp_equivalence、ep_equivalence、
tp_fsdp_equivalence、ep_fsdp_equivalence、pp_equivalence、cp_ulysses_equivalence。
全部通过；Ulysses 旧异常类型断言修正后重跑。显式 loopback 命令格式：
`PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --nproc-per-node=N
--master-addr=127.0.0.1 --master-port=PORT tests/integration_tests/NAME.py`。
TP/EP/FSDP/PP 使用脚本自身的 loss、梯度或多步参数参考；CP 完整模型仅前向，
all-to-all 自身另有反向与 round-trip。GPU/NPU、KCP、torchao 和真实
regional inductor 设备内核尚未验证。


补充 EP×pattern FSDP 数值回归：增加 ep_fsdp_equivalence.py 的
`--pattern-fsdp` 模式（实际路径模式 `model.model.layers.{*}`）。修复前专家权重
落在 size=4 dense mesh，loss 311.819031 对参考 311.874283，专家梯度出现
0.14～0.20 量级误差；修复后专家位于 size=2 sparse mesh，loss 311.874268，
所有路由及专家梯度通过 1e-4 容差。普通与 pattern 路径共用参数级 mesh 分配，
各 unit 的 mesh 和 shard_placement_fn 显式 override 仍优先；缺少 sparse mesh
在变换前拒绝。参考 torchtitan/distributed/fsdp.py 的专家 placement 语义。
复现日志为 /tmp/llmtuner-review-ep-pattern-fsdp-before.log，修复后日志为
/tmp/llmtuner-review-ep-pattern-fsdp-after.log；日志不是仓库提交文件。

普通 EP×FSDP 回归重跑通过。优化器能力探测按生产参数名 group 契约修正，
恢复执行被误跳过的 20 项测试，全部通过。源码清单与磁盘集合核对：149/149，
缺失文件 0、已删除但仍列出的文件 0。

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

- 标准 DCP 加载增加 `DataloaderLayoutLoadPlanner`。仅当检查点含 loader
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

- 复查 config/parallel_config.py：DP shard 为非 bool 整数，允许 -1；
  derive_dp 要求正整数 world size，EP 不参与独立 world 轴乘积。
- test_config.py 新增 20 项非法输入回归；完整 CPU 1591 passed、2 skipped。
- 目标 pre-commit、compileall、diff 检查通过；未运行 GPU/NPU 数值验证。

## FSDP 预取窗口与训练超时校验归并（2026-10-13）

- 复查 config/fsdp_config.py 与 parallel_config.py：预取窗口、训练超时
  复用共享整数下限校验；预取允许零，超时至少为一。
- 新增 19 项回归；完整 CPU 1610 passed、2 skipped；目标 pre-commit、
  compileall、diff 检查通过。GPU/NPU 预取数值与性能未验证。

## FSDP 预取运行时入口（2026-10-13）

- 复查 parallel/fully_shard_parallel/plan.py：直接调用时先验证全部
  窗口计数，再设置模块，非法后向输入不会部分修改前向窗口。
- 新增 12 项回归；完整 CPU 1622 passed、2 skipped；目标 pre-commit、
  compileall、diff 通过。GPU/NPU 验证仍未执行。

## FSDP 忽略 pattern 解析合并（2026-10-13）

- fully_shard_parallel/plan.py 的忽略 pattern 解析复用共享 resolver，
  保留子树排除和空匹配报错；增加无匹配、非法范围回归。
- 完整 CPU 1624 passed、2 skipped；目标 pre-commit、compileall、diff
  通过。未执行 GPU/NPU FSDP 数值验证。

## TP pattern 配置校验提前（2026-10-13）

- 复查 config/tensor_parallel_config.py 和 TP 入口，配置构造拒绝
  重复 pattern 的冲突样式及未实现的 sequence module plan。
- 三项回归覆盖冲突、未实现与合法计划；完整 CPU 1627 passed、2 skipped。
  目标 pre-commit、compileall、diff 通过；未执行 GPU/NPU TP 数值验证。

## TP 目标解析与校验复用（2026-10-13）

- tensor_parallel.py 的 attention 分片判断复用目标解析结果，去掉重复
  计划解析及两个未使用返回值；计划错误先于 head 错误报告。
- head 回归补充合法计划与真实投影；完整 CPU 1627 passed、2 skipped。
  目标 pre-commit、compileall、diff 通过；GPU/NPU 验证未执行。

## TP 原始计划读取统一（2026-10-13）

- tensor_parallel.py 的目标解析复用一次原始计划读取，避免 dense/MoE
  检查读取动态属性时不一致，补齐输入与返回类型。
- 新增属性读取回归；完整 CPU 1628 passed、2 skipped；目标 pre-commit、
  compileall、diff 通过；GPU/NPU 验证未执行。

## TP 计划类型与参数契约（2026-10-13）

- tensor_parallel.py 的计划输入统一为 Mapping，apply_tp 可选 plan
  改为 keyword-only，核心 model/mesh/cfg 位置参数保持；配置从叶模块导入。
- 搜索仓库调用确认无第四个位置参数；完整 CPU 1628 passed、2 skipped。
  目标 pre-commit、compileall、diff 通过；GPU/NPU 验证未执行。

## TP 根投影替换拒绝（2026-10-13）

- tensor_parallel.py 拒绝根 nn.Linear 原地替换，避免空名称子模块导致
  根权重仍复制；module_patterns.py 说明精确列出支持的通配符。
- 两项回归确认拒绝前无模型修改；完整 CPU 1630 passed、2 skipped。
  目标 pre-commit、compileall、diff 通过；GPU/NPU 验证未执行。

## TP 共享权重身份保护（2026-10-13）

- tensor_parallel.py 替换前遍历不去重参数路径，拒绝被选中且具有多个
  路径的共享权重；防止仅替换一个引用而破坏原模型语义。
- 共享模块/tied Parameter 回归通过；完整 CPU 1632 passed、2 skipped。
  目标 pre-commit、compileall、diff 通过；GPU/NPU 验证未执行。

## MoE TP 全目标预校验（2026-10-13）

- tensor_parallel.py 复用专家几何校验，并在批量 MoE 修改前验证所有
  block；共享投影拒绝 bias 和非法维度，避免可预判错误造成部分修改。
- 四项失败路径回归；完整 CPU 1636 passed、2 skipped；目标 pre-commit、
  compileall、diff 通过。GPU/NPU 数值验证未执行。

## Shared expert 属性解析统一（2026-10-13）

- tensor_parallel.py 三处属性选择复用 resolve_shared_expert，修复
  shared_expert=None 时忽略有效 shared_experts 的分片故障。
- 部分和等价回归覆盖两种路径及参数排除 FQN；完整 CPU 1637 passed、
  2 skipped；目标 pre-commit、compileall、diff 通过；GPU/NPU 未验证。

## 验证预处理入口统一（2026-10-13）

- trainer/runtime/validation.py 非 PP 路径复用 Trainer.preprocess，和 PP
  路径共享 batch 搬运与模型预处理；改用 config/validation_config.py、
  datasets/loader_builder.py 的定义模块导入。
- 完整 CPU 1637 passed、2 skipped；目标 pre-commit、compileall、diff
  通过；PP/GPU 验证未执行。

## 可观测性与 checkpoint 计数校验（2026-10-13）

- config/observability_config.py 与 checkpoint_config.py 统一使用严格
  整数下限校验，snapshot 零值保留构造期拒绝的兼容语义。
- 新增 5 项非法输入回归；完整 CPU 1642 passed、2 skipped；目标
  pre-commit、compileall、diff 通过。

## 数据与训练计数及恢复步数校验（2026-10-13）

- data_config.py、training_config.py 合并文档上限及 loss chunk 数的
  整数校验；checkpoint_config.py 先验证 load_step 类型，再比较范围。
- 配置回归增加三项断言；完整 CPU 1642 passed、2 skipped；目标
  pre-commit、compileall、diff 通过。

## Profiler 校验顺序与重复逻辑（2026-10-13）

- observability_config.py 先验证周期字段再做算术运算，可选计数校验
  合并；启用/禁用 profiler 均明确拒绝非法字段类型。
- 新增 16 项回归；完整 CPU 1658 passed、2 skipped；目标 pre-commit、
  compileall、diff 通过。

## Scheduler 与 EMA 整数调度契约（2026-10-13）

- optimizer_config.py 的 scheduler 步数和 EMA 更新间隔复用严格整数
  校验；lr_scheduler.py 的类型导入直接引用配置定义模块。
- 新增 13 项回归；完整 CPU 1671 passed、2 skipped；目标 pre-commit、
  compileall、diff 通过。

## 并行 CLI 复杂字段边界（2026-10-13）

- cli_config.py 的并行配置视图隐藏字典和嵌套列表，parallel_config.py
  明确其 programmatic-only 契约；保留一维 pattern 参数。
- CLI 帮助与配置装配回归通过；完整 CPU 1671 passed、2 skipped；目标
  pre-commit、compileall、diff 通过。

## CLI scheduler 嵌套字段与装配（2026-10-13）

- cli_config.py 隐藏 lr_scheduler_config，llmtuner_config.py 重建基础
  optimizer 配置以覆盖隐藏字段；验证视图默认对象不跨实例共享。
- CLI 训练及 seed checkpoint 回归通过；完整 CPU 1672 passed、2 skipped；
  目标 pre-commit、compileall、diff 通过。

## Snapshot 零频率校验修正（2026-10-13）

- observability_config.py 构造时拒绝零 snapshot 频率，取代先前延迟拒绝；
  profiler 测试覆盖构造后修改配置的防御检查。validation_config.py 复用
  frequency 的共享整数校验。
- 完整 CPU 1672 passed、2 skipped；目标 pre-commit、compileall、diff 通过。

## 本地 GQA head 几何约束（2026-10-13）

- model_config.py 检查 Q/KV head 整除关系，补充五项合法与非法几何回归。
- 完整 CPU 1677 passed、2 skipped；目标 pre-commit、compileall、diff 通过。

## 验证 dataloader 清理归并（2026-10-13）

- trainer/runtime/validation.py 合并两条循环的清理与异常优先级逻辑，
  原有 PP/非 PP 清理失败测试继续验证错误保留与关闭行为。
- 完整 CPU 1677 passed、2 skipped；目标 pre-commit、compileall、diff 通过。

## Runtime loader 返回契约（2026-10-13）

- batch.py、validation.py、trainer.py 的 loader 构造返回类型统一为
  BaseDataLoader；复查 tensor_tree 与多模态 batch 的不同叶子契约。
- 完整 CPU 1677 passed、2 skipped；目标 pre-commit、compileall、diff 通过。

## 梯度裁剪阈值有限性（2026-10-13）

- training_config.py 校验 max_norm 为非 bool 有限数值，保留非正数
  禁用裁剪的语义；复查 accelerator/collectives.py 的阈值消费者。
- 九项回归；完整 CPU 1686 passed、2 skipped；目标 pre-commit、compileall、
  diff 通过。

## Packing 文档上限传参简化（2026-10-13）

- 复查 contracts.py、packing/build.py 和 multimodal/packing.py 的
  上限契约，简化文本 first-fit 可选上限传参，与多模态路径一致。
- 数据专项 279 passed；目标 pre-commit、diff 通过；行为未改变。

## Streaming 重复读取失败状态（2026-10-13）

- sources.py 重复读取先重置后端，再更新本地 epoch，失败路径保留
  本地计数；两项回归覆盖 shuffle 开关，不保证后端状态回滚。
- 数据专项 281 passed；目标 pre-commit、diff 通过。

## Map 数据构建薄封装合并（2026-10-13）

- composition.py 的 map 构建转发函数合并到 build_single_dataset，删除旧函数，
  保持 map/streaming 顺序差异及共享样本变换。
- 数据专项 281 passed；目标 pre-commit 通过。

## DatasetMix 权重输入契约（2026-10-13）

- composition.py 拒绝非数值与 bool 权重，复查拒绝发生在数据源构建前。
- 五项回归；数据专项 286 passed；目标 pre-commit、diff 通过。

## DatasetMix NumPy 实数兼容修正（2026-10-13）

- composition.py 权重使用 Real 校验并规范化为 float，修正合法 NumPy
  标量被拒绝的问题；三项图构建回归通过。
- 数据专项 289 passed；目标 pre-commit、diff 通过。

## CLI 配置定义模块导入（2026-10-13）

- trainer/cli.py 配置导入迁移至定义模块；核对 TP/FSDP/EP 包导出边界。
- 配置/Trainer 专项 271 passed；目标 pre-commit、diff 通过。

## GC 频率直接调用契约（2026-10-13）

- utils/garbage_collection.py 在接管 GC 前验证正整数频率；五项回归
  确认非法输入不修改全局 GC 状态。
- utils 专项 60 passed；目标 pre-commit、diff 通过。

## GC 初始化中断恢复（2026-10-13）

- garbage_collection.py 恢复路径覆盖中断和退出异常；三种异常、两种
  初始 GC 状态回归通过，原异常保留。
- utils 专项 65 passed；目标 pre-commit、diff 通过。

## Profiler 初始化清理复用（2026-10-13）

- components/profiler.py 初始化失败复用退出清理，删除重复实现；已有
  初始化失败和清理异常优先级回归通过，目标 pre-commit、diff 通过。

## 组件配置类型导入统一（2026-10-13）

本轮复查 utils/seeding.py 的归一化说明与类型，专项 70 passed；
datasets/composition.py 在子源构建前规范化相对权重，修复 Grain
缩放中间值溢出。数据专项 292 passed，包含极端尺度实际图与
不可表示比例的提前失败回归；硬件并行数值验证未执行。
本轮累计完整 CPU 1709 passed、2 skipped；tracked-file pre-commit
（排除用户示例）、compileall 和 diff 检查通过。

- 五个组件文件的 TYPE_CHECKING 配置导入迁移至定义模块，延迟语义保持。
- 累计修改完整 CPU 1706 passed、2 skipped；tracked-file pre-commit
  （排除用户示例）、compileall、diff 通过。硬件并行数值验证未执行。

2026-10-14 复查 module_patterns.py：每个 pattern 单次编译，空模块
清单不绕过语法校验；attention_heads.py 修正整除条件说明。
runtime/batch.py 删除训练组装的重复元数据过滤，统一由 preprocess
处理；PP 与非 PP 调用链均经过该边界。

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

FLOPs 窗口解析契约补验：HF 配置声明的 sliding/chunked attention 窗口为零
或负数时，解析层返回 unresolved，主估算返回 0，避免底层严格参数校验
中断训练装配。显式 use_sliding_window=False 仍忽略窗口。新增七组回归，
FLOPs 专项 48 项通过；Ruff、format 与 diff 检查通过。设备索引复核确认
torch.device 已拒绝负数索引，因此没有加入重复的设备检查。

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

`utils/monitoring.py` 使用已解析的 `device_module` 查询 Intel GPU 的计算单元，
保持设备抽象一致并覆盖当前设备索引；非 XPU 环境安全返回未知值。新增监控
回归测试，针对性测试 69 项通过。

## 并行子包显式 API（2026-10-14）

四个并行族的 `__init__.py` 改为显式导出定义模块中的入口，删除通配符重导出，
保持公共 API 兼容并改善可读性。相关并行测试 415 passed、2 skipped。

## 基础探针返回契约（2026-10-14）

分布式启动器探针现在声明有限的返回值集合，监控内存模块和快照函数也声明
返回类型，使工具层接口更适合静态分析和后续重组，同时不改变运行时行为。

## 并行编排接口类型（2026-10-14）

`parallelize_hf_transformers` 明确标注 `mesh` 与 `parallel_dims` 在单设备路径
可为 `None`，并行路径分别使用 `DeviceMesh` 与 `ParallelTopology`，使调用契约
和 IDE 类型提示与实际流程一致。

## 默认进程组接口（2026-10-14）

`accelerator/distributed.py` 使用公开的 `torch.distributed.group.WORLD` 获取
默认进程组，并保留未初始化时的 `None` 语义，避免依赖 PyTorch 私有 API。新增
两项 CPU 回归测试覆盖该边界。

## Slurm 地址解析与失败路径（2026-10-14）

启动地址解析移除 shell 管道，已有 MASTER_ADDR 时跳过主机列表查询；空输出
在进程组创建前失败。启动器 CPU 测试 6 passed，真实 Slurm 环境未验证。

## CUDA 内存统计设备一致性（2026-10-14）

`get_max_cuda_memory` 查询与重置传入相同的设备参数，避免误重置当前设备。
新增 CPU 测试验证指定设备的调用与 MiB 单位转换。

## MPI vendor backend 一致性（2026-10-14）

`init_dist_mpi` 复用设备层的 vendor backend 选择，消除与 torchrun/Slurm
启动路径的行为差异。CPU 测试验证 XPU 本地 rank 绑定与 XCCL 选择。

## Slurm 与 torchrun 初始化复用（2026-10-14）

CPU/CUDA 的 Slurm PyTorch 初始化复用 torchrun 入口，消除重复分支；测试
覆盖设备绑定、rank 环境、额外参数与端口优先级。accelerator 40 项测试通过。

## 配置与种子接口收敛（2026-10-14）

并行族验证函数补齐 ParallelConfig 输入类型；种子开关校验合并到统一入口，
保留便捷调用的默认值。配置与种子测试分别 183 和 44 项通过。

## Checkpoint 删除错误一致性（2026-10-14）

本地与远端删除共用缺失路径处理，其他错误向调用方传播。新增权限、IO 错误
和缺失目录测试；checkpoint 测试 53 passed。

清理线程回归测试确认失败路径有警告且后续删除与退出仍正常，54 项测试通过。

## 四进程 validation 归约验证（2026-10-14）

CPU Gloo 四进程验证 DP x CP 与 DP x TP 的 validation loss 与全 batch
参考一致。范围限于 validation 循环和归约，不覆盖实际模型并行变换。

## Collective 参数与均值契约（2026-10-14）

all_reduce 的 op/group 使用关键字参数；整数 mean 明确拒绝，避免截断。
新增 CPU 失败路径测试，两进程 Gloo sum/max 等价性四项检查通过。

补充浮点 mean 的两进程测试，验证小数结果、dtype 与原地存储保持，五项检查通过。

## PP 注册模块别名清理（2026-10-14）

`partition.py` 的额外模块清理按实际注册名遍历，避免 `named_children()`
去重后遗漏共享模块的别名。仅包含 decoder 子模块的容器予以保留，未分配
的直接别名替换为 Identity；测试验证首阶段 encoder 保留、别名无 checkpoint
参数键。pipeline CPU 测试 36 passed；不代表跨设备 PP 数值验证。

## PP 整体容器分配（2026-10-14）

`partition.py` 先处理模块自身 FQN 的整体归属，再处理容器子 FQN 的局部
分配，修复显式分配 `layers` 时误清空 ModuleList/ModuleDict 的问题。
新增两类容器的回归测试，检查所有权、原始 checkpoint 键和非所属 stage
的清理；保持 HF wrapper 的实际参数路径。

## Loader 共享契约归位（2026-10-14）

BaseDataLoader、DataloaderExhaustedError 和 DP checkpoint 恢复校验迁入
已有 `datasets/contracts.py`；`dataloader.py` 专注 Grain 实现。synthetic、
trainer、validation、loader builder 和测试均从定义模块导入；包级公共
导出保持一致，移除旧实现文件中的共享定义。

## EMA 步数契约（2026-10-14）

EMAConfig 和 EMA 构造复用整数字段校验，要求 start_step/step_bias 为
非负整数、更新间隔为正整数，拒绝布尔与小数，替换 step_bias 的单独
下界判断。EMA.step 在任何平均权重更新前校验 current_step；新增配置、
直接构造和权重保持测试。

EMAConfig 与 EMA 的整数、decay 和 half-life 校验统一复用
`config/optimizer_config.py::validate_ema_schedule`；结构化参数协议允许
运行时对象复用校验而不构造配置参数袋。新增直接构造下的 NaN、Inf、
范围错误测试，并确认异常发生在平均权重分配之前。

## LR 调度长度契约（2026-10-14）

`build_lr_scheduler` 在解析 schedule 长度前检查 training_steps 为正整数；
容器构造复用整数字段校验，避免无效 total_steps 进入 LambdaLR 初始化。
回归测试检查零、负数、布尔值、小数被拒绝，且 optimizer 参数组保持不变。

## 通用接口参数语义（2026-10-14）

`move_tensor_tree` 的输出写入选项 out 和 `canonical_fqn` 的 wrapper
覆盖选项 prefixes 改为关键字参数，避免位置参数隐藏副作用或策略覆盖。
张量树 device 标注补齐已有的字符串支持；仓库调用方无需迁移。

## CastLinear 计算 dtype 契约（2026-10-14）

CastLinear 在参数初始化前检查 compute_dtype，拒绝整数、布尔、复数和
字符串，避免 forward 时断开梯度或迟报错。普通浮点类型（包括 float64）
保留支持；测试验证失败构造不改变 RNG 或原 Linear 参数身份。

## Vocab embedding shard 布局契约（2026-10-14）

TP embedding 使用共享 vocab bounds 验证本地权重形状，拒绝完整词表权重、
错误行数或错误隐藏维度，避免错误 rank offset 导致静默查表错误。测试
验证 max_norm 前拒绝错误布局且参数不变；已有合法空 shard 行为保留。

## FFN checkpoint 合并失败路径（2026-10-14）

FeedForward 的加载 hook 先准备 weight/bias 的全部融合结果，再统一替换
逻辑键。修复 stack 失败时提前 pop 输入，以及 bias 错误时 weight 已被
转换的问题。新增形状不匹配与融合/逻辑 bias 冲突测试，确认所有原条目
及其张量身份保持不变。

QKV 加载 hook 同步采用先准备、后替换的流程；保留 DTensor gather 与
head-group reshape。新增错误 bias 形状和融合/逻辑键冲突测试，验证
失败时 weight 与 bias 的原条目及张量身份均不变。

FFN/QKV 保存 hook 同步先准备全部逻辑投影，再删除融合键并更新输出；
避免错误 bias 的 reshape/unflatten 失败时已消耗合法 weight。新增两项
直接 hook 回归测试，检查原键与张量引用保持不变。

## QKV 空 token 布局（2026-10-14）

QKV head 拆分改用特征轴 unflatten 和 head 轴 flatten，避免空 token
维度与 reshape 的 -1 同时出现导致推导失败；local_head_split 复用同样
的轴拆分语义。新增空输入前向/反向与多维空 token 轴测试。

## MoE 多层统计归约统一（2026-10-14）

sign/quantile bias 更新共用展平拼接、归约、原形状还原流程，移除各自的
stack/unbind 逻辑，避免要求每层专家数或 histogram 形状相同。归约轴与
每轴 collective 次数保持不变。新增不同专家数层的 bias 更新等价及计数
清零测试；多 rank 硬件数值行为仍需对应环境验证。

补充不同专家数及 histogram bins 的 quantile 更新测试，验证 DP/CP
归约、bias 估计和统计清零。CPU Gloo 四进程运行
`tests/integration_tests/quantile_bias_equivalence.py`：DP×CP、DP×TP（EP
关闭/开启）三种统计归约均与参考一致，覆盖大整数计数；不覆盖实际
GPU/NPU 模型变换或 token dispatch。

## Trainer token 计数临时分配消除（2026-10-14）

普通 Batch 的有效 token 计数直接统计每行 labels 的第二个位置起，去掉
完整 next-token target 张量构造及额外 reshape。保持行边界、IGNORE_INDEX
与空序列拒绝语义；packed labels 的计数路径不变，复用已有分母回归测试。

## Validation 双循环合并（2026-10-14）

普通/PP 验证合并到 run_validation_loop，共用 loader 生命周期、token
统计、loss 累积、最终归约与报表；PP 专用 evaluate_pipeline_batch 只负责
微批次准备和 schedule.eval。删除 run_pipeline_validation_loop 的重复
循环，同步更新调用测试和设计/上游文档。Trainer 主体保持完整。

验证 loader 创建后的计数张量初始化纳入 try/finally，避免设备分配失败
时泄漏 loader。普通/PP 两种路径的回归测试确认分配失败后 close 仍执行，
且 cleanup 异常不覆盖原始分配错误。

## CLI 显式参数输入（2026-10-14）

parse_config 接受可选参数序列，未传时仍解析 sys.argv。程序内调用可以
传 list/tuple 或空序列，无需修改全局 argv；测试验证显式输入忽略环境
参数、进程 argv 保持不变，隐式 .args 文件仍禁用。

## 数据构建参数命名统一（2026-10-14）

source、composition、packing、multimodal packing 和 loader 装配统一使用
iteration_policy 参数，替换重复冗长的 dataset_iteration_policy；对应
fixture 和数据测试同步迁移，不保留旧关键字别名。类型仍为
DatasetIterationPolicy；程序内自定义数据配方需改用 iteration_policy=。

## HF source 描述字段合并（2026-10-14）

map/streaming 两种 HF source 的 path/split/name/revision/kwargs 与构造
校验提取到内部 _HuggingFaceSource dataclass，公开 source 类保留明确类型。
构建分支、字段参数、惰性加载和构造后重新校验行为保持一致，复用已有
source 字段冲突与 streaming 状态恢复测试。

## Grain 数据图类型归位（2026-10-14）

GrainDataset 共享类型归入 datasets/contracts.py，composition 与 Grain
loader 从定义模块导入，loader 参数复用该类型。包级公共导出保留，
composition 的公共导出清单移除该契约类型。

## Trainer 组件依赖导入统一（2026-10-14）

Trainer 的 DistributedCheckpointManager、EMA、LRSchedulersContainer
从定义 leaf 模块导入，与 runtime 装配一致；包级导出作为公共 API 保留。
训练主体与组件行为不变，减少实现对包索引导出安排的依赖。

## Dataset recipe 分派入口合并（2026-10-14）

build_dataset 用 recipe 类匹配选择 builder，共用 context/iteration_policy
参数转发，删除三份相同调用结构。类匹配继续接受子类，未知类型保留
TypeError；具体 source、mix、concat 构建语义仍由各自 builder 负责。

## Renderer tokenizer 适配器命名（2026-10-14）

RendererTokenizerWrapper 改名 RendererTokenizerAdapter，明确其接口转换
职责，移除旧导出并同步测试与文档；上游原名保留在来源映射中。
构造、decode 输入和编码输出补齐类型标注，保留 renderers 的调用协议。

## Tokenizer BOS/EOS 布尔配置（2026-10-14）

合并 BOS/EOS 默认开关解析循环，拒绝非布尔且非 null 配置，避免字符串
false 被 bool() 转换为 True 而改变训练 token 序列。新增字符串和整数
开关失败路径测试；合法布尔及未指定的推导行为保留。

added_tokens_decoder 复用已有 token 内容解析，并检查整体 JSON 对象格式，
拒绝空或非法条目而非静默跳过。所有条目验证后才更新 backend 词表；
新增五项失败路径测试，确认合法 BOS 条目不会在失败前部分写入词表。

HF chat_template 命名集合支持 default 或唯一模板选择，列表格式检查名称
及重复项；多个模板无默认项明确拒绝。保留独立 Jinja 文件优先级，新增
真实 tokenizer 文件加载测试覆盖默认、唯一和歧义集合。

模板选择独立为 _select_chat_template 纯函数，构造只负责文件优先级和
编译；集合中选中的 null/缺失内容明确拒绝，顶层 null 仍表示未配置。
新增真实配置加载回归测试覆盖这两类缺失内容。

vocab.txt 保留空行对应的空 token 和行号 ID，避免跳过空行后出现 ID
空洞，使 token ID 超出 get_vocab_size；不对已有 token 重新编号。新增
真实 fallback 加载测试，检查空 token、后续 ID、词表大小和编码结果。

BPE merge 规则严格要求两个 token，拒绝原先被跳过的单字段行或被截断
的多字段行，错误附带文件与行号。新增两项真实 fallback 错误加载测试。

HF tokenizer 记录配置中的 unk_token/unk_id；renderer adapter 不再固定
返回 None，而是暴露实际 unknown token ID。BOS/EOS/UNK 配置与 ID 推导
复用统一循环，保留 token 参数身份；renderer 接口回归检查真实 UNK ID。

配置未声明 UNK 时，从后端 model.unk_token 推导名称及 ID，显式配置优先。
新增配置缺失与空配置的真实加载测试，检查未知文本编码与 unk_id 一致。

MultimodalTokenizer 直接遍历 token 字典，保留 collator 使用的 TOKEN_FIELDS，
为公开 token/ID 属性和构造返回值补齐类型声明。词表验证及数据消费者
接口保持一致，复用现有多模态加载与处理测试。

多模态 collator 在拼接前检查每条样本的 input_ids、labels、positions 和
可选 padding_mask 均为等长一维张量，且 padding_mask 使用 bool dtype。
原实现可能输出长度不一致的模型输入，或在 mask 运算时才报底层错误。
空 batch 明确拒绝；新增长度、维数、mask dtype 和空 batch 回归测试。
TOKEN_FIELDS 继续作为 tokenizer 与 collator 的共享字段契约，未增加包装。

MRoPE 媒体区域检测纳入 positions 归零的文档边界。两篇文档边界处连续
出现相同 image/video placeholder 时，分别消费各自的 grid，避免将跨文档
区域误计为单个媒体而拒绝合法打包输入。新增 image/video 两种回归用例，
核对每篇文档的三轴坐标独立从零开始。

OBELICS/CC12-WD 适配函数保留 sample 字段转换与 tokenizer 参数，将公共
处理选项直接转发给 process_multimodal_sample，删除重复的几何参数声明
及逐项转发。公共处理函数在 texts/images 后使用关键字参数；process_image
在 image 后使用关键字参数，并按几何、resize 策略、预算、归一化排列。
仓库调用方已使用关键字，无需改动；外部位置调用须迁移为关键字调用。

视觉 placeholder 展开直接使用字符串重复构造每个媒体区域，删除按视觉
token 数量增长的临时列表和手动索引；保留计数数量、非负约束与 EOS 行为。
复用现有 placeholder 和多模态处理测试核对输出及边界条件。

视频加载的元数据路径与无元数据 fallback 复用同一段帧数限制、时长与
均匀采样索引计算；仅解码策略保持分支，避免两套采样逻辑日后漂移。

视频接口在 path/video 后使用关键字参数，预算按 min/max 排列。
process_video 明确要求非空 THWC uint8 RGB 输入与正 patch/merge 几何，
拒绝浮点 0–255 输入被当作已缩放像素而产生错误归一化结果；新增布局、
dtype、负几何回归测试。仓库调用方已使用关键字；外部位置调用需要迁移。

文本 packing 入口统一为 build_concat_then_split_packing 与
build_first_fit_packing，删除前者冗余的 dataset 前缀及旧导出；同步 loader
装配、包 API、测试与设计文档。外部调用须使用新的函数名。

DocumentAwareConcatThenSplitIterator 的片段传递复用 TextSequence，删除
input/label/position/mask 四元素位置元组契约，调用方通过具名字段构建行。
保留切片与 checkpoint remainder 行为，复用嵌套 packing 和恢复测试。

loader_builder 的 recipe 解析返回值补齐配方、packer、collator 类型契约；
私有 PackingBuilder Protocol 描述绑定选项后的统一关键字调用签名。
文本路径只选择 packer，复用同一返回语句；保留多模态懒导入边界。

collators.py 定义 CollatorFactory Protocol，明确 context 关键字构造契约；
GrainDataLoader 与 loader_builder 共用该契约，替换接受任意参数的 Callable。
loader 使用 batch_collator 表示实例，避免同一变量从工厂变为实例。

精简 loss.py 模块与 chunked CE 文档，明确模型预处理负责 synthetic 标签
shift、packed 标签已对齐、chunked CE 内部 backward 与 detached 返回值。
保留 TP vocab 参数契约和 FSDP 每 chunk 通信代价说明，删除重复叙述。

chunked CE 在 head 执行前要求 labels 为一维 T 轴张量，避免标量标签的
底层 IndexError 或二维标签在模型执行后才报错。新增三个形状回归用例，
检查 head 未调用、既有 hidden/head 梯度内容与身份保持不变。

vocab-parallel CE 的 reduction 校验移到入口，拒绝非法选项后再读取进程组
或启动 collective，避免三次通信后才失败。新增三种非法选项测试，拦截
rank/world-size 查询与 all_reduce，确认失败路径不进入通信。

HF model 的 batch/mask 分片函数直接从 context_parallel/input_sharding.py
导入，移除实现层对包出口的间接依赖；包公共 API 保持不变。核对训练、
模型与并行实现的同类导入，模块对象导入（distributed/compatibility）保留。

TP token/label 与 padding-mask 分片共用私有 _shard_tp_buffers，统一长度
整除检查和 contiguous shard 调用，保留两个公开语义入口。新增 CPU 替身
测试核对两 rank 的对应切片、seq_dims 与无 load balancer，以及整除失败
不执行分片。该测试不替代真实 TP collective 数值验证。

PyTorch shard helper 与 HeadTailLoadBalancer 独立导入、按实际使用检查能力；
普通 TP/CP 连续分片不再因为缺少 headtail helper 被禁用。错误提示区分
输入分片与 headtail 依赖，删除未经当前版本证明的固定版本承诺。新增
两项 CPU 缺失能力回归测试。

input_sharding 说明明确包含 TP 输入分片，缺失 headtail helper 的错误提示
使用真实配置字段 parallel.context_parallel_load_balancer。新增两 rank
headtail 替身测试，核对 input/labels/positions/padding 的重排一致性及
两次 balancer 构造参数一致；未替代真实 CP kernel 验证。

HF preprocess_inputs 复用 TrainerBatch 类型，准确表达含媒体元数据的字典；
CP/TP mesh 标注 DeviceMesh，可选 attention mask 表达 BlockMask 或 dense
Tensor，补齐 mask 构造返回类型，修正原先将 flex mask 标成 Tensor 的契约。

模型预处理展平后检查 labels/positions/padding_mask 与 input token 数一致，
在 mask 构造或并行分片前拒绝错位的自定义 batch。新增三类字段错位测试，
确认失败时未构建 mask；二维 synthetic 输入仍沿用原先逐行 shift。

FQN optimizer checkpoint 恢复保留空参数组的当前选项。空组没有 FQN 可用于
序列化其配置，原恢复丢失 lr/weight_decay 等字段而使后续 step/scheduler
失败；新增 AdamW flat roundtrip 后执行 optimizer 与 scheduler 的回归测试。

FQN checkpoint 恢复核对组内所有参数的组配置，拒绝将不同保存选项的
参数合并后静默使用首个参数的 lr 等值。新增 float/tensor lr 两项真实
optimizer 回归测试，检查失败时当前参数组与状态未改变。

optimizer 嵌套状态 flatten 保留空字典叶子，避免 save 后丢失并在 load 时
误报缺失；新增根部/内部空字典 flat roundtrip 回归。两个仅本模块使用的
帮助函数改为 _flatten_state/_restore_nested_state，prefix/out 使用关键字。
该回归覆盖内存 flat 格式，不证明 DCP 后端可持久化任意扩展状态。

随后使用真实单进程 PyTorch DCP 保存/加载验证空字典状态，并将该路径
纳入 optimizer 回归测试。恢复前将 nested.count 改为 0，确认 DCP 从磁盘
还原为 3，同时保留两处空字典；不覆盖多 rank DCP 或任意自定义对象。

optimizer 恢复中的参数组重建集中到 _restore_param_group，负责空组配置、
缺失选项及组内一致性检查；unflatten_optim_state_dict 保留状态与参数 ID
映射编排。私有函数显式使用 param_ids/flat_state 关键字，复用现有恢复测试。

OptimizersContainer 的保存/恢复直接调用 expert_shard_map，删除仅转发的
_ep_shards 包装，使用构造时已建立的 model_parts 契约。日志选项使用固定
顺序 tuple，避免 set 遍历使不同进程或运行的配置日志顺序变化。

LambdaLR 恢复对 Tensor learning rate 使用 fill_，与 PyTorch 正常 step
一致，保留外部引用的 lr Tensor 身份；float lr 仍直接赋值。新增真实
AdamW scheduler 恢复后继续 step 的数值与身份回归测试。

核对 151 个源码 Python 模块，均有文件审查清单条目。decoder FSDP 入口在
model/dp_mesh 后将 dtype、策略与开关设为关键字参数，明确原地变换返回
None；同步 placement 测试调用，外部位置选项需迁移。保留 clip_grad_norm_
的 PyTorch 风格位置参数契约，不为外观统一改变标准接口。

decoder FSDP 公共入口拒绝未知 implementation，避免直接调用时将拼错的
custom 选项静默降为 native。新增三种非法选项测试，在任何模型访问或
sharding 前失败；配置构造校验与直接调用契约一致。

FSDP prefetch 在任何 setter 调用前检查所有 unit 的前向/后向接口，避免
后续 unit 缺少方法而留下部分修改的计划；新增合法 unit 后接普通 Module
的失败路径回归，确认前面的 setter 未被调用。

decoder 与 pattern FSDP 共用 get_fsdp_strategy 校验后端，删除 decoder
重复的合法值判断和 custom 导入选择；策略返回类型补齐为 NativeFSDPStrategy
（custom 是其子类）。保留 native fully_shard 调用与现有 placement 验证。

apply_fsdp 直接读取 ParallelConfig 的 pattern、prefetch 和 implementation
字段，删除 getattr 的旧默认兜底，使入口与已验证的完整配置契约一致。

TP 入口直接读取 ParallelConfig 的 colwise/rowwise/sequence pattern 字段，
删除旧默认兜底。核对其余 parallel 配置读取；HF attention head 配置的
可选字段仍使用兼容读取，以支持不同模型布局。

TP 配置 pattern 合并移入 resolve_tp_targets，让解析与应用统一消费 cfg
覆盖项并复用相同优先级；apply_tp 仅编排验证和变换。新增直接解析器测试，
确认 cfg rowwise pattern 优先于传入的同名 colwise 计划。

TP 解析统一建立 raw_plan 后合并配置覆盖，不再改写 plan 参数再二次选择
来源；模型计划只读一次的回归同时覆盖有/无 rowwise 配置 pattern，并
核对目标 style 与优先级。

Column/RowParallelLinear 的输出 reshape 使用已知 feature 维，避免空 token
输入时 0 与 -1 同时出现导致形状推导失败。新增两种 projection × 二维/
单 batch 空输入的 CPU 前后向回归；通信 primitive 使用本地 linear 替身，
该测试只验证 projection 的形状恢复和 autograd，不验证空 collective。

fused TP projection 在通信前要求 [T,K] 或 [1,T,K] 输入，拒绝原先会误接收
的四维多 batch 布局及一维/标量输入，防止 flatten 后的 rank-order gather
混排。新增两类 projection × 三种非法维数的失败路径测试，确认无通信。

TP functional_collectives 仅选择完整、可调用的显式 autograd API 配对，
删除可能断开梯度的非 autograd 回退；缺少能力明确报安装兼容 torch 错误。
新增 single/tensor 配对和 legacy/不完整能力测试，验证版本适配选择契约。

### TP CPU/Gloo 分布式数值复核

使用两个真实进程运行以下脚本，均通过；启动参数为
`PYTHONPATH=. torchrun --master-addr=127.0.0.1 --master-port=<port> --nproc_per_node=2`，
后接相应脚本路径。默认地址在本机解析为不可用的 rendezvous 地址，显式使用
IPv4 loopback 后成功，无需改动通信实现。

| 脚本（位于 `tests/integration_tests/`） | 验证范围 | 最大绝对误差 |
| --- | --- | --- |
| `tp_equivalence.py` | Llama TP logits、分片权重梯度 | 2.980e-08、1.937e-07 |
| `tp_replicated_grad_equivalence.py` | 三步 SGD 的局部 loss、复制参数归约梯度 | 0、2.384e-06 |
| `shared_expert_tp_equivalence.py` | DeepSeek 共享专家 TP 输出、attention/共享专家梯度 | 0、3.262e-07 |

复制参数归约前与参考梯度相差 5.887，确认测试能够识别漏归约。
这些结果验证 CPU/Gloo functional collective 路径，不覆盖 CUDA symmetric-memory
融合实现、NPU 后端或 FSDP/PP/CP/EP 全组合矩阵。

### 训练预处理参数契约简化

`Trainer.preprocess` 与 `trainer/runtime/batch.py::preprocess` 直接接收
`Batch | TrainerBatch`，删除只用于提取 batch 的外层字典参数。训练累积在调用时
提取原始 batch，PP 和验证不再临时构造 `{"batch": batch}`；设备搬运、
token-count 元数据过滤和模型输入生成仍共用同一实现。同步元数据保留回归。
外部调用方应将 `trainer.preprocess({"batch": batch})` 改为
`trainer.preprocess(batch)`；累积统计结构及 trainer 文件布局保持现有契约。

PP 的 `split_pipeline_microbatches` 改为接收 batch 和关键字参数
`num_microbatches`，不再依赖 Trainer 或完整配置，调用方在边界提取配置。
同时修复字典拆分后继承整批 `num_valid_tokens` 的错误：有计数元数据时，
按每个子 batch 的 labels 重算，保留原 batch。新增非均匀有效 token 分布
回归（整批 3，子批分别 1/2），并复用普通 Batch、packed 拒绝与不完整行测试。

PP loss 的 `scalar_loss_fn` 明确声明关键字分母、TP group 和 vocabulary size；
`make_schedule_loss_fn` 直接捕获词表参数，删除只作转发的 `vocab_kwargs` 字典。
显式传入分母时不再提前读取 schedule 默认属性，修复原 `dict.get` 默认表达式
立即求值的问题。新增无默认属性的 schedule 回归，核对平均 CE 数值及梯度；
保留现有显式分母覆盖、动态默认分母和完整词表路径回归。

本轮运行 `tests/integration_tests/pp_equivalence.py` 的双进程 CPU/Gloo
1F1B 路径（显式 127.0.0.1 rendezvous）：四步训练、每步四个 microbatch，
loss 和梯度范数相对完整模型参考的最大绝对误差 4.768e-07，所有 rank 通过。
该证据不覆盖本轮尚未执行的 interleaved/V-block schedule 或 GPU/NPU 路径。

随后补跑双进程 CPU/Gloo 的 Interleaved1F1B 和 ZBVZeroBubble，同样四步训练、
每步四个 microbatch，所有 rank 通过；最大绝对误差分别为 0 和 4.768e-07。
因此以上三种 schedule 均有本轮 CPU 数值证据，GPU/NPU 仍未覆盖。

stage 分配的 input/output weight 改为关键字参数，同步唯一位置调用方，
避免权重传反；保留 num_stages/num_layers 的主要位置参数。删除前置
`num_stages <= num_effective_layers` 已排除的零均分分支，复用现有权重、
stage 数量、完整 layer ownership 回归；修正 PP 集成脚本命令的旧目录路径。

checkpoint 来源解析返回私有只读 `CheckpointLoadSource`，取代路径和三个
布尔值的位置元组；load 按字段访问路径、model-only、HF 与量化格式标志，
不再依赖布尔值排列。描述对象使用 keyword-only 构造，保留恢复优先级和
seed checkpoint 语义。BaseCheckpointManager 构造参数补齐 CheckpointConfig
类型，仅在 TYPE_CHECKING 导入，避免加重运行时配置依赖；复用现有恢复回归。

CLI 将 Trainer 导入延后到配置解析和 seed-checkpoint 前置校验之后，纯配置
解析与 `--help` 不再加载训练引擎。入口生命周期测试从定义模块替换 Trainer，
新增独立子进程测试：屏蔽 trainer 模块后仍能解析 `--steps 1`，防止测试进程
已缓存的模块掩盖依赖。训练、清理与进程组所有权保持现有入口职责。

模型几何同步集中到 `ModelConfig.update_architecture`：聚合配置的
`auto_fill_model` 和运行时 assembly 不再各自维护六个字段的复制列表。
调用方仍负责 HF 来源解析与 composite text_config 解包；ModelConfig 只拥有
可同步字段，缺失字段保留显式值，不引入 HF 运行时依赖。复用本地/组合模型
配置、offline overrides 和真实训练装配的现有回归，保持字段更新行为。

synthetic source 的公共 `iter_from` 在生成任何 batch 前拒绝负数、bool 与
非整数索引，补齐 loader 恢复路径已有的非负游标契约。loader 先构造并校验
source 的形状，再计算 DP 行数，避免非法 batch_size 在 modulo 中先触发
无关 TypeError。新增非法索引四例、非法 batch_size 三例，复用确定性与恢复回归。

HF 模型类解析要求候选为 PreTrainedModel 子类，拒绝将 transformers 的
同名配置类或普通属性当作模型；非法 architectures 声明继续尝试 model_type
映射。新增 AutoConfig 与版本字符串候选回归，确认均解析为 LlamaForCausalLM。
补齐模型工厂的 LLMTunerConfig 入参和模型类返回类型，配置类型只作静态导入。

GC 的 debug 开关拒绝非 bool 值，防止字符串 `"false"` 等真值误启用逐步
collection 和 tensor-cycle 检测；校验先于 gc.disable，新增四个失败路径
确认无全局状态变更。debug 与 collect 的 generation 改为关键字参数，
保留主要频率和日志 reason 的位置参数；现有调用方均已使用具名选项。

AC 入口按已枚举的 block 数量记录日志，不再对任意 nn.Module layers 调用
len；修复合法普通 Module 容器包装完毕后在日志处抛 TypeError 的问题。
新增真实 checkpoint wrapper 的普通容器前后向回归，核对参数身份与梯度，
保留 ModuleList/ModuleDict 的现有布局和包装流程。

TextCollator 在任何张量分配前拒绝空 rows 列表，替代 torch.cat 空列表的底层
错误；含一个空 TextSequence 的 batch 仍合法，输出固定 token 预算的全 padding
张量。新增分配前拒绝与全 padding 语义回归，区分没有行和行内没有 token。

loader 配方解析结果改为私有只读 `_ResolvedRecipe`，按 dataset、build_packing、
collator 字段表达共享装配契约，删除文本/多模态路径的三元素位置解包。
PipelineTextCollator 的选择在共同装配处完成；保持可选媒体依赖懒加载，
不改变图构建顺序或 packing/iteration policy。复用数据流水线与多模态回归。

复核 packing 的 parent 游标、remainder offset 和 finished 校验，保留其现有
状态恢复语义。selective AC 的 mm_recompute_shapes 将 base_fqn/fqns 改为
关键字参数，匹配集合使用 Sequence 类型，明确只读消费契约；同步所有调用，
已有 projection shape 回归改用 tuple，保留错误匹配测试的 list 输入。

compile 删除本地 symmetric-memory 通信组名称缓存，直接使用 PyTorch
enable_symm_mem_for_group 的幂等注册；已检查安装版本源码，其自身按 group_name
维护注册状态并在重复请求时返回。避免两份缓存维护同一职责；保留当前版本的
FutureWarning 屏蔽和能力检查。不宣称修复 PyTorch 自身跨进程组生命周期缓存。
compile TP mesh 与共享 parallel stage runner 的 dense mesh 补齐 DeviceMesh
可选类型；复用既有 compile 配置与非法 mesh 回归，CUDA 路径未运行。

scheduler 容器在构造 LambdaLR 前拒绝重复 optimizer 对象，避免同一参数组
被两套 scheduler 初始化和重复推进。新增真实 SGD 回归：重复对象失败时 lr
保持 0.2，initial_lr 尚未插入；不同 optimizer 的现有 PP 多容器路径保持支持。

补充真实双进程 CPU/Gloo PP checkpoint 证据：运行
`tests/integration_tests/pp_checkpoint_equivalence.py` 的 full/resume 两阶段，
四步训练在 step 2 写入 DCP，然后全新进程从 step 2 恢复并继续两步。
resume 设置 `LLMTUNER_TEST_RESUME_LR=0.003`，恢复后的 loss、模型与优化器状态
均通过脚本对比，loss 最大绝对误差 0。覆盖 FQN 优化器恢复、scheduler 基准
学习率恢复、trainer 计数和 synthetic loader 游标；该次未启用 EMA 或 packed
语料，也未覆盖 GPU/NPU。修正脚本说明的历史游标 replay 描述为绝对索引生成。

随后以 `LLMTUNER_TEST_EMA=dynamic` 在独立目录重跑 PP checkpoint 的 full/resume，
resume 同样改用初始 lr=0.003；真实双进程 Gloo 恢复通过，loss 最大误差 0，
最终动态 EMA 状态与连续训练逐 tensor 相等。增强集成脚本的 EMA 验证：先比较
完整 key 集合，再比较值，避免实际状态缺少部分 keys 时原循环漏检；本次通过
使用增强后的脚本。packed 语料与 GPU/NPU 仍未在本轮验证。

再以 `--packed` 和 `LLMTUNER_TEST_EMA=dynamic` 在独立目录运行两阶段 PP
checkpoint 集成，resume 初始化 lr=0.003。双进程 CPU/Gloo 全部通过，恢复后
loss 最大绝对误差 0，脚本的模型、优化器、token 计数与完整 EMA 状态检查通过。
覆盖本地 JSONL、Grain loader cursor、独立 packed pipeline microbatch 的真实
装配与恢复。CPU attention 使用脚本明确提供的 dense-mask SDPA 替身，故不将
此结果视为生产 CUDA flex attention 验证；GPU/NPU 仍未执行。

再次核对当前 151 个 llmtuner Python 源文件，审查文档均有对应路径引用。
路径清单覆盖与分布式数值证据分别记录，未据此宣称所有硬件/组合已验证。

运行 `validation_reduction_equivalence.py` 四进程 CPU/Gloo，DP×CP 和 DP×TP
验证 loss 与完整 batch 参考一致；覆盖不均匀有效 token、DP-only 分母与序列
轴 loss 归约，并检查 eval/no-grad、loader 关闭、冻结子模块模式恢复和训练
token 计数不变。模型使用显式局部投影与序列切片，该脚本不验证实际 attention
或 TP/FSDP 模型变换。validation token 统计的 dp_mesh 改为关键字参数，明确
归约轴；同步唯一内部调用，不改变归约数学。

辅助损失 step hook 注册时将 model_parts 固定为 tuple，避免外部列表编辑
改变已注册统计范围。新增真实 SGD step 回归，清空调用方列表后原 loss
累积仍正确汇总并清零。运行双进程 CPU/Gloo
`moe_aux_loss_grad_equivalence.py`：CP token-partial 前向归约与 identity backward
相对完整 token 参考通过，router 梯度最大绝对误差 1.192e-07；该脚本验证
辅助损失梯度，不覆盖 EP dispatcher 或 attention 内核。

LocalTokenDispatcher 的 routing scores/expert IDs 展平改用 reshape，支持
合法的非连续切片，不再要求隐式 contiguous 布局。新增连续/非连续输入对照，
核对 dispatch→非线性专家计算→combine 输出、token 与 routing score 梯度；
本地 dispatcher 专项 16 项通过。该证据不覆盖跨 rank all-to-all 后端。

继续运行 `ep_equivalence.py` 的双进程 CPU/Gloo all-to-all 专家前向对比：
8 experts、top-k=2、每 rank 40 tokens，跨 rank 最大相对误差 3.271e-07，
通过 1e-5 阈值。修复集成脚本只断言 rank 0 本地误差的漏检：现在对全 rank
MAX 归约结果统一断言，且先销毁 process group，避免某一 rank 提前异常导致
另一 rank 卡在最后 collective。同步旧测试命令路径。该次不验证 EP backward
或加速器专用 dispatcher。

EP backward 补证：先运行原 `ep_domino_equivalence.py`，再增强 routing scores
和 expert IDs 为经断言的非连续布局并重新执行。两轮真实双进程 CPU/Gloo
均通过普通/Domino × 零接收开关 × checkpoint replay 开关的八种组合。
每 rank token 数分别为 5/7，比较输出、输入梯度、routing score 梯度和本地
expert 参数梯度，rtol/atol=1e-9；replay 组合确认 forward 实际执行两次。
增强后的脚本验证 reshape 修复经过 all-to-all 与重计算仍正确，未覆盖 CUDA/NPU
MC2 或 torchao 内核。

继续核对专家计算与 dispatcher 边界：共用 `_local_reorder` 现在检查 token
输入为二维，routing scores 和 expert IDs 均为 `(tokens, top_k)`，避免实际
路由宽度与配置不一致时按错误 top_k 静默映射 token。新增六组本地及
all-to-all 单 rank 回归，覆盖路由宽度、token 数和额外 score 列不匹配。
完整 CPU 测试 2087 passed、2 skipped；Ruff、format、compileall、diff 检查及
排除受保护 examples/native_alltoall.py 的 pre-commit 全部通过。重新运行
双进程 CPU/Gloo ep_domino_equivalence.py，普通/Domino、零接收与重计算的
八种组合全部通过前向和梯度等价检查。未验证 GPU/NPU 内核。

EP 后端契约继续收敛：删除 token_dispatch.py 重复的后端列表，配置与
model_replacement.py 直接调用统一使用 expert_parallel_config.py 的
validate_expert_dispatcher；EP 度数限制仍由配置校验负责。直接调用选择 MC2
现在准确报告缺少 Ascend 融合算子，不再误报为 CUDA/DeepEP 依赖。四组回归
核对非法/未实现后端在配置及直接调用路径的异常类型与文本完全一致。
删除无实现、无覆盖、无外部引用的 init_buffer 空钩子；保留 mesh 绑定入口。
修正 torchao padding 可用于 per-expert loop、offsets 为累计分段终点等说明。
专项 251 项通过；完整 CPU 测试 2091 passed、2 skipped；Ruff、format、
compileall、diff 检查及排除受保护示例的 pre-commit 通过。本轮未改变通信
算法，也未运行 GPU/NPU 后端。

将 hf_inspection.py 内无训练调用方的 probe_expert_layouts 开发调查工具移至
examples/inspect_hf_experts.py，移除生产模块对应 sys/logger 依赖和旧函数。
新脚本提供 main/CLI，可选模型家族，使用两层模型观察 DeepSeek/GLM4 的后层
专家，并限制 vocab_size=128；任一家族无法调查时返回非零退出码。
实际运行默认 Qwen3MoE、OLMoE、Mixtral、GPT-OSS 以及额外 DeepSeek-V2/V3、
GLM4MoE 七个家族成功。该输出是形状与 activation 调查，不是数值等价证据。
探测模块删除捕获后原样 raise 的包装，修正 F 维度注释与校验实现的矛盾。
非法模型家族实测退出码为 1；EP 替换和 TP MoE 专项 65 项通过，最终完整
CPU 测试 2091 passed、2 skipped。Ruff、format、compileall、diff 检查与
排除受保护示例的 pre-commit 通过；未增加新的硬件数值验证。

HF 模型构建职责整理：将 factory.py 的 materialize_meta_model 完整搬入
models/hf/materialization.py。factory.py 专注 HF config 构建、text config
选择与类解析；新模块负责并行变换后存储分配、非持久 RoPE buffer 重建和
共享参数/buffer 引用恢复。训练装配及测试直接导入新定义模块，不保留旧入口
转发。模块说明、开发指南、上游映射及清单同步更新。模型与 trainer 专项
151 项通过，完整 CPU 测试 2091 passed、2 skipped；Ruff、format、compileall、
diff 检查与排除受保护示例的 pre-commit 全部通过。搬运未改变物化算法，
本轮未新增 GPU/NPU/FSDP 实测。

随机数接口收敛与失败路径修复：删除仅供测试调用的 set_seed 薄转发，统一
使用 seed_everything(seed, *, deterministic, detect_anomaly=False)。调用前
检查 seed 为非 bool 整数且位于 PyTorch 的 [-2**63, 2**64-1] 范围，避免
Python/NumPy 已重新播种后 Torch 才因范围非法失败。新增五组非法输入回归，
直接比较三套真实 RNG 状态和 PYTHONHASHSEED 均未变化；补充最小合法 seed
的重复性验证。训练/PP seed/工具专项 152 项通过；完整 CPU 测试 2096 passed、
2 skipped，Ruff、format、compileall、diff 检查与排除受保护示例的 pre-commit
通过。代码/测试/示例无 set_seed 旧名引用，历史问题记录保留原名；本轮
未执行设备 RNG 或多进程数值验证。

继续审查数据迭代职责，确认 batch_iterator 无生产调用方，五个集成脚本只
用它包装本身无限的 RandomTokenSource。删除该重复层及其两项有限源封装
测试，所有调用改为标准 iter(source)；保留随机源确定性、不同 step 与
checkpoint 游标覆盖，并移除测试 _source 未使用的位置参数。数据耗尽异常
仍由 trainer/runtime/batch.py 统一处理，不改变生产 loader 行为。
完整 CPU 测试 2094 passed、2 skipped（较前轮减少的两项对应已删除接口）；
双进程 CPU/Gloo pp_equivalence.py 默认 1F1B，4 steps × 4 microbatches，
loss/梯度范数最大绝对差 4.768e-07，失败 rank 数 0。Ruff、format、compileall、
diff 检查和排除受保护示例的 pre-commit 通过；代码/测试/示例无旧函数引用。
本轮未运行其余 PP schedule、GPU/NPU 或其他修改脚本对应的并行组合。

checkpoint filesystem 路径协议修复：is_remote、resolve 与 remote join 改用
os.fspath，避免自定义 os.PathLike 的显示字符串被误判为真实路径；类型明确
为 os.PathLike[str]。新增本地和真实 fsspec memory 后端回归，覆盖 URI 判断、
exists/isdir/isfile/listdir/join、删除及重复删除；不涉及外部对象存储服务。
将通用 utils 测试内三项 filesystem 测试归并到 checkpointer/test_filesystem.py，
并增强本地分支测试，确保不会调用远端解析。专项 46 项通过；最终完整 CPU
测试 2096 passed、2 skipped，Ruff、format、compileall、diff 检查与排除受保护
示例的 pre-commit 通过。未验证 S3/GCS 驱动、网络权限或远端 DCP 保存恢复。

优化器相关配置按责任重组：optimizer_config.py 保留 OptimizerConfig 与
ParamGroupConfig；LRSchedulerConfig 搬入 lr_scheduler_config.py，EMAConfig
及共享 validate_ema_schedule 搬入 ema_config.py。CLI、根配置组合、训练配置、
运行时类型引用与测试均直接导入定义模块，config 包继续统一导出公共类。
原配置/优化器专项 331 项通过；补充 12 组调度比例类型和非有限数回归，
拒绝 bool/None/字符串等输入，统一抛 ConfigError，避免 bool 被静默解释为
全程 decay 或非数值输入泄露 TypeError。最终完整 CPU 测试 2108 passed、
2 skipped；python -m llmtuner --help 成功且保留调度参数；Ruff、format、
compileall、diff 检查和排除受保护示例的 pre-commit 通过。开发指南与文件
清单同步更新，本轮未改变调度曲线或 EMA 更新数学，未新增硬件数值验证。

继续核对 EMA 配置与直接构造入口：共享 validate_ema_schedule 现在拒绝
decay/half_life_fraction 的 bool 和非数值输入，合并半衰期的重复分支并统一
ConfigError。decay 的 [0,1) 范围检查本身也拒绝 NaN/Inf，无需重复 isfinite。
扩充七组回归，分别从 EMAConfig 和 EMA 构造调用，并通过阻止
EMAParamOptimizer 分配来确认失败发生在平均权重分配前。EMA 专项 57 项
通过，完整 CPU 测试 2115 passed、2 skipped；Ruff、format、compileall、diff
检查与排除受保护示例的 pre-commit 通过。本轮未改变合法 EMA 参数的更新
数学，未执行新的分布式或 GPU/NPU 验证。

优化器构造流程重组：_build_param_groups 将原两份平行字典合并为按 optimizer
名称索引的 _OptimizerGroups，归集构造参数组和对应日志 patterns。先构建所有
本地 model part 的分组计划，收集参数并验证完整归属，再构造内层优化器，
避免后续 part 配置失败时已经构造部分优化器。两项回归通过替换 AdamW
构造器确认重复 part、后续 pattern 不匹配都在任何构造调用前拒绝。
原优化器专项 154 项通过，新增后 container 专项 25 项通过；最终完整 CPU
测试 2117 passed、2 skipped。双进程 CPU/Gloo Interleaved1F1B（每 rank 两个
本地 part），4 steps × 4 microbatches，loss/梯度范数最大绝对差 0，失败
rank 数 0。Ruff、format、compileall、diff 与排除受保护示例的 pre-commit
通过；本轮未验证 GPU/NPU 优化器内核。

继续检查 optimizer/state_dict.py 的嵌套状态协议：展开后的同名键现在
明确拒绝，避免 cache.value 与 cache={value: ...} 在扁平存储中静默覆盖。
恢复嵌套路径时检查父路径是否已作为 checkpoint 叶节点存在，同时拒绝
用叶值覆盖已建立的分支。新增六组回归覆盖正反插入顺序、标量父节点和
空字典父节点；确认失败不修改 live optimizer state 或 checkpoint 空字典。
状态专项 26 项通过（包含既有空字典 DCP roundtrip）；完整 CPU 测试
2123 passed、2 skipped；Ruff、format、compileall、diff 检查和排除受保护
示例的 pre-commit 通过。有效 checkpoint 的键格式保持不变，未新增设备
或多进程 checkpoint 验证。

公共 QKV 拆头修复：_split 不再以固定 flatten(1,2) 合并维度，改为
flatten(-3,-2) 只合并 KV group 与组内 head，保留任意前导 batch/token 轴。
原二维 token 输入布局不变，forward 文档明确 Q 使用 n_heads、K/V 使用
n_kv_heads。新增五种形状（单 token、二维 token、batch、多级 batch、空序列）
与三个独立逻辑投影 GEMM 比较前向及 input/weight/bias 梯度，并检查连续布局。
QKV 与 TP 投影专项 79 项通过，完整 CPU 测试 2128 passed、2 skipped；Ruff、
format、compileall、diff 与排除受保护示例的 pre-commit 通过。该证据验证
公共投影逻辑，不代表 TP collective 对任意 batch 维度或 GPU attention 已验证。

公共线性构件按实际职责整理：生产使用的 RouterGateLinearFunction 与
RouterGateLinear 搬入 models/common/moe/gate.py；routing 和测试导入同步，
models.common 懒导出继续提供 RouterGateLinear。删除旧 common/linear.py 和
只有测试消费者的 PartialBiasRowwiseLinear 及两项对应测试，不保留转发壳。
现有 TP 入口仍明确拒绝带 bias 的 decoder 投影，不将删除旧类误写成支持
上游的 rowwise bias 路径；上游映射、模块说明、开发指南与清单已同步。
相关专项 143 项通过；完整 CPU 测试 2126 passed、2 skipped（删除两项旧
接口测试）。双进程 CPU/Gloo ep_equivalence.py 包含真实 router gate 调用，
全 rank 最大相对误差 3.271e-07；ep_domino_equivalence.py 的八种专家
dispatch/零接收/recompute 组合也通过前向和梯度检查，后者不覆盖 router。
Ruff、format、compileall、diff 与排除受保护示例的 pre-commit 通过；未验证
CUDA BF16 gate 特殊路径或 GPU/NPU 并行内核。

公共 RoPE 配置归位：RoPEConfig 搬入 config/rope_config.py，构造字段均为
关键字参数，由 config 包导出，移除 models.common 的配置导出。增加偶数
正 dim、正缓存长度、有限正 theta、scaling 枚举校验；Llama 检查频率边界
顺序与原始长度，YaRN 检查 factor>=1、beta、原始长度与 truncate 类型，
实际缩放时检查 theta>1，避免 log 分母为零或无效缩放被静默忽略。
未启用模式的字段不参与校验。yarn_inv_freq 的 base/factor/beta/长度/truncate
改为关键字参数，生产与测试调用同步。新增 18 组非法配置和两个普通 RoPE
忽略未使用 scaling 字段的回归；RoPE 专项 47 项通过，完整 CPU 测试
2146 passed、2 skipped；Ruff、format、compileall、diff 与排除受保护示例
的 pre-commit 通过。该公共构件不替换 HF wrapper 自带 rotary_emb，未新增
设备或分布式 attention 验证。

Attention mask 入口契约整理：create_attention_mask 将不透明的 *args/**kwargs
改为显式 mask modifier 和关键字维度、设备、block size、layout 参数；HF
wrapper 与 CP causal mask 调用同步。保持 flex 延迟加载、NPU eager 路径和
旧版 PyTorch 不支持 separate_full_blocks 时的兼容处理。新增四组替身测试
验证 CPU/NPU builder 选择及新旧 layout 参数转发；这些测试不运行 NPU 内核。
mask 专项 33 项通过，完整 CPU 测试 2150 passed、2 skipped；Ruff、format、
compileall、diff 与排除受保护示例的 pre-commit 通过。

Checkpoint canonical 目录复核：step 正则中的 \d 改为 [0-9]，拒绝混合
ASCII 与 Unicode 数字的目录名，避免发现 step-1٢ 后解析为 12、再加载
不存在的 step-12。新增六组正常、前导零及 Unicode 数字回归；checkpointer
专项 56 项通过，Ruff、format、compileall 和 diff 检查通过。

DocumentAwareConcatThenSplitIterator 恢复流程改为先准备后提交：验证重放样本
和 remainder offset 成功后才更新本地 remainder/finished 状态；重放失败
则使用恢复前父状态回退游标。新增两项回归覆盖越界 offset 与缺失 remainder，
确认异常后状态和后续所有 token 字段与参考流相同。回退依赖父迭代器的
set_state 能正常恢复此前有效状态。数据流水线专项 239 项通过；完整 CPU
测试 2185 passed、2 skipped；Ruff、format、compileall、diff 与排除受保护
示例的 pre-commit 通过。

HF streaming cursor 生命周期复核：HuggingFaceCursorIterator.__next__ 增加
Grain 的 _assert_not_closed 检查，与 document packing iterator 保持一致。
新增有限/重复源两项回归，确认 close 可重复调用，关闭后 next 抛错且不消费
后续样本。完整 CPU 测试 2187 passed、2 skipped；Ruff、format、compileall、
diff 与排除受保护示例的 pre-commit 通过。测试使用本地源替身，不访问 Hub。

PP 输入契约整理：prepare_pipeline_inputs 返回关键字构造的 PipelineInputs
dataclass（args/kwargs/targets），替代三个嵌套列表组成的匿名 tuple。
训练与验证显式访问字段，保留底层 schedule 参数和 stage ownership 语义，
trainer.py 未拆分。完整 CPU 测试 2187 passed、2 skipped；真实双进程
CPU/Gloo 1F1B（4 steps × 4 microbatches）loss/梯度范数最大绝对差
4.768e-07，失败 rank 数 0。Ruff、format、compileall、diff 与排除受保护
示例的 pre-commit 通过；本轮未验证 GPU/NPU schedule。

HF attention 注册接口复核：register_attention_backend 使用 Transformers
AttentionInterface.register 公共接口，移除对 _global_mapping 私有字典的
直接写入。实际调用注册接口后，从 AttentionInterface 实例解析 backend，
确认取得原 flex_attention_adapter。HF wrapper 专项 52 项通过；完整 CPU
测试 2187 passed、2 skipped；Ruff、format、compileall、diff 与排除受保护
示例的 pre-commit 通过。注册验证不执行 flex attention GPU 内核。

lr_scheduler.py 再次复核：LambdaLR 绑定各 inner optimizer，恢复先计算全部
base_lrs/last_lrs 再提交，张量 learning rate 保持原对象；指标名称按优化器
类型、同类型实例和参数组消歧。本轮未改调度数值或 checkpoint 格式，现有
34 项专项通过。清单统计同步核对源码路径集合与表格集合，均为 155 个，
无遗漏、重复或失效条目；路径集合相等不作为实现正确性的证明。

MLA FLOPs 修复：attention_param_term 以 Q/KV 低秩几何识别 MLA，而非仅
根据 q_lora_rank。q_lora_rank=None 时计入直接 Q 投影，K/V 继续使用
kv_a/kv_b 低秩公式，避免将 DeepSeek 的合法混合投影误算成普通 QKV。
两项回归构造实际 DeepseekV3Attention，分别覆盖直接 Q 与低秩 Q，比较
统计项与全部 nn.Linear 权重元素数的两倍。FLOPs 专项 22 项通过；完整
CPU 测试 2189 passed、2 skipped；Ruff、format、compileall、diff 与排除
受保护示例的 pre-commit 通过。本修改仅影响 MFU/FLOPs 估计，不执行新的
attention 数值验证。

HF materialization 预扫描合并：一次 model.modules 遍历收集参数/buffer
别名、待重建的非持久 meta buffer 和需要保留的真实 buffer，替代三次模块
遍历和临时展开的参数/buffer tuple；增加局部容器类型，保留未知 buffer
在 to_empty 前失败、RoPE 重建和共享引用恢复的既定顺序。完整 CPU 测试
2189 passed、2 skipped（含现有 materialization 回归）；Ruff、format、
compileall、diff 与排除受保护示例的 pre-commit 通过。本轮未新增分布式
DTensor materialization 或 GPU/NPU 初始化验证。

CLI 视图薄封装合并：删除仅调用 cli_view 的 tuple 推导函数 cli_groups，
解析器直接按 PARSER_GROUPS 构造视图列表，省去 tuple 到 list 的重复构造。
生产入口、配置测试和上游映射同步，不保留旧函数别名；解析顺序、隐藏字段
与默认值语义保留。配置专项 212 项通过，python -m llmtuner --help 成功；
完整 CPU 测试 2189 passed、2 skipped，Ruff、format、compileall、diff 与
排除受保护示例的 pre-commit 通过。

多模态职责拆分：MRoPE 坐标构造及 text/grid helpers 从 collator.py 移入
datasets/multimodal/mrope.py。build_mrope_positions 显式接收 merge size
和 patch order；collator 专注文本/媒体拼接、padding 与 patch 装配。删除
旧 _build_mrope_positions 方法，生产和六处直接测试调用同步，不保留转发壳。
多模态数据与构建专项 125 项通过；完整 CPU 测试 2189 passed、2 skipped；
Ruff、format、compileall、diff 与排除受保护示例的 pre-commit 通过。
源码与清单均为 156 个不同 Python 路径。此搬运保留既有坐标算法，本轮
没有新增设备或 VLM 训练数值验证。

独立 MRoPE 入口补充边界契约：复用 require_positive_integer 校验 merge size，
检查 tokens 为一维、positions 形状一致；无媒体的空输入返回 (0,3) 坐标，
避免空列表 cat 失败，空输入带媒体 grid 则明确拒绝。新增五组非法 merge
size 与空序列回归，多模态数据专项 125 项通过；Ruff、compileall 和 diff
检查通过。本轮未重跑完整 CPU 套件或设备内核。

MRoPE grid 几何校验：图片与视频 grid 统一要求 (num_media,3) 形状、
int32/int64 dtype 和正尺寸，避免浮点尺寸在 int(item()) 时静默截断，
以及非法维度进入坐标展开。新增十二项图片/视频错误形状、float/bool、
零/负尺寸回归，多模态数据专项 137 项通过；Ruff、format、compileall 与
diff 检查通过。校验运行于既定 CPU 数据处理路径，未新增设备内核验证。

Vision placeholder 数量契约统一：insert_vision_placeholders 在字符串展开
前明确拒绝 bool/float/string/None，数量须为非负整数，保留零数量语义。
扩展四项非法类型回归。包含近期全部 MRoPE 边界测试的完整 CPU 套件
2211 passed、2 skipped；Ruff、format、compileall、diff 与排除受保护
示例的 pre-commit 通过。本轮未新增媒体解码或 VLM 内核运行验证。

HF 状态适配器复核键冲突、PP 全 rank 覆盖检查及 tied lm_head 补全，删除
未使用的 logger 导入和实例。补齐此前 loss 整数契约的十六项直接回归，
覆盖分片三个坐标与 seq_len 的 bool/float/string/None，并确认失败不修改
labels。loss 与状态适配专项 40 项通过；Ruff、format、compileall、diff 和
受影响文件 pre-commit 通过。本轮未重跑完整套件或多进程导出。

补齐 ReduceScatterLinear 构造校验回归：十六项参数组合通过 torch.empty
失败哨兵确认非法维度和非 bool bias 在分配任何参数前被拒绝。TP 投影
专项 36 项通过；包含上轮 loss 十六项回归的完整 CPU 测试 2243 passed、
2 skipped。Ruff、format、compileall、diff 和受影响测试文件 pre-commit
通过。本轮未修改生产数值路径，未新增 collective 硬件验证。

数据包导入边界优化：datasets/__init__.py 使用已有 resolve_export 懒加载
公开名称，保留原 __all__，packing 导出直接指向定义模块。MRoPE 的单字段
整数校验改为入口内检查，消除仅为校验导入 contracts 的 Grain/DCP 依赖。
新增独立进程测试阻止 Grain、HF datasets、torchvision 与 checkpoint 模块
导入，成功导入 MRoPE 并验证跨文档文本坐标。完整 CPU 测试 2244 passed、
2 skipped；Ruff、format、compileall、diff 与受影响文件 pre-commit 通过。

数据包懒导出兼容性补验：新增测试逐项检查 __all__ 与映射表一致，并确认
全部公开符号与定义模块的对象身份相同，覆盖之前只验证 standalone 导入
而未直接枚举的公开 API。补充包说明中的延迟导入边界。两项导入/导出专项
通过，Ruff、format、diff 和受影响文件 pre-commit 通过；本轮未改训练路径。

FLOPs 无效必需几何处理：int_attr 不再接受 bool，hidden/layer/vocab/head
必需字段为零或负数时返回 0，head width 非正时同样停止估计，避免无效
num_attention_heads 在推导 head_dim 时除零或输出负 FLOPs。新增十六组
必需字段零/负/bool 回归，FLOPs 专项 38 项通过；Ruff、compileall、diff
和受影响文件 pre-commit 通过。本轮未重跑完整 CPU 套件。

混合并行编排复核：STAGE_ORDER 与 PP_STAGE_ORDER 均由 stages.py 生成，
模型分割后逐 part 变换并重新绑定 PipelineStage.submod。删除 parallelize.py
未使用的 logger，修正旧注释遗漏 EP/CP 的阶段列表。阶段契约专项 7 项
通过；真实双进程 CPU/Gloo Interleaved1F1B，每 rank 两个 model parts，
4 steps × 4 microbatches，loss/梯度范数最大绝对差 0，失败 rank 数 0。
Ruff、format、compileall 和 diff 检查通过；本轮未改数值路径，未新增
GPU/NPU 或 TP/CP/EP 组合验证。
实际 CPU 编译 smoke 最初受本机 TorchInductor 预编译头时间戳不一致阻塞，
独立 kernel cache 仍复用该预编译头。仅在验证进程设置
TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 后编译成功，17×17 causal
BlockMask 的 to_dense 块布局与 PyTorch eager builder 一致。生产代码未加入
编译异常回退；块布局比较不代表 attention 输出或 GPU/NPU 内核数值验证。

KCP 重复装配复核：此前“重复调用叠加 wrapper”的判断不成立，原实现始终
包装 type(module).forward。撤销新增实例标记，保留重复装配更新通信组的
行为。新增回归使用替身验证两次包装均接收类方法、第二次 forward 使用新
group；不运行 Ascend KCP kernel 或分布式数值等价测试。

TP 投影构造契约收紧：ReduceScatterLinear 在分配权重前检查输入和输出
特征数为非布尔正整数，并拒绝非布尔 bias，错误配置会在模块装配入口给出
明确 ValueError/TypeError。TP 投影专项 20 项通过，Ruff 与 compileall
通过；未新增 GPU collective 数值验证。

loss 辅助函数契约收紧：vocab_shard_bounds 的全局词表大小、TP 度数和 rank，
以及 next_token_targets 的 seq_len 现在明确拒绝 bool 与非整数，避免布尔值
参与分片边界或 reshape 计算。loss 专项 15 项通过，Ruff 与 diff 检查通过。

ParallelConfig 校验顺序调整：先验证各轴度数与 data_parallel_shard_size，
再执行各并行家族校验，避免 CP 的 degree > 1 提前对字符串/None 触发
TypeError。扩展整数契约测试到三个 CP 轴和 EP。配置与 CP 专项 233 项
通过；新增这 16 项参数组合前完整 CPU 测试 2151 passed、2 skipped，
包含 KCP 重复装配回归。Ruff、format、compileall 和 diff 检查通过。

并行配置职责归位：将调度名称解析及错误转换从 ParallelConfig 移入
pipeline_parallel_config.validate_pipeline_parallel_config，将 symmetric
memory 的设备检查移入 fsdp_config.validate_fsdp_config。调度解析继续延迟
导入，既有硬件条件和错误文本保持不变；组合非法配置现在按家族校验顺序
报告首个错误。完整 CPU 测试 2167 passed、2 skipped；Ruff、format、
compileall、diff 与排除受保护示例的 pre-commit 通过。设备检查搬运没有
提供新的 GPU/NPU runtime 验证。

Checkpoint step 入口校验：启用的 BaseCheckpointManager.save 只接受非布尔
非负整数，load 接受 -1（自动选择最新）或非负整数。此前负数/浮点/bool
可进入 step 路径拼接，生成不符合自动发现规则的 checkpoint 名称。新增十组
回归确认非法 step 在存储解析或后端保存前拒绝；未启用或未初始化 manager
继续直接返回 False。完整 CPU 测试 2177 passed、2 skipped；Ruff、format、
compileall、diff 与排除受保护示例的 pre-commit 通过。

训练恢复与词表 loss 接线复核：trainer 状态先验证 step/ntokens_seen 再
提交，未发现新增错误；修正 loss_vocab_kwargs 中“head 始终复制、接线为
no-op”的过时说明，明确依据 logits 宽度选择普通或 TP 分片 cross entropy。
当前完整 CPU 2340 passed、2 skipped，包含近期 Grain layout、策略布尔
校验及 scatter 编译梯度回归；Ruff、format、diff 通过。

Grain 状态布局完整性：version 1 的 batch_layout 必须存在，缺失时在
set_state 前拒绝恢复，防止绕过 batch 兼容性检查。当前 state_dict 始终
写入该字段；synthetic 旧状态兼容行为未调整。新增缺失元数据与游标保持
回归，数据专项通过，Ruff、format、diff 通过。

Packing 包加载边界整理：两个公共 builder 改为 resolve_export 懒导出，
导入 conversions 不再提前加载 build、composition 和 sources。独立进程
确认模块加载状态及 builder 对象身份；数据专项 253 passed，Ruff、format、
compileall、diff 通过，本轮未重跑完整 CPU 套件。

数据迭代策略契约补齐：DatasetIterationPolicy 在配方构建前拒绝非 bool
shuffle/repeat，与 loader 的 repeat 入口约束一致，避免字符串 false 在
图构建时执行 shuffle/repeat 后才发现配置错误。新增八组策略回归；数据
专项通过，Ruff、format、diff 通过，本轮未重跑完整 CPU 套件。

Grain loader 参数校验归位：max_num_documents 由 DatasetBuildContext 统一
校验，删除重复检查；repeat 明确要求 bool，避免字符串 false 实际启用
drop_remainder 并绕过有限 DP 检查。新增四组入口拒绝回归，数据专项
245 passed；本轮未重跑完整 CPU 套件。

Scatter 编译梯度补验：较大 source 的三种几何同时测试 eager 与
aot_eager/fullgraph 训练反向，确认 padding 梯度可被 AOTAutograd 追踪。
专项 11 passed；完整 CPU 在参数化扩展前收集运行，2324 passed、2 skipped，
扩展测试随后单独通过。Ruff、format、diff 通过；该验证不代表 GPU/NPU
Inductor kernel 等价。

Scatter 输入范围修正：撤销上一轮 index/src 同形限制，保留 PyTorch 有效
forward 输入范围。context 保存 source shape，较大 src 的反向 gather 后
补零至原形状，未使用元素梯度为零；MoE 同形路径不增加 pad 分配。三组
较大 source 几何以切片后原生 scatter 的梯度作参考，eager/autograd/compile
专项通过。此前“同形入口拒绝”记录由本条替代。

MoE scatter 自定义梯度契约复核：反向 gather 要求 index/src 同形，入口与
fake 实现明确拒绝形状不同，避免 forward 接受较大 src 后在 backward 才
因梯度形状失败。新增较大 src 回归；scatter eager/autograd/compile 专项
通过，MoE 调用本身使用同形 index/src。FSDP prefetch 前置接口校验已核实，
无新增改动。

Pattern 接口参数顺序统一：module_name_matches 接收模块名作为首参数，
pattern 改为必需关键字，避免两个位置字符串传反，与 TP spec 解析接口的
对象在前、规则在后形式一致。全部实现和测试调用方已更新；pattern/TP
专项通过，Ruff、format、compileall、diff 通过，匹配语义不变。

TP/FSDP pattern 数值语法统一：{*} 和范围界限均使用 ASCII 数字，避免
Unicode 数字在 wildcard 与显式范围中得到不一致的匹配行为。全部调用方
已使用 fullmatch，删除重复首尾锚点。新增 Unicode 与尾换行拒绝回归；
pattern/FSDP/TP 专项通过，Ruff、format、diff 通过。

FFN 推导参数契约复核：ffn_dim_multiplier 明确要求非布尔有限正数，
避免 True 被当作倍率 1 或字符串在 math.isfinite 内触发不明确 TypeError。
新增 bool/string 回归；模型构件专项 71 passed，Ruff、format、compileall、
diff 检查通过，本轮未重跑完整 CPU 套件。

并行家族入口加载统一：CP/EP/FSDP/PP/TP 包索引共用 resolve_export 懒导出，
保持同名运行时入口和原有公共符号。独立进程验证五个包导入不加载运行
入口，以及全部公共对象身份；并行专项 365 passed、2 skipped。完整 CPU
2314 passed、2 skipped；Ruff、format、compileall、diff 通过。

Optimizer 包入口懒加载：公共名称通过 resolve_export 按需加载，导入
state_dict 不再连带加载 EMA、scheduler 和 EP checkpoint 实现。独立进程
检查模块加载状态及全部公共对象身份；optimizer 专项 165 passed，Ruff、
format、compileall 通过，本轮未重跑完整 CPU 套件。

FQN 迁移配套整理：工具单测从 checkpoint 搬到 utils/test_module_names.py，
增加重复 wrapper、完整段匹配及自定义策略覆盖；更新当前文件清单和上游
映射定义路径，历史审查记录保留。专项 60 passed，完整 CPU 2314 passed、
2 skipped；Ruff、format、compileall 和 diff 检查通过。

FQN 工具职责迁移：canonical_fqn 从 checkpoint_keys 搬到无 torch 依赖的
utils/module_names.py，optimizer、EMA、EP 直接导入新定义，checkpoint 包
删除旧导出；checkpoint_keys 只保留格式键。相关 checkpoint/optimizer/EP
专项 255 passed；算法及序列化名称保持不变。

FQN 参数语义整理：canonical_fqn 的 prefixes 更名为 wrapper_segments，
对应常量更名为 _WRAPPER_SEGMENTS，明确匹配路径任意深度的完整段而非
名称前缀。源码与测试无自定义 prefixes 调用方，删除旧参数名；算法不变。
checkpoint 与 EMA 专项通过，Ruff 与 diff 检查通过。

Profiler 复用方案纠正：此前仅验证编号推进，未覆盖退出采样与再次进入在
同一全局 step 写入文件的边界。按一次训练运行的现有契约，成功进入后的
实例禁止再次进入，初始化失败仍允许重试；删除复用计数逻辑，保留固定
起点的恢复调度。训练入口每次创建新实例，行为不变。专项 47 passed，
完整 CPU 2310 passed、2 skipped；Ruff、format、compileall、diff 通过。

Profiler 复用计数补验：每次 trace 创建捕获固定 start_step，成功完成
Profiler.step 后推进全局计数，再次进入从新位置构建 trace 与 memory
profiler。避免动态闭包偏移随迭代变化及复用时回到旧编号。真实 CPU trace
回归检查 100 起始的两次进入推进至 102；专项 48 passed，Ruff、format、
compileall、diff 通过。本轮未重跑完整套件。

Profiler 恢复调度修复：不再启动后覆写 torch profiler.step_num；通过公开
schedule 回调应用恢复 step 偏移，使首个采样动作与后续动作均按全局 step
计算。trace 目录与日志同样应用偏移，内部计数保持局部值。真实 CPU profiler
回归检查初始 warmup、下一步 record 与 iteration_100 输出文件；专项
47 passed，Ruff 与 format 通过。本轮未重跑完整 CPU 套件。

Profiler 进入状态复核：拒绝同实例嵌套进入，避免覆盖运行中的 trace/memory
handle 导致资源泄漏；trace 构建也纳入初始化异常清理，失败后恢复进入状态。
正常退出后仍可重新进入。新增两项生命周期回归，profiler 专项 46 passed；
完整 CPU 2309 passed、2 skipped，Ruff、format、compileall、diff 通过。

Profiler 默认目录职责清理：删除运行模块中无引用的 PROFILE_DIR/MEMORY_DIR，
默认路径只由 ProfilerConfig 定义，运行模块只声明目录内文件布局。修正
observability_config 中关于 CLI 嵌套字段前缀的错误说明；实际为独立平坦
参数组再由 from_groups 装配。profiler 与配置专项 257 passed；Ruff、format、
compileall、diff 通过，本轮未改运行路径或重跑完整套件。

CLI 配置视图字段保真：复制原 dataclass Field 并仅设置 init=False，替代
只重建 default/default_factory 的字段，保留 metadata、repr、compare、hash
与 kw_only 属性，且不修改原配置字段。新增遍历全部隐藏字段的回归；配置
专项 213 passed，Ruff、format、compileall、diff 通过。本轮未重跑完整套件。

GC 生命周期复核：close 恢复自动收集状态后，run 现在明确拒绝继续调用，
避免已结束训练持有的陈旧实例再次触发手动收集。静态 collect 仍可供
checkpointer 独立使用。新增关闭后收集哨兵测试；utils 专项 45 passed，
Ruff、format 与 diff 检查通过，本轮未重跑完整套件。

EMA 参数归属验证前移：先枚举所有 model part 的 trainable 参数组并验证
唯一归属，再创建 EMAParamOptimizer 与平均值。统一使用自有 model_parts
快照，重复 part 或跨 part 共享参数不会先分配重复平均值再失败。新增分配
哨兵回归；EMA 专项 60 passed，Ruff、compileall 和 diff 检查通过。

EMA 初始化顺序整理：先编译 buffer 正则、选择 buffer 并检查可平均 dtype，
再分配参数及 buffer 平均值，避免非法配置在额外复制整套权重后才失败。
复用已选择的 buffer 分组，不重复扫描；错误类型与文本保持原样。新增
分配失败哨兵测试覆盖非法正则和整数 buffer。完整 CPU 2304 passed、
2 skipped；Ruff、format、compileall 和 diff 检查通过。

训练步骤 mesh 参数收敛：_clip_and_check_finite 与 _step_metrics 接收已有
TrainingReductionMeshes 对象，替代分别传递 pp/loss 与 dp/pp/loss/sequence。
减少主流程重复局部变量及指标函数参数数量，补充明确类型；归约与裁剪算法
保持不变，trainer.py 没有拆分。专项 118 passed，完整 CPU 2302 passed、
2 skipped；Ruff、format、compileall 和 diff 检查通过。

训练归约接口命名统一：training_reduction_meshes 更名为
resolve_training_reduction_meshes，与验证的 resolve_loss_reduction_meshes
保持解析函数命名一致，旧实现名删除。新增 16 组 DP/CP/TP/PP 开关组合与
单进程测试，检查 token、loss、sequence、pipeline mesh 的选择及训练/验证
共享语义。训练验证专项 144 passed；完整 CPU 2302 passed、2 skipped，
Ruff、format、compileall 和 diff 通过。组合测试使用拓扑替身，不证明真实
设备 mesh 或 collective 数值等价。

并行公共入口导出收敛：parallel 包的懒映射直接指向各家族同名入口文件，
减少经家族索引再导出的路径。parallelize.__all__ 只声明编排函数，PP 返回
类型的公共定义由 pipeline_parallel 维护。新增公共 API 身份验证，覆盖
全部五个入口。专项 8 passed，完整 CPU 2285 passed、2 skipped；Ruff、
format、compileall 和 diff 检查通过。没有新增硬件并行数值验证。

阶段执行薄封装合并：删除只有一个调用方的 stage_enabled，将 compile
条件留在 parallelize 的唯一执行循环，stages 模块只维护顺序和说明数据。
以真实装配入口的调用记录测试替代重复表达式测试，分别验证 compile 开关
及 TP/EP/CP/AC/FSDP 顺序。AC/compile/阶段专项 63 passed、2 skipped，
新增入口回归随后单独验证；未运行硬件 collective。

并行装配顺序收敛：所有阶段均已支持 PP，删除无实际分支的 Stage.on_pp
与 PP_STAGE_ORDER，完整模型与每个 PP model part 共用 STAGE_ORDER。
apply_stages 不再接收顺序参数，减少闭包传参与两条路径的漂移空间。
删除旧子序列测试，保留阶段顺序、compile 条件及表与执行顺序一致性验证。
专项 45 passed；同步设计文档及上游映射，历史审查条目保留原始记录。

lm_head dtype 职责整理：删除模型构件中的 TORCH_DTYPE_MAP，允许的配置
名称集中为 model_config.LM_HEAD_COMPUTE_DTYPES，HF 装配与配置校验共享
该策略，dtype 转换复用 utils.dtypes.get_dtype。CastLinear 保留纯 torch.dtype
接口；checkpoint export 策略保持独立。删除只枚举旧表的测试，保留真实
logits dtype、共享权重与 checkpoint 键测试。专项 231 passed，完整 CPU
2284 passed、2 skipped；Ruff、format、compileall 和 diff 检查通过。

FLOPs 投影几何复核：将 KV head 的 `or` 默认回退改为显式 None 分支，
拒绝零/负 KV head 与不可整除的 GQA 布局。dense FFN 必须有正宽度，MoE
要求正专家数量、合法 top-k、正专家宽度和非负共享宽度；无法确定有效几何
时返回 0。新增十一组回归，FLOPs 专项 59 passed；完整 CPU 测试
2285 passed、2 skipped，Ruff、format、compileall 和 diff 检查通过。

HF checkpoint shard 映射复核：`models/hf/state_dict_adapter.py` 原先从文件名
提取第一个数字作为 writer 编号，导致无数字文件名被漏掉，版本号前缀造成
编号冲突，不连续编号不符合 PyTorch writer 的 1..N 契约。改为按实际文件名
分组，排序后生成连续编号，保留同文件权重的分组关系；索引须提供非空
weight_map，键和文件名须为非空字符串。新增四种文件命名的真实 CPU DCP
保存/读取往返和七种非法索引回归。专项 20 passed；完整 CPU 测试
2351 passed、2 skipped。Ruff、format、compileall、diff 检查及全仓库
pre-commit 通过（按既定约束排除 examples/native_alltoall.py）。复核两种
checkpoint backend 的映射消费者及最终同步导出调用；未运行分布式硬件
导出或 collective 等价测试，不据此宣称全部并行组合已验证。

DCP manager 内部接口整理：将仅有 manager 自身消费者的 `dcp_save` 改为
`_save_checkpoint`，与 `_load_checkpoint` 的内部职责对应；checkpoint_id
之后的 async_mode、GC、HF 格式参数仅接受关键字，现有调用方均已显式
传参。将 `_flattened_model_states_sd` 改为 `_flatten_model_state`，输入参数
使用 states 表示训练状态容器；更新调用方和异步耗时测试，无旧名兼容壳。
同步修正 GC 尚未接入 trainer、rmtree 忽略所有错误等过时说明，并更新
upstream map 中的配置与 HF adapter 路径。checkpoint 与 trainer 专项
170 passed；完整 CPU 回归 2351 passed、2 skipped，Ruff、format、compileall、
diff 检查及全仓库 pre-commit 通过（排除受保护的 native_alltoall 示例）。
保存格式、异步执行和归约算法未改变；本轮未运行硬件分布式验证。

PP 输入与验证调用链复核：`runtime/pipeline.py` 的单 microbatch 路径原先
按 labels 首维长度切片所有张量，可能截断 packed 多模态 batch 的视觉
patch（patch 数量大于 token 数量）或 MRoPE 坐标轴（首维为 3）。现在
单 microbatch 保留张量完整，仅复制顶层映射并按现有契约重数有效标签；
多 microbatch 的逐行切分及 packed 限制保持原有逻辑。新增 2/8 token
回归，覆盖视觉 patch、grid、三轴坐标和 metadata。删除仅转发一次的
`Trainer.validate_body`，由 validation 模块直接调用自身 run_validation_loop；
更新混合 train/eval 模式恢复的成功/失败测试，保留 Trainer.validate 入口。
初次 pipeline/validation 专项 36 passed；完整 CPU 测试 2353 passed、
2 skipped，Ruff、format、compileall、diff 检查及全仓库 pre-commit 通过
（排除受保护的 native_alltoall 示例）。不增加 trainer 文件拆分；未运行
多模态模型的硬件 PP 数值等价验证，CPU 测试仅证明输入切分与模式恢复契约。

HF 初始化接口与说明整理：复核 materialization 的参数/buffer alias 记录、
未知非持久 meta buffer 预检与 RoPE 重建，以及 cast lm_head 的参数身份
保留逻辑。本轮未改变物化算法。将会修改配置的
resolve_experts_implementation 改为 configure_experts_implementation，
配置作为第一个参数，model_cls 为仅关键字参数；同步模型构造、测试和
当前文件清单，无旧名兼容壳。精简 checkpoint base/filesystem/storage
的模块说明，保留当前职责、配置归属、URI 与删除错误契约；历史来源和
迁移差异继续由 docs 记录。HF wrapper/cast 专项 71 passed；完整 CPU
2353 passed、2 skipped，Ruff、format、compileall、diff 检查及全仓库
pre-commit 通过（排除受保护的 native_alltoall 示例）。本轮未运行硬件
FSDP/TP 物化等价验证。

并行 attention head 读取复核：`parallel/attention_heads.py` 原先使用真值
判断跳过 query head=0，并将 KV head=0 回退为 query head 数量；独立
TP/CP 应用入口因此可能接受非法配置。改为只在 None 时使用缺省规则，
显式 head 数量必须是非布尔的正整数；没有 attention 配置的模型仍跳过，
禁用轴 size=1 保持原有不检查行为。新增 query/KV 两字段的十二组非法
数量回归，attention_heads/TP/CP 专项 85 passed；完整 CPU 测试
2365 passed、2 skipped，Ruff、format、compileall、diff 检查及全仓库
pre-commit 通过（排除受保护的 native_alltoall 示例）。未修改通信算法，
未运行硬件 TP/CP collective 等价验证。

验证语料 mask 生命周期复核：验证配置允许覆盖 dataset，但 HF 模型 mask
原先只由训练语料推导，随机训练配 packed 验证可能沿用 causal 并跨文档
注意。验证期间按覆盖语料和 max_num_documents 设置各本地 part 的 HF
config mask；结束或失败均恢复原字段（包括字段不存在或显式 None）。
同一 config 按身份去重，不触碰训练数据游标。新增 random/packed 覆盖
的成功与失败四组回归；validation 专项 34 passed。硬件 attention 数值
等价未验证。

验证 CP kernel 状态复核：CPFlexKernel.packed 在装配时固定，仅临时修改
HF config 不足以切换随机/packed 验证。验证现在同步修改已安装 kernel
的 packed 标志，并在成功/失败时恢复；扩展原四组回归检查 kernel 状态。
KCP 不支持 packed 文档，增加 packed 验证开始前的拒绝检查，防止仅训练
装配检查被验证 dataset override 绕过。新增测试确认在循环/loader 创建
前报错且保留模型模式；首次测试使用不存在的配置字段已修正为实际
context_parallel_kcp_size。完整 CPU 回归 2370 passed、2 skipped；Ruff、
format、compileall、diff 检查及受影响文件 pre-commit 通过。本轮未改变
CP 通信算法，未运行硬件 CP 数值等价验证。

数据布局 mask 策略合并：新增 DataloaderConfig.attention_mask_type，接受
仅关键字的 dataset override；HF factory 与 validation 复用同一规则，
删除两处重复条件。配置负责描述语料布局，模型负责构造与使用 mask；
方法不修改数据配置或语料选择。新增三种文档容量和三种语料覆盖的九组
回归，原 config/validation/HF wrapper 专项 300 passed；完整 CPU 回归
2379 passed、2 skipped。Ruff、format、compileall、diff 检查及受影响文件
pre-commit 通过；未运行硬件 attention 或并行等价验证。

数据配置职责复核：检查 text `ConcatThenSplit`、text `FirstFitPack` 和
multimodal `FirstFitPack` 均使用 `max_num_documents`，因此新增的
`DataloaderConfig.attention_mask_type` 与实际 packing 上限一致；padding
位置和媒体元数据仍由各自 collator/conversion 负责，未错误迁移到 config。
对应 loader/collator 代码只读该配置，不重复推导 mask。数据 pipeline、
multimodal build/data 专项 401 passed，全仓库 pre-commit 通过（排除
受保护的 native_alltoall 示例）；未运行分布式数据等价验证。

验证状态恢复证据补齐：将原语料覆盖测试扩展为原 mask 字段不存在、显式
None、causal、block_causal 四种状态 × 文档容量 None/1 × 随机/packed
覆盖 × 成功/失败，两个本地 model parts 共享同一配置，检查最终字段与
kernel 状态均恢复。新增 28 个用例；validation 专项 63 passed，完整 CPU
回归 2407 passed、2 skipped；Ruff、
format、diff 和受影响测试 pre-commit 通过。本轮只补充失败路径与状态
恢复证据，不修改生产实现，也不宣称硬件 CP 数值等价已验证。

当前审查清单覆盖复核：扫描 llmtuner 的 157 个 Python 文件，所有当前
路径均出现在文件审查清单中；此项只证明清单覆盖，不证明全部运行语义。
重新读取 Trainer.state_dict/load_state_dict，确认 step/ntokens_seen 先
全部校验再赋值，已有非法字段回归保护部分更新。精简 ValidationConfig
说明，明确 CLI 隐藏来源、临时 loader、模式/mask 恢复，以及 dataset
override 仍复用训练 tokenizer/path/packing 设置；同目录导入使用定义
叶模块的相对路径。Ruff、受影响文件 pre-commit 和 diff 检查通过；无
生产行为变更，未重复运行完整测试。

CLI 生命周期复核：重新读取 parse_config/main 与既有 seed、构造失败、
trainer close、process-group destroy 失败测试。入口保留原始异常，依次
释放 trainer 资源和进程组；seed 在 Trainer 构造前按实际 world size 拒绝
多 rank。精简重复的模块历史说明，保留 flat flags、无隐式 args/config
文件、配置组装、资源归属及运行示例；未修改控制流。CLI 专项 18 passed，
Ruff、受影响文件 pre-commit 通过。

LR scheduler 参数接口复核：重新读取 WSD 构造、inner optimizer 分组、
last_epoch 恢复和 tensor learning rate 身份保留；现有恢复实现预计算全部
组后再写入，已有缺失 initial_lr 回归避免部分更新。构造器 lr_lambda 改
为仅关键字，优化器集合作为唯一位置参数；同步 builder 和全部四处测试
消费者，调度曲线与 checkpoint 格式不变。scheduler 专项 34 passed，
Ruff、format、compileall、diff 和受影响文件 pre-commit 通过。

HF config 来源判定复核：删除按斜杠数量猜测离线架构的规则，仅未存在于
文件系统且属于 Transformers CONFIG_MAPPING 的名称走 for_model。其它
标识交给 from_pretrained，避免不存在的绝对路径被当作 model_type 并
生成错误方向的异常。四组回归覆盖绝对/相对缺失路径、Hub ID 和无斜杠
checkpoint 名称，确认均调用 pretrained 解析；原本地目录与离线构建
回归保留。HF wrapper 专项 56 passed；完整 CPU 2411 passed、2 skipped。
Ruff、format、compileall、diff 及受影响文件 pre-commit 通过。新标识解析
测试使用 monkeypatch，不声称真实 Hub 网络下载已验证。

配置预览来源判定同步：auto_fill_model 仍使用旧 slash-count 规则，与 HF
factory 的 CONFIG_MAPPING 判定不一致。同步为只有已注册且不在磁盘的
架构名跳过 pretrained 读取；Transformers 导入仍位于显式预览调用中，
配置包导入保持轻量。新增四组架构名/缺失路径/checkpoint 名/Hub ID
路由回归，config 专项 226 passed；Ruff、format、diff 与受影响文件
pre-commit 通过。预览仍记录加载失败并保留显式尺寸；未进行 Hub 下载。

优化器配置数值与说明复核：learning_rate、weight_decay、eps 显式拒绝
bool、字符串和 None，并统一为带字段名的 ConfigError；beta entries 同样
拒绝非数值及布尔值，避免 False 被当成合法 beta=0。修正 ParamGroupConfig
缺省 lr 必定报错的错误说明：实际采用 Adam/AdamW 构造器默认值，不继承
外层标量；修正 optimizer 只支持 AdamW 的陈旧说明。新增十六组非法输入
回归，optimizer config 专项 36 passed；完整 CPU 2431 passed、2 skipped。
Ruff、format、compileall、diff
及受影响文件 pre-commit 通过。

优化器字段校验合并：learning_rate、weight_decay、eps 的三段相同数值
判断合并为同一字段遍历；eps 使用严格正数边界，其余允许零。保留上一轮
布尔/类型/有限性拒绝和字段错误信息，减少重复条件，不增加单调用方封装。
全部 optimizer CPU 测试 181 passed，Ruff、format、diff 与受影响文件
pre-commit 通过。无参数分组、更新算法或 checkpoint 格式变更。

CP 入口参数统一：apply_cp 仅接受模型为位置参数，mesh/cfg/parallel_dims
改为仅关键字。装配 partial 已使用关键字；同步 CPU 测试与九个分布式
脚本的所有位置调用，包括双 CP 轴 topology 参数。CP/assembly 专项
29 passed；Ruff、compileall（含 integration scripts）、diff 和受影响文件
pre-commit 通过。仅调整传参契约，未执行硬件分布式脚本。

CP 入口导出边界整理：context_parallel.py 的 __all__ 仅保留 apply_cp，
删除无消费者的 input_sharding 导入转发；包 __init__ 仍从 attention 和
input_sharding 定义模块懒加载公共符号。Ulysses 仅内部消费者的检查改为
_require_ulysses_compatible，mesh 补 DeviceMesh 类型，配置和 mesh 仅关键字。
修正已经过时的 eager __init__ 循环导入注释，保留按需加载 HF backend。
CP/assembly 专项 29 passed；Ruff、format、compileall、diff 与受影响文件
pre-commit 通过。通信与 mask 行为未改变，未执行硬件分布式验证。

EP/FSDP 导出职责复核：EP runtime entry 移除未使用的 FP32 buffer 修复
导入，__all__ 只保留 apply_ep；包 API 仍从 block_conversion 定义处导出。
FSDP runtime entry 的 __all__ 移除 plan 中定义的两个符号，内部实际调用
仍保留，prefetch 测试从 plan 定义处导入。FSDP/EP/assembly 专项
91 passed，受影响文件 pre-commit 和 diff 检查通过。全仓库 Ruff/format
检查发现 tokenizer/huggingface.py 当前另有两行长度超限，非本轮并行
文件问题；未据此报告全仓库门禁通过。未运行硬件分布式验证。

Tokenizer ID 边界证据补齐：当前工作区 huggingface.py 的上一轮两行格式
问题已解决，全仓库 Ruff 与 format 检查通过。重新读取完整显式 token
ID 计划预检、排序追加与追加后核对逻辑；补充空 ID、Unicode/fullwidth
数字、正号和小数形式五组失败测试，确认拒绝前不修改 backend 词表。
本轮未修改生产逻辑，tokenizer 专项 37 passed，受影响测试 pre-commit
与 diff 检查通过。

优化器实现选项前置校验：OptimizerConfig.implementation 的 Literal 注解
不会验证程序化输入，新增构造期拒绝未知值，避免模型装配后才报错；
runtime guard 保留以覆盖可变配置在构造后的修改。新增错误拼写、空串、
None、布尔值四组测试。全部 optimizer CPU 测试 185 passed，Ruff、format、
diff 与受影响文件 pre-commit 通过；设备 fused 能力选择仍由运行时负责。

参数组覆盖校验复核：_build_param_groups 本身不检查剩余参数，但容器
在创建任何 optimizer 前统一调用 _validate_params，已按全体 model parts
拒绝遗漏与重复参数，并给出参数名/形状。初次加入局部重复校验导致两项
原诊断测试失败，已撤回重复检查。保留新增两项测试，直接证明缺失覆盖
在 optimizer factory 调用前失败，以及冻结权重不要求参数组。全部 optimizer
测试 187 passed，受影响测试 pre-commit 通过；生产算法无改动。

Optimizer 类型预解析：构造容器原先逐个解析并创建 optimizer，后续组
类型不支持时前面的实例已经创建。现在在参数覆盖校验后统一解析全部
optimizer factories，再创建实例；新增构造哨兵证明未知后续类型在任何
实例创建前失败。optimizer 专项 188 passed；Ruff、format、compileall、
diff 与受影响文件 pre-commit 通过，更新算法和状态格式未改变。

Optimizer factory 薄封装合并：_resolve_optimizer_factory 只有一个消费者，
且每次调用重建同一 Adam/AdamW 映射。将映射与名称预检合并到容器构造
计划中，只建一次，删除方法与旧引用；继续在任何实例创建前拒绝未知
类型。optimizer 专项 188 passed，Ruff、format、diff 和受影响文件
pre-commit 通过；更新算法与 checkpoint 状态格式不变。

Optimizer 参数计划传参整理：_build_param_groups 保留准确职责名称，
模型为位置参数，param_group_configs 和 impl_kwargs 改为仅关键字；
构造调用直接标明配置列表与设备实现选项，减少相邻参数含义不清。全部
optimizer 测试 188 passed，Ruff、format、compileall、diff 与受影响文件
pre-commit 通过，参数匹配顺序和分组算法不变。

CP 拓扑接口标注：apply_cp.parallel_dims 补充 ParallelTopology | None，
通过 TYPE_CHECKING 导入避免增加运行时依赖。mesh 是实际 CP 通信视图，
parallel_dims 用于双 CP 子轴解析，两者职责明确；未改变调用契约或算法。
CP/topology 专项通过，Ruff、format、diff 与受影响文件 pre-commit 通过。

Betas 容器错误处理：tuple(self.betas) 对 None、bool、标量抛 TypeError，
导致程序化配置绕过统一 ConfigError 语义。捕获该转换错误，附字段与
两个数值/CLI 修复示例；保留异常原因以及合法 iterable 归一化语义。新增
三组非 iterable 回归，optimizer 专项 191 passed；Ruff、format、diff 和
受影响文件 pre-commit 通过。数值更新及状态格式未变。

CompileConfig 输入契约复核：backend 现在要求非空字符串（拒绝空白、
None、bool、整数）；per_block 和 enable_async_tensor_parallel 要求 bool。
避免非法程序化配置延迟到 compile/Inductor 路径才失败。更新原空 backend
诊断测试并新增类型/开关回归。compile/config 专项 249 passed、2 skipped，
新增开关测试后的 config 专项通过；Ruff、format 与受影响文件 pre-commit
通过。未执行硬件 Inductor/async TP 等价测试。

Compile backend 校验顺序：backend 选择移到 async TP 和 MoE Dynamo
全局设置前，flex 模型使用非法 backend 时先拒绝，不先注册对称内存或
开启 capture_scalar_outputs。新增 setup 哨兵回归；不承诺后续编译失败
的全部全局状态回滚。compile 专项通过，Ruff、format、diff 与受影响
文件 pre-commit 通过；未运行硬件 Inductor/async TP 数值验证。

跨模块累计回归：最近配置/CLI/CP 导出与参数契约、optimizer 构造计划、
compile 初始化顺序修改后的完整 CPU 2468 passed、2 skipped；全仓库
Ruff、format、compileall（含 integration scripts）、diff 和 pre-commit
通过（排除受保护的 native_alltoall 示例）。AST 扫描确认五个删除旧名
在实现、测试与示例中无引用：resolve_experts_implementation、dcp_save、
_flattened_model_states_sd、validate_body、_resolve_optimizer_factory。
扫描发现 test_config 的 regex 字符串使用非法 Python 转义，改为 raw
string，regex 含义不变。全局审查与硬件并行验证尚不能据这些门禁认定完成。

AC 配置导出归属整理：VALID_AC_MODES 定义在 activation_checkpoint_config，
运行时只消费该约束；移除 memory.activation_checkpoint.__all__ 的重复
导出及旧兼容说明，测试从配置定义模块导入。运行时仍导入用于 apply_ac
校验，未修改 checkpoint wrapper/RNG/重计算行为。memory 专项通过，
受影响文件 pre-commit 与 diff 检查通过；未执行硬件重计算等价测试。

RegionAC 初始化失败路径复核：缺少可选 torch_remat 依赖时，原 apply_ac
先调用 disable_dynamo_lru_cache，随后才在 wrap_region 中报 ImportError。
现于全局缓存设置之前检查依赖；回归测试用 setup 哨兵确认不修改全局设置，
同时确认各 decoder block 的身份和 forward 保持不变。memory 专项
55 passed；未执行真实 torch_remat 或硬件重计算数值验证。本次不承诺
其他运行时包装失败的事务回滚。

SelectiveAC 包装顺序复核：后续 decoder block 的 FQN 匹配非 Linear 时，
原逐层替换流程会留下前面已包装的 block。full/selective 现在先构建全部
wrapper，再设置 Dynamo 缓存并替换层；保留独立 wrap_selective 的调用
契约，无需重复计算形状。新增真实模型回归确认后续层报错时，原层及参数
身份保持不变且不调用全局设置。region 的原地 forward 注解仍单独处理，
本次不声明其包装异常可全部回滚。

SelectiveAC debug 契约复核：配置原允许 debug=True，但 wrapper 必须提供
自定义 context_fn，PyTorch 非重入 checkpoint 明确拒绝该组合。配置现在
提前抛出 ConfigError，字段帮助说明及类文档同步指出限制；debug=False
行为保持不变。新增配置回归和实际 torch checkpoint 调用回归，以验证
限制来自底层接口，而非仅测试本地条件。未修改保存算子集或重计算策略。

数据装配初始化边界复核：packed build_dataloader 原在加载 tokenizer、
构建图后才由 GrainDataLoader 拒绝非布尔 repeat 或有限多 DP 加载。
将该策略校验提取至 contracts.validate_loader_iteration，装配入口与
直接 loader 构造共享，并在装配入口的资源加载前执行。random 分支保持
原有无限生成语义。新增失败顺序回归，使用最小配置及 tokenizer 哨兵
确认无效策略在读取后续构建字段前被拒绝；未增加有限多 DP 支持。

数据迭代策略结构复核：将只有 apply_iteration_policy 消费的 shard_for_dp
合并到该函数，集中呈现 shuffle → 连续 DP 切片 → repeat，余数分配公式
保持不变，无旧名别名。同步 synthetic 说明。concat 分片测试原每次调用
next(iter(graph))，只重复读取首行；现连续读取整个分片，并与单 rank
完整 shuffle 顺序对照。两份相同语料可产生相同内容，验证的是索引覆盖
及顺序，不能要求其内容集合互斥。

HF 随机访问源包装精简：删除无额外职责的 HuggingFaceRandomAccessDataSource，
build_source 在确认返回值为 datasets.Dataset 后直接返回该对象。Dataset
本身满足 RandomAccessDataSource 的长度和索引协议；源描述、加载选项
校验以及 streaming cursor 保持不变。新增真实本地 Dataset → Grain
MapDataset 回归，确认对象身份、完整行遍历和负索引行为；不涉及 Hub 下载。

ChatProcessor 输入契约整理：在 __call__ 统一执行 messages_fn，单轮
与 renderer 分支均消费消息列表；_tokenize_sample 更名为
_tokenize_single_turn，明确其模板与 prompt-prefix 路径职责。
renderer 分支补齐 TextSequence | None 返回注解。旧私有名称无调用方，
未保留别名；样本溢出、next-token shift、assistant loss mask 不变。

多模态处理命名整理：process_cc12_wd_sample 更名为 process_cc12m_sample，
与 CC12M 配方名一致；更新配方、测试及文件清单，不保留旧别名。
模块长流程图改为职责说明，明确样本处理与 packing/collator 的边界。
图像解码、占位符、标签 mask 与缺图拒绝行为保持不变。

MultimodalProcessor 参数绑定整理：将仅用于调用转发的 tokenizer、图像
及视频选项在构造时用 functools.partial 绑定到 sample_processor，删除
对应独立属性与逐样本转发表。保留 sample= 关键字、自定义处理器选项、
patch 基本检查和长度过滤。内置图像处理器忽略的视频选项仍按原契约传入，
本次不扩大视频支持声明。

MultimodalCollator 媒体接口命名复核：collate_images 同时用于图像与视频
patch 组装，更名为 collate_media，参数 all_images 更名为 media。
更新两个内部调用，无旧名兼容层；图像/视频独立输出、patch_order、grid
及 MRoPE 位置契约不变。复用现有图像与视频 patch/布局测试。

MRoPE 网格构造局部化：将仅一处调用的 vision_grid_positions 合并到
视觉区域处理循环，网格形状缓存不再作为辅助函数参数流转。保持 T/H/W
展平顺序、按形状复用缓存以及 text_len + offset 偏移；有两处消费者的
text_positions 保留。无旧名别名，复用现有多文档、图像/视频和位置测试。

smart_resize 参数契约整理：height/width 保持位置参数，factor 与
min_pixels/max_pixels 改为 keyword-only，避免连续整数位置传参混淆
对齐因子与预算。同步图像路径及测试，视频路径原已使用关键字。
保留预算搜索、比例保护与异常语义，复用紧预算和窄长图像回归。

视频依赖失败路径复核：load_video 原将 PyAV 缺失捕获为普通样本失败，
可能让全部视频样本被丢弃。将懒导入移到解码异常处理之外，缺失时抛出
带 pip install av 的 ImportError；文件和解码异常仍返回 None。
缺依赖测试显式设置 sys.modules，不依赖测试机器是否安装 av。

视频帧数契约复核：load_video 原仅检查上下限关系，允许浮点/布尔帧数，
浮点数可在 np.linspace 处失败并被捕获为样本丢弃。复用 contracts 的
require_positive_integer 检查 min_frames/max_frames，在导入后端之前
拒绝错误配置。新增八个字段/类型组合回归；合法采样与缺失元数据路径
保持不变。

视觉 scatter 写入顺序复核：scatter_vision_embeds 原在覆盖输入后检查
packed bank 总量，多余或不足的特征可留下部分已覆盖输入。现先计算
span 消费总量并检查 bank 行数，再逐段写入。新增 bank 多余/不足回归
确认异常时输入逐元素保持不变；合法融合、dtype 转换与原地返回不变。
本次不声明任意非法 span 或写入异常均可回滚。

QKV 职责复核：确认 _project 由 AllGatherFusedQKVLinear 覆盖，是实际
TP 扩展点，保留该封装。精简 qkv.py 模块说明，集中描述 grouped KV
布局、已有 projection 的注入、leading 维度保持及 state-dict 转换，
删除冗长历史类型检查叙述。本轮该模块无运行时算法修改。

ReduceScatterLinear 构造去重：改为继承 nn.Linear，保留 feature/bias
输入校验，复用标准参数分配与 reset_parameters，删除重复分配和薄
初始化封装。TP forward 与无 TP fallback 保留；state-dict 的 weight/bias
键不变。复用已有固定 seed 初始化、reset、bias、梯度和恢复测试。

ReduceScatterLinear 类型关系后续复核：当前 llmtuner 训练装配没有自动
构造原生 TP projection 类的调用方；HF TP 的 resolve_tp_targets 按
nn.Linear 和 plan 选择目标并替换。继承 nn.Linear 后该类也会进入此
识别范围，不能依据普通投影 CPU 回归宣称自定义原生 TP 构件与 HF TP
变换可任意叠加。AC 的形状与 region 识别同样使用 nn.Linear。
修正 bias 文档：选项要求 bool，False 才令存储的 bias 为 None。
本轮为调用链复核与说明更新，无新增并行组合能力。

VocabParallelEmbedding 输出契约复核：实现只做本地词表查找和外部 token
置零，不执行 TP collective。修正文档，明确 num_embeddings 为全局词表
大小、weight 为本地 shard、输出为局部贡献，完整 embedding 需调用方
求和。HF TP 未自动安装此独立构件；移除含糊的“TP 未接线”叙述。
本轮无运行时算法或通信变更。

词表局部贡献数值复核：新增所有本地 shard 输出相加与完整 nn.Embedding
的输出/梯度对照，覆盖 vocab=7,tp=3 的余数与 vocab=2,tp=4 的空 shard，
含重复 token 和全局 padding row。使用不同输出梯度验证拼接权重梯度。
此测试为 CPU 模拟 rank 的数学验证，无真实进程组通信。越界 token
在本地 mask 路径可被当作外部 token 置零，当前未增加逐 forward 的
设备同步校验；该输入边界仍需结合调用方策略继续审查。

词表越界输入修复：TP embedding 的 mask 原将负数及超出全局词表的
token 静默置零。复用仓库已有 torch._assert_async 模式，在局部查找
及 max_norm 权重修改前检查全局范围，避免 Python bool 的逐步设备同步。
新增负数、词表上界及更大 ID 回归，含合法本地 token，确认 CPU 报错
且权重不变。设备端异步断言不承诺 GPU 异常时可恢复继续训练；未运行
硬件 collective 测试。

RoPE 编译位置边界复核：独立 RoPE 模块不消费 BlockMask，不能在编译时
假定其位置已由 mask 保证。移除 maybe_check_max_pos 的 is_compiling
跳过分支，保持 torch._assert_async 在 Dynamo 图内。Complex/CosSin
负位置回归覆盖 eager 与 torch.compile backend=eager/fullgraph；不将
该 CPU 图验证等同于 Inductor 或硬件验证。

RoPE 缓存接口精简：位置范围检查合并到共享 reshape_for_broadcast 的
索引分支，删除 maybe_check_max_pos 及两个重复调用。共享函数实际只
读取 token 数量，query_shape 参数改为 num_tokens，positions 改为
keyword-only；删除 ComplexRoPE 无效的半宽 query shape 构造。
缓存格式、广播输出和异步断言语义保持不变，无旧检查函数别名。

RoPE 位置形状契约修复：显式 positions 原可将 (1,T)/(T,1) 索引结果
通过 view 静默展平，现共享缓存入口要求一维且长度等于 token 数。
缺省 positions 的 token 数超过缓存容量时给出明确 ValueError，替代
后续 view 形状异常。新增两种 RoPE 格式的错误形状及容量回归，合法
位置索引与广播不变。

RoPE 位置 dtype 修复：bool/uint8 索引会被 PyTorch 当作 mask，尤其
缓存长度等于 token 数且 mask 全真时可静默选择整个缓存。共享入口现
只接受 int32/int64 position IDs，拒绝 mask 和浮点类型；新增两种
格式的 dtype 失败回归及 int32/int64 输出逐元素等价验证。

EMA 状态构件命名整理：EMAParamOptimizer 同时持有参数和 buffer 状态，
不执行训练更新，更名为 EMAStateOptimizer。同步构造、类型注解、错误
提示与测试 patch/import，无旧名别名。公共 EMA 与 checkpoint 状态
键格式不变，复用 EMA 更新、身份校验及恢复回归。

EMA buffer 装配精简：buffer_groups 已表达每个 model part 的匹配集合，
状态构造改为直接列表推导，无匹配诊断使用 any(buffer_groups)，删除
重复累计匹配数的分配循环。无 pattern 时仍不构造 buffer 状态，有
pattern 时仍为每个 part 构造一个状态对象。固定身份检查 helper 保留，
step 与冷启动恢复均复用它；未修改 checkpoint 键或 decay 计算。

EMA 状态校验归属调整：_tracked_averages 只消费状态 optimizer 的 state
与 param_groups，移入 EMAStateOptimizer.tracked_averages，EMA 保留
模型范围解析及更新调度。两类 tensor 分组直接向状态持有对象读取
平均值；新增/丢失 tensor 的拒绝、无 defaultdict 插入及全部组先验证
语义保持不变。删除原方法，无兼容转发。

EMA 装配链路联合回归：检查 training_config 与 runtime/assembly，实际
配置存储字段为 training.ema_config，runtime 通过 training.ema property
读取；EMA 类文档改为与装配调用一致的 property 名称，非删除配置字段。
近期状态对象命名、buffer 构造与身份校验归属调整后，optimizer 服务层
联合测试 191 passed。该回归覆盖普通 optimizer/scheduler/EMA CPU
逻辑，不证明分布式 DCP reshard 或硬件 foreach 等价。

Optimizer 恢复参数契约整理：load_flat_optim_state_dict 与
unflatten_optim_state_dict 的 checkpoint 参数统一为 keyword-only
flat_state，optimizer 保持首个位置参数。更新容器恢复计划与直接测试
调用，区分 flat FQN checkpoint 与原生 integer-key state；序列化键、
分组校验、嵌套恢复及加载顺序保持不变。

Checkpoint 恢复调用复核：全仓 Python 调用扫描确认 flat optimizer 恢复
已使用 flat_state=，容器先为全部 optimizer 构建恢复计划再加载。
BaseCheckpointManager._states_to_load 的 model_only 布尔策略设为
keyword-only，并同步唯一调用，明确恢复范围。保留状态选择函数的
独立职责；模型专用恢复与 exclude_from_loading 的语义不变。

Checkpoint 生命周期与目录清理复核：检查 base、DCP 和 torch_checkpointing
后端的 staging/save 等待、关闭与后台 purge。失败 future 先清引用再等待，
关闭流程继续释放其余资源，未发现需要修改的生命周期行为。临时目录匹配
原使用 Unicode `\d`，与正式目录的 ASCII step 规则不一致，会将包含混合
数字的非标准临时目录加入删除队列；改为复用 `_STEP_DIR_PATTERN`，保留
prefix 转义和当前保存目录保护。两个真实 manager CPU 回归覆盖普通和含
正则元字符的 prefix、规范遗留目录、Unicode/前导零目录和正在保存目录；
仅检查删除队列，不执行后台删除，不代表远端存储或分布式 IO 已验证。

训练装配与 profiler 可读性复核：保留 assembly 的模型、并行、optimizer、
checkpoint 服务边界，压缩 scheduler、EMA、container hooks 与 metrics 的
重复背景注释，保留 inner optimizer 学习率、PP hook 调用频率等维护约束。
Profiler 模块说明集中描述生命周期及设备行为，上游差异由映射文档承载；
两个 builder 和 active handle 补全返回/成员类型。映射文档纠正 leaf_folder
已裁剪的旧描述：当前接口和 worker 子目录测试仍保留，非 torchft 支持。
本轮没有修改训练装配顺序、目录布局或 profiling 调度算法。

PP runtime batch 切分复核：split_pipeline_microbatches 原按 labels 行数
切片全部非标量 tensor，未验证其他字段的 leading axis。row labels 为 4
而 pixel_values 为 6 条时，两个 microbatch 仅覆盖前 4 条，丢失尾部媒体；
3 条 grid 或 3 轴 MRoPE 也会被错误按样本行分配。多 microbatch 路径现在
先检查全部 tensor 是否与 labels 同行数，再切分；不提供新的媒体布局转换，
需要非行布局的调用方使用单 microbatch 或提前构造可切分 batch。
num_pp_microbatches 复用数据 contracts 的正整数校验，先于 batch 访问；
synthetic Batch 要求二维 labels，dict 保留一维 packed 与二维 row labels。
scalar tensor/非 tensor metadata、源 batch 不变、单 microbatch packed
media 保留、有效 label 重计数的既有契约保持。CPU 回归覆盖多/少 leading
entries、非法 degree、labels rank 和 scalar metadata；未验证真实 PP 通信。

AuxLoss 本地汇总与日志收集结构复核：zero_aux_losses 重命名为
rollup_aux_loss_metrics，register_aux_loss_zero_hook 重命名为
register_aux_loss_metrics_hook，明确 step hook 同时汇总并清空实例寄存器。
同步 defining leaf、common 懒导出、trainer assembly 和测试，无旧接口别名；
本条保留旧名称作为迁移记录。collect_aux_loss_metrics 删除仅供一次使用的
_group_acc_or_zero 内部函数和 group_accs 临时字典，直接遍历排序后的
group counts，在该循环读取本地值或构造零贡献，再按 group mesh/PP 归约。
全局 group counts 注册、denominator、gradient 注入和 checkpoint replay
策略未改。模块说明删除已过期的 optimizer/cycle 细节，保留核心数值契约。

MoE 辅助 loss 通信职责调整：PartialToInvariantAllReduce 从模型 loss 文件
移到 accelerator/collectives.py，与通用 all_reduce 放在同一通信层。名称和
前向 clone + SUM、反向 identity 契约保留，MoE loss 从 defining leaf 导入，
原文件删除定义，无转发 wrapper；accelerator package 不新增 eager import。
新增 CPU autograd 回归模拟远端 summand，核对输入不变、输出求和、非线性
下游 loss 的局部梯度及 backward 不再发 collective；该测试不替代实际多进程
或 GPU/NPU 通信等价验证。

GroupedExperts 分段输入与能力接口复核：前向原直接 cumsum token counts，
fallback 分配 new_empty 后按 counts 写输出；counts 总数偏小时尾部未写，
仍可能被当成有效输出使用，fractional/negative counts 也会破坏分段。
现在先检查输入 (R, D)、counts (local E,) 与整数 dtype，使用 tensor
assertion 检查 nonnegative 和 sum=R，随后才执行投影；未使用 Python
item 抽取校验值。CPU 回归用 projection sentinel 确认非法输入在计算前
失败，既有空 token 的 input/weight zero-gradient 契约保持。
删除 grouped_mm_available 单纯转发 has 的包装及导出，constructor 直接
has("torch_grouped_mm")，测试通过构造器验证 default 自动选择。显式
use_grouped_mm、BF16 fused dtype 门控、fallback 算法均保持。模型公共构件、
Domino、dispatch 与 EP replacement 联合 CPU 测试 149 passed；尚未验证
实际 fused GPU kernel 或分布式训练下新增 assertion 的行为与开销。

EP dispatcher 文件职责复核：将 TorchAOTokenDispatcher 与安装提示从
token_dispatch.py 搬到 torchao_dispatch.py。通用模块保留 local/all-to-all
及 per-call metadata，adapter 从该 leaf 导入父类和 metadata，不反向导入。
同步 block_conversion、CPU tests、包说明、开发指南与上游映射；旧模块不
重导出 adapter，无兼容壳。torchao 仍在 constructor 内导入，未新增包级
eager import。纯搬运，padding、unpermute、EP=1 与跨 rank 流程未改；
dispatch 与模型替换联合 CPU 回归 67 passed，真实 torchao/CUDA 未验证。
上游相似度表旧数值对应拆分前文件，已标注拆分后未重算，避免沿用旧证据。

EP group 与 TorchAO padding 参数接口复核：wire_meshes 重命名 set_ep_group，
方法仅安装一个 process group，不构造 mesh；同步模型转换和两份 torchrun
脚本，旧名称无别名。TorchAOTokenDispatcher 的 pad_multiple 改 keyword-only，
直接构造校验拒绝 bool、fractional、string、None 和非正整数，在可选包导入
前失败；CPU 缺依赖测试通过 sys.modules 明确阻断 adapter leaf，避免依赖
机器是否安装 torchao。dispatch 与模型替换 CPU 72 passed，完整 CPU
2541 passed、2 skipped。真实双进程 CPU/Gloo ep_equivalence 最大相对误差
3.271e-7；ep_domino_equivalence 八种 Routed/Domino × zero-receives ×
checkpoint-recompute 组合前向与梯度匹配。该结果不证明 TorchAO/CUDA/NPU
内核或全部混合并行矩阵已验证。

Metrics 配置传递复核：MetricsProcessor 原忽略 config.tag，仅依赖调用方
重复传 tag=config.tag；直接构造配置对象的调用方会丢失记录前缀。现在
constructor 在构建 sinks 前解析 tag：None 使用 config.tag，显式字符串
覆盖配置。trainer assembly 删除重复转传。CPU 回归覆盖配置默认、显式
覆盖、无前缀三种情况；保留 sink 独立加前缀的契约，console 不加前缀。

TP 配置边界复核：enable_sequence_parallel=False 原在 TP=1 时也拒绝，
与错误提示建议 tensor_parallel_size=1 不一致。现在仅 TP>1 时拒绝关闭
sequence parallel，TP=1 的闲置选项允许关闭；实际多 rank TP 仍不提供
replicated-activation 实现。既有拒绝用例改为 TP=2，并加入 TP=1 接受边界。

TP projection 接口清理：全仓运行时搜索确认 validate_tp_projection_preconditions
仅有测试消费者，未被装配调用；删除函数和导出及镜像测试。TP>1 的 SP
配置约束由 tensor_parallel_config 负责，公共 projection 保留直接调用和
无 TP 时的标准 projection fallback。移除 warning doc 中不存在的 backend
前置校验描述，同步 introduction 的 TP=1 开关边界。此项不表示原生 TP
projection 与 HF TP 变换可以任意叠加；实际支持仍由装配路径决定。

Scheduler 保存一致性复核：容器只保存一个 last_epoch 以支持 PP optimizer
数量变化，但公开 iterator 允许独立推进内层 scheduler。state_dict 现在
检查所有 epoch 相同，否则拒绝保存，避免静默丢失进度差异；正常统一 step
与 checkpoint 格式不变。真实 AdamW/LambdaLR 回归覆盖独立推进、保存失败
不修改状态，以及统一恢复后重新可保存。

Scheduler 恢复长度边界复核：非负 last_epoch 仍可能超出 total_steps，原路径
直接求 lambda 会产生负 linear/sqrt rate、cosine 回升或无 decay 的断言。
现在恢复前拒绝 last_epoch>total_steps，并提示使用覆盖 checkpoint 进度的
schedule；拒绝发生在任何 lambda 求值或状态写入前。六组真实双 optimizer
CPU 回归覆盖无 decay/有 decay 与 linear/sqrt/cosine，验证失败不改状态和
学习率，last_epoch=total_steps 仍允许恢复。未改变有效范围内曲线。

Loader 装配入口说明复核：build_dataloader 的 repeat=False 文档原泛称有限
loader，实际 random 分支绕过 iteration policy 始终无限，multi-DP corpus
有限读取被拒绝。重写入口 Args，明确 DP ownership、per-rank token budget、
text PP 独立 packing rows、random row budget、单 DP 有限 corpus 和 recipe
override。删除旧训练路径背景说明，保留 multimodal whole-document packing
与 text selector/PP row grouping 的差异。本轮仅说明调整，未修改算法。

Synthetic loader 迭代结构整理：删除 _slice_each 生成器包装，固定 DP 行边界
在 constructor 保存为 slice，__iter__ 直接取全局 source batch 后切 input
与 labels。checkpoint 恢复仅替换 source iterator，现有 iterator 每次取值
仍读取 self._iterator；游标在 yield 前递增、源 seed/index 生成规则和
checkpoint 布局保持。删除只用于边界计算的 _dp_rank/_rows_per_rank。
复用现有 DP ownership、恢复已持有 iterator、配置一致性和训练流程回归。

Dataset recipe 声明结构整理：将 composition.py 的 WeightedDataset、
DatasetMix、DatasetConcat 与 DatasetRecipe union 移到 SingleDataset 后，
使四种声明及完整组合结构集中在 builder 前。构建函数仍按单数据集、共享
transform/iteration policy、mix、concat 排列。仅调整声明顺序，保留
dataclass 字段、public names、shuffle/shard/repeat 与权重归一算法。

Text packing 长度接口整理：row_lengths(context) 改为
text_packing_lengths(*, num_tokens_per_row)，去掉不读取的 tokenizer、read
options 等 context 依赖；两个 text packer 显式传每行 token 长度。四个
feature 用同一 comprehension 生成长度映射，保持 Grain layout 和 padding
语义。共享 helper 有两个实际消费者，保留其独立布局职责，无旧名称别名。

累计接口调整联合门禁：当前 llmtuner 158 个 Python 文件与本表路径逐项
比较，无缺失条目、无已删除路径；这仅证明 inventory 一致，不证明全模块
数值正确或硬件组合覆盖。上游映射同步 TP=1 的 SP 闲置开关边界。scheduler
保存/恢复校验、synthetic iterator 简化、recipe 声明顺序与 packing 长度接口
调整后完整 CPU 2549 passed、2 skipped；全仓 pre-commit（按用户约束排除
examples/native_alltoall.py）、compileall、diff check 通过。GPU/NPU 和
完整 FSDP/TP/PP/CP/EP 组合验证仍不由上述门禁证明。

HF 模型构建与替换复核：沿 factory → model_setup → CastLinear 与
materialization 检查配置传递、参数身份、训练模式、checkpoint 键和 buffer
别名恢复。本轮未修改运行时行为；精简 factory 的构建说明，纠正真实语料
始终多文档 packing 的过时描述，以及 ParallelConfig 对 TP=1 时允许关闭
sequence parallel 的字段说明。现有 HF wrapper、CastLinear 和配置回归
310 passed；相关模块 Ruff/format 与全工作区 diff check 通过。这些结果
不代表 FSDP DTensor meta materialization 已经过 GPU/NPU 数值验证。

HF 离线复合配置边界修复：factory.py 现在延长实际 text_config 的
max_position_embeddings，避免顶层 seq_len 不作用于文本 decoder。保留
原有更长的文本上下文配置，普通 decoder 的显式 override 行为保持。
arch_overrides 改为 deepcopy：真实 Llava 构造器会向嵌套 text_config 字典
写入 model_type，原浅拷贝会污染调用方配置；回归测试先复现此副作用，
修复后通过。新增 Llava/Gemma3 × 短/长上下文四组真实 HF 配置回归，检查
文本长度及输入字典保持。ModelConfig 文档同步注册架构名判定逻辑。
完整 CPU 2553 passed、2 skipped；全仓 pre-commit（排除用户保护的
examples/native_alltoall.py）、compileall 和 diff check 通过。这里只验证
配置构建，不声称完整多模态训练或硬件混合并行已经验证。

HF 模块接口整理：factory.__all__ 补齐实际由 model 与 runtime assembly
使用的 resolve_model_class/unwrap_text_config，仍由消费者导入 leaf module，
不增加包级 eager 导入。hf 包职责索引补充 model_setup 与 attention_backend。
核实当前 Transformers 严格配置拒绝 tuple architectures，因此撤回假设的
容器兼容修改，不将该情况记录为已修复 BUG。HF wrapper 60 passed，HF
模块 Ruff/format 与全工作区 diff check 通过。

HF checkpoint 接口复核：检查 state_dict_adapter 的单层 model 前缀转换、
collision 拒绝、tied head 补齐/导出过滤、indexed 与单文件键覆盖、shard
分组和 distributed checkpoint 的真实消费者。get_hf_storage_reader 的
from_quantized 改为 keyword-only，DistributedCheckpointManager 显式传递
该模式，避免路径后的无名称布尔值；thread_count 默认与量化拒绝行为不变。
没有新增旧签名兼容包装。现有真实 safetensors/DCP 读写及 checkpointer
回归 95 passed；相关 Ruff/format 与 diff check 通过。PP 键覆盖回归使用
模拟 collective，不能视为实际多进程 checkpoint 验证。

复合模型 checkpoint 共享权重修复：runtime assembly 将顶层 HF config
交给 adapter，而 HFTransformerModel 实际运行 text_config。adapter 构造
时统一解析 text_config，使 to_hf/from_hf 使用实际 decoder 的
 tie_word_embeddings，避免错误拒绝省略 tied lm_head 的 HF checkpoint
或为非共享 decoder 错误补齐 head。真实 Llava 配置回归令顶层开关与文本
开关相反，覆盖共享/不共享两种情况、导出拒绝边界、导入 Parameter/tensor
引用及输入字典保持。联合 adapter/checkpointer/trainer 194 passed；
完整 CPU 2555 passed、2 skipped。全仓 pre-commit（排除保护的
examples/native_alltoall.py）、compileall 和 diff check 通过。未验证
完整多模态训练或 GPU/NPU 分布式 checkpoint 行为。

训练运行时可读性整理：assembly 精简 chunked loss/PP 限制、seed 和 dense
mesh 选择的重复背景说明，保留构建顺序及各阶段职责；validation 的
feasibility 与 validate docstring 合并重复段落，明确有限验证的 DP=1
约束、随机数据无限源、临时 loader 生命周期、模式恢复和 token/loss
归约。纠正文档原有所有 rank 都报告相同验证数值的泛化表述：只有最后
PP stage 持有并报告 loss。此轮仅注释/docstring 调整，不改执行逻辑。
trainer 与 chunked loss 回归 100 passed；runtime Ruff/format、编译及
全工作区 diff check 通过。

数据并行 ownership 结构整理：runtime/batch.py 的 dp_rank_world_size
合并 PP 与非 PP 的重复 mesh 查询及 rank/size 返回，先取得 topology，再
按 pp_enabled 选择 batch 或 dp view。保留 include_singleton_axes=True、
未装配 topology 时 (0,1) fallback，以及 PP 所有 stage 读取相同 DP shard
的契约。未修改 token accounting、数据预算或归约算法。train/validate/
reduction mesh 回归 177 passed；相关 Ruff/format 与 diff check 通过。
mesh ownership 回归主要是 CPU 逻辑证据，不证明硬件组合数值等价。

学习率服务说明优化：lr_scheduler.py 顶层长篇上游比较缩为职责、inner
optimizer 关系与恢复契约；total_steps 参数说明同步现有恢复边界检查。
wsd_factor 与 warmup_steps CLI help 明确首个更新采用
base_lr / warmup_steps，而不是 0，避免与实际 LambdaLR 初始化索引不符。
warmup/stable/decay 曲线、虚拟末步及 checkpoint 格式保持。scheduler 与
配置回归 276 passed；相关 Ruff/format、compileall、diff check 通过。

CLI dataclass 视图接口整理：cli_view 改用公开 dataclasses.fields 构建
真实字段映射，隐藏字段校验与复制复用该映射，去掉对内部
__dataclass_fields__ 的直接依赖。ClassVar 不是真实 CLI 配置字段，若误列
PROGRAMMATIC_ONLY 现在明确拒绝为 stale table。新增 ClassVar 配置回归，
保留已有 default/default_factory/metadata 属性与实际 CLI parsing 回归。
配置测试 236 passed；相关 Ruff/format、compileall 和 diff check 通过。

并行 pattern 接口整理：resolve_module_patterns 将第二份字符串列表
patterns 改为 keyword-only，与 plan_name 一起明确表达选择规则；模块清单
保留首个位置参数。FSDP ignored-module plan 和全部测试消费者同步，无旧
签名包装。ASCII 数值片段、范围、wildcards、无匹配拒绝和空 inventory
先校验 pattern 的语义保持。parallel CPU suite 387 passed、2 skipped；
相关 Ruff/format、compileall 和 diff check 通过。未执行新的硬件 collective
验证，此轮不修改匹配算法或通信路径。

FSDP pattern plan 可读性复核：named 改为 named_modules，selected 改为
unit_options，准确区分模块目录与各 FSDP 单元策略字典。顺序检查覆盖忽略
完整子树、多个 pattern override 合并、深层单元先包裹及 root 最后处理，
prefetch 使用注册顺序而不是推断任意 forward 顺序。仅局部变量命名调整，
不改公共 API、策略合并、通信或 wrapping 算法。FSDP contract 42 passed；
相关 Ruff/format、compileall 与 diff check 通过；未执行硬件数值验证。

并行编排入口说明重组：parallelize.py 精简重复上游比较，明确
STAGE_ORDER、dense module 与 PipelineParallelSetup 返回契约、PP 先拆分
再变换以及 AC/compile 选项共享。核对真实 EP 导入链：model_replacement
负责遍历替换，block_conversion 负责单块转换，二者均仍存在，不把职责
划分误当作旧文件删除。执行逻辑及公开接口保持。assembly stage 回归
8 passed；相关 Ruff/format、编译和 diff check 通过。

GC 工具职责说明整理：garbage_collection.py 精简长背景及日志包装说明，
聚焦显式收集边界、关闭恢复和 setup failure 清理；debug 路径从 defining
leaf module 直接导入 is_main_process，仍只在 debug=True 时加载。收集
频率、主 rank 调试限制及 collector 生命周期不变。utils 回归 45 passed；
相关 Ruff/format、compileall、diff check 通过。

多模态 tokenizer 词表查询修复：MultimodalTokenizer 使用既有 token_to_id
查询基础词表与 added vocabulary，删除自行构造 added-token-only 映射。
原实现会拒绝基础词表中的有效媒体标记，与按词表校验的契约不符。保留
五种标记的显式 ID、缺失 token 拒绝，以及 ID=0 有效的行为。新增真实
WordLevel tokenizer 回归（added decoder 为空），检查所有 ID 与缺失路径；
复用现有 added special-token 和多模态 pipeline 测试。联合 188 passed；
相关 Ruff/format、compileall、diff check 通过；未验证完整 VLM 训练。

Tokenizer 特殊标记恢复结构简化：_infer_special_tokens 合并 BOS/EOS/UNK
属性赋值与 standard token collection，避免对相同配置值进行两次解析；
为 AddedToken 列表补充类型。标准标记顺序、decoder metadata 优先级、
显式 ID 先校验再添加及追加无缺口规则保持。tokenizer/data pipeline/
multimodal 回归 446 passed；相关 Ruff/format、compileall、diff check
通过，未增加无消费者 wrapper 或旧名别名。

Tokenizer/loader 依赖结构复核：loader_builder 的 HuggingFaceTokenizer
导入放到 text branch，多模态 branch 使用其自身入口，synthetic 仍提前
返回。MultimodalTokenizer 继承 HF tokenizer，因此这不减少多模态运行
所需依赖，只明确各构建分支的直接依赖。近期累计完整 CPU 2558 passed、
2 skipped；全仓 pre-commit（排除保护示例）、compileall 和 diff check
通过。导入位置调整后另行复跑 text/multimodal pipeline 回归。

Dataset recipe dispatcher 结构整理：build_dataset 的三个 match 分支直接
调用对应 builder，去掉共同 builder 变量，使 SingleDataset/Mix/Concat
的 node 类型与函数参数保持明确关系。apply_iteration_policy 的 policy
改为 keyword-only，两个内部调用同步；未修改 shuffle/shard/repeat
顺序、mix weights、concat graph 或 seed 派生。pipeline 回归 258 passed；
相关 Ruff/format、compileall 和 diff check 通过。

Dataset source 结构复核：sources.py 的 build_source 说明移除过时迁移
背景，集中说明 description/build、map shuffle 后分片、stream shuffle
前分片与已构建 source passthrough；JSONL index 说明纠正 dataclass 可
哈希的误导，明确 compact offsets 不保留 row 数据。两个 passthrough
isinstance 合并 union，保留字符串拒绝、HF 类型检查和 cursor 行为。
pipeline 258 passed；相关 Ruff/format、编译、diff check 通过。

HF stream 游标隔离修复：真实 IterableDataset 交错迭代复现同一 graph
的两个 Grain cursor 共享 dataset checkpoint 进度，推进一个会污染另一个
get_state。StreamingIterDataset.__iter__ 现在 deepcopy HF dataset，使
每个 cursor 拥有独立 source progress。新增真实生成器交错读取、保存状态
不串扰及第三个 iterator 恢复回归。仅复制 stream 描述及状态，不物化
全语料；自定义 stream 内含不可复制资源时仍可能失败，不声称任意外部
资源可复制。pipeline 259 passed；Ruff/format、编译和 diff check 通过。

HF stream epoch/repeat 恢复验证：新增真实 IterableDataset repeat 跨
多个 epoch、保存于非零 epoch 并恢复的回归，覆盖 epoch reseed 开关。
直接实验显示 HF 原生 shuffle buffer 不保存完整内容，恢复序列可不同；
cursor docstring 明确 exact resume 依赖 source state 覆盖，不承诺任意
外部 HF shuffle graph 精确恢复。仓库 source builder 不调用 HF shuffle，
composition 在分片后使用 Grain checkpointable window shuffle。
pipeline 261 passed，Ruff/format、编译及 diff check 通过。

流式完整 shuffle graph 恢复验证：真实 HF IterableDataset 外包
Grain WindowShuffleIterDataset，读取非窗口边界后保存、继续读取并创建
独立恢复 iterator，后续 25 行跨 repeat epoch 完全一致。验证 source
cursor 与 shuffle buffer 组合状态，不使用假 source 或预生成恢复答案。
完整 CPU 2562 passed、2 skipped；全仓 pre-commit（排除保护示例）、
compileall、diff check 通过。网络 HF 数据、worker 并发和硬件训练不由
此本地生成器测试证明。

模型 projection 构件说明整理：feed_forward.py/activation.py 的顶层
迁移背景缩为当前职责、torchtitan 来源、interleaved w13 与 logical w1/w3
checkpoint 契约及 TP 子类复用关系。检查 checkpoint load hook 的先准备
后改键流程和 activation interface，本轮不修改算术或 hooks。common/TP
projection 回归 115 passed；相关 Ruff/format、compileall、diff check
通过。此轮 CPU 回归不覆盖 CUDA fused GEMM 数值。

MoE router geometry 边界修复：TokenChoiceTopKRouter 在 gate 分配前统一
检查 num_experts/dim/top_k 与可选 group 度数为正整数，拒绝 bool、float
和 string；原大小比较会放过 top_k=True/1.5，使错误延迟到 forward。
保留上界、整除、group capacity 及可选 None 契约。新增五字段四非法
类型回归；routing 53 passed，Ruff/format、compileall、diff check 通过。
此轮不改选择算法或 collective；未验证 GPU router 数值。

Quantile histogram geometry 复核：QuantileBalancer 独立入口在分配
histogram 前拒绝非整数、bool 和非正 num_experts/top_k/num_bins；保留
0 < top_k < num_experts。基础 router 删除已由统一整数检查覆盖的两个
冗余 group 正值检查。新增三字段五非法值回归，routing 68 passed；相关
Ruff/format、编译、diff check 通过。不改变 cutoff、histogram 或 bias
估算算法。

Router 构造参数表达优化：基础 router 的 gate 构造、quantile router
父类构造和 histogram balancer 构造使用显式 geometry 关键字，避免
(dim,num_experts) 与 (num_experts,dim) 在相邻接口间误读。Gate bias 改为
keyword-only，现有全部消费者无需位置布尔值迁移；维度公开顺序保持。
不改 fp32 GEMM、梯度或路由逻辑。routing/common 联合回归 149 passed；
相关 Ruff/format、编译及 diff check 通过。

EP 单块 router 装配合并：block_conversion 将两种 router 共用的
geometry、route norm/scale 与 aux loss 准备集中为 router_options，删除
只为分支重复构造服务的 build_aux_loss 薄包装。quantile 分支保留 sigmoid
及 group 限制和 sign bias 禁用，普通分支保留自身 score/group 选项。
weight copy、冻结状态、训练模式和 shared expert 复用顺序不变。
replacement/routing 109 passed；相关 Ruff/format、编译和 diff check
通过，旧 wrapper 无代码引用。未新增硬件 EP 数值验证。

EP 转换中断清理修复：model_replacement 的候选构建异常回滚覆盖
BaseException，使 KeyboardInterrupt 等中断同样恢复 AuxLoss group counts，
然后原样抛出。新增第一块成功构建含 aux loss、第二块中断回归，检查
全局计数恢复和原 block identity 保持。replacement 42 passed；相关
Ruff/format、编译、diff check 通过。不改变成功安装流程。

EP rank 查询依赖整理：block_conversion 从 accelerator.distributed
直接导入 get_rank/get_world_size，去掉包级 distributed 模块访问，保持
None group 的 local fallback。replacement 42 passed；真实双进程
CPU/Gloo ep_equivalence 通过，最大跨 rank 相对误差 3.271e-7。相关
Ruff/format、compileall、diff check 通过。该脚本不覆盖 GPU/NPU、TorchAO
或完整并行矩阵。

MoE token 统计重复计算整理：layer._forward_tokens 无 padding 时复用
dispatch 已计算的 num_local_tokens_per_expert_E，去掉第二次相同 map.sum；
有 padding 时单独计算 valid_counts_E 后加入统计 buffer。dispatch 输入
仍为完整 token stream counts，eval 不更新 buffer，bias/recompute 语义
保持。padding/bias/routing 101 passed；相关 Ruff/format、compileall、
diff check 通过。不宣称此局部减少求和已带来经过测量的吞吐提升。

MoE 全局专家数契约修复：构造时校验 num_experts 为正整数且与
router.num_experts 一致，避免统计 buffer 与 routing map 形状冲突延迟
到 forward。EP 本地 expert 数不参与该全局比较。新增非法类型与错配
回归，FP32 buffer fixture 改为真实 router 而非不能执行路由的 Linear。
联合 padding/bias/replacement 81 passed；相关 Ruff/format 与 diff 检查
通过，不改一致组件的前向语义。

MoE balancing 系数类型修复：load_balance_coeff 检查 Real 并显式拒绝
bool，再检查有限正值；避免 True 被作为 1 接受、字符串在 math.isfinite
内抛出模糊 TypeError。None 关闭与 quantile 互斥规则保持。扩展现有
coefficient 边界回归。近期累计完整 CPU 2609 passed、2 skipped；全仓
pre-commit（排除保护示例）、compileall 和 diff check 通过。硬件数值
覆盖仍以已运行 integration script 的范围为准。

GroupedExperts 初始化契约复核：逐字段检查 dim/hidden_dim/num_experts
为正整数，替代仅 min 比较；在 capability probe 与参数分配前拒绝非法
类型。use_grouped_mm 明确只允许 bool/None，避免字符串 truthiness
意外开启后端。新增维度与 backend override 回归，common 96 passed；
Ruff/format、编译、diff check 通过。既有合法自动探测与显式后端行为保持。

专家 GEMM/dispatch 边界复核：检查累计 segment offsets、空 token
autograd 路径与 RoutedExperts metadata 生命周期；精简 _grouped_mm
长说明，直接表达 contiguous segments、BF16 fused gate、其他 dtype
fallback 与空输入梯度契约。执行代码未改变。common/token dispatch
127 passed；Ruff/format、compileall、diff check 通过。CPU/fake 后端
回归不作为真实 CUDA grouped GEMM 证据。

Domino slice 结构复核：确认使用 RoutedExperts.forward 共享通信/计算
主流程，每个切片独立 metadata；boundaries 改为实际单一切分点 split_at，
切片张量按 input/scores/expert_ids/counts 及维度命名。不增加包装或改变
分片算法。真实双进程 CPU/Gloo Domino equivalence 的普通/Domino ×
zero receive × checkpoint recompute 共 8 组前向与梯度匹配；Ruff/format、
compileall、diff check 通过。未验证 accelerator stream overlap。

Grain loader 生命周期说明整理：dataloader.py 顶层说明改为当前
batch/collation/thread prefetch/单一游标职责，去掉旧 loader.py 路径与
迁移背景。静态核对 restore 的版本、DP degree、batch layout 校验及
set_state 失败关闭且保留原异常的流程。本轮仅 docstring 调整，执行逻辑
不变；相关 Ruff/format、compileall、diff check 通过。

Collator 公共接口复核：检查 TextSequence 等长字段校验、文本预算
拒绝、padding labels masking 及 PP 有限尾批全 masked row 补齐。
collators.__all__ 补齐 loader builder 实际消费的 PipelineTextCollator；
TextCollator 说明明确 pinning 按设备能力选择，不声称 CPU 页锁定。
本轮仅导出声明与说明调整，执行逻辑不变；Ruff/format、diff check 通过。

Packing recipe 参数命名统一：两个 text packing builder、multimodal
packing builder 与 loader 的 PackingBuilder protocol 首参改为 recipe，
与下方 dataset_graph 区分声明与已构建图。仓库消费者均按位置传 recipe，
无旧关键字调用或兼容包装。text pipeline 262 passed、multimodal 149
passed；相关 Ruff/format、compileall 和 diff check 通过。packing
算法、seed 与 row layout 不变。

多模态 packing 转换参数边界：packing_output_to_multimodal_sample 在
读取字段与构造 padding positions 前复用 require_positive_integer 检查
max_context_length，拒绝零、负值、bool 和非整数，避免独立调用进入
无效 NumPy modulo。正常 builder 的 DatasetBuildContext 原有检查保持。
新增六种非法参数回归；multimodal 155 passed，Ruff/format、compileall
及 diff check 通过。合法 token/media 布局与输入复制策略不变。

Text packing 转换参数契约同步：两个 text conversion 在读取输入前
复用正整数 context length 校验，与 multimodal conversion 一致，避免
零 modulo 或非法位置生成。修正缺省 positions 说明为按 context wrap。
新增两方向六非法长度回归；text pipeline 274 passed，相关 Ruff/format、
compileall 和 diff check 通过。合法 packing 布局与 canonical positions
算法保持。

Packing checkpoint 中断回滚复核：iterators.py 的 set_state 在重放旧
checkpoint 的 remainder 时，KeyboardInterrupt 原先绕过 Exception 回滚，
导致父游标与保留的当前 remainder 不一致。以恢复较早 checkpoint、重放
推进父游标后中断的测试复现，修复为 BaseException 回滚并继续传播原异常。
回归同时比较父游标和后续所有 token 字段。正常切分、文档边界与 padding
逻辑未改动；text/multimodal 430 passed，完整 CPU 2643 passed、2 skipped。
pre-commit（按既定约束排除 examples/native_alltoall.py）、compileall 与
HEAD diff check 通过。本轮未执行新的 GPU/NPU 或混合并行数值验证。

Loader 配方命名与入口可读性整理：loader_builder.py 的私有
ResolvedRecipe 字段 dataset 改为 recipe，与 packing builder 的 recipe
参数一致，明确区分配方声明与已构建 Grain graph。同步两个 resolver 与
packing 调用；仓库搜索无旧字段消费者。精简解析器说明和装配注释，明确
media whole-document packing 与 ordering/sharding 的归属，不改算法或
懒加载位置。完整 datasets CPU 438 passed，Ruff/format、pre-commit
（排除受保护 example）、compileall 与 HEAD diff check 通过。

PP collator padding 规则合并：PipelineTextCollator 的有限尾批
补齐改为构造空 TextSequence 并复用 TextCollator，删除重复的张量初始化、
labels/mask 设置和 positions wrap，去掉只用于重复位置计算的 context
属性。每个补齐批次独立分配，固定 schedule 形状与有效 token 统计保持。
现有尾批测试扩展 context 3/16，断言零 input 与按 context wrap 的 positions。
完整 CPU 2644 passed、2 skipped；pre-commit（排除受保护 example）、
compileall 与 HEAD diff check 通过。未新增 PP 分布式数值验证。

多模态 patch 维度契约统一：MultimodalCollator 与
MultimodalProcessor 构造器复用 require_positive_integer，替代 min 比较。
原检查会接受 bool、正浮点并对字符串/None 抛出无字段上下文的 TypeError；
现在三个 patch 维度均在构造时拒绝非法类型与非正整数。原零尺寸回归扩展为
三个字段 × 六种非法值，覆盖两个构造入口。合法维度与图像处理算法保持。
multimodal CPU 170 passed；相关 Ruff/format、compileall 与 HEAD diff
check 通过。未新增硬件数值验证。

多模态媒体汇总简化：collator.py 先按样本顺序汇总 image/video
列表，预算计数直接复用这两个列表，移除仅用于求和的 images_per_sample
和重复媒体遍历。预算仍为 image 数量加 video 时间 patch 分组数，预算
检查仍在 patch 计算前执行，各类媒体内部顺序保持。复用现有媒体预算、
patch/grid 和 video 测试，multimodal CPU 170 passed；pre-commit
（排除受保护 example）、compileall 与 HEAD diff check 通过。

MultimodalCollator 媒体预算契约补齐：max_images_per_batch
要求非负整数，拒绝 bool、浮点、字符串、None 与负数，避免隐式比较接受
非整数预算或产生底层 TypeError。保留零预算纯文本批次，新增五种非法值
与零预算文本回归。multimodal CPU 176 passed；Ruff/format、compileall、
HEAD diff check 和 pre-commit（排除受保护 example）通过。

MultimodalCollator 只读输入契约整理：collate_text 与
collate_media 参数改为 Sequence，与 __call__ 的序列输入一致；内部只读取、
遍历和索引，无列表修改，移除 __call__ 多余的 list(batch) 复制。媒体汇总
仍生成独立列表，输出拼接不修改源张量。multimodal CPU 176 passed；相关
Ruff/format、compileall 与 HEAD diff check 通过。

多模态处理器参数命名统一：MultimodalProcessor 的
`sample_processor` 改为 `process_sample`，更准确表示传入的是样本处理
函数而不是处理结果；内部 `_process_sample`、三个 multimodal recipe 和
所有测试同步更新，仓库搜索无旧名称。函数签名、懒加载和样本过滤行为不变。
multimodal processor/build 测试 182 passed；Ruff/format、compileall、
HEAD diff check 和 pre-commit（排除受保护 example）通过。

Parallel 入口契约与职责索引复核：核对 apply_fsdp 的 dense/sparse
mesh 判定，修正 parallel/__init__.py 中“所有 degree=1 均为 no-op”的
错误概括。DP shard=1 时 CP、replication 或 expert sharding 仍可能需要
FSDP，不能仅凭 DP 度数跳过装配阶段。README 补齐 stages、attention_heads、
model_compile、token_dispatch 与 torchao_dispatch 职责，明确实现从定义
模块导入、公共包保持懒加载。本轮仅说明修改；相关 Ruff/format 和 HEAD
diff check 通过，不作为新增并行数值验证。

ParallelTopology 内部缓存字段收敛：single/multi axis mesh 与
world mesh 字段设置 init=False、repr=False、compare=False，构造接口
仅接受拓扑几何参数，缓存初始化、查询与 mesh 构造流程保持。仓库无通过
构造参数注入私有缓存的调用。避免内部 mesh 状态参与拓扑值比较和对象显示。
拓扑与装配顺序 CPU 测试 68 passed；定向 pre-commit、compileall 与
HEAD diff check 通过。本轮未改 collective 或 rank 布局。

Topology mesh 查询边界复核：get_optional_mesh 在初始化进程
mesh 前拒绝空轴列表与重复轴列表，避免无效查询进入 collective 初始化或
DeviceMesh 底层切片；合法查询与缓存键顺序保持。新增三个非法轴列表回归，
同时断言 world mesh 尚未创建。topology CPU 63 passed；定向 pre-commit、
compileall 与 HEAD diff check 通过。本轮未改 rank 排列或通信算法。

Topology mesh 构建失败生命周期修复：world/global/single-axis mesh
先在局部变量中完成构建与大小校验，成功后再发布对象缓存。原实现提前设置
world mesh，后续失败后的重试会返回未完成状态。新增注入校验失败、检查缓存
为空、随后成功重试与复用同一对象的回归。topology CPU 64 passed；定向
pre-commit、compileall 与 HEAD diff check 通过。不销毁 PyTorch 内部已创建
的进程组，不声称失败后释放全部 c10d 资源；正常 rank 布局保持。

Topology 状态定义补齐：global mesh 视图注册表声明为内部 dataclass
字段，与 single/multi-axis 缓存采用一致的 init/repr/compare 排除规则。
避免构造后才出现隐式属性，使初始化与失败状态完整可读。补齐 post_init、
几何校验与局部 unflatten 的返回类型；不改变构建顺序或 mesh 切片。
topology CPU 64 passed，定向 pre-commit、compileall 与 HEAD diff check
通过。工作区同时存在其它配置与并行入口变更，本轮未覆盖或提交这些改动。

并行配置重组集成审查（当前未通过）：运行整个 CPU parallel 目录时，
18 个测试模块在收集阶段失败，共同原因是 parallel_config.py 导入已不存在
的 validate_context_parallel_config。复核 CPPlanConfig.__post_init__ 还发现
它读取 context_parallel_kcp_size、tensor_parallel_size、KV/Ulysses degree
等不属于 plan 的字段。应将 plan 自身策略校验与依赖 ParallelConfig 度数的
组合校验分开；配置文件当前正在被同时修改，本轮记录实际失败证据，避免
恢复旧文件覆盖在途重组。这次执行没有产生并行测试通过结果，前轮局部通过
不能代表当前工作区已完成集成验证。

EP checkpoint 接口整理：checkpointing.py 从 accelerator.distributed
定义模块直接导入 rank/world 查询；状态条目筛选的 shard_map、params 参数
改为 keyword-only，save/load 调用明确标注参数，测试 mock 同步。目标文件
定向 pre-commit 与编译通过。实际 pytest 仍被配置重组的缺失 CP 接口阻断，
未获得本轮运行时通过证据；全仓 diff check 另发现 CP 配置的尾随空白。
未覆盖在途配置变更，后续需重跑 EP checkpoint 与整个 parallel 测试。

配置重组恢复后的集成复核：EP checkpoint 定向测试恢复执行并通过
15 cases，验证前轮直接导入与关键字参数整理。整个 parallel CPU 目录当前
67 failed、322 passed、5 skipped；配置已合并回 parallel_config.py，但
ParallelConfig 定义 tp_size/cp_size 等属性，内部校验与运行时仍访问 tp/cp
等旧属性，且组合校验直接读取移入 cp_config/ep_config 的字段。此为当前
实际接口不一致，需整体同步命名与嵌套字段消费者，不能靠本轮 checkpoint
通过推断并行装配正确。配置继续变化，保留现场并等待文件修改归属说明。

ParallelConfig 自身接口修复：组合校验读取 cp_config/ep_config 策略，
度数检查使用明确 tensor/expert_parallel_size，non_dp_sizes 使用 cp_size，
消除配置内部对已移除 tp/cp/ep 属性的访问。默认构造与 DP 推导恢复；实际
执行默认/TP DP 推导、KCP 缺省策略拒绝、KCP+TP 拒绝检查通过。定向
pre-commit 自动移除两个无用导入后重跑通过。现有 plan_config 测试仍为
7 failed、11 passed，失败来自旧构造关键字和 LLMTunerConfig 的旧属性
消费者，需继续迁移；未恢复旧名别名或声称整体配置已验证。

并行配置度数接口迁移复核（当前未完成）：重新读取目标、指南和当前源码清单。
同步 TP、EP、CP、FSDP plan、parallelize 与 topology.from_config 的旧配置
度数访问，使用 tp_size/cp_size/ep_size/pp_size；保留 ParallelTopology 自身
轴字段。配置契约测试改为拒绝旧扁平关键字、校验 *_config 类型，不恢复别名。
当前组合执行 385 passed、26 failed、2 skipped：多数失败来自测试仍使用旧
构造字段；两个 HF TP plan 测试失败另指向安装版 Transformers 提供
base_model_tp_plan 而包装器仅读取 tp_config 的适配缺口，尚待修复验证。
这些结果不代表并行数值验证或全量审查完成。

并行嵌套配置消费者迁移续审：TP/EP/CP/FSDP/PP CPU 批次 411 passed、
2 skipped。修复 loader 的 pp_config.num_pp_microbatches 读取，同步 dataset、
MoE dispatcher、配置和 validation 测试的嵌套配置契约。HF 包装器与 TP plan
解析支持安装版 Transformers 的 _tp_plan/tp_plan；真实 Llama/Qwen3 匹配
回归恢复通过，新增两种接口的独立回归。全套最新运行 2687 passed、3 failed、
2 skipped，最后三项为配置测试的旧属性断言，随后同步；全套仍需重跑。
compileall 与 HEAD diff check 通过，尚未完成全局质量门禁及硬件验证。

并行配置迁移全仓门禁与数值复核：同步 16 个集成脚本的旧扁平策略构造，
保持 CLI 平面参数契约；parallel_plans 文档删除旧名代理承诺，明确嵌套字段
与 degree accessor。完整 pytest 2690 passed、2 skipped；compileall 与
HEAD diff check 通过。pre-commit --all-files 被用户保护的
examples/native_alltoall.py 既存 E501 阻断，其 hook 自动改动已原样撤回；
排除此文件后的所有跟踪文件质量门禁通过。
双 rank CPU/Gloo ep_wiring_equivalence 实际通过：EP=2 vs HF 最大绝对
误差 2.638e-08、EP=2 vs EP=1 为 0、失败 rank 为 0。启动需 PYTHONPATH=.
与显式 127.0.0.1/lo0；默认 rendezvous 遇本机地址解析失败。编译路径另遇
本机 C++ PCH 校验错误，最终 TORCHDYNAMO_DISABLE=1 验证 eager 语义，
不能据此宣称 compile 或 GPU/NPU 并行组合通过。全局逐文件审查继续。

Checkpoint 生命周期续审：本轮读取 filesystem.py、storage.py、base.py、
checkpoint_keys.py、model_state.py、torch_checkpointing_backend.py，并核对
DCP staging/save wait 与发现策略。对照本地 torchtitan/components/checkpointer/
torch_checkpointing.py:510，发现两者 timeout 后均丢失仍在执行的保存 future。
本仓库修复 _wait_for_saving：未完成 future 在 finally 恢复，已完成成功或失败
保持消费一次；避免下一次保存绕过正在写入的 checkpoint。回归覆盖 pending
超时后再次等待、成功、失败，checkpointer CPU 78 passed；定向 pre-commit、
格式、compileall、HEAD diff check 通过。此测试使用真实 concurrent Future，
无需可选后端，但未验证真实 torch_checkpointing IO 或 GPU staging。

DCP 等待中断与 trainer 清理续审：读取 datasets/dataloader.py、Trainer.close
及 DCP 构造失败清理、staging/save 等待。DCP 两个等待阶段原先在 result 前
清空 future，KeyboardInterrupt 导致仍在执行的操作失去跟踪。同步可选后端
的活动 future 保留契约，finally 恢复未完成 staging/save；已完成失败仍消费
一次。新增两阶段中断后重等回归。Grain 恢复失败关闭迭代器并保留原异常，
Trainer 逐资源清理并保留首个异常的现有实现本轮未改动。

学习率调度运行边界续审：读取 optimizer/lr_scheduler.py、ema.py 和
config/lr_scheduler_config.py，核对恢复重算、optimizer-before-scheduler、EMA
固定 identity 与全局 step 老化契约。发现 scheduler load 拒绝超出 total_steps，
但 step 可继续推进并进入负线性学习率或余弦回升区间。容器在修改任何内层
scheduler 前拒绝超过终点，保持正常最后一次更新及其虚拟终点计算。新增
无衰减/完整衰减的超限拒绝与状态不变回归，scheduler 43 passed，定向
pre-commit 通过。EMA 本轮仅阅读，未新增修改或宣称硬件验证。

随机数与通用工具续审：读取 utils/seeding.py、garbage_collection.py、
dtypes.py、monitoring.py，核对配置与 Trainer.seed_everything 消费者。发现
detect_anomaly=False 仅跳过设置，前一次同进程训练开启的 anomaly 模式会
遗留到下一次，配置关闭无效且继续承担栈追踪开销。seed_everything 现在显式
设置请求值，保持 check_nan=False 的 DTensor 适配。新增 true→false 连续
配置回归并在 finally 恢复测试前的 anomaly flags。工具与 PP 种子测试通过，
具体计数见本轮输出；其余读取文件未作外观重命名或声称硬件验证。

公共模型构件与累计集成验证：本轮逐文件阅读 models/common/activation.py、
embedding.py、rope.py、feed_forward.py、cast_linear.py，以及 utils/module_names.py、
lazy_exports.py。核对词表 offset/padding/空 shard、RoPE 位置 bounds 与两种
旋转布局、FFN interleaved 保存恢复及 CastLinear identity/autocast 契约。
现有相关测试 201 passed，本轮未发现有证据的新行为缺陷，未做外观改名。
累计全套初次执行发现 Trainer anomaly 测试仍要求 false 不设置；按上一轮
显式应用 false 的新契约同步断言，重跑完整 pytest 2698 passed、2 skipped。
所有跟踪文件（排除既有保护文件 examples/native_alltoall.py）pre-commit 通过；
补改 Trainer 测试定向门禁通过，compileall 与 HEAD diff check 通过。
这些是 CPU 证据，不覆盖 compile/GPU/NPU 或全并行组合数值验证。

HF 装配辅助与审查清单续核：本轮完整读取 models/hf/attention_backend.py、
factory.py、materialization.py、model_setup.py、flops.py，核对构建入口、
composite text config 传播、设备 backend 选择、meta 非持久 buffer 恢复及
共享参数身份。当前未发现需要修改的明确行为缺陷，沿用针对性测试验证。
当前源码共 153 个 Python 文件，路径与清单核对无漏项；清单删除五个已合并
到 parallel_config.py 的旧 family config 条目，历史迁移日志保留。修正 RoPE
条目已删除 helper 的名字。路径覆盖只证明清单完整，不能等同每条契约或
硬件数值验证完成。

Accelerator launcher 契约续审：读取 tensor_tree.py、collectives.py，核对
嵌套 output 预验证、通信设备搬运及 TP/EP/PP norm 分组；复查 distributed.py
bootstrap 分支，发现 vendor 设备 init_backend 非 torch 被静默忽略，CPU 的
错误也晚于环境解析。PyTorch/Slurm 入口在任何设备和 rank 环境操作前拒绝
非 CUDA 的非 torch engine；CUDA 既有第三方引擎路径保留。新增五设备、
三引擎、两 launcher 的拒绝回归，确认没有 device/PG/subprocess 副作用。
这是 CPU 模拟入口验证，不代表 vendor collective 或第三方引擎数值测试。

Profiler 公共构造入口续审：完整读取 components/profiler.py，检查 trace
退出异常保留、OOM snapshot 与 allocator history 关闭。MemoryProfiler 可直接
构造，原先 freq=0 先启动全局 allocator history，随后 step 才除零失败；非法
step/rank/max_entries 同样没有公共入口保护。现在先校验四项整数范围，失败
不启动 history，避免构造非法实例的全局副作用。新增六项边界回归验证未调用
record_memory_history；实际 GPU allocator 仍待验证。

Compile 与重计算证据续核：本轮读取 parallel/stages.py、model_compile.py、
module_patterns.py、memory/remat_regions.py 及 CompileConfig；核对阶段顺序、
regional annotation 作用域和 projection region 的适配边界。实际执行 CPU
整模型与逐层 aot_eager 输出/梯度等价，两项均通过；compile+memory 批次
81 passed、2 skipped。该证据不覆盖 Inductor、异步 TP、torch_remat 真实引擎
或硬件混合并行。清单当前 153 行/153 独立路径与 153 源文件完全匹配，修正
首页遗留 156 统计；不把路径匹配当作数值验证。已询问可用硬件作业入口，
本地可验证工作继续，硬件要求保持未完成。

MoE 辅助梯度与统计续审：读取 models/common/moe/auxiliary_loss.py、
bias_update.py、gate.py 与 common/aux_loss.py，核对 per-token 归一化、
CP/TP partial→invariant 归约、padding 与重计算指标抑制；本轮未修改实现。
实际双 rank CPU/Gloo moe_aux_loss_grad_equivalence 通过（显式 loopback）：
CP=2 router gradient 最大绝对误差 1.192e-07，参考幅度 9.311e-01，失败
rank=0；直接 backward identity 检查通过。该脚本验证 CP token partial
auxiliary reduction，不包含实际 CP attention kernel 或 EP dispatch 梯度。
辅助 loss、bias update、padding 专项 pytest 结果见本轮输出。

Domino 空 rank 分布式缺陷复核：阅读 TorchAO dispatch 的可选导入与 padding
契约，运行既有 EP Domino 八项 Gloo 等价通过后，扩展 empty-rank 场景。
实际复现 Domino local T=0 触发“at least two tokens”，对端已进入 collective。
移除 rank-local token 下限，始终执行两切片交换（允许空切片），保证参与
顺序一致。扩展 16 项双 rank Gloo 场景全部通过，涵盖普通/Domino、零接收、
checkpoint 重计算与空 token rank；输出、输入/score/expert 参数梯度与参考
在 1e-9 容差内一致。此为真实 CPU collective 验证，不含 accelerator stream
overlap 或 TorchAO/MC2 内核。审查新增场景曾失败，修复后重跑通过。

Domino 调用方与支持声明同步：全仓检索 Domino 下限与配置消费者，没有
遗留 T>=2 限制。distributed_implementation_comparison 文档原先同页同时
宣称 KCP/Domino/custom FSDP 未实现与已迁移，且把未经设备验证的路径标为
完整；按当前入口修正为接线/CPU数值/目标硬件待验证，明确 MC2 早期拒绝、
fixed router debug-only 与 Domino stream overlap 缺口。修正双 CP axis
可以组合的字段说明。文档 pre-commit 与 HEAD diff check 通过。

Domino 修复边界与累计门禁：ep_domino_equivalence 增加单 token rank，
覆盖 None/0/1 本地 token 分配与两类 routed module、零接收、重计算组合。
实际双 rank CPU/Gloo 24 项输出及 input/score/expert gradients 全部与完整
参考一致，容差 1e-9；空/单 token 切片均参与两次 exchange。累计完整 pytest
2734 passed、2 skipped；全部跟踪文件 pre-commit（保护 native_alltoall.py
并排除）、compileall、HEAD diff check 通过。GPU/NPU 与真实可选内核仍
待设备入口，目标未完成。

并行兼容性矩阵入口复核：

- 修正 chunked loss、EP HF 初始加载和两项 TP MoE 规则的实际调用入口；
  修正配置文件路径和 router bias 拒绝理由，保持现有不支持行为。
- 原测试仅检查入口文件存在，无法发现函数迁移。改用 AST 检查限定函数
  存在且调用对应 verdict，避免导入目标模块触发可选依赖。
- 验证：兼容性定向测试 6 passed；受影响文件 Ruff 和 git diff --check
  通过。此次验证覆盖规则元数据一致性，不证明并行运行时数值正确。

TP CPU 分布式与当前门禁复核：

- PyTorch 2.14.0，macOS CPU/Gloo 两进程；使用 loopback rendezvous，
  TORCHDYNAMO_DISABLE=1。运行 tp_equivalence.py：14 个投影替换，
  logits 最大绝对误差 2.980e-08，权重梯度误差 1.937e-07，失败 rank 0。
  此结果不覆盖 CUDA 融合通信、Inductor 或 FSDP/TP 混合路径。
- 当前 pytest -q --disable-warnings：2734 passed、2 skipped、135 warnings。
  compileall 与 git diff HEAD --check 通过。tracked 文件 pre-commit
  首轮自动修正 AST 测试的 UP038，复核改动后第二轮全部通过；遵守既有
  examples/native_alltoall.py 保护例外，不将它纳入本次门禁。
- 当前设备探测：CUDA 不可用、NPU 接口不存在、MPS 可用；GPU/NPU
  集成验证仍待目标环境。训练与验证清理路径复核暂未发现新增确定缺陷。

PP 调度、恢复与组合运行时复核：

- PyTorch 2.14.0，CPU/Gloo，loopback rendezvous，
  GLOO_SOCKET_IFNAME=lo0、TORCHDYNAMO_DISABLE=1。
- run_all.py --master-port 29645 pp_tp_trainer pp_validation_equivalence
  pp_checkpoint_equivalence：3 passed。PP checkpoint 两个独立进程阶段
  从 step 2 恢复后，后续两步 loss 与连续训练绝对误差为 0；
  PP=2、TP=2 四 rank 训练步检查 loss 跨 rank 一致及关闭后进程组可用；
  PP 验证检查有限 loss 和模式恢复，尚无单进程参考的数值比较。
- 两 rank pp_equivalence.py Interleaved1F1B 与 ZBVZeroBubble，
  master port 29646：各训练 4 步、每步 4 microbatch，与完整模型参考
  比较 loss 及梯度范数；最大误差分别 0 和 4.768e-07，容差 1e-05，
  两者失败 rank 均为 0。覆盖虚拟 stage 与 V 布局，不覆盖目标设备内核。

PP 验证数值参考补强：

- pp_validation_equivalence.py 增加完整模型参考：汇集实际 stage 权重、
  strict 加载完整模型，独立计算相同 synthetic 语料及 microbatch 边界
  的 summed CE / valid tokens；初始化参考模型时隔离 RNG。每次验证
  从 1 批扩展到 2 批，验证后续训练与 eval 模式恢复。
- 修正结果输出由最后 stage rank 打印，并显式销毁测试进程组。
- PyTorch 2.14.0，CPU/Gloo 两 rank，loopback master port 29647，
  TORCHDYNAMO_DISABLE=1：两次验证最大误差 4.922e-07，容差
  1e-05，全部通过；定向 pre-commit 和 diff 检查通过。
  覆盖 PP eval 汇总归一化，仍不覆盖 DP/CP/TP eval 组合或 GPU/NPU。

训练指标输入边界复核：`flops_per_token` 与其 attention 辅助函数现在拒绝
负数、布尔值和非整数序列长度；窗口大小必须为正整数。`MetricsProcessor`
同样拒绝负数、布尔值和非整数 `num_flops_per_token`，避免独立调用产生负
TFLOPs 或 MFU。训练配置原有的 `max_seq_len >= 1` 校验保持不变。新增边界
回归后完整 CPU 测试 2267 passed、2 skipped；Ruff、format、compileall 和
diff 检查通过。
