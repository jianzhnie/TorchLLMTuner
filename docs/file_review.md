# 逐文件代码审查

范围：`llmtuner/**/*.py`。状态仅代表实现阅读，不代表全部硬件组合已验证。静态检查不会自动将文件标为已审查。

2026-10-15 最新清单核对：逐文件表格曾与当时 `llmtuner/**/*.py` 集合一致，
后续新增文件的当前核对见 REVIEW-MANIFEST-CP-01。清单覆盖不代表全部
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
| `llmtuner/components/checkpointer/checkpoint_writer.py` | 已阅读；CPU/DCP 数值验证 | GenerationFileSystemWriter；独立 generation shard，本地/file:// 原子 metadata 替换，对象存储整对象发布；memory/file 的同步/线程故障恢复已验证，S3/GCS 待实测；不覆盖断电持久性或并发写入排他。 |
| `llmtuner/utils/module_names.py` | 已阅读 | canonical_fqn；任意深度完整 wrapper 段移除，无 torch 依赖。 |
| `llmtuner/components/checkpointer/distributed_checkpoint.py` | 已阅读 | DistributedCheckpointAsyncMode、FilesystemCheckpointStorage、DistributedCheckpointManager；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/checkpointer/filesystem.py` | 已阅读 | is_remote、resolve、exists、isdir、isfile；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/checkpointer/hf_storage.py` | 已阅读；CPU/DCP 数值验证 | HF generation 导出、索引发布、按索引选取具体 tensor；协调跨 rank 合并/发布失败。local CPU native/custom FSDP 故障恢复精确一致；远程 HF 不支持。 |
| `llmtuner/components/checkpointer/load_planner.py` | 已阅读 | CheckpointLoadPlanner；严格 DCP 加载、旧 RNG/loader metadata 与 compiled 键迁移；flat byte 写回和名称冲突检查，独立模块。 |
| `llmtuner/components/checkpointer/model_state.py` | 已阅读 | ModelCheckpointState 聚合 PP/EP 状态并维护缓存；支持非张量 extra_state，按 shape/dtype/device 刷新张量缓存。 |
| `llmtuner/components/checkpointer/random_state.py` | 已阅读；CPU/DCP 数值验证 | rank 独立 RNG 保存恢复，版本、拓扑及设备校验；实际 CPU dropout 续训和旧 checkpoint 兼容已验证，设备随机流待验证。 |
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
| `llmtuner/models/common/moe/experts.py` | 已阅读，异常协议未验收 | GroupedExperts、RoutedExperts；直接复用 capability probe；检查分段 counts 契约、前向/反向、状态转换和模型兼容性边界。 DOMINO-DIRECT-FSDP-01验证单模块直接hook的延迟计算异常；DOMINO-FAILURE-01其余边界仍开放。 |
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
| `llmtuner/models/kimi/latent_attention.py` | 已阅读，迁移未验收 | 上游 MLA 投影和 HF attention 接口，修正已展开 K/V 的 group 数；CPU 独立公式与梯度通过，模型工厂尚未接通，关联 KCP-MODEL-01。 |
| `llmtuner/models/kimi/normalization.py` | 已阅读，迁移未验收 | Kimi RMSNorm，保留 FP32 归约和上游参数名；目标设备融合路径待适配，关联 KCP-MODEL-01。 |
| `llmtuner/models/kimi/causal_convolution.py` | 已阅读，迁移未验收 | KDA 因果 depthwise 卷积训练构件，独立逐 token 数值/梯度及 packed 重置通过；KDA 模型消费者和设备内核尚未接通，关联 KCP-MODEL-01。 |
| `llmtuner/models/kimi/delta_attention.py` | 已阅读，迁移未验收 | KDA 投影/卷积/gate/归一化训练构件；CPU 递推与上游 WY 输出及梯度对照通过；packed/padding 边界回归通过，工厂及目标设备未验收，关联 KCP-MODEL-01。 |
| `llmtuner/models/kimi/model.py` | 已阅读，迁移未验收 | dense decoder/causalLM 与 HF wrapper 离线构造、前后向及严格模型 state 往返通过；MoE/attn residual、Trainer 完整恢复和并行组合未验收，关联 KCP-MODEL-01。 |
| `llmtuner/models/kimi/moe.py` | 已阅读，迁移未验收 | 可求导本地 Kimi MoE、原始 router/专家键；独立全部专家枚举输出/梯度通过，EP 转换及目标设备未接通，关联 KCP-MODEL-01。 |
| `llmtuner/parallel/expert_parallel/kimi_conversion.py` | 已阅读，迁移未验收 | KIMI-EP-01：ModuleList专家转EP切片，latent/shared生命周期与correction bias参数身份；CPU双rank数值和native Trainer恢复通过，HF映射/其他后端待完成。 |
| `llmtuner/components/checkpointer/kimi_layout.py` | 已阅读，迁移未验收 | KIMI-EP-DCP-01：源布局metadata模板、完整CPU读取、模型/optimizer/EMA转换，单进程真实DCP通过，多rank跨EP待验收。 |
| `llmtuner/models/kimi/checkpoint.py` | 已阅读，迁移未验收 | KIMI-HF-STATE-01：完整grouped与HF逐专家权重映射；CPU单测/EP2通信往返通过，Trainer初始HF加载待验收。 |
| `llmtuner/models/kimi/attention_residual.py` | 已阅读，迁移未验收 | 上游 CPU attention residual 聚合，独立 token 枚举输出/全部梯度通过；decoder block 生命周期尚未接通，关联 KCP-MODEL-01。 |
| `llmtuner/models/kimi/__init__.py` | 已阅读，迁移未验收 | Kimi 架构命名空间；不进行全局注册或导入可选内核，关联 KCP-MODEL-01。 |
| `llmtuner/models/kimi/kimi_linear_config.py` | 已阅读，迁移未验收 | 迁移 vLLM 固定版本配置；嵌套字段自持有，配置持久化回归通过；模型及公开工厂接入待完成，关联 KCP-MODEL-01。 |
| `llmtuner/models/hf/factory.py` | 已阅读 | HF 配置构建与模型类解析；FLOPs 入口归 flops.py，meta 存储物化归 materialization.py。 |
| `llmtuner/models/hf/materialization.py` | 已阅读 | materialize_meta_model；从 factory.py 搬入，负责 meta 存储分配、RoPE buffer 重建和共享参数/buffer 引用恢复；相关回归覆盖。 |
| `llmtuner/models/hf/flops.py` | 已阅读 | quadratic_attention_flops_per_token、MoEGeometry、flops_per_token、int_attr、first_int_attr；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/hf/model.py` | 已阅读 | HFTransformerModel；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/hf/normalization.py` | 已阅读、CPU 数值与算子契约验证 | 显式 aten RMSNorm；混合 dtype、非连续输入、空 batch、独立 HF 梯度及 opcheck；仅 CPU、一阶梯度，目标设备不支持。 |
| `llmtuner/models/hf/model_setup.py` | 已阅读 | resolve_submodule_name、validate_head_config、configure_experts_implementation、apply_cast_lm_head；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/hf/state_dict_adapter.py` | 已阅读 | 单文件/index 共用 key 覆盖校验；真实 DCP 单文件绑定 head 加载回归，FQN 映射冲突明确拒绝；PP 硬件读取待验证。 |
| `llmtuner/parallel/__init__.py` | 已阅读 | __getattr__、__dir__；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/attention_heads.py` | 已阅读 | head_counts、require_heads_divisible_by；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/compatibility.py` | 已阅读 | CompatibilityRule、validation_once_requires_dp1、validation_once_requires_finite_corpus、chunked_loss_pp、chunked_loss_validation；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/context_parallel/__init__.py` | 已阅读 | 包入口及公开导出；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/context_parallel/attention.py` | 已阅读 | cp_all_to_all、reject_unrepresentable_attention_kwargs、run_flex、SeqToHead、HeadToSeq；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/context_parallel/context_parallel.py` | 已阅读 | require_ulysses_compatible、apply_cp；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/context_parallel/compression_windows.py` | 已阅读，迁移未验收 | COMPRESSOR-CP-WINDOW-01及后续CSA批次：变长窗口交换、窗口owner、HCA/CSA overlap与真实indexer；CPU/Gloo输出/梯度和索引参考通过，目标设备/其他轴仍待验。 |
| `llmtuner/parallel/context_parallel/compressor_attention.py` | 已阅读，迁移未验收 | COMPRESSOR-CP-SLIDING-01、COMPRESSOR-CP-THREE-FSDP-01：sliding/HCA/CSA通信、模型安装及无cache decoder；CP2独立HF参考和CP2×custom FSDP2训练恢复通过，padding/packing/AC/compile/目标设备仍开放。 |
| `llmtuner/parallel/context_parallel/input_sharding.py` | 已阅读 | require_torch_cp、resolve_load_balancer、shard_batch_for_cp、shard_padding_mask_for_cp、shard_padding_mask_for_tp；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/context_parallel/kcp.py` | 已阅读 | _kcp_runtime、kcp_kda_forward、apply_kcp；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/expert_parallel/__init__.py` | 已阅读 | 包入口及公开导出；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/expert_parallel/block_conversion.py` | 已阅读 | restore_fp32_state_buffers、build_aux_loss、convert_block；FP32 恢复限定为 MoE/AuxLoss 自有 buffer，shared expert 不使用真值判断；多设备数值待验证。 |
| `llmtuner/parallel/expert_parallel/checkpointing.py` | 已阅读 | mark_experts_ep_sharded、expert_shard_map、named_params、matched_entries、gather_expert_state；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/expert_parallel/domino_dispatch.py` | 已阅读 | DominoRoutedExperts 双切片dispatch/compute/combine编排，逐slice metadata；真实stream overlap待迁移。 |
| `llmtuner/parallel/expert_parallel/mc2_dispatch.py` | 已阅读 / CPU adapter 模拟验证 | MC2GroupedExperts拥有融合forward及参数；MC2-EXPERT-HOOK-02验证native/custom独立专家FSDP与relocation，算子为CPU模拟；Ascend runtime/TP待验证。 |
| `llmtuner/parallel/expert_parallel/mc2_ops.py` | 已阅读 / CPU adapter 模拟验证 | 可选 torch_npu、HCCL 名称、两个融合 autograd 算子及 grouped 权重梯度；真实内核待验证。 |
| `llmtuner/parallel/expert_parallel/expert_parallel.py` | 已阅读 | apply_ep；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/expert_parallel/hf_inspection.py` | 已阅读 | router_of、FusedExperts、fused_experts_of、has_router_weight、is_hf_moe_block、resolve_shared_expert；TP/EP 复用结构探测，后层布局失败在替换前拒绝；多设备数值待验证。 |
| `llmtuner/parallel/expert_parallel/model_replacement.py` | 已阅读 | swap_hf_moe_blocks；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/fully_shard_parallel/__init__.py` | 已阅读 | 包入口及公开导出；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/fully_shard_parallel/fine_grained.py` | 已阅读 | fully_shard_fine_grained、NativeFSDPStrategy、CustomFSDPStrategy、get_fsdp_strategy；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/fully_shard_parallel/fine_grained_param_group.py` | 已阅读 / CPU 分布式数值及 event 单测验证 | Hook-owned all-gather/reduce-scatter 缓冲，跨 hook 等待和释放，多参数 mesh 分组；设备通信及性能待验证。 |
| `llmtuner/parallel/fully_shard_parallel/fine_grained_state.py` | 已阅读 / CPU 分布式数值验证 | 自定义 hook 归属、多上下文 buffer/event 生命周期、root discovery、前向/反向及 reset；共享通信 streams，目标设备与完整上游优化待验证。 |
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
| `llmtuner/parallel/tensor_parallel/checkpointing.py` | 已阅读、CPU恢复与有限规划错误协调验证 | TP全局布局gather/load、packed gate/up、DP DTensor；有限本地规划错误协调、TP×EP×FSDP CPU及aot_eager双向恢复通过；目标设备、拓扑变化尚未验收。 |
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

验证归约、异步 checkpoint 与双 CP 轴运行时复核：

- PyTorch 2.14.0 CPU/Gloo，loopback，TORCHDYNAMO_DISABLE=1。
  run_all.py --master-port 29648 validation_reduction_equivalence
  async_checkpoint_lifecycle_equivalence：2 passed。DP×CP 与 DP×TP
  使用不同有效 token 数的投影模型验证生产 eval 归约，未执行真实并行
  attention 变换；async DCP 两次生命周期，恢复后模型/优化器/scheduler
  与连续运行完全一致，独立 PG 已释放、主 PG 可继续 collective。
- master port 29649 cp_dual_axis_equivalence：4 rank 通过。普通/packed
  eager flex 前向误差均为 0；dense SDPA 替代 attention 的通信梯度
  与完整参考符合 1e-9 容差，并确认旧通信顺序会出现显著数值差异。
- 同步比较文档中仍标为未实现的 KCP、双 CP 轴状态；GPU/NPU、
  flex backward 和 pinned/process staging 验证缺口保留。

MoE 与 FSDP/PP 组合集成复核：

- CPU/Gloo，PyTorch 2.14.0，loopback、TORCHDYNAMO_DISABLE=1。
  port 29650：ep_checkpoint_ac_equivalence 通过，模型/优化器/EMA
  精确恢复到 wrapped EP2、unwrapped EP2、unwrapped EP1；
  pp_ep_equivalence 四 rank 实际 PP2×EP2×dense FSDP 四步训练
  与完整参考 loss 在 1e-4 容差内一致。
- port 29651：TP2×FSDP2 四步训练与完整模型比较，loss 最大误差
  4.768e-07，最终参数最大误差 7.000e-06，失败 rank 0。
- EP×FSDP 首次执行暴露测试迁移遗漏：两处 cfg.ep 已删除，导致
  AttributeError。改为 cfg.ep_size；检索 integration 目录未见其余
  cfg.ep/tp/cp/pp 旧引用。port 29652 重跑默认和 pattern FSDP
  均通过，loss 311.874268 vs 参考 311.874283，专家/路由梯度
  与 mesh 检查失败 rank 0。定向 pre-commit、diff 检查通过。
- 此证据覆盖 CPU 原生 FSDP，不包含 custom FSDP 或设备融合内核。

FSDP 复制/分片与 TP/EP 范数集成复核：

- PyTorch 2.14.0 CPU/Gloo，loopback、TORCHDYNAMO_DISABLE=1。
  port 29653 fsdp_mesh_equivalence.py ddp 与 shard 各训练 4 步：
  loss 与完整参考一致，最终参数最大误差均为 0，失败 rank 0；
  无通信参考的梯度差异约 0.05–0.06，证明测试能识别归约缺失。
- port 29654 三项通过：ep_fsdp_without_dense_dp 验证 TP4×EP2
  在 dense DP1/CP1 下仍为专家应用 efsdp2；该脚本仅检查装配与
  DTensor mesh，未运行前向/反向。ep_grad_norm_equivalence 与
  tp_grad_norm_equivalence 验证实际跨 rank 范数及裁剪，包括 TP/EP
  共用物理 ranks 时专家贡献只计一次。GPU/NPU 验证仍未覆盖。

Dense DP1 下 TP×EP 专家分片数值补强：

- ep_fsdp_without_dense_dp.py 从仅检查装配扩展为真实 TP4×EP2×
  efsdp2 前向/反向；复制变换前 HF 模型为参考，比较 token shard
  logits 和三个专家权重梯度，确认各梯度非零及 expert DTensor mesh。
- 明确关闭 router 辅助 loss，只验证主 CE；直接驱动模型时默认辅助
  loss 需要 trainer 发布 token 分母，首次试跑因此拒绝，此为测试
  设置遗漏，未改变生产 guard。
- PyTorch 2.14.0 CPU/Gloo 四 rank，loopback port 29655，
  TORCHDYNAMO_DISABLE=1：logits 最大误差 1.192e-07，专家梯度
  最大误差 2.561e-09；rtol=1e-4、atol=1e-5，全部通过。
  定向 pre-commit、diff 检查通过；未覆盖 optimizer 更新、router
  辅助 loss、目标设备内核。

TP×EP×CP 前向测试设置复核：

- CPU/Gloo 四 rank cp_tp_equivalence 通过。三轴脚本首次被生产
  CP backend guard 拒绝：默认 CPU SDPA 不支持 CP。测试显式启用
  eager flex；HF float64 参考显式选择 eager experts，避免默认
  grouped-mm 不支持 Double。保持生产环境 guard。
- 原 1e-9 容差忽略 FP32 MoE routing，实测 logits 误差 5.962e-08、
  loss 误差 4.014e-09；依据 ep_wiring_equivalence 同类参考采用
  1e-6 并写明原因。port 29656、TORCHDYNAMO_DISABLE=1 重跑
  三轴通过，定向 pre-commit、diff 检查通过。仅覆盖前向与 loss
  汇总，未覆盖 flex backward、FSDP 装配或目标设备。

CP 梯度与 PP×CP 参考初始化复核：

- PyTorch 2.14.0 CPU/Gloo，port 29657，TORCHDYNAMO_DISABLE=1：
  cp_grad_equivalence（无 attention 的模型，真实 FSDP CP 梯度
  归约与训练参考）和 replicated_attention_cp_equivalence 均通过；
  后者 TP2×Ulysses CP2 复制 MLA 前向误差 1.356e-08，未测 backward。
- PP×CP 脚本参考只按 cfg.seed 重新初始化，未加载实际 stage 权重。
  参照已运行的 PP/PP×EP 测试，增加 DTensor full_tensor 后汇集
  stage state、核对重复键一致、strict 加载完整参考。
  定向 pre-commit 和 diff 检查通过。该脚本训练所需 flex backward
  仍待目标硬件执行，不能将此次静态修复记作运行通过。

Custom FSDP 适配边界与数值复核：

- fine_grained.py 的 custom 策略实际调用 PyTorch 公共 fully_shard，
  distinct hook_module 明确拒绝；不等同于迁移 FSDPTurbo 独立
  param group/state 引擎。配置 docstring 明确此边界，保留原行为。
- fsdp_mesh_equivalence.py 新增第二个参数 native/custom，并输出
  所选实现。CPU/Gloo 两 rank、PyTorch 2.14、port 29658、
  TORCHDYNAMO_DISABLE=1，shard custom 四步训练 loss 与完整参考
  一致，最终参数误差 0、失败 rank 0。此为适配层验证，独立 hook
  引擎语义仍未实现，设备 overlap 仍待验证。定向 pre-commit 通过。

Pattern FSDP hook 错误的装配原子性修复：

- 对照 FSDPTurbo fine_grained_fully_shard/fully_shard.py 与 state.py，
  其 hook 重定位还包含独立通信 context；PyTorch 2.14 state.init
  新增 auto_reshard_after_forward、使用 _fsdp_param_groups，公开
  unshard/reshard 无法直接表达全部上游生命周期语义。迁移缺口保留。
- 当前 pattern plan 若后续单元带非法 hook_module，会在前面单元已
  fully_shard 后才失败。提前校验全部 override：native 拒绝该参数，
  custom 拒绝独立 target，避免该已知错误留下部分模型变换。
- 新增两种策略的回归用例，确认拒绝发生在策略调用前。FSDP 契约
  44 passed；定向 pre-commit 和 diff 检查通过。

当前完整集成清单与质量门禁复核：

- HEAD 19e1614 加当前工作区（包含本轮测试迁移修改），PyTorch 2.14.0
  CPU/Gloo。GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1
  python tests/integration_tests/run_all.py --master-port 29660：
  41 个默认脚本，40 passed、1 failed。唯一失败 pp_cp_equivalence
  在装配阶段被 CPU SDPA 后端 guard 拒绝；该脚本仍需支持 flex
  backward 的环境。没有替换内核、跳过或删除这个失败项。
- 清单默认参数不能替代所有变体：custom/shard FSDP、PP 虚拟
  stage schedule、pattern EP/FSDP 的独立运行证据见此前条目；
  GPU/NPU、Inductor、KCP、TorchAO/MC2、custom 独立 hook 生命周期
  仍不能由该结果证明。
- 当前 pytest：2736 passed、2 skipped、122 warnings（36.64s）。
  tracked 文件 pre-commit 全部通过，沿用 native_alltoall.py 保护
  例外；compileall、git diff HEAD --check 通过。完整集成日志保存在
  /tmp/llmtuner-full-integration-review.log，临时日志不作为仓库产物。

FSDP pattern 参数拼写错误的提前拒绝：

- per-unit override 任意键原先直接传 fully_shard，后续单元的拼写
  错误会在前面单元已变换后触发 TypeError。根据安装 PyTorch
  fully_shard 的 keyword-only 签名预先校验全部键，兼容版本新增
  参数，同时保留 custom hook_module 的独立语义校验。
- 扩展 native/custom 回归测试，含 reshard_after_foward 拼写错误，
  确认策略调用前拒绝。定向 pre-commit 和 diff 检查通过。

Pattern FSDP 全局排除边界修复：

- common | unit override 会让 ignored_params=None/空集合覆盖
  fsdp_ignored_modules 生成的全局排除集合。改为两层集合取并集，
  unit 只能增加排除项；每个 unit 使用新集合，避免共享可变集合。
- 新增 None/空集合回归用例：检查选中单元和 root 都保留全局
  exclusions。FSDP 契约 48 passed，定向 pre-commit 通过。
- PyTorch 2.14.0 CPU/Gloo 四 rank、port 29661、
  TORCHDYNAMO_DISABLE=1，EP×pattern FSDP 等价脚本重跑通过，
  loss/专家梯度/router 梯度检查失败 rank 0；目标设备未验证。

Pattern FSDP 容器选择复核：

- 宽泛 pattern 可选中 ModuleList/ModuleDict，PyTorch fully_shard
  不支持这些无 forward 的容器；逐层包装时可能在子层已变换后才
  报错。预先检查全部单元及 root，明确要求选中子模块或有 forward
  的父模块，保持合法 pattern 行为。父子单元由深到浅处理并去重，
  本轮未发现其排序遗漏。
- 新增两类容器的回归用例，确认策略调用前拒绝。FSDP 契约
  50 passed；pre-commit 首轮 Ruff 类型写法修正后重跑通过，
  diff 检查通过。此轮未改变合法路径的通信或数值语义。

Synthetic loader 缺失恢复游标修复：

- 非空 checkpoint 缺少 steps 时原先默认为 0，可能在模型已恢复
  后静默重读训练数据。改为明确拒绝缺失游标；空 state 保持 no-op，
  旧含 steps 格式保持支持。回归用例确认报错不改变后续数据流。
- 数据管线定向测试 277 passed；pre-commit 通过。CPU/Gloo 两 rank
  port 29662、TORCHDYNAMO_DISABLE=1，PP checkpoint full/resume
  两阶段重跑通过，恢复 loss 与连续轨迹一致。diff 检查通过。

优化器 checkpoint 张量布局复核（2026-10-14）：

- `components/optimizer/state_dict.py`：复现 PyTorch 接受错误形状的 Adam 动量，
  导致恢复后首次更新失败且此前其他 optimizer 已应用 checkpoint 值的问题。
  在反序列化阶段检查直接张量状态的类型和目标形状，保留 step 标量兼容行为；
  container 既有的全量预校验使该错误在应用任何 optimizer checkpoint 值前报告。
- 新增三项回归用例覆盖 `exp_avg`、`exp_avg_sq`、`max_exp_avg_sq`，
  修复前全部失败，修复后通过，并检查两个 optimizer 的状态身份和学习率未改变。
- 验证基线：HEAD `f4b43fd` 加当前工作区；macOS / Python 3.13 / PyTorch 2.14。
  `pytest -q tests/unit_tests/cpu/components/optimizer`：203 passed；
  `pytest -q --disable-warnings`：2746 passed、2 skipped、130 warnings。
  `GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1 python
  tests/integration_tests/run_all.py --master-port 29663 ep_checkpoint_ac_equivalence`：
  CPU/Gloo 两 rank 通过，wrapped EP2、unwrapped EP2 和 EP1 的恢复均 exact。
- 全部现存 tracked 文件的 pre-commit 通过（按既有要求排除
  `examples/native_alltoall.py`）；compileall 与 `git diff HEAD --check` 通过。
  本轮没有变更 collective 或模型分片语义；这些证据不覆盖目标设备内核。

Custom FSDP 生命周期迁移复核（2026-10-14）：

- 逐项对照 FSDPTurbo 的 fully_shard/state/param_group/utils 和安装的
  PyTorch 2.14 私有接口；初始化后的真实对象确认 `_fsdp_param_groups`、
  `reduce_scatter_states` 为列表，`_last_post_reduce_events` 为字典。
  注意通信上下文成员在 `lazy_init` 后才建立，不能在裸构造对象上直接探测。
- 上游 `copy_fsdp_comm_ctx` 共享通信 stream 和 `post_forward_order`；
  多上下文不等于独立 stream。迁移必须同时处理全部 context 的 optimizer
  等待、forward buffer 释放、backward buffer 队列和异常 reset。
- `distributed_implementation_comparison.md` 已补充逐生命周期接口映射及验收
  场景，修正“暂不迁移”和“仅待硬件验证”两处过时状态；fine_grained.py
  docstring 明确当前原生适配不提供 relocated hook / 多 context 引擎。
- 当前状态仍为完整 custom 引擎未实现；本轮没有将适配器改名为完成迁移。
  运行时接口探测通过；FSDP contract 单测 50 passed；受影响文件 pre-commit
  与 `git diff HEAD --check` 通过。本轮修改说明与文档，未改变训练执行逻辑。

FSDP AdamW checkpoint 连续更新验证（2026-10-14）：

- 新增 `tests/integration_tests/fsdp_checkpoint_equivalence.py`，通过实际
  apply_fsdp、OptimizersContainer、LRSchedulersContainer 和同步 DCP manager，
  覆盖 native/custom 原生适配 × dp_shard2/dp_replicate2 四种配置。
- rank 使用不同数据，训练六步，第二步保存；恢复对象使用不同 seed 和 base LR，
  继续四步，逐步 loss/全量梯度及最终参数/Adam 状态/scheduler 与连续训练 exact。
  独立 dense torch.optim.AdamW 参考使用全局 token 分母、weight decay0.1 和
  手工计算的线性 LR 曲线，loss/梯度/参数/动量绝对容差 1e-5。
- 命令：`GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1 python
  tests/integration_tests/run_all.py --master-port 29664 fsdp_checkpoint_equivalence`，
  两 rank CPU/Gloo 通过。基线 HEAD `f4b43fd` 加当前工作区；PyTorch2.14。
- 初次失败来自测试 oracle 未计入 scheduler 的 virtual stable step，以及误从
  container 的基础 param_groups 读取 lr；按现有实现契约修正测试后全部通过，
  本轮未发现新的生产代码 bug。该脚本由 run_all 自动发现，pytest 不运行它。
- 当前完整 CPU suite：`pytest -q --disable-warnings`，2746 passed、2 skipped、
  134 warnings（30.04s）。
- 受影响脚本和全部现存 tracked 文件 pre-commit 通过，仍排除受保护示例
  `examples/native_alltoall.py`；compileall 与 diff 检查通过。这里只覆盖当前
  native/custom 适配、同步保存和同拓扑恢复，不证明完整 custom 引擎、异步保存、
  改变 world size 的恢复或目标设备内核正确。

FSDP checkpoint 改变分片布局验证（2026-10-14）：

- 扩展 `fsdp_checkpoint_equivalence.py` 到 source/resume 的 shard、replicate
  二乘二组合，分别运行 native/custom 适配，共八个实际训练用例。源和目标
  runtime 均检查 embedding DTensor 的本地行数，避免仅改变配置而未改变布局。
- 第二步保存、恢复时立即比较完整模型和 Adam 状态：八个用例均 exact。
  后续四步同布局恢复 loss/梯度/参数/optimizer exact；跨布局绝对容差1e-5。
  每个用例仍对照独立 dense AdamW 六步参考并检查 scheduler 恢复与最终计数。
- 命令：`GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1 python
  tests/integration_tests/run_all.py --master-port 29665 fsdp_checkpoint_equivalence`，
  八个用例通过、脚本 exit0（3.6s）；完整输出暂存于
  `/tmp/llmtuner-fsdp-reshard-review.log`。基线 HEAD `f4b43fd` 加工作区，
  macOS/PyTorch2.14 CPU/Gloo 两 rank。
- 本轮仅扩展分布式测试，生产代码未改变；目标脚本 pre-commit、diff 检查通过。
  既有 CPU suite 结果见上一条，不将其表述为本轮重新运行。
  此证据覆盖固定 world_size=2 的 shard/replicate 转换；不同 world size、
  PP/TP/EP 拓扑变化、完整 custom 引擎及目标设备执行仍未由该脚本验证。

Adam checkpoint step 计数器复核（2026-10-14）：

- 真实复现未校验的 step 状态：两元素 tensor 在下一次 AdamW 更新时报 Scalar
  转换错误；NaN 状态更新产生非有限参数；-1 状态更新触发 ZeroDivisionError。
  `components/optimizer/state_dict.py` 现在在恢复 Adam/AdamW 前检查单值、
  实数、有限、非负且整数语义，错误报告具体 FQN 状态键。
- 新增16个回归用例覆盖 Adam/AdamW × 向量、NaN、Inf、负数、小数、tensor
  bool、Python bool、字符串；检查拒绝后状态身份与 LR 不变，且真实下一次更新
  仍为 step1、参数有限。保留合法整数/浮点整数的旧 checkpoint 转换入口。
- 当前完整 CPU suite：2762 passed、2 skipped、122 warnings（29.18s）；
  optimizer suite：219 passed。两 rank CPU/Gloo 的 EP checkpoint 三种布局
  exact，FSDP checkpoint 八种同/跨 shard-replicate 恢复用例通过。
  命令：`GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1 python
  tests/integration_tests/run_all.py --master-port 29666 fsdp_checkpoint_equivalence
  ep_checkpoint_ac_equivalence`；输出 `/tmp/llmtuner-adam-counter-review.log`。
- 基线 HEAD `f4b43fd` 加工作区，macOS/Python3.13/PyTorch2.14；全部现存
  tracked 文件及新增集成脚本 pre-commit 通过，按既有要求排除
  `examples/native_alltoall.py`；compileall 与 diff 检查通过。
  本轮未改变 collective 或拓扑，不将 CPU 恢复验证扩展为目标设备内核结论。

模型 checkpoint 全量预校验（2026-10-14）：

- `components/checkpointer/model_state.py` 原先对每个 part 依次 strict=False：
  缺少本地权重静默成功；后续 part 形状/类型错误则在前面 part 已更新后失败。
  三个回归场景修复前全部失败。现在在 EP 切片和任意 part 权重应用前检查
  组合模型视图的必需键、Tensor 类型及形状；仍允许其他 part 与辅助状态类别。
- 新增辅助类别与 legacy scalar buffer 兼容测试，保留 PyTorch 将旧版 `[1]`
  tensor 加载到 scalar buffer 的行为。模型状态单测15 passed。
- 两 rank CPU/Gloo 的 ep_checkpoint_ac_equivalence、fsdp_checkpoint_equivalence、
  hf_checkpoint_pp、pp_checkpoint_equivalence 全部通过（port29667），包括
  HF 两 pipeline stage 加载、PP 独立进程恢复、EP 权重/optimizer/EMA 和
  FSDP 同/跨布局 AdamW 连续更新。日志 `/tmp/llmtuner-model-state-review.log`。
- 最终完整 CPU suite：2767 passed、2 skipped、130 warnings（29.63s）；
  Checkpointer 与 trainer 定向 suite 181 passed；全部 tracked 文件及新增集成
  脚本 pre-commit 通过（排除受保护 native_alltoall 示例），compileall 与 diff
  检查通过。基线 HEAD `f4b43fd` 加工作区，PyTorch2.14/Python3.13/macOS。
  预校验不保证用户自定义 load hook 内部异常的事务回滚，也不能回滚 DCP 已
  写入 live storage 的值；此修复只保证上述可提前检查的问题在 part load 前拒绝。

验证默认 RNG 状态隔离（2026-10-14）：

- validation 只恢复模型模式和 attention policy，验证中的随机操作仍推进
  Python/NumPy/PyTorch 默认流；成功、异常两种回归用例修复前均失败。
  `utils/seeding.py::preserve_rng_state` 使用 torch.random.fork_rng 保存 CPU
  和指定设备的默认流，并在 finally 恢复 Python/NumPy；validation 对模式切换、
  loader 构造、前向、报告和清理应用此上下文，避免这些随机操作改变训练默认流。
- 验证 suite65 passed；完整 CPU suite2769 passed、2 skipped、134 warnings
  （31.81s）。实际 MPS 默认 RNG 隔离探测通过；这只验证 RNG API，不是训练内核。
  两 rank PP validation 和四 rank validation reduction 集成脚本全部通过：
  `GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1 python
  tests/integration_tests/run_all.py --master-port 29668 pp_validation_equivalence
  validation_reduction_equivalence`，日志 `/tmp/llmtuner-validation-rng-review.log`。
- 全部现存 tracked 文件 pre-commit（排除保护示例）、compileall、diff 检查通过。
  收尾时 HEAD `3c67bba` 加本轮工作区修改；Python3.13/PyTorch2.14/macOS。
- 边界：该上下文不保存 loader 私有 Generator 或 DTensor 私有 RNG tracker。
  当前 assembly 注册的 checkpoint 附加状态仍仅为 trainer 和 dataloader，
  全局默认 RNG 未持久化；带 dropout 等随机训练的跨进程恢复等价仍需补齐，
  既有无随机训练模型的 checkpoint 数值证据不能覆盖这项需求。

随机训练恢复缺口的实际复现（2026-10-14）：

- 通过生产 DistributedCheckpointManager、OptimizersContainer 和 scheduler
  实际保存 Linear→Dropout(0.5)→Linear 模型第二步状态；新 runtime 恢复后
  第三步 loss 与连续训练差0.04241770878434181。重新加载同一 checkpoint，
  在加载完成后恢复保存的 torch CPU RNG，第三步 loss 和全部参数 exact。
  因而无 dropout 模型的 checkpoint 等价证据不能证明随机训练恢复正确。
- 真实两 rank DCP probe 使用种子100/101保存不同默认 RNG：共用 `rng` tensor
  键时恢复匹配结果为 `[True, False]`，rank1 被另一 rank 的值替代；使用
  `rng.rank_0` / `rng.rank_1` 独立键后两 rank 均 exact。命令为
  `GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --nproc_per_node=2
  --master_addr=127.0.0.1 --master_port=29669 /tmp/llmtuner-rng-rank-probe.py`。
- 本轮获得运行时证据，尚未修改 RNG checkpoint 格式或接线。下一项实施须处理
  rank 独立键、Python/NumPy/CPU/设备流、旧格式缺失状态、恢复执行顺序，以及
  world size/并行坐标变化的兼容策略；不能直接把单份 RNG tensor 加入 states。
  DTensor 私有 RNG tracker 仍需单独核对。环境为PyTorch2.14/macOS CPU/Gloo，
  dropout probe 单进程，rank-key probe 两进程 exit0。

默认 RNG checkpoint 状态接线（2026-10-14）：

- 新增 `components/checkpointer/random_state.py::RNGCheckpointState`，将
  Python/NumPy/torch CPU/所选设备默认状态保存为 rank-qualified opaque payload，
  避免 DCP dedup 丢失 rank 独立流；训练 assembly 注册 `rng_state`。
- DCP 在模型、optimizer、scheduler 和 EMA 加载后显式恢复 RNG；旧 checkpoint
  完全缺少 RNG 类别时 planner 移除该类别加载键并警告，保持旧格式可加载。
  有 RNG 类别但缺少当前 rank 的状态仍由严格 planner 拒绝。world size 或
  device type 不同拒绝，提示通过 exclude_from_loading 的 rng_state 明确放弃
  随机流连续性。model-only/HF 权重加载不恢复训练 RNG。
- 新增 CPU 单测5项：三种默认流精确恢复、版本/world/device 不兼容时状态不变、
  空状态 no-op。新增两 rank DDP dropout 实际训练脚本，rank 使用不同种子，
  Python/NumPy 随机输入与 torch dropout 的两步恢复 loss/参数/Adam/scheduler exact。
  旧格式无 RNG 的真实 DCP 加载探测通过并输出预期警告。
- 完整 CPU suite2774 passed、2 skipped、138 warnings（31.12s）；
  `GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1 python
  tests/integration_tests/run_all.py --master-port 29671 pp_checkpoint_equivalence
  rng_checkpoint_equivalence` 两脚本通过，PP full/resume 独立进程也通过。
  全部现存 tracked 文件加新增文件 pre-commit（排除保护示例）、compileall、
  diff 检查通过。日志 `/tmp/llmtuner-rng-final-{unit,integration}.log`。
- 剩余：同 world size 的并行坐标重新分配尚无检查，DTensor tracker、loader
  私有 Generator、异步 RNG staging 与目标设备训练数值还未覆盖；设备流使用
  已有 get/set_rng_state 接口，CPU 证据不证明所有后端兼容。payload 用 pickle
  保持 Python/NumPy 格式，遵守既有可信训练 checkpoint 的加载边界。

RNG 恢复并行拓扑契约（2026-10-14）：

- random_state.py 现在保存已解析 dp_replicate/dp_shard/CP/TP/PP/EP 及双CP
  轴度数的独立映射，assembly 传入 trainer.parallel_dims；同 world size 的
  并行布局变化或包含 RNG 但缺少所需拓扑信息的旧格式，恢复时提示显式排除
  rng_state。不将不同并行布局下的 rank 随机流视为连续训练状态。
- 五个轴变更回归用例检查拒绝前 CPU RNG 不变，以及外部 topology 变更不影响
  已构造状态对象的元数据。定向 RNG/trainer suite107 passed，PP checkpoint
  与 rank-distinct dropout 实际两rank脚本均通过（port29672）。初次新增注解
  缺少 future annotations 的导入错误已修复，并重新验证成功。
- 该策略保守地拒绝 dp_shard/replicate 等布局变更的 RNG 恢复；模型/optimizer
  DCP 仍可在用户排除 RNG 状态后跨布局恢复。任意自定义 rank 重映射不由度数
  签名证明一致；当前 topology 使用固定轴构造顺序。DTensor tracker仍未覆盖。
- 受影响文件 pre-commit、diff 检查通过；基线3c67bba加工作区。

异步 RNG checkpoint 连续训练验证（2026-10-14）：

- 扩展 rng_checkpoint_equivalence.py 同时执行 disabled/async 保存，async
  分支检查 save_future 已建立；等待 staging 后，在 writer.close 前继续两步
  Python/NumPy 随机输入与 torch dropout 训练。新 runtime 恢复第二步状态后
  继续两步，loss/模型/Adam/scheduler 精确一致，证明当前 CPU staging 路径
  不会因后续随机训练污染已保存 payload 和模型/optimizer 快照。
- 两rank CPU/Gloo 命令：`GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1 python
  tests/integration_tests/run_all.py --master-port 29673 rng_checkpoint_equivalence`，
  sync/async 都通过（2.9s），日志 `/tmp/llmtuner-rng-async-review.log`。
  目标文件 pre-commit、compileall、diff 检查通过；未改生产执行逻辑，因此
  本轮未重复完整 CPU suite，上一轮结果仍是该源码基线的证据。
- future 存在不证明落盘与计算在物理时间上重叠；此测试检查 staging 边界和
  close 等待后的恢复正确性。Pinned/process staging、GPU/NPU 和 DTensor
  私有随机流仍需单独验证；本轮没有发现新的生产缺陷。

PyTorch2.14 DTensor RNG 初始化契约复核（2026-10-14）：

- 安装源码 `_random.OffsetBasedRNGTracker._distribute_region` 直接读取/写回
  默认设备 RNG，`manual_seed` 注释明确不再维护独立 state 副本；因此历史
  “必须额外保存 tracker 状态”的假设不适用于普通2.14 DTensor。LocalTensor
  mode 和显式 generator 仍属于另外的状态来源，未由此结论覆盖。
- `_dispatch.py` 首次隐式创建 tracker 使用 run_state_sync=True、广播全局rank0
  状态，会覆盖 PP stage 的 distinct seed；torchtitan 显式关闭该同步。
  新增 seeding.initialize_dtensor_rng，在 assembly mesh 建立后、模型构建前
  初始化 tracker 并传 run_state_sync=False，不再次 seed，不消费随机数。
  CPU/无mesh跳过，已有tracker保持；更新 RNG checkpoint 与隔离函数说明。
- 两个定向 mock 用例证明参数、初始化一次及CPU跳过，不作为设备数值证据。
  seed/trainer suite105 passed；完整CPU suite2781 passed、2 skipped、134 warnings
  （31.70s）。PP checkpoint 与 sync/async RNG checkpoint 实际CPU脚本通过
  （port29674）；目标文件pre-commit、compileall和diff检查通过。
- CUDA/NPU 上 DTensor random op、PP distinct seed 和恢复后设备 offset 等价
  仍需目标设备验证。当前不创建CPU tracker，符合2.14明确拒绝CPU实例的契约。

FSDP 分片随机训练恢复验证（2026-10-14）：

- rng_checkpoint_equivalence.py 扩展到 DDP/FSDP × disabled/async 四种真实
  两rank配置。FSDP 对两层Linear分别fully_shard再包装root，并检查embedding
  位置的首个Linear权重DTensor本地行数为全局的一半；实际参数、Adam动量
  为分片状态，不以仅配置声明证明FSDP。
- rank-distinct Python/NumPy输入和dropout，第二步保存后等待staging并继续
  两步；新runtime恢复后loss/本地参数/Adam/scheduler均exact。命令为
  `GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1 python
  tests/integration_tests/run_all.py --master-port 29675 rng_checkpoint_equivalence`，
  四配置通过（3.2s），日志 `/tmp/llmtuner-fsdp-rng-review.log`。
- 本轮修改集成脚本，生产源码未改变；pre-commit、compileall、diff检查通过。
  此证据针对原生CPU FSDP及同拓扑恢复，不覆盖custom引擎、设备DTensor随机
  kernel或跨拓扑随机流等价。首层为Linear而非token embedding。

RNG 旧格式兼容与缺失 rank 的 DCP 回归验证（2026-10-14）：

- test_random_state.py 新增两个实际磁盘 DCP 用例：旧格式完全缺少 RNG 类别时
  模型加载成功、默认 CPU RNG 不变；已存在 RNG 类别但仅有另一 rank 键时
  严格 planner 报当前 rank 缺失，模型 target 保持未写入、随机流不变。
  避免旧格式宽容路径吞掉损坏或不兼容 checkpoint 的缺失 rank 状态。
- RNG 单测12 passed；完整 `pytest -q --disable-warnings`：2783 passed, 2 skipped, 118 warnings in 30.67s。
  全部现存 tracked 文件及新增 RNG 测试的 pre-commit 通过（排除保护示例），
  compileall 与 diff 检查通过。基线3c67bba加工作区。
- 本轮没有生产行为改动；将此前临时 probe 的兼容结论固化为回归门禁。
  该用例是单进程磁盘 DCP；rank-distinct 分布式恢复证据见前述集成测试。

Pattern FSDP 共享参数修复：
- 两 rank 原实现复现 embedding/head 参数身份断裂。`plan.py` 在变换前解析
  参数直接所有者及所属单元，按共享关系合并为同一原生 FSDP 状态；与根单元
  共享时统一由根组管理。组内配置冲突在任何变换前拒绝，避免选择任意策略。
- prefetch 以组为单位、按模块注册顺序配置，避免对同一状态重复设置窗口。
- `pattern_fsdp_equivalence.py` CPU/Gloo 两 rank：native/custom adapter ×
  兄弟/嵌套选择四配置均通过共享身份、DTensor 实际分片、三步 SGD 的 loss、
  梯度和参数对比。独立 dense 参考，float32，梯度/参数 atol=2e-6、rtol=2e-5。
  脚本复用公共入口的 Gloo SUM 策略；不证明 custom 多通信上下文或 GPU 行为。
- 本批验证：完整 CPU suite 2784 passed、2 skipped；pattern 和已有 FSDP
  mesh 等价脚本均通过；Ruff、compileall、diff check 通过。pre-commit 检查
  全部 tracked 文件及本批新增文件通过，按用户保护要求排除 native_alltoall.py。
  日志：`/tmp/llmtuner-pattern-full.log`、`/tmp/llmtuner-pattern-integration.log`、
  `/tmp/llmtuner-pattern-precommit.log`。
- 未采用“子单元忽略共享参数、根单元收集”的实验方案：已包装子单元会被原生
  FSDP 遍历跳过。该方案已撤回。

Pattern FSDP 分组依赖复核：
- 新增嵌套共享用例实际复现深度排序错误：先包装父单元后，子单元在反向时报
  `Invariant encountered: value was None`。日志 `/tmp/llmtuner-pattern-order-before.log`。
- `plan.py` 改用父子依赖拓扑排序，跨分支共享造成的依赖环合并为同一状态，
  保留组内策略冲突预检查。新增 CPU 单测验证循环依赖的合并计划（mock 证据）。
- 两 rank CPU/Gloo：native/custom adapter × 三种 pattern 布局六配置通过
  身份、分片、loss、梯度和三步参数更新等价。日志
  `/tmp/llmtuner-pattern-order-after.log`；FSDP contract 52 passed。
- 后续补齐真实跨分支循环训练：`CrossSharedDecoder` 在左右父模块注册对方
  子模块权重，分别调用两分支。两 rank CPU/Gloo native/custom 各四布局共八
  配置通过共享身份、实际分片、loss、梯度及三步 SGD 的独立 dense 等价比较。
  日志 `/tmp/llmtuner-pattern-cycle.log`。目标设备与 custom 多上下文仍未验证。

Pattern FSDP 有效配置比较修复：
- 共享单元 `ignored_params=None` 与默认空集合实际等价，旧比较误报冲突；
  新增回归测试实际先失败。现于分组配置校验前合并全局排除边界，再比较有效
  配置；保留每组独立集合及真正策略冲突的预检查。
- FSDP contract 54 passed；CPU/Gloo 两 rank pattern 八配置数值验证通过，
  日志 `/tmp/llmtuner-pattern-exclusions.log`；受影响文件 pre-commit、
  compileall 和 diff check 通过。

Pattern FSDP 目标设备验证入口：
- 等价脚本新增显式 `--device cuda`，按 LOCAL_RANK 设置设备、使用 NCCL mesh，
  reference/model/batch 均放到同一设备；CPU 路径继续复用 Gloo SUM 策略。
- 目标命令：`PYTHONPATH=. torchrun --nproc_per_node=2
  tests/integration_tests/pattern_fsdp_equivalence.py --device cuda`。
  CPU/Gloo 八配置实际通过，日志 `/tmp/llmtuner-pattern-device.log`；CUDA 环境
  当前不可用，CUDA 分支仍待运行，不计为目标设备数值通过。
- 核对安装版 2.14 FSDPState.init/_init_shared_state：auto reshard 标志、参数组
  列表、组索引和共享通信缓冲保留策略需保留。custom 状态引擎迁移仍未完成。

梯度累积与数据耗尽复核：
- 标准 corpus loader 在 contracts.validate_loader_iteration 拒绝 repeat=False
  且 DP>1，随机数据无限；该路径无需假设有限 DP 已支持。自定义外部 loader 的
  不均匀耗尽没有协调保证，不能据此宣布有限分布式数据支持。
- 新增实际 Adam 参数回归：只有一个 batch、累积步数为二时，整窗读取先抛
  DataloaderExhaustedError，未进入前反向、参数未变、Adam 状态未创建，旧梯度
  已清空。ntokens_seen 记录已读取的四 token，符合当前消耗计数语义。
- 本轮没有确认生产缺陷，未更改 trainer；不足窗取消策略已用回归固定。
  trainer/data pipeline 相关测试结果见 `/tmp/llmtuner-exhaustion-review.log`；
  受影响文件 pre-commit 与 diff check 通过。

梯度累积实际入口数值验证：
- 原“合并 Batch”参考的 loss 差异已归因：HF preprocess 将 Batch 展平成单个
  causal stream，合并 microbatch 改变 attention 上下文，因此该参考不等价。
- accumulation_equivalence.py 保留独立前向，参考路径在 train_step 外手工
  计算 summed cross entropy、全局有效 token 分母及 backward/optimizer 更新。
  两 runtime 显式同步初始参数，rank 和 microbatch 的掩码数不同。
- CPU/Gloo 两 rank FSDP 三步 loss、梯度及 AdamW 参数更新对比通过，float32，
  atol=2e-6、rtol=2e-5。日志 `/tmp/llmtuner-accumulation-reference.log`。
  本轮没有确认累积缩放生产缺陷。尚未覆盖 HSDP、PP 或目标设备累积语义。

HSDP 梯度累积数值验证：
- accumulation_equivalence.py 支持 WORLD_SIZE=4 的 replicate=2/shard=2。
  参考每组 backward 都归约，实际 train_step 在最后一组开启副本 all-reduce；
  三步 loss、梯度和 AdamW 更新 CPU/Gloo 对比通过，float32 atol=2e-6、rtol=2e-5。
  rank 数据及有效 token 不同，rank3 全窗口 masked；全局分母仍正。
- 命令：`GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1 PYTHONPATH=.
  torchrun --nnodes=1 --master-addr=127.0.0.1 --master-port=29685
  --nproc_per_node=4 tests/integration_tests/accumulation_equivalence.py`。
  日志 `/tmp/llmtuner-hsdp-accumulation.log`。仅验证 CPU/Gloo；PP 和目标设备
  累积仍待验证。本轮未修改生产代码。

PP 梯度累积数值复核：
- pp_equivalence.py 新增 schedule 后可选累积次数参数；无 pipeline 参考在
  整个累积窗统计有效 token，再按同样 PP 行分组独立前向，四步 clip/AdamW。
- CPU/Gloo 两 rank，累积=2、PP microbatch=4：1F1B、Interleaved1F1B、
  ZBVZeroBubble 均通过 loss 轨迹和梯度范数对比，rtol/atol=1e-5。
  1F1B loss 最大绝对差4.768e-7。日志 `/tmp/llmtuner-pp-accumulation.log`、
  `/tmp/llmtuner-pp-accumulation-Interleaved1F1B.log`、
  `/tmp/llmtuner-pp-accumulation-ZBVZeroBubble.log`。
- 本轮没有生产修改；该脚本尚未逐参数比较梯度/更新，也未覆盖 PP×DP 或
  目标设备累积。相关 pre-commit 与 diff check 通过。

PP 累积逐参数等价验证：
- pp_equivalence.py 为真实 PP 与无 pipeline 参考保存每步裁剪后的梯度及
  AdamW 更新后的参数，按 FQN 比较每个本地 stage 参数，补足 loss/norm 证据。
- CPU/Gloo 两 rank，累积=2、四个 PP microbatch、四步更新，1F1B、
  Interleaved1F1B、ZBVZeroBubble 均通过，rtol/atol=1e-5。
  日志 `/tmp/llmtuner-pp-parameters.log` 和同前缀的两 schedule 日志。
- 验证范围为 PP-only；PP×DP、设备内核仍未覆盖。本轮仅修改测试，
  相关 pre-commit 与 diff check 通过。

累计工作区质量验证（HEAD 3c67bba + 当前未提交变更）：
- Python3.13/PyTorch2.14/macOS CPU：2788 passed、2 skipped，27.39s；日志
  `/tmp/llmtuner-current-full.log`。本次结果替代较早的全量 CPU 结果。
- CPU/Gloo 两 rank accumulation、pattern FSDP、默认 PP、RNG checkpoint 四
  脚本通过，日志 `/tmp/llmtuner-current-integration.log`；不代表全部组合通过。
- 全部 tracked/untracked 文件 pre-commit 通过，按用户保护要求排除
  examples/native_alltoall.py；compileall、HEAD diff check 通过。日志
  `/tmp/llmtuner-current-precommit.log`。custom 引擎及目标设备缺口仍未完成。

RNG 损坏状态恢复复核：
- 新增 Python/NumPy/CPU RNG payload 损坏的三条回归，调用真实恢复函数，
  确认异常发生后全部 live host stream 状态保持原值；临时生成器校验有效。
- random_state 测试15 passed，相关 pre-commit 和 diff check 通过。
  本轮未发现 host 恢复生产缺陷；device setter 失败原子性仍缺少目标设备证据。

设备 RNG 实际验证：
- 新增 device_rng_equivalence.py，显式 cpu/mps/cuda 设备，比较真实 rand
  序列恢复后的逐位一致性，并检查错误长度 device state 拒绝后 host/device
  状态原值不变。当前 MPS 实际通过（PyTorch2.14/macOS）。
- 命令 `PYTHONPATH=. python tests/integration_tests/device_rng_equivalence.py mps`。
  CUDA 可用相同脚本显式启动；未运行 CUDA，不证明 DTensor 随机算子、GPU
  dropout 续训或通信正确。默认 integration runner 仅运行 CPU API 证据。

FSDP ignored 模块别名修复：
- 同一模块注册 first/alias 时，默认 named_modules 去重导致 ignored alias
  误报不存在；新增回归实际先失败。忽略解析使用完整路径，再按模块 identity
  排除子树并递归收集参数，支持共享别名且避免按路径漏掉 canonical 单元。
- FSDP contract55 passed，CPU/Gloo pattern八配置通过；相关 pre-commit通过
  （首轮格式化自动修改后已重跑）。日志 `/tmp/llmtuner-alias-unit.log`、
  `/tmp/llmtuner-pattern-alias.log`。
- apply pattern 别名选择尚未迁移；需规范化到单一模块并协调重复选中配置，
  不应直接使用完整 named_modules 列表重复 fully_shard。

FSDP apply 模块别名解析：
- apply pattern 使用完整路径匹配，将结果按模块 identity 映射到首次注册路径，
  同一对象只生成一个 FSDP unit；保留既有 pattern 配置按声明顺序合并语义。
  新增“仅别名匹配 + 通配符再次选中”的回归验证无重复单元且 override 保留。
- FSDP contract56 passed，CPU/Gloo pattern八配置数值回归通过；相关
  pre-commit、diff check通过。日志 `/tmp/llmtuner-pattern-apply-alias.log`。
- 此证据验证选择器去重；同一共享模块在一次前向中被重复调用的原生 FSDP
  运行时限制仍须单独核对，不以解析成功宣称任意调用图已支持。

共享模块别名重复调用数值验证：
- pattern_fsdp_equivalence.py 新增 ReusedDecoder，同一 Linear 通过 canonical
  路径和 alias 在一次前向串行调用两次，配置仅选 alias。验证对象身份、实际
  分片和三步 loss/梯度/SGD 参数独立 dense 等价，CPU/Gloo native/custom均通过。
- 当前矩阵两实现×五布局共十配置通过，日志 `/tmp/llmtuner-pattern-reuse.log`。
  相关 pre-commit、diff check通过；不证明任意循环调用图、重计算或GPU行为。

设备 RNG 验证脚本分布式接线修复：
- 原脚本 torchrun 未初始化进程组，生产 RNGCheckpointState 会使用 rank0/size1，
  不能证明 rank 独立键。现按 WORLD_SIZE 初始化 Gloo/NCCL，收尾销毁进程组，
  all_gather 检查各 rank 的 checkpoint 键与随机 payload 均不同。
- CPU/Gloo 两 rank通过，日志 `/tmp/llmtuner-device-rng-ranks.log`；MPS 单进程
  roundtrip和拒绝隔离复跑通过，相关pre-commit通过。CUDA分支仍未运行。

Custom FSDP 策略与2.14生命周期复核：
- 删除无消费者的 supports_hook_module 标记：custom 标为True却拒绝独立hook，
  属性不能准确表达当前适配器能力；入口校验不变，完整迁移需求保持待完成。
- 安装版FSDPState没有旧 _finalize_backward；root callback按各state参数组
  逆序补post_backward，再逐组finalize，并等待reduce_scatter_states事件队列。
  root preforward等待optimizer事件只覆盖自身comm_ctx，多上下文必须显式扩展。
- FSDP contract56 passed；相关pre-commit通过。本轮不宣称custom引擎已迁移。

FSDP plan 职责整理：
- 同文件提取 _resolve_shared_fsdp_units，负责共享参数所有者分组、循环合并、
  child-first排序和有效配置冲突校验，返回完整计划而不变换模型。
  apply_fsdp_by_pattern负责设备策略及实际执行，保留预检查先于变换的边界。
- 复用现有回归：FSDP contract56 passed，CPU/Gloo十组pattern数值配置通过。
  日志 `/tmp/llmtuner-plan-refactor-unit.log`、`/tmp/llmtuner-plan-refactor-numeric.log`。
  pre-commit、compileall、diff check通过；没有新增文件或改变公共入口。

任意 FSDP prefetch 窗口数值验证：
- pattern_fsdp_equivalence.py 新增 --prefetch-window 参数同时设置前后窗口，
  显式窗口5（超过本用例单元数）CPU/Gloo两rank、native/custom×五布局十配置
  均通过三步loss/梯度/参数等价；覆盖共享组、嵌套组、循环组和重复模块调用。
- 命令 torchrun两rank运行pattern脚本并追加--prefetch-window 5；日志
  `/tmp/llmtuner-pattern-prefetch.log`。相关pre-commit和diff check通过。
  仅证明该后端数值行为，不证明GPU通信重叠、内存峰值或任意forward顺序。

MoE backend 接线复核：
- MC2 配置当前主动拒绝，无融合算子运行时；Domino选择双切片RoutedExperts，
  每片独立dispatch metadata，所有rank包含空片仍参与collective。fixed_router
  已接入debug round-robin，未误称训练router优化。
- 修正文档英文段落：Domino stream overlap 尚未实现，需要先实现再设备验证，
  不能表述为仅pending validation。中文矩阵已准确标注实现缺口。
- Domino/fixed router相关CPU单测6 passed；两rank ep_domino_equivalence通过
  前反向、重计算、空/少token场景；日志 `/tmp/llmtuner-domino-current.log`。

EP routing 布尔配置修复：
- ExpertParallelConfig 原先接受 fixed_router="false" 等truthy字符串，导致
  程序化配置意外启用调试路由；moe_quantile_balancing也存在同类输入歧义。
- 两字段构造时要求实际bool，错误含完整ep_config字段名。十个字符串/整数/
  None回归在旧实现全部失败，修复后配置套件通过；相关pre-commit通过。
  日志 `/tmp/llmtuner-ep-flags.log`。CLI正常布尔解析不变。

EP 辅助 loss 系数边界复核：配置和 HF block 替换入口共用有限、非负数值
校验，拒绝 bool、字符串、NaN、Inf 及负值；None 保持继承语义，零禁用。
修复前配置回归 5 失败/3 通过；补齐显式和继承入口测试，确认拒绝后各层
原始 MoE block 身份保持不变。当前工作区 macOS/Python 3.13/PyTorch 2.14：
`pytest -q tests/unit_tests/cpu/test_config.py tests/unit_tests/cpu/parallel/test_ep_model_replacement.py --disable-warnings`
306 通过；全套 `pytest -q --disable-warnings` 2821 通过、2 跳过，日志
`/tmp/llmtuner-ep-coefficient-full.log`。此证据不扩展 EP 硬件或组合覆盖范围。

MoE checkpoint 统计副作用复核：full/selective 的已标记重计算期间跳过
expert token counter 和 quantile histogram 写入，保留路由、输出及梯度计算。
原先依赖统计量统一翻倍的假设，不能保证部分 checkpoint 或不均匀回放时
仍正确。增强现有真实 checkpoint 测试，比较原始 forward 与 forward/backward
后的完整 histogram、token counter、梯度和 bias 更新；旧实现四种配置全部
失败，修复后 MoE routing/padding/aux loss 共 118 通过。region 路径未在此验证。

MoE 多 outstanding forward 的 checkpoint 输入复核：HF 将 padding mask
暂存在 block 属性；后一 microbatch 会覆盖前一 microbatch 的重计算输入。
真实双 forward 后统一 backward 在 full/selective 均复现 router 梯度错误
（最大绝对差约 0.0947）。checkpoint context 现在捕获各 forward 的 MoE
mask，回放时临时恢复，finally 恢复最新暂存值。四个回归配置覆盖不同 mask
及后一 microbatch 无 mask；输出和参数梯度与无 checkpoint 参考一致。
MoE 两套测试 88 通过，activation/EP checkpoint 56 通过；region 和实际
PP×EP 调度尚未在本批次验证。

Checkpoint 异常及重复 backward 复核：嵌套 MoE 注入专家回放失败，
full/selective 均恢复最新 padding mask 和 recompute ContextVar。保留图的
第二次 backward 原先触发单次 generator context 的 AttributeError；改为每次
进入创建新 context，full 模式梯度正确累加。selective 模式保留 PyTorch SAC
缓存消费后仅允许一次 backward 的限制，回归验证原生 RuntimeError 及标记
清理，未绕过缓存生命周期或宣称支持重复 backward。

Checkpoint compile 兼容性复核：新增真实 torch.compile backend=eager 的
full/selective 回归，与独立未包装模型比较输出、输入及参数梯度，两项通过。
该证据覆盖 Dynamo eager 后端，未覆盖 Inductor、MoE 编译或设备通信。
当前源码清单再次核对 154/154，无遗漏、重复和已删除条目；仅证明清单覆盖。
累计 checkpoint 修改后的全套结果见 `/tmp/llmtuner-checkpoint-compile-full.log`；
pre-commit（排除受保护示例）、compileall、HEAD diff 检查通过。

MoE compile 与 PP×EP checkpoint 复核：双 forward 不同/无 padding mask
测试增加 torch.compile backend=eager，八种组合均与未包装参考输出及参数
梯度一致。扩展现有 pp_ep_equivalence.py 的 --activation-checkpoint 参数，
当前 CPU/Gloo 四 rank（PP2、EP2、1F1B、四 microbatch、四 AdamW 更新）
默认/full/selective 均通过独立参考 loss 轨迹比较，rtol/atol 保持 1e-4。
日志 `/tmp/llmtuner-current-pp-ep{,-full-ac,-selective-ac}.log`；实际 checkpoint
装配有日志证据。未比较此组合的独立梯度/参数快照，未覆盖 padding 输入、
Inductor 或目标设备通信；不能据此宣称完整 PP×EP 矩阵验证。

PP×EP 数值证据增强：pp_ep_equivalence.py 每步保存裁剪后梯度和
AdamW 更新后参数，以 canonical FQN 对齐；dense DTensor 完整 materialize，
专家 shard 按 EP rank 拼接，dense 副本额外比较一致性。参考从实际初始
权重独立执行本地全专家训练，不复制 actual 轨迹。CPU/Gloo 四 rank PP2×EP2
默认/full/selective 三种配置、四步均通过 loss/梯度/参数比较，容差仍 1e-4。
日志 `/tmp/llmtuner-pp-ep-snapshots{,-selective,-default}.log`。
该结果补齐上一批独立梯度/参数快照缺口；padding、其他 schedule、设备后端
和更多拓扑仍未覆盖。此次仅增强验证脚本，pre-commit 排除保护示例后通过。

PP×EP 辅助 loss 参考独立性复核：参考训练每步显式设置当前批次 routed
input token 分母，去除对先前 trainer 遗留 AuxLoss 类状态的依赖。脚本新增
--aux-coefficient，实际 stage 验证 aux_loss 实例和系数安装；0.3 放大辅助
梯度后，CPU/Gloo PP2×EP2 full/selective 四步 loss、裁剪梯度、更新后参数
均与独立本地参考一致，容差保持 1e-4。日志
`/tmp/llmtuner-pp-ep-aux{,-selective}.log`。本批次只修正验证脚本，
pre-commit（保护示例排除）、Ruff、格式和 diff 检查通过。

EP×FSDP mesh ownership 复核：当前四 rank CPU/Gloo 默认及 pattern FSDP
均通过 loss、router 梯度、w1/w2/w3 专家梯度对独立 HF 参考比较。
增强 ep_fsdp_equivalence.py 断言，专家 mesh 实际 rank 为对应 EP 坐标
[0,2]/[1,3]，dense router mesh 为 [0,1,2,3]，避免相同 mesh 大小掩盖
错误通信组。新增 rank 检查的 pattern 配置再次通过，日志
`/tmp/llmtuner-ep-fsdp-ranks.log`。脚本模型为 eval，此结果不覆盖 aux loss
或 bias 更新；未执行目标设备后端。

MoE bias 统计归约复核：扩展现有 quantile_bias_equivalence.py，构造真实
MoE 的不同 rank token counts，独立 rank 求和后计算 mean-centered sign
参考。四 rank CPU/Gloo DP×CP、DP×TP 无 EP（只归 DP）、DP×TP 有 EP
（归 DP 与 TP）三种布局均匹配 bias；counter 清零，第二次无新 token 更新
不移动 bias。现有 quantile 大整数 histogram 归约、bias 和清零也通过，日志
`/tmp/llmtuner-bias-axes.log`。此证据覆盖统计更新函数与实际 collective，
没有执行 Trainer optimizer hook 或目标设备；pre-commit 排除保护示例后通过。

MoE bias optimizer hook 接线复核：上述 sign-based 集成改为真实
OptimizersContainer + register_moe_balancing_hook，通过 decoder ModuleList
发现 MoE，再调用 optimizer.step 驱动更新。四 rank CPU/Gloo 三种统计轴
布局均通过独立 bias 参考及清零、第二 step 无变化检查，日志
`/tmp/llmtuner-bias-hook.log`。未设置参数梯度，此证据验证容器 pre-hook 和
collective，不证明 AdamW 参数轨迹或 Trainer 装配；quantile 仍直接调用更新。

Quantile optimizer hook 接线复核：统计集成使用真实 MoE +
QuantileBalancedTopKRouter，替代 SimpleNamespace loss owner，经真实
OptimizersContainer.step 与 register_moe_balancing_hook 选择 quantile 分支。
四 rank CPU/Gloo 三种轴布局均匹配独立 rank-summed histogram（含超过
int32 的总数）、bias 结果及双 scratch 清零，日志
`/tmp/llmtuner-quantile-hook.log`。参数无梯度，未执行 Trainer 装配或路由
forward；保持该证据范围。初次运行因测试 import 遗漏失败，修正后重跑通过。

累计变更与能力文档复核：修正比较文档把 aux-loss 与 bias hook 混为一谈
的描述；支持矩阵集中记录 PP×EP、EP×FSDP、balancing hook 和 checkpoint
compile 的实际证据及限制，删除以入口闭环定义“完整”的泛化表述。
当前累计工作区 CPU 全套日志 `/tmp/llmtuner-current-final-cpu.log`，
pre-commit（排除保护示例）、compileall、HEAD diff 检查通过。
custom FSDP 状态引擎、MC2、Domino stream overlap 仍属实现缺口，未完成。

Memory-budget AC 配置边界复核：拒绝 bool、非数值与范围外值，保留数值
0/1；原实现把 bool 当作预算，字符串/None 抛 TypeError。六项回归修复前
4 失败/2 通过，修复后 AC/config 共 303 通过。保留已有错误消息匹配前缀，
非法配置在设置全局编译预算前失败。此检查不证明 partitioner 实际节省内存。

AC 布尔配置复核：selective preserve_rng_state/debug 与 region
preserve_rng_state 现在要求实际 bool，避免字符串 false 的 truthiness
改变 RNG 保存行为，及 falsy 非布尔值绕过限制。15 项回归修复前 11 失败/
4 通过，修复后 AC/config 共 318 通过；既有 debug=True 和 region RNG=True
拒绝语义保留。pre-commit（保护示例排除）、Ruff、compileall、diff 通过。

AC pattern 输入复核：selective force_recompute_mm_shapes_by_fqns 和
region save/recompute_regions 共用非空白字符串 list 校验，空列表保持合法。
原实现接受字符串并逐字符迭代，空 substring 匹配所有模块，非法元素可在
模型变换中抛类型错误。15 回归修复前全部失败，修复后 AC/config 333 通过；
pre-commit（保护示例排除）、Ruff、compileall、diff 检查通过。

Compile backend 失败边界复核：apply_compile 在改模块 regional 标记或
全局编译选项前通过 PyTorch registry lookup_backend 校验，保留已注册
自定义 backend 支持。未知 backend 原实现先清除模块标记再抛 InvalidBackend；
回归修复前失败，修复后 compile 专项 25 通过/2 跳过。此校验只解决未知
backend 的副作用，不声称后续所有编译失败具有事务回滚。

Async TP backend 契约复核：已安装 PyTorch 的 Inductor post_grad.py
282-283 行调用 micro_pipeline_tp_pass；eager/aot_eager 普通 TP 区域不执行
该 pass，regional Inductor 只标记 attention。CompileConfig 现在要求
async TP backend=inductor，避免接受静默无效配置。两回归修复前均失败，
修复后 compile/config 281 通过/2 跳过；实际设备重叠性能仍未验证。

Async TP 装配失败状态复核：缺少 TP mesh 时原流程先清除 regional 标记，
并给原无标记的子模块新增属性。利用 async TP 只允许 Inductor 的已校验约束，
将其环境检查提前到 model flag 写入前，避免回滚封装；回归确认父标记和子
属性缺失状态均保留。修复前失败，修复后 compile 专项 28 通过/2 跳过。

MoE compile 能力失败复核：capture_scalar_outputs 能力检查提前到 async TP
和 regional 模块标记设置前，复用一次 MoE 检测结果。缺少能力原实现先
清除标记再报错；回归修复前失败，修复后 compile 专项 29 通过/2 跳过。
mock 仅证明失败路径顺序，不构成设备编译或 overlap 的数值证据。

Training 执行开关边界复核：compile/deterministic/detect_anomaly
程序化配置原接受字符串 false、整数和 None，truthiness 可错误启用编译
或确定性行为。构造时要求实际 bool；12 项回归修复前全部失败，修复后
config/trainer 共 364 通过。CLI HfArgumentParser 的布尔解析仍走原入口。

CLI 到执行配置闭环复核：真实 parse_config 对 compile/deterministic/
detect_anomaly 的 true/false 转换为 bool，两项回归通过，程序化非法类型拒绝
没有破坏正常 CLI。累计配置和 compile 失败边界修改后全套 pytest 2890 通过、
2 跳过（macOS/Python 3.13/PyTorch 2.14），日志
`/tmp/llmtuner-cli-config-full.log`；pre-commit（保护示例排除）、compileall、
HEAD diff 检查通过。该 gate 不补足设备并行或剩余迁移实现缺口。

Dataloader shuffle 配置复核：loader_builder 将 shuffle 直接传到 iteration
policy，字符串 false 会被 map/stream 分支当真并改变读取顺序。构造时要求
bool；四回归修复前全部失败，修复后 config/datasets 共 733 通过。
pre-commit（保护示例排除）、Ruff、compileall、diff 检查通过。

SFT 字段名类型复核：local_jsonl_sft 使用的 prompt/response/messages
字段先检查字符串类型，再检查空白，避免 None/int/list 的 strip 调用抛
AttributeError；未使用字段维持原条件校验。12 回归修复前 9 失败/3 通过，
修复后 config/datasets 745 通过；pre-commit（保护示例排除）、Ruff、
compileall、diff 检查通过。

Grain loader checkpoint 输入复核：load_state_dict 先校验 dict，避免 None/
空 list/空 str 静默跳过恢复，以及非空非法类型的 AttributeError；空 dict
既有行为保留。五项回归修复前全失败，拒绝后游标状态不变，data_pipeline
专项 282 通过。pre-commit（保护示例排除）、Ruff、compileall、diff 通过。

Synthetic loader checkpoint 输入复核：RandomTokenDataLoader 在空状态判断前
校验字典类型，修复 None/空列表/空字符串被静默忽略、非空列表或字符串触发
AttributeError 的问题。5 个新增回归在修复前全部失败；修复后检查拒绝保持
游标、已有 iterator 及后续两个 DP 分片批次的 input_ids/labels 与独立 loader
一致。空字典原有兼容行为保持不变。

本批累计验证（2026-10-14，当前未提交工作区，macOS/Python 3.13/PyTorch 2.14）：
`pytest -q --disable-warnings`：2916 passed、2 skipped、135 warnings；
日志 `/tmp/llmtuner-synthetic-state-full.log`。pre-commit 对 tracked/untracked
文件执行，排除受保护的 `examples/native_alltoall.py`，全部通过，日志
`/tmp/llmtuner-synthetic-state-precommit.log`；Ruff、compileall 与
`git diff HEAD --check` 通过。此结果不覆盖目标加速设备上的并行通信或重叠。
后续继续核对 loader 恢复契约及调用链；custom fine-grained FSDP、MC2 等
既有实现缺口和目标设备验证仍待完成。

Streaming cursor 资源清理复核：Grain DatasetIterator.close 仅关闭 parents，
HF cursor 没有 parent，原实现不会关闭持有的源 generator。新增 close 显式
释放支持 close 的底层 iterator，并在 finally 中执行 Grain 的关闭标记。
回归确认主动关闭立即执行 generator finally、重复关闭幂等、清理失败保留
原异常且禁止后续读取；使用真实 datasets.IterableDataset.from_generator
验证源清理传递。第一个用例修复前失败。

验证：当前工作区全量 pytest 2918 passed、2 skipped（27.50s，日志
`/tmp/llmtuner-stream-close-full.log`）；随后增加真实 HF 参数化用例，3 个
资源清理用例全部通过，该新增参数尚未计入上述全量数量。受影响文件
pre-commit、compileall 与 HEAD diff check 通过。证据覆盖主动关闭及失败
行为，不证明任意第三方源均能被关闭；没有 close 方法的源沿用原有处理。
恢复替换旧 iterator 的生命周期仍需接续核对。

Streaming 恢复替换 iterator 的生命周期复核：旧流可能在 generator
finally 中修改源状态，原 set_state 先 restore 后替换 iterator，可能被旧流
延迟清理覆盖。现在校验有效输入后先显式关闭旧流，再 set_epoch/load_state_dict
并构造新流；关闭、状态恢复或 iterator 构造失败时保留原异常，将 Grain cursor
标记关闭，禁止继续读取和再次恢复。非法外层状态仍在关闭旧流前拒绝。

新增源清理修改游标的回归，修复前失败；持有旧 iterator 引用确认主动释放及
恢复后首行正确。另覆盖 close/restore/iterate 三处失败和终止状态。已有真实
HF 恢复及重复 epoch 用例通过。此行为不提供失败恢复事务回滚：下层恢复失败
后 cursor 终止，需新建 loader，与 GrainDataLoader 的恢复失败清理策略一致。

当前累计验证：全量 pytest 2923 passed、2 skipped、125 warnings（27.49s），
日志 `/tmp/llmtuner-stream-restore-full.log`；受影响源文件与测试 pre-commit、
Ruff、格式检查、compileall、HEAD diff check 通过。目标设备并行数值和既有
迁移实现缺口未由这些 CPU 测试覆盖。后续优先回到训练恢复与并行实现缺口。

FSDP grouped state prefetch 复核：PyTorch 2.14 的 grouped fully_shard
将同一 FSDPState 绑定多个模块，set_modules_to_*_prefetch 设置 state 字段。
原窗口规划只检查模块对象身份，decoder 的 tied embedding/norm/head 会覆盖
同一 state 的窗口，并可将该 state 设为自身目标。现按实际 FSDPState 身份
去重，保留给定执行序中的首次出现位置；窗口计数对应独立 state。
回归使用真实 FSDPModule setter 及模拟 shared state，修复前失败，修复后通过；
该用例证明配置语义，不证明通信重叠。CPU contract 57 passed。

两 rank CPU/Gloo pattern_fsdp_equivalence --prefetch-window 5：native/custom
适配器各 5 种共享参数/嵌套布局共 10 配置通过（输出/loss/梯度/SGD 更新）；
日志 `/tmp/llmtuner-prefetch-state-distributed.log`。全量 pytest 2924 passed、
2 skipped、135 warnings，日志 `/tmp/llmtuner-prefetch-state-full.log`；受影响
文件 pre-commit、compileall、HEAD diff check 通过。

重读 FSDPTurbo custom state/param_group/fully_shard 与安装 PyTorch 2.14：
上游仍依赖单个 _fsdp_param_group、reduce_scatter_state 和已移除的
_finalize_backward；当前版本为 groups 列表、RS 队列及逐 group finalize。
完整迁移仍须适配这些契约、hook relocation 和所有 comm context 的根回调
等待/清理，当前 custom 原生适配器不具备该能力。本批不将其标为完成。

Decoder grouped FSDP 真实分布式回归补齐：扩展现有
pattern_fsdp_equivalence.py，新增 decoder 入口布局，以 tied embedding、norm、
head 的真实 grouped fully_shard state 验证 prefetch 去重。显式断言三个模块
共享 state，shared forward 只指向 block，block backward 只指向 shared，
无自身目标和后续模块覆盖。独立 dense reference 对比三次 SGD 的 loss、
完整梯度与更新后参数，容差 atol=2e-6/rtol=2e-5。

两 rank CPU/Gloo、prefetch-window=5：native/custom adapter 各 6 布局，
共 12 配置通过；日志 `/tmp/llmtuner-decoder-prefetch.log`，命令
`PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1 python -m
 torch.distributed.run --nproc_per_node=2 --master_addr=127.0.0.1
 --master_port=29877 tests/integration_tests/pattern_fsdp_equivalence.py
 --prefetch-window 5`（执行时连接为一条命令）。脚本 pre-commit 与 HEAD diff
检查通过。本批仅扩展 integration，不改生产行为；上批全量 CPU 数量不作为
此次分布式覆盖证据。仍未验证 GPU stream overlap 或完整 custom 引擎。

RNG checkpoint 元数据边界复核：原 version/world_size 使用等值比较，
True 与 1.0 在单进程被当成整数 1 接受并执行随机流回退。新增正整数类型
校验拒绝 bool/float，外层 state 和反序列化 payload 必须为字典；空字典兼容
保持。9 个外层/元数据回归修复前全失败，检查非法 metadata 不改变 Python、
NumPy、CPU Torch 随机流；另 5 个非字典 payload 回归通过。

验证：RNG 单测 29 passed；全量 pytest 2933 passed、2 skipped、140 warnings，
日志 `/tmp/llmtuner-rng-metadata-full.log`（随后新增的 5 个 payload 参数仅
计入 RNG 单测，未计入该全量数字）。CPU/Gloo 两 rank RNG checkpoint 等价
脚本 DDP/FSDP × sync/async 通过，恢复后两个 stochastic AdamW 更新的 loss、
参数与优化器状态精确一致；日志 `/tmp/llmtuner-rng-metadata-distributed.log`。
受影响文件 pre-commit、compileall、HEAD diff check 通过。默认 CPU 随机流
证据不覆盖目标设备 DTensor RNG。后续继续 topology metadata 及目标设备恢复。

RNG topology 元数据类型复核：parallel_degrees 的字典等值比较同样
允许 True/1.0 伪装成整数度数。新增所有保存度数的正整数校验，在随机流
修改前拒绝 bool/float；None 仍用于没有 topology 的既有 checkpoint。
8 轴 × 2 非整数值共 16 回归修复前全部失败，修复后拒绝且 CPU 流不回退。

当前全量 pytest：2954 passed、2 skipped、125 warnings（27.50s），日志
`/tmp/llmtuner-rng-topology-full.log`。真实 MPS device_rng_equivalence.py mps
通过精确随机流续接及非法状态隔离，日志 `/tmp/llmtuner-rng-topology-mps.log`。
受影响文件 pre-commit、compileall、HEAD diff check 通过。MPS 证据仅覆盖
设备默认 generator，不覆盖 CUDA/NPU DTensor 随机 kernel、混合并行训练
或完整 custom FSDP；这些待办继续保留。

EMA 非法 checkpoint 冷启动复核：load_state_dict 将任意 falsy 输入
当作缺失 EMA 数据，None/空列表/空字符串会覆盖全部历史平均值。现先要求
字典，只有空字典保留已有冷启动语义。5 个非字典输入回归修复前全部失败，
修复后确认拒绝不改 averages，且下一次真实 EMA firing 仍符合独立 lerp 公式。

当前全量 pytest 2959 passed、2 skipped、143 warnings（27.29s），日志
`/tmp/llmtuner-ema-state-full.log`。两 rank CPU/Gloo
 ep_checkpoint_ac_equivalence.py 验证带/不带 activation wrapper 的 EP2 恢复
及 EP2→EP1 的模型/AdamW/EMA DCP 恢复精确一致，日志
`/tmp/llmtuner-ema-state-ep.log`。受影响文件 pre-commit、compileall、HEAD diff
检查通过。这不是目标设备通信或完整 trainer resume 的覆盖；全局迁移及
设备验证缺口仍保留。

PP Trainer 恢复完整状态断言补齐：pp_checkpoint_equivalence.py 原最终
对比仅 model/EMA/token/loss，optimizer 只检查非空 moments。新增连续训练最终
optimizer 完整 flat state、scheduler state 和 trainer state 保存/逐项精确比较。
两次独立 torchrun 进程启动，PP2/1F1B/4 microbatch、2+2 steps、动态 EMA：
synthetic 与 packed JSONL 两路径均通过，恢复 loss bitwise 相同且所有新增
最终状态断言通过。日志 `/tmp/llmtuner-pp-resume-states.log` 与
`/tmp/llmtuner-pp-packed-resume-states.log`。命令使用 CPU/Gloo、
TORCHDYNAMO_DISABLE=1、LLMTUNER_TEST_EMA=dynamic，脚本 full/resume 两阶段。

packed CPU attention 使用既有 dense-mask SDPA substitute，此证据不覆盖
生产 CUDA flex kernel、随机 dropout 或其他 PP schedules。本批只增强验证
脚本，pre-commit/HEAD diff check 通过，未重复全量 CPU suite。全局实现缺口
及目标设备验证继续保留，后续推进未完成并行能力生命周期。

Pattern FSDP EP1 expert placement 对齐：decoder 默认在 shard degree 大于
expert 数时使用 Shard(1)，pattern 路径此前仅处理 EP>1，默认 Shard(0) 会
沿 expert 轴产生 padding。补齐未启用 EP 的默认 feature placement，普通
参数返回 None 保留原生默认；显式 shard_placement_fn override 优先级不变。
单测修复前缺少 callback 失败，修复后通过；FSDP contract 58 passed。

扩展 pattern_fsdp_equivalence，真实单 expert/两 rank FSDP 断言 expert 轴
完整、placement=Shard(1)，并与独立 dense reference 比较三次 SGD 的 loss、
梯度及参数。native/custom adapter 各 7 布局共 14 配置通过，prefetch window
5，日志 `/tmp/llmtuner-pattern-expert-axis.log`。全量 pytest 2960 passed、
2 skipped、132 warnings，日志 `/tmp/llmtuner-pattern-expert-full.log`；受影响
文件 pre-commit、compileall、HEAD diff check 通过。本批对齐默认 dense mesh
策略；逐 unit mesh override 与专家 placement 的交互仍需核对。CPU/Gloo
结果不证明设备通信重叠或完整 custom 引擎迁移。

Pattern FSDP per-unit mesh override 复核：EP1 expert feature placement
此前按全局 dp_mesh 计算，覆盖 unit mesh 后仍沿用旧 degree。回归覆盖 base/unit
度数 (2,2)、(1,2)、(2,1)，后两种修复前失败。现在保留专家参数/数量映射，
对每个 grouped unit 最终生效的 options.mesh 计算 placement，普通参数使用
None 默认；EP>1 保持 sparse mesh degree。contract 60 passed。

累计全量 pytest 2962 passed、2 skipped、123 warnings（26.84s），日志
`/tmp/llmtuner-unit-mesh-full.log`；两 rank CPU/Gloo pattern FSDP 14 布局配置
重跑通过，日志 `/tmp/llmtuner-unit-mesh-distributed.log`。真实分布式脚本覆盖
默认 mesh 数值；不同 unit mesh 度数的选择仅由隔离 contract 证明，不能扩展
为跨 mesh 训练数值已验证。受影响文件 pre-commit、compileall、HEAD diff
检查通过。完整 custom 引擎和目标设备验证仍未完成。

Per-unit mesh override 分布式数值补齐：pattern_fsdp_equivalence 新增
SingleExpertDecoder 的 block override 为 HSDP mesh (dp_replicate=2,dp_shard=1)，
其余 dense unit 仍 FSDP shard=2。真实 DTensor 断言 expert mesh shape=(2,1)、
local shape=global shape；相同归约 ranks 保证比较的是布局变化而非数据所有权
改变。两 rank CPU/Gloo、prefetch-window=5、native/custom adapter 各8布局
共16配置通过，3次 SGD loss/梯度/参数与独立 dense reference 一致。
日志 `/tmp/llmtuner-unit-hsdp.log`；脚本 pre-commit/HEAD diff检查通过。
本批仅增强 integration，未改生产代码。此证据覆盖明确 HSDP override，
不推广为任意 rank subgroup、任意自定义 mesh 或 GPU overlap 已验证。

累计审查状态复核（2026-10-14）：实际 llmtuner/**/*.py 154 文件与源码
清单154条一一对应，无缺失/废弃/重复条目；文件清单不代表完整运行时验证。
ignored 参数全局/逐 unit 合并与共享组选项冲突的现有针对性用例10 passed，
未发现新生产缺陷，不为无问题检查增加实现修改。当前全部 tracked/untracked
文件 pre-commit 通过，排除受保护 examples/native_alltoall.py；日志
`/tmp/llmtuner-cumulative-precommit.log`，HEAD diff check 通过。

最近多批恢复/FSDP修复累计门禁已核对；全局验收仍被实际实现缺口和设备
验证缺口否定。下一批应优先实现 custom FSDP 的 hook relocation/多通信上下文
生命周期，避免继续用原生适配器的通过结果替代迁移完成。

Custom FSDP 生命周期静态复核：当前适配器已明确拒绝 distinct
hook_module，并在 pattern 变换前校验 hook/未知 override，避免部分模块已
转换后才失败；PyTorch 2.14 原生 state 的 lazy init 会发现 grouped state、
FSDP param groups 和根 callback。针对性 contract（native/custom strategy、
distinct hook、invalid override）通过；这只是拒绝和原生契约检查，不是
FSDPTurbo custom hook relocation 的实现。上游多通信 context、独立 hook
目标、root callback 全局等待/清理仍需真正迁移，不能用当前 adapter 结论替代。

CP 装配失败路径复核：apply_cp 原逐层构造并挂载 kernel，后层缺少
attention 或 kernel 初始化失败时前层已修改。现先收集所有 attention，构造
全部 kernel 后统一挂载；原错误信息保持。缺少第二层 attention 的回归修复前
失败；新增第二个 kernel 构造失败用例确认旧 kernel 身份不变、mesh未设置。
CP 单测23 passed；这是隔离装配失败证据，不证明 CP collective 数值或任意
set_cp_mesh 异常的全局事务回滚。

全量 pytest 2963 passed、2 skipped、127 warnings，日志
`/tmp/llmtuner-cp-assembly-full.log`，随后添加 kernel 构造失败参数单独通过，
未计入该数量。pre-commit 首次指出未使用循环变量，已修正并重跑全部通过；
compileall/HEAD diff check通过。完整 custom FSDP 等全局缺口继续保留。

双 CP 子轴 mesh 度数复核：apply_cp 原只校验合并 cp mesh，总度数正确
不能保证 cp_kv/cp_ulysses 各自与配置一致。新增子轴实际 mesh.size 校验，
在 Ulysses head 校验、kernel 构造和模型修改前拒绝不一致 topology。
两轴各一错度数回归修复前失败，修复后 CP 单测25 passed；证据为隔离
入口校验，未证明子轴 rank membership 或双 CP attention 数值。

全量 pytest 2966 passed、2 skipped、132 warnings（27.44s），日志
`/tmp/llmtuner-dual-cp-degree-full.log`；受影响文件 pre-commit、compileall、
HEAD diff check 通过。双 CP 目标设备数值与完整 custom FSDP 实现缺口继续
保留，不能因增加拒绝校验标记迁移完成。

双 CP rank membership 真实验证：新增 dual_cp_topology.py，4 rank
CPU/Gloo，cp_kv=2/cp_ulysses=2，检查合并 cp 顺序[0,1,2,3]、KV groups
[0,2]/[1,3]、Ulysses groups[0,1]/[2,3]及local rank。独立rank公式检查两轴
交集只有本rank，各组all_reduce SUM与独立rank和一致，运行通过。
日志 `/tmp/llmtuner-dual-cp-topology.log`，启动 --nproc_per_node=4、
master_port=29885、GLOO_SOCKET_IFNAME=lo0。脚本docstring供现有runner发现
rank数，pre-commit/HEAD diff check通过。本批仅新增integration；不证明
DualCPFlexKernel前向/反向数值或目标设备性能，完整custom FSDP等缺口保留。

双 CP collective 数值补齐：新增 dual_cp_attention_equivalence.py，
四rank CPU/Gloo KV2×Ulysses2、float64；真实 DualCPFlexKernel 的 SeqToHead、
KV all-gather、HeadToSeq 及其 autograd 保持不变，仅 run_flex 替换为 SDPA。
独立全序列SDPA作为参考，causal/noncausal 输出和Q/K/V输入梯度均通过
atol=rtol=1e-10。mask切片使用独立 KV-major rank 公式，未复用实现的mask。
日志 `/tmp/llmtuner-dual-cp-attention.log`，启动4rank/port29886。
pre-commit 首次B023闭包检查失败，显式绑定mask后重跑通过；HEAD diff检查
通过。本批仅新增验证。该证据证明此布局的通信数值，仍不证明生产flex mask、
packed/headtail路径、目标GPU kernel或性能，完整custom FSDP等缺口保留。

双 CP mask 索引复核：在既有四 rank 拓扑脚本中加入真实 512-token causal
BlockMask，使用 cp_kv mesh 的 Q shard；每 rank 的 mask.seq_lengths=(256,512)，
并以独立 global query offset `(rank//2)*256` 检查 mask_mod 与预期 causal
关系逐元素一致。该检查覆盖 KV-major mask 轴和 Ulysses-inner rank ordering，
运行日志 `/tmp/llmtuner-dual-cp-mask.log`，pre-commit/HEAD diff check通过。
仍不覆盖 packed/headtail mask、Ulysses full-sequence mask 或 flex GPU kernel；
因此仅证明该 dual CP mask 索引布局，不扩大支持矩阵。

双 CP causal mask 模型调用复核：真实四rank拓扑脚本调用
HFTransformerModel._get_cp_attention_masks，通过隔离模型context向其提供128
本地positions；断言该方法按总CP4构造512全局positions，再按KV2生成
(256,512) mask，mask_mod逐元素符合独立全局query索引。原生BlockMask与
真实mesh shard保持，模型实例仅用stub隔离非相关构建。运行通过，日志
`/tmp/llmtuner-dual-cp-model-mask.log`，pre-commit/HEAD diff check通过。
未发现该causal长度推导缺陷，未改生产代码。测试不覆盖完整HF forward、
packed数据或GPU flex内核，下一步需继续这些实际调用路径。

双 CP packed mask 索引补齐：四rank拓扑脚本新增长度190/145/177三文档，
边界跨越128输入shard和256 query shard。生产 get_document_mask_mod 与 causal
组合生成真实BlockMask，再按KV2分片；独立document_ids+global query公式
逐元素比较通过，并断言结果不同于纯causal以防文档mask被忽略。
日志 `/tmp/llmtuner-dual-cp-packed-mask.log`，pre-commit/HEAD diff check通过。
只证明mask构造及索引，未覆盖HF packed preprocess端到端或生产GPU attention。
本批仅扩展验证，无生产代码修改；全局目标和迁移缺口仍未完成。

双 CP packed SDPA-substitution 数值验证：现有attention等价脚本新增
5/6/5-token三文档，边界跨4-token输入及8-token query shard；独立dense
文档隔离causal mask，实际双CP通信和autograd不变。四rankCPU/Gloo的
noncausal/causal/packed输出、Q/K/V梯度均通过1e-10容差，日志
`/tmp/llmtuner-dual-cp-packed-attention.log`。pre-commit/HEAD diff check通过。
本批仅扩展验证，packed mask直接由独立document ids提供，不证明生产
BlockMask到attention的完整集成；GPU flex kernel与完整custom FSDP仍未完成。

双 CP GQA 通信数值复核：SDPA-substitution 脚本新增Q4/KV2，和既有
Q4/KV4 × noncausal/causal/packed合计6配置。真实四rank CPU/Gloo通信及反向
不变，独立dense SDPA enable_gqa参考，输出与Q/K/V梯度均通过1e-10容差，
日志 `/tmp/llmtuner-dual-cp-gqa.log`。本批仅扩展测试，不证明生产flex GQA
内核或完整HF preprocess集成。pre-commit长度问题修正后复核；全局目标未完成。

双 CP 支持矩阵证据同步（2026-10-15）：复核当前 DualCPFlexKernel 等价脚本，
四 rank CPU/Gloo、float64、KV2×Ulysses2 的六个 mask/GQA 配置均通过，
输出及 Q/K/V 梯度与独立全序列 SDPA 的容差为 1e-10。可复现命令及替换
attention 内核的限制已补入 distributed_implementation_comparison.md；新增
双轴能力行，分别列出真实拓扑/BlockMask 索引和通信数值证据。生产 flex、
目标设备及完整模型验证仍待完成；custom FSDP 仍为原生适配器。

Custom FSDP 实际状态迁移（2026-10-15）：custom 路径改为独立 composable
contract，新增 FineGrainedFSDPState，复用 PyTorch 2.14 初始化 helper、
多参数组和 collective。允许 ancestor hook，pattern 先验证所有单元与共享
分组边界；不混用 native/custom 引擎，要求先执行 custom root。按同一 hook
下 unit 序号分配独立 buffer/event 上下文；通信 streams 和全局 backward
prefetch 顺序共享，前向收尾、反向 final callback 和失败 reset 清理全部
context。该实现已超出原生 adapter，但跨 hook 提前释放等上游优化仍待核对。

新增 fine_grained_fsdp_equivalence.py：两 rank CPU/Gloo，16 个配置覆盖
grad/no-grad 输入、reshard、non-reentrant checkpoint、单/双参数 mesh；
每 parent 两 unit、两 parent hook，forward 故障 reset 后执行双 microbatch
累积及3次SGD。独立 dense 输出、输入/参数梯度和更新参数一致（参数容差
2e-6/2e-5），真实 context/状态清理断言通过。命令：
PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1 python -m
torch.distributed.run --nproc_per_node=2 --master_addr=127.0.0.1
--master_port=29851 tests/integration_tests/fine_grained_fsdp_equivalence.py。
原有 pattern 脚本 --prefetch-window 5 的 native/custom 共16配置通过；
fsdp_checkpoint_equivalence.py 的8个 shard/replicate、native/custom场景
通过（同拓扑 bitwise，跨拓扑容差1e-5）。后者只覆盖默认hook，不能推断
relocated-hook恢复。源文件清单核对155条；目标设备、混合精度、Inductor、
多 outstanding forward、backward故障与relocated-hook DCP继续待完成。

Custom 引擎导入边界补齐：项目允许 torch>=2.12；fine_grained.py 仅在
custom 调用时加载 fine_grained_state.py 的2.14私有接口，原生策略保持
轻量导入。缺少接口或多参数组/reset能力时给出 EnvironmentUnsupportedError
和 native 选择提示。新增独立进程回归阻断 custom engine 导入，验证 native
选择仍成功、custom请求明确失败；该证据不替代真实PyTorch2.12运行。

Custom 状态迁移累计门禁：最终 CPU 单测2969 passed、2 skipped（28.29s）；
全仓 pre-commit 使用 tracked+适用untracked 文件列表，明确排除受保护的
examples/native_alltoall.py，全部通过。compileall 与 git diff HEAD --check
通过。懒加载改动后16配置 relocated-hook 等价脚本复跑通过；最终8配置
默认hook DCP脚本复跑通过，可复现命令：PYTHONPATH=.
GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1 python -m torch.distributed.run
--nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29855
 tests/integration_tests/fsdp_checkpoint_equivalence.py。
下一行动：核对 FSDPTurbo param_group.py 的跨 hook 提前释放与梯度事件
所有权，并补齐 relocated-hook 多上下文的新运行时 DCP 恢复证据；目标设备
验证及MC2/Domino实际机制等全局剩余项保持未完成。没有 commit/push。

Custom 参数组跨 hook 生命周期迁移（2026-10-15）：新增
fine_grained_param_group.py，按2.14 process group/replicate group分组，构造
FineGrainedFSDPParamGroup；未 monkeypatch torch 或替换已初始化对象。
all-gather/reduce-scatter retained state 增加弱引用 hook owner；其他hook的
缓冲先等待 event 再释放，同hook保留，local RS queue cap由原生代码控制。
保留2.14 collective、多参数组索引、post-reduce events及HSDP buffer所有权。
四个回归单测以弱引用验证缓冲存活到wait、同hook保留、nonforward/no-sync
路径和新结果owner。mock event证据不代表目标设备实际通信或性能。

Relocated-hook DCP恢复：原等价脚本改为momentum SGD，step2实际DCP保存，
用seed999/lr0.3新建模型与优化器，加载后继续step3；恢复参数、输出、输入/
参数梯度及momentum与连续训练逐位一致，16配置CPU/Gloo通过。独立dense
三步参数与momentum亦通过2e-6/2e-5比较。它使用PyTorch get/set_state_dict
与DCP，尚未覆盖Trainer/checkpoint manager装配。运行命令：PYTHONPATH=.
GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1 python -m torch.distributed.run
--nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29857
 tests/integration_tests/fine_grained_fsdp_equivalence.py。
原pattern脚本--prefetch-window 5的native/custom16布局复跑通过。源码清单
与磁盘集合已同步156路径；设备、混合精度、backward失败、多outstanding
forward与custom组合矩阵继续待验证。

Custom 参数组累计验证收尾：完整CPU单测2973 passed、2 skipped（28.52s），
67项FSDP contract通过。tracked+适用untracked的全仓pre-commit全部通过，
排除受保护examples/native_alltoall.py；compileall与HEAD diff check通过。
custom参数组替换后默认hook checkpoint脚本8配置复跑通过（29860端口），
同拓扑bitwise、跨拓扑与dense参考1e-5。最新relocated-hook脚本16配置复跑
通过（29859），新增每parent forward中只允许当前hook的all-gather retained
buffer，验证真实路径owner与跨hook清理。下一行动：custom×EP sparse mesh
与Trainer checkpoint消费者，随后backward失败和多outstanding forward。
目标设备、MC2融合和Domino stream overlap等全局剩余项保持未完成。

Custom FSDP × EP稀疏mesh补证（2026-10-15）：既有ep_fsdp_equivalence.py
新增--fsdp-implementation native/custom，保留原生默认并按需导入custom
状态类型。custom断言每层确实使用FineGrainedFSDPState与两个
FineGrainedFSDPParamGroup，避免名义custom或单mesh测试假通过。
四rank CPU/Gloo，EP2×FSDP4，默认decoder和--pattern-fsdp两路径均通过；
loss311.874268 vs独立HF参考311.874283，router/专家梯度容差1e-4，
expert ranks[0,2]/[1,3]与dense[0,1,2,3]正确。命令：PYTHONPATH=.
GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1 python -m torch.distributed.run
--nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29863
 tests/integration_tests/ep_fsdp_equivalence.py --fsdp-implementation custom；
pattern复跑使用29864并加--pattern-fsdp。脚本pre-commit通过。
该证据为eval forward/backward，不涵盖aux、更新、relocated hook、
Trainer checkpoint或目标设备。生产实现本轮未改，不重复完整CPU套件。
下一步继续custom Trainer恢复及backward异常/多forward生命周期。

Custom backward异常reset缺陷修复（2026-10-15）：集成脚本在输入grad hook
抛异常，此时模型参数梯度已产生，root final callback尚未收尾。修复前
reset后参数.grad仍非None，两个rank断言失败。原生group reset仅等待事件与
reshard；custom参数组现先执行该等待，再同步offload事件、清除sharded/
unsharded grad和unsharded_accumulated_grad，遵守失败迭代丢弃契约。
16配置真实CPU/Gloo复跑通过，含异常reset后独立dense三步momentum训练与
DCP新运行时恢复。命令为既有fine_grained_fsdp_equivalence.py，2rank，
loopback/Gloo/Dynamo disable，端口29866。生产与脚本pre-commit通过。
这不证明rank不一致异常可恢复；collective中单rank失败仍需进程级处理。

Custom多outstanding forward补证（2026-10-15）：既有等价脚本新增
--outstanding-forwards，两个microbatch先完成forward，再逆序分别backward。
16个grad-input/reshard/AC/多参数mesh配置全部通过，独立dense梯度、三步
momentum更新及新运行时DCP恢复仍一致；确认每轮root/context收尾状态。
命令：PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1 python -m
 torch.distributed.run --nproc_per_node=2 --master_addr=127.0.0.1
--master_port=29867 tests/integration_tests/fine_grained_fsdp_equivalence.py
--outstanding-forwards。仅CPU/Gloo两forward逆序证据，不代表任意调度或PP。
生产实现未改，复用既有2973通过基线；下一项是Trainer checkpoint消费者。

Custom relocated-hook checkpoint manager补证（2026-10-15）：既有
rng_checkpoint_equivalence.py新增custom同步/异步两个场景，两个linear unit
与root共用root hook，真实forward断言三个独立context。实际调用
DistributedCheckpointManager、OptimizersContainer、scheduler、RNG状态；
rank独立Python/NumPy/torch随机流与dropout，保存step2后继续两步，新运行时
恢复再两步。loss、模型、AdamW和scheduler逐位一致，异步staging后继续
训练及close路径通过。两rankCPU/Gloo六场景通过；命令沿用该脚本、
loopback和Dynamo disable，端口29869。脚本pre-commit与HEAD diff check通过。
这是manager消费者证据，尚未创建Trainer，不归为Trainer装配验证。生产
代码未改，下一步真实Trainer custom FSDP恢复；设备证据仍缺失。

真实Trainer custom FSDP恢复（2026-10-15）：复用pp_checkpoint_equivalence.py，
新增环境选择LLMTUNER_TEST_PP_SIZE=1与LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom，
两rank自动推导FSDP2，保留原PP2默认。修正测试自身仅PP才存在的last-stage
属性访问，非PP两rank均记录归约loss；报告实际并行度而非固定pp2。
独立torchrun full/resume使用dynamic EMA，2+2训练步；loss逐位一致
（4.858562469482422、4.877256393432617），模型、optimizer、scheduler、
trainer计数及EMA最终状态精确比较通过。命令在两phase均设置PYTHONPATH=.
GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1及上述环境和
LLMTUNER_TEST_EMA=dynamic，2rank端口29871/29872；脚本参数full/resume与同一
临时workdir。覆盖实际Trainer装配与默认module hooks，未覆盖Trainer relocated
hook配置、目标设备或PP×FSDP。生产代码未改；下一步迁移剩余机制与混合矩阵。

Custom context初始化资源优化（2026-10-15）：额外context原先lazy_init
创建四个device streams再覆写root streams。改为直接共享root device handle、
四stream与prefetch顺序，独立初始化AG/RS队列和post-reduce事件字典。
新增回归断言三个context仅一次lazy_init、独立队列及各自buffer cap；
68项FSDP contract通过，完整CPU2974 passed、2 skipped（27.64s）。
16配置outstanding-forward/DCP脚本复跑通过（29873），修改范围pre-commit、
compileall及HEAD diff check通过。CPU断言证明分配路径，未测GPU资源/性能。
下一行动转向剩余dispatcher机制与并行声明一致性，保留目标设备缺口。

Domino模块职责迁移（2026-10-15）：双切片dispatch/combine编排从模型专家
权重文件搬到parallel/expert_parallel/domino_dispatch.py；更新block conversion、
CPU与分布式测试叶模块导入，删除旧定义和export。保留GroupedExperts与普通
RoutedExperts职责，未宣称stream overlap已实现。源码清单同步157路径。

Domino整次调用输入边界修复（2026-10-15）：scores/ids截断到第一片长度
时，旧实现第一片dispatch成功，第二片才报shape错误。现在在任一slice
bincount/dispatch前校验完整token/routing shape；两个回归断言截断scores
或ids均不进入第一片dispatch。该检查只验证本rank输入，不声称可协调
rank不一致配置异常。三项Domino CPU专项通过，合法collective数值另复跑。

MoE router/dispatcher组合契约（2026-10-15）：构造时检查dispatcher全局
expert count与router一致、top_k一致，缺少dispatcher给出明确ValueError。
两项回归覆盖专家数/top-k错配，防止到前向shape检查或counts通信才失败。
全量首次运行发现FP32 bias测试使用RoutedExperts(grouped,None)无效fixture，
改为合法LocalTokenDispatcher(256,1)，保留原dtype断言。重跑2978 passed、
2 skipped；修改范围pre-commit、compileall、HEAD diff check通过。未检查
inner_experts数量与全局数量相等，因为EP时它应为本rank专家数。

EP dispatcher调用链累计复核（2026-10-15）：block_conversion在变换前
检查num_experts%ep_size；token_dispatch counts矩阵到rank-major/expert-major
转换与TorchAO local padding/unpermute逐调用metadata核对。counts为一维向量，
分割view保持stride合法，无需为非连续输入盲目改reshape。Domino继续是双切片
顺序执行，真实stream overlap和MC2仍为实现缺口。全量源码清单157条与磁盘
精确匹配；全仓tracked+untracked pre-commit通过，明确排除受保护
examples/native_alltoall.py。复用最近完整CPU2978 passed、2 skipped，
本轮未改生产代码。下一项直接推进dispatcher机制，避免继续只增加边界检查。

Domino 通信发起与结果消费拆分（2026-10-15）：

- `token_dispatch.py` 新增每次交换独立持有的 `PendingAllToAllDispatch`，
  `begin_dispatch` 发起 token exchange，`finish_dispatch` 消费结果并重排；
  普通 `dispatch` 复用两阶段实现，保持既有调用契约。
- `domino_dispatch.py` 对 all-to-all EP 先发起两个切片，再逐片消费、计算和
  combine；本地 dispatcher 保留原路径。新增顺序回归验证两次 launch 均先于
  consume，真实分布式脚本验证不均匀/空输入、零接收、重计算的输出及梯度。
- 验证：相关 CPU 测试 34 passed（新增顺序测试前）；完整
  `pytest -q --disable-warnings` 为 2979 passed、2 skipped；
  `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1 python -m
  torch.distributed.run --nproc_per_node=2 --master_addr=127.0.0.1
  --master_port=29847 tests/integration_tests/ep_domino_equivalence.py`
  的 24 个 CPU/Gloo 配置均通过；compileall、diff check 和排除受保护
  `examples/native_alltoall.py` 的已跟踪/适用未跟踪文件 pre-commit 均通过。
- 未完成：count exchange 的主机同步仍存在；两个 token exchange 提前发起
  不证明通信/计算重叠。下一步先将两个切片的 counts 一并预处理，再实现独立
  accelerator 通信 stream、producer/consumer event 及反向生命周期；目标
  CUDA/NPU 数值与 profiler 验证仍待执行。

Domino 多切片 count 预处理（2026-10-15）：

- `token_dispatch.py::prepare_dispatch_counts` 将 `(slice, expert)` counts
  按 destination rank 打包为一次 all-to-all，接收后恢复 source rank、slice、
  local expert 布局；统一 blocking CPU copy 生成两片 split sizes。
  `DispatchCounts` 明确每片接收 counts 和发送/接收列表的所有权。
- `domino_dispatch.py` 在 token exchange 之前预处理两个切片；
  `begin_dispatch(prepared_counts=...)` 复用结果，避免两次数据交换之间插入
  第二次 count collective 和主机同步。普通单片 dispatch 保持既有路径。
- CPU 顺序测试检查 counts → launch0 → launch1 → consume0/combine0 →
  consume1/combine1；三 rank 布局单测明确检查 wire 打包、两片独立 counts
  以及各 rank split sizes。后者为 mock 布局证据，不是三 rank runtime 验证。
- 验证：`pytest -q --disable-warnings` 为 2980 passed、2 skipped；
  双进程脚本 `tests/integration_tests/ep_domino_equivalence.py` 使用
  `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1 python -m
  torch.distributed.run --nproc_per_node=2 --master_addr=127.0.0.1
  --master_port=29848` 的 24 个 CPU/Gloo 配置前向/输入、score、专家梯度均通过。
  pre-commit 已覆盖已跟踪和适用未跟踪文件，排除受保护示例；compileall 与
  HEAD diff check 通过。
- 未完成：一次 count 主机同步仍存在；GPU 两片 counts 的 bincount 动态长度
  也可能同步。下一步实现 CUDA 专用通信 stream、producer/consumer event
  和 allocator 生命周期，核对 autograd 跨 stream 与 collective 顺序，再用
  目标设备数值及 profiler 证明重叠。NPU 适配及运行时验证仍待完成。

Domino CUDA 第二片 dispatch 调度（2026-10-15）：

- `domino_dispatch.py::_forward_cuda` 增加每设备缓存的 CUDA comm stream。
  dispatch0/permute0 在调用方 stream；comm stream 在 expert0 前记录 producer
  等待，再执行 dispatch1/permute1 并记录 ready event。expert0/combine0 在
  caller stream，消费 input1 之前 wait event。collective 发起顺序固定为
  dispatch0 → dispatch1 → combine0 → combine1。
- 两片仍使用 PyTorch 原生可微 functional collective，跨 stream 反向依赖由
  autograd 管理。生产者输入/counts 以及第二片输出/metadata 分别 record_stream
  到跨 stream 消费方，避免 allocator 提前复用。新 stream 不进入 checkpoint。
- CPU mock 验证 wait/launch/consume/event 次序及双向存储 ownership 标记；
  这只证明 Python 调度逻辑。CUDA 自动反向同步、实际通信顺序与性能尚无
  目标设备证据；没有将其报告为完整 Domino pipeline 迁移。
- `ep_domino_equivalence.py` 增加 `--device cuda` 和 `--trace-dir`，CUDA 使用
  LOCAL_RANK 选设备和 NCCL；保持独立 dense 参考、FP64 容差 1e-9、空 rank、
  零接收、非连续路由和重计算验证。CPU trace 导出成功，两 rank 的 JSON
  均可解析且包含 all-to-all events，不代表 CUDA overlap 已成立。
- 验证：完整 CPU 为 2981 passed、2 skipped；24 个双进程 CPU/Gloo 配置
  在默认及 profiler 模式均通过。命令使用 `PYTHONPATH=.
  GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1 python -m torch.distributed.run
  --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29849
  tests/integration_tests/ep_domino_equivalence.py`；profiler 模式端口 29850
  并加 `--trace-dir /tmp/llmtuner-domino-cpu-traces`。
  全适用文件 pre-commit（排除受保护示例）、compileall 与 diff check 通过。
- 目标设备命令：`PYTHONPATH=. python -m torch.distributed.run
  --nproc_per_node=2 tests/integration_tests/ep_domino_equivalence.py
  --device cuda --trace-dir outputs/domino-traces`。需两张 CUDA/NCCL 设备，核对
  loss/输出/全部梯度，并检查 NCCL dispatch1 和 expert0 kernel 的真实交叠。
  当前机器 torch 2.14 无 CUDA。下一步补非默认 caller stream 和反向生命周期
  验证入口，并继续上游 combine/backward overlap 与 NPU 适配迁移。

Domino combine 通信 stream 与非默认 caller 验证入口（2026-10-15）：

- `_forward_cuda` 将两片 combine 放到 comm stream。expert0 输出生产后记录
  comm wait；第一片 combine 发起后，caller 只等待此前 dispatch1 的 ready
  event，再执行 expert1。expert1 生产后发起 combine1，caller 在最终 event
  等待两片返回结果，再拼接。布局及 collective 顺序未改变。
- expert 输出、第一片 metadata/input 跨 stream record ownership；第二片
  metadata 保持 comm 消费，input/counts 记录 caller，最终输出记录 caller。
  Native autograd 仍负责反向依赖；未声称已实现上游自定义反向重叠调度。
- 调度 mock 更新为检查两次 producer wait、两次 combine 的 comm 归属、
  caller 的 dispatch-ready/final-ready 等待以及 allocator 标记。
- 集成脚本增加 `--nondefault-stream`（需 `--device cuda`）。测试调用方在侧
  stream 执行 forward/backward，显式等待输入生产者并记录存储；返回 default
  stream 前同步，验证输出及梯度。用于暴露依赖默认 stream 的错误；本机未运行。
- 验证：完整 CPU 2981 passed、2 skipped；双进程 Gloo 24 个配置通过，
  使用 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1 python -m
  torch.distributed.run --nproc_per_node=2 --master_addr=127.0.0.1
  --master_port=29851 tests/integration_tests/ep_domino_equivalence.py`。
  pre-commit 覆盖全适用已跟踪/未跟踪文件（保护示例除外），compileall 和
  HEAD diff check 通过。
- 待验证命令：`PYTHONPATH=. python -m torch.distributed.run --nproc_per_node=2
  tests/integration_tests/ep_domino_equivalence.py --device cuda
  --nondefault-stream --trace-dir outputs/domino-side-stream-traces`。
  当前 CUDA 前向 dispatch/combine 调度已加入；实际 GPU 数值、反向 collective
  顺序、allocator 复用压力及 profiler 重叠仍待目标设备验证。NPU 和上游完整
  反向 pipeline 仍存在实现缺口。
- 后续机制定位：MC2 上游实际文件是
  `fsdp_turbo/distributed/expert_parallel/dispatcher_mc2.py`，NPU 算子与反向在
  `fsdp_turbo/ops/npu/grouped_matmul_mc2.py`，依赖 `npu_alltoallv_gmm`、
  `npu_gmm_alltoallv`、`npu_grouped_matmul` 及 HCCL comm name。下一项为迁移
  可选 NPU 算子 adapter 与 grouped expert 权重布局，替换现有无条件拒绝；
  保留缺依赖的可操作错误，禁止普通 all-to-all fallback 冒充融合实现。

MC2 融合 adapter 与 MoE 接线（2026-10-15）：

- 新增 `mc2_ops.py`，移植 FSDPTurbo `ops/npu/grouped_matmul_mc2.py` 的
  fused dispatch/projection 和 projection/combine 及其输入/权重反向。
  可选依赖在调用时加载，要求三项真实 torch_npu 算子和 HCCL comm name。
  移除本仓库不需要的算子内 shared-expert 分支，shared experts 仍由 MoE 管理。
- 新增 `mc2_dispatch.py::MC2RoutedExperts`，参考上游
  `distributed/expert_parallel/dispatcher_mc2.py`，用既有本地 expert 排序、
  一次 count exchange、fused gate/up、activation、fused down/combine、
  score/scatter 形成实际执行路径。gate/up 可微拼接，保留三个参数名称及 checkpoint
  布局。只接受 NPU FP16/BF16 与 EP group；没有 unfused/local fallback。
- 配置不再无条件拒绝 mc2；转换选择专用 RoutedExperts，并在实际构造检查
  optional fused ops。MC2 × TP 因投影 DTensor/collective 布局尚未迁移明确拒绝，
  不宣称该组合完成。默认 all-to-all 路径不导入 torch_npu。
- CPU 模拟测试对独立 F.linear 参考比较两项 adapter 的输出、输入/权重梯度，
  覆盖空专家和全空输入；实际模块模拟测试覆盖独立 w1/w3 拼接、router scores
  梯度和 state-dict 名称。缺包/缺算子错误与配置延迟环境检查均测试。
  模拟结果只证明 adapter 代数及接线，不证明真实 Ascend kernel 正确。
- 新增 `tests/integration_tests/ep_mc2_equivalence.py`：两 NPU/HCCL rank，
  FP16 对独立 CPU dense 参考比较输出、输入、score 及全专家参数梯度，容差
  rtol=1e-2/atol=5e-4；正常、不均匀、零接收、空 rank 和重计算配置。
  命令 `PYTHONPATH=. python -m torch.distributed.run --nproc_per_node=2
  tests/integration_tests/ep_mc2_equivalence.py`；需要真实 torch_npu 三算子。
- 验证：完整 CPU 为 2987 passed、2 skipped；全适用 pre-commit（保护示例
  除外）、compileall 和 diff check 通过。当前机器无 NPU，目标脚本未执行。
  当前源码清单 159 路径；迁移有实际 adapter/消费者，但目标设备数值、空 token
  kernel 支持、TP 组合、混合并行和性能验证仍未完成。下一步继续检查融合算子
  契约、模型转换失败时的原子性，并推进设备 adapter/组合边界。

MoE 整体模块共享与转换失败路径复核（2026-10-15）：

- 确认缺陷：不同 layer slot 引用同一个 HF MoE block 时，转换逐 slot 创建
  新 MoE，导致原有 router/expert 参数及统计状态被拆成独立副本。新增回归在
  修改前确实失败于两层 MoE 身份断言；数值初始化相同不足以证明共享关系保持。
- `model_replacement.py` 对原始 block 身份缓存转换结果，重复 slot 复用一个
  MoE。仍然先构造全部 replacement，再安装；返回值保持被替换 slot 数量。
  原有 AuxLoss 构造失败回滚与 shared-expert dtype/mode 保护继续有效。
- 回归验证完整 block 共享、非注册 moe 引用及 router 参数身份，并与共享
  原始 HF block 的独立模型比较前向；53 个模型转换测试通过，包含较晚 router
  校验失败、构造 KeyboardInterrupt、共享专家 dtype 和子模块模式保护。
- 验证：完整 CPU 2988 passed、2 skipped；全适用 pre-commit（受保护示例
  除外）、compileall、HEAD diff check 通过。此修复没有改变 collective；
  运行时证据是单进程 CPU，不代表共享 block 的所有 FSDP/TP/PP 组合已验证。
- 下一步：核对不同 MoE block 仅共享部分参数的情况，尤其 router gate 和 fused
  gate/up/down；本次修复只覆盖同一整体 block 多处引用。继续检查转换后的
  参数身份与 optimizer/checkpoint 契约；MC2/NPU 与 Domino/CUDA 的设备验证
  缺口继续保留。

不同 MoE block 的部分投影参数共享（2026-10-15）：

- 三项确认缺陷：两个不同 HF MoE block 仅共享 router weight、gate_up_proj
  或 down_proj 时，逐 block 构造会丢失共享。修正测试 fixture 的 aux float
  类型后，三项回归在实现修改前均失败于目标参数身份断言。
- `model_replacement.py` 在 replacement 构造阶段按原始源 tensor 身份记录
  新参数。router/down 一对一复用；fused gate/up 一对二映射，两个 Parameter
  必须成对复用。绑定发生在 dtype/device 转换后、任何 layer 安装前，原有
  construction 失败回滚保持。完整 block 缓存和 projection 缓存分别负责
  模块共享与参数共享。
- 三项 CPU 回归验证新参数身份、独立 HF 参考前向、共享参数的两层累积梯度
  和一次 SGD 更新。前向/更新 atol=rtol=1e-6，梯度 atol=1e-7/rtol=1e-5；
  gate/up gradient/value 拼接后与原始 fused 参数比较。
- 验证：完整 CPU 2991 passed、2 skipped；全适用已跟踪及未跟踪文件
  pre-commit（排除受保护示例）、compileall、HEAD diff check 通过。
  本次未改变 collective；不以 CPU 证明所有分布式共享参数组合正确。
- 覆盖边界：同一源 Parameter 在被转换 MoE blocks 间共享。不同 Parameter
  共用 storage/view、源参数同时被未转换模块使用，以及共享 router buffer/
  balancing 状态仍需核对。下一步沿 optimizer/checkpoint 的参数去重与
  跨模块所有权检查这些边界，继续剩余机制迁移和设备验证。

共享参数别名的 optimizer 分组与恢复（2026-10-15）：

- 确认缺陷：`_build_param_groups` 用去重 named_parameters 枚举，匹配共享
  参数非首个路径的 regex 会被错误判为 matched no parameters。回归在修改前
  明确失败；这会影响共享 MoE projection 及通用 tied 参数的 optimizer 配置。
- `optimizer.py` 枚举全部注册别名，参数组按 id 首次匹配认领，保证共享参数
  只进入一个 group。保存名称固定取首个注册路径再 canonical_fqn，避免切换
  pattern 的匹配别名就改变 optimizer checkpoint FQN。原有 canonical collision
  和跨 model-part 重复归属拒绝保持。
- 回归验证第二别名 pattern 接受、只有一个参数和 optimizer state；保存后
  新建 runtime 改用第一别名 pattern 与不同 lr，再恢复 checkpoint，下一步
  AdamW 更新与连续更新一致，且新模型参数身份共享保持。
- 验证：完整 CPU 为 2992 passed、2 skipped；适用已跟踪/未跟踪文件
  pre-commit（受保护示例除外）、compileall、HEAD diff check 通过。
  本次未改变通信，不据此宣称跨 PP part 共享参数或分布式恢复已验证。
- 下一步继续核对跨已转换/未转换模块的参数所有权、共享 router buffers，
  并处理此前列出的实际迁移机制与目标硬件验证缺口。

MoE 与未转换消费者的未变布局参数身份（2026-10-15）：

- 确认缺陷：router weight 和未分片 down projection 被转换为独立 Parameter，
  即使布局完全不变，也切断源参数被其他未转换模块使用时的共享关系。
  两项回归在实现修改前失败于转换参数与外部 consumer.weight 身份断言。
- `model_replacement.py` 的 source→target 参数映射在一对一转换且类型、
  shape、dtype、device 均相同时直接复用源 Parameter。可保留外部 consumer、
  原有 freeze 状态及参数元数据；gate/up 拆分、EP 真正分片或 tensor 类型变换
  不进入此分支，不将不兼容的原始布局塞入新模块。
- 回归验证 router 和无分片 down 的外部共享身份，并用两个消费者的平方 loss
  验证累积梯度等于 6×parameter；完整 MoE 转换测试 58 项通过，包含原有
  共享 projection 的独立 HF 前向/梯度/SGD 等价及构造失败保护。
- 验证：完整 CPU 2994 passed、2 skipped；全适用 pre-commit（受保护示例
  除外）、compileall、HEAD diff check 通过。未变布局身份保留不代表 fused
  gate/up 外部消费者、EP-sharded 外部消费者或跨 PP 共享已解决。
- balancing hook 现有 collect_moe_layers 已对整体 block 去重；不同 block
  共享 bias buffer 时需要共同统计与单一更新 ownership，不能仅把 bias 引用
  绑定起来。下一步先确认共享 bias 更新契约并补真实回归，再同步转换及 hook；
  保留 gate/up 外部共享、storage/view 共享及设备迁移的剩余事项。

共享 sign-based bias 的统计与单次更新 ownership（2026-10-15）：

- 确认缺陷：不同 MoE 共用 expert_bias_E 时，原 hook 按每层独立 sign 更新。
  单测 first=[8,0,0,0]、second=[0,3,3,3] 会错误抵消；共同 counts 应产生
  [-0.15,0.05,0.05,0.05] 一次 centered sign 修正。回归修改前失败。
- `bias_update.py::update_expert_bias` 按物理 MoE 去重，再按共享 bias 分组；
  所有权的 load_balance_coeff 必须相同，校验早于 collective/状态修改。
  保持既定 DP/CP/按 EP 决定 TP 的归约，然后合并该 bias 各独立 counter 的
  贡献，更新一次并清空所有消费者 counters；已共享 counter 不重复计入。
- HF 转换保存 source bias → 新 FP32 bias 的别名关系，仅对 sign 策略启用，
  各 MoE counters 保持独立。DeepSeek 转换回归验证两 router 共用 bias，
  hook 使用共同 counts 并清空两层统计。没有把 quantile histogram 的不同
  生命周期隐式套到本策略。
- 四进程 `quantile_bias_equivalence.py` 扩展共享 sign bias，用独立 all-gather
  参考检查 DP×CP、DP×TP(EP off/on) 的归约及更新。现有普通 sign、quantile
  归约/清空也通过。命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0
  TORCHDYNAMO_DISABLE=1 python -m torch.distributed.run --nproc_per_node=4
  --master_addr=127.0.0.1 --master_port=29853
  tests/integration_tests/quantile_bias_equivalence.py`。
- 验证过程纠正了测试里的不存在 MoE.num_experts 属性及首次错误使用
  nproc_per_node=2 的启动命令；没有修改参考规则来隐藏失败。最终完整 CPU
  2997 passed、2 skipped；适用全文件 pre-commit（保护示例除外）、compileall
  与 HEAD diff check 通过。
- 未完成：共享 quantile bias/histogram ownership 尚未迁移；外部 bias buffer
  与转换后 FP32 buffer 的身份、storage/view 别名、FSDP/PP 混合配置及真实
  CUDA/NPU 内核仍待验证/处理。下一步核对 quantile 聚合与恢复的共同状态。

共享 quantile bias 的 histogram 聚合和状态恢复（2026-10-15）：

- 确认缺陷：原 update_quantile_expert_bias 对共享 bias 逐 layer 估计，后层
  会用已更新的 bin frame 解读此前采集 histogram，并覆盖前层结果。共同
  histogram/原始 bias 参考单测在实现修改前失败，最大误差约 0.238。
- 按 bias 所有权聚合：物理 MoE 去重，验证 num_experts/top_k/num_bins
  相同后执行既定归约；每个独立 histogram 只贡献一次，共同估计仅执行一次，
  清空所有消费者 histogram 和 token counters。共享 histogram 不重复计入。
- HF 转换的 source-bias 别名映射扩展到 quantile 策略；histograms 保持各自
  scratch 所有权。DeepSeek sigmoid/group-free 两层转换回归验证共享 bias、
  聚合结果及新模型 state_dict/load_state_dict 恢复后的 bias 身份和值。
  histogram 本来为 nonpersistent scratch，更新边界清空后恢复模型保持零；
  此测试不是 DCP 或任意 accumulation 中点恢复验证。
- 四进程 Gloo 真实聚合扩展共享 quantile 用例：独立 gathered histogram
  totals 及共同原始 bias 作为参考，覆盖 DP×CP、DP×TP 的 EP off/on 三配置，
  同时保持 int64 大计数、普通 quantile 和共享 sign 归约/清空验证。
  命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1 python -m
  torch.distributed.run --nproc_per_node=4 --master_addr=127.0.0.1
  --master_port=29854 tests/integration_tests/quantile_bias_equivalence.py` 通过。
- 完整 CPU 2999 passed、2 skipped；全适用 pre-commit（受保护示例除外）、
  compileall 和 HEAD diff check 通过。
- 下一步核对共同状态跨 FSDP/PP 装配及 checkpoint alias 的实际恢复，补充
  policy 冲突和更新所有权复用的边界；保留外部 FP32 bias buffer、跨布局参数
  共享及 CUDA/NPU 迁移/验证的未完成项。

EMA 共享 buffer 别名选择及恢复（2026-10-15）：

- 复查 Trainer 的 finite→staging wait→optimizer→scheduler→EMA 顺序及梯度
  分类，没有在本次路径确认新的训练缺陷。继续 buffer consumers 时确认 EMA
  的 named_buffers 去重隐藏第二别名，使只匹配该路径的配置静默漏掉 buffer。
  新回归修改前失败，EMA 输出 matched no buffers 且 state 为空。
- `ema.py::_selected_buffers` 遍历全部别名，按 tensor 身份只选择一次，保留
  首个注册名称作为状态 FQN。构造和每次 update/cold-start 的身份检查复用
  相同选择契约，避免二者识别的 buffer scope 不一致。
- 回归：仅选择 second_bias，首个状态名 first_bias，decay=.5 更新零→2 后
  平均为1，证明没有重复平均。补充新 runtime 用 first_bias pattern 恢复
  checkpoint，再更新到4，平均与连续过程一致，物理 buffer 共享保持。
- 验证：EMA suite 66 passed；完整 CPU 3000 passed、2 skipped；随后新增的
  同测试恢复分支单独再跑 1 passed。全适用 pre-commit（保护示例除外）、
  compileall、HEAD diff check 通过，最终测试/文档范围重新执行 pre-commit。
- 下一步回到剩余实际机制：设备 adapter、Domino 反向 pipeline、MC2 TP 布局
  与 custom FSDP/PP 生命周期仍有实现或验证缺口；避免继续仅扩展稳定别名
  测试而延后迁移。共有状态跨 FSDP/PP 和真实 DCP 恢复保持未完成项。

PP × custom FSDP 反向收尾与恢复复核：

- 四 rank PP2×FSDP2 的真实 Trainer 暴露生产缺陷：PyTorch 2.14
  PipelineStage.perform_reduce_grad 使用 fully_shard.state(module)，不能取得
  自定义 composable API 注册的 state。partition.py 增加
  ComposablePipelineStage，通过 FSDPModule._get_fsdp_state() 获取实际状态，
  保留同步/reshard 标志、全部参数组 post_backward、root final callback 和
  最后梯度缩放的顺序。普通模块仍委托原生 stage；不修改全局 torch registry。
- pp_checkpoint_equivalence.py 支持四 rank，并仅由最后全局 rank 写 loss
  参考。恢复时直接 torch.equal(DTensor) 会在默认进程组归约；不同 PP stage
  状态条目数量不同导致测试 collective 错位。修复为逐 rank 比较本地 shard，
  同时核对 dtype、全局 shape、placement 和 mesh，最终统一归约失败状态。
  faulthandler 实际调用栈确认原卡点为测试 EMA 比较和最终 loss broadcast。
- 当前工作区验证：native/custom 均以 CPU/Gloo float32、1F1B、4 microbatch、
  dynamic EMA 完成四步训练，step2 DCP 恢复至新建 Trainer 再训练两步；loss
  max diff=0，模型、AdamW、scheduler、trainer计数、EMA逐位相同。
  使用 LLMTUNER_TEST_WORLD_SIZE=4、LLMTUNER_TEST_PP_SIZE=2、
  LLMTUNER_TEST_FSDP_IMPLEMENTATION=native/custom、LLMTUNER_TEST_EMA=dynamic，
  torchrun --nproc_per_node=4 执行 pp_checkpoint_equivalence.py full/resume。
- 本批全量 CPU：3000 passed、2 skipped、152 warnings；pipeline 单测39 passed。
  全部适用文件 pre-commit 通过（按既定例外排除 examples/native_alltoall.py），
  compileall、git diff HEAD --check 通过；文档补充后相关 pre-commit 再次通过。
  不推断其他 schedule、relocated hooks、混合精度、目标设备或 PP×CP 已验证。
  下一步审查 PP 多本地 stage 的 custom FSDP 收尾及相应 schedule 数值路径。

多本地 PP stage × custom FSDP 复核：

- 阅读 partition、pipeline_parallel、trainer/runtime/pipeline、FSDP state
  final callback、optimizer state 和 metrics rank 归属；对照本地 torchtitan
  pipeline_parallel.py 的 loop/V 分配和安装 PyTorch 2.14 的分离 input/weight
  backward、REDUCE_GRAD 行为。本批未确认新的生产缺陷，不做外观重构。
- checkpoint脚本原固定最后全局rank写参考不适用于V布局，改为现有
  get_metrics_rank接口统一解析full写入和resume广播；可用
  LLMTUNER_TEST_PP_SCHEDULE选择Interleaved1F1B/ZBVZeroBubble及六层模型。
- pp_equivalence支持四rank custom FSDP，按PP mesh local rank判断stage
  所有权；DTensor参数/梯度在所属mesh内full_tensor后比较。独立整模型参考
  按DP2各自四microbatch的实际行边界计算，保留全局token分母和AdamW/clip。
- 实际CPU/Gloo float32、PP2×custom FSDP2、每rank两个local stages：
  Interleaved1F1B与ZBVZeroBubble均四步等价通过，loss最大差异分别
  4.768e-7、9.537e-7；裁剪梯度、梯度范数及更新参数容差1e-5。
  两种调度分别step2 DCP保存、新建Trainer恢复再训练两步，模型、AdamW、
  scheduler、trainer及dynamic EMA逐位一致，loss差异0。
- 复现：LLMTUNER_TEST_WORLD_SIZE=4、LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom，
  torchrun --nproc_per_node=4 pp_equivalence.py <schedule>；恢复脚本另设置
  LLMTUNER_TEST_PP_SIZE=2、LLMTUNER_TEST_PP_SCHEDULE=<schedule>、
  LLMTUNER_TEST_EMA=dynamic，在相同独立workdir运行full/resume。
  不覆盖目标设备、AC或其他调度。下一步检查PP×CP实际训练调用链。

PP × CP 训练流程复核：

- 阅读 runtime/batch、runtime/pipeline、HF preprocess/_shard_for_cp/forward、
  CP entry/input_sharding/attention、FSDP dense mesh及loss归约。确认loader
  DP轴排除CP，microbatch先按行拆分再CP分片；全局有效token分母只DP归约，
  FSDP shard mesh包含CP，梯度缩放与CP平均归约配合。源码清单159条与实际
  159个Python文件一致，无缺漏及失效路径；清单覆盖不等于设备验证完成。
- pp_equivalence加入CP度数和序列长度场景选择；dense参考按真实loader DP
  degree划分microbatch，参数逻辑副本计数单独包括CP。最初测试扩展使用
  不存在的context_parallel_size字段，构造失败后修正为现有KV轴配置。
- 新增pp_cp_equivalence.py：四rank PP2×KV-CP2，seq256、四microbatch，
  保留真实Trainer、CP token/position/label及BlockMask分片、KV AllGather
  autograd、native/custom FSDP及PP schedule；仅attention计算替换为
  BlockMask.mask_mod展开的CPU SDPA。明确拒绝非CPU，不能证明生产flex
  sparse索引/内核、目标设备或headtail/packed/双CP组合正确。
- 实际custom FSDP四步独立dense比较：1F1B累积1、1F1B累积2、
  Interleaved1F1B累积1均通过，loss最大差异4.768e-7；梯度范数、裁剪后
  参数梯度及AdamW更新参数容差1e-5。没有新的生产缺陷证据，不做无收益搬运。
- 复现：PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1
  torchrun --nproc_per_node=4 tests/integration_tests/pp_cp_equivalence.py
  <1F1B/Interleaved1F1B> <accumulation>；默认custom，可用
  LLMTUNER_TEST_FSDP_IMPLEMENTATION=native切换。PP×CP恢复及目标设备仍待验证。
  native FSDP的1F1B累积1同样通过，loss最大差异4.768e-7。
  本批全量CPU3000 passed、2 skipped；全部适用文件pre-commit通过
  （排除受保护examples/native_alltoall.py），compileall与diff检查通过。
  下一步核对packed/headtail生产mask与PP预处理契约。

PP × CP headtail 边界复核（待确认差异）：

- 对照安装PyTorch _HeadTailLoadBalancer与_create_cp_block_mask，token按
  head/tail chunk重排；mask_mod同时恢复local Q和gathered KV的原始索引。
  将LLMTUNER_TEST_CP_LOAD_BALANCER接入pp_equivalence已有CP配置，支持真实
  headtail Trainer验证；保留原有1e-5断言，不扩大容差。
- 实际四rank custom FSDP、PP2×KV-CP2、1F1B四步CPU SDPA替代测试失败：
  loss最大差异4.768e-7、梯度与norm通过，但首层v_proj.weight更新参数
  最大差异1.239e-5，rank0/1均失败。第二次运行稳定复现，不能声明通过。
- 增加各rank失败收集、最大误差与对应梯度诊断，避免仅打印rank0的错误。
  尚未确认是数值重排/AdamW近零梯度放大还是生产机制缺陷；不得据此修改
  生产代码或宣布修复。下一行动检查首步最大偏差坐标梯度并用独立高精度
  或相同输入重排参考区分浮点误差与通信/布局错误。

Headtail 更新差异数值诊断：

- 首步最大参数偏差坐标梯度为-5.187757779e-9与-6.204572855e-9。
  使用实际lr=3e-4、eps=1e-8的AdamW首步公式
  lr*g/(abs(g)+eps)计算，两梯度预测差异1.239458475e-5，与实测1.239e-5
  吻合。说明该坐标参数差异由近零梯度放大产生，不能据此单独判定CP错误。
- pp_cp_equivalence新增LLMTUNER_TEST_CP_FP64_DIAGNOSTIC=1显式诊断模式：
  模型参数转换double，独立dense与PP loss均在double logits计算CE；保留
  FP32 scalar返回以匹配Trainer sentinel/collective契约。HF内部仍有FP32
  计算，因此这不是全路径纯FP64证明，也不能替代生产FP32验收。
- 仓库入口真实四rank headtail/custom/1F1B四步诊断通过：最大梯度差异
  5.373e-9、最大参数差异4.221e-7，loss差异4.768e-7。原FP32参数测试
  失败状态与1e-5断言保持不变；下一步用独立重排参考定位剩余梯度舍入来源。
- 临时诊断初版double scalar loss与非最后stage FP32 sentinel不一致，Gloo
  abort；修正测试scalar契约后通过，不将该人为注入失败计为生产缺陷。
- pp_equivalence报告所有rank最大梯度/参数误差，便于区分loss通过与更新
  偏差。本批相关pre-commit、compileall、diff检查通过；仅测试及文档修改，
  未重跑全量CPU单测，上一批3000 passed、2 skipped。未改生产逻辑。

Headtail 独立重排参考复核：

- 新增LLMTUNER_TEST_CP_REORDER_REFERENCE=1诊断选择。整模型参考用独立
  chunk公式pair(r,2*cp-1-r)重排K/V与dense mask列；Q/output仍为原始顺序。
  不调用生产load-balancer、CP sharder或collective，保持独立数值参考。
- 运行前对MHA/GQA用FP64验证该排列输出及Q/K/V梯度与未排列SDPA等价，
  容差1e-12。然后真实四rank PP2×KV-CP2 headtail/custom/1F1B四步FP32
  训练按原1e-5断言比较：loss最大差异4.768e-7、梯度1.043e-7、参数7.400e-6，
  通过。原未重排参考默认行为和失败记录保持，未扩大容差。
- 结合AdamW首步公式与double诊断，证据支持attention KV归约顺序改变
  近零梯度、AdamW放大的解释；不支持把1.239e-5直接当布局错误修生产代码。
  该结果仍是CPU SDPA替代，未验证生产flex sparse block索引与目标设备。
- 复现：LLMTUNER_TEST_CP_LOAD_BALANCER=headtail与
  LLMTUNER_TEST_CP_REORDER_REFERENCE=1，四rank运行pp_cp_equivalence.py 1F1B。
  下一步推进packed PP×CP预处理与mask覆盖，保留目标设备验收缺口。

Packed PP × CP 真实语料复核：

- 阅读packing/build、packed输入preprocess、PP行拆分与finite validation
  global loss/token归约。扩展pp_real_corpus_equivalence四rank PP2×KV-CP2，
  seq256、custom FSDP；模型初始/最终DTensor在所属mesh还原后与独立整模型
  参考比较。CPU保留生产CP mask predicate、分片/通信/autograd，只替代
  attention算术；CP1继续既有dense-mask SDPA路径。
- 首轮fixture语料48行不足增长后的完整token batch，finite validation
  报zero batches；按seq长度比例扩大至192行后通过，保留末尾全padding
  microbatch的non-vacuity检查，不修改生产loader耗尽或丢尾语义。
- 实际四rank PP2×KV-CP2/custom、1F1B、三步AdamW：loss轨迹
  [4.867416,4.458224,4.272079]，最终参数最大差异1.071e-6；有限验证参考
  4.178464，实际报告在1e-5内，恢复train模式与全padding microbatch检查通过。
  标签已由collator对齐，CP分片保持文档位置/标签/mask契约。
- 复现：LLMTUNER_TEST_CP_SIZE=2、LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom，
  PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1，四rank运行
  tests/integration_tests/pp_real_corpus_equivalence.py。全rank失败收集使报告
  不再仅依据rank0判断通过。生产flex sparse索引/目标设备、headtail packed、
  双CP与checkpoint恢复仍待验证。原CP1两rank基线通过，参数差异7.451e-9。
  本批全量CPU3000 passed、2 skipped；适用文件pre-commit（保护例外排除
  examples/native_alltoall.py）、compileall、diff检查均通过。
  下一步推进PP×CP packed的DCP恢复路径。

Packed PP × CP checkpoint恢复复核：

- 扩展pp_checkpoint_equivalence配置KV-CP度数及seq长度，并使packed CP路径
  使用真实CP attention backend/collective，CPU仅替代attention算术。检查
  DCP loader布局planner、rank RNG、模型/优化器/EMA state顺序，无新生产缺陷。
- 实际四rank PP2×KV-CP2/custom FSDP、1F1B四microbatch、seq256、packed
  local_jsonl、dynamic EMA：连续四步loss为
  [4.928006,4.527804,4.349355,4.267666]；step2 DCP在独立新进程Trainer恢复
  后两步loss逐位一致，差异0；各rank模型shard、AdamW、scheduler、trainer
  counters、EMA全部精确一致。真实Grain loader自动恢复，未手动fast-forward。
- 三个集成脚本的重复CPU flex/backend/SDPA patch配置合并至
  pp_cp_equivalence.cpu_cp_attention上下文，入口显式CPU限制；所有消费者
  直接导入该定义，共享测试替代机制不改变独立整模型参考。生产代码未改。
- 复现：LLMTUNER_TEST_WORLD_SIZE=4、LLMTUNER_TEST_CP_SIZE=2、
  LLMTUNER_TEST_SEQ_LEN=256、LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom、
  LLMTUNER_TEST_EMA=dynamic，四rank分别运行pp_checkpoint_equivalence.py
  full/resume <同一workdir> --packed。目标设备、其他schedule、headtail/双CP
  恢复仍待验证。合并测试上下文后同一恢复用例再次通过；适用文件pre-commit
  （排除保护文件）、compileall、diff检查通过。本轮未重跑全量CPU单测，
  上轮3000 passed、2 skipped。下一步检查共享参数跨PP与checkpoint所有权。

任意共享参数跨PP stage所有权修复：

- 已有apply_pp只检查embedding/lm_head tying。任意decoder层共享同一
  Parameter可被逐stage deepcopy静默复制，梯度/更新及checkpoint语义改变。
  新回归在修复前实际失败（进入deepcopy），同stage共享回归原已通过。
- partition.split_model_into_stages在任何local stage复制前遍历全局模块
  assignment，用实际Parameter身份记录stage所有权；跨stage给出
  UnsupportedCombinationError、两消费者FQN和stage编号，提示同stage放置。
  所有rank对同一模型和partition作对称检查，不依赖本rank归属。
  不将同rank多个stage误当可共享；各stage仍有独立FSDP/optimizer生命周期。
- 同stage共享继续支持，回归比较两层共享Q投影的完整前向、累计梯度和
  SGD更新与未分割模型一致，身份保持。参数storage view与共享buffer不由
  此身份检查覆盖。跨stage共享同步并未实现，提前拒绝仅修复静默语义破坏，
  不能表述为共享权重迁移完成。
- 验证：pipeline单测41 passed；全量CPU3002 passed、2 skipped；扩展同stage
  数值断言后相关2测试再通过。两rank真实Interleaved1F1B四步训练独立dense
  比较通过，loss差异0、梯度4.936e-8、参数1.192e-7。
  下一步检查FSDP共享参数分组及跨模块buffer/恢复别名契约。

Meta buffer重建别名修复：

- materialization.py在to_empty前按注册路径重建RoPE inv_freq，但未知名字
  的同一物理meta buffer别名仍在unresolved中，因此同模块cache_alias或
  父模块external_alias会误报unsupported non-checkpoint buffers。新回归
  修复前实际触发RuntimeError；最初测试缺少nn导入的fixture错误已修正后
  才取得该生产失败证据。
- 以分配前tensor_paths身份组为依据，将已重建/保留的buffer结果传播至
  同tensor的全部buffer路径，再执行未知buffer校验。既有最终身份恢复保留，
  即使外部别名先于RoPE owner遍历也使用已重建值；不以名字猜测内容。
- 增加同模块/父模块两个meta别名场景，检查重建值与对象identity。
  既有真实buffer保留、persistent buffer加载、tied parameter梯度和未知
  meta非persistent buffer拒绝回归继续覆盖；storage view共享尚不在此范围。
  HF wrapper相关原63测试通过（扩展父模块参数化后包含于全量结果）。
- 本批未修改FSDP分组：embedding/head既有group路径已阅读，任意跨unit
  参数及DTensor/设备materialize仍需专门生命周期验证。下一步继续核对
  FSDP shared unit与buffer alias checkpoint恢复，不宣称全局完成。
- 全量CPU3004 passed、2 skipped；全部适用文件pre-commit通过（排除保护
  examples/native_alltoall.py），compileall与diff检查通过。

默认FSDP dense层共享参数分组修复：

- pattern路径已有共享拥有者合并，但默认decoder仍独立fully_shard每层。
  实际两rank custom FSDP最小模型两Linear共享weight，分组后两rank均
  identity=False，断言失败；修正最初probe漏传pp_enabled后取得生产证据。
- 默认decoder在任何参数替换前按dense block Parameter身份合并连接的
  层组，保留层注册顺序，grouped fully_shard一次管理全部消费者，后续
  层跳过重复分组；普通独立层仍单独分组。既有state去重prefetch继续复用。
- pattern_fsdp_equivalence新增SharedLayerDecoder默认路径，保留embedding/
  head tying，同时两dense层共享投影；检查同一sharded Parameter与state、
  独立dense loss/梯度及三次SGD更新，native/custom两rankCPU/Gloo均通过。
  其余既有pattern/cycle/alias/mesh场景全部通过。不覆盖MoE混合层共享、
  层与embedding/norm交叉共享、storage views或目标设备。
- 不宣称任意默认共享布局完整支持；pattern已有分组机制与默认dense修复
  分别记录。全量CPU3004 passed、2 skipped；分布式脚本18场景通过；
  适用文件pre-commit（既定保护例外）、compileall、diff检查通过。
  下一步验证共享dense层DCP新运行时恢复，以及剩余共享边界。

默认FSDP共享dense层Trainer恢复复核：

- pp_checkpoint_equivalence新增LLMTUNER_TEST_SHARED_LAYERS=1场景，在HF
  构造完成/FSDP变换前使前两decoder层Q投影共享Parameter，要求PP1。
  每次train_step及load后断言同一Parameter与同一FSDP state；采用真实
  Trainer/DCP、canonical optimizer FQN、dynamic EMA，不是裸state_dict mock。
- 实际CPU/Gloo两rank custom/native默认FSDP full/resume：step2保存，连续
  四步与新进程恢复后两步，loss差异0，模型shard、AdamW、scheduler、trainer
  counters与EMA逐位一致；共享身份始终保留。共享参数没有额外optimizer副本。
- 恢复脚本报告改为收集全部rank失败，只在global verdict为0时打印通过，
  避免其他rank参数/EMA不一致时rank0仍打印all checks passed。
- 复现：LLMTUNER_TEST_PP_SIZE=1、LLMTUNER_TEST_SHARED_LAYERS=1、
  LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom/native、LLMTUNER_TEST_EMA=dynamic，
  两rank运行pp_checkpoint_equivalence.py full/resume <同一独立workdir>。
  不推断MoE/跨dense边界共享、storage view、设备、meta HF载入或PP跨stage共享。
  下一步推进任意共享边界的分组与恢复契约；原全局目标仍未完成。

默认FSDP dense层与decoder边界共享修复：

- 两rank CPU/Gloo最小decoder将Linear bias与末尾LayerNorm bias绑定。
  变换前身份一致，原默认路径native/custom两实现的两rank均变为不同Parameter，
  确认边界unit先分片、层unit后分片会静默拆开共享参数。
- fully_shard_parallel.py将既有embedding、norm/head（或tied边界组）与dense
  层纳入同一变换前所有权计划；按实际Parameter连接合并并只分片一次。
  保留原边界分组；共享组只要一个消费者要求不reshard便保持unsharded，
  always策略仍全部reshard。未共享的分组及其策略保持原行为。
- pattern_fsdp_equivalence增加tied/untied两个层与norm共享bias场景，断言
  共享Parameter与state、embedding/head身份、三次SGD后的别名；与独立dense
  参考比较loss、完整梯度及参数。native/custom两rank、prefetch窗口2的
  22场景通过；最小复现修复后四个rank/engine结果均shared_after=True。
- 仍未覆盖dense与MoE共享、root额外参数别名、storage view和目标设备。
  新边界布局的Trainer/DCP恢复尚待验收；已有dense层恢复证据不能外推。
  下一步验证新边界共享的真实Trainer恢复及MoE混合所有权。
- 本批全量CPU3004 passed、2 skipped；适用文件pre-commit全部通过
  （排除保护文件examples/native_alltoall.py），compileall及diff检查通过。

默认FSDP层/norm边界共享Trainer恢复复核：

- pp_checkpoint_equivalence新增LLMTUNER_TEST_SHARED_BOUNDARY=1，要求PP1。
  HF构造后/FSDP变换前绑定layer0.input_layernorm.weight与final norm.weight；
  每步和load后检查真实共享Parameter与同一FSDP state，可与原shared layers
  开关组合。该场景采用Qwen3 RMSNorm weight，补充此前Linear/LayerNorm bias
  的独立dense数值等价测试，不将两种fixture混为同一证据。
- 实际native/custom各两rank CPU/Gloo full/resume，dynamic EMA，step2保存、
  连续四步与新进程恢复后两步比较；loss最大差异0，模型本地shard、AdamW、
  scheduler、trainer counters和EMA逐位相等，全rank汇总通过。
  full loss为4.8555808067、4.8522520065、4.8585534096、4.8772649765。
- 复现：PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1，
  LLMTUNER_TEST_PP_SIZE=1、LLMTUNER_TEST_SHARED_BOUNDARY=1、
  LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom/native、LLMTUNER_TEST_EMA=dynamic；
  两rank运行pp_checkpoint_equivalence.py full/resume <同一独立workdir>。
  本项证明该RMSNorm边界布局的Trainer恢复，不外推MoE/root/storage view或设备。
  下一步推进dense与MoE以及root额外Parameter别名的默认所有权计划。
- 本批仅修改集成测试与记录；全部适用文件pre-commit通过（保护示例排除），
  compileall和diff检查通过。未重跑CPU全套，上一批结果为3004 passed、2 skipped。

默认FSDP root与dense层共享参数修复：

- 两rank CPU/Gloo最小decoder在root注册layers[0].weight的external_weight
  别名；默认分片前身份一致，native/custom两实现两rank分片后均False。
  默认计划此前只包含边界与dense层，root最后独立处理剩余注册导致共享拆开。
- 变换前收集计划子树之外各模块的直接Parameter注册，作为root所有权；
  不递归收集root全部参数，避免把所有独立层错误合并。root共享连接组延后
  到最后统一fully_shard，其他分组保持独立，root保持不reshard策略。
- 新增RootSharedDecoder，通过root别名与layer内部投影实际共同参与前向。
  native/custom两rank CPU/Gloo、prefetch窗口2，24场景通过；与独立dense
  参考比较loss、完整梯度、三次SGD更新，并检查共享身份。
  PyTorch会将group列表内被root包含的子模块折叠为root管理，故子层不是
  独立FSDPModule；首次测试错误读取子层state，已修正为检查实际root state，
  未弱化共享身份或数值断言。
- 不覆盖root与MoE共享、EP专家placement、storage views及目标设备。
  root额外子树别名已纳入所有权收集，但本次数值fixture为root直接注册。
  新root布局Trainer/DCP恢复未验证；下一步验证恢复并推进MoE混合共享。
- 全量CPU3004 passed、2 skipped；适用文件pre-commit通过（排除保护示例），
  compileall与diff检查通过。全局目标仍未完成。

默认FSDP root共享参数Trainer恢复复核：

- pp_checkpoint_equivalence新增LLMTUNER_TEST_SHARED_ROOT=1（要求PP1），
  HF构造后/FSDP前在wrapper root注册layer0 Q projection weight的
  external_weight别名。每步和load后检查root别名与实际层参数同一身份。
  root注册改变canonical参数FQN，采用真实Trainer、AdamW和DCP检验恢复。
- native/custom分别两rank CPU/Gloo full/resume、dynamic EMA：step2保存，
  新进程恢复到step4，与连续四步的loss、模型shard、AdamW、scheduler、
  trainer counters和EMA逐位一致；loss最大差异0，所有rank检查通过。
  full loss为4.8555808067、4.8522510529、4.8585624695、4.8772563934。
- 复现：PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1，
  LLMTUNER_TEST_PP_SIZE=1、LLMTUNER_TEST_SHARED_ROOT=1、
  LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom/native、LLMTUNER_TEST_EMA=dynamic；
  两rank执行pp_checkpoint_equivalence.py full/resume <同一独立workdir>。
  本fixture的root别名仅注册，实际计算由Q projection消费；此前独立dense
  RootSharedDecoder已验证root别名也参与计算时的loss/梯度/SGD等价。
- 未覆盖MoE/root共享、EP专家placement、storage views、设备或meta加载。
  下一步推进默认MoE混合共享所有权及其独立数值参考。
- 本批仅扩展集成测试和文档，适用文件pre-commit（保护示例排除）、
  compileall和diff检查通过；未重跑全量CPU，上一批3004 passed、2 skipped。

默认FSDP dense/MoE混合共享与专家placement修复：

- 新增MoESharedDecoder：dense层与MoE层共享Linear投影，MoE还实际计算
  GroupedExperts单专家输出。修复前两rank native在共享Parameter断言处失败，
  证明默认MoE独立分片仍会拆开dense/MoE共享，而不是测试或环境失败。
- 默认所有权计划现在包含所有decoder层、边界和root，统一连接分组。
  在变换前一次收集各专家原Parameter的placement；共享组按参数保留专家
  sparse mesh/Shard(1)与普通参数dense mesh，不为合并组统一错误placement。
  同一专家Parameter要求冲突Shard维度时在任何分片前拒绝。
  移除逐MoE分支重复fully_shard和placement闭包，所有分组使用同一执行路径。
- 两rank CPU/Gloo native/custom、prefetch窗口2的26场景通过；新场景共享
  Parameter/state保持，专家Shard(1)与E维不切分，独立dense loss、全部梯度和
  三次SGD更新等价。该fixture为单专家确定分配，不证明多专家dispatcher。
- 四rank真实qwen3_moe、EP2/FSDP4/efsdp2 default路径native/custom既有验证
  通过；新增--shared-projection共享两MoE层Q权重，native/custom各再通过。
  共享Parameter/state保持，dense参考累计Q梯度、router/专家梯度与loss符合
  既有1e-4容差，专家mesh size2、dense/router mesh size4及rank成员正确。
  此真实EP场景只验证loss/梯度和placement，未比较optimizer更新或DCP。
- 全量CPU3004 passed、2 skipped；本批新混合布局DCP、root专家别名、
  storage views、CUDA/NPU及symm_mem dense scope共享state策略仍待验收。
  下一步复核混合state的symm_mem范围和真实Trainer恢复，不宣称全局完成。
- 适用文件pre-commit全部通过（排除保护文件examples/native_alltoall.py），
  compileall及diff检查通过。

共享FSDP state的symm_mem范围修复：

- 原enable_fsdp_symm_mem按FSDPModule的moe_enabled逐个启用。共享dense/MoE
  state经dense消费者被错误启用；root管理普通MoE子树也漏排除；all范围对
  同state重复配置。六个CPU回归在修正fixture的FSDPModule构造方式后，
  实际三失败三通过，失败为dense-grouped、dense-managed和all-grouped。
- 遍历模型时传播最近FSDP state及MoE子树上下文，先建立state所有权和
  sparse state集合，再按scope选择并且每state配置一次。嵌套独立MoE state
  不使外部dense root被排除，普通MoE子树则属于其root state；MoE父模块
  下独立包装的子模块继承MoE上下文。未修改设备通信内核。
- CPU六回归通过；pattern_fsdp_equivalence的真实native/custom共享混合state
  增加dense/all选择检查，仅拦截设备enable调用，实际state发现与选择不替换。
  两rank CPU/Gloo、prefetch窗口2的26场景通过，loss/梯度/三次SGD仍等价。
  该检查证明策略与去重，不证明CUDA/NPU symmetric-memory通信或性能。
- 下一步验证真实MoE共享布局Trainer恢复及root专家别名；storage views、
  target-device symm_mem kernel和性能仍待验收，全局目标未完成。
- 全量CPU3010 passed、2 skipped；适用文件pre-commit通过（保护示例排除），
  compileall及diff检查通过。

真实EP共享MoE层Trainer恢复复核：

- pp_checkpoint_equivalence新增LLMTUNER_TEST_MOE=1和LLMTUNER_TEST_EP_SIZE，
  使用offline qwen3_moe、8专家/top2、专家hidden48、HF eager初始化。
  LLMTUNER_TEST_SHARED_LAYERS=1在FSDP前绑定两MoE层Q projection。
  实际MoE类型、专家DTensor sparse mesh degree与router dense mesh degree
  增加断言，避免普通dense/fallback运行被误记为EP恢复。
- native/custom分别四rank CPU/Gloo、PP1/EP2/FSDP4/efsdp2、dynamic EMA，
  实际Trainer/DCP step2保存，新进程恢复到step4。模型shard、AdamW、
  scheduler、trainer counters和EMA逐位一致，loss差异0；每步和load后
  共享Parameter与state保持。增加mesh断言后两实现分别再恢复通过。
  full loss为4.8834443092、4.8733038902、4.8456087112、4.8653020859。
- 复现：PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1，
  LLMTUNER_TEST_WORLD_SIZE=4、LLMTUNER_TEST_PP_SIZE=1、LLMTUNER_TEST_MOE=1、
  LLMTUNER_TEST_EP_SIZE=2、LLMTUNER_TEST_SHARED_LAYERS=1、
  LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom/native、LLMTUNER_TEST_EMA=dynamic；
  四rank运行pp_checkpoint_equivalence.py full/resume <同一独立workdir>。
- 本项证明共享dense Q投影的真实MoE布局恢复，未验证共享专家Parameter、
  root专家别名、storage views、TP/PP/CP混合或设备；下一步检查专家别名
  与root参数管理，复用已有分组及placement证据。全局目标仍未完成。
- 本批只修改集成测试与文档；适用文件pre-commit（保护示例排除）、
  compileall与diff检查通过，未重跑CPU全套，上一批3010 passed、2 skipped。

HF MoE布局变换的外部专家别名保护修复：

- root注册HF fused gate_up_proj别名时，conversion把内部参数拆成w1/w3，
  root原参数仍保留，两套Parameter可独立进入optimizer。新CPU回归修复前
  实际未报错并安装两层replacement，证实静默共享破坏。
- 转换前枚举真实将被安装的layer slot子树，收集其外的所有直接Parameter
  注册/FQN。不能按候选block对象的全部别名路径排除，否则root挂载整个
  原HF block的路径会被误认为已替换；根参数/整个block别名均增加回归。
- 既有router/EP1 down布局不变且复用source时继续支持。fused split或EP
  slice使外部source无法复用时给UnsupportedCombinationError，包含外部FQN、
  布局变化及要求适配external consumer的提示。检查在installation前完成，
  原layer/root参数身份和AuxLoss group counts保持；不自动猜测外部计算语义。
- 两新增CPU回归与现有sharing/external消费者共7 targeted测试通过。
  真实四rank EP2/FSDP4 orchestration新增--external-expert-alias，以root
  注册HF down_proj全专家权重，EP分片将改变shape；全rank明确拒绝且原
  layer和root身份不变，脚本exit0。完整CPU3012 passed、2 skipped。
- 这是防止静默破坏的guard，不是外部fused/full专家消费者迁移完成；
  conversion后的专家参数root别名/FSDP placement仍需数值与DCP验收。
  下一步检验转换后别名，区分注册身份与消费者全专家布局的不同契约。
- 适用文件pre-commit通过（排除保护示例），compileall及diff检查通过；
  全局目标仍未完成。

EP转换后root专家别名的checkpoint索引修复：

- ep_fsdp_equivalence新增--converted-root-expert，只在真实apply_ep后/FSDP前
  注册local down expert的root别名。native/custom四rank EP2/FSDP4 loss与
  完整专家参考的router/专家梯度均通过1e-4，身份与efsdp mesh size2保持。
  这不同于未适配外部完整HF专家消费者；不移除前一批保护。
- pp_checkpoint_equivalence新增LLMTUNER_TEST_SHARED_ROOT_EXPERT=1，
  同样在真实EP后绑定别名。首次custom四rank full成功但resume失败，loss
  最大差异2.503e-4，模型/AdamW/EMA偏离。不能把前向等价解释为恢复正确。
- checkpointing.expert_shard_map此前仅识别GroupedExperts注册FQN，遗漏root
  canonical alias；index_checkpoint_parameters默认去重又隐藏原专家路径，
  导致模型别名/optimizer/EMA没有完整EP gather，DCP把不同EP坐标当成副本。
  改为先按真实Parameter身份确定EP group，再索引所有注册FQN；live index
  使用remove_duplicate=False。同身份跨group冲突在collective前明确拒绝。
- 新增model/optimizer/EMA三类root与module别名gather/slice回归，加共享参数
  不同group拒绝测试。原普通无别名路径保留；尚不支持storage view身份推导。
- 新workdir重新保存后，native/custom四rank PP1/EP2/FSDP4、dynamic EMA，
  step2保存/new process恢复至step4：loss差异0，模型shard、AdamW、scheduler、
  trainer counters及EMA逐位相等，root别名和专家mesh断言保持。
  full loss为4.8840322495、4.8683495522、4.8488407135、4.8653321266。
- 复现：LLMTUNER_TEST_WORLD_SIZE=4、LLMTUNER_TEST_PP_SIZE=1、
  LLMTUNER_TEST_MOE=1、LLMTUNER_TEST_EP_SIZE=2、
  LLMTUNER_TEST_SHARED_ROOT_EXPERT=1、LLMTUNER_TEST_EMA=dynamic、
  LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom/native，四rank运行
  pp_checkpoint_equivalence.py full/resume <同一新workdir>。
  修复前异常布局checkpoint可能已丢失专家坐标内容，不宣称可恢复旧损坏数据。
- 尚未验证target-device、storage views或任意外部完整专家消费者；下一步
  检查跨模型部分共享身份及checkpoint key冲突，继续全局未完成项。
- 全量CPU3016 passed、2 skipped；适用文件pre-commit通过（保护示例排除），
  compileall与diff检查通过。

跨model parts优化器与EMA checkpoint命名提前校验：

- ModelCheckpointState已有raw key冲突保护，optimizer/EMA state_dict也已有
  合并冲突检查；但各part独立构造时只检查本part名称，不同Parameter用相同
  checkpoint FQN可先训练，到首次save才失败。新增三个构造前回归在修复前
  实际进入optimizer/EMA state allocation并失败，非推测性的重复覆盖。
- optimizer在全部param group计划与Parameter身份验证后、任何inner optimizer
  创建前检查全局canonical param_names，报出part编号和冲突名称。
  EMA在任何average clone前检查全部parameter/buffer组canonical名称，
  同时覆盖跨part参数/参数及参数/buffer冲突。合法同part别名去重保持。
- 原九个测试fixture采用两个同名独立Sequential，与全局checkpoint FQN
  契约不符；改为first/second唯一前缀，仍修改原实际Parameter/buffer，保留
  多optimizer、hook次数、替换/冻结/删除失败前不修改average等全部断言。
  第一次完整回归九个fixture失败已处理；相关98测试和最终全量均通过。
- 四rank真实Interleaved1F1B、PP2/FSDP2 custom、每rank两个local stages、
  dynamic EMA full/resume通过：step2恢复到step4，loss差异0，模型、AdamW、
  scheduler、trainer counters与EMA逐位一致，证明合法全局FQN配置未误拒绝。
- 最终全量CPU3019 passed、2 skipped。未实现跨PP stage共享同步；配置中
  的buffer跨part身份及storage view仍需分别审查。下一步继续checkpoint
  buffer别名与恢复边界，不把提前命名保护表述为共享同步能力。
- 适用文件pre-commit通过（保护示例排除），compileall和diff检查通过；
  全局目标仍未完成。

checkpoint buffer导出视图缓存修复与别名恢复：

- ModelCheckpointState与torchtitan/components/checkpointer/base.py的ModelWrapper
  原采用_is_alias_of判断无需刷新；共享storage并不表示同一view。buffer导出
  hook改为转置但shape仍相同时，缓存返回旧值。新回归修复前实际数值失败，
  两个非对角元素错误；不能以storage复用证明导出state正确。
- 刷新直接比较本地Tensor的shape/stride/storage_offset，DTensor不经过
  autograd-aware to_local。同storage同view仍无复制；同storage不同view
  分离缓存为导出值clone，避免cached.copy_写坏模型活跃buffer或重叠copy异常。
  后续缓存与live storage独立，普通刷新继续in-place并复用新缓存storage。
  合并原单调用方_shares_storage薄封装，注明视图变化时的一次缓存分配例外。
- transpose/offset两个导出回归检查导出值、模型buffer不变及之后缓存storage
  复用；测试仅验证state-dict导出，不宣称无配套load hook的变换可直接恢复。
  新实际CPU DCP保存/新model加载验证普通persistent buffer的同对象别名、
  tail storage view、offset和值保持；alias结构由新model构造，DCP恢复内容。
- 四rank custom EP2/FSDP4 root专家alias新运行时resume复核通过，loss差异0，
  模型/AdamW/scheduler/trainer/EMA逐位一致，确认普通DTensor刷新路径未回归。
  复用已有正确step2 checkpoint及独立连续参考，未重复full phase。
- 全量CPU3022 passed、2 skipped；未验证target-device pinned async staging、
  跨PP stage buffer同步或任意load hook的恢复事务。下一步检查异步checkpoint
  staging与资源生命周期，继续原全局目标未完成项。
- 适用文件pre-commit通过（保护示例排除），compileall与diff检查通过。

异步checkpoint中断关闭的资源所有权修复：

- DCP interrupted wait已有保存live future的逻辑，但BaseCheckpointManager.close
  捕获后仍执行_close销毁stager/process group并_initialized=False；worker
  尚在运行时失去资源。staging/saving两新回归修复前实际release且失败。
- close捕获错误后检查尚未完成的staging/save future；仍活跃时保留初始化
  状态与资源并抛出首个错误，后续完成后可重试close。已完成但失败的future
  仍按原流程消费并清理全部资源；不将失败改为成功，不重复释放已完成资源。
- Trainer.close原先先清空checkpointer引用，失败后不能重试；新consumer回归
  修复前引用为None而失败。若checkpointer仍_initialized，恢复该资源引用，
  独立metrics/dataloader/GC资源继续正常清理；再次close完成后引用清空。
- pp_checkpoint_equivalence新增LLMTUNER_TEST_ASYNC_MODE=async；full phase
  捕获最终参考状态后先Trainer.close，再检查metadata和启动独立resume。
  两rank CPU/Gloo custom FSDP、dynamic EMA实际thread async DCP full/resume
  step2恢复至step4通过，loss差异0、模型/AdamW/scheduler/trainer/EMA逐位一致。
  pinned process stager未在CPU路径启用，future中断回归只验证生命周期协议。
- 复现：LLMTUNER_TEST_PP_SIZE=1、LLMTUNER_TEST_ASYNC_MODE=async、
  LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom、LLMTUNER_TEST_EMA=dynamic，
  两rank运行pp_checkpoint_equivalence.py full/resume <同一新workdir>。
- 不宣称已验证target-device pinned staging、真实OS信号注入或worker失败
  后的跨rank故障恢复。下一步核对async失败时checkpoint保留/清理与load选择，
  继续原全局目标未完成项。
- 最终全量CPU3025 passed、2 skipped；适用文件pre-commit通过（保护示例
  排除），compileall与diff检查通过。

checkpoint发布前保留旧恢复点修复：

- _purge_stale_checkpoints原在新save前保留keep_latest_k-1，K=1时唯一旧
  完整checkpoint会先进入删除队列；同步或异步保存失败则没有恢复点。
  新回归修复前同步失败场景实际排队旧checkpoint；异步测试首先在排队
  断言失败后等待未完成fixture Future，已终止该特定pytest进程并修正fixture
  finally完成Future，避免将测试清理挂起误记为生产collective问题。
- 保存前保留K个已有完整checkpoint；只有同步成功或异步future.result成功
  后，将新发布checkpoint纳入K个保留窗口。失败清除pending retention，
  中断继续保留future与pending policy。DCP与torch_checkpointing backend
  同步最终保存/异步完成路径均接入；不在后台callback线程执行保留扫描。
- 同步/异步成功失败四场景验证新save成功后才排队旧checkpoint，失败保留
  旧checkpoint；相关67测试通过。未改变既有purge_exempt与废弃目录规则。
- 下一步检查模型only最终export与可恢复checkpoint的不同保留窗口及
  存储发布原子性；当前未覆盖真实磁盘写失败/跨rank故障注入或设备staging。
- 全量CPU3029 passed、2 skipped；适用文件pre-commit通过（保护示例排除），
  compileall和diff检查通过。全局目标仍未完成。

最终模型导出与训练恢复点保留窗口修复：

- 新实际CPU DCP测试保存step1完整状态，K=1后step2 final model-only export。
  修复前step1进入删除队列，自动resume无恢复点；此前K=0 auto-resume跳过
  export测试不能证明retention正确。新增回归实际修复前失败。
- Base retention使用backend的_is_resumable_checkpoint区分完整训练checkpoint
  与model-only export，各自保留K项；保存成功才淘汰的既有规则继续适用。
  purge_exempt分别保护两类产物；seed checkpoint沿既有resume契约分类。
  配置keep_latest_k说明同步更新，K=0仍不启用淘汰。
- 真实metadata回归验证step2导出不排队step1，auto-resume仍选1；step3完整
  save仅排队step1，保留step2 export，auto-resume选3。旧marker fixture只
  测publication/retention时明确替换resumable判定，避免空.metadata当真实DCP。
- 相关68 checkpointer测试通过；本项使用单进程实际DCP，不宣称跨rank磁盘
  故障恢复或原子重写同step目录已验证。下一步检查同step覆盖与存储发布
  失败后的加载选择，继续全局未完成项。
- 全量CPU3030 passed、2 skipped；适用文件pre-commit通过（保护示例排除），
  compileall与diff检查通过。

DCP模型flatten与训练状态类别冲突修复：

- 加载选择与存储发布路径继续阅读；native metadata解析错误目前显式传播，
  未擅自将损坏checkpoint跳过并恢复旧步数。同step目录原子重写尚未实现。
- _flatten_model_state此前sd.update(model.state_dict())可覆盖同名optimizer等
  state类别，导致保存内容悄然变为模型tensor；新回归修复前实际未报错。
  合并前比较模型与辅助state顶层keys，报出冲突，保留原states对象不变。
  普通HF参数FQN不冲突时保存格式保持；不新增重命名兼容层。
- 下一步核对同step完整checkpoint与final export重写及存储发布失败边界。
  本项不证明原子IO或损坏checkpoint可恢复；全局目标继续未完成。
- 全量CPU3031 passed、2 skipped；pre-commit首次E501已修复并重跑相关
  文件通过，其余适用文件检查通过（保护示例排除），compileall/diff通过。

同step最终导出覆盖完整checkpoint修复：

- 实际CPU DCP save(1)完整状态后修改模型，再save(1,last_step=True)。
  修复前最终model-only metadata覆盖原完整状态，resumable判定失败；新增
  回归实际失败，前一批不同step保留窗口不能保护同目录覆盖。
- Base提供最终保存路径解析：仅model-only且同step已有resumable checkpoint
  时写入step-N/model-export，保留完整目录与metadata；普通最终导出和完整
  final save路径不变。DCP及torch_checkpointing final路径共用，配置说明更新。
  嵌套export随所属完整checkpoint被retention删除，未作为独立step计数。
- 回归验证两份产物均存在、root仍resumable、export非resumable、显式load(1)
  恢复原完整模型且值精确一致。既有backend HF mock补齐Filesystem storage，
  不跳过路径解析；原首次mock AttributeError已处理。
- 本项避免model-only覆盖完整状态，不实现一般同step完整重写原子发布。
  下一步检查完整checkpoint重复保存与写失败，以及嵌套导出使用文档。
  全局目标仍未完成。
- 首次全量验收暴露旧backend mock缺失metadata文件名，以及retention worker
  在marker检查后删除旧目录的竞态。mock补齐实际路径依赖；DCP元数据扫描
  仅捕获FileNotFoundError并返回不可恢复，损坏和权限错误仍向上传播。
  新增三种异常回归，checkpoint定向测试73 passed；本轮全量3035 passed、
  2 skipped（153 warnings），进程正常结束；日志为
  /tmp/llmtuner-checkpoint-race-units.log。
- 适用文件pre-commit全部通过（排除examples/native_alltoall.py），
  compileall与git diff HEAD --check通过。证据限CPU DCP与可选backend mock，
  不证明目标设备、完整同step重写原子性或可选backend实际存储行为。

同step完整DCP重写失败保护：

- 核对安装的PyTorch 2.14 FileSystemWriter：默认shard使用固定文件名，finish
  删除旧metadata再rename。重复save同step时，新shard可在metadata发布前
  覆盖旧内容。实际同步/线程异步故障回归均失败：metadata字节不变，但恢复
  的模型已变为新值，确认旧checkpoint内容损坏。
- 新增checkpoint_writer.py，将本地原生DCP存储发布职责从manager分离。
  generation前缀写入coordinator SavePlan并分发至各rank；成功写完shard后
  序列化并flush/fsync临时metadata，os.replace发布，不先删除旧metadata。
  reader与checkpoint目录格式保持兼容；同步、线程及进程异步入口均接入。
- 回归覆盖同步/线程异步 × shard写后失败/metadata替换失败，比较旧metadata、
  模型、Adam状态与scheduler，随后验证重试成功及普通DCP reader读取新值。
  4项通过；未以mock成功代替实际DCP文件读写。
- 实际2进程CPU/Gloo fsdp_checkpoint_equivalence.py：native/custom ×
  shard/replicate源目标8场景均通过。每项注入rank0线程异步shard写后失败，
  每个rank收到CheckpointException，fresh runtime恢复旧模型及优化器后续训；
  同拓扑精确一致，跨拓扑及独立dense参考atol=1e-5。完整命令：
  PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1 python -m
  torch.distributed.run --nproc_per_node=2 --master_addr=127.0.0.1
  --master_port=29941 tests/integration_tests/fsdp_checkpoint_equivalence.py。
  进程exit=0；日志/tmp/llmtuner-generation-fsdp.log。
- 上游torchtitan/components/checkpointer/dcp.py仍交给DCP默认writer；本项为
  llmtuner保存失败契约的必要适配，不宣称是上游已有机制。保留旧generation
  及失败文件至整个step目录被retention删除，避免破坏已读取旧metadata的reader。
  重复同step保存会增加磁盘占用；不提供并发writer排他或断电持久性保证。
- 实现当前只覆盖本地原生DCP。remote/HF保持原存储路径；进程pinned异步入口
  接入但未运行。下一行动检查remote发布契约、HF writer及目标设备pinned流程。
  全局目标仍未完成。
- 全量CPU3039 passed、2 skipped（157 warnings），exit=0，日志
  /tmp/llmtuner-generation-units.log；适用文件pre-commit全部通过，排除受保护
  examples/native_alltoall.py；compileall及git diff HEAD --check通过，保护
  示例无HEAD差异。源码清单与160个当前Python文件一致，无重复、
  遗漏或失效条目；清单覆盖不证明所有机制正确。

HF分片导出发布与索引读取复核：

- 实际HF DCP保存后修改权重，合并文件写完后注入失败；修复前重新通过
  initial_load_path/initial_load_in_hf装配fresh model，恢复的权重已被覆盖。
  初始测试误用显式load(step)恢复model-only export，修正为合法初始权重
  导入入口后确认真实故障（1 failed、2 passed），未改变训练resume语义。
- 新增hf_storage.py负责HF存储契约，将导出及合并从DCP manager移出。
  每次导出写入独立.export-generation/sharded；各rank处理所属输出分片，
  使用DCP _DistWrapper协调合并与coordinator发布，失败广播给所有rank。
  root索引用os.replace发布并指向generation内的不可变文件；失败时旧索引
  和旧权重保留。缺少初始化进程组时也发布合并后的索引。
- IndexedHuggingFaceStorageReader逐文件按weight_map选择具体tensor，忽略
  未索引文件及同文件未索引键，保留无索引single-file加载和DCP sharding
  offset元数据读取。回归验证两文件包含相同FQN时选择索引指定值；不再
  使用目录扫描顺序隐式覆盖。HF adapter返回此reader，旧HF格式仍可导入。
- 修复effective checkpoint folder边界：本地配置子目录与remote dump folder
  拼接后也是remote，Base在状态/后台资源初始化前拒绝HF导出；读写helper
  直接调用也拒绝remote URI，避免Path将协议转换为错误本地路径。
- CPU回归包含无进程组单/双分片导入、未发布根文件忽略、合并/发布失败
  后fresh model恢复，以及有效remote目录拒绝。2进程CPU/Gloo native/custom
  FSDP × 单/双输出分片 × rank1合并失败/rank0发布失败8场景通过；每个rank
  捕获CheckpointException，旧导出及成功重试后的新导出精确恢复，原完整
  step checkpoint仍resumable。命令：PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0
  TORCHDYNAMO_DISABLE=1 python -m torch.distributed.run --nproc_per_node=2
  --master_addr=127.0.0.1 --master_port=29942
  tests/integration_tests/hf_checkpoint_equivalence.py，exit=0；日志
  /tmp/llmtuner-hf-generation-distributed.log。
- 参考torchtitan/components/checkpointer/dcp.py与安装PyTorch 2.14的
  hf_storage.py/_consolidate_hf_safetensors.py，保留分布式并行合并、格式和
  数据布局；generation发布及失败协调为本项目适配。旧/失败generation
  随step目录retention清理；重复导出增加磁盘占用，未证明断电持久性。
- Remote原生DCP仍使用上游FsspecWriter；其假设空目录和原子文件创建，
  当前不具备本地generation发布保护。下一行动用真实fsspec后端复现同step
  写失败并核对metadata替换契约；不将CPU/HF结果视为remote或目标设备验证。
  全局目标仍未完成。
- 全量CPU3047 passed、2 skipped（163 warnings），exit=0，日志
  /tmp/llmtuner-hf-generation-units.log；适用文件pre-commit全部通过，排除
  受保护examples/native_alltoall.py；compileall及git diff HEAD --check通过。
  当前161个源码文件与审查清单一致，无遗漏、重复或失效路径；此统计仅
  证明清单覆盖，不表示全局机制验收完成。

fsspec原生DCP generation发布与失败恢复：

- 实际memory://及file://，同步/线程异步 × shard写后失败/发布前失败8回归
  修复前全部失败：shard固定名称覆盖旧内容；默认finish先rm旧metadata，
  rename失败后旧metadata消失。未使用mock文件操作代替实际DCP读写。
- 复用GenerationFileSystemWriter，构造remote writer时懒加载fsspec适配，
  data plan沿用coordinator generation prefix。file://解析为本地路径，使用
  fsync临时metadata与os.replace；memory/S3/GCS先pickle.dumps完成序列化，
  然后pipe_file整对象替换，不预删旧metadata，不使用copy/delete rename。
  目录及metadata格式兼容默认FsspecReader，回归额外用普通dcp.load读取。
- 8项故障恢复与重试回归通过；首次验收一个memory异步故障未注入，查明
  fsspec按线程缓存实例，修正为patch filesystem类的pipe_file后实际触发失败。
  未把测试注入错误当作生产实现缺陷。
- 两项实际memory保存后“提交已成功但确认丢失”回归：异常继续传播，随后
  load恢复完整新generation；远程异常不能证明提交未发生，也不能承诺一定
  恢复旧版本。验证不等同真实S3/GCS网络故障；没有删除或回滚有效新状态。
- 不支持没有已约定原子整对象替换的协议，writer在数据写入前明确拒绝；
  FTP拒绝回归仅mock协议元数据。memory filesystem是进程内dict，无法跨rank
  或保存subprocess共享；manager按实际backend protocol拒绝多rank及
  async_with_pinned_mem的memory组合，在创建额外PG前失败，两项guard回归
  通过。单进程同步/线程memory保留支持；此拒绝不表示共享内存存储已迁移。
- 实际2进程CPU/Gloo，fsdp_checkpoint_equivalence.py --fsspec：native/custom
  × shard/replicate源目标8场景通过；rank0线程异步shard写后失败传到所有rank，
  fresh runtime恢复旧checkpoint继续训练，同拓扑精确一致，跨拓扑与独立
  dense参考atol=1e-5。命令：PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0
  TORCHDYNAMO_DISABLE=1 python -m torch.distributed.run --nproc_per_node=2
  --master_addr=127.0.0.1 --master_port=29943
  tests/integration_tests/fsdp_checkpoint_equivalence.py --fsspec；exit=0，日志
  /tmp/llmtuner-fsspec-generation-distributed.log。file://证据不是云端验证。
- 脚本增加互斥的--checkpoint-root，可在S3/GCS前缀下创建唯一测试子目录，
  保留产物供检查；--help通过。目标环境需s3fs/gcsfs及相应访问凭据，运行：
  PYTHONPATH=. torchrun --nproc_per_node=2
  tests/integration_tests/fsdp_checkpoint_equivalence.py
  --checkpoint-root s3://BUCKET/PREFIX（或gs://）。本机未执行云端命令。
- S3/GCS仅机制接入，真实性能、网络错误及服务端一致性待验证；pinned-memory
  保存subprocess及CUDA/NPU待验证。旧/失败generation随step retention删除，
  不提供并发writer排他或断电保证。下一行动验证新writer在保存subprocess的
  序列化/文件IO生命周期，设备pinned传输与云端验证继续保留缺口。
  全局目标仍未完成。
- checkpoint专项97 passed；最终全量CPU3060 passed、2 skipped
  （200 warnings），exit=0，日志/tmp/llmtuner-fsspec-generation-final-units.log。
  适用文件pre-commit全部通过，排除受保护examples/native_alltoall.py；
  compileall及git diff HEAD --check通过。修改未新增源码路径，当前161个
  Python文件与审查清单一致；清单覆盖不表示所有硬件机制正确。

保存subprocess与同步staging返回值契约：

- 增加fsdp_checkpoint_equivalence.py --process-save-cpu，实际使用DCP
  PROCESS executor、DefaultStager shared CPU storage；显式关闭pinning、
  async staging及nonblocking copy，不替换保存executor或writer实现。
- 首次真实2rank运行失败：stager同步完成时PyTorch async_save返回upload
  Future，manager只接受AsyncSaveResponse而抛TypeError；worker此时已启动
  且实际写入完成，manager却未保存Future引用，close无法等待该任务。
  mock最小回归修复前同样失败；已核对安装PyTorch async_save实际返回分支。
- process模式兼容Future和AsyncSaveResponse：同步stage无staging_future，
  直接持有upload；异步stage保留两Future原语义。回归确认未完成upload被
  持有，完成后maybe_wait_for_saving清空；所有合法返回都纳入retention/close。
- 修复后8配置实际process保存、线程写后故障及恢复先通过；随后将故障
  injector改为可序列化writer subclass，使rank0在保存subprocess内完成shard
  写入后抛错。第二次2rank CPU/Gloo验证native/custom × shard/replicate
  源目标8配置通过，所有rank捕获CheckpointException，fresh runtime恢复
  原模型/Adam状态/LR并继续训练；同拓扑精确一致，跨拓扑/dense oracle 1e-5。
  多次manager创建/关闭后仍可复用PyTorch worker处理后续请求。
- 命令：PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1 python -m
  torch.distributed.run --nproc_per_node=2 --master_addr=127.0.0.1
  --master_port=29946 tests/integration_tests/fsdp_checkpoint_equivalence.py
  --fsspec --process-save-cpu；exit=0。日志/tmp/llmtuner-process-worker-failure.log。
  首次失败日志/tmp/llmtuner-process-generation.log；Future回归失败日志
  /tmp/llmtuner-process-future-before.log。均为CPU float32、本地file://。
- 参考torch.distributed.checkpoint.state_dict_saver.async_save及
  _async_process_executor：daemon为PyTorch进程级单例，按上游设计活到主进程
  退出；manager仅关闭自身stager/PG并等待自身upload，不擅自销毁共享worker。
  不将manager close解释成daemon已退出。并发manager/改变rank成员关系未验证。
- 本项不证明pinned-memory异步device staging、CUDA/NPU或云端传输正确。
  全局目标仍未完成；下一行动检查异步staging与optimizer更新的实际等待顺序，
  以及stage缓存复用时的训练快照一致性。
- checkpoint专项98 passed；全量CPU3061 passed、2 skipped
  （214 warnings），exit=0，日志/tmp/llmtuner-process-future-units.log；
  适用文件pre-commit全部通过（排除examples/native_alltoall.py），
  compileall及git diff HEAD --check通过。

异步快照与前向状态变更边界复核：

- Trainer 的 staging 等待移到首个前向之前；BatchNorm running buffers 和
  Embedding(max_norm) 可在前向修改持久状态，不能等到 optimizer step 才保护快照。
  validation 同样在执行模型之前等待 staging，避免 eval 中的权重修改污染快照。
- 两项训练参数化回归及验证回归在修复前失败，修复后通过；相关测试 167 passed。
- 新增 `tests/integration_tests/async_snapshot_equivalence.py`：延迟真实
  DefaultStager 读取，检查共享 CPU storage 和缓存复用，使用真实 PROCESS 上传。
  两次独立双 rank full/resume 调用均退出 0；custom FSDP、PP1、动态 EMA 下，
  模型持久 buffer、优化器、scheduler、trainer、EMA 和续训 loss 精确一致。
- 命令：`PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1
  LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom LLMTUNER_TEST_EMA=dynamic
  python -m torch.distributed.run --nproc_per_node=2 --master_addr=127.0.0.1
  --master_port=<PORT> tests/integration_tests/async_snapshot_equivalence.py
  <full|resume> <WORKDIR>`；两个阶段顺序运行并使用同一目录。
- 本证据禁用了 pinning/nonblocking device copy，不证明 CUDA/NPU staging。
  日志：`/tmp/llmtuner-async-snapshot-{full,resume}.log`。
- 本批累计验收：CPU 全量 3064 passed、2 skipped；pre-commit 通过（排除
  受保护的 `examples/native_alltoall.py`），compileall 与 HEAD diff 检查通过。
- 后续审查：同一 manager 的 load 与未完成 save 是否具备一致的状态所有权；
  本批尚未处理该路径，也未宣布全局审查完成。

恢复入口与未完成保存所有权复核：

- `components/checkpointer/base.py::load` 在合法 step 校验之后、存储发现之前，
  等待 staging 和 upload。此前可在 staging 读取 live state 时覆盖状态，或在
  上传结束前发现旧 checkpoint / 尚无 checkpoint。
- 四项回归修复前失败：等待顺序、staging 失败、upload 失败，以及真实单进程
  DCP 异步保存的完成 Future 被 gate 持有时调用 latest load。后者使用真实
  DCP 上传和恢复，但 gate 控制的是完成句柄，不证明远端发布延迟。
- 修复后 checkpointer 定向 102 passed；失败阻止存储发现和状态恢复，活跃任务
  继续由现有 Future 等待/close 契约管理。非法 step 仍先报错。
- 累计验收：CPU 全量 3068 passed、2 skipped；适用 pre-commit、compileall、
  HEAD diff 检查通过，pre-commit 排除 `examples/native_alltoall.py`。
- 此修改不提供多线程 manager 并发 API，也不解决多 rank 非对称 staging 失败。
  后续按全局待办转向其他调用链；上述分布式失败协调仍属于未完成项。

并行梯度范数低精度累加与空 stage 复核：

- `accelerator/collectives.py::clip_grad_norm_` 的 PP/TP/EP p 次幂累加使用
  原始 norm dtype，FP16 的有限局部 norm=300 在平方时变成 inf，导致误报非有限
  并可能清零梯度。并行累加现将 FP16/BF16 scalar norm 提升到 FP32，保留
  FP64；单设备路径维持原有契约。上游 torchtitan 当前 PP 代码也存在该累加
  风险，本仓库按数值契约修正，不照搬错误行为。
- 无 TP/EP 的 PP stage 若没有 grad，get_total_norm 返回 CPU zero；现先放到
  首个 gradient / parameter device，无参数时使用通信设备，再进入 PP collective。
- 新增四项修复前失败的 CPU 回归：PP/TP/EP FP16 有限 norm 及无梯度 meta
  parameter 的 collective device 检查。meta 检查是布局证据，不是设备执行证据。
- 真实双 rank CPU/Gloo `parallel_grad_norm_equivalence.py` 退出0；独立 FP64
  向量 [300,400] 参考 norm=500，分别验证 PP/TP/EP 归约和裁剪（梯度容差1e-3）；
  PP 一端无梯度时两端 norm=400。仅验证 collective 原语，不证明完整模型布局。
- 命令：`PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 python -m torch.distributed.run
  --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29949
  tests/integration_tests/parallel_grad_norm_equivalence.py`。
- 累计验收：CPU 全量 3072 passed、2 skipped；适用 pre-commit（排除受保护
  `examples/native_alltoall.py`）、compileall 和 HEAD diff 检查通过。
- 尚未验证 CUDA/NPU 空梯度 collective；此修复针对跨 rank p 次幂累加，未改变
  PyTorch 本地低精度 norm 本身的舍入/范围限制。日志
  `/tmp/llmtuner-parallel-norm-{before,distributed,units}.log`。

无参数 collective 参与者补充复核：

- 上批修复只覆盖有 parameter 无 grad；TP/EP 完全无参数时仍生成 CPU zero，
  PP 无参数时使用默认 group 通信设备。现使用实际 TP/EP/PP mesh group 查询
  通信设备，避免后端与默认组不同导致 tensor device 不匹配。
- 累计验收：CPU 全量 3075 passed、2 skipped；适用 pre-commit（排除
  `examples/native_alltoall.py`）、compileall 和 HEAD diff 检查通过。
- 三项修复前失败的 meta-device 回归通过，验证 group 传递及空 tensor 设备；
  定向范数测试 13 passed。真实双 rank CPU/Gloo 脚本扩充 PP/TP/EP 各自一端
  完全无参数场景，两端 norm=400、另一端梯度裁剪正确，退出0。
- 命令同 `parallel_grad_norm_equivalence.py`，本次端口29950；日志
  `/tmp/llmtuner-empty-mesh-{before,distributed,units}.log`。meta/mock 只验证设备
  选择契约，仍无 CUDA/NPU 实际 collective 证据；不扩展完整模型能力声明。

数据 loader、collator 与恢复消费者复核：

- 逐段重读 `datasets/contracts.py`、`synthetic.py`、`dataloader.py`、
  `collators.py`、`loader_builder.py`，并核对 trainer 和 validation 消费者、
  当前工作区 diff 及对应 CPU 回归。未发现本组新的确定性缺陷。
- synthetic 按绝对索引生成、DP 连续行切片；恢复先校验 DP/layout/cursor，
  已持有 iterator 随后读取更新后的 stream。后退恢复明确要求 fresh loader。
  修正两处测试 docstring 的旧“恢复需重放/drain”描述；测试逻辑未改。
- Grain 在 set_state 前校验版本、rank 和 batch layout；失败关闭 iterator，
  cleanup 失败不覆盖恢复原异常。validation 每轮创建并关闭独立 loader，
  不复用训练 cursor。有限 DP corpus 在构建时拒绝，尚无协调耗尽实现。
- collator 输出独立 storage，padding 强制 IGNORE_INDEX；PP 有限末批使用
  完全 masked 空行补齐 schedule 形状，计数仅包括有效 label。synthetic
  repeat=False 仍为无限流，builder 文档已明确，不能当作有限 corpus 证据。
- 证据复用上一批全量 CPU 3075 passed、2 skipped：已有 active iterator seek、
  非法状态不改变 stream、配置变化拒绝及恢复失败清理测试；本次只改说明，
  不重复运行稳定训练测试。此结论不证明数据源身份变化可检测、streaming
  远端故障恢复或有限多 rank 数据耗尽已实现。
- 下一范围：数据 composition/packing 与 source 变更身份，核对持久化状态
  是否足以拒绝会改变训练数据语义的恢复。全局目标保持未完成。

有限 concat packing 尾部数据复核：

- 重读 composition、packing/build 与 conversions，确认默认 concat packing
  无条件 filter(full) 丢弃有限流末行，少于一行的验证集会全部消失。
- `packing/build.py` 仅在 repeat=True 时过滤不完整行；有限流保留末行，已有
  conversion 用 segment_ids 将 padding labels 屏蔽为 IGNORE_INDEX。
- 最小真实 Grain regression：3个有效 token、row size8，修复前0行失败；修复
  后保留1行，前三labels完整，其余mask/IGNORE_INDEX正确。使用直接TextSequence
  source隔离处理器，未以参考实现替换packing算法。
- 验收：data/validation 定向361 passed，全量CPU 3076 passed、2 skipped；
  pre-commit首次格式化新测试，复跑受影响文件通过；其余适用文件通过，排除
  `examples/native_alltoall.py`。compileall/HEAD diff检查通过。日志
  `/tmp/llmtuner-finite-tail-{before,tests,units,precommit}.log`。
- 数据源身份方面，当前 batch_layout 不能证明原始内容、tokenizer或recipe未变；
  该恢复检测缺口仍待进一步处理，不能用此尾部修复声明数据恢复全部正确。

有限 packing 文档容量与 loader 数值闭环：

- 逐段复核 `packing/iterators.py` 的文档切分、tail padding、remainder游标、
  finished状态及恢复回滚；文档容量路径保留有限末行，无新增确定性缺陷。
- 新增 `test_finite_packing_loader_preserves_all_targets` 四配置：concat/split
  与first-fit分别结合无容量限制/每行一个文档。11个已知target经过真实Grain
  packing、thread prefetch、collator、loader后全部出现且仅出现一次，统计总数11，
  padding均为IGNORE_INDEX。4 passed，日志 `/tmp/llmtuner-finite-packing-matrix.log`。
- 参考为独立已知target集合，允许first-fit重排；不以同一packer的第二个入口
  作为oracle。source通过patch注入TextSequence，未覆盖tokenizer、源读取或网络。
- 此批只增加上述具体尾部风险的测试，生产代码无变动；复用上一批3076 passed
  全量结果，新增用例已独立执行。数据源身份检测及设备并行验证仍未完成。

数据源适配与身份缺口实证：

- 逐段复核 `datasets/sources.py`：JSONL按二进制offset读取、忽略空行、检查
  dict row，pattern解析拒绝空匹配/重复resolved路径；HF streaming先DP split，
  Grain cursor deepcopy source，shuffle buffer由Grain管理。核对安装HF的实际
  state_dict实现返回deepcopy，load_state_dict保存待恢复状态。
- 真实本地HF IterableDataset（id0..4）两个cursor独立；在id1后保存，新cursor
  恢复并跨repeat边界输出 [2,3,4,0,1,2,3]，与连续流一致。无需网络；不证明
  远端数据源、HF buffer shuffle或Arrow batched-map精确恢复。
- 已确认未解决缺口：两个真实GrainDataLoader布局相同（row1/context1/DP1），
  输入来源分别base10/base100；前者消费1行后state恢复到后者被接受，下一label
  从12变为102。故layout校验无法保障source identity，不能宣称配置/内容变化
  后仍正确续训。实验退出0；进程解释器退出时Grain析构出现模块清理警告，
  本结论仅使用close之前的实际输出，不计作清理已验证。
- 下一行动：为builder生成可序列化data identity并纳入loader恢复契约，区分
  配置identity与内容fingerprint；同步DCP metadata/planner及旧checkpoint策略，
  使用同布局异内容、seed/packing/tokenizer变化回归。自定义recipe与远端
  mutable source需要明确身份供应契约，不能用对象repr或路径代替内容证据。

loader 数据身份恢复契约基础实现：

- `GrainDataLoader` 新增可选非空字符串 `data_identity`，graph owner提供稳定
  identity，记录于现有batch_layout并在set_state之前校验。未提供时保留原有
  直接graph调用契约；提供后拒绝缺少identity或不同identity的checkpoint。
- 直接恢复回归修复前因缺少契约失败，修复后通过；真实DCP save/load验证
  saved identity被读取、不同identity拒绝且target cursor不动，匹配identity
  恢复成功。2 passed，日志 `/tmp/llmtuner-data-identity-{before,tests,units}.log`。
- 累计验收：CPU全量3082 passed、2 skipped；适用pre-commit（排除
  `examples/native_alltoall.py`）、compileall及HEAD diff检查通过。
- 这只是loader能力基础，`loader_builder.py`尚未自动提供identity；上批同布局
  异内容的生产缺口仍未关闭。下一行动为生产builder构建配置/内容fingerprint，
  补齐local JSONL/tokenizer、远端revision与自定义recipe边界及兼容性说明。
- 没有放宽DCP strict planner；启用identity的loader不能从无该字段的旧DCP
  checkpoint恢复。builder启用前须增加旧checkpoint迁移/拒绝的明确说明和测试。

本地文本 builder 数据身份接入：

- `loader_builder.py::local_text_data_identity` 使用SHA256按块读取匹配JSONL和
  tokenizer.json/vocab/merges/config/chat-template内容，结合data配置、seed、
  repeat和实际dataset override生成versioned identity。生产local_jsonl/SFT
  builder向GrainDataLoader传入该值；random和注册/远端recipe尚未接入。
- 指纹在PP token budget等纯配置校验之后、tokenizer/graph构建之前生成。
  首次全量发现优先级回归（在budget校验前读取配置/文件），已调整顺序；
  保留原测试，不通过修改fixture掩盖错误。
- 生产入口回归覆盖同路径corpus内容变化、seed变化、tokenizer配置字节变化，
  恢复前拒绝；旧checkpoint缺少identity同样拒绝。各实验使用独立corpus拷贝。
  内容哈希是保守字节身份，空白变化也可拒绝；不锁定训练期文件，不保证构建
  与读取之间没有并发文件修改，每rank构建额外扫描文件带来I/O成本。
- introduction增加旧checkpoint规则：可显式排除dataloader重新开始数据流，
  但不算精确续训。DCP严格缺失字段行为保留，未补造旧identity。
- 验收：修正后全量CPU3085 passed、2 skipped；随后新增legacy用例的四项
  定向测试全部通过。适用pre-commit首次通过，后续变动文件复跑通过；
  compileall及HEAD diff检查通过。日志 `/tmp/llmtuner-builder-identity-*.log`。
- 注册HF/多模态/自定义recipe的自动identity、远端immutable revision及实际
  完整trainer恢复仍待继续验证；不能据此宣称所有数据身份缺口已关闭。

数据身份跨 rank 资产目录与真实 PP 恢复复核：

- 上批identity包含dataset/tokenizer路径，真实packed PP2 full成功、fresh resume
  失败：各rank使用data-rank-N目录，DCP统一loader metadata中的identity不能
  与另一rank相同内容的路径identity匹配。失败日志
  `/tmp/llmtuner-packed-identity-resume.log`，没有修改oracle掩盖问题。
- identity现排除dataset_path/tokenizer_path及纯性能num_prefetch_batches，保留
  按顺序JSONL和tokenizer内容hash及处理配置。新增相同内容不同目录与prefetch
  深度的identity等价回归；内容变化/seed/legacy拒绝仍通过，定向7 passed。
- 重新生成checkpoint，独立双rank full/resume均退出0：PP2、native FSDP、
  四microbatch、local_jsonl、动态EMA；续训loss差异0，模型/优化器/scheduler/
  trainer/EMA精确一致。CPUpacked attention明确使用dense SDPA测试替代，
  不证明CUDA flex生产内核。日志 `/tmp/llmtuner-packed-identity-fixed-*.log`。
- 命令：`PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHDYNAMO_DISABLE=1
  LLMTUNER_TEST_EMA=dynamic python -m torch.distributed.run --nproc_per_node=2
  --master_addr=127.0.0.1 --master_port=<PORT>
  tests/integration_tests/pp_checkpoint_equivalence.py <full|resume> <WORKDIR>
  --packed`；阶段依次执行，同一目录。
- 累计CPU3087 passed、2 skipped；适用pre-commit（排除
  `examples/native_alltoall.py`）、compileall、HEAD diff检查通过。
- 注册/远端recipe身份、训练期间文件变化、多rank非对称身份错误的协调仍待
  审查。全局目标保持未完成，下一行动转向recipe与处理器的语义身份边界。

文本监督边界与 renderer 空输出复核：

- 逐段重读 `datasets/text/processors.py` 和 `renderer.py`：单轮prompt prefix
  必须精确匹配，label屏蔽到prompt长度-1；多轮loss_mask与target一起shift，
  保留每个assistant turn监督，超长sample按现有契约drop。
- 确认renderer返回0/1token会生成空TextSequence，推迟故障到下游packing或
  有效token检查。现明确要求至少两个token构成prediction pair，否则ConfigError
  提示检查renderer/messages，不静默接收空监督输入。
- 两项边界回归修复前未报错失败；修复后通过。最初测试替身遗漏types命名空间
  导致NameError，先修正fixture再取得真实before证据，不将fixture失败算产品bug。
- 验收：renderer定向19 passed，CPU全量3089 passed、2 skipped；适用
  pre-commit（排除 `examples/native_alltoall.py`）、compileall和HEAD diff检查
  通过。
- 测试通过可选renderer接口替身，真实renderers依赖和实际模板输出数值尚未
  验证；未宣称安装/支持所有renderer类。日志
  `/tmp/llmtuner-renderer-empty-{before,units}.log`。全局目标保持未完成。

多模态 source padding 与 packing 契约复核：

- 重读multimodal collator/video/processors/placeholders/packing/recipes/mrope及
  __init__，确认source padding在packing输入转换中被丢弃，输出仅识别packer
  segment_ids新增padding，导致原padding作为有效target与attention token。
- packing现携带padding_mask token feature，新增padding填True；输出合并
  source mask和packing mask，屏蔽labels且不修改原sample。位置重建只针对
  packer新增尾部，内部source padding不重置后续真实token位置。
- 回归修复前丢失中间mask失败，修复后保留[False,True,False]，对应target
  IGNORE_INDEX；源labels未变。定向与全量结果见验收补充；日志
  `/tmp/llmtuner-mm-padding-{before,tests,units}.log`。
- 验收：多模态定向183 passed，全量CPU3090 passed、2 skipped；适用
  pre-commit（排除 `examples/native_alltoall.py`）、compileall与HEAD diff
  检查通过。
- 默认processor不含padding时使用全False保持原契约；shape/dtype消费者
  继续校验。未证明真实video解码、远端媒体或VLM模型并行训练；schema新增
  feature对旧多模态packing cursor恢复的兼容性仍需核对。

多模态 padding feature 的真实 Grain 重建恢复：

- 新增 `test_multimodal_grain_packing_restores_source_padding`，5个长度3的
  Text/media dict sample，每个包含内部padding。真实FirstFit packing两个bin，
  保存首行后state，重建graph恢复后逐tensor比较连续流，labels/input/positions/
  mask精确一致；10个独立已知有效target全保留且无重复，5个padding target屏蔽。
- 1 passed，日志 `/tmp/llmtuner-mm-padding-restore.log`。source通过注入简单
  dict隔离媒体处理，未替换Grain算法，迭代器finally关闭。
- 核对安装Grain `transformations/packing.py` 的get/set_state：保存parent/
  next_row/counter，恢复时reset并重放；没有序列化新增mask buffer。仅能据此
  说明当前实现的恢复机制，旧版本/schema cursor兼容仍未由此证明。
- 此批只补上一批schema变更的具体恢复风险；生产代码未变，复用上一批CPU
  全量3090 passed、2 skipped。新增用例与受影响pre-commit独立通过。
- 下一项：旧schema cursor重放及缺字段策略，再转回模型/并行剩余未闭环机制。
  全局目标保持未完成，真实VLM训练与远端媒体仍无本批证据。

旧多模态 packing feature schema 游标恢复验证：

- 新增 `test_multimodal_packing_resumes_legacy_feature_schema`：使用实际Grain
  FirstFit算法、旧length/padding_struct（无padding_mask字段）及旧转换输入，
  保存首行后parent/row/counter状态；退出legacy构建patch后用当前生产graph
  恢复，逐tensor对比旧图连续读取。开启/关闭bin shuffle两配置，2 passed。
- 输入样本不包含source padding，验证新增feature不改变原有正常文档的布局、
  target和cursor。若历史样本已有内部padding，修复后的label mask应有意改变，
  不承诺与旧bug行为数值一致。
- legacy patch只还原字段布局，算法仍是安装Grain真实实现；结论不跨Grain
  版本，也不证明HF源、远端媒体、VLM/并行模型训练。日志
  `/tmp/llmtuner-mm-legacy-schema.log`。
- 生产代码未变，此批新增用例与受影响pre-commit通过，HEAD diff检查通过；
  复用此前全量3090 passed、2 skipped。上一批新schema恢复1项也独立通过。
- 下一范围回到模型与并行：复核TP投影和复制参数梯度归约的参数身份与冻结
  状态，继续全局尚未闭环机制；设备/远端验证缺口保留。

TP row projection 反向 wgrad dtype 契约复核：

- 重读tp_projections.py与tensor_parallel/linear_ops.py，确认
  LinearReduceScatter.backward无条件torch.mm(out_dtype=float32)，但PyTorch
  此参数仅支持CUDA fp16/bf16输入。FP32反向不能使用这一调用；FP64也不能
  被无条件降精度。现只在半精度输入使用显式FP32累加，其他dtype普通matmul。
- FP32/FP64两项回归修复前失败，修复后对独立dy@W、dy.T@X参考一致，dtype
  保持。symm fused gather用接口替身，matmul是真实CPU算子；这证明算术分支，
  不证明CUDA fused collective或symm内核支持FP64。
- 验收：linear_ops定向6 passed，全量CPU3095 passed、2 skipped；适用
  pre-commit（排除 `examples/native_alltoall.py`）、compileall和HEAD diff检查
  通过。
- TP参数身份/冻结状态与复制梯度归约仍继续审查，没有因本项修复声明完整
  TP已验证。日志 `/tmp/llmtuner-tp-wgrad-{before,units}.log`。

TP参数替换与复制梯度归约静态复核：

- 逐段复核tensor_parallel.py的dense constructors、plan/target解析、apply_tp，
  及trainer._allreduce_replicated_tp_grads。生产变换以当前Parameter.requires_grad
  构造分片，wrapped.train(inner.training)保留模式；共享投影别名、bias、
  非整除维度在改变模型/访问group之前拒绝，共享权重不被静默解除。
- tp_sharded_param_ids在归约时解析登记名称到当前对象，避免FSDP参数替换后
  旧id失效；expert参数纳入分片集合。复制梯度SUM（不平均），DTensor对
  local shard归约；同part.parameters按身份去重，多part共享则由optimizer
  的参数所有权检查拒绝。未发现本组新的确定性缺陷。
- 已核对现有frozen/shared-before-mutation回归；前提未变，复用上一批CPU
  全量3095 passed、2 skipped，不重复运行稳定测试。静态结论不证明动态
  unused gradients在TP rank间不对称时collective安全，也不覆盖CUDA overlap。
- 下一范围：模型attention与CP布局/掩码边界，及未解决的跨rank异常协调；
  全局目标保持未完成。

固定容量 varlen metadata 的 kernel 最大长度复核：

- 重读attention/masks.py，固定容量路径验证document数量，却直接把
  max_context_length声明为max_q/max_k，没有检查cu_seqlens实际segment长度。
  positions=arange(8)、capacity1、context4原先返回长度8但bound4的不一致
  metadata。现对固定cu_seqlens差分做异步上限断言，不增加host同步。
- 新增修复前不报错的回归，修复后RuntimeError明确指出max_context_length。
  masks定向34 passed；合法文档与padding容量现有用例保持通过。
- 累计CPU3096 passed、2 skipped；适用pre-commit（排除
  `examples/native_alltoall.py`）、compileall和HEAD diff检查通过。
- 本项证明metadata契约和CPU断言，不证明GPU varlen内核或CP数值；上游语义
  对齐仍以实际输入与内核契约为准。日志 `/tmp/llmtuner-varlen-bound-*.log`。
- 下一步继续CP layout、GQA与padding跨rank路径；全局目标未完成。

双CP轴 GQA非二次幂本地heads 数值复核：

- 重读context_parallel/attention.py的SeqToHead/HeadToSeq、KV gather、dual
  kernel及run_flex，核对安装HF flex实现：CUDA non-power-of-two local query
  heads显式repeat_kv，CPU eager启用GQA；dropout在HF/CPU两边均拒绝。
- 扩充 `dual_cp_attention_equivalence.py` Q6/KV2（ratio3，Ulysses2后Q3/KV1），
  配合已有Q4/KV4、Q4/KV2，共9种head/mask配置。真实4rank CPU/Gloo KV2×
  Ulysses2，noncausal/causal/packed，输出和QKV梯度与独立FP64dense SDPA
  参考在rtol/atol1e-10内一致，进程退出0。仅扩充具体GQA边界，不重复宣称
  旧稳定用例为新实现。
- 命令：`PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 python -m torch.distributed.run
  --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29955
  tests/integration_tests/dual_cp_attention_equivalence.py`；日志
  `/tmp/llmtuner-dual-cp-gqa3.log`。随后print增加Q_heads避免日志中同KV配置混淆。
- attention算术被明确替换为SDPA，保留真实all-to-all/KV allgather autograd；
  不证明生产flex、CUDA kernels或padding路径。此批生产代码未变，复用此前
  CPU全量3096 passed、2 skipped；脚本/文档pre-commit及HEAD diff检查通过。
- 下一步核对CP输入/attention mask对padding与load balancer的对齐；全局未完成。

CP→TP 输入及 padding 对齐复核：

- 阅读 HF wrapper 的 preprocess_inputs、_shard_for_cp、_shard_for_tp、
  get_attention_masks 与 _apply_attention，以及 context_parallel/input_sharding.py。
  CP 先切 tokens/labels/positions/padding；TP 再切 tokens/labels/padding，
  positions 保留 CP 长度，供投影 gather 后的 attention/RoPE 使用。
- 新增 test_input_sharding.py，使用实际 PyTorch 私有分片 helper 与 DTensor
  local partition，fake process group 提供 CP2×TP2 的四种 rank 坐标；
  contiguous/headtail 共8例。独立按连续区间或头尾chunk构造索引，检查全部
  token字段、内部padding和输入未被修改；8 passed，Ruff check/format通过。
- fake backend不验证真实通信、attention或目标设备；此批未确认新的生产缺陷。
  padding_mask当前只传给MoE路由/统计，attention按positions/文档边界构建，
  因而本项不能证明内部padding被attention排除。下一步追踪多模态源内部padding
  的语义，确定是否需要attention键屏蔽及其CP/TP/PP传播，避免误判正常尾padding。
- 仅新增定向测试与记录，生产代码未变，不重复全量CPU；全局审查仍未完成。

多模态 batch 到 HF 文本 decoder 的输入消费复核：

- 追踪 multimodal/processors.py、collator.py、packing.py 到 HF
  preprocess_inputs/forward；内置processor未生成内部token padding，视觉patch
  padding与token padding不可混同。已有源padding保留不证明attention排除语义。
- 确认另一生产缺陷：collator产生pixel_values、视频、grid及可选MRoPE，
  HF wrapper仅取input/labels/positions/padding，随后调用文本_decoder；媒体
  被静默丢弃，训练可按占位token继续。该入口没有视觉encoder或媒体融合路径。
- preprocess_inputs现在在分片/attention之前逐项拒绝非None的pixel_values、
  pixel_values_videos、grid_thw、grid_thw_videos、mrope_positions，使用
  UnsupportedCombinationError指出具体未消费字段及可行选择；None媒体和
  special_tokens元数据仍允许文本处理。此修复不是完整视觉模型迁移。
- 五个回归修复前均未抛错；修复后HF+多模态定向250 passed，全量CPU
  3102 passed、2 skipped，退出0。全仓适用pre-commit排除受保护
  examples/native_alltoall.py后通过；compileall及HEAD diff检查通过。
  日志 /tmp/llmtuner-media-guard-{before,targeted,units}.log。
- introduction区分多模态数据管线与模型消费能力。剩余：真实视觉适配器尚未
  实现；通用内部token padding的attention契约仍须明确及验证，不能将本次
  输入拒绝视为CP/TP/PP多模态支持完成。下一步补齐未读模块与跨rank失败协调。

训练取数的跨rank耗尽与异常协调：

- 阅读 trainer/runtime/batch.py、trainer.train_step 的累积预读与train循环。
  原batch_generator仅本rank抛DataloaderExhaustedError；有限DP分片不等长时，
  其他rank仍yield并进入token denominator/model collective，不能协调取消step。
- batch_generator在fetch及labels结构检查后、yield前对默认world group执行
  MIN状态归约；错误优先于耗尽。所有rank一起耗尽，源错误rank保留原异常，
  其他rank报明确peer失败；不记录未yield批次的metrics。默认world包含PP/CP/TP
  同步peer，不能只归约DP。单进程不增加collective。
- 新data_fetch_coordination.py用真实两rank CPU/Gloo，第二次fetch分别模拟
  仅rank0耗尽、ValueError及缺labels。修复前rank1错误yield导致脚本失败；
  修复后三例均协调，tokens只计成功首批，并能继续barrier，最终脚本退出0。
  命令：PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 python -m torch.distributed.run
  --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29957
  tests/integration_tests/data_fetch_coordination.py。
- 全量CPU3102 passed、2 skipped；适用全仓pre-commit自动整理一处import后
  定向重跑通过，compileall及HEAD diff检查通过，保护示例未纳入工具输入。
  日志 /tmp/llmtuner-fetch-{before,after,final,units}.log。
- 本机制每个分布式fetch增加一次scalar collective和状态host同步，未测设备
  性能；仅覆盖训练batch_generator，不能协调初始化、验证或模型执行中的异常，
  也不能解救阻塞于next的loader。取消step已预读的游标回滚仍未实现。下一步
  检查验证固定步数的数据耗尽与训练取消step保存状态契约，全局未完成。

验证循环复用跨rank取数契约：

- 阅读validation.py的create/run/finalize路径。正steps使用repeat loader，
  但空DP分片或source异常仍可能单rank结束；原next后直接break不协调其他rank
  的count/model/loss collective，训练取数修复未覆盖该路径。
- 提取batch.py::fetch_batch承载world范围的fetch/labels结构/耗尽协调，训练
  batch_generator只负责metrics，验证在计token/forward前复用。避免两份
  collective顺序和异常优先级实现；不放开现有DP>1完整有限验证限制。
- 扩充data_fetch_coordination.py运行真实run_validation_loop，替换loader与
  简单模型算术，使用实际两rank CPU/Gloo mesh、token/loss归约和SPMD上下文。
  不对称耗尽时两边只评估共同首批，独立sum参考loss1.5，其他例ValueError/
  缺labels两边协调失败且不报告loss，所有loader关闭、metrics不计尾批。
  首次脚本失败是测试topology缺SPMD方法，修正后六种训练/验证场景退出0。
- 命令同data_fetch_coordination.py两rank启动，端口29958；日志
  /tmp/llmtuner-validation-fetch-distributed.log。训练/验证定向174 passed；
  全量CPU3102 passed、2 skipped，日志 /tmp/llmtuner-validation-fetch-units.log；
  全仓适用pre-commit（排除保护示例）、compileall及HEAD diff检查通过。
- 每次fetch的scalar collective/host同步成本未测；模型和预处理异常、loader
  初始化/阻塞、token元数据错误尚未协调。验证当前按共同成功前缀报告，不构成
  DP不等长完整语料评估。下一步核对取消训练step的cursor/计数保存与未读模块。

有限训练耗尽后的最终保存与计数契约：

- 阅读train_step预读、microbatch token计数、train耗尽分支及checkpoint最终
  保存策略。现有不完整累积窗口回归明确ntokens_seen为已消费token，游标/
  计数不回滚并不等同于权重已更新；避免擅自改为optimizer有效token语义。
- 确认正常有限退出遗漏最终save：耗尽分支只撤销step并break；最后已完成
  更新不在interval时，close仅等待已有保存，无法保留最新权重。现在step回退
  后若step>0调用save(last_step=True)，使用既有最终策略，未训练新运行不创建
  step-0。已保存同step的重写由此前checkpoint generation机制负责。
- 新train-loop回归完成0/1step后耗尽：1step用例修复前缺最终save失败，
  修复后校验completed step、normal/final save顺序及资源close；0step不保存。
  回归以替身train_step/checkpointer隔离循环行为，不证明实际DCP写入或恢复。
- trainer+checkpointer定向212 passed；全量CPU3104 passed、2 skipped；全仓
  适用pre-commit（排除保护示例）、compileall及HEAD diff检查通过。日志
  /tmp/llmtuner-exhaust-save-{before,targeted,units}.log。
- introduction明确耗尽窗口游标/计数、最终保存及model-only不能精确续训。
  不完整窗口不会执行optimizer；最终保存仍服从enable及导出配置。后续需补
  真实有限数据耗尽产物的fresh-runtime恢复证据；跨rank模型异常和设备验证
  仍未完成，全局目标继续。

有限数据耗尽最终checkpoint的真实fresh-runtime恢复：

- 在上一批最终save修复之上补test_finite_training_final_dcp，用真实Trainer、
  模型前后向、Adam/scheduler、synthetic checkpointable loader与DCP文件写入；
  islice仅限制三个batch，累积2，计划8step、interval100、最终完整保存。
- 训练完成1step后耗尽，第三batch不更新参数，但cursor3/ntokens48；验证
  step-1/.metadata实际存在、step-2不存在。新建Trainer load(1)后模型与
  optimizer张量rtol/atol0一致，scheduler、trainer和loader状态完全一致；
  下一batch等于独立RandomTokenSource.iter_from(3)参考，证明取消窗口消费
  的游标与权重completed step在当前契约下可恢复。
- 单进程CPU真实DCP用例1 passed；测试文件pre-commit修正行宽后重跑通过。
  此批只增测试/记录，生产未变，复用上一批全量3104 passed、2 skipped，
  不将本例当作多rank或设备保存恢复证据。日志/tmp/llmtuner-finite-final-dcp.log。
- 该窗口恢复缺口已补足单进程证据；下一批转向剩余文件覆盖及模型/parallel
  未解决契约，不继续无依据扩展稳定checkpoint测试。全局目标未完成。

多本地model parts的optimizer pattern归属：

- 阅读optimizer.py参数计划/构造/参数校验，lr_scheduler.py全实现及对应配置。
  原_build_param_groups要求每规则在每part命中，分别针对first/second本地part
  的合法规则会在第一part构建时失败；新回归修复前ValueError复现。
- 空本地匹配现在跳过，通过配置索引记录成功归属，在全部本地计划完成后
  校验每条规则至少实际分配参数。仍在optimizer构造前检查未分配参数、重复
  所有权、FQN冲突和非法optimizer。按索引避免同pattern多次配置被误认为有效。
- 新two-part测试不同lr与独立AdamW更新rtol/atol0一致；新增重复规则遮蔽
  多part拒绝回归。原later_pattern_mismatch测试改为第一part均覆盖、第二part
  参数未分配的真正非法情况，保留构造前拒绝断言，并非弱化参数所有权检查。
- optimizer定向230 passed（重复规则回归随后加入全量）；全量CPU3107 passed、
  2 skipped；全仓适用pre-commit排除保护示例、compileall、HEAD diff通过。
  日志/tmp/llmtuner-group-parts-{before,targeted,units}.log。
- scheduler静态核对warmup/virtual final step、预更新lr metrics、共同epoch、
  restore初始lr及越界拒绝，未发现新确认缺陷。多part修复不改变collective，
  不证明跨PP rank规则裁剪；构造文档明确仅本地parts，无implicit collective。
  后续检查全局PP配置规则分配及尚未覆盖模块；全局目标仍未完成。

PP stage专用optimizer规则的装配与跨rank验证：

- 继续追踪assembly._build_optimizer_stack，统一cfg.optimizer未经stage过滤
  传给每个rank；上一批仅本地parts匹配仍拒绝只在其他PP stage命中的合法规则。
- OptimizersContainer新增显式pipeline_mesh，可按PP group MAX归约规则的实际
  分配索引；Trainer装配传入PP mesh。没有mesh的直接构造仍无collective。
  本地参数归属/冲突检查保留；规则全pipeline均未分配才拒绝，先匹配仍生效。
  API要求所有PP rank同时构造并使用同序规则，不自动使用默认world通信组。
- 新pp_optimizer_groups.py真实两rank Gloo，stage0/1不同lr，独立AdamW更新
  rtol/atol0一致；globally unused与重复遮蔽规则两边均拒绝且barrier可继续。
  首次脚本错误读取container.param_names，改为inner optimizer后退出0。
- pp_equivalence增加LLMTUNER_TEST_OPTIMIZER_GROUPS=1模式，真实Trainer PP2
  embedding/lm_head/catch-all三规则（lr相同以隔离归属）4step对独立dense参考：
  loss最大差4.768e-7、gradient3.260e-8、parameter1.192e-7，失败rank0，退出0。
  默认规则PP2原有路径同样通过。端口29959/29960；日志
  /tmp/llmtuner-pp-groups.log、/tmp/llmtuner-pp-{groups,stage-groups}-training.log。
- 全量CPU3107 passed、2 skipped；全仓适用pre-commit排除保护示例、compileall、
  HEAD diff检查通过。未验证设备后端或不同rank配置、构造前局部异常的协调；
  不宣称全部PP×FSDP/TP/EP组合完成。下一步继续未读文件及这些异常边界。

MoE辅助loss指标的当前runtime层数归属：

- 阅读aux_loss.py、moe/auxiliary_loss.py、gate.py及bias_update/layer统计消费。
  RouterGate强制FP32，balance counts/probabilities前向Partial→Invariant、
  identity backward；padding仅过滤统计，recompute抑制metric写入。
- 确认_group_counts是进程内所有构造实例累积，先构造其他模型后当前runtime
  rollup只汇总当前parts，却按旧总层数除，指标被稀释。Trainer现在向collector
  显式传model_parts，按distinct local AuxLoss实例计数，PP轴SUM层数，不归约
  复制data轴。无当前实例的旧group跳过；注入梯度及loss分母不变。
- 新CPU回归previous2/current1，当前raw metric2正确报告2；aux定向33 passed。
  新aux_loss_runtime_metrics.py真实PP2 Gloo，历史模型rank各5/6层，当前各1/2层，
  独立sum6/count3=2，两rank一致、rollup清空instance scratch，退出0。
- 真4rank PP2×EP2 Trainer四步loss仍匹配独立dense参考（脚本tol1e-4），退出0；
  该脚本不单独证明每项aux metric，而上述PP2脚本负责层数语义。端口29961/29962，
  日志/tmp/llmtuner-aux-runtime-{pp-ep,metrics}.log。
- 全量CPU3108 passed、2 skipped；全仓适用pre-commit（保护示例排除）、
  compileall及HEAD diff检查通过。日志/tmp/llmtuner-aux-runtime-count{-units,}.log。
- standalone collector不传parts仍使用构造注册计数；global group key registry、
  step denominator及group_acc仍未完全runtime隔离，不宣称多个交错Trainer安全。
  不同rank历史metric group key不一致可能导致collective顺序风险，后续需按
  runtime重组注册表。目标仍未完成，下一范围这些所有权与未覆盖文件。

Aux metric group key按当前runtime对齐：

- 上一批仍遍历进程构造_group_counts；各rank历史分组不同或某stage空注册表时，
  group tensor collective数量/顺序可不一致。现在model_parts路径收集distinct
  当前实例的Counter，PP先all_gather_object当前keys并排序统一，再归约sum/count；
  空stage参与key交换，无当前group的dense runtime不报告旧指标。
- 扩充aux_loss_runtime_metrics.py：两rank历史AuxLoss子类不同，当前层数1/2；
  再清空rank1注册表且使其无当前loss，rank0一层raw3，双方mean3且barrier可继续。
  真实CPU/Gloo退出0，命令端口29962；/tmp/llmtuner-aux-runtime-keys.log。
- 新CPU dense runtime排除stale group回归；aux定向34 passed。四rank真实
  PP2×EP2四step仍匹配dense reference，退出0；日志
  /tmp/llmtuner-aux-runtime-keys-{targeted,pp-ep}.log。
- 全量3108 passed、2 skipped是在新增最后一个测试前运行，之后仅加测试/文档，
  生产前提未变；不虚报全量3109。全仓适用pre-commit排除保护示例、compileall、
  HEAD diff通过。日志/tmp/llmtuner-aux-runtime-keys-units.log。
- 每次PP指标收集新增一次对象key交换，性能与NCCL/HCCL未验证；后续可在
  runtime构建时登记稳定layout以减少开销。group_acc及_step_denominator仍
  共享，交错Trainer完整隔离未解决；standalone无parts路径仍使用注册表。
  本项仅解决当前Trainer的group选择/PP顺序，不宣称全部生命周期隔离完成。

工具层职责与GC进程状态所有权：

- 重新核对161个源码路径与台账161条不同路径：无遗漏、重复或stale路径。
  阅读utils/module_names.py、dtypes.py、lazy_exports.py、__init__.py、monitoring.py、
  garbage_collection.py，定位optimizer/EP checkpoint、model_setup、包导出、
  profiler/metrics/checkpointer及Trainer消费者。路径覆盖不证明全部runtime正确。
- module_names只删完整wrapper segment，消费者检查FQN冲突；lazy exports按
  明确mapping导入leaf，未知属性抛AttributeError；dtype入口由配置约束；
  monitoring CPU可选history为None，设备probe来自选定device module。未因
  helper体积小合并有独立契约模块，硬件峰值和memory API未做设备实测。
- 确认两个GC owner重叠时，原first.close会提前enable。现在class锁保护active
  owner计数与首次状态，最后close才恢复，非LIFO也有效；初始化collect失败
  通过既有close释放自己的owner，保留其他owner。close仍幂等。
- 四种初始状态/关闭顺序回归修复前1失败3通过，修复后通过；新增第二owner
  初始化失败仍保持首owner的用例。旧positive-frequency测试补实际close，避免
  只手动enable泄漏生命周期。utils定向84 passed（随后失败路径用例加入全量）。
- 全量CPU3114 passed、2 skipped；全仓适用pre-commit排除保护示例、compileall
  及HEAD diff通过；日志 /tmp/llmtuner-gc-owners-{before,targeted,units}.log。此为进程GC策略，不证明
  allocator性能、设备内存释放或多个交错Trainer的完整状态安全。
- 下一步继续通用状态/设备工具和跨模块职责，Aux runtime隔离仍保留待办。

设备/张量树与grouped GEMM能力探测对齐：

- 阅读accelerator/tensor_tree.py、device.py、capabilities.py的probe入口，以及
  utils/seeding.py；核对collectives.get_comm_device、GroupedExperts调用和现有
  tensor tree/seed测试。树搬运保留namedtuple/defaultdict并在输出修改前检查
  结构；不支持scalar元数据，Trainer的metadata batch不能直接替换为该helper。
- 确认grouped_mm_runs原probe始终CPU，weight连续，而生产GroupedExperts在
  训练设备执行weight.transpose(-2,-1)；CPU成功/失败不能决定GPU路径，stride
  也可能造成误判。probe现使用get_current_device并同样转置expert weights，
  inputs/weights/int32 offsets同device，保留异常fallback和缓存。
- 新CPU/meta两例用算子替身检查实际张量device/dtype/stride；修复前均失败，
  修复后通过。它们只证明probe输入契约，不证明CUDA/NPU内核。accelerator+
  MoE定向236 passed；全量CPU3116 passed、2 skipped；全仓适用pre-commit
  排除保护示例、compileall、HEAD diff通过。
- 日志/tmp/llmtuner-grouped-probe-{before,targeted,units}.log。probe仍仅代表
  一组形状、当前选定设备与前向支持，非所有专家尺寸/反向证明。seed的fork_rng
  及DTensor tracker边界未发现本批新确认缺陷；确定性开关的全局生命周期继续待审。

FSDP pattern/prefetch语义与上游静态复核：

- 逐段阅读fully_shard_parallel/plan.py、module_patterns.py，核对默认decoder
  prefetch入口、expert placement、FSDPTurbo的set_modules_to_prefetch及安装
  PyTorch2.14 FSDPState._pre_backward/FSDPParamGroup.pre_backward。参数所有权
  在变换前解析，跨层共享/cycle合并state，冲突effective options先拒绝；
  ignored根排除整子树，hook覆盖在shard调用之前检查。未确认本批新行为缺陷。
- PyTorch default_prefetch由backward显式state列表是否为空决定，空列表触发
  默认_backward_prefetch。config的count0仅表示无显式窗口；decoder EP另保留
  默认1单元，而pattern直接用count。纠正helper“exact”措辞及字段说明，
  不把0误解为禁用引擎所有预取，不修改引擎策略。
- 对比文档5.2/5.4/13仍宣称pattern/custom/window未迁移，已按当前生产路径
  更新为已实现机制及设备验证缺口；CPU offload示例改用FSDPConfig嵌套字段。
  pattern注册顺序与任意forward执行顺序不可等同，性能和设备重叠仍待验证。
- 仅说明/docstring变更，适用pre-commit、HEAD diff通过；复用此前CPU3116
  passed、2 skipped，不重复训练测试。静态依据不能证明任意pattern配置的
  后端值校验均在变换前完成；无效覆盖值造成partial mutation仍需继续核对。
- 下一批检查逐unit effective options在调用引擎前的完整校验和异常所有权；
  全局审查目标保持未完成。

FSDP pattern逐单元reshard策略预检：

- 确认plan.py仅预检override键、hook及共享参数选项冲突，后续unit的非法
  reshard_after_forward可能在前序unit变换后才由引擎拒绝，留下部分变换模型。
  安装PyTorch2.14的fully_shard/_get_post_forward_mesh_info及custom入口均已
  核对。本批在全部grouped effective options解析后、首次shard之前检查策略。
- 保留None/bool及合法整数；整数必须是实际unit mesh分片轴大小的正因子，
  HSDP不使用总mesh大小，per-unit mesh覆盖生效。sharded dp_mesh_dims的整数
  策略按原生/custom契约提前拒绝，replicate-only named DP无post-forward
  sharding策略。尚未宣称所有override值和placement回调均已完整预检。
- 非法字符串、0、负数、不整除值及超过shard轴大小的10项回归修复前均失败；
  修复后通过且assert零shard调用。另覆盖合法None/bool/1/2/4与unit mesh，
  以及native/custom SPMD整数拒绝。目标文件92 passed。
- 完整CPU3132 passed、2 skipped；该次收集后补充的2项SPMD回归包含在上述
  92项定向结果中。真实2rank CPU/Gloo pattern_fsdp_equivalence.py的26种
  native/custom布局通过独立dense更新参考，包含共享参数、prefetch及HSDP。
  日志/tmp/llmtuner-pattern-reshard-{before,units,integration}.log。
- 下一行动检查mp/offload policy、ignored_params与placement失败是否仍可能
  延迟到部分变换后；此批未修改collective或数值语义，不证明目标设备内核、
  重叠性能及所有并行组合。全局目标保持未完成。

FSDP pattern policy/exclusion输入边界复核：

- plan.py新增变换前的mp_policy/offload_policy类型检查、shard_placement_fn
  callable检查、ignored_params的set[Parameter]检查。原先非法precision或
  placement对象可能延迟失败；字符串排除项不会匹配实际Parameter，可能静默
  失效。native按公开policy类型契约拒绝None，custom保留None默认初始化语义。
- 8项非法覆盖回归修复前均失败，修复后确保零shard调用；4项None策略回归
  区分native/custom。既有有效policy覆盖测试改用实际MixedPrecisionPolicy，
  不再用无法运行的object替身。ignored全局边界和合法unit覆盖既有测试保留。
- 完整CPU3146 passed、2 skipped；2rank CPU/Gloo pattern等价脚本再次通过
  26种native/custom布局；适用pre-commit、compileall、HEAD diff通过。
  日志/tmp/llmtuner-pattern-policy-{before,units,integration}.log。
- 未提前执行placement回调，避免改变有状态用户函数；回调返回值、用户代码
  异常和实际device分配失败仍可能发生在变换中，未承诺任意引擎失败可回滚。
  下一行动转向训练运行时AuxLoss共享状态隔离与异常清理，保留该引擎失败边界
  作为待办。CPU证据不证明CUDA/NPU内核及重叠；全局审查未完成。

AuxLoss运行时分母、指标与失败累积隔离：

- 确认当前model_parts计数仍读取共享group_acc：模型B的rollup覆盖A尚未收集
  指标；共享_step_denominator也不能表达交错运行的模型分母。两个新回归均
  在修复前失败。Trainer现在显式传model_parts，把分母写入实际AuxLoss实例；
  optimizer hook保存每个实例的指标快照，runtime收集只读取当前实例快照。
- 新step清除instance_acc和旧快照，丢弃未到optimizer hook的失败累积；rollup
  按实例身份去重，避免重叠model parts再次覆盖快照。无model_parts的独立调用
  保留类默认分母/group registers。生产运行时不再依赖其数值，但构建registry
  仍用于现有has_pending_counts入口，不宣称全局Torch/Trainer状态已完全隔离。
- 定向AuxLoss/Trainer147 passed。完整suite首次2项失败定位为staging测试
  使用Trainer.__new__却未提供真实runtime的model_parts；补齐测试模型归属，
  原有快照与实际buffer/weight变化断言保持。最终3148 passed、2 skipped。
- 2rank CPU/Gloo aux_loss_runtime_metrics.py通过异构/空PP stage；真实4rank
  pp_ep_equivalence.py --activation-checkpoint full --aux-coefficient 0.3四步
  loss、梯度和AdamW参数匹配独立dense参考（tol1e-4）。适用pre-commit、
  compileall和HEAD diff通过；上游映射同步实例状态与当前证据。
- 日志/tmp/llmtuner-aux-isolation-{before,targeted,units-final,pp-ep,pp-metrics}.log。
  未证明同一个模型的并发执行安全、CUDA/NPU/compile region全部组合或失败后
  引擎状态通用恢复。下一行动核对模型/optimizer异常后Trainer退出及资源所有权；
  全局目标仍未完成。

Trainer退出链与trace部分启动资源复核：

- 阅读CLI.main、Trainer构造/train/close、Profiler生命周期、checkpoint
  manager.close/writer和GrainDataLoader恢复/close。现有Trainer释放全部资源并
  保留首个错误，训练和CLI保留原始异常；interrupted checkpoint future保留
  资源owner，尚未确认这些路径的新缺陷。writer发布与进程组所有权保持。
- 确认Profiler.build_torch_profiler在底层__enter__返回前仅持局部句柄，若
  torch profile.start已获取资源后抛错，外层__exit__拿不到句柄。PyTorch2.14
  的start先_transit_action再进入step record_function，确有部分启动边界。
  现在在该失败点尝试__exit__；停止再次失败仅记录，不掩盖原始BaseException。
- RuntimeError/KeyboardInterrupt×cleanup成功/失败4项回归修复前均失败，
  修复后通过。补充实际CPU profiler先成功启动再注入异常的测试：退出导出真实
  gzipped JSON trace，随后新的trace可正常启动和导出。Profiler文件58 passed；
  Profiler+Trainer先前定向168 passed。
- 完整CPU3152 passed、2 skipped；全套收集后新增的真实trace回归已包含在
  58项定向结果。适用全仓pre-commit、compileall、HEAD diff通过；保护
  examples/native_alltoall.py。日志/tmp/llmtuner-profiler-entry-{before,
  targeted,units}.log。设备trace、 allocator历史和跨rank失败仍非CPU证明。
- 下一行动检查数据源/处理配方的恢复身份和远端数据变化边界，避免继续重复已
  覆盖的退出测试；全局审查及目标设备验证仍未完成。

配置驱动注册/远端语料的声明恢复身份：

- 复核loader_builder、sources、text recipes、composition及tokenizer实际文件
  读取。local JSONL已有内容哈希；其他recipe传data_identity=None，虽直接
  GrainDataLoader可接收调用方身份，但配置/CLI无对应入口。HF revision是加载
  参数，不是完整配方或内容证明，不能仅凭固定名称宣称精确恢复。
- DataloaderConfig新增可选data_identity，校验非空字符串，synthetic拒绝
  无效选项。corpus_data_identity替代仅local命名的内部函数，绑定调用方
  snapshot ID、seed、repeat、数据配置及tokenizer资产；local继续哈希JSONL。
  未设置字段时去除新增None键，保持既有local-text-v1指纹算法；注册/远端未
  声明时保持旧行为。无消费者旧函数名清理完成，CLI由既有配置view自动接通。
- 真正Grain loader注册recipe回归覆盖相同身份恢复的下一批input/labels/
  positions/padding exact一致，声明ID/seed/tokenizer变化拒绝游标；local
  新声明变化仍拒绝，CLI字段及非法/unused identity覆盖。定向597 passed
  （后续补充3项包含于完整suite）；完整CPU3163 passed、2 skipped。
  全仓适用pre-commit、compileall、HEAD diff通过，保护native_alltoall示例。
  日志/tmp/llmtuner-corpus-identity-{targeted,units}.log。
- introduction明确调用方必须固定实际远端版本、在配方或内容变化时更新ID；
  相同声明不能证明远端内容一致，不验证网络Hub/对象存储内容，也不锁定构建后
  文件。未宣称自动任意recipe指纹或远端精确恢复完成。下一行动继续检查HF
  streaming cursor状态与shuffle/epoch恢复，目标设备矩阵仍是全局未完成项。

HF streaming重复epoch重置失败状态复核：

- 阅读HuggingFaceCursorIterator、HF source构建、真实重复epoch与Grain window
  shuffle恢复用例。每cursor source deepcopy、显式set_state旧generator关闭
  和失败关闭已有覆盖；确认__next__重复重置缺少相同失败边界，set_epoch/
  load_state_dict/iter可能修改来源后失败，下一次读取仍进入不一致状态。
- 重复重置现在捕获BaseException并关闭Grain cursor，原始异常继续抛出，
  epoch仅在成功重置后提交。正常耗尽与首个新epoch样本读取不改动；不将任意
  source的读取错误解释为安全可重试。3项epoch/restore/iterate失败回归修复前
  均失败，修复后验证关闭后next/set_state不再接触source。
- 数据pipeline319 passed，包含真实HF跨epoch恢复与Grain窗口buffer恢复；
  完整CPU3166 passed、2 skipped；适用全仓pre-commit、compileall和HEAD
  diff通过。日志/tmp/llmtuner-stream-reset-{before,targeted,units}.log。
  上游映射删除过期“逐字一致”描述，记录当前source隔离与失败适配。
- CPU本地generator不证明网络流重连、服务端内容版本和HF外部shuffle精确
  恢复；HF自身buffer shuffle仍不采用。下一行动核对全量文件覆盖与残余必需
  实现/设备验证待办，避免把当前局部通过视为全局完成。

全量清单与custom relocated-hook Trainer恢复闭环：

- 当前161个Python源码与表格161个不同路径完全一致，missing/stale/duplicate
  均为空；全部状态包含已阅读。只证明阅读清单覆盖，不证明全部实现及设备组合。
  实际torch2.14 CPU，CUDA不可用，torch_npu/fla/torchao/renderers未安装。
- 对照custom state/param group及迁移矩阵，选取CPU可运行的Trainer恢复缺口：
  pp_checkpoint_equivalence.py新增RELOCATED_HOOKS选项，模型构建后按实际
  named_modules路径设置四attention projection的ancestor hooks，仍走真实
  并行装配/Trainer/AdamW/DCP/EMA。运行中断言四不同context及hook归属。
- 首次fixture误用layers属性路径，模型注册路径不同而被pattern guard拒绝；
  改为实际模块路径。resume首次在forward前断言四context，确认其由lazy_init
  创建，改为初始化后强断言；每个训练step继续验证四context，无弱化数值参考。
- 2rank CPU/Gloo full/resume独立进程、PP1/FSDP2/custom、dynamic EMA，
  2+2步恢复loss、模型、AdamW、scheduler、trainer counters和EMA逐位一致。
  日志/tmp/llmtuner-relocated-trainer-{full,resume}.log；临时workdir
  /var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-relocated-trainer-0xkpcawl。
  仅扩展集成脚本/文档，复用CPU3166 passed、2 skipped基线；适用质量门禁通过。
- 比较文档移除单Trainer装配未验证的过期结论。仍缺PP relocated组合、混合
  精度/compile/目标设备、MC2 NPU、KCP内核、Domino完整设备重叠证据；下一行动
  验证PP relocated pattern如何绑定分割后本地parts，不能以PP1代替该组合。
  全局目标保持未完成。

PP全模型FSDP pattern与复制对象绑定修复：

- 确认parallelize原先把同一pattern配置直接用于每个part，跨stage具体FQN
  在其他stage报未命中；hook_module/ignored_params仍持全模型原始对象，不能
  指向PP deepcopy后的本地执行。resolve_pipeline_fsdp_configs在模型分割后、
  local transforms前全模型解析计划，按实际注册FQN选择每个part单元并重绑
  hook及Parameter排除集合；mesh/policy保持identity，原配置不改动。
- 全局未命中仍拒绝；没有选中child的part使用显式空FQN root计划，确保剩余
  参数仍由root FSDP管理。ignored pattern也按本地模块筛选。原模型hook祖先
  和exclusion类型预检保留，避免非法值在重新绑定时被掩盖。单元命名仍以实际
  注册路径为准，不把HF属性路径猜作FQN。
- CPU回归覆盖复制对象hook/ignored identity、mesh identity、原配置未变、
  root-only计划及全局未命中。最初fixture用Sequential删除导致重编号，改为
  保留注册键以匹配真实PP分割语义。FSDP定向106 passed，完整3168 passed、
  2 skipped；最后补充的原模型预检和root计划断言已由定向suite复核。
- 真实4rank PP2×FSDP2/custom/1F1B/4microbatch/dynamic EMA独立full/resume
  2+2步：loss差异0，model/AdamW/scheduler/trainer/EMA逐位一致，训练中确认
  四attention投影使用复制后的ancestor hook及四不同通信context。脚本扩大
  RELOCATED_HOOKS选项为PP可用；比较文档更新证据。
- 日志/tmp/llmtuner-pp-{pattern-units,relocated-full,relocated-resume}.log，
  workdir /var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-pp-relocated-i45jdjr8。
  全仓适用pre-commit、compileall、HEAD diff通过。AC/compile后的命名、
  有状态placement回调捕获的旧Parameter及其他PP schedules仍需独立核对；
  下一行动检查这些变换与pattern契约，设备矩阵和全局目标仍未完成。

FSDP pattern透明wrapper路径及PP/AC组合复核：

- 确认AC的_checkpoint_wrapped_module、whole-model compile的_orig_mod让
  原始具体pattern不再命中。plan解析加入逻辑路径别名，物理路径仍保留；wrapper
  边界选择外层模块，实际内部单元仍返回canonical物理路径，共享参数规划不变。
  真实checkpoint_wrapper/torch.compile(eager)回归两项；禁用逻辑映射对照均
  报ignored pattern未命中，修复后FSDP108 passed，完整CPU3170 passed、2 skipped。
- fine_grained_fsdp_equivalence.py真实2rank默认16配置通过，包含独立dense
  参数/梯度、失败reset、重计算与DCP；日志/tmp/llmtuner-fsdp-wrapper-integration.log。
- checkpoint harness增加LLMTUNER_TEST_AC，可运行真实full AC组合。4rank
  PP2/FSDP2/custom/relocated/full AC的full阶段失败：SAC backward报告
  profiler._record_function_enter_new调用index8无对应storage。未绕过操作
  缓存或弱化断言。相同PP/custom/full AC但默认module hooks对照四步通过，
  将剩余风险定位到region内relocated hooks/recompute操作序列。
- 失败日志/tmp/llmtuner-pp-hook-ac-full.log，对照/tmp/llmtuner-native-hook-ac-audit.log。
  wrapper路径修复不代表该组合通过；compile仅验证真实wrapper路径，没有
  宣称分布式compiled数值。下一行动检查SAC effect缓存及FSDP重计算hook行为，
  修复该确认失败后重新运行full/resume。适用质量门禁通过；全局目标未完成。

PP relocated hooks/full AC确认失败修复与恢复闭环：

- 核对PyTorch2.14 FSDPState._pre_forward/_post_forward和SAC dispatch。
  PRE_BACKWARD分支负责重计算，不注册新forward生命周期；PP多个microbatch
  共用relocated ancestor后state可能IDLE，replay误走正常forward，导致额外
  post_forward profiler操作触发SAC invocation index8不存在。默认hooks/full
  AC四步对照通过，前批relocated/full AC失败保留作为before证据。
- FineGrainedFSDPState._pre_forward检查本项目显式in_checkpoint_recompute
  标记并设PRE_BACKWARD，调用原生重计算unshard/cast逻辑；未更改SAC缓存、
  未跳过collective或弱化比较。CPU两分支回归与FSDP110项通过。
- 原失败4rank PP2×FSDP2/custom/relocated/full AC/1F1B/dynamic EMA的
  full四步及独立resume均通过：loss差异0，model/AdamW/scheduler/trainer/
  EMA逐位一致；与默认hook AC对照loss一致。2rank fine_grained16配置也通过
  独立dense输出、input/parameter gradients、momentum、reset及DCP参考。
- 完整CPU3170 passed、2 skipped（随后新增2分支回归包含于110定向结果）。
  日志/tmp/llmtuner-hook-ac-{recompute-fix,recompute-resume,fine-grained,units}.log；
  workdir /tmp/llmtuner-hook-ac-recompute-fix。全仓适用pre-commit、compileall、
  HEAD diff通过。比较文档同步full AC证据，前批待修复项现已关闭。
- 仍未验证selective/region checkpoint、外部checkpoint未使用本项目标记的
  多microbatch组合、compiled运行时、目标设备及全部调度。下一行动扩展
  selective AC同配置并核对多本地stage，保持完整范围及全局未完成状态。

Selective AC与多本地stage relocated FSDP恢复验证：

- 接续确认失败修复，运行真实4rank PP2×FSDP2/custom/relocated/dynamic EMA
  两组独立full/resume：1F1B/selective AC，Interleaved1F1B/full AC。后者
  六decoder层、每rank两个本地stage。扩展集成断言遍历全部model_parts，
  各part均验证custom root、四context及复制后ancestor hook归属。
- 两组2+2步恢复loss差异均0，完整model/AdamW/scheduler/trainer/EMA状态
  逐位一致。不是独立dense训练数值证明，独立参考仍由fine_grained16配置等
  已有脚本提供；本批新增证据专门覆盖实际装配/调度/恢复组合。
- 日志/tmp/llmtuner-hook-{1F1B-selective,Interleaved1F1B-full}-{full,resume}.log。
  workdir分别为/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/
  llmtuner-hook-1F1B-qxm7roxl及llmtuner-hook-Interleaved1F1B-hjo1d5zm。
  脚本说明同步AC变量和多part断言；比较文档删除多本地stage未验证的过期描述。
- 仅集成断言与文档修改，复用完整CPU3170 passed、2 skipped和FSDP110项
  基线；适用pre-commit、compileall、HEAD diff通过。region/compiled组合、
  其他调度、目标设备和多Trainer同模型并发仍未证明；下一行动检查whole-model
  compile/custom hook路径的运行时契约，保持全局未完成。

Whole-model compiled custom hooks运行时恢复验证：

- checkpoint集成脚本增加LLMTUNER_TEST_COMPILE_BACKEND，真实TrainingConfig
  设置compile及CompileConfig backend。2rank PP1/FSDP2/custom relocated
  attention hooks、dynamic EMA、whole-model aot_eager的独立full/resume
  2+2步loss差异0，model/AdamW/scheduler/trainer/EMA逐位一致。
- 补充每part OptimizedModule及Dynamo unique_graphs非零断言，重新运行
  独立resume通过，避免仅凭wrapper或入口证明编译执行。日志
  /tmp/llmtuner-hook-compile-{full,resume,graphs}.log；workdir
  /var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-hook-compile-2nkmtn4o。
- 无生产行为修改，脚本/文档适用pre-commit、compileall、HEAD diff通过；
  复用CPU3170 passed、2 skipped及FSDP110项基线。本证据是相同配置恢复
  一致性，不证明Inductor内核、compile×AC、PP compiled调度及目标设备数值。
  下一行动核对compiled参数命名与跨compile配置checkpoint迁移契约；全局未完成。

Compile配置切换的checkpoint名称修复与未关闭数值差异：

- compiled→非compiled真实2rank恢复先报EMA missing key；canonical_fqn
  原先只移除AC wrapper，现也移除_orig_mod，optimizer/EMA使用稳定逻辑名。
  随后模型missing lm_head key暴露ModelCheckpointState仍使用compiled
  state_dict；该view现在解包实际OptimizedModule到原始模型保存/加载，保留
  EP ownership/cache逻辑。真实CPU模型双向加载及wrapper命名回归通过。
- 新compiled full→非compiled resume已完成加载与后续训练，loss差异0，
  但最终参数/optimizer/EMA逐位比较失败。未修改容差或断言，未宣称跨compile
  数值等价；下一行动量化差异并核对AOT backward与eager更新路径。缺键问题
  已解决，数值问题仍开放。集成reference仅规范化wrapper键名，张量严格比较。
- 新格式同compiled配置resume重新运行逐位通过。定向checkpoint/optimizer/
  utils336 passed，完整3174 passed、2 skipped；适用全仓pre-commit、
  compileall、HEAD diff通过。历史带_orig_mod的compiled checkpoint尚无自动
  迁移，不能宣称格式向后兼容；需要后续旧格式加载规划。
- 日志/tmp/llmtuner-cross-compile-before.log、/tmp/llmtuner-cross-final-
  {full,resume,compiled-resume,units}.log；workdir
  /var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-cross-compile-final-371n7mh4。
  全局目标仍未完成，保持跨执行模式差异和旧格式迁移为必需待办。

跨 compile 恢复差异诊断：

- 范围：`tests/integration_tests/pp_checkpoint_equivalence.py` 的最终模型、optimizer、EMA shard 比较；2 rank CPU/Gloo、custom FSDP、重定位 hook、dynamic EMA，从 compiled checkpoint 恢复为 eager。
- 修改：精确比较失败时输出 tensor 类型、布局、本地 metadata 或最大绝对/相对误差与不等元素数；保持全部精确相等断言。
- 验证：实际 `torchrun` resume 退出 1，确认最终参数存在数值差异；已输出参数样本最大绝对误差为 3.7252903e-09 至 7.4505806e-09，未触发布局或 dtype 差异诊断。日志 `/tmp/llmtuner-cross-diagnostics.log`。Ruff 检查通过。
- 未完成：尚未证明差异仅由执行模式的浮点舍入导致，跨模式恢复等价仍未通过；旧 `_orig_mod` checkpoint 映射仍待处理。
- 下一行动：比较恢复后首次更新前的模型、optimizer、EMA 状态，区分加载差异与后续 eager/compile 计算差异，再验证旧 checkpoint 键迁移。

跨 compile 恢复保存点状态复核：

- 范围：2 rank CPU/Gloo、custom FSDP、重定位 attention hook、dynamic EMA，compiled `aot_eager` full 后分别 eager/compiled resume。
- 修改：集成脚本在 step 2 checkpoint 保存点额外序列化独立的模型、optimizer、scheduler、EMA、trainer 与 token 计数快照；恢复后首次更新前递归精确比较。快照提取复用于最终状态；旧测试产物无快照时仍保留原有最终验收，有意 LR override 不执行本项完全一致比较。
- 验证：新 full 退出 0；eager resume 两 rank 的 pre-update failures 均为 0，但最终精确比较失败、退出 1；相同 compiled resume 的保存点和最终全部检查通过、退出 0，编译图非空断言保持启用。由此定位本用例差异产生于后续执行模式变化，而非保存点状态加载。
- 证据：`/tmp/llmtuner-cross-split-{full,resume,control}.log`；产物 `llmtuner-cross-split-y2ktdd0w`。本结果尚未证明跨模式计算误差满足独立数值参考，未放宽精确断言。
- 未完成与下一行动：核对跨 eager/compile 的独立计算参考与支持契约；继续实现旧 `_orig_mod` checkpoint 严格映射和冲突检测。全局目标未完成。

旧 compiled checkpoint 键迁移：

- 范围：DCP planner 对旧 `_orig_mod` model、optimizer state/param_groups 和 EMA 键的恢复。
- 修改：仅移除完整 `_orig_mod` 路径段，建立保存键到当前 canonical 目标的映射；tensor 保留 live destination 身份，byte metadata 通过原 dictionary path 写回。规范键与旧键映射冲突时拒绝加载，其他缺失键保持严格检查。
- 验证：checkpoint 单元测试 104 passed，新增实际 DCP tensor、LR 标量、EMA 迁移及名称冲突拒绝测试。真实旧 2 rank CPU/Gloo custom/relocated hook/aot_eager/dynamic EMA checkpoint 已恢复；原测试参考文件仍有旧名称导致首轮 keys 比较失败。复制参考文件并仅规范化键名、保留全部数值及原 checkpoint 后再次执行 resume，退出 0，最终所有精确检查通过。日志 `/tmp/llmtuner-legacy-reference-resume.log`，原始首次结果 `/tmp/llmtuner-legacy-compiled-resume.log`。
- 质量门禁：全量 CPU suite 3176 passed、2 skipped、217 warnings；全仓适用 pre-commit 首轮修复 1 项后重跑全部通过，排除受保护的 `examples/native_alltoall.py`；compileall 与 HEAD diff 空白检查通过。日志 `/tmp/llmtuner-legacy-all-units.log`。
- 限制：此次验证覆盖旧格式到当前 canonical 格式、相同 compiled 执行模式；跨模式最终数值差异仍未关闭。
- 下一行动：完成本批全量门禁；核对跨模式独立计算参考及余下并行验证缺口。

旧 compiled checkpoint flat planner 标量写回复核：

- 发现：关闭 `flatten_state_dict` 时，旧键迁移的 tensor 仍通过共享 destination 恢复，但 byte 标量写到旧键而未回写 canonical 目标；实际 DCP 回归复现 LR 保持 0.0 而非保存值 0.25。
- 修改：planner 使用独立工作字典并记录迁移目标；flat byte load 写回原 canonical dictionary。默认展开路径继续使用 native mappings。
- 验证：新增展开开/关两种实际 DCP 用例，覆盖多层 `_orig_mod`、LR 标量、tensor 与目标键集合；新增迁移后缺失 bias 严格拒绝测试。checkpoint suite 107 passed；Ruff/格式通过。真实旧 2 rank compiled checkpoint 恢复退出 0，loss 最大差异 0，全部最终精确比较通过（`/tmp/llmtuner-flat-planner-resume.log`）。全量 suite 3179 passed、2 skipped、214 warnings（`/tmp/llmtuner-flat-planner-units.log`）；适用全仓 pre-commit、compileall 与 HEAD diff 检查通过，排除受保护 `examples/native_alltoall.py`。
- 下一行动：完成门禁与真实多进程对照，再继续跨模式计算参考审查。全局目标未完成。

Custom FSDP compile 独立 dense 参考扩展（未收尾）：

- 修改：`fine_grained_fsdp_equivalence.py` 增加 `--compile-backend aot_eager`，按生产顺序 compile 后应用 FSDP；optimizer 与 dense 参考参数使用 canonical 名称。每个配置独立 reset Dynamo/cache/counters，两 rank 都须生成编译图；原输出、梯度、momentum、更新参数、异常 reset 和 DCP 精确恢复检查保持。
- 发现与证据：首轮累计计数的 16 项数值比较通过，但不构成每项 compile 证据。加强为逐配置图断言后，前两个无 recompute 用例通过，首次 recompute 用例 unique_graphs=0，退出 1；日志 `/tmp/llmtuner-compiled-dense-nonvacuous.log`。graph-break 日志显示 native FSDP 状态查询和 checkpoint 高阶图捕获相关 eager fallback；移除 trace 中测试观察并未解决，已撤销该尝试。
- 未完成：当前新增 compile 模式在 recompute 组合失败；不能宣称 16 项编译矩阵通过，也未解决 HF 跨模式恢复的数值差异。
- 下一行动：核对脚本内直接 checkpoint 与生产 activation-checkpoint wrapper 路径的差异，确定应修复的生产机制或测试装配；保留逐配置图断言后重新取得结果。

Compile × AC context 类型契约修复（矩阵仍有失败）：

- 根因：生产 `checkpoint_contexts` 将 native SAC recompute mode 包在普通 context manager 内，不满足 torch.compile 要求两个 context 均为 TorchDispatchMode 的断言；实际生产 full wrapper 复现。
- 修改：重计算 context 继承 native SAC mode，保留其 storage、mutation policy、dispatch 和每次进入的 counter reset；在 native mode 进入/退出外围管理重计算 marker 与 MoE mask，进入异常时也恢复 marker。集成脚本编译/重计算组合改用生产 `wrap_full`，按 AC→compile→FSDP 装配，hook 指向对应 wrapper；owner 检查解开 AC wrapper 后仍要求同一实际 block。
- 验证：新增实际 aot_eager full-wrapper 输出、输入/参数梯度及 mode 类型回归，AC suite 80 passed；Ruff/格式通过。两 rank dense-reference 编译矩阵前 9 项通过，含 recompute；随后 input_grad=True、multiple_groups=True 配置遇到 Dynamo `dictionary changed size during iteration`，脚本退出 1。日志 `/tmp/llmtuner-ac-dispatch-dense.log`。
- 未完成：本批全部矩阵、全量门禁及多参数组 Dynamo 异常尚未收尾；当前证据不能证明编译矩阵全部通过。
- 下一行动：按报错栈定位 Dynamo 字典变化的触发状态与生产私有 API 调用，修复后保留逐配置图断言重新验收。

Compile × AC 独立 dense 矩阵收尾复核：

- 定位：上批 Dynamo dictionary 异常发生在测试 Block.forward 的 `_get_fsdp_state()` 内部弱引用 registry 查询；不是数值计算路径。trace 中排除这项 Python 内部观察，eager 观察及每次 backward 后 assert_clean 保留，逐配置与逐 rank 图断言不变。
- 验证：实际 2 rank CPU/Gloo/aot_eager/custom FSDP 的 16 种 input_grad×reshard×recompute×multiple_groups 配置全部通过、退出 0。重计算使用生产 full AC wrapper；独立 eager dense 参考覆盖输出、输入/参数梯度、momentum 和三次参数更新，原参考容差保持；DCP 新运行时恢复比较仍精确相等。每配置独立 reset Dynamo/counters，两 rank 非空编译图。日志 `/tmp/llmtuner-compile-ac-clean.log`。
- 修改：bare full-wrapper 单测也独立清空 counters，避免之前用例生成的图使验证空转。生产 context 类型修复后全量 suite 3180 passed、2 skipped、218 warnings；两 rank `--outstanding-forwards` 额外 16 配置全部通过、退出 0，含先两次 forward 再逆序 backward 和 DCP 恢复（`/tmp/llmtuner-compile-ac-outstanding.log`）。适用全仓 pre-commit、compileall 与 HEAD diff 检查通过，排除受保护 `examples/native_alltoall.py`。
- 限制：证据为 CPU float32/aot_eager 的小型独立 dense 模型，不证明 HF 跨模式训练精确相等、Inductor、CUDA/NPU 或目标设备重叠性能。
- 下一行动：收尾 full suite、多 outstanding forward 和质量门禁，然后继续训练/并行剩余缺口。

生产 HF full AC × custom FSDP × compile 恢复验证：

- 范围：真实 HF Trainer、2 rank CPU/Gloo、重定位 q/k/v/o attention hook、custom FSDP、full AC、aot_eager、dynamic EMA。针对刚修复的生产 SAC dispatch context 类型契约，覆盖真实层装配与完整训练更新。
- 证据：`pp_checkpoint_equivalence.py` full/resume 均退出 0；两 rank 的保存点快照加载后精确一致，最终模型、optimizer、scheduler、EMA、trainer 与 token 计数精确一致，loss 最大差异 0，编译图非空断言通过。日志 `/tmp/llmtuner-hf-fullac-compile-{full,resume}.log`。
- MoE/EP 补充：Qwen3MoE、EP2、同一 full AC/custom FSDP/aot_eager/dynamic EMA 配置的 full/resume 也均退出 0；两 rank pre-update failures=0，最终精确比较全部通过，loss 最大差异 0。日志 `/tmp/llmtuner-moe-fullac-compile-{full,resume}.log`。
- 限制：当前为相同 compile 模式恢复，不构成独立 dense MoE 数值参考；尚未证明跨模式精确恢复、目标设备或 Inductor。
- 下一行动：取得 MoE/EP 编译重计算的 full/resume 最终结果，核对尚未覆盖的状态统计和恢复契约。

PP × EP × full AC × compile 独立梯度参考：

- 修改：`pp_ep_equivalence.py` 增加 `--compile-backend aot_eager`，每 rank 断言本地 model parts 为 OptimizedModule 且生成编译图；初始参考权重按 canonical 名称收集。独立参考继续使用未编译、无 AC、单进程 EP1 模型，从实际 PP 初始状态重建，不复用并行 forward。
- 验证：4 rank CPU/Gloo、PP2×EP2、full AC、aot_eager、router aux coefficient=0.3，四步 loss、clip 后梯度、AdamW 更新参数全部通过原有 1e-4 容差；实际专家 shard 与 PP 分割断言保持，退出 0。日志 `/tmp/llmtuner-pp-ep-compile-reference.log`。
- selective AC：同一 4 rank/aux=0.3/aot_eager 配置也全部通过、退出 0，四步 loss、梯度与参数保持原 1e-4 容差；日志 `/tmp/llmtuner-pp-ep-selective-compile-reference.log`。适用全仓 pre-commit、compileall、HEAD diff 检查通过，排除受保护 `examples/native_alltoall.py`。本批只扩展显式 torchrun 脚本；最近生产代码状态的 full CPU suite 为 3180 passed、2 skipped。
- 限制：本项不覆盖 CUDA/NPU、Inductor 或所有并行组合；HF 跨模式精确恢复差异仍单列待办。
- 下一行动：收尾 selective AC 与适用质量门禁，更新能力矩阵。

Qwen3 compile 与原生 HF 独立轨迹参考：

- 范围：`test_model_compile.py` 新增四步 Qwen3 CPU/aot_eager 训练比较；参考直接调用原生 HF CausalLM.forward，不走 llmtuner 的 flatten forward，也不应用 compile/AC/并行变换。
- 验证：同初值与相同输入，逐步比较 logits、cross entropy、clip 后全部参数梯度和 AdamW 更新参数，使用 torch.testing 的既有 float32 默认容差；独立 reset Dynamo/counters 并确认生成图。新增用例通过，日志 `/tmp/llmtuner-native-hf-compile.log`。
- 限制：该参考确认本用例在 float32 数值容差内的计算正确性，不证明跨模式最终状态 bitwise 一致，也不能单独解释原跨模式恢复中的每个差异；恢复脚本原精确断言未改动。全量 suite 3181 passed、2 skipped、225 warnings（`/tmp/llmtuner-native-hf-all-units.log`），适用全仓 pre-commit、compileall 与 HEAD diff 检查通过，排除受保护 `examples/native_alltoall.py`。
- 下一行动：收尾质量门禁，并继续尚未解决的全局审查与目标设备验证缺口。

Checkpoint load planner 职责归位：

- 结构：将包含 loader/RNG/compiled 兼容规则的 planner 从 DCP manager 移至 `components/checkpointer/load_planner.py`，统一命名 `CheckpointLoadPlanner`；manager 仅负责 I/O 与生命周期编排，定义模块直接导入，旧类名已删除且不留 alias。
- 消费者：同步 manager、checkpoint/RNG 测试及文档；保留原有严格缺失键、冲突和标量写回契约。逐文件清单更新为 162 源文件。
- 验证：checkpointer suite 180 passed，实际 DCP 旧格式迁移等回归保持；Ruff 与格式通过，旧名搜索在 llmtuner/tests/docs 中无残留。全量 suite 3181 passed、2 skipped、228 warnings；适用全仓 pre-commit、compileall 与 HEAD diff 检查通过，排除受保护 `examples/native_alltoall.py`。源码与清单 162 文件集合完全一致，无遗漏、重复或失效路径。
- 下一行动：完成门禁，继续训练服务和进程级状态所有权审查。

Allocator history 停止失败的资源所有权修复：

- 根因：MemoryProfiler.close 在调用 backend stop 前清除 active 标记；Profiler.__exit__ 同时提前清空 owner。若 backend stop 抛异常，allocator 可能仍在记录，但后续 close 不再尝试停止。
- 修改：仅 stop 成功后清除 active 标记；外层 shutdown 失败时保留 memory_profiler 引用，允许显式重试退出，成功后仍幂等。原训练/OOM 异常继续保留，不被清理错误替代。
- 验证：更新既有 OOM×shutdown failure 四分支回归，模拟首次 stop 失败、第二次成功，断言 owner 和 active 标记保留、重试调用发生且最终重复 close 不再调用 backend。Profiler suite 58 passed；Ruff/格式通过。仅为 CPU 故障注入，不代表真实设备 allocator 验证。全量 suite 3181 passed、2 skipped、210 warnings（`/tmp/llmtuner-profiler-retry-all-units.log`）；适用全仓 pre-commit、compileall 与 HEAD diff 检查通过，排除受保护 `examples/native_alltoall.py`。
- 下一行动：完成门禁，继续进程级资源与并行能力剩余事项。

HF 模型构建非对称异常协调：

- 根因：HF config/模型构建是本地阶段，单 rank 抛异常后，成功 rank 原先直接进入 TP/EP/FSDP 等变换，可能执行其他 rank 永远不会参与的 collective。
- 修改：在本地 `_build_model` 后、任何模型并行变换前，使用现有通信设备和 world MIN reduction 协调成功状态；失败 rank 保留原 Exception，成功 peer 抛明确的另一 rank 构建失败错误。单进程保留原异常。范围不包含建组失败、挂起构造器、进程被杀或后续 collective 内异常。
- 回归：新增真实两 rank `trainer_initialization_coordination.py`，分别在 rank0 的 config factory、rank1 的 HF model constructor 注入异常，所有 rank 均断言未进入 `_apply_parallelism`，退出后 barrier；最后用同一 group 新建真实 Trainer 并更新。
- 验证：Trainer unit suite 111 passed；两种 rank-local 异常协调已在真实 Gloo 通过。首轮 standalone 因 TCPStore 地址失败；IPv4 重跑通过协调后，在健康更新被本机旧 Inductor PCH mtime 错误阻断。新独立 cache 仍引用同一全局 PCH；按 PyTorch 实际环境开关 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0` 关闭 header 预编译优化后重跑，保留全部模型/通信/更新断言，完整脚本退出 0，两个 rank 新建 Trainer 更新成功（`/tmp/llmtuner-model-build-no-pch.log`）。全量 suite 3181 passed、2 skipped、225 warnings；适用全仓 pre-commit、compileall 与 HEAD diff 检查通过，排除受保护 `examples/native_alltoall.py`。
- 下一行动：取得完整退出结果与质量门禁，再覆盖剩余初始化阶段的失败边界。

Rank-local 初始化失败边界统一与 dataloader 协调：

- 根因：某 rank 的 dataloader 构建失败后，peer 原本继续进入 checkpointer，异步配置可在内部创建新 group，造成参与者不匹配。
- 修改：`accelerator/collectives.py::synchronize_local_errors` 提供明确的 group-local phase 契约，复用现有设备与 MIN reduction；assembly 的模型和 dataloader 本地阶段共用此边界。保留原异常 identity，peer 获得阶段错误，单 rank 不通信。禁止把带内部 collective 的阶段盲目放入该边界；挂起/被杀进程仍不在恢复范围。
- 回归：集成脚本重命名为 `trainer_initialization_coordination.py`，同步引用并删除旧路径；新增 rank0 loader 构建失败、rank1 持有已构造 loader 的场景，两 rank 均未调用 checkpointer builder，peer loader 确认关闭。随后真实 Trainer 更新通过。原模型 config/constructor 非对称故障也保持通过，脚本退出 0（`/tmp/llmtuner-loader-build-coordination.log`）；本机 IPv4 与关闭 PCH 缓存优化条件同上一批。
- 验证：新增单 rank 成功/原异常 identity CPU 回归；collectives+Trainer 专项 116 passed。全量 suite 3183 passed、2 skipped、233 warnings（`/tmp/llmtuner-local-phase-all-units.log`）；适用全仓 pre-commit、compileall、HEAD diff 检查通过，排除受保护 `examples/native_alltoall.py`。
- 下一行动：完成门禁，再审查 optimizer 内部 PP collective 前的局部规划错误。

Optimizer PP 局部规划/初始化异常协调（world 边界待收尾）：

- 根因：参数组 canonical FQN 冲突在 PP rule-match all_reduce 前抛出，peer 仍进入 collective；重复参数和实际 optimizer 构造失败也只影响本地 rank。
- 修改：使用 synchronize_local_errors 分别包围本地参数规划与 rule-match 后的本地校验/构造，PP match collective 保持原组。新增 initialization_group，Trainer 显式传入 world，使错误传播到其他 DP 子组；独立 container 默认使用 PP 组，非分布式不新增通信。scheduler/EMA/hooks 本地装配另用 world 阶段边界。
- 验证：实际两 rank PP 脚本新增 canonical 名称冲突、重复本地 parts 两项非对称故障，均协调退出，后续 barrier 通过；既有 stage-specific AdamW 独立参数更新与未使用规则拒绝保持，退出 0。首轮测试期望错误文本不符已按实际明确错误修正。日志 `/tmp/llmtuner-pp-optimizer-coordination-final.log`。
- 未完成：新增显式 world 组以及 PP 子组之外错误传播尚须真实 4 rank 验证；当前世界范围更新的专项 suite 正在取得结果，全量门禁尚未执行。不得将两 rank 证据表述为 DP×PP 已通过。
- 下一行动：运行 PP2×DP2 非对称构造故障验证，随后完成全部门禁。

Optimizer PP2 × DP2 world 异常传播验收：

- 回归：扩展 `pp_optimizer_groups.py` 同时支持 2 rank 与 4 rank；4 rank 建立两个 PP2 子组，rule matching 使用各自 PP 组，initialization_group 使用 world。仅 global rank0 的 canonical 冲突和重复 parts 错误，其他 3 rank 均在对应阶段收到错误，随后 world barrier 通过。
- 验证：两种 world size 的独立 stage-specific AdamW 参数更新及无效规则拒绝保持通过，均退出 0，日志 `/tmp/llmtuner-pp-optimizer-world-{2,4}.log`。单 PP 子组缺失规则故障也通过，已匹配的另一个 DP 副本收到初始化错误，全部 rank 的后续 world barrier 通过、脚本退出 0（`/tmp/llmtuner-pp-optimizer-subgroup.log`）。
- 质量门禁：full suite 3183 passed、2 skipped、220 warnings（`/tmp/llmtuner-optimizer-world-all-units.log`）；pre-commit 首轮 line-length 经 formatter 修复后重跑全部通过，compileall 与 HEAD diff 检查通过，排除受保护 `examples/native_alltoall.py`。设备 collective 和挂起/被杀 rank 不由该边界恢复。
- 下一行动：取得子组与全部门禁最终结果，然后核对剩余全局状态/迁移能力待办。

Metrics 初始化非对称异常协调：

- 根因：模型、optimizer 和 loader 初始化已协调后，某 rank 的日志后端构造失败仍会使其他 rank 返回健康 Trainer，随后进入不匹配的训练 collective。
- 修改：metrics/FLOPs 的本地构造阶段复用 world 错误边界；成功 peer 的 metrics owner 挂在 Trainer 上，构造失败统一清理并关闭。仅用于本地构造，不包裹 backend 内未知 collective 或挂起调用。
- 验证：两 rank CPU/Gloo，rank1 MetricsProcessor factory 抛原 ValueError，rank0 返回带 close 的 metrics owner；两 rank 均构造失败，peer owner 确认关闭，随后 barrier 和真实新 Trainer 更新通过，退出 0。模型、loader 既有故障场景保持；日志 `/tmp/llmtuner-metrics-build-coordination.log`。本项为真实 c10d 通信与注入 metrics backend 故障，不代表外部 WandB 服务验证。
- 质量门禁：全量 suite 3183 passed、2 skipped、219 warnings（`/tmp/llmtuner-metrics-build-all-units.log`）；适用全仓 pre-commit、compileall、HEAD diff 检查通过，排除受保护 `examples/native_alltoall.py`。
- 下一行动：继续核对未闭合的 checkpointer 内部建组前失败边界及全局待办。

Checkpoint 建组前本地预检异常协调：

- 根因：适配器缺失、存储解析和异步模式校验原先只在本地失败，成功 peer 仍可能进入模型 EP 状态收集或异步 Gloo 建组。
- 修改：DistributedCheckpointManager 先在 world 本地错误边界完成有效目录、HF adapter、async mode 和 memory:// 策略检查，再构造基类模型状态并建组；禁用 checkpoint 保持无存储和无通信。ModelCheckpointState 构造含 EP gather，明确留在本地协调区间之外。
- 回归：真实两 rank Gloo，分别让 rank0/rank1 缺 HF adapter，失败 rank 保留 ValueError，peer 获得 Checkpoint preflight 错误；所有 rank 断言未调用 new_group，随后 barrier 和同组新建真实 Trainer 更新通过，退出 0。既有模型、loader、metrics 非对称错误场景保持通过，日志 `/tmp/llmtuner-checkpoint-preflight-gloo.log`。
- 验证：checkpoint 专项 107 passed；适用全仓 pre-commit、compileall、HEAD diff 检查通过，排除受保护 examples/native_alltoall.py。全量 CPU suite 3183 passed、2 skipped、226 warnings，日志 `/tmp/llmtuner-checkpoint-preflight-all-units.log`。
- 未完成：模型 state_dict hook/EP gather 前的本地准备及建组后线程启动失败尚未整体协调；本批不宣称处理 collective 内失败、挂起或进程被杀。下一行动：继续审查模型状态收集前的本地准备及其协调边界。

Checkpoint 建组后保留线程启动异常协调：

- 根因：异步 Gloo group 创建后，单 rank 的 purge thread 启动失败只清理本地资源；成功 peer 原先返回健康 manager，可能继续进入不匹配的后续 collective。
- 修改：队列、线程构造和启动作为本地 Checkpoint services construction 阶段协调；任一 Exception 令全部 rank 退出构造，双方均调用现有 _close 释放线程和异步组。原始本地异常保留，peer 收到阶段错误；BaseException 仍做本地清理，不宣称可跨 rank 恢复中断或进程死亡。
- 验证：真实两 rank Gloo 分别注入 rank0/rank1 OSError，确认两边 async group 均已注销、成功 peer 的线程已停止，world barrier 及同组全新 Trainer 更新通过，脚本退出 0（`/tmp/llmtuner-retention-coordination-gloo.log`）；checkpoint 专项 107 passed。
- 门禁：适用全仓 pre-commit、compileall、HEAD diff 检查通过，排除受保护 examples/native_alltoall.py；全量 CPU suite 3183 passed、2 skipped、218 warnings（`/tmp/llmtuner-retention-coordination-all-units.log`）。
- 下一行动：模型状态初始化中 expert_shard_map 和本地 state_dict 可能在 EP gather 前失败；检查其协调边界，须考虑 state_dict hook 自身包含 collective 的情况，不能直接包装为纯本地区间。

EP checkpoint 状态键通信顺序修复：

- 根因：gather_expert_state 原先按 state_dict 插入顺序发起 all_gather；各 rank 同一键集合但顺序不同，同形状专家张量可能交叉拼接且不报错。迭代器还可能在前面已通信后才发现后续形状错误。
- 修改：先完整解析并校验本地 expert entries，再按状态键排序执行通信；输出映射保留原键和值语义。只保证相同键集合的稳定顺序，不宣称跨 rank 键缺失、形状不一致或 state_dict hook 内异常已协调。
- 验证：真实两 rank Gloo 故意反向插入 first/second 同形状张量，逐键与独立显式参考零容差比较通过；随后同组 Trainer 更新通过，脚本退出 0（`/tmp/llmtuner-ep-checkpoint-order-gloo.log`）。CPU 专项 20 passed，新增后续键形状错误时首次通信尚未发生的断言。
- 门禁：适用全仓 pre-commit、compileall、HEAD diff 通过，新增单测再次通过 pre-commit，排除受保护 examples/native_alltoall.py。全量 CPU suite 3183 passed、2 skipped、207 warnings（`/tmp/llmtuner-ep-checkpoint-order-all-units.log`）；新增单测在该轮收集后添加，已单独通过。
- 下一行动：继续处理模型状态构造的本地元数据检查与 EP gather 前错误协调，明确 PP/EP 子组和可能含 collective 的 hook 边界。

EP checkpoint 通信前元数据与局部错误协调：

- 根因：固定键顺序后，单 rank 的空/缺失状态、dtype/shape 差异或参数索引异常仍能让 peer 进入不匹配的 all_gather；原先本地 ValueError 不能让其他 rank 退出。
- 修改：gather_expert_state 先捕获纯本地参数索引、entry 解析与 manifest 构造错误，再在每个已知 EP 子组交换成功状态和 manifest（键、逻辑 shape、dtype、layout、DTensor 类型）。不一致时全部子组成员跳过 tensor gather 并报错，本地原异常保留。多个组按成员列表排序；其他健康组完成自身协议后再抛已记录错误，避免仅参与健康组的 peer 被遗留。实际 tensor full_tensor/gather 留在本地准备之外。
- 回归：新增 `tests/integration_tests/ep_checkpoint_coordination.py`，真实 2/4 rank Gloo，分别在 EP local rank0/1 注入缺失 EMA 键、空 state、dtype 差异、合法但跨 rank 不同 shape、本地坏 shape、模型 parts canonical 参数冲突；全部断言未调用 tensor gather，随后 world barrier 和同组健康快照零容差通过。4 rank 使用两个 EP 子组及不同 stage 键；额外真实 DTensor 在 efsdp2×EP2 上 full_tensor 后拼接，与显式参考零容差一致。日志 `/tmp/llmtuner-ep-checkpoint-preflight-{2,4}.log`、`/tmp/llmtuner-ep-checkpoint-preflight-dtensor.log`，均退出 0。
- 验证：相关 CPU 39 passed；全量 3184 passed、2 skipped、233 warnings（`/tmp/llmtuner-ep-checkpoint-preflight-all-units.log`）；适用全仓 pre-commit、compileall、HEAD diff 通过，排除受保护 examples/native_alltoall.py。新增脚本后续 DTensor 分支再按受影响文件跑门禁。
- 限制：依赖各成员已有一致且非空的 shard_map/组归属；不协调调用本函数前的 state_dict hook、expert_shard_map 失败，或 full_tensor/collective 内错误，不覆盖 CUDA/HCCL 设备证据。metadata exchange 增加每 EP 组每次 state export 的通信成本。
- 下一行动：继续检查 ModelCheckpointState 构造的本地模型类型/分片映射准备及 state_dict hook 契约，处理当前函数之前的失败传播。

Checkpoint 模型准备阶段的 world 异常协调：

- 根因：ModelCheckpointState 在首次 EP 状态收集前执行模型类型检查和 expert_shard_map；非法 model part 或共享参数的冲突 EP group 只在单 rank 抛错，peer 继续进入状态通信。
- 修改：ModelCheckpointState 增加显式 initialization_group，仅把本地模型集合复制、类型检查、compile 解包和分片映射构造放进 Checkpoint model preparation 边界；BaseCheckpointManager 在已初始化的 distributed 环境传入 world。直接构造状态 wrapper 默认不新增 world collective，保留 rank-local dense 使用方式。state_dict hook 和 EP gather 明确位于边界外。
- 回归：真实两 rank，分别在 rank0/rank1 注入非法 part 与共享 EP 参数组冲突，健康 peer 的模型标记真实 world EP group；所有 rank 在建 async group 前退出并通过 barrier，同组真实新 Trainer 更新通过，退出 0（`/tmp/llmtuner-model-preparation-ep-gloo.log`）。初轮无健康 EP 标记的协调测试也通过（`/tmp/llmtuner-model-preparation-gloo.log`）。
- 门禁：checkpoint CPU suite 180 passed；全量 3184 passed、2 skipped、215 warnings（`/tmp/llmtuner-model-preparation-all-units.log`）；适用全仓 pre-commit、compileall、HEAD diff 通过，排除受保护 examples/native_alltoall.py；强化健康 EP 标记后的脚本再跑对应门禁。
- 限制与下一行动：本阶段不处理 model.state_dict hook 内异常、parts 输出键冲突、通信内故障或进程终止。继续检查可在 hook 前完成的 canonical FQN/参数身份冲突校验与已知 hook 的通信契约，不能把任意 hook 假定为本地操作。

Checkpoint 首次模型状态合并冲突协调：

- 审查：QKV._split_qkv_on_save 对 DTensor 执行 redistribute；FeedForward._split_w13_on_save 的 DTensor reshape/layout 运算也不能一概当作纯本地。因此不把任意 model.state_dict hook 纳入 synchronize_local_errors。
- 根因：首次 state_dict 导出后，某 rank 的 parts 输出键重名抛 ValueError，成功 peer 随后进入 EP gather 或异步建组。只在 hook 前检查参数名不能覆盖 hook 生成的逻辑名称，还可能误拒绝 hook 合法改名。
- 修改：先取得各 part 的导出，再在纯本地合并阶段用显式 initialization_group 协调重复键错误，之后才调用 gather_expert_state。manager 首次初始化使用 world；常规 state_dict/save/restore 的刷新不新增 world collective，直接 wrapper 默认保持本地契约。
- 回归：两 rank 分别在 rank0/rank1 注入原始键重叠、hook 把 distinct other 参数导出为 weight 的逻辑键重叠；健康 peer 持有真实 EP group。失败 rank 保留 overlap ValueError、peer 收到 Checkpoint model state merge 错误，所有 rank 未建 async group，后续 barrier 与同组新 Trainer 更新通过，脚本退出 0（`/tmp/llmtuner-state-merge-gloo.log`）。
- 验证：checkpoint CPU 180 passed；适用全仓 pre-commit、compileall、HEAD diff 通过，排除受保护 examples/native_alltoall.py；全量 CPU suite 3184 passed、2 skipped、209 warnings（`/tmp/llmtuner-state-merge-all-units.log`）。
- 限制与下一行动：未协调 hook 内抛错、通信失败和后续保存刷新中的局部失败；不能假定任意外部 hook 无 collective。继续核对全局剩余缺口，优先处理实际暴露的保存/恢复失败协议，避免以该初始化阶段证据宣称全路径异常可恢复。

DCP writer 本地构造失败的保存协议协调：

- 根因：GenerationFileSystemWriter 在进入 DCP save/async_save 前本地构造，路径或文件系统初始化失败可能仅影响一个 rank，而成功 peer 进入后续保存 collective。
- 修改：writer 构造使用 Checkpoint writer construction world 本地错误边界；原失败 rank 保留异常，peer 在 DCP 启动前收到阶段错误。仅包裹无 collective 的 writer/path setup，不包裹 DCP 内部通信或任意 state_dict hook。
- 回归：新增 `tests/integration_tests/checkpoint_save_coordination.py`，真实两 rank CPU/Gloo，同步和 threaded async 各注入 rank0/rank1 writer OSError。先实际保存 step1，再失败 step2；确认没有遗留 save_future，旧 step1 仍可实际 DCP 零容差加载，随后同一步 step2 重试实际保存/加载并与当前模型零容差一致，所有场景通过、退出 0（`/tmp/llmtuner-writer-coordination-gloo-retry.log`）。首轮测试误设 keep_latest_k=1，被配置拒绝；修正为现有合法契约 2，保留原失败日志 `/tmp/llmtuner-writer-coordination-gloo.log`。
- 门禁：全量 CPU 3184 passed、2 skipped、207 warnings（`/tmp/llmtuner-writer-coordination-all-units.log`）；适用全仓 pre-commit、compileall、HEAD diff 通过，排除受保护 examples/native_alltoall.py；测试配置修正后按受影响文件重跑门禁。
- 限制与下一行动：本项不覆盖实际远端服务、pinned staging/PROCESS upload、DCP collective 内错误或进程终止。继续审查 DefaultStager 构造和首次 async_save 启动前的局部异常，核实 PyTorch 内部 staging/collective 契约再选择协调位置。

Pinned checkpoint stager 构造失败与不对称重试协调：

- 源码核实：PyTorch 2.14 DefaultStager.__init__ 只初始化本地 StateDictStager、executor 和设备 stream；async_save 的 Stateful 转换可能收集 EP 状态，PROCESS executor 首次启动的 _ProcessGroupInitInfo 包含 broadcast，不能把整个 async_save 包成纯本地区间。
- 根因：某 rank 的 DefaultStager 构造失败时，成功 peer 可能继续进入首次 PROCESS 建组；若重试时仅 stager is None 的 rank 进入新错误边界，又会产生不匹配的 collective。
- 修改：pinned 模式的所有 rank 均进入 Checkpoint stager construction 本地协调阶段，内部仅在 stager 缺失时构造。成功 peer 在失败后保留其完整 stager，重试时复用，close 仍由 manager 释放；边界结束前尚未启动上传。
- 回归：扩展 `tests/integration_tests/checkpoint_save_coordination.py`，两 rank 分别注入 stager OSError，断言两边 async_save 均未调用、无 staging/upload future、旧 step1 实际零容差恢复。重试使用真实 DefaultStager（CPU/shared-memory、同步 staging）和真实 PROCESS 上传，成功 peer 的 stager 身份保持，step2 零容差加载、close 后 stager 已释放。保留同步/thread writer 故障场景，全脚本退出 0（`/tmp/llmtuner-stager-coordination-gloo.log`）。CPU 适配明确关闭 pinning/nonblocking，不代表目标设备证据。
- 门禁：full suite 3184 passed、2 skipped、216 warnings（`/tmp/llmtuner-stager-coordination-all-units.log`）；适用全仓 pre-commit、compileall、HEAD diff 通过，排除受保护 examples/native_alltoall.py。
- 限制与下一行动：不覆盖 upstream ctor 尚未返回的部分资源、stage 异步失败、首次 PROCESS 内部 broadcast/子进程启动失败或设备 stream 错误；继续核对异步 staging future 的完成/失败传播契约及既有上传清理规则。

异步 staging 失败协议、独立通信组与恢复验收：

- 根因：PyTorch PROCESS executor 直接 source_future.result()，单 rank staging 失败时其 peer 仍可能进入子进程保存通信。另 PyTorch 2.14 async_save 对调用返回前已完成的 staging future 不检查异常，会错误标记 staging_completion 成功。
- 修改：pinned/PROCESS manager 增加专用 Gloo staging_pg；CoordinatedStager 在本地快照 ready/failure 后执行该组状态协调，再放行上传，避免与训练及首次 PROCESS broadcast 共用组。直接 stage 抛错转换为失败 future，与异步失败统一处理；manager 使用自身 completion future，保留本地异常与 peer 阶段错误。底层 stager 由 manager 关闭；部分建组失败清理已拥有的组，常规关闭也释放 staging_pg。
- 回归：真实两 rank Gloo，分别 rank0/rank1 的直接 stage OSError 和真实 executor future 异常；两个等待点均报错，没有 step2 metadata，旧 step1 实际零容差加载，同一步重试真实 shared CPU staging/PROCESS 上传与恢复通过。沿用 writer/stager 构造失败场景，脚本退出 0（`/tmp/llmtuner-staging-failure-gloo.log`）。新增单进程真实 group 测试确认第二个组创建失败时第一个已注销，checkpoint suite 181 passed。
- 正常路径：延迟实际 staging 直到 Trainer wait，真实 CPU shared-memory cache 复用、PROCESS 上传、custom FSDP 与 dynamic EMA，full/resume 两个独立 torchrun 均退出 0；恢复前 split state、最终模型/optimizer/scheduler/EMA/trainer 状态检查全通过，loss max abs diff 0。日志 `/tmp/llmtuner-coordinated-staging-{full,resume}.log`，产物 `/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-coordinated-staging-iaymveak`。
- 门禁：最终 full suite 3185 passed、2 skipped、228 warnings（`/tmp/llmtuner-staging-failure-final-all-units.log`）；适用全仓 pre-commit、compileall、HEAD diff 通过，排除受保护 examples/native_alltoall.py。
- 限制与下一行动：仅 CPU/shared-memory，关闭设备 pinning/nonblocking 的测试适配不构成目标设备证据；不恢复 collective 内失败、进程终止或 PROCESS 创建失败。新增每个 pinned manager 的 staging 组和每次快照的成功状态通信开销。下一步核对并行/恢复矩阵剩余缺口与 CPU 可执行项，继续推进全局审查。

EP/efsdp × full AC/compile × PROCESS 延迟快照组合验收：

- 目的：前批 staging 协调在 dense custom FSDP 上已通过；进一步覆盖模型、AdamW 和 EMA 的 EP 全量状态收集，及真实专家 DTensor 在 efsdp 子组的物化，防止新增 staging 组与该通信顺序冲突。
- 实际执行：`async_snapshot_equivalence.py full/resume WORKDIR`，分别两 rank 和四 rank、独立 full/resume 进程；两者均设置 `LLMTUNER_TEST_PP_SIZE=1`、`LLMTUNER_TEST_MOE=1`、`LLMTUNER_TEST_EP_SIZE=2`、`LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom`、`LLMTUNER_TEST_AC=full`、`LLMTUNER_TEST_COMPILE_BACKEND=aot_eager`、`LLMTUNER_TEST_EMA=dynamic`；四 rank 另设 `LLMTUNER_TEST_WORLD_SIZE=4`。IPv4 rendezvous、GLOO_SOCKET_IFNAME=lo0、关闭旧 PCH 缓存；script 内启用 async_with_pinned_mem，但 CPU stager 适配关闭实际 pinning/nonblocking。
- 结果：2rank efsdp1×EP2、4rank efsdp2×EP2 均完整训练4步、step2新运行时恢复再训练2步，所有进程退出0。现有脚本断言实际MoE expert/router DTensor mesh、OptimizedModule及非空compile图；每rank root buffer共享CPU缓存第一次cache_reused=False、第二次True。更新前split state与最终模型、AdamW、scheduler、EMA、trainer计数比较全通过，loss最大误差0，无放宽断言。
- 证据：`/tmp/llmtuner-ep-coordinated-staging-{full,resume}.log`、`/tmp/llmtuner-efsdp-ep-coordinated-staging-{full,resume}.log`；产物分别 `/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-ep-coordinated-staging-xs1_691m` 与 `/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-efsdp-ep-coordinated-staging-_atslyl5`。已更新并行验证矩阵。
- 限制：相同aot_eager模式CPU/Gloo数值证据，不涵盖跨eager/compile逐位恢复、Inductor、CUDA/NPU、设备拷贝/性能或并发manager。本批仅补足运行时证据与文档，没有额外行为修改；按文档范围跑质量检查。
- 下一行动：返回全局矩阵剩余可执行项，继续审查独立参考下的训练数值及已知跨eager/compile恢复差异，目标设备缺口仍保持未验证。

跨 eager/compile 恢复的首次更新梯度定位：

- 修改：新增 `tests/integration_tests/compile_trajectory_diagnostics.py`，仅观察真实 Trainer.optimizer.step 前后的本地参数及 clip 后梯度，复用完整 full/resume harness，保留其所有最终严格断言。每次更新写独立 trace，不改训练语义或容差。
- 执行：真实2rank CPU/Gloo、custom FSDP、重定位attention hooks、dynamic EMA；aot_eager full、同编译resume、eager resume 三个独立进程。前两者退出0，eager最终严格比较退出1，未将预期失败改为通过。日志 `/tmp/llmtuner-cross-update-trace-{full,control,eager}.log`，产物 `/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-cross-update-trace-xuosp5ii`。
- 证据：逐rank逐键比较trace，compiled控制的step3/4更新前参数、梯度、更新后参数均与连续训练逐位相同。eager step3更新前参数完全一致，clip后梯度两rank均有12个键不同，最大绝对差分别1.1175871e-8/1.4901161e-8；step3更新后9/8键不同，最大1.1920929e-7/3.7252903e-9。step4梯度25键不同，最终17键不同。由此把首个观察到的偏差定位在首次恢复更新的反向或裁剪之前/期间，尚未区分前向、中间梯度及归约路径。
- 限制：trace仅记录clip后梯度，不能证明具体AOT算子根因，也不能单独证明差异符合独立参考；跨模式精确恢复仍失败。无生产行为修改，本批按新增诊断脚本和文档运行门禁，不重复稳定全量测试。
- 下一行动：捕获clip前梯度和前向输出，定位AOT/eager首次分歧，再与独立原生HF参考比较；继续保持原始失败和严格最终检查。

跨执行模式首次分歧进一步定位至反向/归约：

- 扩展 compile_trajectory_diagnostics.py，在真实 loss_sum 调用处保存 logits，并在 _clip_and_check_finite 前保存梯度、调用后保存全局梯度范数；保持原始最终严格比较，未改生产代码或容差。
- 两轮真实2rank full/compiled-resume/eager-resume：前两阶段退出0，eager最终严格比较仍退出1。最终trace中step3更新前参数和前向logits在两个rank均逐位相同；裁剪前已有12个梯度键不同，最大绝对误差rank0=1.4901161e-8、rank1=1.8626451e-8；全局梯度范数两路径均为1.324188470840454且逐位相同。compiled恢复控制所有观察值与连续训练逐位一致。
- 因此首个观察到的分歧位于相同logits之后的反向/分布式归约，而非参数恢复、前向输出或梯度裁剪。尚未区分AOT backward算子和归约路径，不能证明未知误差仅为舍入或宣称跨模式精确恢复通过。
- 证据：`/tmp/llmtuner-cross-preclip-trace-{full,control,eager}.log`、`/tmp/llmtuner-cross-logits-trace-{full,control,eager}.log`；最终产物 `/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-cross-logits-trace-1hi__6pg`。诊断脚本pre-commit、HEAD diff检查通过；无生产修改，未重复全量CPU测试。
- 下一行动：对首次更新中分歧参数的局部反向梯度与归约后梯度分别取证，或使用不含FSDP通信的独立原生HF参考隔离AOT backward；严格失败状态保留。

无通信原生 HF/eager/AOT 反向隔离：

- 扩展 compile_trajectory_diagnostics.py --isolated-backward：同初值Qwen3、seq32/vocab128/hidden64/2层，原生HF CausalLM.forward、eager wrapper、aot_eager wrapper，单线程CPU、无process group/FSDP，独立targets与cross entropy，确认生成compile图。输出及全部梯度按既有float32默认容差检查，同时报告逐位差异，不要求必须出现差异。
- 结果：三方logits逐位相同，所有参数梯度逐位相同，仓库脚本退出0（`/tmp/llmtuner-isolated-backward-repo.log`）。初轮临时实验遇既有Inductor PCH header编译错误，按已核实环境设置 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 后通过，未关闭compile；保留 `/tmp/llmtuner-isolate-aot.log` 与 `/tmp/llmtuner-isolate-aot-no-pch.log`。
- 结论限制：当前无通信最小输入不能复现分布式恢复的12键差异，不足以排除更复杂输入、累积或FSDP hook对AOT backward的影响；跨模式严格恢复仍失败，未放宽断言或调整生产计算。
- 下一行动：在真实首次恢复更新捕获FSDP reduce-scatter前的完整局部梯度与归约后梯度，检查两条路径的输入、累积顺序和归约调用顺序；也可将真实输入/全量参数导出到隔离对照以缩小用例。新增脚本及文档按范围跑pre-commit和diff门禁，无生产修改不重复全量suite。

FSDP foreach_reduce 前的局部完整梯度取证：

- 修改：compile_trajectory_diagnostics.py 在原生 _fsdp_param_group.foreach_reduce 入口观察参数FQN、调用序号和未归约完整梯度；仅复制CPU诊断张量后调用原函数，不改通信、hook或数值路径，原最终严格检查保持。
- 运行：真实2rank custom FSDP/relocated attention hooks/dynamic EMA，compiled full和compiled resume退出0，eager resume最终严格比较仍退出1。日志 `/tmp/llmtuner-cross-reduce-trace-{full,control,eager}.log`，trace产物 `/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-cross-reduce-trace-7ow1o9hd`。
- 首次恢复更新step3：两rank的25个参数归约入口顺序与compiled参考一致；compiled resume的所有归约输入逐位一致。eager在foreach_reduce前已有12个键不同，最大绝对差rank0=1.8626451e-8、rank1=1.3038516e-8，涉及第0层attention、MLP、norm与embedding；第1层、最终norm/lm_head梯度逐位一致。因此首个观察分歧已早于reduce-scatter，不能归因于collective调用顺序；更精确的backward算子/共享输入梯度累加机制仍待定位。
- 验证限制：仅CPU/aot_eager、固定输入与checkpoint；不是跨模式精确恢复通过证据，不放宽断言。无生产修改，诊断脚本和文档pre-commit及HEAD diff检查通过。
- 下一行动：核对第1层到第0层的输入梯度及残差/qkv分支累加；将真实输入与参数用于无FSDP的复现，区分AOT图分割与hook交互。全局目标保持未完成。

真实输入与权重的无FSDP反向重放：

- 扩展诊断脚本，在 Trainer._forward_backward_body 外侧记录实际inputs、labels、global_valid_tokens和positions，不改模型图或原始恢复断言。相同compiled checkpoint再resume取证退出0（`/tmp/llmtuner-real-input-capture.log`），首次恢复更新每rank一个128token batch、全局有效token248。
- 增加 `--replay-backward WORKDIR RANK`，从真实step2 DCP加载完整dense Qwen3参数，使用捕获的step3输入和实际summed CE/global token分母，在无process group/FSDP的eager/AOT模型重放；当前仅支持该固定dense诊断配置，不宣称通用模型导出。rank0临时脚本结果logits与全部梯度逐位一致（`/tmp/llmtuner-replay-real.log`）；rank1仓库脚本也退出0，logits及全部梯度逐位一致（`/tmp/llmtuner-replay-real-rank1.log`）。诊断脚本与文档pre-commit、HEAD diff检查通过。
- 证据继续缩小范围：真实输入/权重的无FSDP重放仍未复现先前归约前12键差异；不能据此证明FSDP错误或接受差异。生产计算与最终严格断言保持不变。
- 下一行动：在相同输入上比较native/custom FSDP及重定位/默认hooks，隔离FSDP图分割与backward分支累加交互。全局目标未完成。

跨compile恢复默认hook/native/custom对照收敛：

- 本批无生产修改，使用现有 compile_trajectory_diagnostics.py，对native和custom FSDP默认module hooks分别执行compiled full、compiled resume控制、eager resume，各为独立两rank CPU/Gloo进程，PP1、dynamic EMA、aot_eager。明确移除 LLMTUNER_TEST_RELOCATED_HOOKS，其他配置沿用已失败重定位场景。
- 两种实现的三个阶段全部退出0，最终模型/AdamW/scheduler/EMA/trainer精确检查通过，loss差异0。额外逐rank比较step3 logits、foreach_reduce输入、clip前梯度及更新后参数，全部逐位一致，调用顺序也一致。
- 对照结论：现有差异依赖已测重定位attention hooks配置；native默认和custom默认均不能复现，不应描述为所有compile/FSDP组合均失败。未证明重定位实现存在数值错误，仍需隔离同一attention下多投影unit的backward累加与AOT图边界。原重定位最终严格失败保留。
- 证据：`/tmp/llmtuner-cross-default-{native,custom}-{full,control,eager}.log`；产物 `/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-cross-default-native-t15_a5qc`、`/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-cross-default-custom-sfqi_irp`。
- 下一行动：在重定位attention的q/k/v/o投影间逐个切换hook归属，比较attention输入梯度和其分支累加，保持同模式控制与最终严格断言；目标设备与其他模型不由此验证。文档门禁通过，全局目标未完成。

单投影与QKV重定位的跨模式对照：

- 扩展诊断入口 LLMTUNER_TEST_TRACE_PROJECTIONS=q_proj 或 q_proj,k_proj,v_proj，使用实际pattern配置和attention hook_module，运行后逐投影断言真实hook owner；不修改生产配置或原始恢复断言。该入口直接复用_run，避免既有四投影专用断言误用于子集。
- 真实2rank CPU/Gloo custom FSDP/dynamic EMA/aot_eager：Q-only和QKV各compiled full/compiled control退出0，eager resume最终严格检查退出1。两者step3 logits逐位一致，归约前均12键不同，最大绝对差rank0=1.8626451e-8、rank1=1.3038516e-8；compiled控制归约输入逐位一致。差异无需多个重定位单元同时存在。
- 证据：`/tmp/llmtuner-cross-projections-{q,qkv}-{full,control,eager}.log`；产物 `llmtuner-cross-projections-q-15zeywbj`、`llmtuner-cross-projections-qkv-4lnx3ru4`（系统临时目录）。严格失败保留，不放宽容差；相关脚本和文档门禁通过。
- 限制：pattern选择也改变未选择参数的FSDP归属，不能仅凭Q-only失败就认定Q hook本身有错，亦不能归因于多通信context。下一行动：保持相同projection pattern，仅将hook_module从attention切回projection自身，对照相同分组下的图边界与梯度累加。

相同Q-only分组切换hook owner的反证：

- 修改诊断入口 LLMTUNER_TEST_TRACE_LOCAL_HOOKS=1，保持Q-only pattern及参数归属不变，hook_module从attention改为Q projection自身，并断言真实owner。非生产配置，无断言或容差调整。
- 真实两rank custom FSDP/dynamic EMA/aot_eager，compiled full和compiled control退出0，eager恢复最终严格比较退出1。step3 logits逐位一致，归约前仍12键不同，最大绝对差rank0=1.8626451e-8、rank1=1.3038516e-8，和attention owner配置相同。
- 此证据反证前批“只在重定位配置复现”可作为根因判断：差异无需重定位hook，细粒度projection pattern/分组的影响仍存在。此前native/custom默认decoder分组通过的证据保持有效，不以修复hook重定位为既定方向。
- 日志 `/tmp/llmtuner-cross-q-local-hook-{full,control,eager}.log`，产物 `/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-cross-q-local-hook-n2oyvj4i`；脚本及文档pre-commit、HEAD diff通过。
- 下一行动：在相同projection分组下比较native与custom，检查FSDP autograd边界引入的AOT graph partition和输入梯度累加顺序；严格跨模式失败保持开放。全局目标未完成。

原生FSDP相同Q-only分组复现：

- 扩展诊断pattern入口支持native FSDP，native不传hook_module override，逐投影断言state._modules确实指向Q projection；custom保留已有owner断言。保持同样Q-only分组与真实Trainer恢复流程，未修改生产代码。
- 真实两rank CPU/Gloo、dynamic EMA、aot_eager：compiled full和compiled control退出0，eager resume最终严格比较退出1。step3 logits逐位相同，foreach_reduce前12键已不同，最大绝对差rank0=1.8626451e-8、rank1=1.3038516e-8，与custom Q-only和重定位配置相同。
- 该证据说明复现不依赖custom state、hook重定位或多context实现；默认decoder单元native/custom均通过，细粒度Q-only原生也失败。根因仍须隔离原生FSDP autograd边界与AOT图分割/梯度累加顺序，不能宣称已修复或已证实仅为浮点舍入。
- 日志 `/tmp/llmtuner-cross-native-q-pattern-{full,control,eager}.log`；产物 `/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-cross-native-q-pattern-6zmjz7hm`。诊断脚本及文档门禁通过；无生产变更，不重复稳定全量suite。
- 下一行动：在无分布式模型中显式建立同样Q projection的AOT图边界或导出原生FSDP的AOT backward图，核对共享输入梯度累加顺序；当前跨模式严格失败与设备验证缺口仍开放。

无FSDP的Q projection显式图边界重放：

- 诊断脚本新增测试专用 `LLMTUNER_TEST_REPLAY_Q_BREAK=1`，仅对Q projection的forward使用 `torch.compiler.disable`，其余模型保持aot_eager；未修改生产实现或严格恢复断言。
- 使用真实step2完整checkpoint及step3捕获输入，rank0/rank1分别独立重放，无通信、无FSDP。两个进程均退出0，logits逐位一致，梯度在float32默认容差内，但同样12键不逐位一致；embedding最大绝对差分别1.862645149230957e-8与1.30385160446167e-8，与真实归约前案例一致。无显式边界的既有重放两rank均逐位一致。
- 复现不需要FSDP：显式AOT图边界已足以触发同样键集合和误差幅度。尚未逐元素核对差值张量，也未证明具体梯度累加顺序，不能将幅度一致等同于根因完全确定。
- 复现命令：`LLMTUNER_TEST_REPLAY_Q_BREAK=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. python tests/integration_tests/compile_trajectory_diagnostics.py --replay-backward <capture-root> <rank>`；capture-root须含真实checkpoint/step-2和trace-resume-aot_eager/step-3-rank-<rank>.pt。两rank原始日志为 `/tmp/llmtuner-q-break-replay-{0,1}.log`。
- 下一行动：保存重放梯度并逐元素比较真实FSDP归约输入，随后导出分段AOT backward图核对共享输入分支累加；跨模式严格恢复仍失败，目标设备验证仍开放。

无FSDP重放与真实归约输入的逐元素核对：

- 扩展 `compile_trajectory_diagnostics.py --replay-backward` 的Q图边界诊断，分别读取eager/aot_eager真实step3的 `gradients_before_reduce`，校验参数键集合并比较全部完整局部梯度；不使用clip后或归约后状态作替代。
- rank0/rank1均退出0，两种模式各25个参数全部逐位一致，四个 `replay_vs_fsdp_gradient_differences` 列表均为空。日志 `/tmp/llmtuner-q-break-compare-{0,1}.log`。因此已测配置中的12键跨模式分歧完整复现于无FSDP显式Q图边界路径，不只是最大误差碰巧相同。
- 这将首个差异定位到分段编译反向执行，未证明具体算子或共享输入累加顺序。跨模式严格恢复仍未通过，不能认定FSDP所有行为均正确，也不能放宽恢复断言。
- 本批仅修改诊断测试与记录；相关pre-commit通过。下一行动：导出该无FSDP最小重放的AOT backward分段，核对Q/K/V共享输入梯度的累加顺序，构造独立的小规模多分支参考。

AOT backward图导出及简化分支反证：

- 用 `TORCH_LOGS=aot_graphs` 配合既有Q图边界rank0重放命令导出真实分段图，进程退出0，保留 `/tmp/llmtuner-q-break-aot-graphs.log`。对应attention backward图中，V输入梯度view_12与K输入梯度view_15经 `aten.add.Tensor` 形成add_9；Q projection在禁用编译的边界外，Q输入梯度不在该图内累加。这确认了分段梯度汇合结构，未直接观察eager的运行时累加次序。
- 独立临时三分支实验使用固定seed7、输入128×64、三组64→128线性投影，Q设显式边界，其余aot_eager。对 `q*k+v` 及带Q/K RMSNorm的变体分别运行，均退出0，前向及输入梯度逐位一致；独立计算三条输入梯度后，eager/compiled均逐位对应 `(v+k)+q`。简化模型未复现，不能将真实问题直接归因于三分支加法不结合性。
- 原真实Qwen重放的完整梯度对应证据仍有效，跨模式严格恢复仍失败。下一行动：保留真实attention的SDPA、Q/K norm及残差依赖，隔离分支汇合；避免继续用已证实不复现的简单逐元素三分支扩大随机规模。本批无生产或测试代码修改，仅新增诊断证据记录。

反向残差汇合的首次重结合差异：

- 诊断重放新增测试专用 `LLMTUNER_TEST_REPLAY_ADDS=<path>`，仅在backward中用TorchDispatchMode捕获形状1×128×64的add/add_输入，不包裹forward编译。rank0重放退出0，全部25参数仍逐位对应真实FSDP归约输入，观察器未改变已测结果。命令沿用Q_BREAK重放并增加上述变量，日志 `/tmp/llmtuner-backward-adds.log`，捕获 `/tmp/llmtuner-backward-adds.pt`。
- 两模式前6次加法的输入逐位一致。首个不同汇合处，eager记录6/7的输入可写为 `(r+a)+b`，AOT记录6/7为 `r+(a+b)`；r、a、b分别跨模式逐位相同，两个中间结果也分别逐位等于对应独立重算。两种重结合结果的最大绝对差9.313225746154785e-10，与捕获的汇合结果差异一致。这是残差与分段分支的三路汇合，不能仅用无残差Q/K/V小模型解释。
- 此证据确认已测rank0首次差异来自浮点加法重结合；尚未据此修改数值契约或放宽跨模式最终断言。下一行动：将残差路径纳入独立小模型复现，并核对rank1同一汇合；评估能保持生产编译机制的确定性累加实现及其代价。跨模式严格恢复仍失败，设备验证仍开放。

rank1残差重结合确认与独立小模型：

- rank1启用REPLAY_ADDS完成真实输入重放退出0，25参数仍逐位对应FSDP归约输入。前6次加法输入相同，首次三路汇合的r/a/b及两个中间结果分别逐位对应；eager `(r+a)+b` 与AOT `r+(a+b)` 最大绝对差4.656612873077393e-10。rank0/rank1均确认该结构，不再仅为rank0观察。
- 新增诊断入口 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. python tests/integration_tests/compile_trajectory_diagnostics.py --residual-backward`。固定seed7，128×64输入、三组64→64线性、共享输入RMSNorm、Q/K RMSNorm、Q显式图边界和残差；完全不依赖HF/FSDP/checkpoint。退出0，前向逐位相同，输入梯度最大绝对差4.76837158203125e-7；输出仅为诊断，没有宣称等价通过。去掉共享输入prenorm的对照仍逐位相同。
- 日志 `/tmp/llmtuner-backward-adds-rank1.log`、`/tmp/llmtuner-residual-minimal.log`。相关pre-commit通过。下一行动：在独立小模型导出prenorm backward，检查残差与RMSNorm内部两个梯度分量的汇合，评估固定该边界累加的编译兼容方案。生产与原严格恢复断言未变，问题未关闭。

独立小模型确认RMSNorm反向两分量与残差重结合：

- 导出 `--residual-backward` 的AOT图，prenorm backward返回 `add(mul_2,mul_6)`：直接分量与方差分量在图内先相加，再与图外残差汇合。日志 `/tmp/llmtuner-residual-minimal-graphs.log`，退出0。
- 小模型新增反向add输入捕获及机制断言：两模式前4次add输入逐位相同；最后三份输入分别为残差g、prenorm直接梯度、方差梯度，跨模式均逐位相同。断言eager输入梯度逐位等于 `(residual+direct)+variance`，compiled输入梯度逐位等于 `residual+(direct+variance)`。诊断运行退出0，保持最大绝对差4.76837158203125e-7，日志 `/tmp/llmtuner-residual-merge-assertions.log`。
- 这解释最小复现及真实追踪的首个三路汇合，未证明通过更改本项目FSDP即可消除编译器重分组。下一行动：检查上游RMSNorm实现/编译边界契约与PyTorch已有保持backward边界的机制；先在独立模型验证候选，再决定生产适配。禁止通过关闭compile或放宽严格恢复断言将问题标为关闭。

可微分输入副本候选的独立验证：

- 核对本地torchtitan `torchtitan/models/common/norm.py::GatedRMSNorm.forward` 使用F.rms_norm和local_compile；并非HF Qwen的手写RMSNorm，不能直接假设其反向边界相同。PyTorch strict_mode的autograd为未实现，不能用作可训练边界方案。
- 独立复现新增 `LLMTUNER_TEST_CLONE_PRENORM=1`：仅在共享输入RMSNorm前执行x.clone，保留Q图边界与aot_eager。仓库诊断命令退出0，六次add输入全部逐位一致，前向及输入梯度逐位一致，maxabs=0；日志 `/tmp/llmtuner-clone-prenorm-repo.log`。原无clone机制断言保留，候选仅在测试。
- clone在eager中建立梯度汇合边界，使norm直接/方差梯度先合并后传给原输入；增加一次张量复制且改变原eager舍入路径。未验证真实HF、FSDP、AC、Inductor、DTensor及设备内存代价，不能宣称生产修复或所有模式确定性。
- 下一行动：在真实Qwen无FSDP重放两种模式均给残差前norm输入建立此边界，比较全部参数梯度；保持已有原实现追踪作为对照，不将新实现梯度与旧路径混为恢复通过证据。

真实Qwen重放验证归一化输入副本候选：

- `--replay-backward` 支持CLONE_PRENORM，在两侧每层input_layernorm/post_attention_layernorm注册测试专用输入clone hook；在deepcopy与compile前完成，不改生产HF实现。候选模式明确跳过与旧实现归约梯度的同路径对应检查，仍比较候选eager/AOT的全部梯度，并新增逐位断言。
- 使用同一真实step2 checkpoint和step3捕获输入，rank0/rank1均退出0，logits逐位相同、25个参数梯度全部逐位相同，差异列表均为空。命令为既有Q_BREAK重放加 `LLMTUNER_TEST_CLONE_PRENORM=1`；日志 `/tmp/llmtuner-clone-qwen-{0,1}.log`。原无clone的失败及逐位归约对应证据保留。
- 该结果是单步无分布式候选验证，不是原实现修复或最终恢复通过。尚需真实FSDP新轨迹的full/compiled resume/eager resume、AC及设备验证；内存复制代价与适用模型范围未确定。下一行动：将相同测试变换在assembly前施加于真实Trainer，运行Q-only细粒度FSDP的全新轨迹与严格恢复。

真实Trainer细粒度FSDP候选的严格恢复验证：

- 诊断assembly配置在parallelism前给每层两处残差前norm添加输入clone hook，受CLONE_PRENORM测试开关控制；生产代码未改。新建轨迹目录 `/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-clone-trajectory-0ikepmob`，不覆盖原失败。
- 实际两rank CPU/Gloo、Q-only pattern/custom FSDP、attention hook owner、dynamic EMA。四步compiled(aot_eager) full、同模式resume及eager resume分别独立torchrun，均退出0。原pp_checkpoint_equivalence的恢复前完整状态和最终模型/optimizer/EMA/scheduler/trainer严格断言均保留；两rank pre-update state failures=0，恢复loss最大绝对差0。日志 `/tmp/llmtuner-clone-trajectory-{full,control,eager}.log`。
- 运行方式：各阶段 `LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom LLMTUNER_TEST_TRACE_PROJECTIONS=q_proj LLMTUNER_TEST_CLONE_PRENORM=1 PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=<free-port> tests/integration_tests/compile_trajectory_diagnostics.py <full|resume> <fresh-root>`；full及compiled控制设置COMPILE_BACKEND=aot_eager，eager不设置。
- 候选闭合了该配置的严格恢复实验，未闭合生产缺陷：尚未选择正式HF适配机制、界定适用norm/模型、验证AC/native FSDP/Inductor与目标设备，也未量化复制代价。下一行动：审查HF适配入口并设计有明确范围的梯度汇合边界，避免无条件复制所有模型norm或依赖测试hook。

无额外数据复制的view边界候选：

- 生产接入审查定位到HF构造适配层：模型行为变换应在HF构建时对eager/compile一致施加，不能仅在apply_compile改变已保存数值路径。当前不做无范围的全模型norm hook替换。
- 在独立prenorm残差复现中，以x.view_as(x)替代x.clone，同样前向与输入梯度逐位一致，maxabs0；view共享存储，避免clone的数据副本，但会增加autograd节点，编译/DTensor契约仍需实测。
- 真实Qwen诊断新增测试专用VIEW_PRENORM选项，rank0真实checkpoint/输入重放退出0，logits与全部参数梯度逐位一致，日志 `/tmp/llmtuner-view-qwen-0.log`。该结果未覆盖rank1和真实FSDP，不能沿用clone候选的恢复通过结论。
- 下一行动：验证view候选rank1与两rank新轨迹严格恢复，并检查TP非连续输入和AC；若有效，优先采用不复制数据且限定已知纯norm的模型适配。生产代码未修改。

view候选真实FSDP与full AC严格恢复：

- rank1无FSDP真实输入重放退出0，logits/全部25参数梯度逐位一致，日志 `/tmp/llmtuner-view-qwen-1.log`。普通非连续转置张量view_as另验证共享storage、保持stride和平方和反向；此检查不代表DTensor/TP证据。
- 诊断assembly支持VIEW_PRENORM，在并行变换前为两处残差前norm建立共享存储view输入。两rank CPU/Gloo、Q-only custom FSDP/attention hooks/dynamic EMA，分别无AC和full AC新建轨迹，各执行compiled full、compiled resume、eager resume，六个torchrun均退出0。原始恢复前与最终完整状态逐位断言保留，loss差异0。
- 无AC目录 `/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-view-trajectory-e5d2j71p`，full AC目录同父路径 `llmtuner-view-ac-trajectory-ph8ad6ag`；日志 `/tmp/llmtuner-view-trajectory-{full,control,eager}.log` 与 `/tmp/llmtuner-view-ac-trajectory-{full,control,eager}.log`。沿用clone批次命令，将CLONE_PRENORM替换为VIEW_PRENORM，full AC再加 `LLMTUNER_TEST_AC=full`。
- 候选避免clone数据副本且覆盖重计算路径，但仍仅为测试适配；生产接入、native FSDP、TP/DTensor、Inductor/目标设备均未完成。下一行动：用可保持模块身份/FQN的HF归一化适配实现共享输入view边界，限定已知纯RMSNorm类并验证导出、参数身份和独立HF数值；不得仅按属性名称替换任意norm。

生产Qwen3残差归一化梯度汇合适配：

- `models/hf/model_setup.py::stabilize_residual_norm_gradients` 在HF构造阶段对精确Qwen3RMSNorm/Qwen3MoeRMSNorm类型的input_layernorm/post_attention_layernorm添加顶层可deepcopy、kwargs兼容的view输入pre-hook。只适配已知纯norm，保持原模块/Parameter身份、权重与FQN，重复调用不重复注册；HFTransformerModel在模型构造后调用。两模式一致应用，不依赖compile开关，不修改HF类或全局注册。
- 适配改变原eager浮点汇合顺序；旧checkpoint仍严格加载相同键，但旧实现训练轨迹不能宣称逐位延续。共享存储view避免clone副本，不承诺Inductor消除view后的全部设备确定性。
- 新增两类CPU回归：参数身份/键保持、幂等、原生HF前向及梯度参考、非连续输入/kwargs和deepcopy。全量pytest最终3187 passed、2 skipped，退出0，日志 `/tmp/llmtuner-production-norm-all-units.log`。
- 无候选VIEW/CLONE开关的真实两rank CPU/Gloo、custom Q-only FSDP/attention hooks/full AC/dynamic EMA，新轨迹compiled full、compiled resume、eager resume均退出0，完整状态严格断言通过、loss差异0。目录 `/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-production-norm-trajectory-8xmjq08y`，日志 `/tmp/llmtuner-production-norm-trajectory-{full,control,eager}.log`；运行方式同view/full AC批次但移除VIEW_PRENORM。
- 全部适用tracked/untracked文件pre-commit通过，明确排除受保护examples/native_alltoall.py；compileall通过。未新增生产文件，逐文件清单路径不变，本批重新审查model.py/model_setup.py的构造调用链。下一行动：生产native FSDP对照及TP/DTensor view布局检查，再扩展MoE；Inductor与目标设备仍待验证，整体目标未完成。

生产norm边界的native、MoE/EP与TP×FSDP复核：

- 两rank CPU/Gloo、full AC/dynamic EMA/Q-only细粒度分组：native FSDP使用projection自身hooks，custom MoE使用Qwen3MoE+EP2及attention hooks。两组各compiled full、compiled resume、eager resume均退出0，原始恢复前及最终完整状态逐位断言保持，loss差异0；未启用VIEW/CLONE测试变换。
- native目录 `/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-production-norm-native-mm_189j0`；MoE目录同父路径 `llmtuner-production-norm-moe-p08bl3d1`。日志 `/tmp/llmtuner-production-norm-{native,moe}-{full,control,eager}.log`。native沿用生产轨迹命令设FSDP_IMPLEMENTATION=native、TRACE_LOCAL_HOOKS=1；MoE设MOE=1、EP_SIZE=2，保留custom。
- 四rank `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=<free-port> tests/integration_tests/tp_fsdp_equivalence.py` 退出0，真实Qwen3 Trainer TP2×FSDP2四步与单进程参考比较，loss最大差4.768e-7、参数最大差7.000e-6，既有容差不变，failed ranks=0。日志 `/tmp/llmtuner-production-norm-tp-fsdp.log`；覆盖实际DTensor分片训练，不代表跨模式TP恢复或所有view layout。
- 本批未修改生产源码，仅补充生产适配的并行证据。下一行动：Inductor真实编译数值与恢复契约，以及其他仍开放的设备/生命周期待办；CPU/Gloo结论不推广到CUDA/NPU内核，整体目标仍未完成。

CPU Inductor真实轨迹及独立原生HF数值对照：

- 两rank CPU/Gloo、custom Q-only FSDP/full AC/dynamic EMA，backend=inductor；compiled full及compiled resume均退出0，eager resume最终严格检查退出1，保留失败。恢复前完整状态无差异，step3参数相同；但logits已有rank0=1.7881393432617188e-7、rank1=1.4901161193847656e-7差异，25个归约前梯度键不同，更新后16键不同。这与此前仅反向重结合的aot_eager案例不同，生产view适配不承诺消除Inductor前向融合差异。
- 目录 `/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-production-norm-inductor-8ayuu02z`；日志 `/tmp/llmtuner-production-norm-inductor-{full,control,eager}.log`。运行方式同生产full AC轨迹，将COMPILE_BACKEND从aot_eager设为inductor；不启用候选变换。未放宽严格断言。
- `--isolated-backward` 支持环境选择backend，原生HF参考删除本项目norm hook，仅使用其原始CausalLM forward。补充直接compiled-vs-native logits及所有参数梯度比较，采用原有torch.testing float32默认容差，确认unique_graphs>0；`LLMTUNER_TEST_COMPILE_BACKEND=inductor TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. python tests/integration_tests/compile_trajectory_diagnostics.py --isolated-backward` 退出0，日志 `/tmp/llmtuner-production-inductor-direct-native.log`。梯度和logits非逐位一致，该计算正确性证据不替代跨模式严格恢复。
- 本批仅改诊断测试和文档，相关pre-commit通过。下一行动：核对Inductor浮点融合/重关联选项与实际生成代码，判断可否在保留编译机制下建立约定的跨模式数值契约；继续保留严格失败为开放项，不能将默认容差计算通过当作逐位恢复通过。

Inductor前向归约的独立复现：

- 本机PyTorch2.14已为use_fast_math=False、cpp.enable_unsafe_math_opt_flag=False、cpp.enable_floating_point_contract_flag=off；cpp_builder确认相应编译选项，不能通过重复关闭这些开关解释或消除当前差异。
- 新增 `--inductor-norm`：seed42、128×64输入，返回pow2/mean/rsqrt/norm，确认编译图且保持float32默认容差。实际退出0；pow2逐位一致，默认mean最大差2.384185791015625e-7，cpp.simdlen=1标量路径仍差3.5762786865234375e-7；最终输出差分别4.76837158203125e-7、7.152557373046875e-7。生成代码确认向量和标量路径不同归约，首次差异在mean，不足以判断rsqrt独立误差。
- 运行 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. python tests/integration_tests/compile_trajectory_diagnostics.py --inductor-norm`；日志 `/tmp/llmtuner-inductor-norm-repo.log`，生成代码 `/tmp/llmtuner-inductor-norm-probe.log`。不依赖HF/FSDP/checkpoint，关闭SIMD未恢复逐位一致；相关pre-commit通过。
- 下一行动：核对ATen与生成归约累加树及保持归约算子边界的可训练编译机制；不继续重复已关闭fast-math选项。生产未改，Inductor跨模式严格恢复失败仍开放，该实验未证明整模型全部差异来源。

可训练ATen mean边界的隔离实验：

- 新增 `--inductor-mean-boundary` 诊断，用测试命名空间custom_op仅保留ATen mean末维归约，提供fake及expand/末维长度的autograd公式，其余pow/rsqrt/乘法继续Inductor。无生产注册或全局lowering修改。
- seed42、128×64实际运行退出0，unique_graphs=1；平方、mean、rsqrt及最终norm四份前向结果全部逐位一致，新增严格断言。输入梯度仍非逐位一致，最大绝对差9.5367431640625e-7，保持float32默认容差断言并通过。归约边界在该前向实验有效，但反向存在独立融合/运算顺序差异，不能将保留mean推广成完整训练修复。
- 命令 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. python tests/integration_tests/compile_trajectory_diagnostics.py --inductor-mean-boundary`；日志 `/tmp/llmtuner-mean-boundary-repo.log`。相关pre-commit通过。ATen完整SumKernel实现未包含在本地安装头文件中，不声称已核对其完整累加树。
- 下一行动：用同一输入及上游梯度隔离norm backward公式，区分custom mean backward与其外部融合；在完整训练证据前不接入生产opaque算子。Inductor跨模式严格恢复失败保留，整体目标未完成。

固定上游梯度的norm backward归约隔离：

- 新增 `--inductor-norm-backward`，seed42/128×64固定输入、随机上游g及同一eager计算的scale，逐段返回direct、sum(g*x)、scale^3、variance、mean_grad、square_grad、indirect和最终梯度。手工eager公式与原生autograd逐位一致（无loss反向），确认参考。
- Inductor首次在sum(g*x)出现差异2.1457672119140625e-6，direct/scale^3/square_grad逐位一致，最终梯度差2.384185791015625e-7。测试命名空间custom_op仅保留该sum为ATen边界，其余公式仍编译，八项返回结果全部逐位一致。custom sum是固定张量公式的前向隔离探针，未注册高阶梯度，也不用于训练生产。
- 命令 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. python tests/integration_tests/compile_trajectory_diagnostics.py --inductor-norm-backward`；实际退出0，原float32默认容差全部通过，日志 `/tmp/llmtuner-norm-backward-boundary.log`。
- 已将前后向归约差异分别隔离；不代表整模型所有算子均可逐位一致。下一行动：评估完整norm前后向opaque边界的可训练实现及独立梯度契约，然后决定是否适合生产，不能通过全面回退eager或放宽恢复断言关闭问题。生产未改，Inductor跨模式严格恢复仍失败。

可训练opaque RMSNorm的一阶独立契约：

- 新增 `--inductor-opaque-norm` 测试专用custom_op前后向，前向保留原ATen pow/mean/rsqrt/weight顺序，backward使用已独立核对的公式并作为opaque边界，提供fake和一阶autograd注册。外层保持Inductor fullgraph编译，没有修改生产norm或注册生产算子。
- seed42/128×64、随机权重和固定随机上游梯度：输出、输入梯度、权重梯度全部逐位一致（误差0）；2×4 float64有限差分gradcheck通过。实际仓库诊断命令退出0，日志 `/tmp/llmtuner-opaque-norm-repo.log`，命令 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. python tests/integration_tests/compile_trajectory_diagnostics.py --inductor-opaque-norm`。
- 限制：仅一阶普通CPU张量；不支持高阶autograd声明，未验证混合精度、DTensor、AC或整模型所有融合算子。保留ATen执行有性能代价，尚无生产采用决定。
- 下一行动：只在诊断中将真实Qwen纯RMSNorm替换成等式边界，先比较整模型前向以判断还有无其他差异源；若仍不同，不把norm单点修复当成跨模式恢复解决方案。Inductor严格跨模式失败仍开放，生产本批未改。

真实Qwen的opaque norm与独立HF参考：

- 新增 `--inductor-opaque-qwen`，仅给诊断实例的精确Qwen3RMSNorm绑定opaque forward，原生HF参考在变换前deepcopy并删除本项目view hook，避免参考共用候选算子；生产与全局HF类未修改。
- 首次真实attention反向暴露非连续Q/K norm的fake stride错误，原始日志 `/tmp/llmtuner-opaque-qwen-native-reference.log` 保留退出1。测试算子反向显式contiguous并使fake相同，未放宽metadata断言；修正后退出0，日志 `/tmp/llmtuner-opaque-qwen-contiguous.log`。
- `LLMTUNER_TEST_COMPILE_BACKEND=inductor TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. python tests/integration_tests/compile_trajectory_diagnostics.py --inductor-opaque-qwen` 实际编译图确认，候选eager/Inductor logits和全部参数梯度逐位一致。两者分别及直接compiled对原生HF logits/梯度均通过既有float32默认容差；原生参考梯度非逐位相同，原因包含norm梯度边界与残差汇合差异，不能宣称原HF轨迹逐位保持。
- 该证据覆盖随机单batch的完整Qwen前后向，不是多步FSDP严格恢复；dtype断言限定float32，DTensor/混合精度/高阶及性能未实现。下一行动：opcheck包含非连续输入的fake/alias/autograd契约，并将候选限制在测试中运行真实Inductor恢复，不凭单batch通过接入生产。原生产跨模式严格失败仍开放。

opaque norm算子契约与真实Inductor严格恢复：

- `LLMTUNER_TEST_NORM_OPCHECK=1 ... --inductor-opaque-norm` 对连续及转置非连续3D输入，前向/反向算子各opcheck的schema、autograd注册、faketensor、aot_dispatch_dynamic全部SUCCESS；float64 gradcheck及float32逐位比较仍通过，退出0，日志 `/tmp/llmtuner-opaque-opcheck.log`。不意味着反向算子支持高阶梯度。
- 新增测试专用OPAQUE_NORM装配路径（实际变量 `LLMTUNER_TEST_OPAQUE_NORM=1`），在parallelism前为诊断实例绑定候选norm，保持生产模块与全局HF类不改。两rank CPU/Gloo、Q-only custom FSDP/attention hooks/full AC/dynamic EMA，Inductor full、Inductor resume、eager resume均退出0，原完整状态逐位断言通过、两rank pre-update failures=0、loss差异0。
- 目录 `/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-opaque-norm-inductor-lcz0zkym`；日志 `/tmp/llmtuner-opaque-norm-inductor-{full,control,eager}.log`。沿用生产Inductor轨迹命令加 `LLMTUNER_TEST_OPAQUE_NORM=1`，各阶段一致应用；原无opaque的跨模式失败记录保留。
- 此结果闭合该测试候选的多步恢复，不代表生产路径已修复。候选仅float32/普通CPU、一阶；保留ATen边界带来性能与编译融合取舍，混合精度/DTensor/device仍未验收。下一行动：核对能否以明确配置与设备能力契约提供可维护的生产归一化机制，先处理混合dtype及任意前导维/冻结权重的autograd语义，避免将测试实现直接复制到生产。

候选norm一维权重梯度缺陷与冻结契约：

- 审查发现测试候选对一维输入使用sum(tuple(range(ndim-1)))，空tuple在PyTorch表示全维归约，会错误返回标量权重梯度。已改为仅ndim>1归约前导维，一维直接保留逐元素g*norm；生产未使用该候选，不影响已有生产状态。
- 新增 `LLMTUNER_TEST_NORM_SHAPES=1`，覆盖(4,)、(0,4)、(3,4)、(2,3,4)，各输入/权重都训练、仅输入训练、仅权重训练，共12配置。前向与原生公式、所需输入/权重梯度逐位一致；冻结项grad保持None。原Inductor逐位验证及float64 gradcheck仍通过，实际退出0，日志 `/tmp/llmtuner-opaque-shapes.log`。
- 命令 `LLMTUNER_TEST_NORM_SHAPES=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. python tests/integration_tests/compile_trajectory_diagnostics.py --inductor-opaque-norm`；相关pre-commit通过。下一行动：混合dtype的HF精确cast/梯度cast契约；候选未达生产适用条件，Inductor生产跨模式严格失败仍开放。

opaque候选低精度cast与权重dtype契约：

- 测试候选对fp16/bf16输入采用FP32归一化，结果回转输入dtype后乘权重；backward先按乘法输入dtype回转，再转FP32计算norm梯度，最后回转输入梯度，权重梯度归约后转权重dtype。fake输出dtype按输入/权重promote_types，不再总是假设输入dtype。
- `LLMTUNER_TEST_NORM_DTYPES=1` 覆盖fp16/bf16/fp32输入×三种权重dtype，共9组合，与独立原生Qwen3RMSNorm比较输出dtype/逐位前向/逐位输入及权重梯度，全部通过，退出0。日志 `/tmp/llmtuner-opaque-dtypes.log`。这9项为普通CPU eager契约，不是低精度Inductor或设备证据。
- 修改后重新运行SHAPES=1及OPCHECK=1，12个维度/冻结配置、连续非连续算子检查、原float32 Inductor逐位对照及float64数学公式gradcheck仍通过，退出0，日志 `/tmp/llmtuner-opaque-contracts-after-dtype.log`。float64 gradcheck用于数学公式，不声明HF强制FP32的float64语义一致。
- 相关pre-commit通过；生产未改。下一行动：真实混合dtype Inductor输出/fake验证及设备契约评估，然后明确候选性能与支持范围再决定生产接入；原生产Inductor跨模式严格失败仍开放。

九种dtype组合的实际Inductor契约：

- 在既有DTYPES矩阵增加 `LLMTUNER_TEST_NORM_COMPILE_DTYPES=1`，每组独立reset Dynamo计数并以Inductor/fullgraph调用候选，反向后断言unique_graphs>0，避免只验证eager。
- fp16/bf16/fp32输入×三种权重dtype共9组合均退出前完成逐位输出/输入梯度/权重梯度与原生Qwen3RMSNorm比较，输出dtype一致；整次进程退出0，原独立float32逐位与float64 gradcheck仍通过。日志 `/tmp/llmtuner-opaque-compiled-dtypes.log`。
- 命令 `LLMTUNER_TEST_NORM_DTYPES=1 LLMTUNER_TEST_NORM_COMPILE_DTYPES=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. python tests/integration_tests/compile_trajectory_diagnostics.py --inductor-opaque-norm`。这是普通CPU张量编译证据，不覆盖CUDA/NPU或FSDP mixed precision。生产未改。
- 下一行动：明确DTensor的自定义算子分发/placement契约与性能取舍，再决定生产可维护实现；不能将CPU dtype矩阵推广为全设备机制完成。生产Inductor跨模式严格失败仍开放。

候选自定义norm的真实DTensor分发缺口：

- 新增 `LLMTUNER_TEST_NORM_DTENSOR=1` 两rank诊断，输入4×8采用Shard(0)，权重8采用Replicate，真实CPU/Gloo调用候选norm。两个rank均报NotImplementedError：Operator llmtuner_diagnostic.norm.default does not have a sharding strategy registered。用all_gather_object确认一致错误后正常清理，探针退出0；此退出0表示预期缺口被确认，不是能力通过。
- 命令 `LLMTUNER_TEST_NORM_DTENSOR=1 PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=<free-port> tests/integration_tests/compile_trajectory_diagnostics.py --inductor-opaque-norm`；日志 `/tmp/llmtuner-norm-dtensor-probe.log`。
- 核对torch.distributed.tensor.experimental.register_sharding接口可为自定义算子定义output/input placement策略。前导维分片可局部归一化，hidden维分片不能直接局部执行；反向复制权重的梯度需声明Partial(sum)，不能误标Replicate导致丢失归约。尚未注册策略或宣称实现。
- 下一行动：在测试命名空间定义前导维分片及复制策略，并以独立dense输入/权重梯度核对真实两rank反向和partial归约；生产本批未改，Inductor生产跨模式问题与设备验证仍开放。

候选DTensor前后向布局策略及hidden重分布：

- 测试命名空间注册前向Replicate与前导维Shard策略，权重保持Replicate；反向input gradient同分片，前导维分片的weight gradient标Partial(sum)。没有hidden维局部策略，允许DTensor按复制策略重分布后执行。
- 真实两rank CPU/Gloo分别测试复制、Shard(0)序列、Shard(1)hidden输入；输出和两类梯度full_tensor与独立dense公式通过原float32默认容差。序列分片断言权重梯度Partial，hidden分片断言输出Replicate，确认未错误局部归一化。全部rank完成并退出0，日志 `/tmp/llmtuner-norm-dtensor-hidden.log`；原无策略失败日志保留。
- 命令沿用NORM_DTENSOR=1探针，当前语义变为布局数值验证。注册仅覆盖一维mesh测试，未验证多轴、编译DTensor或设备，不声明hidden高效分布式归约；hidden路径需要实际all-gather。
- 下一行动：核对实际TP norm的本地/DTensor输入契约与多轴策略，再评估生产机制；当前候选仍测试限定，生产Inductor严格跨模式失败仍开放。

四rank二维mesh候选布局验证：

- NORM_DTENSOR探针新增 `LLMTUNER_TEST_NORM_MESH_2D=1`，真实四rank CPU/Gloo构造2×2 mesh，覆盖(R,R)、(S0,R)、(S0,S0)。完整输出及输入/权重梯度与独立dense参考通过float32默认容差，逐轴权重梯度Replicate/Partial(sum)断言通过，进程退出0。
- 命令 `LLMTUNER_TEST_NORM_DTENSOR=1 LLMTUNER_TEST_NORM_MESH_2D=1 PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=<free-port> tests/integration_tests/compile_trajectory_diagnostics.py --inductor-opaque-norm`；日志 `/tmp/llmtuner-norm-dtensor-2d.log`。full_tensor物化S0/S0时PyTorch警告顺序all-gather成本，保留警告，不将数值验证当通信性能优化。
- 核对tensor_parallel.py的apply_tp明确拒绝额外SequenceParallel变换：norm消费已有sequence shard。候选DTensor策略不应额外复制该归约职责；实际生产自定义TP路径仍需独立候选集成，不能仅凭模块说明宣称运行输入类型已验证。
- 下一行动：编译DTensor候选前后向并检查Partial输出，随后整理生产采用的性能/生命周期边界；本批仅诊断与文档，生产Inductor跨模式严格失败仍开放。

候选Inductor DTensor前后向与二维mesh：

- NORM_DTENSOR增加 `LLMTUNER_TEST_NORM_COMPILE_DTENSOR=1`，各layout独立reset计数，以Inductor/fullgraph执行，backward后断言unique_graphs>0；其余dense参考/placements/Partial断言保持。
- 两rank一维mesh复制/S0/S1及四rank二维mesh(R,R)/(S0,R)/(S0,S0)全部完成，输出和两类梯度通过既有float32默认容差；权重Partial(sum)及hidden重分布断言通过，两进程组均退出0。日志 `/tmp/llmtuner-norm-compiled-dtensor.log`、`/tmp/llmtuner-norm-compiled-dtensor-2d.log`；PyTorch opaque值弃用及顺序all-gather警告保留。
- 运行沿用NORM_DTENSOR探针，添加COMPILE_DTENSOR=1及PCH=0；二维再加MESH_2D=1并使用四rank。没有更改生产算子/注册，CPU编译证据不能推广目标设备或完整TP Trainer。
- 下一行动：整理可维护生产归一化实现的明确配置与注册生命周期，评估是否保留ATen边界及其性能代价；候选CPU单/多轴契约已具证据，避免重复相同矩阵。生产Inductor跨模式严格失败与硬件项仍开放。

候选opaque norm生产采用的CPU性能代价：

- 增加NORM_BENCHMARK=1诊断，以torch.utils.benchmark单线程、每配置3次预热、blocked_autorange至少0.3秒，固定上游梯度计算forward+backward并清空grad，排除编译。实际退出0，日志 `/tmp/llmtuner-norm-benchmark.log`。
- 输入128×64：原生eager45.08µs、融合Inductor36.64µs、opaque Inductor84.74µs；512×1024：分别1474.71/732.76/1903.81µs。opaque约融合路径2.31/2.60倍。仅本机CPU局部算子结果，未测峰值内存或完整Trainer吞吐，不推广GPU/NPU。
- 命令 `LLMTUNER_TEST_NORM_BENCHMARK=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. python tests/integration_tests/compile_trajectory_diagnostics.py --inductor-opaque-norm`；修正benchmark闭包显式绑定循环变量，避免后续复用时捕获漂移。
- 生产设计决定：不得无条件用opaque替换当前已验证view适配；需要显式的模型数值边界选项，eager与compile同选项运行，记录不兼容轨迹的限制。下一行动：在模型配置定义保留ATen norm边界的明确选项和校验，再做可维护生产实现/注册生命周期；性能证据不作为放弃严格恢复的理由。整体目标未完成。

配置到生产归一化后端复核：修复 `LLMTunerConfig` 缺失 `norm_backend`
转发属性导致 HF factory 静默选择 native 的缺陷，factory 改为显式读取属性。
新增 ModelConfig → HF 配置 → 模型实例 → forward 的 native/aten 回归，
并在 checkpoint 集成脚本断言实际 Qwen norm 已绑定生产 aten 实现。
`pytest -q tests/unit_tests/cpu/models/test_hf_wrapper.py -k norm_backend
--disable-warnings`：4 passed；本批四个文件 Ruff 与 `git diff --check` 通过。
生产双 rank custom FSDP、full AC、dynamic EMA、Inductor 连续运行、
同模式恢复及跨 eager 恢复均退出 0，保持原严格状态断言；生产 norm 绑定断言通过。
命令：`python /tmp/llmtuner-run-production-aten.py`，底层运行仓库
`compile_trajectory_diagnostics.py` 的 full/resume；配置包括 PP1、q_proj
细粒度 FSDP、CPU/Gloo、norm_backend=aten。临时 launcher 和日志仅作补充证据。
下一动作：核验生产算子契约、更新新增源码清单，再完成累计质量门禁。

生产 ATen RMSNorm 契约复核：新增独立 HF 参考的 36 项输入/权重 dtype ×
shape 数值回归，覆盖一维、空 batch、多维和非连续输入，输出及输入/权重梯度
逐位一致；另验证 float64 输入拒绝与真实生产算子的 schema、fake、autograd、
动态 AOT opcheck。`pytest -q tests/unit_tests/cpu/models/test_normalization.py
--disable-warnings`：38 passed。当前生产源码与逐文件表核对为 163 项。
此结果不覆盖目标设备、高阶梯度或所有分布式布局。

本批累计门禁：完整 CPU suite 为 3229 passed、2 skipped、221 warnings，
最终退出 0；适用已跟踪/未跟踪文件的 pre-commit 全通过，排除受保护
`examples/native_alltoall.py`；compileall、HEAD diff 空白检查通过。
源码 163 路径与表格集合完全相等，无重复条目。全局目标仍有必需验证缺口，
本批通过不能替代 KCP/MC2/Domino 目标设备及其他未关闭组合的验证。

MoE 编译装配失败的全局配置复核：`parallel/model_compile.py::apply_compile`
在开启 capture_scalar_outputs 后，torch.compile/逐层 compile 装配异常会遗留
该全局标志。现于异常退出恢复进入装配前的值，包括 KeyboardInterrupt；成功
返回仍保留延迟 tracing 所需设置。新增 4 项故障注入回归覆盖初始 true/false
及 RuntimeError/KeyboardInterrupt。编译模块测试 34 passed、2 skipped；
累计完整 CPU suite 3233 passed、2 skipped、206 warnings，退出 0。
Ruff、适用全仓 pre-commit、compileall、HEAD diff 检查通过。
限制：此修复针对同步装配异常，不回滚已编译的部分 block，不解决成功 Trainer
共存时的全局 Dynamo/Inductor 配置所有权或延迟 tracing 失败；继续保留该缺口。

逐层编译事务边界复核：PyTorch Module.compile 原地写入
`_compiled_call_impl`，后续 block 装配失败时会留下部分编译模型。
`apply_compile` 现在在每次调用前保存原 callable，异常时逆序恢复全部已尝试
block，包含抛出异常的 block；保留已有编译 callable、Module/Parameter 身份
及 state-dict 键。新增 6 项回归覆盖三层失败位置和 RuntimeError/KeyboardInterrupt，
编译模块测试 40 passed、2 skipped。成功路径保持原行为。
此事务仅处理同步 compile 入口装配，不声称恢复延迟 tracing 的全局编译缓存，
async TP 注册和成功模型之间的全局开关所有权仍需继续审查。
适用全仓 pre-commit、compileall、HEAD diff 检查通过；完整 CPU suite
3239 passed、2 skipped、203 warnings，最终退出 0。

Regional Inductor 失败回滚复核：backend 选择会改写模型及所有子模块的
`_regional_inductor_enabled`。同步编译装配失败时现在恢复各模块先前值及属性
是否存在，避免重试/复用模型继承失败尝试的执行路径。backend 选择自身也纳入
异常边界；新增 inductor/aot_eager 两条失败回归，混合 true/false/无属性。
编译模块测试 42 passed、2 skipped；适用全仓 pre-commit、compileall、
HEAD diff 检查通过。完整 CPU suite 3241 passed、2 skipped、208 warnings，
退出 0；async TP 注册与成功模型的全局
配置所有权仍未由本批解决，不宣称完整 Trainer 事务回滚。

异步 TP 编译失败的 pass 状态复核：`apply_compile` 在 setup 前保存
Inductor `_micro_pipeline_tp`，将 async setup 纳入现有失败边界；注册阶段或后续
compile 入口抛错时恢复原值。新增 4 项故障注入覆盖初始 true/false 与两个阶段。
编译模块测试 46 passed、2 skipped；Ruff、适用全仓 pre-commit、compileall、
HEAD diff 检查通过；完整 CPU suite 3245 passed、2 skipped、212 warnings，
最终退出 0。
核对本机 PyTorch 2.14 `_symmetric_memory.enable_symm_mem_for_group`：注册记录
归 PyTorch 进程/通信组所有，重复调用幂等，目前无对称注销 API；本项目不直接
删除私有 store 或 C++ group 信息。失败回滚只恢复 Inductor pass，不承诺注销
PyTorch 注册或解决成功 Trainer 共存的全局设置隔离。设备 overlap 未验证。

归一化构造设置的重复调用复核：native→aten 留下原 view hook，aten→native
未解除实例 forward，导致配置与实际后端不一致。现重设前只移除本模块拥有的
view hook（含 kwargs 索引），恢复 native 时仅解除自己绑定的 aten forward；
用户 hook 保留。新增混合后端六次重复设置回归，验证 hook、forward 与配置
一致性。HF wrapper suite 77 passed；适用全仓 pre-commit、compileall、
HEAD diff 检查通过；完整 CPU suite 3246 passed、2 skipped、205 warnings，
最终退出 0。
契约仍为构造期、并行变换/compile 前设置；不支持已编译或已分片模型热切换。

生产 RMSNorm 的 DTensor 数值闭环：新增独立脚本
`tests/integration_tests/aten_norm_dtensor_equivalence.py`，直接调用生产
`llmtuner::rms_norm`，以原生 HF Qwen3RMSNorm autograd 为独立参考。
真实 2-rank Gloo eager 与 Inductor、4-rank 2×2 mesh Inductor 均退出 0；
比较完整输出、输入梯度与权重梯度，使用 PyTorch float32 默认容差。
覆盖复制、序列分片、隐藏维分片重分布、双轴序列分片；断言序列分片的权重
梯度为 Partial(sum)、隐藏维输入对应复制输出。编译用例断言 unique_graphs>0。
命令：`PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2
--master_addr=127.0.0.1 --master_port=<空闲端口>
tests/integration_tests/aten_norm_dtensor_equivalence.py`；编译增加
`LLMTUNER_TEST_COMPILE_BACKEND=inductor` 和本机禁用 PCH 的环境变量
`TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0`，四 rank 改 nproc 为 4。
新脚本 pre-commit 与 HEAD diff 检查通过。本批未改生产代码，复用上一批
完整 CPU suite。此结果不涵盖目标设备、混合精度分布式或高阶梯度。

能力矩阵与上游映射一致性复核：将生产 native/aten、真实 DTensor、严格恢复
及同步 compile 失败回滚的当前证据集中到 distributed_implementation_comparison
顶部，明确历史批次状态不代表当前状态。upstream map 删除过期固定模块数，
指向实时逐文件清单；区分本地 torchtitan norm 参考与本项目 HF 归一化适配。
设计文档补充算子注册所有权、配置传递、构造期设置和恢复/融合限制。
仅文档变更，定向 pre-commit 与 HEAD diff 检查通过；复用已通过的生产数值证据。
剩余全局开关共存、目标设备及未关闭组合验证继续开放。

Custom relocated FSDP 混合精度边界复核：新增
`tests/integration_tests/fine_grained_mixed_precision.py`，真实 2-rank CPU/Gloo
比较 custom 两个独立参数单元共用 Block hook 与原生 FSDP Block 边界。
32 配置（参数 bf16/fp16 × reduce fp32/bf16 × output fp32/bf16 × reshard
开/关 × 输入梯度开/关），每配置三次 SGD，输出、输入梯度、完整参数梯度和
更新后参数均通过 PyTorch 默认容差；断言输出 dtype、FP32 master/梯度。
原始 per-Linear native 参考失败：梯度最大差异 0.00439453125，不能直接归因于
custom 错误，因为 native 每个 Linear 的 output cast 让 tanh 在 fp32 执行，
custom relocated hook 让 tanh 在参数 dtype 执行。换为同 Block 计算边界后
通过，未调整容差。原失败日志 `/tmp/llmtuner-fine-grained-mixed-precision.log`
及匹配边界结果日志保留为诊断补充。
命令：`PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2
--master_addr=127.0.0.1 --master_port=<空闲端口>
tests/integration_tests/fine_grained_mixed_precision.py`。脚本 pre-commit、
HEAD diff 检查通过；本批无生产修改，复用已有完整 CPU suite。
不覆盖 AC/compile、DCP、设备 stream 或 overlap；本批结论是相同执行边界下
与原生 FSDP 的混合精度等价，不是所有 hook 迁移均保持原精度轨迹。

Custom FSDP 混合精度 AC/compile 复核：扩展已有
`fine_grained_mixed_precision.py`，支持 `--checkpoint` 与
`--compile-backend aot_eager|inductor`，仅 custom 路径在 FSDP 前应用生产
wrap_full/torch.compile，原生同 Block 边界保持 eager 参考。
真实双 rank CPU/Gloo 下，full AC、aot_eager、full AC+aot_eager、Inductor
四组均覆盖 32 配置、三次更新并退出 0；保留原默认容差与全部梯度/dtype断言，
编译配置额外要求 unique_graphs>0。full AC+Inductor 首配置失败，首个参数梯度
最大绝对误差 0.00439453125；未放宽容差或关闭 AC。无 FSDP/通信的最小隔离
重现：bf16 双 Linear+tanh，Inductor 单独的全部梯度逐位一致；增加生产 wrap_full
后，第一层 weight/bias 梯度误差 0.0078125/0.00390625，输出与末层梯度一致。
该差异在去掉 FSDP 后仍存在，不能归因于 custom 参数组或 collective。
仓库脚本 `--local-probe` 保留可复现比较与逐参数误差，下一步定位 tanh 的
重计算融合与低精度舍入边界，判断其数值契约；失败矩阵保持开放。
新增脚本路径未变，定向 pre-commit 与 HEAD diff 检查通过；本批无生产改动。
DCP 与设备 stream/overlap 不在此结果范围内，不能以小型 Block 替代完整模型矩阵。

AC+Inductor 低精度差异根因闭环：检查 PyTorch 2.14 Inductor config，
`emulate_precision_casts=False` 会省略低精度算子融合间的 downcast/upcast。
独立显式 FP32 tanh 中间值参考与默认 AC+Inductor 的全部梯度逐位一致；
native BF16 tanh 参考在首层 weight/bias 分别相差 0.0078125/0.00390625。
脚本 `--local-probe` 新增上述独立参考及精确梯度断言，避免只打印误差。
设置 `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1` 后，真实两 rank full AC+
Inductor 的完整 32 配置/三次 SGD 矩阵退出 0，保持原生 eager 参考与原断言。
这确认当前失败来自编译重计算的数值策略，不是 FSDP 通信/状态缺陷。
默认融合路径原失败保留；需要 eager 低精度舍入契约时可显式启用此 PyTorch
设置，必须在首次导入/编译前设置。未自动改写全局开关，不推广到所有算子或设备，
性能代价尚未测量。下一步继续 DCP/多运行时生命周期与剩余模块风险。

多 custom FSDP runtime 状态隔离复核：新增
`tests/integration_tests/fine_grained_runtime_isolation.py`，真实双 rank CPU/Gloo
同时保留两个不同 seed 的 relocated-hook 模型，采用不同 reshard/multi-group
策略，交错前向及正反顺序 backward，各完成三次带 momentum 的 SGD。
B forward 尚未反向时注入 A forward 异常并 reset，B 后续反向/更新仍与独立
dense 参考一致；普通与非重入重计算两配置均退出 0。逐次比较输出、归约后
完整参数梯度和更新参数，并检查每个 runtime 收尾、state_ctx 与通信 context
身份不共享。所有数值断言保持 PyTorch 默认容差。
命令：`PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2
--master_addr=127.0.0.1 --master_port=<空闲端口>
tests/integration_tests/fine_grained_runtime_isolation.py`。
脚本 pre-commit 与 HEAD diff 检查通过；本批无生产修改。
限制：此证据是同进程多模型 FSDP 生命周期隔离，不覆盖多个完整 Trainer 的
全局随机数/编译设置、线程并发、目标设备 streams 或不同进程组拓扑。

Allocator history 共享所有权复核：MemoryProfiler 原来每个实例构造都会重启
进程级记录，每个 close 都停止，可能清除或中断其他活动 profiler。
现以带锁的弱 owner 集合共享记录：首 owner 启动、最后 close 停止，关闭失败
保留 owner 供重试；重叠 max_entries 不同在改动 allocator 前明确拒绝。
新增正反关闭顺序与容量冲突回归，Profiler suite 61 passed；Ruff、适用全仓
pre-commit、compileall 与 HEAD diff 检查通过；完整 CPU suite
3249 passed、2 skipped、210 warnings，最终退出 0。
所有 owner 仍须显式 close；不声称接管框架外直接启动的 allocator recorder，
也未验证实际 CUDA/NPU allocator。数据恢复身份读取复查未发现本批新增缺陷。

Allocator snapshot 发布完整性复核：MemoryProfiler 原先直接向最终 pickle 写入，
序列化/读取失败会留下截断文件，同 step 重写也会破坏已有快照。现写同目录
唯一临时文件，关闭成功后 os.replace 原子发布，失败清理临时文件。
新增首次写入与已有完整快照两种中途序列化失败回归，验证无半文件、旧内容保持。
Profiler suite 63 passed；适用全仓 pre-commit、compileall、HEAD diff 检查通过。
完整 CPU suite 3251 passed、2 skipped、220 warnings，最终退出 0。
此保证为文件可见性，不承诺断电持久性；设备 allocator
本身仍需目标设备验证。关闭后的 step 不再读取快照，现有实现保持该行为。

Metrics 关闭失败重试复核：LoggerContainer 原先关闭前清空 owner 列表，
失败 sink 因而失去引用。现继续关闭所有 sink，同时保留失败项供下次 close；
成功项不再重复关闭。沿调用链修复 Trainer.close 对失败 metrics owner 的
丢弃，其他资源仍完成清理。更新 container 回归并新增 Trainer 故障注入，
覆盖原异常身份、失败项保留、成功项去除与后续幂等重试。
初轮 metrics+trainer suite 184 passed；最终完整 suite 3252 passed、2 skipped、
220 warnings，退出 0。适用全仓
pre-commit 后补充修改的四个文件定向门禁通过，compileall、HEAD diff 通过。
不保证 sink 自身能重试任意外部故障；本改动确保所有权不会在首次错误时丢失。

WandB run 引用复核：原 logger 调用模块级 wandb.log/finish，其他初始化改变
当前 run 后可能写入或关闭错误 run。现保存 init 返回的 run 并使用该实例，
finish 成功后清空引用，失败仍可重试；关闭后 log 明确报错。
新增替身回归：模块当前 run 与 init 返回值不同，验证只有返回 run 收到日志
与关闭，另一个 run 不变；既有前缀/关闭测试改为实例契约。Metrics suite
74 passed；完整 CPU suite 3253 passed、2 skipped、226 warnings，退出 0。
适用全仓 pre-commit、compileall、HEAD diff 通过。本批未连接 WandB 服务。
限制：保持 WandB init 默认复用策略；若 init 本身返回共享 run，仍不能据此
宣称多个 logger 各有独立运行。真实外部服务行为及复用所有权策略继续开放。

Scheduler 共同进度契约复核：容器公开内层 scheduler，但 step 原先只检查
各自上限；一个内层被单独推进后，容器继续推进不一致 epoch，直到 checkpoint
才暴露错误。现复用 state_dict 的共同 epoch 校验，在改动任何 lr 前拒绝。
新增真实两 optimizer 回归，断言失败后 scheduler 状态与 lr 不变；恢复共同
进度后容器继续正常更新。scheduler suite 44 passed；适用全仓 pre-commit、
compileall、HEAD diff 检查通过；完整 CPU suite 3254 passed、2 skipped、
209 warnings，最终退出 0。
此修复防止已有进度分歧继续扩散，不保证任意用户 lambda/optimizer 异常的
全更新事务回滚，也不改变训练中 optimizer→scheduler 的既定顺序。

EMA 固定 scope 元数据复核：对象身份不变的 buffer resize 或 .data dtype
变更可绕过原身份校验，参数组先更新后 buffer 组失败，留下部分平均值更新。
tracked_averages 现统一校验 live/average shape、dtype、device，所有组预检
完成后才进入更新/cold-start 复制。新增 4 项 shape/dtype × step/cold-start
回归，断言失败前后全部平均值逐位保持。EMA suite 72 passed；适用全仓
pre-commit、compileall、HEAD diff 检查通过；完整 CPU suite 3258 passed、
2 skipped、211 warnings，最终退出 0。
本批不支持运行时迁移已构造 EMA 的 scope/dtype/device；报错要求模型最终
确定后重建。DTensor placement 变化及任意底层 kernel 故障不在此预检保证内。

多运行时 distributed EMA 数值复核：扩展
`fine_grained_runtime_isolation.py`，两个同时存活的 custom FSDP 模型各有独立
生产 EMA，fixed decay=0.5；每次 SGD 后与各自独立 dense lerp 平均值比较。
沿用不同 reshard/multi-group、交错前后向及一方 forward 失败 reset 场景。
真实两 rank CPU/Gloo，普通与非重入重计算均完成三次更新并退出 0；比较
完整 DTensor EMA 值及原有输出/梯度/参数/收尾断言，默认容差保持。
命令沿用该脚本 torchrun 两 rank IPv4 启动；定向 pre-commit 与 HEAD diff
检查通过。本批仅修改集成脚本，复用上一批完整 CPU suite 结果。
验证支持既有固定布局下的独立 EMA 状态，未证明动态 placement/mesh 变更、
不同通信组或完整 Trainer 全局状态隔离；未根据疑点添加额外生产限制。

冷 optimizer 状态初始化失败复核：init_optim_state 原先只恢复 lr/grad，
Adam 分配中途失败或 step post-hook 失败可留下非空 partial state；重试据此
跳过初始化，后续保存/更新损坏。现异常时删除本次冷参数的新状态，恢复原有
空 state 容器，已存在的非空状态不进入该清理范围。
新增真实 AdamW 分配与 post-hook 失败回归，验证无 partial state、参数不变、
lr/grad 保持，重试得到 step0 且首次真实更新为 step1。state-dict suite
47 passed；适用全仓 pre-commit、compileall、HEAD diff 通过；完整 suite
3260 passed、2 skipped、216 warnings，最终退出 0。
该恢复不回滚用户 hook 自行修改的模型或任意外部状态，不承诺任意 optimizer
kernel 的权重事务；本批修复初始化新状态与重试契约。

MoE 路由构造边界复核：TokenChoiceTopKRouter 原先接受 NaN/Inf route_scale
并在所有选中分数上直接相乘，导致无效训练输出；非 bool route_norm/debug
值如字符串 false 会按 truthiness 启用错误路由策略。现 gate 分配前要求有限
实数 scale 与实际 bool 开关，quantile 路径复用同一校验，未限制合法有限缩放。
新增 16 项回归，含非法 scale 不进入 gate 分配；router/padding suite
116 passed。适用全仓 pre-commit、compileall、HEAD diff 通过；完整 suite
3276 passed、2 skipped、217 warnings，最终退出 0。
已阅读空 token、padding 筛选和 quantile recompute 抑制路径，本批未确认新的
空 token 缺陷；不将构造参数校验解释为运行时任意非有限 tensor 的检测。

TP 自定义 realizer 构造失败复核：原 apply_tp 在每次构造后立即安装 dense
投影，且先修改 MoE；后续构造器抛错会留下部分分片模型。新增回归在原实现
实际失败（首个 Linear 已变为 ColumnParallelLinear）。现先准备全部 dense
替换对象，成功后才执行 MoE 变换和安装，保持既有 attention 边界与训练模式。
回归验证原模块/参数身份和值不变、MoE 变换未触发，以及修正计划后可重试。
这不是任意构造器副作用、MoE 内部失败或全局 symmetric-memory 注册的事务回滚。

验证：`pytest -q tests/unit_tests/cpu/parallel/test_tp.py
 tests/unit_tests/cpu/parallel/test_tp_moe.py --disable-warnings`：70 passed。
真实两 rank IPv4 CPU/Gloo `tests/integration_tests/tp_equivalence.py` 退出 0，
14 个投影完成替换，输出最大绝对差 2.980e-08、梯度 1.937e-07，原容差保持。
完整 `pytest -q --disable-warnings`：3277 passed、2 skipped、201 warnings，
39.04s，退出 0；适用 tracked/untracked 全仓 pre-commit 通过，保护文件
examples/native_alltoall.py 排除；compileall 与 HEAD diff 检查通过。
日志：`/tmp/llmtuner-tp-preparation-{units,equivalence,precommit}.log`。
下一步继续核对 MoE 变换失败路径和 pattern 生命周期；CUDA/NPU 与完整并行
组合验证仍未关闭，本批 CPU 证据不外推目标设备行为。全局目标未完成。

MoE shared expert 重复 TP 变换复核：resolve_tp_targets 原先只排除本次
待变换 MoE 子树；已带 _tp_moe_boundary 的块仅计数，其 plain Linear shared
投影在第二次调用时会被 dense realizer 再次分片。扩展现有 shared expert
回归，原实现实际失败（原 Linear 变为 ColumnParallelLinear）。现保留已变换
块的路径，并与新块一起排除 dense 命中，维持同一拓扑下重复应用的身份和值。
回归比较全部模块、参数身份与精确参数值；TP/MoE 定向 suite 70 passed。
本批不增加动态 mesh、TP 度数变更支持，不改变 collective 或分片数值公式；
多设备 shared expert 重复应用数值验证仍开放。
完整 CPU suite 3277 passed、2 skipped；适用全仓 pre-commit、compileall、
HEAD diff 检查通过。日志 `/tmp/llmtuner-tp-moe-reapply-units.log` 和
`/tmp/llmtuner-tp-moe-reapply-precommit.log`。下一步核对跨块共享参数及模型
变换后的布局身份契约，目标设备与组合矩阵缺口保持未完成。

根 MoE TP 路径复核：以 MoE block 自身作为 apply_tp 入口时，根路径为空，
旧子树排除前缀构造成 '.'，未排除 shared expert dense 命中。扩展 shared
expert 回归覆盖 root/nested 两种入口，root 在旧实现实际失败。现空路径使用
空前缀，覆盖全部后代，首次及再次应用均保持单次 feature shard。
定向 TP/MoE suite 71 passed；完整 CPU suite 3278 passed、2 skipped、
213 warnings，47.17s，退出 0。适用全仓 pre-commit、compileall、HEAD diff
通过；保护文件排除。
额外两 rank CPU/Gloo 诊断 `/tmp/llmtuner-root-moe-tp-check.py` 直接调用生产
apply_tp 与实际边界 collectives，使用测试 MoE/shared MLP 构件及未分片参考。
固定路由下首次/重复变换输出、输入梯度、shared expert 梯度全部通过严格
float64 容差，日志 `/tmp/llmtuner-root-moe-tp-fixed-routing.log`，退出 0。
首次随机路由诊断在输入梯度严格 FP64 容差失败，rank1 最大绝对差约
3.66e-4；构件内部路由强制 FP32，尚未独立定位差异，原日志
`/tmp/llmtuner-root-moe-tp-equivalence.log` 保留，不将固定路由结果解释为该
失败已关闭或完整 HF 随机路由数值通过。没有修改生产路由或放宽断言。
后续：用独立路由梯度参考定位随机路由差异；跨块共享参数、目标设备和组合
矩阵仍待验收。全局目标保持未完成。

随机 MoE TP 输入梯度差异定位：新增自包含两 rank 集成脚本
`tests/integration_tests/tp_moe_equivalence.py`，直接手动切分 dense 参考的
packed gate/up、down 和 shared expert 权重，独立构造 partial expert 图，
不调用生产 sharding/边界 collective 构造参考。partial 梯度按 rank 求和后，
生产 TP 的首次/重复应用输入梯度在 rtol=1e-9、atol=1e-10 下匹配；输出与
shared expert 梯度继续严格对照 dense。参考 oracle 编写时一度遗漏 top-k
专家累加，诊断失败后已纠正，未改生产代码或放宽断言。
FP32 路由模式下，独立 partial 与 dense 输入梯度重现同一最大绝对差
3.65791686e-4；`--router-fp64` 在相同种子/权重/输入下还通过 dense 输入
梯度严格比较（误差见日志）。因此上批随机路由差异已有计算顺序/FP32
路由舍入的独立证据，不据此修改生产全局 dtype、通信或容差。
真实 CPU/Gloo 两 rank，脚本默认与 --router-fp64 均退出 0，日志
`/tmp/llmtuner-tp-moe-{fp32,fp64}-oracle.log`。沿用 IPv4 torchrun 启动。
定向脚本 pre-commit、HEAD diff 检查通过；生产代码未修改，复用上一批
全量 CPU 3278 passed/2 skipped 证据。本脚本不覆盖 router 参数梯度、
全部 routed expert 参数梯度及真实 HF/设备后端，这些能力不据此验收。
下一步离开该已定位诊断，推进全局台账剩余训练生命周期与设备验证缺口。

可选 checkpoint 后端构造失败复核：TorchCheckpointingManager 在 backend
manager.build 成功后若 retention 线程启动失败，_initialized 尚未置位，
析构调用 public close 会直接返回，导致 manager 无清理。现对后续本地服务
构造与启用日志增加失败清理边界，显式调用 _close，清理异常记录后保留原始
构造异常。未改变可选依赖懒加载与正常存储策略。
新增 4 项 RuntimeError/KeyboardInterrupt × cleanup success/failure 回归，
原实现确认 calls=[] 失败；修复后保留原异常身份，backend.close 恰调用一次，
未初始化实例的再次 public close 不重复释放。backend 使用替身，不声称真实
第三方进程/存储资源已验证。Checkpoint suite 112 passed；全量 CPU suite
3282 passed、2 skipped、205 warnings，38.45s，退出 0。适用全仓 pre-commit、
compileall、HEAD diff 通过，保护文件排除。
来源核对：torchtitan 01b1b69e3 的
components/checkpointer/torch_checkpointing.py 同样先 build 再启动线程；
本项目增加显式失败清理，并修正模块溯源路径与过时的测试/性能表述。
日志 `/tmp/llmtuner-backend-construction-{units,precommit}.log`。
全局剩余：真实可选包 I/O/子进程生命周期、目标设备及组合矩阵未验收；继续
检查其他训练生命周期契约，不因本批替身测试关闭硬件或后端缺口。

临时 checkpoint backend 生命周期复核：HF 加载与最终同步保存都在 finally
直接 close 临时 manager，操作失败并伴随清理失败时，原 RuntimeError 或
KeyboardInterrupt 被 OSError 覆盖。新增两路径回归在旧实现确认失败。
现将共同资源所有权合并为 temporary_backend_manager context manager：
操作失败仍尝试 close，清理错误记录而原异常身份保持；操作成功时 close
错误直接传播，避免将未成功收尾报告为成功。manager.build 失败不虚构已获得
的资源，后续状态装载/GC 仅在正常关闭后执行。
新增 6 项 load/save × RuntimeError/KeyboardInterrupt/success 回归，验证原始
异常或 cleanup 异常身份与调用顺序。Checkpoint suite 118 passed；完整 CPU
suite 3288 passed、2 skipped，最终退出 0；适用全仓 pre-commit、compileall、
HEAD diff 通过，保护文件排除。日志
`/tmp/llmtuner-temporary-backend-{units,precommit}.log`。
本批只验证本地控制流与 backend 替身，未运行真实可选包或硬件通信。继续
保留其存储、子进程、完整恢复及设备矩阵缺口；全局目标未完成。

TP MoE 全参数梯度缺口补验：扩展 tp_moe_equivalence.py 的独立手动分片
oracle，计算全部 reference 参数梯度，按 packed gate/up、down、shared
布局抽取每 rank 目标。直接调用生产 Trainer._allreduce_replicated_tp_grads
（最小 runtime 提供真实 TP mesh 与 model_parts），验证仅复制 router 被
SUM，routed/shared shards 不被重复归约；首次/再次 apply_tp 后全部参数梯度
匹配独立参考。随后真实 SGD lr=0.001 更新所有参数，与独立预期更新比较。
FP64 router 额外对照 dense router 参数梯度；FP32 router 则对照独立 partial
归约 oracle，保留已定位的 FP32 舍入边界，未放宽已有断言。
两 rank CPU/Gloo 默认及 --router-fp64 均退出 0；日志
`/tmp/llmtuner-tp-moe-full-grad-{fp32,fp64}.log`。适用全仓 pre-commit 通过，
HEAD diff 通过；仅修改集成测试与文档，复用前批完整 CPU 3288 passed/
2 skipped 结果。并行能力矩阵新增对应具体配置行。
本批关闭该 fixture 的全部参数梯度与单次 SGD 验证缺口，不代表完整 Trainer
训练、真实 HF block、FSDP/AC/compile 组合或设备后端已验证。全局目标未完成。

验证输入初始化协调复核：run_validation_loop 原先在 fetch_batch 的 world
协调前直接创建 loader、分配 token tensor、调用 iter；单 rank 本地异常会使
其他 rank 进入无人匹配的 fetch collective。现将这些无 collective 的本地
操作放入 synchronize_local_errors，外部 finally 保留并关闭成功创建的 loader；
构造未返回 loader 的 rank 不访问未赋值资源，原异常与 cleanup 优先级保持。
扩展 data_fetch_coordination.py，两 rank CPU/Gloo 分别注入 rank0 构造与
iter 错误，rank0 保留原异常、rank1 收到阶段错误；已创建 loader 恰关闭一次，
双方后续 barrier 正常。既有耗尽/decode/malformed 场景全部通过，脚本退出 0。
Validation 单测 66 passed；完整 CPU 3288 passed、2 skipped、211 warnings，
41.30s，退出 0。全仓 pre-commit 首轮发现新增测试闭包 B023，已显式绑定
循环变量并定向重跑通过；其他全仓 hooks 通过。compileall、HEAD diff 通过。
日志 `/tmp/llmtuner-validation-initialization{,-units,-precommit}.log`。
本协调不处理阻塞 loader、进程退出或 loader 内部 collective 失败；未来引入
此类初始化通信必须重新划分本地边界。目标设备及完整组合验证保持未完成。

验证 loader 收尾协调复核：成功验证循环后，单 rank loader.close 异常会
在 finalize_validation 的 loss collective 前退出，其他 rank 仍进入汇总。
现仅在循环成功路径对本地 cleanup 做 world 错误协调；已失败路径沿用本地
close 与原异常优先规则，避免故障展开时添加不匹配通信。此边界要求 loader
close 本身不含 collective，不能处理阻塞、进程丢失或模型通信内部故障。
扩展 data_fetch_coordination.py，rank0 close 注入 OSError，rank0 保留原异常，
rank1 收到 Validation input cleanup 阶段错误；两 rank 都断言 finalize
未调用、各 close 一次，并成功进入后续 barrier。两 rank CPU/Gloo 全场景
退出 0；Validation 单测 66 passed。完整 CPU suite 3288 passed、2 skipped，
最终退出 0；适用全仓 pre-commit、compileall、HEAD diff 通过，保护文件排除。
日志 `/tmp/llmtuner-validation-cleanup{,-units,-precommit}.log`。
本批关闭成功循环后的清理错误传播缺口；目标设备与完整验证并行组合仍待验收。

验证结果发布协调复核：finalize_validation 中无 loss 的 PP 阶段原先直接
返回，末阶段非有限 loss 或 metrics sink 异常不能传给这些阶段。现预检与
发布分别协调本地错误，loss all_reduce 保持在协调边界外；无 loss 阶段同样
参与发布协调。预检覆盖空 batch/token 与本地 loss clone，失败时不进入汇总。
扩展两 rank data_fetch_coordination.py，rank0 持 loss、rank1 无 loss，注入
sink OSError、NaN 和 zero batches，原异常在故障 rank 保持、另一 rank 收到
阶段错误，后续 barrier 成功；全脚本退出 0。此为实际 world 协调测试，未运行
PP schedule 本身。Validation suite 66 passed，完整 CPU 3288 passed、2 skipped、
217 warnings，39.05s，退出 0；适用全仓 pre-commit、compileall、diff 通过。
四 rank validation_reduction_equivalence.py 首轮因历史 __new__ fixture 缺失
checkpointer 字段失败；补齐 checkpointer=None 后 DP×CP 与 DP×TP 均与完整
batch 独立 loss 参考匹配，退出 0，未放宽数值断言。最终定向 pre-commit 通过。
日志 `/tmp/llmtuner-validation-reporting{,-units,-precommit}.log` 与
`/tmp/llmtuner-validation-reporting-reductions{,-retry}.log`。
约束：metrics.log_validation 应为本地 sink；通信内部故障、阻塞或 rank 丢失
不在此协调保证内，目标设备和完整 PP 验证仍待验收。全局目标未完成。

真实 PP 验证发布故障补验：扩展 pp_real_corpus_equivalence.py，在正常有限
验证后追加末阶段 sink OSError，实际运行生产 Trainer.validate 与 pipeline
forward-only schedule。末阶段保留原异常身份，其他 stage 收到 reporting
阶段错误；逐 module training flag 与 CPU RNG 精确恢复，各 rank barrier 成功。
原有三次训练 loss、最终参数及有限验证完整模型参考断言保持，无容差调整。
CPU/Gloo IPv4 torchrun 两配置退出 0：PP2/CP1/native、两 rank，seq64，最终
参数最大差 3.725e-09，验证参考 4.180420；PP2/KV-CP2/custom、四 rank，
seq256，参数最大差 3.960e-06，验证参考 4.163111，容差均 1e-5。
两者均真实 1F1B、packed local_jsonl、Grain 数据与最终含 padding 的有限验证；
CPU 明确以 dense-mask SDPA 替代 attention 内核，不外推生产 flex 或设备。
日志 `/tmp/llmtuner-pp-validation-reporting.log` 与
`/tmp/llmtuner-pp-cp-custom-validation-reporting.log`；后者启动环境
LLMTUNER_TEST_CP_SIZE=2、LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom，均设
PYTHONPATH=.、GLOO_SOCKET_IFNAME=lo0、TORCHDYNAMO_DISABLE=1。
仅集成测试与文档修改，定向 pre-commit/diff 通过，复用前批完整 CPU suite。
这补齐对应实际 PP schedule 的发布错误证据，未验其他 schedule/硬件组合。

合成数据构造 seed 契约复核：RandomTokenSource 原先不检查 seed，bool 被
隐式接受、float/str/None 延迟至首次生成报错，越界整数通过 modulo 静默映射，
与训练 seed_everything 的整数/范围契约不一致。读取 HEAD 版本独立执行确认
上述六种非法输入构造均被接受。现 source 构造时要求非 bool Python int，
范围 [-2**63, 2**64-1]，loader 复用；合法 seed 的生成公式和状态格式保持。
新增 12 项非法 seed × source/loader 回归及 4 项合法边界确定性恢复回归，
恢复后 input/labels 精确匹配。数据管线 suite 335 passed；完整 CPU suite
3304 passed、2 skipped、233 warnings，37.92s，退出 0；适用全仓 pre-commit、
compileall、HEAD diff 通过，保护文件排除。日志
`/tmp/llmtuner-synthetic-seed-{units,precommit}.log`。
该校验属于数据构造输入契约，不外推跨设备 RNG 或完整训练恢复等价。
全局目标未完成，后续继续处理台账中实现与验证缺口。

Tokenizer 回退词表 ID 契约复核：vocab.json 原先直接传入 WordLevel/BPE，
后端实际接受 {'[UNK]':0,'hello':3} 并报告 vocab size=2、hello ID=3，也接受
两个 token 共用 ID=0。这与本 wrapper 的 get_vocab_size/追加 token ID 按数量
分配契约不一致。现回退词表构造前要求非空 token→整数 ID 映射，排除 bool，
ID 必须唯一且从零连续；JSON 键顺序不受限。tokenizer.json 快速加载路径不变，
本批不声称修复所有序列化后端词表或支持稀疏词表。
新增 7 项非法映射/ID 布局及 1 项乱序合法映射回归，实际 hello 编码保持 [1]。
Tokenizer suite 47 passed，完整 CPU suite 3312 passed、2 skipped、215 warnings，
35.78s，退出 0。适用全仓 pre-commit、compileall、HEAD diff 通过；保护文件排除。
日志 `/tmp/llmtuner-vocab-layout-{units,precommit}.log`。
本批另核对当前生产源码与逐文件表：163 文件/163 行，无遗漏/失效/重复；
该覆盖核对不等于全部运行时或目标设备验收。全局目标仍未完成。

当前工作区 DCP 保存进程恢复复核：执行两 rank CPU/Gloo
`tests/integration_tests/fsdp_checkpoint_equivalence.py --process-save-cpu`，
IPv4 torchrun，PYTHONPATH=.、GLOO_SOCKET_IFNAME=lo0。native/custom ×
shard/replicate 来源与恢复目标共 8 配置全部完成，最终退出 0。实际 PROCESS
executor 与可序列化 generation writer 在保存子进程写 shard 后注入 rank0
OSError，所有 rank 收到 CheckpointException；fresh runtime 从原发布 step2
恢复并续训到 step6，恢复后更新前模型、AdamW 完整状态精确比较通过。
连续/恢复 loss、全部梯度、最终参数、Adam moments/counters、scheduler/LR
均通过；同拓扑 atol=0，跨拓扑和独立 dense oracle atol=1e-5，原断言保持。
日志 `/tmp/llmtuner-dcp-process-save-audit.log` 含全部 8 个终态场景结果。
本批仅补充当前状态的真实后台保存证据，未修改生产代码或测试，文档门禁通过；
复用前批 CPU 3312 passed、2 skipped 的适用结果。使用 CPU shared staging，
明确禁用 pinning、async device staging、nonblocking copy；多 manager 并发、
设备 pinned transport、低精度和 relocated hook 的完整 DCP 仍未由此证明。
下一步优先推进未覆盖组合/生命周期，不将该已通过矩阵重复扩展为全局完成。

Custom relocated FSDP 混合精度 DCP 恢复闭环（2026-10-15）：
- 新增 `tests/integration_tests/fine_grained_checkpoint_equivalence.py`，真实两rank
  CPU/Gloo，BF16/FP16 compute、FP32 master/reduction × reshard false/true ×
  full AC false/true，共8配置；Linear参数组hooks迁移到共同Block。
- 生产 OptimizersContainer AdamW、scheduler、EMA与同步DistributedCheckpointManager：
  两步后保存，以不同seed构建新运行时恢复，更新前逐位比较模型及全部服务状态；
  相同输入继续step3，输出、完整参数梯度、更新后模型及服务状态逐位一致。
- 命令：`PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2
  --master_addr=127.0.0.1 --master_port=<空闲端口>
  tests/integration_tests/fine_grained_checkpoint_equivalence.py`（连接为一条命令）。
  session16032已取得终态，退出0；日志 `/tmp/llmtuner-relocated-mixed-dcp.log`，
  每rank各8条PASS。该脚本通过ruff及pre-commit；生产代码未因本批修改。
- 边界：同步同拓扑CPU恢复等价，不是独立dense正确性参考；未覆盖目标设备、
  compile、异步transport、拓扑变化、数据/RNG状态或完整Trainer装配。

WandB 初始化复用所有权修复（2026-10-15）：
- 继续关闭前述默认复用缺口。真实WandB 0.28.0离线复现：已有run时默认
  init返回同一对象，finish返回值会结束外部run；原实例引用修复不足以隔离所有权。
- `components/metrics.py` 显式 `reinit="create_new"`，每个logger持有独立run，
  保留环境指定实验身份与实例log/finish契约。需要SDK支持该策略；不回退到
  finish_previous或return_previous。参考torchtitan/observability/metrics.py，
  本仓库主动隔离嵌入式使用的run生命周期，不沿用其模块级log/finish。
- 新增有/无外部run的初始化回归；metrics suite 76 passed。真实SDK离线验证
  返回不同run，logger关闭及重复关闭后外部run仍存活并能log，退出0。
  未连接远端服务，未验证resume/fork服务端行为或旧版SDK。
- 完整CPU suite最终结果见 `/tmp/llmtuner-wandb-ownership-units.log`，退出0。
- 适用全仓pre-commit退出0（排除受保护examples/native_alltoall.py），
  日志 `/tmp/llmtuner-wandb-ownership-precommit.log`；compileall与HEAD diff通过。

Trainer 构造失败的执行策略回滚（2026-10-15）：
- 独立进程故障注入确认：seed_everything后装配抛ValueError，原构造清理虽
  释放资源，deterministic/anomaly/check_nan及种子/设备环境设置仍泄漏。
- `utils/seeding.py::rollback_execution_settings_on_error` 保存本次播种会改变的
  deterministic/warn_only、cuDNN deterministic/benchmark、未初始化内存填充、
  anomaly/check_nan及四个环境变量；Trainer构造异常时在资源清理后恢复，
  保留原异常。成功构造保留运行设置，避免改变后续训练与lazy compile语义。
- 新增ValueError/KeyboardInterrupt回归：已有值与缺失环境变量均精确恢复，
  原异常身份不变；另验证成功构造保留策略。原部分资源清理故障测试继续通过。
  初轮Trainer suite 114 passed；新增成功路径由完整suite覆盖。
- 最终完整CPU suite 3317 passed、2 skipped、211 warnings，退出0；日志
  `/tmp/llmtuner-setting-rollback-units.log`。适用全仓pre-commit退出0，
  排除受保护文件；compileall及HEAD diff通过，台账追加后定向格式检查通过。
- 限制：本批只关闭失败构造的播种策略泄漏；不宣称RNG流、成功Trainer.close、
  compile全局开关或并发运行时所有权已恢复。下一批继续处理这些状态，不能
  用本次回滚证明完整全局状态隔离。目标设备环境变量仅验证值恢复，未证明
  已初始化通信库能重新读取变量。

Trainer 构造失败的主机 RNG 回滚（2026-10-15）：
- 新增4个回归，在原执行策略回滚实现上先运行，4 failed：装配播种并消耗
  Python/NumPy/CPU随机数后失败，清理即使继续消耗随机数，调用方原序列均应恢复。
  覆盖ValueError/KeyboardInterrupt × 清理成功/失败；保留原装配异常身份。
- 扩展 `utils/seeding.py::rollback_execution_settings_on_error`，进入时快照
  三类主机默认RNG，构造异常时在资源清理后恢复；成功路径保留训练播种结果。
  修复后Trainer suite 119 passed，29 warnings，退出0。
- 失败证据 `/tmp/llmtuner-rng-rollback-before.log`；修复后定向日志
  `/tmp/llmtuner-rng-rollback-target.log`。不涉及私有loader generator。
- 设备RNG仍待处理：CUDA manual_seed_all使用_lazy_call(seed_all=True)，
  未初始化设备有延迟播种队列，简单快照当前设备不足以撤销所有设备播种。
  本批无设备数值证据，不宣称全生命周期、设备或并发RNG隔离完成。
- 最终完整CPU suite 3321 passed、2 skipped、208 warnings，退出0；日志
  `/tmp/llmtuner-rng-rollback-units.log`。适用全仓pre-commit退出0，日志
  `/tmp/llmtuner-rng-rollback-precommit.log`，排除受保护文件；compileall及
  HEAD diff通过，追加台账后定向门禁通过。

Trainer 后续装配失败的 compile 策略回滚（2026-10-15）：
- `apply_compile`既有回滚仅覆盖函数内部错误；生产compile返回成功、后续
  dataloader等装配失败仍泄漏scalar capture及async TP全局设置。
  新增ValueError/KeyboardInterrupt用例在修复前均失败，日志
  `/tmp/llmtuner-compile-policy-before.log`，未将该失败归为编译器误差。
- `parallel/model_compile.py::rollback_compile_settings_on_error`在进入时保存
  Dynamo capture_scalar_outputs与Inductor _micro_pipeline_tp，异常恢复原值；
  Trainer仅在cfg.training.compile开启时懒导入并包围整个装配和失败清理。
  成功运行保留lazy tracing策略，不改变现有apply_compile内部回滚。
- 测试实际调用生产apply_compile、torch.compile eager backend并成功返回，
  随后注入装配异常；MoE发现用替身触发scalar配置，async TP只注入开关，
  不声称真实TP注册、通信或编译前后向已由该测试验证。
- 剩余：成功Trainer退出的策略所有权、并发运行时、设备RNG与symmetric-memory
  注册生命周期仍开放；本批不撤销PyTorch拥有的group注册。
- Trainer与compile专项167 passed、2 skipped；完整CPU suite 3323 passed、
  2 skipped、209 warnings，退出0。日志 `/tmp/llmtuner-compile-policy-target.log`
  和 `/tmp/llmtuner-compile-policy-units.log`。适用全仓pre-commit退出0，
  排除受保护文件；compileall、HEAD diff及追加台账后的定向门禁通过。

TP 跨MoE共享参数的变换前校验（2026-10-15）：
- 新增共用experts模块、单个packed Parameter、外部参数别名三类回归，修复前
  3 failed：逐块切片会重复切片共享模块，或替换参数后静默解除共享关系。
  失败日志 `/tmp/llmtuner-tp-shared-moe-before.log`。
- `tensor_parallel.py::apply_tp`复用全模型remove_duplicate=False参数路径表，
  在任何dense构造或MoE变换前检查待切片expert/shared-expert参数别名。
  与既有dense共享参数边界一致，明确拒绝无法保留身份的布局；报错包含全部路径。
  回归检查所有参数身份、数值与block类保持不变，不以部分变换失败收尾。
- TP专项74 passed；真实两rank CPU/Gloo生产MoE前后向/完整梯度/更新参考脚本
  退出0，日志 `/tmp/llmtuner-tp-shared-moe-distributed.log`。该fixture无跨block
  tied专家；不将常规TP通过视为共享参数支持完成。
- 共享参数的alias-preserving TP仍为实现缺口，本批仅防止静默破坏；EP路径
  及目标设备共享布局未验收，不新增组合支持声明。
- 完整CPU suite 3326 passed、2 skipped、221 warnings，退出0；日志
  `/tmp/llmtuner-tp-shared-moe-units.log`。适用全仓pre-commit、compileall、
  HEAD diff通过；保护文件排除，追加文档后定向门禁通过。

真实两rank Trainer构造失败状态回滚验证（2026-10-15）：
- 扩展 `trainer_initialization_coordination.py`，通过training配置开启deterministic
  与detect_anomaly，分别在model config、模型实例、dataloader、metrics装配注入
  单rank故障。各rank独立保存并精确比较Python/NumPy/CPU RNG、确定性与
  anomaly/check_nan、cuDNN/填充策略及四个环境变量；健康peer同样必须恢复。
- 原始异常身份、peer错误语义、资源关闭和后续barrier保持检查；最终同进程
  新建真实FSDP Trainer执行更新，loss有限，两rank均打印fresh update passed。
- 首跑退出1为测试错误：给LLMTunerConfig只读代理属性赋值；修正为training
  字段。第二次回滚与协调断言全部通过，但最终更新遭Inductor预编译头mtime
  错误退出1；日志 `/tmp/llmtuner-initialization-rollback-distributed-retry.log`。
- 最终两rank CPU/Gloo，使用本机既有环境例外
  `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0`，生产配置不变，退出0。
  命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0
  TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=2
  --master_addr=127.0.0.1 --master_port=<空闲端口>
  tests/integration_tests/trainer_initialization_coordination.py`（连接为一条命令），
  日志 `/tmp/llmtuner-initialization-rollback-distributed-final.log`。
- 定向pre-commit及HEAD diff通过。本批仅集成脚本与文档变化，复用上一批
  完整CPU 3326 passed、2 skipped证据；不宣称设备RNG、并发或死rank恢复。

Checkpoint Adam状态预初始化的零分母权重损坏修复（2026-10-15）：
- 审查保存/恢复前置路径发现init_optim_state虽临时lr=0，Adam仍计算归一化
  更新，eps=0或FP16默认1e-8下溢时0/0产生NaN，零学习率无法阻止权重损坏。
  新增Adam/AdamW × FP32 eps0/FP16 eps1e-8共4回归，修复前全部失败；日志
  `/tmp/llmtuner-optimizer-denominator-before.log`。常规默认配置eps要求正值，
  但FP16正epsilon下溢仍存在；直接optimizer和显式param group也可使用eps0。
- `components/optimizer/state_dict.py::init_optim_state`仅在Adam状态分配步骤
  暂用可表示的epsilon1.0，finally恢复用户epsilon、lr和原grad引用，既有
  新状态归零及异常撤销机制保留。首个真实更新与独立fresh optimizer逐位一致。
- 新增post-step hook抛RuntimeError/KeyboardInterrupt回归：原异常、epsilon、lr、
  grad身份、权重及state撤销均检查，移除故障hook后可重试。该修复不保证任意
  optimizer hook外部副作用回滚，也不改变真实训练使用用户epsilon的数值语义。
- 优化器专项初轮242 passed；最终完整CPU suite3332 passed、2 skipped、
  229 warnings，退出0；日志 `/tmp/llmtuner-optimizer-denominator-units.log`。
- 真实两rank custom FSDP低精度DCP脚本8配置复跑退出0，日志
  `/tmp/llmtuner-optimizer-denominator-dcp.log`；其FP32 master验证生产恢复链
  回归，不代替FP16 master的独立梯度参考或设备验证。
- 适用全仓pre-commit退出0，compileall与HEAD diff通过，追加文档后定向
  门禁通过；受保护文件排除。本批不改变持久化格式，数据loader预检无新增修改。

Adam checkpoint预初始化的weight decay溢出修复（2026-10-15）：
- 延续上一批零分母检查，有限最大FP16/FP32权重配weight_decay=10时，
  coupled Adam仍在lr0状态分配中溢出并写NaN。4个Adam/AdamW对照用例在
  修复前2 failed、2 passed；日志 `/tmp/llmtuner-optimizer-decay-before.log`。
- `init_optim_state`在Adam/AdamW状态分配时同时临时设置weight_decay=0，
  与epsilon一起保存/恢复，避免分配过程依赖权重幅值。真实更新保留用户
  weight_decay；异常finally也恢复，新增hook异常断言覆盖非零decay还原。
- 新回归检查权重逐位不变、配置恢复、step及moment为零；AdamW作为
  未出现该耦合溢出的对照保留。生产持久化格式不变。
- 优化器专项248 passed；完整CPU suite3336 passed、2 skipped、219 warnings，
  退出0。日志 `/tmp/llmtuner-optimizer-decay-target.log` 与
  `/tmp/llmtuner-optimizer-decay-units.log`。适用全仓pre-commit退出0，
  受保护文件排除，compileall和HEAD diff通过；追加台账后定向门禁通过。
  本批不涉及collective、mesh或通信算法，未重复上一批分布式DCP实验。

OptimizerConfig默认算法选择修复（2026-10-15）：
- 全字段传递审查发现optimizer_type虽暴露Adam/AdamW，默认catch-all参数组
  却硬编码AdamW；请求Adam实际执行decoupled decay，配置与数值不符。
  Adam运行类型回归及5个非法值回归修复前6 failed、AdamW对照1 passed，
  日志 `/tmp/llmtuner-optimizer-type-before.log`。
- `config/optimizer_config.py`默认组使用self.optimizer_type，并对非Adam/AdamW
  立即抛ConfigError。默认AdamW不变，显式参数组继续采用自己的optimizer_name。
- 覆盖真实container两步更新与独立torch Adam/AdamW逐位比较，使用非零decay；
  增加生产CLI parser的--optimizer_type Adam到默认组、显式组优先级回归。
  初轮优化器专项255 passed，后追加2个配置测试由完整suite验收。
- 公共字段不更名；用户此前配置Adam而实际得到AdamW的行为现在纠正。已有
  optimizer state不会因本次修复自动转换算法，恢复时须保持期望的优化器类型。
- 完整CPU suite3345 passed、2 skipped、222 warnings，退出0；日志
  `/tmp/llmtuner-optimizer-type-units.log`。适用全仓pre-commit退出0，排除
  受保护文件；compileall与HEAD diff通过，台账追加后定向门禁通过。

Adam AMSGrad恢复布局修复（2026-10-15）：
- 独立复现AMSGrad checkpoint加载到普通AdamW后返回成功，amsgrad=True但
  max_exp_avg_sq丢失，下一步抛KeyError。反向布局也会错误要求源中不存在状态。
- unflatten_optim_state_dict先解析保存的group选项，再按保存amsgrad决定
  max_exp_avg_sq的必需性及形状检查；恢复选项与状态一致，缺失仍明确失败。
- 双向布局回归比较后续真实更新及全部状态逐位一致，专项259 passed。
  初次数值测试未复制checkpoint张量导致源/目标状态共享，已用deepcopy隔离，
  原失败日志保留/tmp/llmtuner-amsgrad-target.log；修正后日志
  /tmp/llmtuner-amsgrad-target-final.log。测试参考问题与生产缺陷分开记录。
- 证据限平坦状态加载；DCP按目标布局规划时跨AMSGrad选项仍需单独验证，
  不宣称分布式跨选项恢复完成。持久化键不变。
- 完整CPU suite3347 passed、2 skipped、212 warnings，退出0，日志
  /tmp/llmtuner-amsgrad-units-final.log。全仓适用pre-commit初次通过；测试
  deepcopy修改后定向format修复并重跑通过，compileall、HEAD diff通过。

真实DCP Adam/AMSGrad同配置恢复补证（2026-10-15）：
- 新增test_adam_dcp_resume_restores_complete_state_before_next_update，生产
  OptimizersContainer、Adam、非零coupled decay、scheduler及同步DCP实际I/O。
  普通/AMSGrad两配置先更新保存，再以不同seed新建运行时加载；更新前逐位
  比较模型、完整optimizer与scheduler，继续相同下一步后loss/模型/optimizer
  逐位一致。专项2 passed、4 warnings，退出0。
- 该证据是CPU单进程真实DCP，不是分布式collective或跨AMSGrad配置验收。
  跨选项DCP目标键规划仍开放，不将上一批flat状态双向支持外推到DCP。
- 当前全量源码清单与台账核对结果以上述实际命令输出为准，阅读覆盖与
  运行时验收分开；本批无生产修改，不改变持久化格式。
- 清单163个生产文件、163条唯一记录，无遗漏或失效路径。完整CPU suite
  3349 passed、2 skipped、228 warnings，退出0，日志
  /tmp/llmtuner-adam-dcp-units.log；受影响文件pre-commit、compileall及
  HEAD diff通过。全局剩余实现和设备验证缺口保持开放。

真实Trainer默认算法选择与完整恢复验证（2026-10-15）：
- 新增test_trainer_selected_optimizer_resumes_complete_next_update，真实Trainer
  分别用OptimizerConfig optimizer_type Adam/AdamW、非零decay、for-loop。
  显式断言实际内层类，训练step1保存，连续运行step2；关闭后新建Trainer
  加载step1，先精确比较模型、optimizer、scheduler、trainer计数、synthetic
  loader游标，再执行step2，全部状态与连续轨迹逐位一致。
- 专项2 passed、4 warnings，退出0，日志
  /tmp/llmtuner-trainer-adam-resume.log。相较container及独立DCP测试，该证据
  覆盖公开配置到Trainer装配和数据游标；不覆盖设备、跨算法迁移或随机dropout。
- 本批仅测试与文档变化，完整CPU基线复用前批3349 passed、2 skipped，
  新增两项单独执行。受影响文件pre-commit与HEAD diff通过。

低精度梯度范数先提升精度修复（2026-10-15）：
- 四个40000的有限FP16梯度范数为80000，原get_total_norm返回FP16 inf，
  导致误判非有限或裁剪清零；原并行路径后置float转换不能挽回局部溢出。
  foreach开关两回归修复前均失败，日志/tmp/llmtuner-gradnorm-before.log。
- accelerator/collectives.py在局部norm计算前将FP16/BF16梯度临时转FP32，
  原梯度存储和裁剪仍保留原dtype；与独立FP32范数及裁剪参考精确一致。
  代价是norm阶段临时FP32张量，不宣称设备性能改善。
- collectives/Trainer专项130 passed，完整CPU结果记录于
  /tmp/llmtuner-gradnorm-units.log；不改变mesh分组和通信顺序。
- 完整CPU3353 passed、2 skipped、222 warnings，退出0；真实两rank
  CPU/Gloo parallel_grad_norm_equivalence 的PP/TP/EP参考通过，退出0，
  日志/tmp/llmtuner-gradnorm-distributed.log；未验收目标设备峰值内存。
  受影响文件pre-commit、compileall及HEAD diff通过。

低精度局部范数溢出的真实分布式补证（2026-10-15）：
- parallel_grad_norm_equivalence.py新增每rank四个40000的FP16梯度，局部
  norm80000已超FP16范围；PP/TP/EP × foreach false/true共6配置，用独立
  float64全向量参考比较全局norm和裁剪梯度。norm容差1e-7、FP16梯度1e-3，
  依据FP32计算及FP16存储精度，不沿用仅标量300/400的较弱覆盖。
- 两rank CPU/Gloo运行退出0，新6配置和原空梯度/无参数peer场景均通过；
  日志/tmp/llmtuner-gradnorm-local-overflow.log。命令为PYTHONPATH=.
  GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1
  --master_port=<空闲端口> tests/integration_tests/parallel_grad_norm_equivalence.py。
- 本批仅集成测试和文档，复用前批完整CPU3353 passed、2 skipped；定向
  pre-commit与HEAD diff通过。不证明DTensor低精度、完整混合并行训练或设备
  内存/性能；这些缺口保持开放。

低精度DTensor范数与裁剪验证（2026-10-15）：
- parallel_grad_norm_equivalence.py新增真实distribute_tensor Shard(0)参数/梯度，
  FP16/BF16 × foreach开关4配置。使用已量化实际输入的独立float64全向量norm
  参考，核对返回FP32、Shard placement不变及完整裁剪梯度；BF16参考从实际
  存储值构建，避免把40000量化误差误判为通信误差。
- 两rank CPU/Gloo退出0，日志/tmp/llmtuner-gradnorm-dtensor.log；既有局部
  溢出PP/TP/EP及空peer用例同次通过。定向pre-commit和HEAD diff通过。
  本批仅测试/文档，复用完整CPU3353 passed、2 skipped；不证明完整FSDP
  autograd/optimizer链、二维mesh或目标设备性能/峰值内存。

低精度范数的临时存储优化（2026-10-15）：
- 前批实现同时保留全部低精度grad.float副本，增加训练峰值存储。
  改为每张量linalg.vector_norm指定FP32计算，只保留标量norm后汇总，
  普通FP32/FP64路径不变；不再由Python列表持有全部FP32梯度副本。
- 专项130 passed；真实两rank PP/TP/EP低精度局部溢出、FP16/BF16真实
  DTensor foreach开关及空peer脚本退出0，日志
  /tmp/llmtuner-norm-scalar-distributed.log。追加混合FP16/FP64的L1/L2/inf
  参考测试，collectives专项通过；不宣称设备实际内存或速度收益已测量。
- 完整CPU结果见/tmp/llmtuner-norm-scalar-units.log；全仓适用pre-commit、
  compileall及HEAD diff通过，受保护文件排除。

DCP跨AMSGrad配置的目标规划修复（2026-10-15）：
- 将真实Adam DCP恢复测试扩为源/目标amsgrad两开关4配置；修复前2 failed、
  2 passed，日志/tmp/llmtuner-amsgrad-cross-dcp-before.log，确认flat修复不足。
- CheckpointLoadPlanner根据保存metadata为本rank已有exp_avg_sq补齐maximum
  moment目标，沿相同mapping写回原嵌套字典；源无maximum时移除目标预分配
  项，使后续按保存amsgrad进行严格校验，不保留零值伪装已恢复状态。
- 修复后4配置真实同步DCP、不同seed新建运行时、更新前完整状态及下一步
  更新逐位一致，日志/tmp/llmtuner-amsgrad-cross-dcp-after.log。仅CPU单进程，
  compiled legacy键、多维mesh、目标设备跨选项仍待验证，不宣称全矩阵完成。
- 完整CPU3358 passed、2 skipped、225 warnings，退出0，日志
  /tmp/llmtuner-amsgrad-cross-dcp-units.log；定向pre-commit、compileall及
  HEAD diff通过。

旧compiled名称与AMSGrad目标补齐组合检查（2026-10-15）：
- 新增真实DCP Stateful回调回归，保存state._orig_mod.weight的exp_avg_sq及
  maximum，目标仅预分配canonical exp_avg_sq；默认planner恢复新增maximum
  到canonical键且数值正确，1 passed、2 warnings。未发现默认生产路径缺陷。
- 最初普通dict测试不接收DCP新增键，随后测试callback清空自身输入导致失败；
  改为生产Stateful契约并复制回调字典后通过。诊断期间非默认flat路径修改
  撤去，未以测试参考错误修改生产语义。非默认flat planner不新增支持声明。
- 定向pre-commit与HEAD diff通过；本批最终仅测试/台账变化，复用前批完整
  CPU3358 passed、2 skipped，新增用例单独通过。设备和多mesh缺口保留。

AMSGrad新增目标的compiled名称冲突校验（2026-10-15）：
- 旧键复核发现saved_names只检查已有目标，新补齐maximum状态不在目标中，
  canonical与_orig_mod同义键可绕过歧义检查。新增真实DCP回归修复前1 failed，
  日志/tmp/llmtuner-amsgrad-ambiguity-before.log。
- planner将本rank已有exp_avg_sq对应的潜在maximum目标纳入原冲突检查，
  保存键有歧义时规划阶段拒绝，保持唯一旧名迁移及不同PP part所有权边界。
  清除前批测试无用的重复destination赋值。
- checkpoint专项124 passed、95 warnings，退出0；日志
  /tmp/llmtuner-amsgrad-ambiguity-target.log。定向pre-commit、compileall及
  HEAD diff通过；CPU真实DCP证据，不外推远端存储或多rank全矩阵。
- 完整CPU3360 passed、2 skipped、222 warnings，退出0，日志
  /tmp/llmtuner-amsgrad-ambiguity-units.log；追加台账后格式门禁通过。

Custom FSDP跨AMSGrad低精度恢复矩阵（2026-10-15）：
- fine_grained_checkpoint_equivalence.py扩为BF16/FP16 compute、FP32 master/
  reduction × reshard × full AC × 源/目标AMSGrad开关，共32配置；保留ancestor
  relocated hooks、生产AdamW/scheduler/EMA/DCP和不同seed新建运行时。
- 两rank CPU/Gloo32配置全部通过，退出0，每rank32条PASS；恢复后更新前
  模型/optimizer/scheduler/EMA、下一步输出/完整梯度及更新后状态逐位一致。
  日志/tmp/llmtuner-amsgrad-fsdp-distributed.log。既有模型仅4参数，无PP/EP；
  不外推拓扑变化、compile、异步transport、数据/RNG或目标设备恢复。
- 命令PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2
  --master_addr=127.0.0.1 --master_port=<空闲端口>
  tests/integration_tests/fine_grained_checkpoint_equivalence.py。
- 本批仅集成脚本与文档；复用前批完整CPU3360 passed、2 skipped证据。
  首次pre-commit发现新增docstring超长行，换行后定向门禁通过，HEAD diff通过。

DCP形状预检及目标未写入复核（2026-10-15）：
- 新增真实DCP保存/加载回归，源长度2目标3及源3目标2；当前PyTorch2.14
  DefaultLoadPlanner已有Size mismatch检查，在read前拒绝，两种目标保留
  原始-7值。无需重复增加生产形状校验。
- 首轮测试错误匹配shape而实际消息为Size mismatch，2 failed仅是测试参考
  错误；改为实际错误契约后2 passed，4 warnings，退出0。失败诊断日志
  /tmp/llmtuner-dcp-shape-before.log；定向pre-commit和HEAD diff通过。
- 本批仅测试/台账，复用前批完整CPU3360 passed、2 skipped。该证据仅普通
  tensor尺寸检查，不代替DTensor跨mesh、存储损坏或读取中途失败事务保证。

Adam必需状态的嵌套替代拒绝（2026-10-15）：
- 复核components/optimizer/state_dict.py及OptimizersContainer恢复预检，发现
  缺失step/exp_avg/exp_avg_sq/max_exp_avg_sq时，同名前缀的子键可被恢复为dict，
  绕过计数和moment检查并污染优化器状态。4项回归修复前均未抛错，日志
  /tmp/llmtuner-adam-nested-before.log；问题为恢复期校验缺陷。
- 仅原状态为dict时允许嵌套恢复；必需非dict状态缺失即KeyError。回归同时
  确认第二个优化器预检失败时，第一个优化器lr及两者原状态身份不变。
  既有通用嵌套状态、空dict、AMSGrad跨选项恢复测试保留。
- 优化器专项263 passed、2 warnings；完整CPU3366 passed、2 skipped、
  247 warnings，退出0，日志/tmp/llmtuner-adam-nested-units.log。无collective
  或布局修改，本批不扩展分布式支持声明。首次全仓适用pre-commit发现新增
  测试装饰器行超长并由format修正，随后复跑；compileall及HEAD diff检查通过。
- 该检查不保证读取中途失败的事务性，也不证明跨设备恢复；现有全局能力
  缺口继续保留。下一步审查恢复时group选项及多优化器状态提交的异常边界。

PP2 × custom FSDP2跨AMSGrad恢复（2026-10-15）：
- 扩展pp_checkpoint_equivalence.py公开ParamGroupConfig入口，新增环境变量
  LLMTUNER_TEST_AMSGRAD，full/resume可独立选择；不修改生产实现。
- 四rank CPU/Gloo，PP2 × custom FSDP2、1F1B、4 microbatch、float32、dynamic
  EMA，两方向AMSGrad 1→0和0→1均在独立torchrun进程训练2步保存/再训练2步，
  新建Trainer恢复。四次进程退出0，各方向4rank更新前完整状态无差异，最终
  model/optimizer/scheduler/trainer/EMA精确一致，后续loss最大绝对差0。
- 命令：PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 LLMTUNER_TEST_WORLD_SIZE=4
  LLMTUNER_TEST_PP_SIZE=2 LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom
  LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_AMSGRAD=<源或目标0/1>
  TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=4
  --master_addr=127.0.0.1 --master_port=<空闲端口>
  tests/integration_tests/pp_checkpoint_equivalence.py <full或resume> <共享工作目录>。
- 原始状态与四份日志位于
  /var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-pp-amsgrad-431cfa9k；
  文件名1-0-full/resume.log和0-1-full/resume.log。以上摘要保留验收结果。
- 本批仅脚本与文档，复用前批完整CPU3366 passed、2 skipped；定向pre-commit
  及HEAD diff检查通过。未覆盖EP、legacy compiled、混合精度、其他PP schedule
  或目标设备跨配置恢复；全局缺口保持开放。

EP2 × custom FSDP4跨AMSGrad恢复（2026-10-15）：
- 审查random_state.py的rank限定payload、拓扑预检、host预校验及manager最后
  恢复RNG顺序；已有CPU失败保持流测试覆盖该契约，本批未发现需修改的host问题。
  device setter失败后的设备状态及分布式读取事务性仍未据此证明。
- 沿专家状态归属复核expert_parallel/checkpointing.py的完整expert gather、
  load slice、DTensor mesh/placement恢复，再运行真实Trainer公开配置链路。
- 四rank CPU/Gloo、qwen3_moe eager experts、PP1/EP2/custom FSDP4、float32、
  dynamic EMA；AMSGrad源/目标1→0及0→1，各full/resume为独立torchrun进程。
  4次进程均退出0，两方向各4rank更新前完整状态无差异，后续两步loss最大
  绝对差0，最终model/optimizer/scheduler/trainer/EMA精确一致。
- 命令沿用前批pp_checkpoint_equivalence.py full/resume及IPv4四rank启动，
  增加LLMTUNER_TEST_MOE=1 LLMTUNER_TEST_EP_SIZE=2 LLMTUNER_TEST_PP_SIZE=1，
  其余WORLD_SIZE=4、FSDP_IMPLEMENTATION=custom、EMA=dynamic、源/目标
  AMSGRAD开关不变。日志/保存状态目录
  /var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-ep-amsgrad-z_a6r1i4，
  日志1-0-full/resume.log及0-1-full/resume.log；摘要保留上述实际结果。
- 本批仅补充能力矩阵及台账，文档门禁通过；复用前批CPU3366 passed、2 skipped。
  不外推EP拓扑变化、共享专家、TP、compile、混合精度、目标设备或性能。
  已覆盖的跨AMSGrad分支不再重复扩展；下一步回到compile恢复及生命周期缺口。

Trainer loader关闭失败的所有权修复（2026-10-15）：
- 复核trainer.close、GrainDataLoader.close及CLI异常保持边界。Trainer先清空
  loader引用，迭代器关闭抛错时未恢复，导致后续close无法重试worker/预取资源。
  实际GrainDataLoader.close委托路径加故障注入，RuntimeError及KeyboardInterrupt
  两项修复前失败，日志/tmp/llmtuner-loader-cleanup-before.log。
- Trainer保留关闭失败的dataloader，继续释放其余资源，最后传播首次异常；
  再次close仅重试失败loader，已成功metrics/GC不重复关闭。更新close文档为
  实际异常契约，未拆分trainer.py。测试检查原异常身份和两次关闭调用顺序。
- cleanup/close专项14 passed；完整CPU3368 passed、2 skipped、234 warnings，
  退出0，日志/tmp/llmtuner-loader-cleanup-units.log。相关及全仓适用pre-commit、
  compileall、HEAD diff检查通过，受保护native_alltoall未进入自动修复。
- 本批不证明真实worker被中断后的内部可恢复性，只保证Trainer不丢弃清理
  所有权；并发close、成功运行的全局策略所有权及设备RNG回滚仍有缺口。
- 核对当前normalization.py、配置及较新台账，aten生产实现已接入并有严格
  CPU恢复证据；历史候选“未接入”不作为当前状态，避免重复实现。

真实Grain预取线程中断关闭的重试验证（2026-10-15）：
- 审查datasets/dataloader.py、loader_builder.py及当前Grain
  ThreadPrefetchDatasetIterator.close/_stop_prefetch：join抛错时保留线程引用，
  后续close可继续stop/join并清理父迭代器。无需复制上游线程生命周期实现。
- 新回归使用真实Grain MapDataset、TextCollator及ThreadPrefetch预取线程，
  先消费batch启动worker，仅第一次worker.join注入KeyboardInterrupt。
  Trainer首次保留loader及原异常，第二次关闭完成真实join，验证worker不再
  存活、内部线程引用清空、关闭后next被拒绝，第三次Trainer.close不重复join。
  finally恢复真实join并清理，避免测试失败遗留线程。
- 新回归1 passed；数据管道专项336 passed、10 warnings，退出0。首次
  pre-commit由format修正测试签名长行，复跑全部适用检查通过，HEAD diff通过。
  本批仅测试/台账，复用上一批完整CPU3368 passed、2 skipped证据，新增回归
  单独执行通过；没有生产变化或全设备退出正确性的新增声明。
- 该证据关闭上一批本机Grain正常worker在join入口中断后的重试缺口；不保证
  卡死的数据源能取消、不覆盖OS进程死亡或并发close。全局策略所有权和硬件
  并行验证仍需推进。

TP共享专家投影完整几何预检（2026-10-15）：
- 审查tensor_parallel.py的共享参数拒绝、shared MLP切片和所有block预检顺序。
  原实现只检查每个共享投影分片维整除性，gate输入维、up宽度或down输出维
  不兼容仍先改变所有block。三种复现修复前3 failed、原预检4 passed，日志
  /tmp/llmtuner-shared-geometry-before.log。
- 变换前验证共享gate/up同为(F,D)、down为(D,F)，D与routed experts一致；
  失败报告路径及三种实际shape，不要求共享F等于routed F，保留独立容量。
  扩展既有跨block预检回归，确保拒绝时两block原类、所有Parameter身份保持。
- TP MoE专项31 passed；真实两rankCPU/Gloo生产TP边界前向、所有梯度、
  两次SGD更新和重复apply路径退出0，日志
  /tmp/llmtuner-shared-geometry-distributed.log。命令PYTHONPATH=.
  GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1
  --master_port=<空闲端口> tests/integration_tests/tp_moe_equivalence.py。
- 完整CPU3372 passed、2 skipped、242 warnings，退出0，日志
  /tmp/llmtuner-shared-geometry-units.log。首次门禁长行由format修正后复跑通过；
  全仓适用pre-commit、compileall及HEAD diff通过，保护文件排除。
- 共享专家跨注册路径的别名保留实现仍未完成，本批仅修复完整几何预检，
  不将拒绝配置等同该迁移完成，不外推目标设备或全组合数值支持。

TP同一shared MLP gate/up权重绑定支持（2026-10-15）：
- 原apply_tp对所有MoE待切片参数别名一律拒绝。本批支持同一shared MLP的
  gate/up同参数或同Linear两种绑定：二者切片轴、布局、消费边界相同。
  helper先捕获全部原参数，按身份缓存一次切片，两个投影绑定同一新Parameter，
  不重复切片，保留requires_grad及更新后的feature metadata。
- preflight仅允许这两个完整注册路径的别名集合；跨block routed/shared专家、
  外部消费者别名及gate/down不同轴绑定继续在模型改变前拒绝。
  两rank×两种绑定CPU回归检查身份、冻结、独立torch.chunk参考及feature大小；
  两种负例检查拒绝后类、参数身份与值不变。TP MoE专项37 passed。
- 真实两rankCPU/Gloo脚本新增--tied-shared-projections。默认、绑定、绑定加
  --router-fp64三次均退出0；独立partial专家图与dense参考核对输出/输入梯度/
  全部参数梯度、重复apply身份及SGD更新，保留原严格容差。日志
  /tmp/llmtuner-tp-shared-tie-{baseline,tied,tied-fp64}.log。
  命令PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2
  --master_addr=127.0.0.1 --master_port=<空闲端口>
  tests/integration_tests/tp_moe_equivalence.py [上述参数]。
- 完整CPU3376 passed、2 skipped、246 warnings，退出0，日志
  /tmp/llmtuner-tp-shared-tie-units.log；随后仅新增两负例，专项37 passed。
  全仓适用pre-commit、compileall及HEAD diff通过，保护文件排除；格式门禁
  初次长行由format修复后复跑。工作区暂存状态外部变化，未主动暂存或覆盖。
- 本批推进别名保留能力的已验证子集，不关闭跨block共享专家迁移；尚无该绑定
  与FSDP/EP/compile/保存恢复组合或目标设备证据。全局目标仍未完成。

P0：TP切片checkpoint被DCP复制去重（2026-10-15，未修复）：
- 扩展pp_checkpoint_equivalence.py的公开ParallelConfig入口，支持环境变量
  LLMTUNER_TEST_TP_SIZE；保持原更新前完整状态及后续严格恢复断言。
- 真实两rank CPU/Gloo、PP1/TP2、默认native FSDP（DP shard1）、float32、
  dynamic EMA，独立full/resume进程2+2步。full退出0，resume退出1；更新前
  model/optimizer/EMA已错误，后续step3/4 loss差0.005269527和0.020684242。
  无放宽断言、跳过失败或生产修改。源/恢复日志和状态目录：
  /var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-tp-checkpoint-0nw5rekj。
- 独立读取.metadata和实际DCP tensor确认：layers.0.mlp.gate_proj.weight
  两rank源切片不同，但保存metadata.size只有本地(64,64)，实际存储tensor
  等于rank1源切片、不等于rank0。该参数原完整形状应为(128,64)。因此不是
  浮点容差问题：不同TP切片在同FQN下按普通复制tensor被去重。
- ModelCheckpointState、OptimizersContainer（及复用它的EMA）目前只调用
  EP gather/load适配，未给普通TP参数表达全局布局；FSDP局部DP DTensor也不能
  单独表达外层TP归属。故TP训练等价证据不等于checkpoint恢复支持。
- 复现：PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 LLMTUNER_TEST_WORLD_SIZE=2
  LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_TP_SIZE=2 LLMTUNER_TEST_EMA=dynamic
  TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=2
  --master_addr=127.0.0.1 --master_port=<空闲端口>
  tests/integration_tests/pp_checkpoint_equivalence.py <full或resume> <同一目录>。
- 下一动作：实现TP参数布局描述及checkpoint全局张量gather/load切片，覆盖dense
  col/row、packed gate/up、shared aliases；模型/optimizer/EMA共用契约，组合
  EP/FSDP时保持通信顺序与所有权。不可用rank后缀掩盖全局布局、禁止TP保存或
  放宽断言作为完整修复；旧checkpoint丢失的rank切片不可凭空恢复。
- 验收：上述生产Trainer原严格断言通过、TP2×FSDP2及MoE/绑定布局补证，
  metadata为完整全局形状，所有TP rank切片正确，完整状态与后续更新一致。
  实现未修复，验证失败，P0保持开放；本批脚本/文档门禁通过不代表该行为通过。

TP checkpoint全局布局修复续接（2026-10-15）：
- 基线HEAD 500f166956c46093355ec082555e4b9782cba054；续接时原适配文件已不在
  源码，未复用历史通过作为当前证据。本轮新增tensor_parallel/checkpointing.py，
  TP入口记录同名核心入口内实际变换的path/dim/packed/group，model/optimizer/
  EMA共用全局gather与load slice；FSDP先full_tensor后TP gather，load恢复DP
  placements；同FQN不再将不同TP值交给DCP按复制去重。packed按gate/up分半
  重排，标量step不变，旧缺失切片checkpoint不能无损迁移。
- 当前源码实际执行：PP1/TP2两rank native及TP2×custom FSDP2四rank，dynamic
  EMA、float32、独立full/resume各2+2步，四次进程退出0；更新前完整状态及
  最终model/optimizer/scheduler/trainer/EMA精确一致，后续loss差0。DCP metadata
  gate_proj全球shape(128,64)，已不再是错误本地(64,64)。
- 命令与原P0复现相同；四rank改WORLD_SIZE=4、nproc=4、
  FSDP_IMPLEMENTATION=custom，TP_SIZE=2、PP_SIZE=1、EMA=dynamic、PCH=0。
  状态及full/resume日志目录：
  /var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-tp-dcp-current-6q__3z9j，
  子目录2-native、4-custom。结果由原严格断言及独立metadata读取确认。
- tp_moe_equivalence.py默认及--tied-shared-projections两rank脚本均退出0，
  packed/shared全局state、实际本地权重、绑定身份和下一次Adam更新精确一致；
  日志/tmp/llmtuner-tp-dcp-current-{plain,tied}.log。完整CPU3378 passed、
  2 skipped、230 warnings，退出0；/tmp/llmtuner-tp-dcp-current-units.log。
  全仓适用pre-commit、compileall和HEAD diff通过，保护文件排除。
- 实现状态：部分修复；正常CPU路径通过，仍需规划错误协调回归、TP×EP、
  topology变更及目标设备验证。_index_layouts或entries本地失败发生在metadata
  collective之前时需协调；不能因正常恢复通过关闭全部失败路径验收。
  下一动作：将本地规划错误作为group manifest报告，再注入rank局部损坏测试。

TP checkpoint本地规划错误协调（2026-10-15）：
- 在当前未提交TP适配基础上修复_entries/参数索引及shape检查可能在metadata
  collective前单rank抛错的问题。先从注册布局发现group，本地准备捕获Exception，
  再向各group报告ready/manifest；所有metadata报告完成后才抛本地原异常或peer
  错误，禁止失败rank的peer进入tensor gather/DTensor full_tensor。
  加载也共用规划阶段，完整shape校验在任何distribute_tensor之前。
- 新增tp_checkpoint_coordination.py，真实两rankCPU/Gloo，保存/加载 × shape、
  missing_parameter、packed奇数、manifest dtype不一致8配置，两rank均抛错，
  每次后续barrier成功，最后正常gather/load精确重试通过，退出0；日志
  /tmp/llmtuner-tp-checkpoint-coordination.log。命令PYTHONPATH=.
  GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1
  --master_port=<空闲端口> tests/integration_tests/tp_checkpoint_coordination.py。
- 两rank tp_moe_equivalence.py --tied-shared-projections重新执行退出0，
  packed/shared真实DCP及下一Adam更新精确一致；日志
  /tmp/llmtuner-tp-coordination-moe.log。完整CPU3378 passed、2 skipped、
  233 warnings，退出0，日志/tmp/llmtuner-tp-coordination-units.log。
- 适用全仓pre-commit、compileall及HEAD diff通过，保护文件排除。先前TP2/
  custom FSDP2正常Trainer恢复证据见上一批；本批未修改布局数学或容差。
- 限制：group注册本身须各rank一致；损坏到无法发现group、进程死亡、
  KeyboardInterrupt及collective内部失败不承诺可恢复。TP×EP及目标设备仍待
  验证；本批不将有限故障模型推广为全checkpoint事务性。

TP×EP checkpoint转换顺序复核（2026-10-15）：
- ID TP-EP-CKPT-01 / P1 / tensor_parallel/checkpointing.py；调用方为
  ModelCheckpointState与OptimizersContainer（EMA复用后者）。
- 静态确认保存先gather_expert_state再gather_tensor_parallel_state，加载逆序。
  EP gather将专家维扩至全局，但TP _prepare_state仍用live parameter的
  EP-local shape校验所有维度；同一专家同时标记TP/EP时存在shape拒绝风险。
  尚未取得真实四rank复现，不将此推断写为已验证运行时缺陷。
- 实现状态：待修复判定；验证状态：静态调用链已核对，组合运行待执行。
  验收：TP2×EP2保存/加载模型、Adam及EMA全局专家布局，并在独立运行时
  恢复后比较更新前状态和下一更新；同时保留单轴与packed gate/up契约。
- 下一动作：以真实二维TP/EP组及注册专家参数最小复现，明确中间布局各轴
  所有权，再调整TP形状规划；不以放宽所有shape检查代替布局契约。

TP×EP checkpoint中间布局修复（2026-10-15）：
- TP-EP-CKPT-01从静态疑点确认为真实四rank CPU/Gloo缺陷：TP2×EP2，
  packed参数本地(2,4,3)，EP gather后(4,4,3)，TP仍要求(2,4,3)。
  /tmp/tp_ep_checkpoint_probe.py原始复现所有rank确认ValueError并通过barrier，
  退出0，日志/tmp/tp-ep-checkpoint-probe.log。
- 生产TP适配显式使用expert_shard_map扩大中间专家维，两方向均保持该维全局，
  TP/EP同轴分片明确拒绝。加载时EP专家不提前distribute_tensor，交由后续
  EP切片恢复live DP布局；其他TP参数维持原分发契约。
- 新增tests/integration_tests/tp_ep_checkpoint_layout.py：独立arange全局
  (4,8,3)参考，二维真实TP/EP组，packed gate/up分别切片；模型、exp_avg与
  ema_params三类键的全局值和恢复本地切片均rtol=atol=0，四rank退出0。
  命令PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=4
  --master_addr=127.0.0.1 --master_port=<空闲端口>
  tests/integration_tests/tp_ep_checkpoint_layout.py；日志
  /tmp/tp-ep-checkpoint-layout.log。代码状态HEAD 500f166及未提交TP适配。
- 完整pytest -q --disable-warnings：3378 passed、2 skipped、224 warnings，
  35.18s，退出0，/tmp/tp-ep-checkpoint-units.log；适用全仓pre-commit（保护
  文件排除）、compileall、Ruff和HEAD diff检查通过。
- 实现状态：中间形状缺陷已修复；验证状态：CPU适配器精确往返通过。
  未关闭：真实Trainer TP×EP DCP、独立恢复与下一Adam/EMA更新，TP×EP×FSDP
  及设备数值；尤其EP后置分发的DTensor分支尚无本批运行证据。

TP×EP真实Trainer DCP恢复（2026-10-15）：
- TP-EP-CKPT-01：当前HEAD 500f166及未提交TP适配，macOS/Python3.13/
  PyTorch2.14 CPU/Gloo，world4、TP2、EP2、PP1、custom FSDP、float32、
  dynamic EMA，真实qwen3_moe，独立full/resume torchrun进程。
- 首轮full退出1，测试router mesh断言沿用TP1假设，把TP轴计入DP mesh。
  依据fully_shard_parallel.py的独立DP子mesh契约，router期望改为
  world/PP/TP；expert仍使用world/PP/EP，未删除任何数值或恢复断言。
  原失败日志位于llmtuner-tp-ep-trainer-fgtlibs5/full.log（系统临时目录）。
- 修正后full/resume均退出0；保存step2后连续再训2步，与独立恢复后2步
  比较。四rank更新前完整状态failures=0；最终模型、optimizer、scheduler、
  trainer、动态EMA逐位一致，后续loss max_abs=0。full losses为
  [4.879703521728516,4.869268894195557,4.842996597290039,4.858843803405762]。
- 证据根目录/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/
  llmtuner-tp-ep-trainer-f9g8xikm，含full.log、resume.log、split/final快照与DCP。
  命令：PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 LLMTUNER_TEST_WORLD_SIZE=4
  LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_TP_SIZE=2 LLMTUNER_TEST_EP_SIZE=2
  LLMTUNER_TEST_MOE=1 LLMTUNER_TEST_EMA=dynamic
  LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom
  TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=4
  --master_addr=127.0.0.1 --master_port=<空闲端口>
  tests/integration_tests/pp_checkpoint_equivalence.py <full|resume> <同一目录>。
- 本批仅修改测试mesh断言与用例说明；生产实现未变化，复用上一批CPU3378
  passed、2 skipped及compileall证据。目标设备、拓扑变化、跨compile恢复和
  多轴局部异常协调不在本配置证据内；不外推所有TP/EP组合。

TP×EP跨compile恢复失败定位起点（2026-10-15）：
- 当前源码集合核对：164个llmtuner/**/*.py，对应台账164个唯一条目，
  无遗漏/重复/失效路径；此为清单完整性，不代表全部运行时验收。
- ID TP-COMPILE-ATTN-01 / P1 / tensor_parallel.py attention边界、
  models/common/tp_projections.py、HF attention及compile调用链。
  配置沿上一批world4/TP2/EP2/custom FSDP/float32/dynamic EMA；full为
  eager，resume设置LLMTUNER_TEST_COMPILE_BACKEND=aot_eager，其余不变。
- 实际full退出0、resume退出1。四rank pre-update state failures=0，
  随后首次编译前向SDPA fake tensor报错：Q/K/V形状(1,2,128,16)，
  mask为(1,1,64,64)，无法broadcast。未得到恢复后更新数值验收；不将
  加载成功外推为跨compile完整支持，也不把尚未定位的错误归因于DCP。
- 证据目录/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/
  llmtuner-tp-ep-crosscompile-ndu4e8ow，full.log/resume.log及独立split状态。
  命令同上一批pp_checkpoint_equivalence.py full/resume，唯resume增加
  LLMTUNER_TEST_COMPILE_BACKEND=aot_eager。代码HEAD 500f166+未提交TP适配，
  本批未改生产代码或测试断言。
- 实现状态：待定位；验证状态：失败。下一动作：使用相同公开配置在全新
  compiled Trainer首步复现，区分恢复影响与TP compile attention边界问题；
  对照gather前后hidden、position/rope及mask形状，保留数值与编译执行断言。
  不通过关闭compile、跳过mask或放宽恢复断言关闭此问题。

TP compile attention与checkpoint解耦诊断（2026-10-15）：
- TP-COMPILE-ATTN-01：全新world4/TP2/EP2/custom FSDP/aot_eager Trainer
  full首步同样失败，退出1；没有load，证明checkpoint恢复不是必要条件。
  日志目录/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/
  llmtuner-tp-ep-freshcompile-qowy84qt/full.log。
- 移除EP和MoE，world2/TP2/PP1/custom/aot_eager/qwen3全新Trainer仍失败，
  退出1，SDPA Q序列256而mask128；目录同临时根下
  llmtuner-dense-tp-compile-wyh2e09r/full.log。命令沿前批full，将world改2，
  移除LLMTUNER_TEST_MOE/EP_SIZE/EMA，full设置COMPILE_BACKEND=aot_eager。
- 静态调用链：HFTransformerModel._apply_attention在普通SDPA返回mask=None，
  HF decoder据边界前本地序列构造mask；GatherSequenceFirst在attention内部
  才收集完整序列。编译时显式mask与完整Q不一致是当前根因假设，尚需独立
  张量形状证据核实，不直接按该假设修改mask语义。
- 下一动作：定位HF mask生成处使用的inputs_embeds/cache_position与完整positions
  的形状，确认compile/eager分支；为TP提供与gather后序列一致的因果mask，
  保留packed/CP/sliding-window契约与前后向数值参考后再验收。
  本批无代码修改，不重跑无关CPU套件；文档格式/diff检查通过。

TP SDPA compile mask修复与专家动态计数后续失败（2026-10-15）：
- TP-COMPILE-ATTN-01：核对HF decoder在输入embedding后、attention gather前
  调用create_causal_mask；编译显式mask使用本地长度。HFTransformerModel
  forward现对无CP、完整positions长于本地输入的SDPA路径，按完整长度调用
  HF原生mask工厂，保留full/sliding规则。Qwen3接收mask mapping，Qwen3MoE
  接收单tensor；首轮错误传dict到MoE失败证据保留resume-mask-fix.log。
- 两rank dense TP2/custom/aot_eager真实Trainer full四步和step2保存退出0，
  日志临时根llmtuner-dense-tp-maskfix-fb6czxnl/full.log；原相同配置mask形状
  失败已消失。尚缺独立dense数值参考及packed/CP组合回归，不宣称完整验收。
- 四rank eager→compiled TP2/EP2恢复再次执行：完整更新前状态仍通过，SDPA
  已通过，随后experts.py:170的counts.tolist()/if count触发
  GuardOnDataDependentSymNode Ne(u0,0)，退出1。日志原crosscompile目录
  resume-mask-fix2.log。新增TP-COMPILE-EXPERT-01 / P1，下一动作检查空token、
  动态分组GEMM与编译捕获契约，不能关闭compile或删除梯度来绕过。
- 当前改动仍为进行中：CPU3378 passed、2 skipped、229 warnings，32.69s
  （/tmp/tp-compile-mask-units.log，执行于首次mask实现；其后修正MoE单tensor
  格式，最新集成结果如上）；最新全仓适用pre-commit、compileall、diff通过。
  下一批补充mask工厂格式/滑动窗口回归及完整最新CPU门禁。

专家动态计数compile修复与跨compile数值失败（2026-10-15）：
- TP-COMPILE-EXPERT-01：GroupedExperts fallback去掉counts.tolist后的if count，
  使用torch._check(count>=0)、end边界及torch.narrow动态segment；空segment
  仍执行F.linear以保留零梯度，不禁用compile、不替换原dtype。
- 新增test_expert_compile_segments.py，float64独立逐专家F.linear参考，
  counts=(2,0,3)/(0,5,0)/(0,0,0)，aot_eager fullgraph=True输出、输入和
  全部参数梯度rtol=atol=1e-12通过，3 passed。
- 原四rank eager→compiled恢复重跑，更新前四rank状态failures=0，现完成
  后续训练，无SDPA形状或专家动态guard错误；最终状态比较仍失败，退出1。
  首个rank0模型键embed_tokens.weight max_abs=0.000906927511，
  q_proj.weight=0.00046589179，非可忽略舍入，原严格断言保留。
  日志原crosscompile目录resume-expert-fix.log。
- 新增TP-COMPILE-NUMERIC-01 / P1 / TP attention mask与编译前后向调用链；
  根因待定位。下一步以同一批数据及保存点固定状态，比较eager/compiled
  前向loss、attention输出、首个梯度差异，对照独立全序列因果参考；
  不以容差放宽、关闭compile或重写连续训练参考关闭该缺口。
- 最新pytest全量3381 passed、2 skipped、236 warnings，34.20s退出0，
  /tmp/tp-expert-compile-units.log。适用全仓pre-commit、compileall及HEAD
  diff已通过，生产修改仍未提交。

TP跨compile首步梯度隔离（2026-10-15）：
- TP-COMPILE-NUMERIC-01：临时/tmp/tp_compile_gradient_probe.py在
  OptimizersContainer.step前保存首步实际梯度，同一step2 checkpoint独立
  eager/compiled恢复；eager退出0、compiled退出1（原最终严格比较）。
  输出位于原crosscompile目录eager-gradient-<rank>.pt及
  compiled-gradient-<rank>.pt，入口日志同前缀.log。
- 四rank梯度键一致，compiled两层input_layernorm.weight梯度均为0，eager
  非零；rank0范数0.003707833与0.004274307，首层最大差0.001428295。
  embedding梯度rank0最大差0.026783919。首步loss相同而梯度已不同，
  将定位重点前移至backward，不将后续loss偏离归为checkpoint加载失败。
- 仅resume设置公开LLMTUNER_TEST_NORM_BACKEND=aten再次运行，退出1，
  两层input norm梯度仍为0；原生RMSNorm公式并非目前足够解释。
  aten-gradient-<rank>.pt及aten-gradient.log保留；不修改生产默认norm。
- /tmp/tp_collective_backward_probe.py真实两rank：gather后平方sum的eager
  梯度精确为4；fullgraph编译被functional_collectives的可选public导入
  AttributeError阻断，退出1，/tmp/tp-collective-backward-probe.log。
  该编译结果未触达反向，不能认定collective backward错误。
- 本批无生产修改。下一动作：预解析实际funcol callable后重跑独立compiled
  gather与reduce-scatter梯度，再对照attention边界前后梯度；保持原数值断言。

TP gather独立compiled backward排除实验（2026-10-15）：
- TP-COMPILE-NUMERIC-01：/tmp/tp_collective_resolved_probe.py在compile外
  使用生产functional_collectives解析实际all_gather callable；真实两rank
  CPU/Gloo，x=ones(2,3)，gather平方和，独立梯度参考为2*x*world=4。
  eager与aot_eager fullgraph=True均精确通过，退出0；日志
  /tmp/tp-collective-resolved-probe.log。
- /tmp/tp_collective_graphbreak_probe.py保留生产all_gather_along及其内部
  可选导入，fullgraph=False，与实际Trainer允许graph break的策略一致。
  eager/compiled梯度同样精确通过，退出0；日志
  /tmp/tp-collective-graphbreak-probe.log。两项均用torchrun两rank、IPv4
  空闲端口、PYTHONPATH=.、GLOO_SOCKET_IFNAME=lo0；代码状态HEAD500f166
  加既有未提交改动，本批生产未修改。
- 结论限该独立collective：没有证据支持all-gather编译反向普遍失效。
  fullgraph可选导入异常与真实模型零梯度分别处理，不将独立通过外推FSDP
  组合。下一动作：缩小到norm→gather→三投影→rowwise reduction的残差块，
  固定权重与输入，对照FSDP前后及graph break边界的输入/参数梯度。

TP norm/gather残差块及custom FSDP组合隔离（2026-10-15）：
- TP-COMPILE-NUMERIC-01：/tmp/tp_norm_boundary_probe.py独立模块为
  nn.RMSNorm→TP all_gather→Linear/SiLU/Linear→reduce_scatter→残差，
  固定seed41、float64，真实两rank CPU/Gloo；eager与aot_eager fullgraph
  前向、输入梯度及三类参数梯度rtol=atol=1e-12通过，退出0。
  日志/tmp/tp-norm-boundary-probe.log。
- /tmp/tp_norm_fsdp_probe.py相同模块，生产顺序compile(fullgraph=False)
  后fully_shard_fine_grained，eager参考亦加custom FSDP；本地DTensor参数
  梯度和输入梯度按相同容差通过，退出0，日志/tmp/tp-norm-fsdp-probe.log。
  FSDP/TP在该隔离用例共用world组，不能视为生产正交mesh组合证明。
- 命令均PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun两rank、IPv4
  空闲端口；当前HEAD500f166及既有未提交修改，本批无生产改动。
- 简化norm/collective/残差/root FSDP组合不足以复现真实HF的零梯度；
  下一动作保留HF attention及decoder子模块FSDP边界进行缩减，检查图边界
  与参数unshard/reshard生命周期。原真实失败继续开放，不以该通过关闭。

TP collective解析图边界修复（2026-10-15）：
- TP-COMPILE-NUMERIC-01：native FSDP compiled恢复亦复现input norm零梯度，
  更新前状态通过、最终比较失败，native-gradient.log保留；custom引擎非必要条件。
- 临时tp_resolved_trainer_probe.py仅在compile外解析实际autograd callable并
  替换resolver，无tensor/通信变化，真实四rank恢复全部严格检查通过，退出0
  （resolved-gradient.log）。据此修改linear_ops.functional_collectives缓存
  环境解析结果，apply_tp装配时预解析，避免可选public import失败在tensor
  图内产生异常边界。仍懒加载、仍要求完整autograd pair。
- assume_constant_result候选触发Dynamo内部NotImplementedError，已撤去，
  原resume-production-resolved.log保留；实际生产采用普通缓存与装配预解析。
- 无诊断替换重跑pp_checkpoint_equivalence.py resume，world4/TP2/EP2/
  custom FSDP/float32/dynamic EMA/aot_eager，独立eager保存点，四rank更新前
  与最终完整状态逐位通过、后续loss差0，退出0；证据原crosscompile目录
  resume-production-cache.log。命令沿前批，仅生产实现更新；HEAD500f166
  加未提交TP checkpoint、mask、expert与resolver修复。
- 本配置TP-COMPILE-NUMERIC-01修复验证通过；native最新修复、反向跨模式、
  Inductor/目标设备尚待运行，不外推。下一步补入长期独立collective编译回归，
  并复核resolver缓存对可选导入mock测试的影响及最新完整门禁。

TP collective缓存最新质量门禁（2026-10-15）：
- 首轮完整CPU因三个resolver mock用例共享新缓存失败，保留
  /tmp/tp-collective-cache-units.log（3378 passed/3 failed/2 skipped）。
  test_linear_ops.py增加缓存隔离fixture，未删除后端选择与缺autograd拒绝断言。
- 最新全量pytest 3381 passed、2 skipped、218 warnings，35.69s退出0，
  /tmp/tp-collective-cache-final-units.log；适用全仓pre-commit退出0，
  compileall和HEAD diff通过。Ruff首次自动整理导入，复检通过。
- 本批真实跨compile恢复退出0与原始失败及诊断均见上一批。下一动作补充
  持久化集成回归及native修复后验证，未将有限CPU配置扩展到设备完整性。

TP编译回归持久化及native恢复（2026-10-15）：
- 新增tests/integration_tests/tp_compile_collectives.py，真实两rank Gloo，
  生产resolver预解析后生产gather/scatter wrappers以aot_eager fullgraph捕获；
  ones输入、平方sum，独立解析梯度均为4，eager/compiled精确通过，退出0。
  /tmp/tp-compile-collectives-regression.log；命令torchrun两rankIPv4、
  PYTHONPATH=.、GLOO_SOCKET_IFNAME=lo0。此不单独复现FSDP零梯度原故障，
  原故障由真实Trainer集成验收覆盖。
- 原eager/custom保存点，resume切native FSDP+aot_eager，其他world4/TP2/
  EP2/float32/dynamic EMA不变，生产无诊断替换，恢复前后及最终完整状态
  逐位通过、loss差0、退出0；原crosscompile目录
  resume-native-production-cache.log。实际同时覆盖custom→native状态恢复。
- test_linear_ops.py新增resolver缓存后不重复后端探测契约，定向7 passed。
  生产代码未变化，复用上一批全量3381 passed/2 skipped；本轮新增测试
  定向与真实集成结果单列，不改写历史全量计数。HEAD500f166+未提交修复。
- 下一步：反向compiled→eager与Inductor/设备证据仍开放；继续扩大其他迁移
  和失败生命周期审查，避免仅以此CPU组合宣布全局验收。

TP×EP compiled→eager完整恢复（2026-10-15）：
- 当前HEAD500f166+未提交TP布局/mask/专家动态切片/collective resolver修复。
  真实world4/TP2/EP2/PP1、custom FSDP、CPU/Gloo float32、dynamic EMA，
  full设置LLMTUNER_TEST_COMPILE_BACKEND=aot_eager，resume移除该变量，
  各为独立torchrun进程。两phase均退出0。
- step2保存后连续再训2步，与独立eager恢复后2步比较：更新前四rank完整
  状态failures=0，最终模型/optimizer/scheduler/trainer/EMA逐位一致，
  后续loss最大绝对差0。full losses=[4.879703521728516,4.869268894195557,
  4.842996597290039,4.858843803405762]。
- 证据目录/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/
  llmtuner-tp-ep-compile-to-eager-lxanm1m_，含full.log/resume.log、split/final
  快照与DCP。命令沿原pp_checkpoint_equivalence.py full/resume四rank，
  GLOO_SOCKET_IFNAME=lo0、PYTHONPATH=.、IPv4空闲端口；编译变量如上。
- 本批无生产修改；复用最近3381 passed/2 skipped全量结果与后续7项resolver
  单测及collective持久化集成证据，不重复无关门禁。文档格式/diff检查通过。
  本CPU/aot_eager配置双向恢复通过；Inductor、设备、拓扑变化及多轴故障
  协调仍开放，不将双向恢复推广为全部并行矩阵完整验收。

TP checkpoint非strided布局规划拒绝（2026-10-15）：
- ID TP-CKPT-LAYOUT-01 / P1 / tensor_parallel/checkpointing.py。
  静态确认manifest不含layout且局部检查不拒绝sparse；可在metadata成功后
  于contiguous或tensor collective前单rank失败。未运行原版挂起复现，
  因此不宣称已实测原版死锁。
- _prepare_state现在要求匹配tensor为strided，并在manifest包含layout；
  仍由既有ready/error报告传播，不让健康peer进入后续tensor collective。
- tp_checkpoint_coordination.py新增rank0 sparse，保存/加载共10类配置
  两rank均收到错误，每次barrier及最终健康精确重试通过，退出0；日志
  /tmp/tp-checkpoint-layout-coordination.log。命令同原两rank脚本，IPv4、
  PYTHONPATH=.、GLOO_SOCKET_IFNAME=lo0；HEAD500f166+未提交累计修复。
- 范围限已注册一致TP组中的本地规划异常，不含进程死亡、注册损坏、
  collective内部失败及DP交叉组传播。跨组故障仍开放。
- 最新CPU suite 3382 passed、2 skipped、239 warnings，35.66s退出0，
  /tmp/tp-checkpoint-layout-final-units.log；适用全仓pre-commit、compileall
  及HEAD diff均通过。

TP checkpoint无存储meta状态规划拒绝（2026-10-15）：
- TP-CKPT-LAYOUT-01续：meta tensor符合strided/shape条件却无实际存储，
  静态确认可延迟到tensor collective失败；本批在_prepare_state先拒绝meta，
  manifest增加device.type（不比较本地设备编号）。不宣称原版死锁已实测。
- rank0 meta tensor新增保存/加载两项故障；真实两rank CPU/Gloo共12配置
  全部协调抛错，后续barrier和正常gather/load精确重试通过，退出0；日志
  /tmp/tp-checkpoint-meta-coordination.log。命令沿tp_checkpoint_coordination.py
  两rank IPv4、PYTHONPATH=.、GLOO_SOCKET_IFNAME=lo0。HEAD500f166+累计修复。
- 能力边界：仅本地规划及同TP组metadata；设备类型不一致的实际CUDA/CPU
  混合组未运行，DP跨组故障、注册损坏和collective内部故障仍待验证。
- compileall/diff通过；全仓pre-commit首次报告新增用例行宽，formatter修复后
  受影响文件复检通过。完整CPU 3382 passed、2 skipped、244 warnings，
  35.61s退出0（/tmp/tp-checkpoint-meta-units.log）。

MC2上游逆算子与多来源计数复核（2026-10-15）：
- 参考FSDPTurbo HEAD0af4eca242c14ebff254dab1a38045f1395fa520，
  fsdp_turbo/ops/npu/grouped_matmul_mc2.py及distributed/expert_parallel/
  dispatcher_mc2.py。逐项核对本项目mc2_ops.py、mc2_dispatch.py：forward
  trans_gmm_weight=True、dispatch/combine逆通信、weight grad矩阵方向、
  rank-major counts按来源rank求和均一致；未发现该部分新增确认缺陷。
- 适配差异保留：gate/up参数在调用时可微cat；shared expert在MoE层外执行；
  激活使用GroupedExperts.activation_fn；torch_npu/HCCL延迟解析且拒绝缺算子。
  upstream act_limit语义由模型activation适配负责，本批未验证所有模型。
- 原CPU模拟仅world1；新增三项多来源rank计数回归，涵盖world2/world3、
  非均匀和全零，独立期望逐专家列和，int64输出。此仅布局算术，未执行
  融合通信。真实NPU脚本ep_mc2_equivalence.py有空专家/空rank/重计算入口，
  当前无NPU未运行。MC2×TP仍为必需迁移缺口，不能以配置拒绝视为完成。
- 下一动作审查Domino stream/event及张量生命周期，优先可独立验证的机制；
  保留KCP/MC2/设备重叠证据缺口，全局目标未完成。

Domino stream与正常生命周期复核（2026-10-15）：
- 已阅读domino_dispatch.py完整forward/_forward_cuda、token_dispatch.py的
  begin/finish/permute/unpermute/combine；对照FSDPTurbo HEAD0af4eca的
  distributed/expert_parallel/domino_dispatcher.py。当前CUDA生产顺序为
  dispatch0→dispatch1→combine0→combine1；producer wait、ready/completed
  events与跨stream record_stream对应正常数据依赖，无本批新增确认缺陷。
- 差异与缺口：上游torch.accelerator及自定义Domino autograd调度，本项目
  CUDA stream配原生autograd；NPU专用调度与完整反向重叠仍未迁移。
  本地expert抛错后未形成跨rank取消协议，不能用单rankfinally同步stream
  解决其他rank已进入combine的挂起；登记DOMINO-FAILURE-01/P1，下一动作
  明确通信外的expert失败协调边界及所有rank一致collective次序，再故障注入。
- pytest tests/unit_tests/cpu/models/test_domino_experts.py：5 passed，1.28s；
  mock event顺序不能证明真实设备生命周期。实际两rank CPU/Gloo执行
  tests/integration_tests/ep_domino_equivalence.py退出0，非均匀/奇数token、
  空rank/零接收、非连续routing及重计算的输出/输入/score/专家梯度通过
  既有独立全专家参考和原容差。日志/tmp/ep-domino-review.log，命令
  PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2
  --master_addr=127.0.0.1 --master_port=<空闲端口> <脚本>。
- HEAD500f166+已有未提交修复，本批无生产改动；CUDA/NCCL、NPU/HCCL与
  trace重叠未运行，不能以CPU结果关闭迁移。文档格式/diff检查通过。

KCP本地mask边界复核（2026-10-15）：
- ID KCP-MASK-01 / P2 / context_parallel/kcp.py::kcp_kda_forward。
  输入限定[1,local_sequence,hidden]，mask直接逐token乘输入/输出，原先
  未验证mask形状或设备。新增必须为本地[1,sequence]且同设备检查，
  在可选runtime加载、projection及任何CP kernel之前失败。
- test_cp.py新增四种非法mask形状（1D、4D、错误batch、错误sequence），
  assert runtime未触达；定向29 passed。设备不匹配未在目标设备运行。
- 最新完整CPU3389 passed、2 skipped、242 warnings，35.84s退出0，
  /tmp/kcp-mask-review-units.log；受影响pre-commit、compileall、HEAD diff通过。
  该计数包括前批MC2新增三项，本批KCP四项。HEAD500f166+累计未提交修复。
- 已阅读本项目KCP完整runtime/forward/installation；检查上游FSDPTurbo
  modules/kcp.py发现其显式Triton driver选择，当前适配直接导入Ascend kernel，
  driver选择差异待设备环境核实，不能仅按存在差异修改全局driver。
  KCP真实状态交换/梯度、hybrid/packed及设备内核证据仍未完成。
- 下一动作核对上游KCP模型forward与本项目mask/状态上下文契约，明确
  任意额外mask是否能由KDA状态传递表达；CPU输入校验不代表迁移完成。

KCP状态forward与完整hybrid迁移边界（2026-10-15）：
- 参考FSDPTurbo HEAD0af4eca的distributed/context_parallel/kcp/kda_state.py、
  kcp_context_parallel.py及modules/kcp.py。纯KDA的q/k/v卷积、beta sigmoid、
  gate flags、完整head布局、输出gate/norm与当前本项目forward一致。
- KCP-EMPTY-01/P2：空本地序列会构造[0,0]全局cu_seqlens，上游要求严格
  递增；本项目新增非空序列检查，确保在可选依赖及CP kernel之前报错，
  新增runtime未触达回归。该输入不能用于宣称KCP空rank支持。
- KCP-HYBRID-01/P1：上游完整模式通过cp_plan.kcp_function_patches为
  full_attention/compressor_full_attention/transposed compressor安装Ulysses
  MLA，同时KDA走state CP；本项目apply_kcp显式拒绝hybrid，尚无该函数patch
  装配。实现状态为部分迁移，非单纯设备验证缺口；能力矩阵对应行已修正。
- 验收要求：保留HF模型实际MLA调用和压缩布局，生产配置触达hybrid patch，
  KDA/softmax各自状态和序列/head通信布局正确，独立前后向参考及目标内核
  验证；不能用拒绝配置关闭此迁移。下一动作识别本项目实际支持的KDA/MLA
  模型工厂与上游patch目标，确定实例patch契约，避免复制上游全局patch假设。

KCP公开模型工厂可达性核查（2026-10-15）：
- KCP-MODEL-01 / P1 / models/hf/factory.py及KCP入口。
  实际查询当前Transformers CONFIG_MAPPING_NAMES与
  MODEL_FOR_CAUSAL_LM_MAPPING_NAMES：kimi_linear/kimi_k3四项均None；
  已安装models中也无对应Kimi实现。factory离线依赖AutoConfig.for_model，
  类解析仅查transformers导出的PreTrainedModel，不加载仓库外部模型类。
- FSDPTurbo HEAD0af4eca models/kimi/__init__.py显式注册仓库内KimiLinearConfig、
  KimiK3Config和AutoModel，模型还依赖kimi_k3_cp.py的顶层sequence/position
  适配；本项目无对应注册或本地模型迁移。因此当前生产KCP函数可直接patch
  结构匹配实例，但上游Kimi公开配置→模型→训练入口尚不完整。
- 实现状态：部分；验证：当前环境与源码证据确认缺口，不是仅缺Ascend设备。
  验收需本地Kimi配置/类注册、HF wrapper的decoder/embedding/head适配、
  checkpoint命名/共享参数及实际hybrid attention调用均接通；保持可选依赖
  延迟加载并保留上游来源，不能仅注册空类或依赖外部FSDPTurbo安装通过。
- 下一动作从上游KimiLinear模型最小依赖闭包核对可迁移组件，先确定单模态
  causal LM的生产路径，再接hybrid CP与Kimi K3 composite模型契约。
  KCP-MODEL-01与KCP-HYBRID-01独立跟踪，不将内核mock视为模型可达。
- 本批只读审查与文档更新，复用上一批CPU门禁；无新增运行时验收声明。

KimiLinear最小依赖闭包与上游缺失配置（2026-10-15）：
- KCP-MODEL-01续：读取上游modeling_kimi_linear.py导入/类定义与KDA构造、
  modules/short_conv.py及gated RMSNorm。模型直接依赖ShortConvolution、
  ops.attn_res、ops.kda、ops.rms_norm、tensor_cache、设备/日志适配；还有
  SituAndMul按需加载fused_situ_glu，不能只复制模型文件即视为可运行。
- 当前FSDPTurbo HEAD0af4eca且git status干净，但其relative import的
  models/kimi/configuration_kimi_k3.py不存在；rg --files和git ls-files均无
  该文件。KimiLinearConfig/KimiK3Config缺失是当前指定参考的具体来源缺口。
  本批未猜写配置默认值或绕过HF模型契约，也未修改上游仓库。
- 可复用接口：ShortConvolution为带activation属性的depthwise Conv1d，
  其权重形状与本项目KCP卷积调用一致；KDA q/k/v、f_a/f_b、b、gate与
  o_norm/o_proj成员吻合。正常纯KDApatch结构可沿用，但这不证明模型工厂完成。
- 解除来源缺口：取得同版KimiLinearConfig/KimiK3Config的可信源码/固定revision，
  或从实际checkpoint config推导并与模型构造/导出逐字段验收；同时迁移上述
  ops的必要语义与本项目设备抽象。缺失文件不阻断其他并行审查。
- 下一动作推进不依赖该配置的KCP kernel adapter调用契约回归，并复核其他
  必需迁移的实现缺口；保持KCP-MODEL/HYBRID开放。仅文档变更，格式/diff通过。

KCP adapter runtime调用契约回归（2026-10-15）：
- 检查apply_kcp安装流程，未发现需修改的实际wrapper构造失败路径，未为
  人工注入的无关异常新增抽象或回滚。生产实现本批保持不变。
- 新增test_kcp_runtime_contract.py，两项tensor/tuple kernel返回形式，
  检查build_cp_context全局边界[0,8]、kernel_size=3、三次conv与KDA共享
  context、无initial/final state、beta sigmoid只执行一次、safe gate与
  skip_kda_recompute透传；identity测试内核输出及输入梯度保持通过。
- pytest定向2 passed，1.23s退出0，受影响pre-commit/diff通过。该参考为
  参数契约测试替身，不能证明KDA数学、实际state exchange或设备梯度；
  明确不计为目标设备数值验收。HEAD500f166+既有未提交修复。
- 上批空序列生产校验已由30项CP测试通过，最近全量3389/2skip运行在其前；
  累计交付仍需最新完整CPU门禁。KCP-MODEL/HYBRID及设备缺口保持开放。

累计门禁与双CP kernel边界复核（2026-10-15）：
- 最新pytest -q --disable-warnings：3392 passed、2 skipped、231 warnings，
  36.53s退出0，/tmp/llmtuner-review-current-units.log。包含KCP非空校验、
  kernel adapter两返回形式与前批所有累计测试；不复用旧计数冒充本次。
- Git已跟踪及适用未跟踪文件（排除examples/native_alltoall.py）的全仓
  pre-commit退出0；compileall与HEAD diff通过。当前源码164项与逐文件
  清单无遗漏/失效，未将清单覆盖等同于全部运行时验收。
- 已读DualCPFlexKernel完整实现及apply_cp策略入口：Q/K/V先SeqToHead，
  K/V再沿KV组gather，输出HeadToSeq；mask必须已由wrapper按KV Q窗口构造。
  既有dual_cp_attention_equivalence六配置与cp_dual_axis的生产eager flex
  前向证据见能力矩阵，本批未重复执行，也未新增其设备backward声明。
- 当前HEAD500f166及累计未提交源码/测试/文档，未提交推送。下一动作优先
  检查公开双CP输入mask与load-balancer组合，再推进DOMINO-FAILURE-01等
  跨rank异常路径；KCP模型来源、MC2×TP、Domino设备pipeline仍开放。

双CP mask与load-balancer组合审查（2026-10-15）：
- 阅读parallel_config.py的Ulysses轴组合校验、apply_cp的双轴mesh校验和
  _require_ulysses_compatible，以及HFTransformerModel.set_cp_mesh/
  _shard_for_cp/_get_cp_attention_masks：双轴输入按总CP连续分片，Q在
  Ulysses内拼接，mask仅按KV子mesh切分，packed mask在position切分前构造。
- headtail在含Ulysses轴时由配置与运行时双重拒绝，因此当前支持路径不存在
  headtail重排后仍用连续Q偏移的静默错误。本批未发现需修改的确认缺陷，
  也未把该拒绝声明为headtail双CP迁移完成。
- 复用既有独立双轴通信六配置证据；本轮实际pytest
  tests/unit_tests/cpu/test_parallel_plan_config.py
  tests/unit_tests/cpu/parallel/test_input_sharding.py --disable-warnings：
  27 passed，2.32s退出0。未运行设备flex backward或新完整HF双轴组合。
- 代码HEAD500f166+已有未提交改动，本批无生产修改；下一动作优先推进
  已登记DOMINO-FAILURE-01或其他实际失败协议，避免给充分覆盖路径重复加测试。

Domino局部专家异常真实复现（2026-10-15）：
- DOMINO-FAILURE-01/P1由静态风险升级为确认失败协议缺口。
  /tmp/domino_failure_probe.py真实两rank CPU/Gloo，EP2、4全局专家、top2、
  3本地token，rank0 GroupedExperts.forward注入ValueError，超时设5秒。
  rank0报告原异常，rank1在combine报Gloo send timeout，进程终止另报告
  1个unwaited collective。日志/tmp/domino-local-failure-probe.log。
- 脚本捕获预期异常后退出0，仅表示复现执行完毕，不能解释为协议通过；
  未通过后续barrier或健康重试。正常两rank数值证据仍见前批，不覆盖异常。
- 阅读fully_shard_parallel/plan.py与默认分组逻辑：默认decoder unit外层
  hook在专家调用前unshard，但任意pattern可把FSDP单元/hook放在专家子树。
  因而inner_experts调用不能一般性包装为synchronize_local_errors的纯本地
  阶段；其内部可能已有collective，单组finally同步也不能释放peer combine。
- 验收需区分通信前/内核前本地准备、已有dispatch消费与expert FSDP生命周期，
  失败rank和peer在同一组按相同顺序完成/取消协议，再验证barrier及健康重试；
  CUDA还需stream/event/allocator收尾。进程死亡与collective内部错误另列。
  下一动作先确认可保证无collective的专家执行边界，或为该边界显式提供
  上层所有组协调接口；不以只捕获Exception或关闭Domino宣称修复。
- 当前HEAD500f166+累计未提交改动，本批无生产修改；文档格式/diff通过。

Kimi 配置候选来源与消费者核查（2026-10-15）：
- KCP-MODEL-01 / P1 / models/hf/factory.py 与上游 Kimi 模型。
  已取得可追溯候选：本地 vLLM HEAD
  `4e4eab73ed2b4e00931a4636e926b8df1cb1fe50` 的
  `vllm/transformers_utils/configs/kimi_linear.py`、`kimi_k3.py`，
  两文件 git status 无修改，保留 Apache-2.0 及贡献者版权声明。
  该来源解除“无可信配置候选”的来源缺口，不代表与 FSDPTurbo 同版。
- 本轮执行 Python importlib 按文件独立加载 KimiLinearConfig（不导入 vLLM
  包），使用 hidden_size=16、heads=2、layers=2、两层 KDA 的配置。
  AST 提取 FSDPTurbo modeling_kimi_linear.py 中直接 config.<field> 的
  29 个不同成员，实例均具备；to_dict/from_dict 完整字典往返相等，
  is_kda_layer(0/1) 均为 True，退出 0。该检查只覆盖直接成员存在性，
  不证明 getattr 隐式字段、嵌套配置约束、默认值语义或模型数值。
- KimiK3Config 还依赖 vLLM logger 与线性配置模块，迁移时需改为本项目
  定义模块导入，不能增加 vLLM 运行依赖。线性模型依旧直接依赖上游
  short_conv/attn_res/kda/rms_norm/tensor_cache，配置不能单独接入并声明支持。
- 实现状态：KCP-MODEL-01、KCP-HYBRID-01 均保持部分实现、未验收。
  下一动作按候选配置逐项适配本地模型与可选算子依赖，验证 HF 五段式
  wrapper 的构造、前后向、保存恢复，再接 hybrid MLA/压缩器 CP。
  本轮未改生产代码；既有 3392 passed/2 skipped 门禁只按原台账复用。

KimiLinear 配置迁移第一批（2026-10-15）：
- KCP-MODEL-01：新增 models/kimi/kimi_linear_config.py，保留 vLLM
  固定来源与 Apache-2.0 标识，保留模型字段/默认值/一基层号语义。
  不依赖 vLLM，不导入可选内核。模型、算子和公开工厂尚未接入，未注册
  AutoConfig/AutoModel；该新增构件不表示公开训练入口已支持 Kimi。
- 候选实现把 rope_theta 写入调用方 rope_parameters，且持有调用方
  linear_attn_config 的列表引用；迁移后 deepcopy 持有嵌套配置。
  原 assert 配置检查改为显式 ValueError，避免 python -O 跳过。
- 本轮实际验证：test_kimi_config.py 4 passed；使用真实临时目录
  save_pretrained/from_pretrained 对比完整字典（只排除 HF 加载路径字段），
  覆盖 hybrid 层选择、MLA/MoE、未知 kernel 选项、输入字典所有权及缺失列表。
  pytest -q --disable-warnings 全量 3396 passed、2 skipped、241 warnings，
  36.52s，退出 0；日志 /tmp/llmtuner-kimi-config-units.log。
  HEAD500f166 加既有未提交修复及本批配置；compileall、排除受保护示例的
  tracked/untracked 全文件 pre-commit 通过，diff HEAD --check 通过。
- 清单与当前源码集合重新核对，无遗漏、重复或失效路径；新增两行均标记
  迁移未验收。下一步迁移 Kimi 模型/算子适配，接通 HF wrapper 的真实
  前后向与完整 checkpoint；KCP-MODEL/HYBRID 保持开放，NPU 未验证。

Kimi MLA 构件迁移与 head 布局修正（2026-10-15）：
- KCP-MODEL-01：新增 models/kimi/latent_attention.py 与 normalization.py，
  从 FSDPTurbo 0af4eca242c14ebff254dab1a38045f1395fa520 的
  models/kimi/modeling_kimi_linear.py 迁移 MLA 投影与 RMSNorm，保留原版权、
  Apache/MIT 条款及参数键。类名 KimiMLAAttention → KimiLatentAttention；
  本项目无旧消费者，未增加兼容别名。尚无完整模型/工厂入口，迁移未验收。
- 已确认布局差异：kv_b_proj 输出 num_heads 份 latent K/V，原上游却按
  num_heads/num_key_value_heads 设置 num_key_value_groups。HF SDPA 会据此
  再扩展 K/V，heads=2、kv_heads=1 时不符 query heads；迁移设置实际 group=1。
  eager 路径直接使用已展开 K/V，不进行额外 repeat。保留 mla_use_nope 契约，
  不声明未实现的旋转位置编码支持；显式 ValueError 替代该上游 assert。
- 本轮实际验证 test_kimi_latent_attention.py：4 passed，query low-rank
  开/关 × output gate 开/关，FP32、heads=2/kv_heads=1、不等 Q/V head dim、
  5-token causal mask；独立显式投影/softmax/聚合公式比较输出、输入梯度、
  全部参数梯度，atol=rtol=1e-6；HF SDPA 同输入输出也满足相同容差。
  RMSNorm 在参考中共用，因此该实验不独立验收 norm 数学或设备融合路径。
- 累计 pytest -q --disable-warnings：3400 passed、2 skipped、228 warnings，
  35.42s，退出 0，/tmp/llmtuner-kimi-mla-units.log；HEAD500f166 加既有
  未提交改动及本批构件。compileall、适用 tracked/untracked pre-commit
  （排除受保护示例）、git diff HEAD --check 均通过。
- 下一动作迁移 KDA/卷积执行接口及 decoder/causalLM，接通模型工厂后验证
  实际训练与恢复；MoE 上游本体训练会 NotImplementedError，必须适配本项目
  router/专家机制，不能照搬 inference-only 实现。KCP hybrid、目标设备、
  MC2×TP、Domino 异常传播等既有缺口均保持开放。

Kimi 因果卷积训练构件（2026-10-15）：
- KCP-MODEL-01：新增 models/kimi/causal_convolution.py，继承 Conv1d，
  保留上游 channels×1×kernel 权重及 activation='silu' 属性；ATen 执行
  depthwise 左侧历史卷积并裁掉右侧未来位置。packed 边界分段执行，各文档
  从零历史开始；空普通输入仍保留输入和权重的零梯度路径。
- 来源为 FSDPTurbo 0af4eca 的 modules/short_conv.py 参数布局及 Kimi
  KDA 消费者，许可证说明沿用现有 Kimi 构件。差异：此构件是训练用 ATen
  实现，不接受推理 cache，不冒充上游融合或 KCP state-exchange 内核；
  packed 边界 tolist 和 Python 分段仍需后续编译/设备适配。
- 本轮实际 test_kimi_convolution.py 3 passed：float64 普通与 packed
  [0,1,4,6]，独立逐 token/lag 加权求和参考比较输出、输入及权重梯度，
  atol=rtol=1e-12，另覆盖空输入零梯度。未使用生产 conv1d 作为参考。
- 本轮全量 pytest -q --disable-warnings：3403 passed、2 skipped、
  244 warnings，35.71s，退出0，/tmp/llmtuner-kimi-convolution-units.log；
  HEAD500f166 加此前未提交改动与本构件。compileall、适用全文件
  pre-commit（排除受保护示例）、diff HEAD --check 通过。
- 实现状态仍为部分：新构件尚无完整 KDA/decoder 生产消费者，不能作为
  模型入口接通的证据。下一动作继续迁移 KDA gate/kernel 调用、decoder 和
  工厂，验证真实训练与恢复；目标设备性能和 KCP 通信单独验收。

Kimi KDA 训练构件与独立 WY 对照（2026-10-15）：
- KCP-MODEL-01：新增 delta_attention.py，迁移上游 KDA 的投影名称、
  卷积权重、A_log、dt_bias、低秩或全秩 output gate、head RMSNorm。
  新增 KimiGatedRMSNorm 保持 KCP 调用所需 (output, gate) 接口。
  dt_bias 明确零初始化，避免照搬上游 empty 未初始化参数；加载仍使用原键。
- CPU 实现用逐 token state 递推；非 CPU 分支按需导入 FLA 或 Ascend
  chunk_kda，显式 sigmoid beta 一次，关闭内核 beta sigmoid（上游 NPU
  adapter 也作此转换）。无全局算子注册或 FSDPTurbo 运行依赖。
  模型消费者、packed/padding 和 cache 尚未实现，当前显式拒绝，不能据此
  关闭完整模型迁移要求。设备内核调用仅静态检查，未验收。
- 本轮实际 /tmp/check_kimi_recurrent.py：读取固定上游
  ops/cpu/chunk_kda.py，AST 仅移除 registry 导入/注册 decorator，原数学
  实现保持不变；seed14、FP32、B1/T7/H2/K3、chunk8，lower_bound=None/-3，
  对比本项目递推与上游独立 WY chunk 算法：输出 atol/rtol=2e-6，
  q/k/v/g/beta/A_log/dt_bias 七类梯度 atol/rtol=3e-6，均通过、退出0。
  这是 CPU 数学对照，不证明目标设备内核或跨 rank state exchange。
- 持久测试 test_kimi_delta_attention.py 3 passed：两种 output gate 的
  完整投影训练梯度存在且有限，以及首 token 无历史时独立闭式结果。
  全量 pytest -q --disable-warnings：3406 passed、2 skipped、230 warnings，
  38.32s，退出0，/tmp/llmtuner-kimi-kda-units.log；HEAD500f166 加现有
  staged/unstaged 改动与本批新增。工作区出现外部暂存，未调整其暂存状态。
  适用全文件 pre-commit、compileall、diff HEAD --check 通过。
- 下一动作接通 Kimi decoder/causalLM 和公开配置工厂，处理文档/padding
  边界并验证完整恢复；hybrid KCP、NPU/CUDA、MC2×TP、Domino 失败协调
  等全局缺口保持开放。新增文件逐文件台账仍标记迁移未验收。

KDA 文档与 padding 边界适配（2026-10-15）：
- KCP-MODEL-01：delta_attention.py 不再一律拒绝 cu_seqlens 或 mask。
  packed 只接受 batch1、整数一维、从0严格递增到序列长度的边界；按文档
  分段执行，卷积和 delta state 同时重置。布尔 attention_mask 使用 HF
  语义 True=有效位，与 Trainer padding_mask 的 True=padding 必须由未来
  decoder 明确转换；当前无完整 decoder 消费者，不混用两种语义。
- padding 每行移除无效 token 后执行，index_copy 填回原位置；全空行不
  启动内核，保留全部参数零梯度路径。不接受 mask 与 cu_seqlens 同时提供。
  Python 分段/索引、host tolist 仍有 compile 和性能局限，目标设备未验收。
- 本轮实际 test_kimi_delta_attention.py 5 passed（新增2）：packed 两文档
  对比独立调用且后一文档 loss 对前一文档梯度严格为零；不连续有效位及
  全 padding 行验证有效输出、padding 输出及输入梯度为零、参数梯度有限。
  这些是边界不变量，与此前独立 WY 数学对照分开记账，不外推 KCP 通信。
- pytest -q --disable-warnings：3408 passed、2 skipped、240 warnings，
  40.54s，退出0，/tmp/llmtuner-kda-boundaries-units.log；HEAD500f166 加
  既有 staged/unstaged 改动与本轮 delta_attention.py 修改。适用全文件
  pre-commit（保护示例排除）、compileall、diff HEAD --check 均通过。
- 下一步接通 decoder/causalLM、配置工厂和 Trainer 文档边界转换，再验证
  实际训练与完整 checkpoint；KCP hybrid、设备内核及其他迁移缺口仍开放。

Kimi dense decoder 与 HF 工厂接通第一批（2026-10-15）：
- KCP-MODEL-01：新增 models/kimi/model.py，保持上游 embed_tokens/layers/
  norm/lm_head、MLP gate/up/down 参数名；层内使用已迁移 KDA/MLA。
  factory.py 支持 kimi_linear 离线配置与惰性模型类解析；本地/Hub 配置读取
  前注册轻量 KimiLinearConfig，不加载可选内核。MoE/attention residual
  构造仍显式拒绝，不能据此关闭完整迁移；KimiK3 尚未迁移。
- decoder 根据 position_ids 的零起点构造文档边界，KDA 按文档重置；
  MLA 无显式 mask 时构造因果/同文档 additive mask。该行为尚需独立混合
  模型、padding 和 PP/CP 本地分段验收，不能由纯 KDA 测试外推。
- 本轮实际 test_kimi_model.py：1 passed；真实 build_model_config →
  HFTransformerModel，两层 dense KDA、hidden8、heads2、seq4、vocab32，
  前向 logits [4,32]、全部梯度存在且有限、新建 wrapper strict 模型
  state_dict 加载后输出一致。此为模型 state 往返，未验收 optimizer/
  scheduler/数据/RNG 的 Trainer 完整恢复。
- 首次失败为本机 Inductor 预编译头 mtime 过期；保留失败归属为环境缓存，
  用 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 在测试命令中关闭缓存
  后通过，未修改生产全局开关。相同环境全量 pytest -q --disable-warnings：
  3409 passed、2 skipped、229 warnings，54.51s，退出0；
  /tmp/llmtuner-kimi-model-units.log。HEAD500f166 加已有 staged/unstaged
  及本批模型/工厂改动。pre-commit 首次整理一个 import，复查后重跑全部
  适用文件通过；compileall、diff HEAD --check 通过。
- 下一动作适配 MoE 训练和 attention residual、完整 Trainer/恢复与 hybrid
  KCP；当前公开配置仅 dense 子集接通，其他缺口与设备验证保持开放。

Kimi 混合 decoder padding 与 MLA mask 修正（2026-10-15）：
- KCP-MODEL-01：实际检查发现新接入 decoder 把二维有效位 mask 原样交给
  MLA，缺失因果约束且布局错误；KDA 则只接受 bool，但 HF 常用整数0/1。
  现统一验证二维形状/二值语义并转换为 bool 给 KDA，MLA 显式合并 causal
  与有效 key mask 成四维 additive mask。没有改变 Trainer padding_mask
  的相反语义；wrapper 到 decoder 的 padding 透传仍需后续接通验收。
- latent_attention.py eager 原把 bool mask 当加法偏置，已改为屏蔽语义；
  全屏蔽 query 行在 softmax 前清零 scores、softmax 后清零概率，避免
  NaN 输出和梯度，并与 SDPA 零输出契约一致。
- 本轮实际定向6 passed：加入 KDA→MLA dense 模型、非连续有效位0/1
  mask，与去 padding 独立模型调用比较有效 logits（FP32、1e-6），
  全参数梯度有限；其余复用本轮原有 MLA 独立投影测试。
  随后追加 eager bool/all-masked vs 原生 SDPA 测试，MLA 定向5 passed，
  输出1e-6、梯度有限。该追加测试在全量采集后创建，因此全量结果不包含它。
- 本轮全量 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q
  --disable-warnings：3410 passed、2 skipped、238 warnings，38.85s，
  退出0；/tmp/llmtuner-kimi-hybrid-padding-units.log。HEAD500f166 加既有
  staged/unstaged 与本轮模型/mask修改。适用全文件 pre-commit（追加测试前）、
  compileall、diff HEAD --check 通过；追加测试另经 Ruff 与定向验证。
- 下一动作继续接通 wrapper padding 与完整 Trainer 验证、MoE/attn residual
  及 hybrid KCP；这些 CPU 非并行模型证据不证明设备内核/CP 通信或完整恢复。

Kimi wrapper 有效位透传与 padding 零位置（2026-10-15）：
- KCP-MODEL-01：HFTransformerModel 将 Trainer padding_mask 的 True=padding
  明确取反为 decoder valid_token_mask；保留 attention mask 独立传递。
  decoder 验证其形状/dtype/device，仅有效 token 的 position=0 形成文档
  重置，前导 padding 保留零起点；MLA 文档编号也使用过滤后的 reset。
- KDA cu_seqlens 分段可同时传段内有效位 mask，解决 packed 与 padding
  共同出现时此前直接拒绝的问题；每段内部仍按有效位压紧并填回。
  该构件的 Python 分段和 host 边界提取仍未验收 compile/性能。
- 实际定向8 passed：真实 HF wrapper、单层 KDA、非连续 padding、padding
  positions重复0，与压紧输入输出对比1e-6；原有模型/边界测试也通过。
  全量 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q
  --disable-warnings：3412 passed、2 skipped、226 warnings，43.26s，
  退出0，/tmp/llmtuner-kimi-wrapper-padding.log；HEAD500f166 加既有
  staged/unstaged 与本批改动。compileall、适用全文件 pre-commit 通过。
- 全量后补正 MLA 文档编号使用有效 reset，扩展混合模型测试带 padding
  重复零 positions，另执行 test_kimi_model.py；数值证据仍仅 CPU、非并行。
  下一动作完整 Trainer 更新/恢复和 MoE/attention residual、hybrid KCP；
  全局迁移与设备验收仍未完成。

Kimi 真实 Trainer/custom FSDP 完整恢复（2026-10-15）：
- KCP-MODEL-01：扩展 pp_checkpoint_equivalence.py 的公开配置场景，
  LLMTUNER_TEST_KIMI=1 选择 kimi_linear 和两层 KDA；不 patch 模型、
  optimizer、恢复断言或生产算子。其他既有场景默认保持原样。
- 本轮执行两组独立 full/resume（每阶段 torchrun 两个 Gloo rank），
  IPv4 master_addr=127.0.0.1、GLOO_SOCKET_IFNAME=lo0、PYTHONPATH=.，
  LLMTUNER_TEST_KIMI=1、PP_SIZE=1、FSDP_IMPLEMENTATION=custom，
  TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0；第二组额外 EMA=dynamic。
  模型 hidden64、heads4、KDA head16、两层、kernel3、vocab128、seq32、
  FP32/CPU、TP/CP/EP1、AC/compile关闭；连续4步，step2保存，新运行时续2步。
- 四阶段均退出0。两rank恢复后更新前完整 runtime 状态 failures=0；
  完整恢复验收沿用脚本原有严格模型、Adam moments、scheduler、trainer、
  loader/RNG与启用EMA状态及最终状态比较；后续 loss 差0，未放宽任何断言。
  两组连续 loss=[4.874064,4.863374,4.880116,4.850170]（此处仅6位展示），
  resumed 后两步精确相等。首次真实 Trainer 装配没有新增生产缺陷。
- 证据目录分别为 /var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/
  llmtuner-kimi-trainer-_r3j5qol 与 llmtuner-kimi-trainer-fv6zx6b5；
  full.log/resume.log、split/final状态快照与DCP由脚本生成。
  HEAD500f166 加当前 staged/unstaged Kimi/并行修复；本轮仅修改测试场景。
  适用全文件 pre-commit、compileall、diff HEAD --check通过；CPU全量复用
  上批3412/2skip及最后模型定向3通过，无生产变化无需重复全量。
- 实现仍为部分：本证据只关闭 dense KDA/customFSDP2 同模式恢复的验收
  缺口，不证明 Kimi MoE、hybrid attention、KCP/TP/PP、compile 或目标设备。
  下一动作继续迁移 MoE/attention residual 与 hybrid KCP，保留其余矩阵缺口。

Kimi 本地可求导 MoE 迁移（2026-10-15）：
- KCP-MODEL-01：新增 kimi/moe.py，从 FSDPTurbo 0af4eca 的模型迁移
  router sigmoid/softmax、group top2 限制、correction bias、原分数归一化及
  scaling；保留 gate.weight/e_score_correction_bias 和 experts.N.w1/w2/w3
  键。删除上游 checkpoint 任意截断，保持严格 shape 加载；bias明确零初始化。
- 本地专家按路由索引计算并 index_add 合并，训练不再调用上游 inference-only
  no_grad 路径；shared experts 仅非零数量时构造，支持 latent down/up/norm
  构件。decoder 按 first_k_dense_replace/moe_layer_freq 选择 sparse block。
  attention residual 仍未迁移；EP block 检测/转换尚未接通，保持实现缺口。
- 本轮 test_kimi_moe.py 1 passed：seed4、FP32、E4/top2、hidden8/F12、
  五token，独立线性投影/sigmoid/topk 权重与全部专家枚举参考，比较输出、
  输入及参数梯度。group/shared/latent/设备数值及并行尚未验收。
- 全量测试（关闭本机失效 Inductor PCH 缓存）3413 passed、2 skipped、
  233 warnings，39.16s，退出0，/tmp/llmtuner-kimi-moe-units.log；
  HEAD500f166 加既有 staged/unstaged及本批模型/MoE修改。
  compileall、适用全文件 pre-commit、diff HEAD --check通过。
- 下一动作补齐 EP 检测/布局转换与独立 grouped/shared/latent 数值参考，
  验证公开 Trainer 的 MoE 更新与恢复，继续 attention residual/hybrid KCP。

Kimi grouped/shared/latent MoE 验收与组配置边界（2026-10-15）：
- KCP-MODEL-01：扩大 test_kimi_moe.py 独立枚举参考，覆盖普通、group2/
  retained1+shared1、group2/retained1+shared1+latent4+norm 三配置。
  专家选择参考用两专家组显式求和/argmax，不调用生产 router；参考计算
  全部专家再稀疏权重加和，比较输出、输入及全部可导参数梯度通过。
  shared/latent 子模块在参考中复用，因此只证明组合语义，不独立验收其
  Linear/RMSNorm 数学。FP32，hidden8、E4/top2、expert12、五token。
- 确认非法分组原会在 forward topk/reshape 时报底层错误或从排除组取专家；
  KimiMoEGate 构造增加整除、retained范围、组内top2容量及top_k容量校验。
  新增三项失败配置回归（E5/group2、E4/group4、retained容量不足）。
- EP 静态复核：hf_inspection 仅融合 gate_up/down，MOE_LAYER_ATTRS 仅mlp；
  当前 Kimi block_sparse_moe + ModuleList w1/w2/w3 未进入转换路径。
  需要明确 checkpoint 分组键迁移和 latent wrapper，不能仅改检测器后
  视为支持；本轮未修改 EP 布局或声明 EP 验收。
- 全量 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q
  --disable-warnings：3418 passed、2 skipped、241 warnings，45.02s，
  退出0，/tmp/llmtuner-kimi-moe-variants.log。HEAD500f166 加既有
  staged/unstaged 与本批 router 校验/测试。pre-commit 首次格式化一文件，
  复查并重新执行全部适用文件通过；compileall/diff HEAD --check通过。
- 下一动作继续迁移 Kimi EP 识别与专家/checkpoint布局，随后真实 Trainer
  MoE/EP 更新恢复；attention residual、hybrid KCP 与目标设备仍未完成。

Kimi router 非连续与空 token 修复（2026-10-15）：
- KCP-MODEL-01：复核 EP 参数转换契约时发现本地 router flatten view
  不接受部分非连续 B/T 布局；改为 reshape。所有 token×expert/group
  reshape 均显式给出专家维度，消除空token [0,-1] 的歧义。
- 新增 noncontiguous/empty 两项回归：B2、T3或0、hidden8、E4/top2、
  group2/retained1，输出与连续副本一致，反向输入有限且所有专家梯度存在。
  首次空输入测试失败于 score_mask 的第二处 [0,-1]；补正后定向8通过。
- 首次全量采集旧实现，1 failed、3419 passed、2 skipped，37.90s，
  /tmp/llmtuner-kimi-router-layout.log，保留该失败记录；补正后重新执行
  TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings：
  3420 passed、2 skipped、245 warnings，39.04s，退出0，
  /tmp/llmtuner-kimi-router-layout-fixed.log。HEAD500f166 加既有
  staged/unstaged 与本轮router布局/测试改动；compileall、适用全文件
  pre-commit（保护示例排除）、diff HEAD --check通过。
- 本轮未接通 EP 转换。仍需 Kimi ModuleList 参数堆叠、检查点键双向转换、
  correction bias 与 latent wrapper 契约；保持 EP 与全局迁移未验收。

Kimi 本地 MoE 的真实 Trainer/FSDP 恢复（2026-10-15）：
- KCP-MODEL-01：pp_checkpoint_equivalence.py 增加 KIMI_MOE=1 场景，
  通过真实 ModelConfig arch_overrides 设置 E4/top2、group2/retained1、
  expert48、shared1，不使用模型替身、不更改状态或数值断言。
- 本轮实际两阶段 IPv4 torchrun nproc2，KIMI=1、KIMI_MOE=1、PP_SIZE=1、
  FSDP_IMPLEMENTATION=custom、EMA=dynamic；CPU/Gloo/FP32，hidden64、
  heads4、两层KDA、seq32、vocab128、TP/CP/EP1、AC/compile关闭；
  PYTHONPATH=.、GLOO_SOCKET_IFNAME=lo0、PCH缓存关闭。
  连续4步step2保存，独立runtime恢复再2步，两阶段退出0。
- 两rank恢复后更新前状态 failures=0；脚本完整模型、Adam、scheduler、
  trainer、loader/RNG与动态EMA恢复和最终状态严格比较通过；后续loss差0。
  连续loss（展示6位）[4.860115,4.870937,4.872646,4.873839]。
  证据：/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/
  llmtuner-kimi-trainer-zwg38ya5/{full,resume}.log 及状态快照/DCP。
- HEAD500f166 加既有 staged/unstaged 修复与本轮集成场景；工作区有外部
  暂存状态变化，未调整索引。本轮仅测试/文档变更，CPU全量复用上一批
  3420/2skip；适用全文件pre-commit、compileall、diff HEAD --check通过。
- 该证据关闭本地MoE+customFSDP2同模式恢复缺口，不证明EP分片、latent
  分布式、compile、attention residual、hybrid KCP或目标设备正确性。
  下一步仍需逐专家布局到EP grouped参数及checkpoint键转换，保持全局未完成。

Kimi decoder 输入契约复核（2026-10-15）：
- KCP-MODEL-01：开始EP契约核查时观察到外部HEAD更新为f188f67
  feat(kimi)，保留当前索引与全部已有改动，未执行commit/push。
- 确认 KimiTextDecoder 只检查use_cache，却会在kwargs中静默丢弃非空
  past_key_values，导致用户给定历史被忽略；现明确拒绝非空cache。
  position_ids 新增整数类型、B/T形状、设备校验，避免错误边界元数据
  在KDA分段或MLA mask构造中被接受。推理cache仍未实现，未冒充支持。
- test_kimi_model.py 本轮4 passed，新增非空cache与错误位置长度回归。
  全量 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q
  --disable-warnings：3421 passed、2 skipped、243 warnings，40.26s，
  退出0，/tmp/llmtuner-kimi-decoder-inputs.log。代码状态HEADf188f67加
  既有未提交修复与本轮decoder校验/测试；compileall、适用全文件
  pre-commit、diff HEAD --check通过。
- EP逐专家布局/checkpoint转换尚未实现；下一动作继续该迁移，
  attention residual、hybrid KCP、MC2×TP、Domino异常协调及目标设备
  等全局缺口仍保持开放。

KDA 持久独立 affine 状态参考（2026-10-15）：
- KCP-MODEL-01：将多 token 数学验收补入仓库持久测试，而非仅依赖临时
  上游 WY 实验。参考把每个 value column 的状态更新写为4×4增广 affine
  变换，并累乘时间前缀；不调用生产 recurrent_kda 的状态更新或输出helper。
- seed19、FP32、B1/T5/H2/K3，lower_bound=None/-3 两配置；输出比较
  atol=rtol=2e-6，q/k/v/raw_decay/beta/A_log/dt_bias 七类梯度比较
  atol=rtol=3e-6。定向 test_kimi_delta_attention.py 7 passed，退出0。
  该证据独立检查历史 decay 与 delta 修正，不证明 fused/KCP设备通信。
- 本轮仅新增测试，生产未变；HEADf188f67加既有未提交修复。Ruff检查及
  格式通过，CPU全量复用上一批3421/2skip；下一动作继续EP布局转换，
  attention residual/hybrid KCP与其他必需迁移仍未完成。

Kimi SiTU 激活实际实现迁移（2026-10-15）：
- KCP-MODEL-01：确认本地 dense/expert MLP 直接 ACT2FN['situ'] 查找，
  未注册该激活导致配置不可构造。新增 common/activation.py::SituGLU，
  使用既有(gate,up)契约；KimiMLP/KimiExpertMLP显式选择，shared复用
  KimiMLP。不全局修改HF ACT2FN，不添加可选内核导入。
- 对齐 FSDPTurbo 0af4eca ops/cpu/situ_glu.py：FP32 gate/up，
  beta*tanh(gate/beta)*sigmoid(gate)，可选linear_beta*tanh(up/linear_beta)，
  输出还原输入dtype；显式校验beta与linear_beta有限范围。当前为ATen，
  NPU融合性能/数值未验收，不把此实现当作设备融合算子迁移完成。
- test_situ_activation.py 4 passed：linear_beta=None/0/1.7 独立显式
  公式输出与gate/up梯度；dense/expert两类真实MLP前后向通过。
  全量 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q
  --disable-warnings：3427 passed、2 skipped、226 warnings，36.97s，
  退出0，/tmp/llmtuner-situ-units.log。HEADf188f67 加已有未提交修复与
  本轮激活/MLP改动；compileall、适用全文件pre-commit、diff检查通过。
- 下一动作仍为Kimi EP逐专家布局/checkpoint转换与attention residual、
  hybrid KCP，未关闭整体模型迁移或目标设备验证缺口。

Kimi attention residual 聚合构件（2026-10-15）：
- KCP-MODEL-01：新增 attention_residual.py，对齐 FSDPTurbo 0af4eca
  ops/cpu/attn_res.py；历史block与当前prefix拼接，RMS仅用于投影打分，
  softmax权重聚合原始幅值，FP32计算后恢复dtype。保留上游版权与许可证。
- 本轮独立逐token/逐block枚举参考，history0/3、tokens2/hidden4、FP32，
  输出和prefix/history/projection/norm梯度atol=rtol=1e-6，定向2通过。
  首次history0参考未连接空history梯度而失败，修正参考加入数学零依赖，
  生产实现未改。Ruff/格式、compileall、diff检查通过。
- 尚未接入decoder的block生命周期和最终聚合，不能声明attention residual
  配置可训练；下一动作接通层间prefix/block状态并验证完整模型与恢复。
  本轮新增构件，累计门禁交付需后续完整回归；设备融合/并行未验证。

Kimi attention residual 层间生命周期接通（2026-10-15）：
- KCP-MODEL-01：decoder layer注册上游self_attention_res/mlp_res的norm/proj
  原键，按layer_idx/block_size在边界保存prefix，attention前聚合旧block，
  attention后prefix累加、MLP前再次聚合；decoder最后output_attn_res聚合。
  构造不再一律拒绝attention residual，block_size显式正整数校验。
- 实际测试history0/3独立聚合参考加三层KDA block_size1/2训练，共4通过。
  首次测试假设所有参数都有梯度而失败；上游第一层空history不执行attention
  打分，修正为精确断言仅layer0 self_attention_res_norm/proj两权重无梯度，
  其余全部存在且有限。未人为创建梯度或改变上游执行语义。
- 初次全量旧断言2 failed、3429 passed、2 skipped，36.15s，
  /tmp/llmtuner-attnres-model.log；修正后全量PCH关闭pytest：3431 passed、
  2 skipped、249 warnings，36.51s，退出0，
  /tmp/llmtuner-attnres-model-fixed.log。HEADf188f67加既有未提交修复与
  本批model/attention_residual构件；适用pre-commit、compileall、diff通过。
- 实现状态部分：完整层间生命周期尚需独立数值oracle和真实Trainer恢复，
  PP需跨stage传递block历史，不能由当前普通hidden输出外推；KCP/设备
  融合、EP转换等仍开放。下一动作验证生命周期数值及恢复并接并行契约。

Attention residual 完整参考与真实 Trainer 失败（2026-10-15）：
- 本轮增加历史list逐block评分/加权参考，三层KDA、block1/2，比较完整
  logits及参数梯度；共用KDA/MLP/norm子模块，仅独立验证block生命周期。
  一次未固定seed运行block2的embedding梯度超2e-6，随机状态未留存，不能
  重现该精确输入；随后固定seed19两配置均通过（定向4），此通过不关闭
  首次差异。下一步保留多seed诊断并报告首键实际误差，禁止只选通过seed。
- ATTNRES-FSDP-01/P1：集成脚本新增LLMTUNER_TEST_ATTN_RES设置。
  实际KIMI1/ATTN_RES1/customFSDP2/EMA dynamic/CPU FP32双rank full阶段
  失败退出1，resume未启动。accelerator/collectives.py::_local_norm在
  torch.nn.utils.get_total_norm的foreach_norm收到混合Tensor/DTensor，
  两rank相同报错；未改生产裁剪或弱化断言。日志目录
  /var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-kimi-trainer-y6l1vlxz。
- 实现状态部分/验证失败。下一动作定位首个普通梯度的参数所有权、FSDP
  state和unused参数交互，再修复真实根因并重新执行full/resume。
  本轮仅测试场景/参考与台账变化；Ruff/diff通过，未重跑全量，不把
  seed19参考通过声明为生命周期全面验收。HEADf188f67+既有改动。

ATTNRES-FSDP-01 参数归属诊断（2026-10-15）：
- 本轮实际重复双Gloo rank的真实Trainer full失败，临时诊断仅在裁剪前打印
  参数/梯度类型与root state，不改变模型执行或梯度。output_attn_res_norm
  和output_attn_res_proj的weight均为Parameter/Tensor，而root
  _fsdp_param_group为None，两rank一致。证明是新增输出参数未纳入root
  sharding，不能靠裁剪跳过或混合范数封装关闭缺陷。
- 日志 llmtuner-kimi-trainer-3n92qqfp/full.log 与
  llmtuner-kimi-trainer-t8bmbc5w/full.log 位于既有macOS临时根目录。
  full退出1，resume未运行；无生产变更。
- 已读取fully_shard_parallel.py边界/root单位构造与PyTorch2.14
  _get_modules_and_states，下一动作检查_get_managed_modules在norm/head
  多模块边界状态下的遍历截断，确定为何decoder剩余参数被排除。
  缺陷保持开放；目标设备与其余迁移未验收。

ATTNRES-FSDP-01 根因修复与 PP 回归收尾（2026-10-15）：
- HEAD f188f679e2325ded37ae1da46153c4ae6b511682 + 当前未提交改动。
  HFTransformerModel.named_children 原先只呈现五个标准部件，导致 FSDP
  DFS 漏掉 decoder 的 output_attn_res_norm/proj；root 参数组为空，
  裁剪遇到普通 Tensor 与 DTensor 混合。现于标准部件后呈现真实注册父模块，
  保持 state_dict 路径；新增递归遍历测试覆盖额外参数与 buffer。
- 首轮全量失败保留：/tmp/llmtuner-traversal-fix.log，退出1，
  3425通过/6失败/2跳过。新增父模块被 PP 当作未分配普通模块替换为 Identity，
  导致阶段参数丢失；partition 现识别包含已呈现部件的结构父模块并保留。
  不通过修改原 PP 数值或共享参数断言绕过回归。
- 本轮 pytest -q tests/unit_tests/cpu/models/test_hf_wrapper.py
  tests/unit_tests/cpu/parallel/test_pipeline.py --disable-warnings：
  退出0，119通过。完整 CPU 结果见本批后续门禁记录。
- 本轮 LLMTUNER_TEST_ATTN_RES=1 LLMTUNER_TEST_EMA=dynamic
  python /tmp/run_kimi_trainer.py：full/resume均退出0。生产脚本为
  tests/integration_tests/pp_checkpoint_equivalence.py；临时启动器仅安排
  两次独立 torchrun、IPv4端口及日志。CPU/Gloo world2、FP32、custom FSDP2，
  TP/PP/CP/EP1、AC/compile关闭、两层dense KDA、block_size1、动态EMA。
  两rank恢复后更新前state failures=0；四步loss为
  [4.877277374267578,4.855770111083984,4.872187614440918,4.856586456298828]，
  恢复后两步及最终完整状态严格一致，loss最大绝对差0。
  原始full.log/resume.log在macOS临时根目录llmtuner-kimi-trainer-m47l814t。
- 实现已修复此参数归属缺陷；Kimi PP历史传递、EP转换、目标设备验证仍开放。
  生命周期独立参考的多seed梯度误差问题仍开放，不能用seed19结果关闭。

ATTNRES-FSDP-01 本批最终门禁（2026-10-15）：
- TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings：
  退出0，3432通过、2跳过、230警告，42.97秒；
  /tmp/llmtuner-traversal-pp-fix.log。该环境变量仅用于本机测试缓存问题。
- 排除 examples/native_alltoall.py 后，对Git已跟踪及适用未跟踪文件执行
  pre-commit run --files：最终退出0，全部hook通过。初次因partition注释
  超88字符失败，已修正；未改动受保护示例。
- python -m compileall -q llmtuner 与 git diff HEAD --check 均退出0。
  复用前述119项定向测试与双rank恢复证据；全局目标及其他必需迁移仍未完成。

ATTNRES-NUMERIC-01 / P2 / Kimi attention residual 独立参考复核（2026-10-15）：
- 路径与调用方：tests/unit_tests/cpu/models/test_kimi_attention_residual.py，
  KimiLinearForCausalLM 的三层KDA、block_size1/2；生产实现未变。
  HEAD f188f679e2325ded37ae1da46153c4ae6b511682 + 既有未提交改动。
- 契约与复现：原逐候选除以sqrt、标量加权求和的生命周期参考，在seed0..31
  共64例中有7例不满足原2e-6梯度容差：seed1/block1及
  seed1,2,5,6,8,17/block2。首个失败多为embed_tokens.weight，
  seed8/block2为layers.0.self_attn.k_proj.weight。
  仅换rsqrt或预乘norm/projection仍有6例失败；seed1/block1首个失败键
  embedding的违规元素最大绝对差5.9604644775390625e-6（原参考）；
  rsqrt+权重预乘变体为6.198883056640625e-6。
  临时诊断记录/tmp/attnres_order_results.log，运行退出0（显式收集失败，
  不视为原参考通过）；首次因/tmp脚本缺PYTHONPATH退出1，设PYTHONPATH=.后运行。
- 根因边界：该数学参考改变FP32 RMS归一化、候选归约及权重求和计算顺序，
  与固定上游ops/cpu/attn_res.py的FP32执行契约不同。
  保留原失败，不以筛seed或放宽断言关闭问题。
- 实现/验证：生命周期参考独立维护history列表，但聚合遵循FSDPTurbo
  0af4eca242c14ebff254dab1a38045f1395fa520 ops/cpu/attn_res.py 的FP32顺序，
  不调用生产attention_residual。原梯度2e-6容差保持；logits加强为逐位一致。
  seed0..31 × block1/2全64例通过，覆盖全部参数梯度及恰好两个unused参数。
  另保留逐token独立数学公式测试，扩为seed0..7 × history0/3全16例，
  输入、历史、projection、norm梯度均维持1e-6容差。
- pytest -q tests/unit_tests/cpu/models/test_kimi_attention_residual.py
  --disable-warnings：退出0，80通过，4.62秒。
  上游顺序参考若强行要求梯度逐位相等，64例embedding均有舍入差异；
  例如seed1/block2最大9.5367431640625e-7，history堆叠图不同不能宣称位相等。
  正式测试未要求此前不存在的梯度逐位契约。
- 验收：关闭本配置CPU生命周期参考待定位项；不由此证明全模型FP64误差界、
  任意seed、其他dtype、PP/EP、编译或目标设备正确。
  下一动作继续Kimi EP的专家布局、latent wrapper与checkpoint双向适配。

ATTNRES-NUMERIC-01 最终门禁（2026-10-15）：
- TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings：
  退出0，3508通过、2跳过、248警告，44.01秒；
  /tmp/llmtuner-attnres-multiseed.log。较前批增加76个参数化场景。
- 全部适用文件pre-commit初次仅B023失败（循环中的错误消息lambda未绑定name）；
  已用默认参数绑定，随后针对本批测试与文档重跑全部适用hook，退出0。
  排除受保护examples/native_alltoall.py，其他文件沿用该次已通过结果。
- python -m compileall -q llmtuner、git diff HEAD --check、定向ruff均退出0。
  本批无生产行为变化，前批真实Trainer/FSDP恢复证据继续有效，未重复运行。
- 全局目标仍进行中。Kimi EP下一批必须同时检查block_sparse_moe发现、
  ModuleList到stacked权重转换、latent维度与shared ownership、correction bias
  Parameter到状态buffer的显式迁移、模型/optimizer/EMA键双向映射，
  禁止仅扩大探测器后直接调用现有fused-only转换路径。

KIMI-EP-01 / P1 / Kimi EP 生产转换接通（2026-10-15）：
- 生产路径：apply_ep → swap_hf_moe_blocks → kimi_conversion.convert_kimi_block。
  新增llmtuner/parallel/expert_parallel/kimi_conversion.py（已阅读/CPU数值通过，
  迁移未全面验收）；模型block_sparse_moe纳入共享发现与训练hook。
  HEAD f188f679e2325ded37ae1da46153c4ae6b511682 + 既有未提交改动。
- 契约/实现：Kimi ModuleList w1/w2/w3复制为当前EP rank的GroupedExperts切片，
  native alltoall分发，SwiGLU/SituGLU保持原语义。router在模型宽度路由，
  latent down在分发前，norm/up在加权combine后；shared expert使用原输入。
  router、shared、latent模块及correction bias保留原参数身份/模式/dtype。
  bias继续为Parameter，不启用原模型没有的sign更新；不是静默变成buffer。
  不同冻结状态或专家共享无法在stack后保留，安装前明确拒绝。
  TP、quantile及非alltoall后端的Kimi适配仍未实现，拒绝不视为迁移完成。
- CPU定向pytest test_ep_model_replacement.py + test_kimi_conversion.py：
  初版76通过；含12个Kimi输出/输入梯度/专家及latent/shared/router梯度场景，
  topk1/2、普通/latent、SiLU/Situ、非零correction bias，输出1e-6、梯度2e-6。
  新转换实例strict native state往返通过。后续补充整专家模块共享拒绝测试。
- 独立通信数值：PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun
  --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29839
  tests/integration_tests/kimi_ep_equivalence.py，退出0。CPU/Gloo/FP32，
  EP2/TP1/PP1/CP1，无FSDP/AC/compile/EMA；8场景包含latent有无、
  SiLU/Situ、rank0空输入或不均匀3/5token，均含shared与group2/top2。
  对照独立完整逐专家Kimi实现，比较输出、输入梯度、EP切片对应的全局专家
  梯度和rank本地router/shared/latent梯度；容差同上。
  日志/tmp/kimi-ep-equivalence-fixed.log。初次门控错用不存在的distributed
  capability退出1，修为仓库spmd_types门控后实际执行成功，未绕过collective。
- 真实Trainer native恢复两场景：LLMTUNER_TEST_KIMI_MOE=1
  LLMTUNER_TEST_EP_SIZE=2 LLMTUNER_TEST_EMA=dynamic
  python /tmp/run_kimi_trainer.py，普通场景full/resume均0；增加
  LLMTUNER_TEST_KIMI_LATENT=1的latent场景同样均0。
  生产脚本pp_checkpoint_equivalence.py；custom FSDP2/EP2/其余轴1，
  FP32 CPU/Gloo、AC/compile关闭、dynamic EMA。新runtime更新前完整状态
  两rank零失败，后续loss差0，最终完整状态严格一致。
  普通四步loss[4.860115,4.870937,4.872646,4.873839]；
  latent四步[4.859194,4.879417,4.875623,4.871466]（此处显示6位）。
  macOS临时根目录llmtuner-kimi-trainer-3tm4147t、-rp9cue1l留存原始日志。
- 持久化边界：native模型/optimizer/EMA使用转换后FQN并由现有EP gather/slice
  保存恢复，本批真实恢复已覆盖；与未转换Kimi checkpoint不自动互通。
  原始HF逐专家键与grouped键双向映射、跨EP开关恢复仍缺；不宣称HF兼容完成。
  下一动作完成该映射并验证HF导出/加载、native跨拓扑边界，以及更多设备组合。

KIMI-EP-01 最终门禁（2026-10-15）：
- 首轮全量3522通过/2跳过/240警告，35.37秒；日志
  /tmp/llmtuner-kimi-ep-suite.log。补充同一专家模块别名拒绝后，定向15通过；
  最终全量3523通过/2跳过/238警告，37.78秒、退出0，
  /tmp/llmtuner-kimi-ep-final.log。
- 全量适用pre-commit（Git跟踪+适用未跟踪，排除受保护示例）退出0；
  后续仅注释/上游来源记录变化的定向pre-commit同样退出0。
  compileall、git diff HEAD --check均退出0。
- 本批不关闭HF双向映射与跨格式恢复；不会把native恢复外推为原格式兼容。

KIMI-HF-STATE-01 / P1 / Kimi EP 与原HF权重布局双向映射（2026-10-15）：
- 新增models/kimi/checkpoint.py（已阅读/CPU通信验证，完整迁移未验收）：
  full grouped w1_EFD/w2_EDF/w3_EFD与逐专家w1/w2/w3.weight双向转换；
  router.gate→gate、expert_bias_E→gate.e_score_correction_bias，latent模块
  去除/恢复routed_experts前缀。shared键不变；碰撞、缺失专家、越界索引、
  rank错误及未gather本地专家切片明确报错，不丢弃未知键。
- HFTransformerStateDictAdapter新增keyword-only kimi_expert_parallel=False，
  默认行为不变；trainer装配仅对kimi_linear且EP>1启用。from_hf不靠缓存
  猜目标布局；普通未转换Kimi仍保持原格式。映射在现有EP gather后和slice前，
  支持meta加载模板；native optimizer/EMA序列化不改，跨EP开关训练状态未实现。
- 生产加载限制：assembly的EP初始HF加载拒绝仍保留，尚未由映射单测外推为
  真实Trainer HF初始化可用；须补完整公开配置/磁盘加载后再解锁。
- 单测初次26通过/2失败：两个完整模型forward触发macOS Inductor陈旧PCH
  mtime错误，映射/strict加载断言已通过。仅测试命令设
  TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0后，
  pytest -q tests/unit_tests/cpu/models/test_kimi_checkpoint.py
  tests/unit_tests/cpu/models/test_state_dict_adapter.py --disable-warnings：
  退出0，28通过。后续增加真实safetensors写读后strict加载，最终结果见门禁。
- 双rank验证：PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun
  --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29839
  tests/integration_tests/kimi_ep_equivalence.py，退出0；日志
  /tmp/kimi-ep-hf-mapping.log。EP2/其余轴1、CPU/Gloo/FP32、无FSDP/AC/compile/EMA，
  普通/latent × SiLU/Situ × 不均匀/空rank共8场景，在前批梯度对照外新增
  ModelCheckpointState真实收集→HF导出→另一独立模型导入/EP切片，
  原始HF全部键/权重逐位一致，重新分片后的输出逐位一致。
- HEAD f188f679e2325ded37ae1da46153c4ae6b511682 + 既有未提交改动。
  实现部分/CPU模型映射通过；HF初始加载生产验收、native跨EP训练状态迁移、
  FSDP分片HF存储及目标设备仍开放。下一动作验证Trainer公开HF初始化链路。

KIMI-HF-STATE-01 最终验证（2026-10-15）：
- TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings：
  退出0，3528通过/2跳过/229警告，37.16秒；
  /tmp/llmtuner-kimi-hf-mapping-suite.log。含真实safetensors文件写读回归。
- 重跑真实Trainer：LLMTUNER_TEST_KIMI_MOE=1 LLMTUNER_TEST_KIMI_LATENT=1
  LLMTUNER_TEST_EP_SIZE=2 LLMTUNER_TEST_EMA=dynamic
  python /tmp/run_kimi_trainer.py，full/resume均0；macOS临时根目录
  llmtuner-kimi-trainer-lmu4r4gv。custom FSDP2/EP2/其他轴1、CPU/Gloo/FP32，
  AC/compile关闭，dynamic EMA；完整状态/后续loss严格一致，loss差0。
  该测试验证adapter装配未破坏native恢复，不代表HF初始化已验收。
- 全适用文件pre-commit排除受保护示例后全部通过；compileall与diff检查退出0。
  初次Ruff两处测试长行已修正，未修改断言。全局目标仍未完成。

KIMI-HF-INIT-01 / P1 / Trainer EP 原始HF初始化（2026-10-15）：
- 路径：trainer/runtime/assembly.py → meta HFTransformerModel → EP/FSDP →
  materialize_meta_model → DistributedCheckpointManager.load → adapter → EP slice。
  读取当前目标与工作区；前批已形成Kimi双向权重映射，本批生产入口按
  text_config.model_type==kimi_linear放行EP初始HF加载，其他家族继续拒绝。
- 确认缺陷：真实首次运行在meta分配时失败，MoE.tokens_per_expert_E非持久
  buffer无重建规则；/tmp/kimi-hf-initial-01.log，torchrun退出1，两rank同错。
  materialization现仅对native MoE计数器重建FP32零值，未知buffer仍拒绝。
  新CPU回归使用真实meta转换证明计数器在存储分配后为全零。
- 新tests/integration_tests/kimi_hf_initial_load.py：真实HF原始逐专家
  safetensors写盘、公开CheckpointConfig初始加载、真实Trainer meta/FSDP/EP，
  收集后全部权重逐位对照源模型，前向1e-6对照，执行一次真实optimizer更新。
  未mock模型/通信/加载器，不以strict=False忽略状态。
- 本轮命令公共前缀：LLMTUNER_TEST_KIMI=1 LLMTUNER_TEST_KIMI_MOE=1
  LLMTUNER_TEST_KIMI_LATENT=1 LLMTUNER_TEST_EP_SIZE=2 LLMTUNER_TEST_PP_SIZE=1
  PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0
  TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0，另设
  LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom，执行torchrun --nproc_per_node=2
  --master_addr=127.0.0.1 --master_port=29839
  tests/integration_tests/kimi_hf_initial_load.py /tmp/kimi-hf-initial-03，
  退出0；日志同路径.log。前一修复后/tmp/kimi-hf-initial-02仅权重与前向
  验证同样退出0；03增加更新。CPU/Gloo/FP32、custom FSDP2/EP2，其余轴1，
  latent32+norm/shared1/group2/top2，AC/compile/EMA关闭。
- HEAD f188f679e2325ded37ae1da46153c4ae6b511682 + 既有改动。
  实现已接通限定Kimi HF初始化；默认native FSDP同场景验证与最终CPU结果
  见本批后续记录。跨EP开关native optimizer/EMA迁移、FSDP下HF导出存储、
  TP/PP/CP组合及目标设备仍不由该结果推导完成。

KIMI-HF-INIT-01 最终验收补充（2026-10-15）：
- 同命令仅FSDP实现改native、目录/tmp/kimi-hf-initial-native，退出0，
  全权重精确、源模型前向1e-6及一次optimizer更新均通过。日志同目录名.log。
- TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings
  3529通过/2跳过/230警告，35.71秒、退出0，
  /tmp/llmtuner-hf-initial-suite.log。
- 全适用pre-commit（排除examples/native_alltoall.py）全部通过，
  python -m compileall -q llmtuner与git diff HEAD --check退出0。
  本批未执行提交推送，全局迁移仍有未闭合项。

KIMI-HF-EXPORT-01 / P2 / Kimi EP/FSDP实际HF导出存储（2026-10-15）：
- 扩展tests/integration_tests/kimi_hf_initial_load.py的真实Trainer测试，生产
  实现无变化。initial HF加载后训练两步，保存完整native step2，再同step
  last_save_model_only=True/last_save_in_hf=True/export_dtype=float32导出。
  实际调用DistributedCheckpointManager.save与save_hf_checkpoint，读取
  发布index指向的全部safetensors分片，全部键/权重对照导出前精确相等。
  原始Kimi模型strict加载产物后前向对照1e-6；运行时全部模型/optimizer/
  scheduler/trainer/EMA状态精确不变，导出后继续一次真实更新成功。
- 同step导出确实写入checkpoint/step-2/model-export，完整native checkpoint
  留在step-2。最终custom场景进一步对所有原checkpoint文件SHA256前后精确
  比较；不是仅检查文件存在或只比metadata。
- CPU/Gloo/FP32、FSDP2/EP2/其他轴1、latent32+RMSNorm/shared1/group2/top2，
  AC/compile关闭。公共环境KIMI=1、KIMI_MOE=1、KIMI_LATENT=1、EP_SIZE=2、
  PP_SIZE=1（均LLMTUNER_TEST_前缀）；PYTHONPATH=.、GLOO_SOCKET_IFNAME=lo0、
  TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0。执行torchrun --nproc_per_node=2
  --master_addr=127.0.0.1 --master_port=29839
  tests/integration_tests/kimi_hf_initial_load.py /tmp/kimi-hf-export-02，
  FSDP_IMPLEMENTATION=custom，EMA关闭，退出0。
  前一/tmp/kimi-hf-export-01仅导出/前向/继续更新版本同样退出0。
- FSDP_IMPLEMENTATION=native、EMA=dynamic，目录/tmp/kimi-hf-export-native，
  同命令退出0；精确运行状态包含动态EMA，未增加文件hash时检查metadata不变。
  最终custom+dynamic EMA+全部文件hash版本目录
  /tmp/kimi-hf-export-custom-final，端口29840，最终状态见后续记录。
- 日志均为对应目录名.log；HEAD f188f679e2325ded37ae1da46153c4ae6b511682
  +既有改动。此批增加验证证据，不外推目标设备、其他dtype、PP/TP/CP组合，
  不宣称native跨EP开关optimizer/EMA布局迁移完成。

KIMI-HF-EXPORT-01 最终状态与门禁（2026-10-15）：
- /tmp/kimi-hf-export-custom-final.log：torchrun退出0，custom FSDP2/EP2
  dynamic EMA、全checkpoint文件hash不变、全部运行state精确、HF文件权重
  精确、strict原模型前向与导出后继续更新均通过。
- pre-commit对本批脚本和两份文档全部适用hook通过，git diff HEAD --check
  退出0。仅集成测试与文档变化，复用前批3529通过/2跳过CPU单测及compileall
  证据（KIMI-HF-INIT-01），未将其记为本轮执行。
- 下一可执行项：Kimi native跨EP开关恢复的模型/optimizer/EMA布局迁移；
  hybrid KCP、MC2×TP、Domino故障协调与设备证据等全局缺口仍开放。

KIMI-EP-OPT-01 / P1 / 跨EP布局optimizer与EMA转换契约（2026-10-15）：
- 当前范围：models/kimi/checkpoint.py新增convert_kimi_optimizer_state，
  输入为完整EP收集后的flat FQN状态；逐专家moments/AMSGrad/ema_params
  与grouped专家轴stack/unbind，router/bias/latent键同步映射。
  此函数to_hf标志表示采用HF逐专家FQN，不声称HF格式包含optimizer状态。
- 合并前逐专家step及param_groups超参数必须一致；缺专家、非法索引、
  moment形状/dtype不一致或目标键碰撞均拒绝。不选择某一专家覆盖不同历史。
  复制合并/拆分后的标量step，避免两个独立runtime计数器共享存储。
- 独立真实AdamW+AMSGrad：逐专家更新一步→flat转换→新grouped参数/optimizer
  初始化→加载moments/step/groups→同一新梯度再更新，两种参数逐位一致；
  所有flat字段双向精确往返。EMA tensor单独逐位往返。
- 首轮1失败/3通过：测试为分配state直接调用真实step改变了目标权重；修为
  既有init_optim_state零lr分配。同时检查发现collapsed step需clone，已修。
  之后4通过；再补scalar不共享与缺失专家拒绝测试，结果见门禁。
- 实现部分/CPU转换通过：该转换器尚未接入DCP生产load planner，不能宣称
  Trainer跨EP开关native完整恢复已实现。下一步以源checkpoint metadata
  构造读取模板，在通信前协调布局识别，再同时迁移model/optimizer/EMA，
  独立runtime验收更新前全部状态与后续更新。
  HEAD f188f679e2325ded37ae1da46153c4ae6b511682 + 既有改动；未提交推送。

KIMI-EP-OPT-01 本批门禁（2026-10-15）：
- pytest -q tests/unit_tests/cpu/models/test_kimi_optimizer_migration.py
  --disable-warnings：最终退出0，6通过，0.99秒。
- 全量命令TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q
  --disable-warnings：3533通过/2跳过/228警告，36.41秒、退出0；
  日志/tmp/llmtuner-kimi-optimizer-migration.log，
  该次收集在后两个测试添加前，新增两个由上述定向结果覆盖。
- 全适用pre-commit排除受保护示例后全部通过；compileall、diff检查均0。
  本批函数未接入生产加载，前批生产HF/native恢复证据不受影响，未重跑通信。

KIMI-EP-DCP-01 / P1 / native DCP跨专家布局生产读取路径（2026-10-15）：
- 新components/checkpointer/kimi_layout.py，已阅读/CPU真实DCP文件验证，
  双rank跨EP训练恢复尚未验收。DistributedCheckpointManager非HF加载先
  检查Kimi源metadata：同布局返回既有sharded加载；individual/grouped变化
  时按源planner路径构建完整CPU读取模板，再转换模型/optimizer/EMA。
  optimizer先于scheduler，RNG最后；缺类别、混合布局、分配或转换异常在
  collective之前的local错误边界协调。不把改变目标模板当作读取源状态。
- 适配成本：跨布局迁移每rank读取完整CPU tensor，明确增加主机峰值内存；
  同布局仍用原sharded路径。不能外推为大模型无额外内存迁移或性能优化。
  PyTorch planner metadata路径是依赖，缺失时明确失败。
- 首轮真实磁盘测试两个方向均失败：模型前向通过，但下一次AdamW更新报
  betas=None。DCP flat加载的byte记录需要写回original_state_dict，否则
  外层Stateful装配用占位None覆盖读取结果。专用源planner修复该写回，
  保留失败证据，不删param_groups字段。
- tests/unit_tests/cpu/models/test_kimi_dcp_layout.py：真实dcp.save/load、
  AdamW+AMSGrad、latent/shared Kimi，两方向individual↔grouped，恢复模型
  前向1e-6、后续更新前向2e-6；新增真实EMA保存/转换/加载全部字段逐位一致。
  最终定向退出0，2通过，4警告，2.48秒。非mock存储/转换，单进程CPU。
- HEAD f188f679e2325ded37ae1da46153c4ae6b511682 + 既有未提交改动。
  实现部分：生产入口已接入，但真实多rank EP开关、数据/RNG拓扑变化、
  compiled FQN/PP路径及失败后参数部分应用仍需验证；不宣称整体迁移完成。
  下一动作真实Trainer两阶段跨EP配置验收更新前完整状态及后续更新。

KIMI-EP-DCP-01 本批门禁与未闭合验收（2026-10-15）：
- 首轮全量2失败/3535通过/2跳过，35.72秒，源byte字段回写缺陷；
  /tmp/llmtuner-kimi-dcp-migration.log。修复后全量退出0，
  3537通过/2跳过/235警告，34.65秒；/tmp/llmtuner-kimi-dcp-final.log。
  其后补EMA精确断言、缺类别提前校验，定向最终2通过，2.42秒。
- 真实同布局回归：LLMTUNER_TEST_KIMI_MOE=1 LLMTUNER_TEST_KIMI_LATENT=1
  LLMTUNER_TEST_EP_SIZE=2 LLMTUNER_TEST_EMA=dynamic
  python /tmp/run_kimi_trainer.py，full/resume均0，custom FSDP2/EP2，
  CPU/Gloo/FP32、其他轴1、AC/compile关闭。完整更新前/最终状态和后续loss
  严格一致。原始日志macOS临时根目录llmtuner-kimi-trainer-_s1451w2。
  该结果只证明同布局路径未回归，不替代跨EP开关分布式验收。
- 全适用pre-commit排除受保护示例后全部通过；后续本批文件定向门禁同样0，
  compileall和diff检查0。全局目标保持未完成，未commit/push。

KIMI-EP-CROSS-01 / P1 / 双rank EP1→EP2真实恢复（2026-10-15）：
- 首轮full0/resume1，普通CPU源完整tensor复制到dense FSDP DTensor报混合
  Tensor/DTensor，原日志macOS临时根目录llmtuner-kimi-cross-vy5rq4c7。
  kimi_layout现按目标state模板重建dense mesh/placements，EP expert切片
  仍归原state wrapper；optimizer/EMA同样恢复目标DTensor布局。
- 第二轮full0/resume1推进至RNG拓扑校验，源目录llmtuner-kimi-cross-8t51kand。
  拒绝是既有明确契约，未弱化；RNGCheckpointState新增无副作用
  validate_state_dict，跨布局在模型/optimizer应用之前验证拓扑。
  保存恢复仍RNG最后应用，新回归证明validate不改变CPU随机流。
- 新tests/integration_tests/kimi_ep_layout_resume.py使用公开
  checkpoint.exclude_from_loading=[rng_state]，明确放弃精确随机流连续性。
  对源split独立快照转换后比较完整model/optimizer/EMA/trainer/scheduler，
  全部精确相等，再执行后续两步loss对照1e-6。不是修改原严格恢复脚本断言。
  新脚本首次完整状态/训练通过但读错losses文件名失败，修为losses_full.json。
- 成功命令：LLMTUNER_TEST_KIMI=1、KIMI_MOE=1、KIMI_LATENT=1、EMA=dynamic、
  EP_SIZE=2、PP_SIZE=1、FSDP_IMPLEMENTATION=custom（均同前缀），
  PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0
  torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29839
  tests/integration_tests/kimi_ep_layout_resume.py
  /var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-kimi-cross-8t51kand。
  退出0，两rank exact pre-update state/later loss passed；
  /tmp/kimi-cross-resume-correct.log。源full使用相同脚本配置EP_SIZE=1、
  pp_checkpoint_equivalence.py full，新独立torchrun恢复为EP2。
- CPU/Gloo/FP32、custom FSDP2、其余轴1、latent32/shared1/group2/top2、
  AC/compile关闭、动态EMA。反向EP2→EP1、多轴、设备、故障原子性未验收。
  实现部分；精确随机流跨拓扑仍拒绝，不能由本数值模型无随机前向外推。
- 定向RNG+磁盘布局测试48通过/8警告，2.83秒；HEAD f188f679e2325ded37ae1da46153c4ae6b511682
  +既有改动；全量结果见后续门禁。

KIMI-EP-CROSS-01 门禁补充（2026-10-15）：
- 全量TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings
  取得退出0，原日志/tmp/llmtuner-kimi-cross-suite.log；该次收集后增加的
  无副作用RNG验证由48项定向测试覆盖。
- 本批全部受影响文件pre-commit通过，compileall/diff退出0。
  前批全适用文件门禁沿用，无关文件未重复自动修改。未提交推送。

KIMI-EP-CROSS-02 / P2 / EP2→EP1反向真实恢复（2026-10-15）：
- 复用KIMI-EP-DCP-01已成功的源full训练checkpoint与独立split-state快照，
  macOS临时根目录llmtuner-kimi-trainer-_s1451w2；源custom FSDP2/EP2，
  当前源码状态HEAD f188f679e2325ded37ae1da46153c4ae6b511682+既有改动。
  本轮独立torchrun恢复目标custom FSDP2/EP1，其他轴1，CPU/Gloo/FP32，
  latent32/norm/shared1/group2/top2、dynamic EMA、AC/compile关闭。
- 初次恢复已完成，但独立参考在转换源model快照时失败：快照保存本地EP
  专家，而optimizer/EMA快照已gather。修复tests/integration_tests/
  kimi_ep_layout_resume.py的参考，从两个rank独立快照拼回完整专家轴，
  然后进行布局转换。未改生产加载或断言；显式限制该参考world_size=2，
  不把按world rank拼接外推为多轴expert rank顺序。
- 成功命令环境LLMTUNER_TEST_KIMI=1、KIMI_MOE=1、KIMI_LATENT=1、
  EMA=dynamic、EP_SIZE=1、PP_SIZE=1、FSDP_IMPLEMENTATION=custom（同前缀），
  PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0
  torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29839
  tests/integration_tests/kimi_ep_layout_resume.py
  /var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-kimi-trainer-_s1451w2。
  退出0，两rank exact pre-update state/later loss passed；更新前model、
  optimizer、EMA、trainer、scheduler精确一致；后续两步loss1e-6通过。
  /tmp/kimi-reverse-resume-fixed.log；首轮失败/tmp/kimi-reverse-resume.log。
- 沿用公开exclude_from_loading=[rng_state]，不保证跨拓扑精确随机流。
  出现源DTensor快照反序列化DeviceMesh警告，compile关闭，此证据不证明
  编译恢复。生产实现无变化；本批只脚本/文档，复用前批CPU/compileall证据，
  定向Ruff/diff通过。多轴、其他EP度数和目标设备仍开放。

KIMI-EP-PREFLIGHT-01 / P1 / 跨布局模型键/形状通信前校验（2026-10-15）：
- 发现：kimi_layout将完整CPU源张量distribute_tensor后才由ModelCheckpointState
  检查缺键/形状，损坏状态可能在局部rank报错时让其他rank进入通信。
  ModelCheckpointState原校验提为无通信validate_state_dict，load复用同一规则；
  kimi_layout在synchronize_local_errors转换边界内提前调用，先于所有
  distribute_tensor与模型copy。保持原scalar兼容和PP其他部分键语义。
- 新真实DCP回归：缺shared down权重、shared down形状错误，两类转换失败
  后目标全部权重逐位不变。正常双向磁盘迁移/AdamW/EMA同时通过，
  pytest -q tests/unit_tests/cpu/models/test_kimi_dcp_layout.py --disable-warnings，
  退出0，4通过/8警告，3.40秒。不宣称所有恢复错误都事务回滚。
- HEAD f188f679e2325ded37ae1da46153c4ae6b511682+既有修改；模型键/形状
  静态校验无新增collective，复用当前state wrapper缓存full形状。
  生产通信故障、optimizer后置验证失败与部分应用仍开放，下一动作扩大
  optimizer完整状态预校验并验证rank局部损坏失败协调。

KIMI-EP-PREFLIGHT-01 门禁（2026-10-15）：
- TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings
  3540通过/2跳过/253警告，35.98秒，最终退出0；
  /tmp/llmtuner-migration-validation.log。
- 双rank反向EP2→EP1重复前批成功命令，源_s1451w2，custom FSDP2、动态EMA，
  显式排除rng_state，退出0；/tmp/kimi-reverse-preflight.log，两rank
  更新前精确状态及后续loss通过，证明正常生产迁移未回归。
- 全适用pre-commit排除受保护示例后全部通过，compileall/diff退出0。
  全局目标继续，未commit/push。

### KIMI-EP-TRAINING-PREFLIGHT-01：跨布局训练状态提前校验

- 优先级：状态损坏；路径：`components/checkpointer/kimi_layout.py`，由 DCP 恢复入口调用。
- 确认问题：optimizer/EMA 缺键或形状错误原本在模型应用后才由各自加载器拒绝，导致失败恢复已改变模型。
- 修改：转换前取得目标完整训练状态模板（可能包含 EP/TP gather，留在本地错误同步边界外）；转换后、模型写入前同步检查必需键、张量类型及完整逻辑形状。后续 placement 复用模板；AMSGrad 最大动量是否必需按保存的配置判断。
- 验收证据：真实 DCP 损坏 optimizer/EMA × 缺键/形状四例，均报错且目标所有模型张量逐元素不变；正常两个方向恢复及下一次更新继续通过。`pytest -q tests/unit_tests/cpu/models/test_kimi_dcp_layout.py --disable-warnings`：8 passed、16 warnings、exit 0。
- 本轮全量：`TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings`：3544 passed、2 skipped、264 warnings、35.41s、exit 0。代码为 HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加现有未提交迁移及本批修改。
- 门禁：适用 Git 跟踪及未跟踪文件运行 `pre-commit run --files ...`（排除 `examples/native_alltoall.py`），全部通过；`python -m compileall -q llmtuner`、`git diff HEAD --check` exit 0。日志 `/tmp/llmtuner-kimi-state-preflight.log`、`/tmp/llmtuner-state-precommit.log`。
- 边界：本批证明缺键/张量形状错误提前拒绝，不证明所有 optimizer 超参数错误或通信内部异常具有事务回滚能力；新增模板 gather 的真实跨 EP 分布式路径仍需复跑。下一步先复跑 EP2→EP1，再继续 KCP hybrid 等必需迁移缺口。

### KIMI-EP-TRAINING-PREFLIGHT-02 / KCP-MLA-DETECTION-01

- 本轮范围：跨 EP 恢复生产路径、AMSGrad 保存配置切换，以及 KCP 对本地 Kimi MLA 的能力检查。HEAD 仍为 `f188f679e2325ded37ae1da46153c4ae6b511682`，含现有未提交迁移与本批修改。
- EP2→EP1：真实 Trainer 两进程 CPU/Gloo、FP32、custom FSDP2、TP/PP/CP1、动态 EMA、latent MoE，AC/compile 关闭。命令采用 `LLMTUNER_TEST_KIMI=1 LLMTUNER_TEST_KIMI_MOE=1 LLMTUNER_TEST_KIMI_LATENT=1 LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_EP_SIZE=1 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29839 tests/integration_tests/kimi_ep_layout_resume.py /var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-kimi-trainer-_s1451w2`；exit 0，两个 rank 均验证更新前模型/optimizer/EMA/trainer/scheduler 精确一致及后续两步 loss 1e-6。显式排除 RNG，未证明随机流连续性或更新后全部参数精确相同。日志 `/tmp/kimi-training-preflight-reverse.log`。
- AMSGrad：保存 true、目标初始 false 时，最大二阶动量也必须按保存配置提前检查；原模板没有该键时以完整 exp_avg_sq 形状补充期望。新增缺键/形状两例均确认模型未写入；跨布局 DCP 测试 10 passed、20 warnings。
- KCP-MLA-DETECTION-01/P1：原 hybrid 检测只识别 q/k/v 三 projection，本地 MLA 的压缩 KV 布局漏过，可能只 patch KDA 而留下局部 softmax。现纳入 `kv_a_proj_with_mqa`/`kv_b_proj` 结构，q_lora_rank=None/4 两例均在依赖加载前拒绝。此项只关闭静默放行缺陷，不关闭 KCP-HYBRID-01 必需迁移；已检查本地 latent_attention 调用与上游 `distributed/context_parallel/kcp/kcp_context_parallel.py`、`ulysses/ulysses_attention.py` 的 full/compressor/transposed 三 patch 契约，下一步需接实例级 attention callable 和全局 mask，而非复制全局 monkey patch。
- 定向命令 `pytest -q tests/unit_tests/cpu/parallel/test_cp.py tests/unit_tests/cpu/models/test_kimi_dcp_layout.py --disable-warnings`：42 passed、20 warnings、exit 0。
- 门禁：适用文件 pre-commit（排除受保护 example）exit 0；compileall 与 diff check exit 0。全量 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings`：3548 passed、2 skipped、245 warnings、36.46s、exit 0，日志 `/tmp/llmtuner-kcp-mla-preflight.log`。能力矩阵纠正 Kimi Linear 注册已接通、K3/compressor 仍缺失的旧描述。

### KCP-HYBRID-MLA-01：本地 Kimi MLA 的真实 Ulysses 分支

- 优先级 P1，关联 KCP-HYBRID-01（仍开放）。审查路径：`models/kimi/latent_attention.py`、`models/hf/model.py` 的 attention seam、`parallel/context_parallel/kcp.py` 与现有 CP autograd 交换；上游固定来源及适配差异见 upstream map 新条目。
- 实现：`apply_kcp` 对本地 MLA 安装实例 callable，KDA 保持原 state CP；MLA 以已有 SeqToHead/HeadToSeq 交换完整展开的 q/k/v，使用 global causal SDPA 后恢复 BSHD。q/head 与 value 维度不同也保留；不复制配置中的 GQA KV。重装替换 callable、不新增参数/状态键；KV cache 明确拒绝。
- mask：验证本地 square causal/padding mask 的全部元素，恢复 padding 后 gather 得到全局 key validity；不静默丢弃 additive bias、packed 或任意 document mask。局部校验在 CP 子组同步，collective 留在本地边界外。HF wrapper 的 KCP 分支不构造/传递 flex BlockMask，避免目标设备选 flex 时错误进入不同布局。
- 新脚本：`tests/integration_tests/kimi_mla_cp_equivalence.py`；`PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29841 tests/integration_tests/kimi_mla_cp_equivalence.py`。CPU/Gloo、FP32、CP2、4 query heads、单配置 KV head 但投影已展开、Q/K dim4 与 V dim3；q_lora None/4 × output gate false/true × padding false/true 共8组。独立无 CP 的完整序列 eager MLA 参考（其投影实现已有独立方程单测），输出 1e-6、输入梯度2e-6、归约参数梯度3e-6；两 rank 均通过。rank0 非因果 mask 还验证两 rank 同步拒绝，随后 barrier 成功。日志 `/tmp/kimi-mla-cp-final.log`。
- 单测新增成功安装/重装时参数身份与名称不变、非法 head degree 安装前拒绝、HF sdpa/flex 两 backend 的 KCP mask seam；全量 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings`：3551 passed、2 skipped、264 warnings、37.26s、exit0，日志 `/tmp/llmtuner-hybrid-mla-unit.log`。HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加现有工作区和本批改动；之后仅增加非空/完整4D mask shape 校验，真实分布式脚本复跑覆盖最终版本。
- 质量门禁：适用全文件 pre-commit（排除受保护 example）、compileall、diff check 均 exit0；原始日志 `/tmp/llmtuner-hybrid-mla-precommit.log`。未运行目标设备，无性能声明。
- 仍需验收：本脚本只运行真实 MLA 分支，没有用 mock 冒充完整 KDA 内核；安装单测的 KDA/runtime 是明确测试替身。完整 hybrid Trainer、KDA Ascend 运行、K3/compressor/transposed、AC/compile 与多轴组合仍开放，下一步由公开 Kimi hybrid 配置验证整个 KDA/MLA 混合执行及 mask/padding 边界。

### KCP-PADDING-STATE-01：防止清零 padding 改变 KDA 状态语义

- P1 / `parallel/context_parallel/kcp.py::kcp_kda_forward`，调用链为本地 Kimi decoder → KCP KDA wrapper。已阅读 `models/kimi/delta_attention.py`、decoder mask 分支及上游固定 commit `0af4eca242c14ebff254dab1a38045f1395fa520` 的 `distributed/context_parallel/kcp/kda_state.py`。
- 确认缺陷：本地 KDA padding 先 compact，旧 KCP wrapper 只清零输入/输出；无效 token 仍推进卷积/decay，内部 padding 后的有效结果改变。回归固定 seed41 直接比较真实本地 KDA 的 compact 与清零路径，确认差异，不放宽容差。
- 修改：KDA 局部输入与能力检查移入 `synchronize_local_errors(..., group=group)`，边界内没有 collective；遇到非全有效 mask，在任何可选 kernel/context 构造前报可操作错误。删除已不适用的零乘 mask。全有效 mask 仍通过。正确 padding 状态通信仍为部分实现缺口，拒绝不能替代迁移验收。
- 单测 `pytest -q tests/unit_tests/cpu/parallel/test_kcp_runtime_contract.py --disable-warnings`：5 passed。首次 fixture 缺少公开配置必需的 kda_layers/full_attn_layers 报错，补全合法配置后通过；未改变数值断言。
- 实际分布式：`PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29841 tests/integration_tests/kimi_mla_cp_equivalence.py`，CPU/Gloo、FP32、CP2；既有8组真实 MLA 输出/梯度保持通过，新增 rank0 KDA mask 带 padding、rank1 全有效时双方均在 runtime/kernel 前失败，后续 barrier 成功，exit0。日志 `/tmp/kimi-kcp-padding.log`；该失败用例用真实 KimiDeltaAttention，不 mock runtime，不宣称运行 KDA 内核。
- 全量 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings`：3554 passed、2 skipped、249 warnings、36.59s、exit0；日志 `/tmp/llmtuner-kcp-padding-unit.log`。适用全文件 pre-commit（排除保护 example）、compileall、diff check exit0，日志 `/tmp/llmtuner-kcp-padding-precommit.log`。HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加现有工作区及本批改动。
- 当前依赖证据：PyTorch2.14，fla 与 triton_ascend_kernels 的 find_spec 均 None，CUDA false、torch 无 npu；完整 target KDA/hybrid 数值仍未运行。不能以 MLA CPU 通过替代。
- 上游来源纠正：Kimi K3 example 的 KCP 实际 patch 普通 eager MLA+KDA；compressor/transposed 类型由 DeepSeek V4 example 使用。K3 多模态顶层 CP 和 compressor 两段 KV 通信应独立推进；同步更新 upstream map。下一步处理正确 padding 状态布局或独立 compressor 迁移，不继续扩张已经覆盖的 checkpoint 局部测试。

### COMPRESSOR-CP-BOUNDARY-01：压缩前窗口与全局条目归属

- P1 / 必需迁移 KCP-HYBRID-01 的独立 compressor 分支；本轮为实证审查，未声称实现完成。阅读本项目 `context_parallel/{context_parallel,attention}.py`、HF attention adapter/model setup，并核对 FSDPTurbo 固定 `0af4eca242c14ebff254dab1a38045f1395fa520` 的 `ulysses_attention.py` compressor 两 wrapper 与 DeepSeek V4 example。
- 当前环境 Transformers5.9.0 已注册 DeepseekV4Config，源码路径为安装包 `transformers/models/deepseek_v4/modeling_deepseek_v4.py`；本项目未发现专用 compressor 接线。普通 CP 安装 CPFlexKernel，不会按 sliding/compressed 两段重排，也不负责压缩前窗口状态。Ulysses 的 query/K/V 独立 SeqToHead 要求每个输入头数可均分，V4 单头共享 KV 不满足；不能把 query heads 能均分视为 KV 可均分。
- 已执行真实 HCA 最小实验：hidden16、heads4、head_dim8、rope4、q_lora8、o_groups2/o_lora4、HCA压缩率4、sequence6、FP32、seed47，position_bias清零。完整调用 HCACompressor 输出 `(1,1,1,8)`；分别对 positions0:3 与3:6 的两个局部 shard 调用均输出 `(1,1,0,8)`。断言完整条目数1且局部合计0通过，exit0。无需通信即可证明“先局部压缩后交换 KV”丢失跨 shard 窗口。
- 可复现：从 Transformers leaf 导入 `DeepseekV4Config`、`DeepseekV4HCACompressor`，配置 `num_hidden_layers=1, layer_types=['heavily_compressed_attention'], mlp_layer_types=['moe'], compress_rates={'compressed_sparse_attention':4,'heavily_compressed_attention':4}` 与上述维度；`x=torch.randn(1,6,16); positions=torch.arange(6)[None]`；调用 `compressor(x,torch.empty(0),positions,None,0)` 与两个3-token切片，比较 `result[0].shape[2]`。实验依赖 Transformers5.9.0 实现，未修改第三方安装包。
- 同一源码还确认 HCA 无 cache 时 `first_window_position=0`，压缩 RoPE 的位置从局部0开始，block_bias却使用传入的全局 position_ids；CSA另含前窗 overlap。窗口刚好整除也不能据此跳过位置/overlap检查。eager attention保留 per-head sink，并返回BSHD；通用CPU flex明确不支持s_aux。不得通过忽略sink或容差扩大来制造通过。
- 实现状态：未迁移/已确认边界；验证状态：CPU最小实验与静态调用证据，不是CP数值通过。验收需：跨shard压缩窗口/CSA overlap交换、全局压缩条目索引及RoPE、两段KV重排、共享KV/GQA扩展与sink head归属、sliding+compressed mask、前后向/更新和生产入口。下一动作先实现压缩前窗口所有权，避免只复制attention wrapper；随后不整除与整除窗口的独立完整模型参考。
- 代码状态 HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加既有未提交改动，本轮未改生产代码。仅文档按范围验证，不重复训练测试。KCP纯KDA设备依赖、MC2×TP和Domino失败协议等其他缺口仍开放。

### COMPRESSOR-CP-WINDOW-01：可微分的压缩窗口重分配

- P1，关联 COMPRESSOR-CP-BOUNDARY-01；实现新增 `parallel/context_parallel/compression_windows.py`，支持等长连续 CP shards，以窗口最后token的rank为owner。变长 all-to-all 只搬运完整窗口所需特征，反向用交换的split sizes恢复源token梯度；全局不足窗口尾部丢弃，零窗口owner/零全局窗口仍保留autograd边。
- `exchange_compression_windows` 先在CP子组协调局部形状/窗口检查，再all-gather固定dtype元数据核对batch/sequence/width/window/dtype，避免不一致split进入数据通信。返回B×owned_windows×window_size×D及全局首window ID。
- `compress_hca_windows` 使用真实HCA projection、position_bias、FP32 softmax、kv_norm和模型partial RoPE，位置按全局window ID计算。来源 Transformers5.9.0 `DeepseekV4HCACompressor.forward`，Apache-2.0头和来源已保留。投影留在源rank，不将完整序列复制到每个rank。
- 验证脚本 `tests/integration_tests/compression_window_equivalence.py`：CP2、CPU/Gloo、FP32、batch2，local length/window `(3,4),(4,4),(5,3),(2,7),(1,2),(5,1)`；输出窗口与完整序列reshape参考逐元素相等，非线性加权loss的输入梯度逐元素相等。真实HCA local length3/4/5、compress_rate4，对独立未修改的完整序列HF HCA输出1e-6、输入梯度2e-6、归约参数梯度3e-6通过；包含rank0没有窗口、rank1拥有跨界窗口，以及非零rank的全局RoPE位置。
- 命令：`PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29842 tests/integration_tests/compression_window_equivalence.py`；exit0。首次末版dtype不一致用例失败：元数据漏填dtype索引，rank0未拒绝（原日志保留 `/tmp/compression-window-dtype-failure.log`）；补入dtype索引后复跑，长度/dtype不一致时双方在数据交换前拒绝且barrier成功，日志 `/tmp/compression-window-final.log`。HF rope配置发出attention_factor提示，未屏蔽；双方使用同一合法配置。
- 实现状态是支持模块已实现、模型生产接线未完成：尚未安装到Trainer/DeepSeek V4 attention，不能称完整compressor CP已迁移。返回窗口数可不均匀，后续KV交换必须支持此布局；CSA overlap、global mask/sink、cache/compile及目标设备仍开放。下一步将HCA窗口输出接入attention，并按global ID构造压缩mask，不能沿用等长compressed all-to-all假设。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加现有工作区和本批文件。适用全文件pre-commit（排除受保护example）、compileall、diff check exit0；日志 `/tmp/compression-window-precommit.log`。全量 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings`：3554 passed、2 skipped、261 warnings、37.07s、exit0，日志 `/tmp/compression-window-unit.log`。

### COMPRESSOR-CP-ATTENTION-01：HCA两段KV与sink的真实通信

- P1，新增支持模块 `parallel/context_parallel/compressor_attention.py`，从窗口owner布局接收shared KV；query/sliding KV走既有Ulysses autograd，压缩条目按每rank实际window数变长交换，保持sliding全序列在前、compressed全序列在后。sink按query head owner取片，仅参与softmax分母；输出恢复本地BSHD。
- global mask按token位置与window ID分别构造：sliding满足因果和window限制，compressed仅允许已完成窗口。局部检查在CP子组协调，随后固定元数据all-gather检查shape/window设置；数据collective不置于本地校验边界。未提供padding/CSA/cache语义。
- 验证 `tests/integration_tests/hca_attention_equivalence.py`：`PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29843 tests/integration_tests/hca_attention_equivalence.py`，CPU/Gloo FP32、CP2、batch2/heads4/dim3；local length/rate/window `(3,4,2),(4,4,4),(5,3,20),(2,7,1)`。独立完整序列显式索引mask+softmax方程：输出1e-6、q/sliding/compressed/sink梯度2e-6均通过；包括零条目owner、不均匀count和全局零压缩条目。日志 `/tmp/hca-attention-final.log`，exit0。
- 范围：本模块仍未由apply_cp/Trainer安装；窗口模块与attention模块已分别有真实通信证据，尚不能宣称完整模型迁移。下一步组合真实HCA投影、压缩、输出逆RoPE与分组projection，验证模型级mask和入口；CSA overlap仍开放。当前eager显式attention使用二次内存，未验收目标内核/性能/compile。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加现有工作区及本批新文件。适用全文件pre-commit（排除保护example）、compileall、diff check exit0；日志 `/tmp/hca-attention-precommit.log`。全量 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings`：3554 passed、2 skipped、257 warnings、37.13s、exit0，日志 `/tmp/hca-attention-unit.log`；之后新增元数据一致性检查，分布式脚本对最终实现复跑通过。

### COMPRESSOR-CP-HCA-MODULE-01：真实HCA完整attention模块组合

- P1，`compressor_attention.py::hca_attention_forward` 组合已有窗口/attention原语，跟随 Transformers5.9.0 `DeepseekV4Attention.forward` 的projection、query norm、partial RoPE、输出逆RoPE和grouped output projection；Apache-2.0来源头保留。检查HCA层类型、全局连续position_ids，拒绝cache和外部mask，检查在CP子组本地错误边界内。
- 扩展 `tests/integration_tests/hca_attention_equivalence.py` 的真实模块用例：CPU/Gloo FP32、world/CP2、batch2、hidden16、query heads4、shared KV1、head dim8/rope4、q_lora8、o_groups2/o_lora4、sequence6/local3、compress_rate4、sliding_window3；rank0无压缩窗口、rank1处理跨界窗口。非零sink和position_bias，独立完整HF模块（未patch）接full sliding mask并自行拼接compressed block_bias；与本地CP forward输出2e-6、输入梯度3e-6、归约后全部参数梯度4e-6通过。之前四组attention方程参考继续通过。
- 命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29843 tests/integration_tests/hca_attention_equivalence.py`，exit0，两个rank均打印full HCA module passed。日志 `/tmp/hca-module.log`。HF rope提示保留，无容差调整/选择性seed。
- 仍未安装到apply_cp或Trainer，未声称完整模型/所有层已支持。下一步需公开入口选择专用路径，处理decoder原有mask、CSA/sliding层、padding与多层模型；没有以HCA单层代替完整compressor迁移。目标设备/compile/cache未验收。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加原工作区与本批改动；适用全文件pre-commit（排除保护example）、compileall、diff check exit0，日志 `/tmp/hca-module-precommit.log`。全量 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings`：3554 passed、2 skipped、263 warnings、36.74s、exit0，日志 `/tmp/hca-module-unit.log`。

### COMPRESSOR-CP-DTYPE-01：HCA数据交换前dtype一致性

- P1 / `compressor_attention.py::hca_ulysses_attention`；审查发现固定元数据只比较shape/window，未比较query dtype，本地KV也未检查dtype/device。不同rank数据字节数可能在all-to-all内部失败。
- 修改：本地错误同步边界校验floating query、KV同dtype/device和sink设备；固定long元数据加入query dtype索引，各rank在数据通信前一致拒绝peer dtype不匹配。sink允许独立浮点精度以保留混合精度参数语义。
- 实际命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29843 tests/integration_tests/hca_attention_equivalence.py`：exit0，原四组attention与真实HCA模块前后向全部通过；新增rank0局部KV dtype错误、两个rank query/KV各自合法但peer dtype不同两例，双方提前拒绝、barrier成功。CPU/Gloo FP32正常路径，错误路径包含FP64；日志 `/tmp/hca-dtype-validation.log`。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加现有工作区；ruff、compileall、diff check通过；本批只新增失败校验，复用上一批全量3554 passed/2 skipped记录，不将其写成本轮执行。生产安装、完整模型、CSA与目标设备依然开放，下一步回到模型级接线。

### COMPRESSOR-CP-DECODER-01：消除训练decoder隐式cache接线冲突

- P1，核对 Transformers5.9.0 `DeepseekV4Model.forward`：即使use_cache=False，past_key_values=None仍创建DynamicCache并传各层，只将返回cache置None。真实tiny模型hook证实attention接收DynamicCache、返回None，输出(1,6,16)且backward成功。此行为不是自动等价于无cache的CP分片状态。
- 新增 `compressor_attention.py::hca_decoder_forward`，遵循HF embedding→hyperconnection展开→decoder layers→hc_head→norm，使用全局position_ids构造main/compress RoPE，不创建cache；attention全局mask由已验证HCA forward负责。源码来源/Apache-2.0保留。工厂尚无安装入口，非法cache/mask明确拒绝。
- 扩展真实两进程脚本：小型DeepseekV4Model、vocab32/hidden16/moe_width16/E4/top2/shared1/hc_mult2、单HCA层、CP2、FP32、sequence6/local3；独立未修改HF decoder跑完整输入并use_cache=False，对手动安装HCA forward+stateless decoder的输出3e-6、归约后全部参数梯度6e-6比较通过，含embedding/MoE/hyperconnection/norm。原HCA模块及dtype失败协调继续通过。
- 命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29843 tests/integration_tests/hca_attention_equivalence.py`，exit0，两rank均stateless HCA decoder passed；日志 `/tmp/hca-decoder.log`。没有放宽既有断言，decoder新增自身容差。
- 仍开放：apply_cp公开安装、HF wrapper模型级mask/forward、Trainer数值与恢复、CSA/sliding混合层、padding/compile/目标设备。此记录证明真实单层decoder，不是全生产支持；下一步实现公开安装并拒绝尚未实现的层组合，同时保留完整迁移缺口。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加现有工作区；本轮执行适用pre-commit、compileall、diff check；全量3554 passed/2 skipped沿用前批，不宣称本轮重跑。

### COMPRESSOR-CP-INSTALL-01：公开apply_cp安装全HCA decoder

- P1 / `context_parallel.py::apply_cp` → `compressor_attention.py::apply_hca_context_parallel`；DeepSeek V4模型在通用flex校验前选择专用入口，使用真实CP mesh。完整安装前检查全部本地层：仅HCA、Ulysses、连续分片、无load balancer/packing/dropout、TP/PP1、query heads可分。检查完成后再替换attention与stateless decoder forward，保留参数身份和checkpoint名称。
- HF wrapper记录hca_ulysses策略，绕过通用flex/DSA mask；实际forward在CP子组本地错误边界协调拒绝padding或外部mask，再进入decoder。该路径不宣称支持padding/packed/CSA/sliding混合层/TP/PP，相关完整迁移仍开放；没有把拒绝配置作为总体完成。
- 真实两进程脚本已改走 `apply_cp`，使用真实 `ParallelConfig(context_parallel_ulysses_size=2, cp_config=ContextParallelConfig(context_parallel_strategy='ulysses'))` 和 `init_device_mesh('cpu',(2,),mesh_dim_names=('cp',))`；轻量wrapper承载真实HF decoder并记录set_cp_mesh，断言所有parameter identity不变。原完整HF参考的decoder输出3e-6/归约全部梯度6e-6保持通过。该测试不是实际HFTransformerModel/Trainer，不外推后者通过。
- 命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29843 tests/integration_tests/hca_attention_equivalence.py` exit0，日志 `/tmp/hca-install.log`。首次脚本编辑有空import导致SyntaxError，修正测试语法后重跑，未改生产断言。现存数值/失败协调用例全部通过。
- 下一步使用完整HFTransformerModel/Trainer公开配置验证forward、更新及恢复；检查模型级mask和剩余层类型。生产入口已触达支持机制，验证仍停留真实CP decoder层级。设备数值/compile/多轴未验收。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加现有工作区和本批；适用全文件pre-commit排除保护example、compileall、diff check exit0；日志 `/tmp/hca-install-precommit.log`。全量 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings`：3554 passed、2 skipped、260 warnings、36.60s、exit0，日志 `/tmp/hca-install-unit.log`。

### COMPRESSOR-CP-WRAPPER-01：完整HF wrapper前后向与更新

- P1 / `models/hf/model.py::__init__` 与公开HCA安装调用链。真实wrapper首次构造失败，原因是通用CPU backend选择sdpa，Transformers5.9.0 DeepseekV4ForCausalLM明确不支持；错误发生于CP安装前。现在DeepSeek V4明确选择其支持的eager，其他模型保持既有设备选择。原失败日志 `/tmp/hca-wrapper-backend-failure.log`保留。
- 扩展 `hca_attention_equivalence.py`：真实HFTransformerModel、真实ParallelConfig/mesh/apply_cp，完整无CP wrapper为独立运行时参考。单HCA层、CPU/Gloo、FP32、CP2、vocab32/hidden16/E4/top2/shared1、sequence6/local3，比较logits3e-6、归约后全部参数梯度6e-6、一次SGD(lr0.01)后全部参数2e-6，两个rank全部通过。原模块/decoder/失败协调仍通过；不是测试替身wrapper。
- 命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29843 tests/integration_tests/hca_attention_equivalence.py`，exit0，日志 `/tmp/hca-wrapper.log`。参考数值未改容差，无选择性重跑seed。
- 边界：本轮接通实际wrapper，不证明Trainer/FSDP/AdamW/EMA/保存恢复；CSA、混合层、多轴、目标设备/compile仍未完成。eager backend无性能收益声明。下一步公开Trainer配置与恢复证据。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加现有工作区/本批；适用全文件pre-commit排除保护example、compileall、diff check exit0，日志 `/tmp/hca-wrapper-precommit.log`。全量 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings`：3554 passed, 2 skipped, 241 warnings in 36.32s，exit0，日志 `/tmp/hca-wrapper-unit.log`。

### COMPRESSOR-CP-TRAINER-01：真实HCA Trainer完整恢复

- 新脚本 `tests/integration_tests/hca_trainer_checkpoint.py`，复用pp checkpoint harness的full/resume严格快照验证，仅替换公开配置工厂，不patch生产模型/optimizer/通信。全HCA DeepSeek V4：hidden16/heads4/KV1/head_dim8/rope0.5/E4/top2/shared1/hc_mult2、sequence32/local16、batch8、CP2/DP1/TP/PP/EP1、CPU/Gloo/FP32、AC/compile关、AdamW及动态EMA。DP1没有FSDP sharding，不能写成CP×FSDP通过。
- 命令两次独立torchrun：`LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_EMA=dynamic PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29844 tests/integration_tests/hca_trainer_checkpoint.py full|resume /var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-hca-trainer-0nnb6yw2`，full和resume均exit0。
- full训练4步、保存step2及独立split/final快照；新进程恢复后两rank均pre-update state failures=0，比较模型/optimizer/scheduler/EMA/trainer/token counter；随后2步loss与连续训练完全一致，`max abs diff=0.000e+00`。最终参数、optimizer、EMA、scheduler/trainer等harness检查均all checks passed。loss序列4.860137462615967、4.857418060302734、4.860584259033203、4.85555362701416。数据游标通过后续batch/训练对比间接覆盖，未单独比较RNG payload；未排除RNG加载。
- 日志 `/tmp/hca-trainer-full.log`、`/tmp/hca-trainer-resume.log`；HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加原工作区，新批只增加集成配置脚本、无生产变化。复用上一批全量3554 passed/2 skipped，新增脚本pre-commit、compileall、diff check exit0。
- 实现/验证进展：全HCA子集已由公开配置到实际Trainer训练/恢复，不再是仅支持模块或手动wrapper安装。完整compressor迁移仍部分：CSA/sliding混合、padding、CP×FSDP及其他轴、AC/compile、目标设备性能未验收。下一步推进CSA overlap或代表性CP×FSDP组合，不以本次单配置关闭全局目标。

### COMPRESSOR-CP-FSDP-01：HCA CP2×custom FSDP2完整恢复

- 扩展 `hca_trainer_checkpoint.py`：公开ParallelConfig读取LLMTUNER_TEST_HCA_DP_SIZE及FSDP实现，默认CP2/DP1保持；本次显式DP-shard2/world4，不patch生产路径。
- 配置：PyTorch2.14、CPU/Gloo/FP32、CP Ulysses2、dp_shard2、TP/PP/EP1、全HCA单层、hidden16/E4/top2/shared1、batch8/seq32、AdamW、动态EMA、AC/compile关、custom fine-grained FSDP。实际日志mesh `pp=1, dp_replicate=1, dp_shard=2, cp=2, tp=1, ep=1`；harness的custom state类型检查仍启用。
- 两个独立命令：`LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_HCA_DP_SIZE=2 LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29845 tests/integration_tests/hca_trainer_checkpoint.py full|resume /var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-hca-fsdp-xn06wkxx`；full/resume均exit0。
- 连续4步、step2保存，新进程四rank均pre-update state failures=0；模型/AdamW/scheduler/EMA/trainer/token counter快照比较通过；恢复后两步loss差0，最终训练状态harness检查全部通过。full losses `[4.860430717468262,4.8570661544799805,4.860596656799316,4.855550289154053]`。参考是同配置连续训练，不能据此证明对CP1/DP1数值等价；独立完整HF参考由之前CP2 wrapper测试承担。
- 日志 `/tmp/hca-fsdp-full.log`、`/tmp/hca-fsdp-resume.log`。代码HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加现有工作区；本批无生产变更，仅集成配置扩展。pre-commit、compileall、diff check通过，复用上批CPU3554 passed/2 skipped而非本轮重跑。
- 能力仅确认这一个CP×custom FSDP配置；native FSDP、AC/compile、CSA/混合层、padding、多设备仍开放。下一步推进CSA overlap实质迁移或另一已声明能力缺口，不以成功恢复替代所有组合验收。

### COMPRESSOR-CP-CSA-OVERLAP-01：CSA前窗Ca的可微halo交换

- P1，新增 `compression_windows.py::compress_csa_windows`。阅读Transformers5.9.0 CSA compressor和Indexer：各窗口Cb与前窗口Ca做联合softmax，不能按局部窗口重置。复用完整窗口重分配，跨owner仅传末窗Ca/gate，按全局final-token owner定位，可跳过零窗口rank；反向恢复halo梯度，输出采用全局window RoPE。保留Apache-2.0来源。
- 实际CPU/Gloo CP2：local length3/4/5/1、ratio4、batch2、hidden16/head_dim8、seed71，真实CSA projection/position_bias/norm/RoPE；HF完整CSA前向作为压缩参考，仅将离散indexer.forward替换为empty selection以隔离本批，不宣称indexer验证。输出1e-6、输入梯度3e-6、归约压缩参数梯度4e-6通过；indexer参数双方无梯度。全局零窗口的HF输出为无grad常量，本地显式验证输入和参数零梯度。
- 首次平方和loss在length4参数梯度失败：maxdiff5.073496e-6、reference max3.8023337e-4、均finite，输出/输入梯度通过。RMSNorm输出平方和接近常量导致抵消，换为同seed固定随机upstream以检查非退化VJP，原容差不变；保留 `/tmp/csa-window-initial-failure.log`、`/tmp/csa-window-diagnostic.log`。随机VJP前三例通过，零窗口参考无grad的测试调用错误随后按常量语义校验，未删除该case。
- 命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29842 tests/integration_tests/compression_window_equivalence.py`，最终exit0，日志 `/tmp/csa-window-final.log`；既有窗口6组/HCA3组和metadata失败检查保持通过。
- 实现状态：CSA窗口支持模块已实现，indexer top-k/selection mask、attention及公开安装未接通；完整CSA迁移仍开放。不能以测试替身indexer关闭验收。下一步真实indexer全局compressed keys与局部query评分/causal top-k，再接CSA attention。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加现有工作区；适用全文件pre-commit排除保护example、compileall、diff check exit0，日志 `/tmp/csa-window-precommit.log`；全量 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings`：3554 passed, 2 skipped, 254 warnings in 36.34s，exit0，日志 `/tmp/csa-window-unit.log`。

### COMPRESSOR-CP-CSA-INDEXER-01：真实全局key/局部query top-k

- P1，`compression_windows.py::select_csa_windows`，来源Transformers5.9.0 DeepseekV4Indexer.forward（现有Apache-2.0头）。复用CSA overlap压缩计算indexer entries，变长all-to-all gather保留全局window顺序；局部query按全局position RoPE，FP32 ReLU dot×weights评分，按已完成窗口threshold屏蔽future，top-k及-1 sentinel沿用HF语义。无padding伪造key，不因零owner改变全局索引。
- 离散selection显式no_grad，与HF仅返回Long topk indices的不可微契约一致；attention实际compressed values仍走独立可微压缩，未将其detach。local position/q_residual校验在CP子组本地错误边界内，数据collective在边界外。
- 集成脚本扩展真实indexer比较（没有patch indexer）：CP2、CPU/Gloo FP32、local length3/4/5/1、ratio4、batch2、index heads2/dim8、q_lora8、seed71、topk1/3，选择索引与完整序列HF indexer局部切片逐元素相等，包含-1 future及全局空keys。原CSA压缩比较仍单独隔离indexer，不能混淆两份证据。
- 命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29842 tests/integration_tests/compression_window_equivalence.py` exit0，日志 `/tmp/csa-indexer-final.log`；8组真实selection与既有窗口/HCA/CSA梯度用例通过。
- 尚未接selection mask到CSA attention或公开安装；全HCA限制仍在，不宣称CSA Trainer已支持。下一步将全局selection转成attention可见mask，再验真实CSA模块输出/梯度，后续才扩展混合层安装。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加现有工作区。适用全文件pre-commit排除保护example、compileall、diff check exit0；日志 `/tmp/csa-indexer-precommit.log`。全量 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings`：3554 passed, 2 skipped, 269 warnings in 37.34s，exit0，日志 `/tmp/csa-indexer-unit.log`。

### COMPRESSOR-CP-CSA-ATTENTION-01：真实selection接入CSA模块

- P1，`compressor_attention.py` 的forward现识别CSA并调用可微overlap压缩+真实离散indexer；attention gather局部selection元数据，scatter成全局compressed可见mask，与已完成窗口因果约束相交。保持shared KV、sink、两段KV顺序和输出逆RoPE/group projection。selection形状、long dtype、device、范围在CP本地错误边界检查，presence/topk长度加入固定元数据避免rank分歧。
- 扩展 `hca_attention_equivalence.py` 真实HF CSA模块参考，无mock indexer：CP2、CPU/Gloo FP32、batch2/hidden16/heads4/KV1/head_dim8、index heads2/dim8/topk1、ratio4、sequence10/local5、sliding3，包含两个global compressed entries，非退化topk过滤。输出2e-6、输入梯度3e-6、全部有梯度参数归约后4e-6通过；indexer仅离散输出，双方对应参数grad None。HF原始完整CSA自行构建selection block bias，不复用本地mask。
- 命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29843 tests/integration_tests/hca_attention_equivalence.py`，exit0，日志 `/tmp/csa-attention-final.log`；原HCA attention/decoder/完整wrapper更新与dtype失败路径继续通过。首次较小sequence6也通过，但global仅1条目，主动扩为sequence10覆盖真实选择，不以退化例验收。
- 生产安装仍限制全HCA，CSA模块尚未通过公开入口安装；下一步放开HCA/CSA混合层并验真实wrapper/Trainer，不以本模块通过代替全迁移。padding/滑动独立层、AC/compile、设备及多轴仍开放。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加现有工作区。适用全文件pre-commit排除保护example、compileall、diff check exit0；日志 `/tmp/csa-attention-precommit.log`。全量 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings`：3554 passed, 2 skipped, 252 warnings in 37.58s，exit0，日志 `/tmp/csa-attention-unit.log`。

### COMPRESSOR-CP-MIXED-01：HCA+CSA公开入口与Trainer恢复

- P1，专用安装器现允许HCA/CSA两类层，仍提前拒绝独立sliding层；全部配置验证先于forward替换。此前hca命名的内部入口已扩展语义，后续应重命名为compressor职责并同步消费者，避免长期保留误导名称。
- 实际wrapper测试改为两层 `[heavily_compressed_attention, compressed_sparse_attention]`、真实apply_cp安装，对完整未修改HF wrapper logits3e-6/梯度6e-6/一次SGD更新参数2e-6通过。CPU/Gloo CP2、FP32、无mock indexer；命令沿用 `torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29843 tests/integration_tests/hca_attention_equivalence.py`，PYTHONPATH=.、GLOO_SOCKET_IFNAME=lo0；exit0，日志 `/tmp/csa-mixed-wrapper.log`。
- `hca_trainer_checkpoint.py` 增加LLMTUNER_TEST_CSA=1的公开两层配置和index heads2/dim8/topk1，其他E4/top2/shared1/hc_mult2/seq32/batch8。两次独立进程命令 `LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_CSA=1 LLMTUNER_TEST_EMA=dynamic PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29844 tests/integration_tests/hca_trainer_checkpoint.py full|resume /var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-csa-mixed-t9dsnuyd`：full/resume均exit0。
- CP2/DP1（无FSDP sharding）、AdamW/动态EMA、AC/compile关；step2恢复两rank更新前状态failures0，继续两步loss差0，最终状态harness all checks passed。连续loss `[4.848858833312988,4.849308967590332,4.8496294021606445,4.860266208648682]`。日志 `/tmp/csa-mixed-full.log`、`/tmp/csa-mixed-resume.log`。
- 实现状态：HCA/CSA混合已接公开入口并CPU训练恢复；不外推CSA×FSDP、sliding/padding、AC/compile、其他轴/目标设备。下一动作先准确重命名扩展后的内部入口，再处理独立sliding与失败生命周期等剩余机制。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加现有工作区；适用全文件pre-commit排除保护example、compileall、diff check exit0，日志 `/tmp/csa-mixed-precommit.log`。全量 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings`：3554 passed, 2 skipped, 260 warnings in 36.15s，exit0，日志 `/tmp/csa-mixed-unit.log`。

### COMPRESSOR-CP-NAMING-01：入口职责命名与消费者迁移

- 新增路径已支持HCA/CSA，旧hca入口名称误导职责。重命名映射：`hca_ulysses_attention`→`compressor_ulysses_attention`，`hca_attention_forward`→`compressor_attention_forward`，`hca_decoder_forward`→`compressor_decoder_forward`，`apply_hca_context_parallel`→`apply_compressor_context_parallel`；内部strategy `hca_ulysses`→`compressor_ulysses`。同步apply_cp、HF wrapper及集成消费者，删除旧名而不加别名。`compress_hca_windows`与`compress_csa_windows`仍准确区分算法，保留。
- 此为新增未提交内部API名称迁移，不改用户ParallelConfig策略ulysses、模型parameter/state-dict键或DCP格式。历史溯源/批次旧名保留，不机械改写旧证据。运行时strategy只记录于model，不作为训练checkpoint schema。
- `rg`核对llmtuner/tests旧四函数及内部策略无引用。真实CPU/Gloo CP2脚本 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29843 tests/integration_tests/hca_attention_equivalence.py` exit0，HCA/CSA模块、混合wrapper logits/梯度/更新和失败协调通过，日志 `/tmp/compressor-naming.log`。随后仅调整提示文本/docstring。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加原工作区及本批；pre-commit首次发现docstring超长，按88列修正，未忽略gate。compileall、diff check通过，最终适用pre-commit通过；全量 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings`：3554 passed, 2 skipped, 262 warnings in 36.58s，exit0，日志 `/tmp/compressor-naming-unit.log`。
- 完整迁移仍开放：独立sliding层、padding、CSA×FSDP、AC/compile、目标设备及其他全局必需机制。下一步推进独立sliding层的同一sharedKV/sink通信，避免为其复制HCA专用壳。

### COMPRESSOR-CP-SLIDING-01：三类层统一生产入口

- P1，`compressor_attention.py` 支持独立sliding_attention：compress_rate=None明确表示无压缩，compressed entries为空，窗口计数/因果mask分支显式处理；不伪造压缩率或条目。继续复用sharedKV/head交换、sink与输出逆RoPE。安装器允许sliding/HCA/CSA三类，非法类型仍在替换forward前拒绝。
- 真实公开安装wrapper测试升级三层 `[sliding_attention, heavily_compressed_attention, compressed_sparse_attention]`，其他配置保持CPU/Gloo FP32、CP2/DP1/TP/PP/EP1、heads4/KV1/hidden16。对独立完整未修改HF wrapper logits3e-6、归约全部梯度6e-6、一次SGD更新参数2e-6通过，实际sliding边界跨CP shard。此前真实HCA/CSA模块与输入失败协调继续通过。
- 命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29843 tests/integration_tests/hca_attention_equivalence.py`，exit0，日志 `/tmp/sliding-compressor.log`。没有测试替身模型/通信或放宽断言。
- 验证边界：三层wrapper数值已验，三层Trainer恢复尚未跑；CSA×FSDP、padding/dropout、AC/compile、目标设备/性能、TP/PP组合保持开放。下一步用三层公开Trainer配置验生命周期，再推进未闭合故障协议或其他必需迁移。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加现有工作区；适用全文件pre-commit排除保护example、compileall、diff check exit0，日志 `/tmp/sliding-compressor-precommit.log`；全量 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings`：3554 passed, 2 skipped, 257 warnings in 36.97s，exit0，日志 `/tmp/sliding-compressor-unit.log`。

### COMPRESSOR-CP-THREE-FSDP-01：三类层训练与恢复

- P1；公开Trainer配置加入 `LLMTUNER_TEST_SLIDING=1`，与CSA开关组合生成sliding/HCA/CSA三层；不修改生产实现、不替换通信或模型。更新测试说明，默认配置保持HCA。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加现有暂存/未暂存工作区；本批仅扩展 `hca_trainer_checkpoint.py` 的测试配置与说明。CPU/Gloo FP32，CP2×dp_shard2/custom FSDP，TP/PP/EP1，AdamW/动态EMA，AC/compile关闭。
- 实际命令（full和resume分别独立执行）：`LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_HCA_DP_SIZE=2 LLMTUNER_TEST_CSA=1 LLMTUNER_TEST_SLIDING=1 LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29845 tests/integration_tests/hca_trainer_checkpoint.py full|resume /var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-three-cp-fsdp-myk5en5l`。
- full/resume均exit0；四rank更新前状态failures=0，恢复后两步loss最大绝对差0，最终模型/AdamW/scheduler/EMA/trainer/token计数检查通过。连续loss `[4.846107482910156,4.862561225891113,4.849610805511475,4.858764171600342]`。日志 `/tmp/three-cp-fsdp-full.log`、`/tmp/three-cp-fsdp-resume.log`。未独立比较RNG payload，不据此宣称全部随机状态验收。
- 实现/验证结论：三类层组合的CPU实际Trainer训练恢复通过；不外推padding/packing/dropout、AC/compile、TP/PP/EP或目标设备。生产实现未变，复用COMPRESSOR-CP-SLIDING-01的全量单测3554 passed/2 skipped及生产数值参考；本轮不重跑无关全量单测。
- 本轮受影响三文件pre-commit exit0（YAML无适用文件跳过），测试脚本ruff check/format、compileall及git diff HEAD --check均通过；仅测试配置和文档变更，未重跑累计全文件pre-commit。
- 下一动作：继续DOMINO-FAILURE-01的已确认专家局部失败协议；其余全局必需迁移仍开放。

### DOMINO-PREFLIGHT-01：通信前局部输入失败协调

- P1 / `parallel/expert_parallel/domino_dispatch.py::DominoRoutedExperts.forward`。完整routing shape校验、ID计数此前在rank本地直接抛错，peer继续进入prepare_dispatch_counts，存在挂起路径；负ID使bincount失败，越界ID使counts长度失配。该问题独立于DOMINO-FAILURE-01的专家执行异常。
- 仅将无collective的shape/ID校验及两slice本地准备放入EP组 `synchronize_local_errors`；显式要求int64及 `[0,num_experts)` ID。失败rank保留ValueError，成功peer接收带phase的RuntimeError。局部EP1继续直接执行；未包裹可能触发FSDP通信的inner_experts。每次EP调用增加一次通信前状态归约；CUDA性能/重叠影响待目标设备实测，不宣称免费恢复。
- `tests/integration_tests/ep_domino_equivalence.py` 增加rank0独有四种错误：shape、负ID、越界ID、错误dtype。每例双方捕获预期错误，再barrier、复用同一模型/dispatcher执行健康forward、再barrier；真实通信无mock。原有24项RoutedExperts/Domino组合的输出、输入/score/参数梯度对独立本地专家FP64参考，rtol/atol均1e-9，包含不均匀/空/单token、空接收与实际checkpoint重计算，继续通过。
- 实际命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29846 tests/integration_tests/ep_domino_equivalence.py`，CPU/Gloo EP2、TP/PP/CP/FSDP1；exit0，日志 `/tmp/domino-preparation.log`。定向 `pytest -q tests/unit_tests/cpu/models/test_domino_experts.py --disable-warnings`：5 passed，exit0。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加现有工作区及本批两文件修改。适用全文件pre-commit（排除保护example）exit0，日志 `/tmp/domino-preparation-precommit.log`；compileall、diff HEAD --check通过。
- 本轮全量 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings` ：3554 passed、2 skipped、268 warnings in 36.39s，exit0，日志 `/tmp/domino-preparation-unit.log`。
- 实现与CPU准备阶段验收通过；DOMINO-FAILURE-01保持未关闭：dispatch在途/专家hook已有通信的异常不可由本批保证，仍需明确FSDP生命周期与取消/收尾协议。下一动作继续该边界，CUDA/NCCL stream/event与性能保持待验。

### REVIEW-MANIFEST-CP-01：新增CP文件清单与专家异常边界复核

- 上一轮DOMINO-PREFLIGHT-01为已验证进展；本轮重新读取目标文件，核对当前源码与逐文件主表。发现漏列 `compression_windows.py`、`compressor_attention.py`，无失效路径或重复。已按此前逐批阅读/数值证据补录，不将本轮路径枚举冒充重新逐行阅读；两文件均保留“迁移未验收”。
- 实际集合核对：以 `Path('llmtuner').rglob('*.py')` 为源码集，提取以源码路径为首列的Markdown主表行，用Counter验证唯一性；补录后集合差均空、重复为空。此证据只证明清单集合完整，不证明全局验收。
- DOMINO-FAILURE-01边界复核：阅读 `models/common/moe/experts.py` 的GroupedExperts/RoutedExperts完整实现，以及 `fully_shard_parallel/plan.py` 的pattern选择和expert placement、`domino_dispatch.py` 的调用位置。GroupedExperts激活函数可注入任意nn.Module，且pattern FSDP能在后代/外部hook_module安排通信；仅根据GroupedExperts类型、当前参数是否Tensor或顶层是否FSDPModule，均不能证明整个调用无collective。主表experts.py改为异常协议未验收，避免将阅读状态误当缺陷关闭。
- 下一动作：为已有dispatch消费/专家调用/收尾明确上层跨EP与sparse FSDP组的一致阶段协议；先区分无FSDP局部专家和含hook专家，不能在未证明无通信的调用体外套单EP组错误归约。通信内部/进程死亡仍属独立故障模型；不关闭Domino或以拒绝组合代替迁移。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加原工作区；本轮仅改台账，无生产代码变化。复用DOMINO-PREFLIGHT-01的3554 passed/2 skipped及真实两rank正常/失败前置验证，不记为本轮执行。 本轮主表集合断言、文档pre-commit和git diff HEAD --check均exit0，Python/YAML无适用文件的hook跳过。

### DOMINO-DRAIN-01：专家失败时等待已发起dispatch

- P1 / `parallel/expert_parallel/domino_dispatch.py`，关联DOMINO-FAILURE-01。CPU两slice先启动dispatch0/1，专家0失败使dispatch1无人消费；旧真实复现退出出现unwaited collective警告。正常路径不变，异常路径保留已成功返回的pending对象，逐个materialize其routed_input，等待已有通信后才抛出原异常。某次wait失败仍尝试其余wait，原异常为主异常、cleanup异常作为cause；不在错误路径新增collective，不包裹专家为纯本地阶段。
- 单测两例覆盖正常wait和第一次wait失败，均要求等待两个transfer且保留原异常对象。定向pytest：7 passed，exit0。通过在独立Python进程中将 `git show HEAD:llmtuner/parallel/expert_parallel/domino_dispatch.py` 编译到该模块命名空间后执行同一新增测试，旧实现2 failed/5 deselected、exit1（waited为空），日志 `/tmp/domino-drain-before.log`；仅进程内替换，无工作区回退。
- 真实故障复现命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29847 /tmp/domino_failure_probe.py`，CPU/Gloo EP2、FP32、4专家/top2/本地3token、process-group timeout5秒，rank0专家forward注入ValueError。旧 `/tmp/domino-local-failure-probe.log` 两rank退出均报告unwaited；当前 `/tmp/domino-drain-failure.log` 无该警告，rank0保留原ValueError，rank1仍在combine报5秒send timeout。探针捕获预期错误exit0，只证明复现执行完毕，不能标为异常恢复通过；没有健康barrier/retry。
- 正常路径命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29848 tests/integration_tests/ep_domino_equivalence.py`，exit0，日志 `/tmp/domino-drain-equivalence.log`；24种正常配置的独立参考FP64输出/梯度1e-9及四种输入失败后的barrier/健康重试继续通过。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加已有工作区及本批异常收尾/两单测。全量 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings`：3556 passed、2 skipped、256 warnings in36.89s，exit0，日志 `/tmp/domino-drain-unit.log`。适用全文件pre-commit排除保护example、compileall、diff HEAD --check exit0，日志 `/tmp/domino-drain-precommit.log`。
- 本批关闭CPU已发起dispatch遗留等待的局部缺陷，DOMINO-FAILURE-01继续开放；CUDA独立stream/event收尾、通信内部失败、跨EP/FSDP错误协调尚无验收。等待现有work不能取消peer已进入的combine，下一动作仍需一致的专家失败阶段协议，不能将无警告等同进程组健康。

### DOMINO-LOCAL-EXPERT-01：局部专家异常协调与重试

- P1，关联DOMINO-FAILURE-01；前两批只修复输入准备与dispatch等待，专家故障仍使peer进入combine。本批为可证明无collective的标准专家计算加入双方状态协调，先解决该明确边界，原含hook/FSDP/DTensor/compile异常协议仍未关闭。
- `domino_dispatch.py::_can_coordinate_experts` 在两个dispatch前只判断一次：精确GroupedExperts类型和未替换的forward/_grouped_mm、精确内置SwiGLU/SituGLU及原forward、全部后代无forward/prehook/编译call、无全局forward/prehook、参数和输入为普通Tensor/Parameter。EP组min归约资格，任一rank不满足则双方都沿已有调用路径执行，不能一方同步另一方执行内部通信。不把未知模块或含FSDP hook的调用包入局部同步。
- `_run_experts` 对已满足边界的计算，在返回combine之前协调原异常；某rank失败，其他rank获得带phase的RuntimeError，双方不进入该slice combine。两slice均采用；CPU异常继续等待已发起dispatch。CUDA异常在通信stream materialize已排队dispatch和已存在combine0，再让compute stream等待；没有目标CUDA证据，不能据模拟宣称设备恢复。
- 调度审查发现“每slice计算前资格归约”会等待dispatch1/combine0，破坏两处重叠；已将资格归约前移到dispatch0前，计算后的错误归约仍增加同步开销。无新增计算前阻塞归约，保留dispatch1/expert0、combine0/expert1的调度关系；实际重叠/性能待CUDA trace与测量，不宣称无开销。
- `ep_domino_equivalence.py` 在标准实现内部通过rank0独有F.linear异常，分别失败slice0/1；不是替换专家forward而绕过资格判断。双方捕获预期异常后barrier，同实例重试前向与反向，检查输入/score/所有专家参数梯度存在、输出和输入梯度有限，再barrier。另验仅rank0注册prehook时双方资格均false，防止分歧collective。
- 实际命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29849 tests/integration_tests/ep_domino_equivalence.py`；CPU/Gloo FP64独立参考24项正常输出/梯度1e-9，故障/重试FP32，EP2、TP/PP/CP/FSDP1；exit0，日志 `/tmp/domino-expert-boundary-final2.log`。原四种输入错误恢复仍覆盖。
- 单测新增无hook/有prehook/后代posthook/自定义activation/编译call/远端资格失败六例；CUDA stream模拟增加slice0/1错误，验只等待已排队结果、保持原异常、不额外combine；正常调度单测确认资格判断在counts与dispatch前。最终定向pytest：15 passed、exit0。模拟不证明CUDA/NCCL数值或真实allocator生命周期。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加已有工作区及本批生产/测试修改。适用全文件pre-commit排除保护example、compileall、diff HEAD --check exit0，日志 `/tmp/domino-expert-boundary-precommit.log`；最终全量 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings`：3564 passed、2 skipped、268 warnings in36.00s、exit0，日志 `/tmp/domino-expert-boundary-unit-final.log`。随后仅补齐CUDA错误收尾对已存在output1的等待，重跑定向15项与受影响gate。
- 实现：标准无通信专家故障协调已接生产调用路径；验证：CPU双rank通过后仅关闭此子边界。DOMINO-FAILURE-01整体仍开放，剩余含专家FSDP hook/跨EP+sparse FSDP组、通信内部/设备异步fatal错误、目标设备stream/event与性能。下一动作推进hook生命周期内的局部计算错误捕获与上层一致收尾，不能将资格false当作迁移完成或配置拒绝。

### DOMINO-ERROR-DOMAIN-01：跨EP组传播与外层FSDP重试

- P1 / `parallelize.py`→`expert_parallel.py::apply_ep`→`domino_dispatch.py`，关联DOMINO-FAILURE-01。前批仅EP组协调：EP2×efsdp2时，一个EP组抛错不能通知另一个仍将参与后续dense/sparse FSDP通信的EP组。资格检查也须在相同完整域一致，不能仅在每个EP子组一致。
- 生产入口仅对Domino解析现有loss mesh的group，传入apply_ep新增可选error_group，绑定所有本地Domino模块。该mesh在当前PP stage内覆盖dp_replicate×dp_shard×CP×TP，与sparse的dp_replicate×efsdp×EP覆盖相同rank集合；不创建新group、不跨PP stage。输入准备、专家资格与局部专家错误统一用该域；token dispatch/combine仍用原EP group。独立apply_ep/手工构建不传时保留EP域，调用方承担外层域选择。
- error_group只是运行时ProcessGroup引用，不进state_dict、不改模型/optimizer/DCP键。新增参数为可选，未删除旧API。专家本体有hook/DTensor/compile等情形仍不进入局部同步，不宣称已支持这些异常路径。
- 扩展 `tests/integration_tests/ep_fsdp_equivalence.py --domino-expert-failure`：通过真实公开ParallelConfig和parallelize入口选择Domino，断言错误域size4；在首层标准专家内部rank0独有F.linear异常，四rank必须全部捕获对应异常，再barrier、显式调用PyTorch2.14根 `reset_iter_state()`、清梯度、barrier，在同实例执行既有独立完整HF模型参考的loss/router/全部专家梯度验证。模型eval、autograd开启，不是完整Trainer/optimizer更新或router训练统计恢复验收。
- custom命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29850 tests/integration_tests/ep_fsdp_equivalence.py --fsdp-implementation custom --domino-expert-failure`，exit0，日志 `/tmp/domino-fsdp-domain.log`；native命令改port29851、implementation native，exit0，日志 `/tmp/domino-fsdp-domain-native.log`。CPU/Gloo FP32、dp_shard4/EP2/efsdp2、TP/PP/CP1、8专家top2、每rank16token、AC/compile关闭；两者均all four ranks aborted/root reset complete、failed ranks0、all checks passed。loss311.874268，独立ref311.874283，沿用tol1e-4；未调整容差。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加累计工作区及本批三生产文件/集成脚本修改。定向Domino+EP单测18 passed、exit0；全量 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings`：3564 passed、2 skipped、252 warnings in36.43s、exit0，日志 `/tmp/domino-fsdp-domain-unit.log`。适用全文件pre-commit排除保护example、compileall、diff HEAD --check通过，日志 `/tmp/domino-fsdp-domain-precommit.log`。验证后仅补测试说明及float(loss.detach())消除测试警告，不改数学参考。
- 子边界：外层native/custom FSDP包围标准局部专家时，跨EP故障传播和显式root reset后的数值重试通过。DOMINO-FAILURE-01保持开放：专家内部/直接专家hook、grouped/relocated hook、多PP/TP/CP错误传播、训练统计/累积失效及CUDA真实stream/event和性能仍待验；通信内部或rank死亡另属故障模型。下一动作定位直接专家FSDP hook的计算错误与posthook收尾边界。

### DOMINO-DIRECT-FSDP-01：直接专家hook的延迟计算异常

- P1 / `models/common/moe/experts.py`→`parallel/expert_parallel/domino_dispatch.py`，关联DOMINO-FAILURE-01。直接专家FSDP包含通信pre/posthook，不能把整次__call__当局部计算。新增显式 `ExpertErrorCapture` 记录与forward可选error_capture参数，把原计算体归入 `_compute`；普通调用保持立即抛错。捕获仅覆盖计算体，不覆盖pre/posthook；失败返回同storage的detached输入占位供posthook收尾，外层检查记录并协调抛错，绝不传给combine或backward。
- 记录为不注册pytree的opaque dataclass，避免FSDP输入cast复制空list导致错误记录丢失；每次调用清理旧error，运行时局部对象不进模块state_dict。捕获模式检查weights已materialize为普通Tensor/Parameter，防止计算体意外进入DTensor通信。forward公共名称未变，新增可选keyword；参数名/checkpoint键不变。
- 资格判断扩展到PyTorch原生生成的精确FSDPModule/GroupedExperts组合类；只接受单模块state、唯一pre/posthook handle及bound callback均匹配该state的已知直接hook。编译、自定义forward/_compute/_grouped_mm、额外或全局hook、自定义activation仍排除。直接hook在局部捕获之外正常运行，然后用前批完整错误域传播记录的计算异常。grouped/relocated/custom hook不由此自动验收。
- 单测：异常记录保留原异常对象，实际forward posthook在传播前运行，正常不捕获调用仍立即抛错，pytree保持记录identity，复用记录清旧error；正常输出和input/全部参数梯度与无capture路径逐元素相等。Domino定向17 passed、exit0。第一次直接FSDP探针因返回输入view产生FSDP warning，改为detach占位以避免为失败输出注册无用pre-backward view hook；保留 `/tmp/domino-direct-expert-custom.log` 原证据，最终两实现无该warning。
- 扩展 `ep_fsdp_equivalence.py --expert-fsdp` 用公开pattern `model.model.layers.{*}.mlp.routed_experts.inner_experts` 为每个专家建立FSDP单元。故障wrapper断言coordinate_experts确实true，rank0标准F.linear内抛ValueError；四rank都收到错误，reset前断言专家state为IDLE，barrier后显式root reset，再在同实例对独立完整HF参考验loss/router/全部专家梯度，未降低原tol1e-4。
- 最终custom命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29852 tests/integration_tests/ep_fsdp_equivalence.py --fsdp-implementation custom --domino-expert-failure --expert-fsdp`；native改port29853、implementation native。均exit0，all four ranks aborted/root reset complete、failed ranks0/all checks passed；loss311.874268/ref311.874283。日志 `/tmp/domino-direct-expert-custom-final2.log`、`/tmp/domino-direct-expert-native-final.log`。CPU/Gloo FP32、EP2/efsdp2/dp_shard4、TP/PP/CP1、eval但autograd开启、8专家top2、每rank16token，AC/compile关；不是Trainer/optimizer/训练统计恢复验收。
- 两rank正常/故障与重计算脚本 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29854 tests/integration_tests/ep_domino_equivalence.py` exit0，日志 `/tmp/domino-defer-equivalence.log`；24项FP64独立参考1e-9及原输入/局部专家故障后重试通过，不外推直接FSDP×AC/compile。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加累计工作区及本批两生产文件/两测试文件。全量 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings`：3566 passed、2 skipped、251 warnings in39.98s、exit0，日志 `/tmp/domino-direct-expert-unit.log`；适用全文件pre-commit排除保护example、compileall、diff HEAD --check exit0，日志 `/tmp/domino-direct-expert-precommit.log`。
- 子边界已实现且CPU验证：直接单专家native/custom FSDP hook中的局部计算异常可完成posthook，再一致传播并root reset重试。DOMINO-FAILURE-01整体仍开放，剩余grouped/relocated hook、编译/其他轴、多通信上下文训练生命周期及CUDA/NCCL真实收尾与性能。通信hook内部故障/进程死亡不由局部捕获恢复。下一动作推进grouped/relocated hook的识别与一致收尾验收。

### DOMINO-GROUPED-RELOCATED-01：共享专家与迁移hook的异常收尾

- P1 / `domino_dispatch.py::_can_coordinate_experts`，关联DOMINO-FAILURE-01。直接单模块handle检查不能识别PyTorch2.14 grouped FSDP的MultiHandle/wrapped callback，导致共享专家合并state后失去局部计算错误协调。扩展为核对专家属于state._modules、专家上唯一pre/posthook的ID归属state handle（含MultiHandle），并沿有循环保护的__wrapped__链核对最终state callback；原额外hook/自定义计算/compile等限制保持。通信hook仍在compute捕获体外，不把通信错误当局部计算错误恢复。
- `ep_fsdp_equivalence.py --shared-experts` 在独立HF参考中真实共享两层全部专家权重，并在真实EP转换后将对应local专家参数绑定为同一对象，再交给生产pattern FSDP按共享参数合并state；该测试fixture显式构造共享别名，不声称HF转换自动保留所有共享别名。断言两个专家同state，首层posthook后已从待执行集合移除、第二层仍待执行；显式root reset后集合为空/state IDLE，同实例loss/router/共享专家梯度对独立参考通过。
- `--relocated-expert-hooks` 通过实际FSDPConfig pattern override的hook_module=model迁移专家hook到共同祖先；只适用custom。错误发生时祖先尚未posthook，专家state FORWARD为预期，不伪称异常前已IDLE；root reset后断言IDLE、待执行集合空，所有通信上下文all_gather_state=None、reduce_scatter_states与post_forward_order为空。独立state与共享group两类都验证至少两个context真实存在。
- 四项真实四进程命令均以 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=4 --master_addr=127.0.0.1` 开始，脚本均 `tests/integration_tests/ep_fsdp_equivalence.py --domino-expert-failure --expert-fsdp`：①port29855/custom/--shared-experts，日志 `/tmp/domino-grouped-custom.log`；②port29856/native/--shared-experts，日志 `/tmp/domino-grouped-native.log`；③port29857/custom/--shared-experts/--relocated-expert-hooks，日志 `/tmp/domino-grouped-relocated.log`；④port29858/custom/--relocated-expert-hooks，日志 `/tmp/domino-relocated-multictx.log`。implementation用 `--fsdp-implementation custom|native`，每项最终exit0、all four ranks aborted/root reset complete、failed ranks0/all checks passed。
- CPU/Gloo FP32、dp_shard4/EP2/efsdp2、TP/PP/CP1、两层8专家top2、每rank16token、AC/compile关、eval且autograd开启。共享参考loss311.769608/分布式311.769592；独立专家参考311.874283/分布式311.874268；原tol1e-4不变。每项先验收异常后的state/barrier/root reset，再同实例完成独立HF参考前后向。未覆盖optimizer更新/Trainer训练统计，也未覆盖通信hook内部故障。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加累计工作区及本批生产资格判断/集成脚本修改。全量 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings`：3566 passed、2 skipped、263 warnings in35.33s、exit0，日志 `/tmp/domino-grouped-unit.log`；适用全文件pre-commit排除保护example、compileall、diff HEAD --check exit0，日志 `/tmp/domino-grouped-precommit.log`。
- 子边界实现且CPU验证：grouped FSDP的计算异常传播与显式reset，共同祖先relocated hook的多context reset与数值重试。DOMINO-FAILURE-01整体仍未关闭：训练模式router统计/梯度累积失效、完整Trainer恢复、其他并行轴、编译及CUDA/NCCL stream/event/性能仍待验证；通信内部失败/进程死亡另列。下一动作检查训练模式失败迭代的router统计及已有梯度是否被完整丢弃。

### MOE-FAILED-COUNTS-01：成功返回后提交块内token计数

- P1 / `models/common/moe/layer.py::_forward_tokens`；训练模式原先在routed/shared专家前提交tokens_per_expert_E，专家异常仍污染累计计数。把原padding过滤/非重计算计数逻辑移到两条专家路径成功返回之后；数学公式、buffer键及成功forward计数不变。失败保留调用前已有累积，不清除其他成功microbatch计数。这是块内提交边界，不是整个训练step回滚。
- 新增routed/shared两种异常单测：预置非零prior计数，失败保持逐元素不变/原异常identity，恢复后前后向成功、只追加一次实际routing map计数。定向routing+padding：118 passed、exit0。同一新增测试在独立进程内加载HEAD旧layer.py执行：2 failed/84 deselected、exit1，日志 `/tmp/moe-count-commit-before.log`，不回退工作区。
- 集成脚本 `ep_fsdp_equivalence.py` 故障阶段启用train模式、预置首MoE计数3，要求四rank故障后原计数不变；后续eval独立参考和reset断言保持。首跑遗漏AuxLoss step denominator，在专家前触发既有配置校验，exit1，日志 `/tmp/moe-count-commit-fsdp.log`；补齐 `AuxLoss.set_step_denominator(world*TOKENS_PER_RANK, model_parts=[model])` 的正常装配，不关闭aux或弱化错误断言。
- 最终命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29860 tests/integration_tests/ep_fsdp_equivalence.py --fsdp-implementation custom --domino-expert-failure --expert-fsdp --shared-experts --relocated-expert-hooks`，exit0，日志 `/tmp/moe-count-commit-fsdp-final.log`。CPU/Gloo FP32、dp_shard4/EP2/efsdp2/其他轴1，共享专家grouped state/root relocated hook，AC/compile关；所有rank失败首块计数不变，root reset、eval同实例loss/router/专家梯度参考tol1e-4通过，loss311.769592/ref311.769608。
- 两rank原Domino脚本port29859正常/局部失败/重试/重计算exit0，日志 `/tmp/moe-count-commit-domino.log`。全量 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings`：3568 passed、2 skipped、275 warnings in33.06s，exit0，日志 `/tmp/moe-count-commit-unit.log`。HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加累计工作区及本批layer/单测/集成脚本修改。适用全文件pre-commit排除保护example、compileall、diff HEAD --check通过，日志 `/tmp/moe-count-commit-precommit.log`；随后仅补训练集成fixture denominator并重跑实际四rank。
- 关闭失败MoE自身token计数污染；DOMINO-FAILURE-01整体开放。已阅读routing.py：QuantileBalancer.observe与AuxLoss指标在router内部早于专家执行，其他层已经完成的统计和已累积梯度也不由本批回滚；Trainer.train异常当前关闭资源并抛错，不提供自动step重试。下一动作明确整个失败累积窗口的统计/梯度丢弃接口与恢复契约，保留quantile/aux、Trainer及目标设备缺口。

### MOE-NEW-WINDOW-01：新训练窗口丢弃路由临时统计

- P1 / `trainer/trainer.py::train_step`；新step已清参数梯度、AuxLoss.set_step_denominator已清辅助指标，但MoE token counter与quantile histogram此前只在成功optimizer prehook排空。失败窗口或无balancing hook运行可遗留统计，下个step混入旧观测。
- train_step在zero_grad后遍历本地parts中的MoE，no_grad清tokens_per_expert_E，QuantileBalancedTopKRouter同时清required_bias_histogram_EB；不改expert_bias、参数、optimizer/scheduler/EMA或checkpoint格式。不在microbatch之间清理，窗口内部仍累积；不加入collective。不自动重试失败窗口，FSDP root reset/数据游标/RNG仍由完整恢复契约负责。
- 新增两种路由测试：预置非零计数/直方图及真实参数梯度，进入实际train_step，在LR读取阶段检查梯度None、统计全零、expert_bias逐元素不变，再主动终止避免无关装配。当前2 passed/125 deselected；同测试独立进程加载HEAD旧Trainer：2 failed/125 deselected、exit1，日志 `/tmp/moe-new-window-before.log`。
- 首轮全量2 failed/3568 passed/2 skipped，均为旧精简测试通过__new__创建Trainer未配置model_parts，在新入口访问时报AttributeError；补齐两个fixture的真实nn.Module parts，不放宽生产初始化要求，保留 `/tmp/moe-new-window-unit.log`。最终 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 pytest -q --disable-warnings`：3570 passed、2 skipped、250 warnings in33.42s、exit0，日志 `/tmp/moe-new-window-unit-final.log`。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加累计工作区及本批trainer/测试修改。适用全文件pre-commit排除保护example、compileall、diff HEAD --check通过，日志 `/tmp/moe-new-window-precommit.log`，该gate在补两fixture前执行，补fixture后的trainer/测试/台账pre-commit、compileall和diff HEAD --check也exit0（YAML无适用文件跳过）。
- 实现关闭新step统计泄漏的入口缺口，不宣称整个失败step原子回滚；单测未证明完整Trainer健康重试、FSDP待处理状态、optimizer部分更新或设备错误恢复。下一动作按实际Trainer失败点核对数据/RNG、路由bias optimizer prehook与更新失败的状态归属；DOMINO-FAILURE-01和目标设备缺口继续开放。

### OPTIMIZER-PARTIAL-FAILURE-01：确认顺序更新的恢复边界

- P1 / `components/optimizer/optimizer.py::OptimizersContainer.step`、Trainer.train_step及MoE bias prehook。阅读确认多个inner optimizer依次更新、bias prehook在参数更新前运行；后续optimizer失败可留下此前参数/moments及bias已改变。Trainer传播错误，scheduler/EMA在optimizer成功之后，train仅成功step进入save；不存在隐含原子回滚或自动重试保证。
- 新增真实双optimizer/双part测试：先warm Adam并deepcopy全部模型/optimizer快照，给两part非零梯度，在第二optimizer注入原始异常；确认第一part已变、第二part未变，然后完整恢复两part和FQN optimizer state，逐键精确检查并清梯度。此测试明确zero_grad不能回滚部分更新，不通过吞异常制造恢复。未模拟分布式通信或声称disk DCP覆盖。
- step docstring明确非原子更新和checkpoint恢复要求；不引入每步全模型快照或改变optimizer数学。故障发生在prehook内部时同样可能有局部提交，仍需完整恢复，具体bias故障注入后续覆盖。
- 实际命令 `pytest -q tests/unit_tests/cpu/components/optimizer/test_optimizer_container.py --disable-warnings`：40 passed、exit0；受影响两文件pre-commit、compileall、diff HEAD --check exit0。HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加累计工作区及本批docstring/新增契约测试。生产行为未变，复用MOE-NEW-WINDOW-01全量3570 passed/2 skipped，不记为本轮执行。
- 下一动作：完整Trainer从最近成功checkpoint恢复、重新读取失败窗口与RNG/路由统计/optimizer/EMA一致性；局部异常后的同实例继续只在明确未更新且已完成FSDP reset的子契约内成立，不能推广到部分optimizer提交。DOMINO-FAILURE-01整体及目标设备验证继续开放。

### TRAINER-RNG-SNAPSHOT-01：恢复后更新前显式RNG比较

- 验证缺口 / `tests/integration_tests/pp_checkpoint_equivalence.py::_runtime_state`。原harness只比较模型/optimizer/scheduler/EMA/trainer，不应由loss相等推断所有RNG恢复。本批增加可选LLMTUNER_TEST_RNG_STATE=1，调用生产RNGCheckpointState取得Python/NumPy/CPU/当前设备及拓扑payload；decode后NumPy数组转Tensor，复用逐键比较，避免pickle字节或数组==的错误判据。默认关闭以保留既有跨拓扑排除RNG场景；不改生产checkpoint。
- 两次独立进程实际命令 `LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_CSA=1 LLMTUNER_TEST_SLIDING=1 LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29861 tests/integration_tests/hca_trainer_checkpoint.py full|resume /tmp/llmtuner-rng-compressor-20261015`，full/resume均exit0。CPU/Gloo FP32、CP2/DP1/TP/PP/EP1、三层sliding/HCA/CSA、动态EMA、AC/compile关。更新前两rank包含RNG的逐键状态failures0，后两步loss差0，最终既有状态检查all checks passed。日志 `/tmp/compressor-rng-full.log`、`/tmp/compressor-rng-resume.log`。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加累计工作区及本批测试快照变化；受影响脚本pre-commit和diff HEAD --check exit0。仅测试修改，复用MOE-NEW-WINDOW-01全量3570 passed/2 skipped，不记为本轮执行。
- 范围仅同拓扑CPU更新前随机状态恢复；设备RNG字段为None，不证明CUDA/NPU，未独立验最终RNG或故障后完整Trainer恢复。下一动作在真实故障/新runtime恢复场景启用该快照与数据游标检查；全局目标及DOMINO-FAILURE-01仍开放。

### TRAINER-FAILED-COMMIT-RESUME-01：真实参数提交失败后的新runtime恢复

- `pp_checkpoint_equivalence.py` 新增可选LLMTUNER_TEST_FAILED_UPDATE=1恢复场景：先load既有split checkpoint，实际data_iterator读取下一窗口，真实train_step/optimizer.step完成参数更新后注入RuntimeError，断言更新确已执行，close失败runtime并barrier；新建独立Trainer，再走正常load及全部更新前/后续训练比较。不是zero_grad伪回滚，不改生产异常处理，不自动保存失败step。
- 实际命令 `LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_CSA=1 LLMTUNER_TEST_SLIDING=1 LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_FAILED_UPDATE=1 PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29862 tests/integration_tests/hca_trainer_checkpoint.py resume /tmp/llmtuner-rng-compressor-20261015`，exit0，日志 `/tmp/compressor-failed-update-resume.log`。复用TRAINER-RNG-SNAPSHOT-01独立full参考/step2 checkpoint，来源未修改。
- CPU/Gloo FP32、CP2/DP1/TP/PP/EP1、sliding/HCA/CSA三层、动态EMA、AC/compile关。两rank均discarded failed runtime after optimizer commit；新runtime更新前模型/AdamW/scheduler/EMA/trainer/token计数及Python/NumPy/CPU RNG逐键failures0，后续两步loss差0，最终既有状态all checks passed。后续loss比较同时验证实际loader位置续读，但未独立逐键比较loader payload。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加累计工作区及本批测试修改；脚本pre-commit exit0，生产未改，复用MOE-NEW-WINDOW-01全量3570 passed/2 skipped。不外推EP/FSDP故障、部分单optimizer内部更新、设备异步错误、完整train()退出或最终RNG。下一动作把相同故障恢复机制用于Domino×custom FSDP生产Trainer；全局目标仍开放。

### DOMINO-TRAINER-FAILED-COMMIT-01：EP×custom FSDP新runtime恢复

- 测试配置 `pp_checkpoint_equivalence.py` 增加LLMTUNER_TEST_DISPATCHER公开ExpertParallelConfig选择；Domino场景断言真实DominoRoutedExperts存在且error_group已由生产装配绑定，不以配置值代替机制触达。默认alltoall不变，无生产修改。
- full命令 `LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_MOE=1 LLMTUNER_TEST_EP_SIZE=2 LLMTUNER_TEST_DISPATCHER=domino LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29863 tests/integration_tests/pp_checkpoint_equivalence.py full /tmp/llmtuner-domino-failed-update-20261015`；resume同配置增加LLMTUNER_TEST_FAILED_UPDATE=1、phase改resume。两独立torchrun均exit0，日志 `/tmp/domino-trainer-full.log`、`/tmp/domino-trainer-failed-resume.log`。
- CPU/Gloo FP32、qwen3_moe两层、dp_shard4/EP2/efsdp2、TP/PP/CP1、AdamW/动态EMA、AC/compile关闭。失败runtime已load step2并真实执行下一窗口optimizer更新，随后各rank注入RuntimeError并close；新建Trainer重新load。四rank更新前模型/AdamW/scheduler/EMA/trainer计数及Python/NumPy/CPU RNG逐键failures0，后续两步loss差0、最终state all checks passed。连续loss `[4.884032249450684,4.868349552154541,4.848840713500977,4.865332126617432]`。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加累计工作区及本批测试配置/触达断言；脚本pre-commit exit0，diff HEAD --check通过；无生产变化，复用MOE-NEW-WINDOW-01全量3570 passed/2 skipped，不记为本轮执行。
- 已验收该配置下更新提交后失败、丢弃runtime并从成功checkpoint恢复的路径；loader位置由续训数值验证，未独立比较payload，未验最终RNG。不是Trainer.train整循环故障、单rank通信故障、梯度累积失败、quantile路由或CUDA/NPU恢复；DOMINO-FAILURE-01整体及全局迁移仍开放。下一动作扩大局部专家单rank异常后的完整Trainer恢复与非持久统计检查。

### DOMINO-TRAINER-LOCAL-FAILURE-01：单rank计算异常后新Trainer恢复

- 测试harness增加LLMTUNER_TEST_FAILED_EXPERT=1：正常load既有step2 checkpoint，在首个真实Domino `_run_experts` 内断言计算协调开启，仅rank0的真实F.linear抛ValueError，其他rank必须收到Domino expert computation阶段错误；实际train_step读取数据、执行router与通信，不替换forward输出或跳过collective。关闭失败Trainer、barrier、新建独立Trainer后正常load并复用严格状态比较及续训。
- 实际命令 `LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_MOE=1 LLMTUNER_TEST_EP_SIZE=2 LLMTUNER_TEST_DISPATCHER=domino LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_FAILED_EXPERT=1 PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29864 tests/integration_tests/pp_checkpoint_equivalence.py resume /tmp/llmtuner-domino-failed-update-20261015`，exit0，日志 `/tmp/domino-trainer-expert-resume.log`。独立连续参考复用DOMINO-TRAINER-FAILED-COMMIT-01的full与checkpoint，未重新生成或改写参考。
- CPU/Gloo FP32、dp_shard4/EP2/efsdp2、TP/PP/CP1、两层qwen3_moe、动态EMA、AC/compile关。四rank均discarded runtime after single-rank expert failure；新runtime更新前模型/AdamW/scheduler/EMA/trainer/token计数/Python与NumPy与CPU RNG比较failures0，后续两步loss差0，最终状态all checks passed。失败runtime没有原地reset重试，没有宣称通信内部异常恢复。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加累计工作区及本批测试修改。受影响脚本pre-commit、diff HEAD --check通过；生产未改，复用MOE-NEW-WINDOW-01全量3570 passed/2 skipped，不记为本轮执行。
- 已验收该配置的单rank局部专家错误→全部rank退出当前step→新runtime恢复成功checkpoint→一致续训；非持久统计由新runtime重建与新step清理，但尚未逐项快照，量化路由/累积中途错误、其他并行轴、compile和CUDA/NPU仍待验。DOMINO-FAILURE-01及全局迁移保持开放，下一动作补非持久统计/loader更新前显式比较，避免仅由loss推断其相等。

### TRAINER-LOADER-SNAPSHOT-01：失败恢复后loader逐键比较

- 测试harness新增LLMTUNER_TEST_LOADER_STATE=1，将真实dataloader.state_dict加入独立split快照，恢复后首次data_iterator之前逐键比较，不以loss等价替代游标证据。默认关闭以兼容旧参考artifact；本批重新生成独立full参考，未向旧快照补造预期。
- full实际命令 `LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_MOE=1 LLMTUNER_TEST_EP_SIZE=2 LLMTUNER_TEST_DISPATCHER=domino LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1 PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29865 tests/integration_tests/pp_checkpoint_equivalence.py full /tmp/llmtuner-domino-loader-20261015`；resume同命令增加LLMTUNER_TEST_FAILED_EXPERT=1并改phase resume。两独立torchrun均exit0，日志 `/tmp/domino-loader-full.log`、`/tmp/domino-loader-resume.log`。
- CPU/Gloo FP32、dp_shard4/EP2/efsdp2、TP/PP/CP1、Domino/custom FSDP、AdamW/动态EMA、AC/compile关。先注入rank0专家计算异常、协调退出并丢弃runtime，再新Trainer load；四rank更新前状态包含synthetic loader的steps/dp_world_size/batch_layout(seed/vocab/batch/seq)、模型/optimizer/scheduler/EMA/trainer及CPU RNG，failures0。续训loss差0、最终既有状态通过；loader最终payload尚未独立比较，真实map/stream/packed loader不由此证明。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加累计工作区及本批测试快照变化；受影响脚本pre-commit及diff HEAD --check exit0。生产未改，复用MOE-NEW-WINDOW-01全量3570 passed/2 skipped，不记为本轮执行。下一动作补非持久router/aux统计在失败恢复后新窗口的显式检查；完整全局迁移及目标设备仍开放。

### TRAINER-EMPTY-SCRATCH-01：恢复后的非持久统计显式断言

- 测试harness新增LLMTUNER_TEST_EMPTY_SCRATCH=1；新Trainer正常load之后、读数据/更新之前，遍历实际MoE/AuxLoss断言tokens_per_expert_E和instance_acc为空，_step_metric/_runtime_denominator为None；quantile实例存在时另验histogram为空。要求moe_count>0且aux_count>0，防止未触达即通过。此场景普通router，quantile分支未触达，不据条件分支声明验证。
- 实际命令沿TRAINER-LOADER-SNAPSHOT-01四进程resume配置，port29866、同workdir `/tmp/llmtuner-domino-loader-20261015`，增加 `LLMTUNER_TEST_FAILED_EXPERT=1 LLMTUNER_TEST_EMPTY_SCRATCH=1`，其余 `LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_MOE=1 LLMTUNER_TEST_EP_SIZE=2 LLMTUNER_TEST_DISPATCHER=domino LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1 PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29866 tests/integration_tests/pp_checkpoint_equivalence.py resume /tmp/llmtuner-domino-loader-20261015`；exit0，日志 `/tmp/domino-scratch-resume.log`。
- CPU/Gloo FP32、EP2/efsdp2/dp_shard4、其他轴1、custom FSDP/Domino、动态EMA、AC/compile关。单rank专家故障后新runtime恢复，四rank各2MoE/2AuxLoss scratch为空，包含loader/RNG的更新前failures0，续训loss差0、最终状态通过；不证明失败runtime统计已自动回滚，明确验证丢弃runtime后重建。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加累计工作区及本批测试断言；pre-commit初次E501两处，拆行后全部适用hook通过，diff HEAD --check通过；生产未改，复用MOE-NEW-WINDOW-01全量3570 passed/2 skipped。下一动作quantile路由真实生产配置和失败恢复；目标设备及全局必需迁移仍开放。

### QUANTILE-DOMINO-RESUME-01：公开量化路由配置与故障恢复

- 测试harness增加LLMTUNER_TEST_QUANTILE=1，公开ModelConfig选择deepseek_v3，n_routed_experts8/top2/shared1、n_group/topk_group1、first_k_dense_replace0、q/kv LoRA16、qk nope/rope8、v16；公开ExpertParallelConfig.moe_quantile_balancing=True，sigmoid模型来源符合生产转换契约，不把Qwen softmax改写成sigmoid假验证。其他hidden64/two layers/vocab128保持harness配置。
- full命令 `LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_MOE=1 LLMTUNER_TEST_QUANTILE=1 LLMTUNER_TEST_EP_SIZE=2 LLMTUNER_TEST_DISPATCHER=domino LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1 PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29867 tests/integration_tests/pp_checkpoint_equivalence.py full /tmp/llmtuner-quantile-domino-20261015`；独立resume同配置增加LLMTUNER_TEST_FAILED_EXPERT=1与LLMTUNER_TEST_EMPTY_SCRATCH=1、phase改resume。均exit0，日志 `/tmp/quantile-domino-full.log`、`/tmp/quantile-domino-resume.log`。
- CPU/Gloo FP32、dp_shard4/EP2/efsdp2、其他轴1、Domino/custom FSDP/动态EMA、AC/compile关。rank0局部专家故障传播后丢弃runtime，新Trainer恢复：每rank真实两quantile router，要求quantile_count==moe_count，token计数/直方图空；该模型aux_count=0，不假称aux指标已验（普通router上一批单独覆盖aux>0）。模型含expert bias、AdamW/scheduler/EMA/trainer/loader/Python与NumPy与CPU RNG更新前failures0，后续loss差0，最终状态通过。连续loss `[4.864121437072754,4.875164985656738,4.863265037536621,4.877156734466553]`。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加累计工作区及本批测试配置/触达断言；受影响脚本pre-commit和diff HEAD --check通过，生产无改动，复用MOE-NEW-WINDOW-01全量3570 passed/2 skipped。未证明量化更新对独立数学参考（已有CPU单测另有证据），不外推设备/compile/累积中途故障或原地通信恢复；全局迁移继续。下一动作推进未覆盖的累积中途故障及实际目标内核缺口。

### QUANTILE-ACCUMULATION-RESUME-01：已有梯度时的局部专家故障恢复

- 验证缺口 / `tests/integration_tests/pp_checkpoint_equivalence.py`。公开 TrainingConfig 接入 LLMTUNER_TEST_ACCUMULATION（默认1）；故障注入新增 LLMTUNER_TEST_FAILED_EXPERT_CALL（默认1）。本批设累积2、首次 routed module 第3次专家调用注入 rank0 F.linear 异常，且所有rank在该调用前断言实际参数梯度非空，确认经过前一个microbatch反向，避免只测试空梯度窗口。
- full实际命令：`LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_MOE=1 LLMTUNER_TEST_QUANTILE=1 LLMTUNER_TEST_EP_SIZE=2 LLMTUNER_TEST_DISPATCHER=domino LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1 LLMTUNER_TEST_ACCUMULATION=2 PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29868 tests/integration_tests/pp_checkpoint_equivalence.py full /tmp/llmtuner-quantile-accumulation-20261015`。full session34852确认exit0；独立resume同命令phase改resume并增加 `LLMTUNER_TEST_FAILED_EXPERT=1 LLMTUNER_TEST_FAILED_EXPERT_CALL=3 LLMTUNER_TEST_EMPTY_SCRATCH=1`，session82294最终exit0。
- CPU/Gloo FP32、dp_shard4/EP2/efsdp2、TP/PP/CP1、custom FSDP/Domino、DeepSeek V3 quantile、动态EMA、AC/compile关。四rank均协调异常、关闭故障runtime、新建Trainer恢复；各2MoE/2quantile的临时统计为空（aux=0），模型含expert bias、optimizer/scheduler/EMA/trainer/loader/RNG更新前failures0。连续loss `[4.870572566986084,4.8695292472839355,4.875195503234863,4.873786449432373]`，恢复后两步最大loss差0，最终状态all checks passed。原始日志 `/tmp/quantile-accumulation-full.log`、`/tmp/quantile-accumulation-resume.log`。
- HEAD `f188f679e2325ded37ae1da46153c4ae6b511682` 加累计工作区及本批测试选项；受影响脚本pre-commit全部适用hook通过、diff HEAD --check通过。生产未改，复用MOE-NEW-WINDOW-01的3570 passed/2 skipped，不记为本轮执行。只证明完整checkpoint重建恢复，不证明失败窗口原地重试、通信故障恢复或CUDA/NPU重叠。DOMINO-FAILURE-01整体仍开放；下一动作转向未覆盖的必需内核、布局与多轴组合缺口，避免重复已通过子路径。

### DUAL-CP-EVIDENCE-02：当前 GQA 矩阵与生产 flex 前向复核

- 证据一致性 / `context_parallel/attention.py::run_flex`、`DualCPFlexKernel` 与双轴集成脚本。矩阵仍写6配置，但现有脚本已包含Q6/KV2（历史“双CP轴 GQA非二次幂本地heads 数值复核”）；本批修正为9配置，保持SDPA替换与真实flex的证据边界。
- 实际命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29869 tests/integration_tests/dual_cp_attention_equivalence.py`，session22882 exit0，日志 `/tmp/dual-cp-nine-configs.log`。CPU/Gloo FP64、KV2×Ulysses2、其余轴1、AC/compile无：三类mask×三种head组合全部输出/QKV梯度在1e-10内匹配独立全序列SDPA；attention算术替换为SDPA，生产collective/autograd未替换。
- 实际命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29870 tests/integration_tests/cp_dual_axis_equivalence.py`，session81297 exit0，日志 `/tmp/dual-cp-production-forward.log`。同四rank、HF Qwen3两层/Q4/KV2、512tokens，普通causal和多文档packed真实CPU eager flex前向与独立模型参考通过rtol/atol1e-9；另包含真实通信反向的SDPA参考及旧错误通信顺序的非等价断言。mask创建解除compile为测试适配，不宣称生产编译已验证。
- 最小直接调用 `flex_attention(q,q,q).sum().backward()`（CPU、q shape1×2×8×4、requires_grad=True）明确返回NotImplementedError：FlexAttention does not support backward on CPU。故生产flex反向仍需CUDA等受支持设备；未改生产后端、未用SDPA替换宣称完成。HEAD f188f679e2325ded37ae1da46153c4ae6b511682 加累计工作区；本批只更新文档，复用MOE-NEW-WINDOW-01全量门禁，目标设备/TP/PP双轴组合继续开放。下一动作推进其他不依赖设备的布局与训练调用链缺口。

### RNG-RESTORE-COORDINATION-01：常规 DCP 的 rank 局部恢复异常传播

- P1 / `components/checkpointer/distributed_checkpoint.py::_load_checkpoint` → `RNGCheckpointState`。DCP读取后RNG原先仅各rank独立load；一个rank元数据错误或本地setter失败时，其他rank可能正常返回进入训练collective。本批在常规DCP末尾新增全rank本地校验与本地恢复两个错误边界；校验统一成功后才开始恢复，局部原异常保留，其他rank收到阶段RuntimeError。边界内部不包含DCP/model collective；模型/optimizer此前可能已加载，不声明整个load原子回滚。
- `tests/integration_tests/rng_checkpoint_equivalence.py`在每个真实manager正常恢复前分别给rank0的validate_state_dict/load_state_dict注入ValueError，所有rank断言原错误或准确阶段消息并继续barrier；校验失败另比较Python/NumPy/torch实际流未变，再正常load及两次随机dropout/AdamW更新，与连续参考逐位比较。DDP/native FSDP/custom relocated多context，各同步/异步，共6配置全部通过。
- 实际命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29871 tests/integration_tests/rng_checkpoint_equivalence.py`，session29450 exit0，日志 `/tmp/rng-restore-coordination-final.log`。CPU/Gloo FP32、两rank独立Python/NumPy/torch seed、AC/compile关闭。首轮session63813 exit1，日志 `/tmp/rng-restore-coordination.log`：测试错误地比较含Tensor的pickle字节，非状态语义；已改实际生成器状态比较并保留失败证据，未弱化随机轨迹断言。
- HEAD f188f679e2325ded37ae1da46153c4ae6b511682 加累计工作区及本批两文件变更。最新全量 `pytest -q --disable-warnings` exit0：3571 passed、2 skipped、260 warnings、35.99s，日志 `/tmp/rng-coordination-units.log`。全部Git已跟踪及适用未跟踪文件pre-commit（排除受保护examples/native_alltoall.py）、compileall与diff HEAD --check均exit0；首次Ruff报告注入闭包未绑定failure，已显式绑定并通过全门禁。
- 已实现并验证常规DCP本地RNG失败传播；不覆盖进程死亡、通信内部失败或设备setter的实际故障。Kimi跨expert布局分支已有前置协调校验，但末尾RNG setter仍需单独协调及迁移场景验证，登记RNG-KIMI-RESTORE-01（P1，路径kimi_layout.py，验收为单rank setter异常传播且正常跨布局恢复保持一致），下一批处理；其他Stateful回调失败不由此自动协调。全局目标保持未完成。

### RNG-KIMI-RESTORE-01：跨专家布局末尾 RNG 恢复协调

- P1 / `components/checkpointer/kimi_layout.py::load_changed_kimi_layout`。前置layout conversion已协调RNG metadata校验，但模型/训练状态应用后末尾setter独立执行，单rank失败仍可使其他rank进入训练通信。新增本地RNG restoration错误边界，所有参与rank进入；不把模型转换、distribute_tensor或state collectives放入边界，不承诺整个加载事务回滚。
- 新 `tests/integration_tests/kimi_rng_restore_coordination.py` 使用真实小型Kimi latent/shared MoE、实际DCP保存读取、ModelCheckpointState与生产layout迁移。两个rank分别保存默认CPU RNG，两个方向individual→grouped/grouped→individual，rank0 setter注入原ValueError，另一rank收到准确阶段RuntimeError；随后barrier和正常重载成功，torch随机流逐位恢复、目标模型输出与源布局在1e-6内一致。没有使用实际EP切片/完整Trainer/设备setter，不能由此外推跨拓扑随机连续性；已有真实Trainer跨EP证据仍明确排除RNG。
- 命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29872 tests/integration_tests/kimi_rng_restore_coordination.py`，session67029 exit0，两个方向均打印RNG failure coordinated；日志 `/tmp/kimi-rng-coordination.log`。HEAD f188f679e2325ded37ae1da46153c4ae6b511682 加累计未提交状态与本批一处生产边界/新集成脚本。
- 本轮完整 `pytest -q --disable-warnings` exit0，3571 passed、2 skipped、262 warnings、35.51s，日志 `/tmp/kimi-rng-coordination-units.log`。全部Git适用文件pre-commit排除examples/native_alltoall.py后exit0、compileall和diff HEAD --check exit0；初次Ruff行宽由format修正，最终门禁通过。生产文件集合无新增，仅新增显式torchrun测试，逐文件清单路径不变。
- 本项本地setter失败传播契约关闭；进程死亡/通信内部错误/设备RNG setter故障及其他Stateful回调异常保持独立缺口。下一动作扩展checkpoint其他本地恢复阶段的错误边界审查，并转向未闭合KCP/MC2布局能力；全局目标继续。

### SCHEDULER-RESTORE-COORDINATION-01：常规 DCP 本地 scheduler 失败传播

- P1 / `components/checkpointer/distributed_checkpoint.py::_load_checkpoint` → `components/optimizer/lr_scheduler.py::LRSchedulersContainer.load_state_dict`。实际scheduler加载仅校验epoch并重建各optimizer的LR，不包含collective；原单rank失败可让其他rank继续进入模型TP/EP状态转换或RNG协调。现所有rank在scheduler加载处参与本地错误边界，无scheduler的rank同样进入；原异常保留，成功rank收到阶段RuntimeError。不声明此前DCP写入可回滚。
- 扩展 `rng_checkpoint_equivalence.py`，注入目标改为显式target/method/phase三元组，新增rank0真实scheduler实例load_state_dict失败；与已有RNG校验/恢复注入共同运行。真实两rank CPU/Gloo FP32，DDP/native FSDP/custom relocated FSDP × sync/async共6配置，局部异常及peer阶段错误断言、后续barrier、正常load与两次随机dropout/AdamW更新逐位一致全部通过。
- 命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29873 tests/integration_tests/rng_checkpoint_equivalence.py`，session33681 exit0，日志 `/tmp/scheduler-load-coordination.log`。HEAD f188f679e2325ded37ae1da46153c4ae6b511682 加累计工作区与本批scheduler边界/测试扩展；AC/compile关闭，未验证实际设备或通信内部故障。
- 全量 `pytest -q --disable-warnings`：3571 passed、2 skipped、265 warnings、35.15s、exit0，日志 `/tmp/scheduler-load-units.log`；全部适用Git文件pre-commit（排除examples/native_alltoall.py）、compileall、diff HEAD --check通过。当前178源码与逐文件表178行核对，无遗漏/失效/重复，仅证明清单一致而非全部运行时验收。
- 本项常规DCP的scheduler本地失败传播关闭。审查确认ModelCheckpointState.load_state_dict内部包含TP/EP转换，不能整体塞入local-only边界；下一动作界定其本地preflight与collective应用阶段，并检查Kimi迁移的scheduler/其他回调。全局必需迁移及硬件验证继续开放。

### MODEL-LOAD-PREFLIGHT-COORDINATION-01：常规 DCP 模型校验先于 TP/EP 转换

- P1 / `distributed_checkpoint.py::_load_checkpoint` → `model_state.py::validate_state_dict/load_state_dict`。后者先本地校验键/形状，再执行TP/EP状态转换；原manager没有跨rank协调前置校验，单rank校验失败可让peer进入后续转换通信。现manager在应用模型前统一执行已有纯本地validate_state_dict，错误协调成功后才进入模型加载。保留直接ModelCheckpointState调用的既有rank-local契约，不将含collective的加载函数放入本地错误边界。
- 在真实manager集成脚本rng_checkpoint_equivalence.py增加rank0模型validate_state_dict注入，按target/method/phase驱动；DDP/native FSDP/custom relocated FSDP × sync/async六配置均验证原异常/peer准确阶段错误、后续barrier及正常重载，两次随机dropout/AdamW更新的loss/参数/optimizer/scheduler逐位一致。CPU/Gloo FP32、两rank、AC/compile关闭；注入验证协调机制，不假称真实损坏文件覆盖（键/形状损坏另有既有单测）。
- 命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29874 tests/integration_tests/rng_checkpoint_equivalence.py`，session9035 exit0，日志 `/tmp/model-load-coordination.log`。另同env、port29875执行 `tests/integration_tests/tp_checkpoint_coordination.py`，session17381 exit0：实际TP gather/load的shape、missing parameter、packed、manifest、sparse、meta共12故障场景协调、健康重试通过，日志 `/tmp/model-tp-checkpoint-coordination.log`；该脚本直接调用TP state函数，非manager×TP整体证明。
- HEAD f188f679e2325ded37ae1da46153c4ae6b511682 加累计工作区与本批前置边界/测试扩展。全量pytest -q --disable-warnings exit0：3571 passed、2 skipped、259 warnings、35.91s，日志 `/tmp/model-load-units.log`。全部适用文件pre-commit（排除examples/native_alltoall.py）、compileall和diff HEAD --check均通过。
- 本项manager已有模型键/形状校验的传播缺口关闭。不覆盖DCP已写入live tensors的回滚、自定义load hook异常、TP/EP通信内部故障。下一动作回到必需迁移的KCP padding state通信及MC2×TP布局适配，避免将已验证checkpoint局部子路径扩展为全局完成；全局继续。

### MC2-EXPERT-HOOK-01：融合入口绕过独立专家 forward 生命周期（开放）

- P1 / `parallel/expert_parallel/mc2_dispatch.py::MC2RoutedExperts.forward` → GroupedExperts参数；消费者为FSDP pattern指定inner_experts的独立包装。审查MC2 TP阻断时确认更基础的入口差异：MC2直接cat w1/w3并调用融合函数、读取w2，完全不调用inner_experts；普通RoutedExperts调用inner_experts.forward。绑定该模块的FSDP前后hook因而不会执行。默认祖先FSDP单元可能已materialize，不能把此风险外推为所有默认MC2配置必失败。
- 已核对上游FSDPTurbo commit `0af4eca242c14ebff254dab1a38045f1395fa520` 的 `distributed/expert_parallel/dispatcher_mc2.py`：其forward直接绑定持有gate_up_proj/down_proj的专家模块；本项目将融合forward放在父RoutedExperts，参数所有者仍为子GroupedExperts，生命周期归属不同，不能仅照搬数学。
- 实际CPU探针复用test_mc2_routed_experts_preserve_separate_weights_and_router_gradients（torch_npu三算子为既有EmulatedOps，明确不是目标算子），给构造的MC2/reference两个GroupedExperts各注册forward_pre_hook，调用真实两个RoutedExperts并完成梯度比较；MC2 hook次数0、reference次数1，exit0。通过MonkeyPatch临时包装GroupedExperts.__init__及runpy加载测试，不修改生产全局。证明入口绕过hook；尚未用真实FSDP+NPU验证报错/梯度后果。
- 实现状态：未修复；验证状态：CPU适配器机制复现。验收：融合计算必须经参数所属模块的正常forward生命周期，保留独立专家FSDP及ancestor relocation语义、参数身份/FQN、融合算子与router/权重梯度；真实native/custom独立专家包装测试及NPU数值仍需补齐。下一动作设计专属MC2专家compute入口或等价模块归属，避免通过手动执行私有FSDP hooks或拒绝组合代替迁移。MC2×TP仍单独开放。
- HEAD f188f679e2325ded37ae1da46153c4ae6b511682 加已有工作区；本批只更新证据文档，生产无改动，复用MODEL-LOAD-PREFLIGHT-COORDINATION-01全量3571/2skip。新证据使MC2生命周期不能记为完成；全局继续。

### MC2-EXPERT-HOOK-02：参数所有者承载融合 forward

- 关联MC2-EXPERT-HOOK-01/P1。新增同文件MC2GroupedExperts继承GroupedExperts，直接继承其参数构造、activation与checkpoint schema，在forward内执行融合dispatch/GEMM/activation/combine；MC2RoutedExperts仅负责排序/count准备、调用inner_experts及加权还原。block_conversion在MC2公开配置分支直接构造该专家类型，不修改已有对象class、不手动调用FSDP私有hooks、不移动参数FQN、不增加兼容壳。参数仍w1_EFD/w3_EFD/w2_EDF，constructor要求匹配专家类型，错误输入明确拒绝。
- 同步消费者：CPU adapter测试与NPU ep_mc2_equivalence.py构造MC2GroupedExperts；普通参考仍GroupedExperts。单测显式断言专家pre/post hook顺序、Parameter身份及原checkpoint键，保留独立参考输出/输入/router/三组权重梯度比较；定向9 passed、exit0。NPU脚本仅更新入口，本机未运行，不声明真实融合验证。
- 新mc2_fsdp_lifecycle.py：两rank CPU/Gloo FP64，真实native FSDP、custom FSDP、custom ancestor hook relocation共3配置，独立专家单元和root包装；融合三算子用既有EmulatedOps（通信world_size1，明确非EP/NPU通信），FSDP真实all-gather/反向归约。输出、输入梯度、full expert权重梯度与独立GroupedExperts逐项1e-12匹配，全部exit0。命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29876 tests/integration_tests/mc2_fsdp_lifecycle.py`；最终session47737、日志 `/tmp/mc2-fsdp-lifecycle-final.log`。初稿临时patch Sequential.forward通过但有Ruff闭包警告，已改显式ExpertOwner模块并复跑最终稿，无生产全局patch。
- HEAD f188f679e2325ded37ae1da46153c4ae6b511682 加累计工作区及本批2生产文件/单测/集成消费者/新脚本。全量pytest -q --disable-warnings：3571 passed、2 skipped、263 warnings、36.17s、exit0，日志 `/tmp/mc2-hook-units.log`；全部适用Git文件pre-commit排除examples/native_alltoall.py后通过，compileall及diff HEAD --check通过。生产源码未新增文件，表路径仍一致。
- 实现已修复hook绕过，CPU真实FSDP生命周期数值通过；MC2-EXPERT-HOOK-01的NPU融合×FSDP目标验收仍待运行，未关闭全部设备条件。TP布局适配、EP×FSDP真实融合、重计算/compile及性能仍独立开放；下一动作继续MC2×TP参数布局和复制梯度契约，不以CPU算子替身作为迁移全部完成。

### MC2-DISTRIBUTED-ADAPTER-01：真实跨 rank 传输的融合适配器排列参考

- 验证缺口 / mc2_ops.py的两融合autograd逆变换及mc2_dispatch.py的send/recv计数。此前EmulatedOps仅world_size1，不能证明多rank source-major/expert-major排列。新增tests/integration_tests/mc2_distributed_adapter.py，用真实Gloo all_to_all_single模拟融合算子通信，明确实现rank→expert排列及逆排列，继续调用生产AllToAllGroupedMatmul/GroupedMatmulAllToAll及MC2RoutedExperts。测试替身不作为NPU算子实现或性能证据。
- 两rank CPU/Gloo FP16，E4/top2、D4/F3，各rank两专家；不均匀tokens2/3、空输入rank0、零接收rank1三配置。独立全量GroupedExperts/LocalTokenDispatcher参考比较输出、输入/路由score梯度及本rank专家权重梯度，atol5e-4/rtol1e-2（沿既有NPU脚本FP16契约），全部通过。命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29877 tests/integration_tests/mc2_distributed_adapter.py`，最终session13070 exit0，日志 `/tmp/mc2-distributed-adapter-final.log`。初稿相同数值通过，但Ruff闭包绑定/行宽警告已修正，再执行最终稿通过。
- TP调用链复核：tensor_parallel.py的moe_deferred_to_ep在ep_size>1时不切F维/不安装TPMoeSequenceBoundary，原HF专家完整交给EP，MoE直接处理T/tp序列片段；block_conversion.py的MC2×TP拒绝理由“需要fused projection layout adapter”不能由该实际路径证明。现有test_apply_tp_defers_the_moe_blocks_to_ep_when_ep_is_on本轮定向1 passed/36 deselected，明确参数shape及identity内容保持。没有据此立即删除限制：公开TP×EP×MC2配置转换、dense TP边界、复制梯度及完整模型数值仍需连通验收；本批适配器测试不含TP投影。
- HEAD f188f679e2325ded37ae1da46153c4ae6b511682 加已有工作区及本批新测试；无生产变化。新脚本pre-commit与diff HEAD --check通过，复用MC2-EXPERT-HOOK-02最新完整3571 passed/2 skipped门禁，不记本轮全量。源码清单无新增生产文件。下一动作以公开配置串接TP dense边界与MC2 EP，验证实际布局后调整过时guard；NPU融合/性能与所有其他迁移缺口保持开放。

### MC2-TP-PUBLIC-01：公开 TP2/EP2 候选接通，梯度差异待定位（开放）

- P1 / block_conversion.py的MC2×TP guard、TP deferral及MC2融合反向。依据前批确认的EP拥有完整F布局，当前工作区候选删除该过时guard并清理无用UnsupportedCombinationError导入；这是未验收候选，不能宣称MC2×TP支持完成。新tests/integration_tests/mc2_tp_ep_equivalence.py经公开ParallelConfig/ExpertParallelConfig、apply_tp→apply_ep构建真实两层Qwen3MoE，TP2/EP2共享两rank、CPU/Gloo FP16，其他轴1，AC/compile/FSDP无；MC2融合算子为DistributedOps替身，真实all-to-all与dense TP通信。
- 非空断言：TP realizer存在、TP后HF专家完整(8,96,64)，EP后专家(4,48,64)；对照公开alltoall dispatcher模型，并实际调用Trainer._allreduce_replicated_tp_grads。前向两rank逐位相同(max diff0)，但原atol5e-4/rtol1e-2下全参数梯度比较失败：rank0五键、rank1七键，首键embedding max abs0.015625；rank0其他失败为首层q/k/v/o projection。未放宽容差、未挑seed、未按loss缩放规避差异；此参考隔离dispatcher差异，不单独证明全部TP数学正确。
- 命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29878 tests/integration_tests/mc2_tp_ep_equivalence.py`。初次session78877因非空断言只识别ColumnParallelLinear失败（真实realizer还包括RowParallelLinear/ColwiseLinearNoGather），修正为既有验证集合；第二次session81882真实梯度失败；诊断版session16838收集所有失败后仍assert，exit1。日志 `/tmp/mc2-tp-ep.log`、`/tmp/mc2-tp-ep-final.log`、`/tmp/mc2-tp-ep-diagnostics.log`均保留。
- HEAD f188f679e2325ded37ae1da46153c4ae6b511682 加累计工作区及本批候选/新脚本。完整CPU pytest exit0：3571 passed、2 skipped、264 warnings、36.07s，日志 `/tmp/mc2-tp-candidate-units.log`；这不覆盖显式torchrun失败，不能作为候选验收。受影响文件pre-commit及diff检查通过，最终诊断扩展另执行格式/引用检查。
- 实现状态：候选接通、未验收；验证状态：CPU适配器全模型梯度失败、NPU未运行。下一动作定位首个专家输入cotangent差异，分离FP16分支累加顺序与融合反向公式/传输错误，保持原失败断言；若需数值契约调整须独立参考证明，不能凭单测绿关闭。候选未commit/push，完成前不声称生产支持；全局继续。

### MC2-TP-ROUNDING-01：实际模型 gate/up 反向分歧定位（验收仍开放）

- 关联MC2-TP-PUBLIC-01。测试新增可选LLMTUNER_MC2_BOUNDARY_DIAGNOSTICS=1，记录两模型各层RoutedExperts输入/输出及其梯度，返回值不改变；GradientDiagnosticOps在真实融合反向操作数上计算分支FP16求和与FP64参考。默认断言/容差/seed保持，生产本批无修改。
- 两rank最后一层输入/输出/输出梯度逐位相同，但输入梯度diff rank0=1.52587890625e-5、rank1=7.62939453125e-6；第一层输出梯度已受后层差异影响。实际四次gate/up反向的fused_vs_split均7.62939453125e-6。FP64 oracle下融合结果max error分别3.7154e-6、3.1552e-6、3.7075e-6、3.7549e-6，分支结果分别4.8798e-6、5.0582e-6、6.6255e-6、6.9072e-6。部分融合输出与oracle.half并非逐位相同，不能声称融合内核总是一次精确舍入；此数据仅证明已捕获算术差异与误差范围。
- 因果对照：另加LLMTUNER_MC2_SPLIT_GRAD_DIAGNOSTIC=1，仅在测试替身内将gate/up反向输入梯度改为两个FP16乘积相加，其他前后向/通信/参数梯度代码不变。两rank两层MoE输入梯度diff全部0，完整参数梯度原atol5e-4/rtol1e-2断言通过。这个干预关闭了所测模型差异来源定位，不是生产修复；没有把CPU替身分支求和带入真实融合算子或更改生产全局开关。
- 实际命令：`LLMTUNER_MC2_BOUNDARY_DIAGNOSTICS=1 PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29879 tests/integration_tests/mc2_tp_ep_equivalence.py`，session92580 exit1（原梯度验收仍失败），日志 `/tmp/mc2-tp-real-operands.log`；相同命令加LLMTUNER_MC2_SPLIT_GRAD_DIAGNOSTIC=1、port29880，session72491 exit0，日志 `/tmp/mc2-tp-split-diagnostic.log`。先前只捕获边界session42567 exit1，日志 `/tmp/mc2-tp-boundaries.log`；独立seed7的8×48/48×64 FP16矩阵探针也复现fused/split maxdiff0.015625，但不以该小探针代替实际模型证据。
- HEAD f188f679e2325ded37ae1da46153c4ae6b511682 加累计工作区及本批测试诊断；新脚本pre-commit与diff检查通过，复用MC2-TP-PUBLIC-01完整CPU3571 passed/2 skipped，本轮不重复全量。最终通过输出明确标识split-gradient diagnostic，不能与默认融合验收混淆。
- 下一动作：基于独立高精度完整模型参考界定FP16融合误差，而非单方面要求两种舍入图完全相同或放宽原断言；补目标NPU内核参考和公开配置恢复/更新。MC2×TP候选未验收、原失败保留，全局必需迁移仍开放。

### MC2-TP-FP64-REFERENCE-01：独立未并行 HF 完整模型误差测量

- 关联MC2-TP-PUBLIC-01（未关闭）。新测试选项LLMTUNER_MC2_FP64_REFERENCE=1用_model().half().double()重建与被测模型完全相同的FP16量化初始权重，执行原始HF全序列forward/backward；无TP/EP替换，无融合autograd。loss仍与被测路径相同的output.float().square().sum()，所以这是FP64模型计算、FP32 loss表达式参考，不能称全链路每次操作FP64。比较完整复制参数embedding、各层norm、final norm和lm_head；sharded projection和转换专家键尚无独立映射，不由此宣称全梯度参考已完成。
- 命令 `LLMTUNER_MC2_FP64_REFERENCE=1 PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29881 tests/integration_tests/mc2_tp_ep_equivalence.py`，session34266 exit1，日志 `/tmp/mc2-tp-fp64-reference.log`。诊断执行完毕后原参数比较仍失败；非启动或测试通过，明确保留原验收。
- 两rank普通与融合output max error一致，rank0=4.938267636660498e-4、rank1=4.046748734332084e-4。归约后复制参数各rank一致：embedding普通max error0.025959305465221405/relative norm0.001271631052016478，融合0.02615823596715927/0.001248997540462914；首层input norm普通relative0.0013996789962615354、融合0.0013356528330254777；末层post norm普通0.00174521657260389、融合0.0017585632161642884。final norm与lm_head两路径误差完全相同，relative分别0.0007007011523285324/0.0009130258490462665。结果不支持融合系统性更差，也不足以单独定义安全容差。
- CPU/Gloo FP16被测、两rankTP2/EP2，其他轴1、无FSDP/AC/compile，融合为CPU替身。HEAD f188f679e2325ded37ae1da46153c4ae6b511682 加累计工作区及本批诊断；脚本pre-commit/diff通过，生产无改动，复用MC2-TP-PUBLIC-01完整3571/2skip。未调整容差、seed、loss尺度或默认断言。
- 下一动作建立投影/专家FQN与本地shard的独立HF梯度映射，并核实router选择是否跨精度改变；取得全参数/更新误差后再决定数值契约，保留NPU内核与矩阵缺口。不能把本轮平均误差测量作为全局或MC2×TP验收完成。

### MC2-TP-HF-GRADIENT-MAP-01：全部参数独立映射与路由一致性

- 关联MC2-TP-PUBLIC-01/FP64-REFERENCE-01。测试新增固定Qwen3 fixture的显式hf_parameter_view：q/k/v投影按输出维TP切片、o投影按输入维切片；w1/w3映射HF gate_up两半再按EP专家维切片，w2映射down专家维，router.gate映射HF gate；其他参数直接同名。每个参数先断言与独立HF初始权重逐位相同，再比较形状及梯度；不再跳过未匹配键，最终强制compared等于两个模型全部参数数目。
- 每rank完成54视图（每模型27参数），全量映射权重身份内容及梯度shape检查通过。给两个生产router及原始HF gate安装仅记录的forward hooks；所有rank两层topk选择与全序列FP64 HF对应片段逐位一致，排除本配置跨精度selection变化。没有把router计算改为FP64或强制固定路由。
- 实际命令 `LLMTUNER_MC2_FP64_REFERENCE=1 PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29882 tests/integration_tests/mc2_tp_ep_equivalence.py`，session76991 exit1，日志 `/tmp/mc2-tp-routes-reference.log`。原先不带路由捕获的全参数映射session81357同样完成54视图后exit1，日志 `/tmp/mc2-tp-all-gradient-reference.log`。退出失败来自保留的普通FP16对MC2 FP16逐元素断言（rank0五键/rank1七键），不是映射或路由断言；不记为验收通过。
- 独立HF梯度relative norm error最大：普通alltoall0.0021439104082561626，MC2 0.002062427889309677，均为第二层q_norm.weight；下一大项第二层k_norm，普通0.0021198246858727247、融合0.0019161654733686059。包含全部TP/EP shard后仍未见融合整体误差显著恶化，但一个seed/模型结果不足以普遍证明数值界限；没有修改atol/rtol、默认loss或断言。
- 本批测试/文档修改，HEAD f188f679e2325ded37ae1da46153c4ae6b511682 加累计工作区；受影响pre-commit/diff通过，复用MC2-TP-PUBLIC-01全量3571/2skip。FP64 oracle仍执行FP32 loss表达式，CPU替身不证明NPU内核。下一动作保留默认失败及目标设备验收，转向其他独立迁移缺口；该局部误差来源已有边界/实际操作数/干预/完整HF参考证据，避免继续堆叠同一seed实验替代实际设备与更新验证。

### KCP-SHARD-LENGTH-01：状态 context 构造前统一校验长度

- P1 / `parallel/context_parallel/kcp.py::kcp_kda_forward`。context全局cu_seqlens按local_sequence×group.size构造，原先各rank独立计算，无等长检查；不等长输入会构造不同全局边界并进入目标kernel通信。现本地形状/能力错误协调之后，在组内用int64[local_sequence,-local_sequence]一次MIN归约得到min/max，所有rank一致拒绝不等长，再加载可选kernel/runtime。collective保持在local-only错误边界之外；复用get_world_size/get_comm_device，不自行推断HCCL/NCCL设备。
- `kimi_mla_cp_equivalence.py`新增真实KimiDeltaAttention输入rank0长度3/rank1长度4、无padding，双方明确ValueError含equal local sequence lengths，后续barrier成功；本机fla/triton runtime缺失，成功得到该ValueError也证明在可选加载前拒绝。既有8组MLA独立数值/梯度、rank局部非法mask、KDA padding错误协调保持通过。
- 命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29883 tests/integration_tests/kimi_mla_cp_equivalence.py`，session19500 exit0，日志 `/tmp/kcp-unequal-shards.log`。CPU/Gloo FP32、CP2、其他轴1，AC/compile无。仅验证前置元数据和MLA已有数学，不执行KDA目标kernel或实际state exchange。
- HEAD f188f679e2325ded37ae1da46153c4ae6b511682 加累计工作区与本批KCP检查/测试扩展。全量pytest -q --disable-warnings：3571 passed、2 skipped、267 warnings、35.49s、exit0，日志 `/tmp/kcp-unequal-units.log`；全部适用Git文件pre-commit排除examples/native_alltoall.py后通过、compileall/diff HEAD --check通过。无新增生产文件，源码表路径不变。
- 此项关闭等长context前置检查缺口，不将拒绝不等长/padding当作KCP完整迁移。新增每次forward的小型归约及host读取有成本，目标设备性能未测；不能缓存一次后忽略可变序列。KCP-PADDING-STATE-01正确compact/state通信和设备内核仍开放；下一动作确认目标runtime对非均匀有效token/context分区的实际契约，再实现padding，或推进不依赖该外部runtime的独立缺口。

### KCP-STATE-GEOMETRY-01：状态通信几何前置一致性

- P1 / context_parallel/kcp.py。context只采用Q卷积核大小，原先没有本地Q/K/V核大小一致性或跨rank头数/维度一致性检查。新增本地matching Q/K/V校验；沿KCP-SHARD-LENGTH-01一次MIN归约扩展metadata为sequence/kernel/heads/key_dim/value_dim及其负值，统一验证min/max，不增加collective次数。通信留在local-only错误边界之外。
- 新单测覆盖本地不一致卷积，真实两rankKimiDeltaAttention用各自一致Q/K/V但rank间kernel3/4，双方在runtime加载前报matching geometry错误并成功barrier。命令PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29884 tests/integration_tests/kimi_mla_cp_equivalence.py，最终session5646 exit0，日志/tmp/kcp-geometry-final.log；既有8组MLA和padding/长度错误传播通过。CPU/Gloo FP32、CP2、其他轴1，未执行KDA设备kernel。
- 初次完整CPU5失败（/tmp/kcp-geometry-units.log）：新增检查抢在原参数错误之前访问精简fixture缺少属性。已把新检查放在原能力校验之后保留错误优先级；第二次1失败（/tmp/kcp-geometry-units-final.log）为新增fixture缺少head_k_dim/gate_lower_bound，补齐后定向39 passed。最终完整pytest session39621 exit1：2 failed、3572 passed、2 skipped、265 warnings、43.19s（/tmp/kcp-geometry-units-complete.log）；新增test_legacy_cross_layout_checkpoint_without_rng_restores_model两个方向失败于缺少rng_state，与本批KCP无关但累计门禁未通过。当前工作区测试集合较先前增加2项，按外部状态保留，不覆盖；下一动作检查Kimi旧checkpoint无RNG兼容策略。不以失败轮次计完成。
- HEAD f188f679e2325ded37ae1da46153c4ae6b511682加累计工作区；适用全仓pre-commit及最终受影响文件pre-commit、compileall/diff通过。上游modules/kcp.py只透传全局cu_seqlens/group/kernel，无rank局部有效长度公开参数；本机runtime缺失，尚不能证明padding compact后的分区可直接接入。KCP-PADDING-STATE-01及真实设备状态通信继续开放；不以新增拒绝代替迁移。

### KIMI-LEGACY-RNG-01：旧格式兼容与当前 rank 缺失的边界回归

- 路径：`components/checkpointer/kimi_layout.py`、`random_state.py`、`load_planner.py` 与 `tests/unit_tests/cpu/models/test_kimi_dcp_layout.py`。当前 HEAD `0b812f4fc3f95d4e4eb735d55abe4486a867c2ad` 已包含旧 Kimi checkpoint 缺少整个 RNG 类别时的兼容修复；本批核验该实现，不将已有生产改动计为本批修复。前批 KCP-STATE-GEOMETRY-01 记录的两项 legacy RNG 失败在当前代码下已消除。
- 新增双向跨布局回归：individual → grouped、grouped → individual，真实 DCP 保存仅含 rank_1 的 RNG 类别，rank_0 加载必须抛出 KeyError；逐项确认目标模型权重和 CPU RNG 均未改变。整个类别缺失可兼容与类别存在但缺少当前 rank 的损坏状态保持区分；未放宽验证或数值容差。
- 本轮定向 `pytest -q tests/unit_tests/cpu/models/test_kimi_dcp_layout.py`：14 passed、28 warnings，3.80s、exit0。完整 `pytest -q --disable-warnings`：3576 passed、2 skipped、269 warnings，41.16s；完整日志 `/tmp/kimi-legacy-current-units.log`。已有会话日志已收尾核对，无需重复启动。代码状态为上述 HEAD 加既有文档工作区及本批两个回归用例。
- 本批全部适用 Git 文件 pre-commit（排除 `examples/native_alltoall.py`）、compileall 与 `git diff HEAD --check` 已通过；本次台账补记另检查格式与 diff。未新增生产源码文件。当前审查确认 RNG 的空字典 no-op 有明确既有单测契约，未把该兼容行为当作缺陷修改。
- 范围：CPU 单进程真实 DCP 的跨布局恢复失败边界；不外推设备 RNG、分布式 rank 局部错误或 compile 跨布局验收。MC2-TP-PUBLIC-01 的原始 FP16 梯度失败、KCP padding/state exchange 与实际 CUDA/NPU 组合验证仍开放，全局审查未完成。下一项检查 Kimi 跨布局迁移与普通 planner 的 compiled FQN 兼容边界，先确认生产保存格式再决定是否需要修改。

### KIMI-COMPILED-LAYOUT-01：跨专家布局恢复旧 compiled 参数名

- P1 / `components/checkpointer/kimi_layout.py::load_changed_kimi_layout`。普通 `CheckpointLoadPlanner` 已支持旧 `_orig_mod` 名称，跨布局分支却根据原 planner 路径选择模型并转换，带前缀的模型键被遗漏或无法匹配目标。真实 `torch.compile(source, backend="eager").state_dict()` 经 DCP 保存复现两个方向恢复失败：原实现专项 2 failed、2 passed（`/tmp/kimi-compiled-layout-before.log`，exit1）。当前生产 ModelCheckpointState 已解包顶层 OptimizedModule，per-block 使用 Module.compile 保持名称；本项明确兼容历史命名，不声称当前保存仍有同样问题。
- 修复：复用 `utils/module_names.py::canonical_fqn`，按 `_orig_mod` 完整路径段识别源布局并重建模型、optimizer、EMA 的稳定 planner 路径；DCP 读取保留原 storage FQN。规范化后两键落到同一目标时，在源数据读取和模型复制前明确拒绝。非模型训练类别及用户排除的 optimizer/EMA 保持原选择语义；未新增包装类或公开名称。
- CPU 真实 DCP 回归：原有双向权重、AMSGrad optimizer、EMA 和下一步真实更新用例增加 compiled 源维度，并覆盖旧 optimizer/EMA FQN；另新增 model/optimizer/EMA 三种冲突用例，逐项检查目标权重未改变。最终 `pytest -q tests/unit_tests/cpu/models/test_kimi_dcp_layout.py --disable-warnings`：19 passed、35 warnings、3.71s、exit0（`/tmp/kimi-compiled-layout-final.log`）。保持原 atol/rtol。torch.compile 仅用于生成旧包装名称，不作为 Inductor 执行验证。
- `kimi_rng_restore_coordination.py` 增加源命名维度：两 rank CPU/Gloo、两种源布局 × 普通/旧 compiled 名称共4配置；真实 DCP，rank0 RNG setter 故障注入，双方完成错误传播和 barrier，随后再次恢复并逐位比较 rank-local 后续随机数、按1e-6比较模型输出，全部exit0。命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29885 tests/integration_tests/kimi_rng_restore_coordination.py`；日志 `/tmp/kimi-compiled-rng-coordination.log`。最终只调整行宽格式，未改变测试逻辑。不含真实EP分片、FSDP或目标设备。
- 当前 HEAD `0b812f4fc3f95d4e4eb735d55abe4486a867c2ad` 加既有工作区及本批生产/测试修改。完整 `pytest -q --disable-warnings`：3581 passed、2 skipped、285 warnings、45.51s、exit0（`/tmp/kimi-compiled-all-units.log`）。全部适用 Git 文件 pre-commit 排除受保护示例后通过；后增加的集成脚本初次行宽检查失败，自动格式化后受影响 pre-commit 全通过。compileall、diff 检查通过。当前逐文件表与源文件集合核对无遗漏、重复或失效路径，未新增生产源码。
- 本项旧名跨布局兼容已修复并通过上述CPU验收；实际编译执行、TP/EP/FSDP设备组合不由此关闭。MC2-TP-PUBLIC-01原逐元素梯度失败、KCP-PADDING-STATE-01、Domino设备重叠及目标内核矩阵仍开放；全局目标继续。下一动作检查 checkpoint 状态类别识别与名称转换的其他分支，并按迁移矩阵推进独立未验项。

### FIXED-ROUTER-EP-01：公开配置、独立专家算式与更新验证

- 验证缺口 / `config/parallel_config.py::ExpertParallelConfig.fixed_router` → `apply_ep` → `swap_hf_moe_blocks` → `TokenChoiceTopKRouter` → alltoall/Domino。本批阅读配置、转换、routing/layer、AuxLoss及dispatcher调用链，生产未修改。新增 `tests/integration_tests/fixed_router_ep_equivalence.py`，公开配置启用 fixed_router/EP2/aux coefficient0.03；真实离线Qwen3MoE转换，断言flag及本rank专家shape(4,48,64)，避免只验证手工router对象。
- 独立参考直接使用原HF权重、固定 `(token*K+slot)%E` ID、FP32 softmax/gather/normalize与逐专家linear/SiLU/linear/index_add，不使用生产router、dispatcher、GroupedExperts或aux注入。参考显式计算padding过滤的f/p辅助loss并加入目标；生产仍经AuxLossInjection backward。每rank独立token流，专家参考梯度真实all-reduce后按EP rank切片。
- 两rank CPU/Gloo，EP2/E8/top2，其他轴1，无FSDP/compile；alltoall/Domino × full AC开关 × unequal(5/7)、空source(0/3)、稀疏(1/2)、全padding(3/5)共16场景。FP64专家、FP32 routing；输出/专家梯度/一步真实SGD更新atol=rtol=1e-10，输入及gate梯度/更新atol1e-7、rtol1e-6。全参数初始内容逐位检查；有效token统计逐位匹配，aux metric按不含coeff的契约匹配，AC用router调用次数确认真实replay且计数/metric不重复。全部通过。
- 最终命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29889 tests/integration_tests/fixed_router_ep_equivalence.py`，exit0，日志 `/tmp/fixed-router-ep-updates.log`。前序测试参考错误保留：`/tmp/fixed-router-ep-numerics.log`误把coeff加入metric；`/tmp/fixed-router-ep-numerics-final.log`对非连续gate/up梯度view直接Gloo归约，导致错误参考。修正为原有metric契约和contiguous clone后原数值容差不变，16场景通过（`/tmp/fixed-router-ep-numerics-contiguous.log`），再增加SGD验收得到最终结果。未据测试fixture错误修改生产或放宽断言。
- HEAD `0b812f4fc3f95d4e4eb735d55abe4486a867c2ad` 加既有工作区及本批新脚本/文档。脚本pre-commit/diff检查通过。复用 KIMI-COMPILED-LAYOUT-01 的完整3581 passed/2 skipped门禁，本批仅新显式torchrun脚本，不计为再次执行pytest；无新生产文件。生产源码清单不变。
- 验证范围为debug round-robin/drop-free语义；不宣称capacity dropping、TorchAO padding、MC2融合、TP/PP/CP/FSDP组合或CUDA/NPU重叠通过，router gate复制梯度的Trainer归约也不在该block级测试内。设备矩阵与其他迁移开放项继续；下一动作扩展公开Trainer fixed routing组合及设备验证，或推进其他可运行独立缺口。

### FIXED-ROUTER-UPSTREAM-01：修正 FSDPTurbo 迁移的分配及权重语义

- P1 / `models/common/moe/routing.py`、`parallel/expert_parallel/model_replacement.py`、`config/parallel_config.py`。对照 FSDPTurbo commit `0af4eca242c14ebff254dab1a38045f1395fa520` 的 `distributed/expert_parallel/utils.py::fixed_router_for_debug`，上游输出 `(token + slot) % E` 和 detached `1/K`。此前公开 fixed_router 实际启用 torchtitan `_debug_force_load_balance`：`(token*K+slot)%E` 且gather真实score，再应用route_norm/route_scale。两者语义不同；FIXED-ROUTER-EP-01 只能证明旧实现内部自洽，其迁移等价结论不足，由本项修正。
- 实现：router新增明确的fixed_router构造属性，统一执行上游固定ID/权重，Kimi与通用HF转换均连接该属性；torchtitan `_debug_force_load_balance` 保持原行为。公开配置字段名称不变，但fixed_router=True的调试行为有意更正；普通学习路由、checkpoint键及参数布局不变。gate仍计算供本项目显式auxiliary loss使用，固定路由本身不向gate传播梯度；上游helper未包含aux loss，本项目仍遵循配置的注入策略。无容量丢弃，不将该debug开关解释为容量控制。
- 单测4项覆盖route_norm开关×route_scale0/3，断言上游ID、精确1/K、detached score和top-k计数；布尔校验增加3项非法fixed_router输入。保持原torchtitan已有测试。此前已运行86项router测试通过，此后新增最终7项由完整套件覆盖。
- 独立参考脚本 `fixed_router_ep_equivalence.py` 同步改为上游公式及固定权重，其他loss/梯度/SGD/AC断言与容差保持。两rank CPU/Gloo EP2、alltoall/Domino × full AC开关 × unequal/empty/sparse/all_padding共16场景全部exit0。命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29890 tests/integration_tests/fixed_router_ep_equivalence.py`，日志 `/tmp/fixed-router-upstream-ep.log`。这是修正后机制的证据，旧FIXED-ROUTER-EP-01日志不能代替该结果。
- HEAD `0b812f4fc3f95d4e4eb735d55abe4486a867c2ad` 加既有工作区与本批改动。完整 `pytest -q --disable-warnings`：3588 passed、2 skipped、272 warnings、42.08s、exit0（`/tmp/fixed-router-upstream-all-units.log`）。全仓适用Git文件pre-commit初次仅新增参数化行宽失败，自动格式化后重跑；compileall/diff通过。更新并行对比及上游映射，明确两种调试策略区别。未新增生产源码，逐文件表路径不变。
- 当前关闭公开fixed_router分配/权重偏离来源的问题；不关闭MC2真实NPU、CUDA重叠、TP/PP/CP/FSDP组合及容量/padded dispatcher验证。下一动作核对fixed_router的EP1可达性和Trainer复制梯度/冻结gate处理，继续其他必需迁移缺口；全局目标未完成，无commit/push。

### FIXED-ROUTER-ENTRY-01：直接转换入口策略校验与无 gate 梯度路径

- P1 / `parallel/expert_parallel/model_replacement.py::swap_hf_moe_blocks`。公开转换API只校验dispatcher/aux系数，直接传fixed_router=True、quantile_balancing=True绕过ExpertParallelConfig的冲突检查；fixed选择绕过quantile观察，但后续quantile更新仍被安装。新增真实DeepSeek模型回归，原实现不抛错并替换两个block：`/tmp/fixed-quantile-before.log`，1 failed/62 deselected、exit1。现入口直接构造已有ExpertParallelConfig复用策略校验，不重复实现组合规则；在替换前拒绝冲突。推导自HF的aux系数仍由原后续校验负责。专项63 passed、9.69s、exit0（`/tmp/fixed-quantile-after.log`）。
- 无aux行为验证：扩展 `fixed_router_ep_equivalence.py` 至系数0/0.03，累计32场景；系数0时独立参考不注入aux、不为gate制造零梯度，双方gate.grad严格为None，SGD之后gate权重不变，专家输出/梯度/更新保持原容差。命令两rankCPU/Gloo、端口29891，其余沿FIXED-ROUTER-UPSTREAM-01；最终exit0（`/tmp/fixed-router-noaux-ep.log`）。含alltoall/Domino、full AC开关、四种token/padding场景。
- `ep_fsdp_equivalence.py` 新增 --fixed-router，公开parallelize入口，独立HF权重经explicit_reference计算原始专家算式；aux0、四rank CPU/Gloo FP32、DP shard4/EP2/efsdp2、其余轴1、无AC/compile。分别native/custom真实FSDP，断言router无梯度、专家mesh2/router mesh4、专家梯度及全局loss与独立参考在既有1e-4内一致。命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29892 tests/integration_tests/ep_fsdp_equivalence.py --fixed-router`；custom追加 `--fsdp-implementation custom` 并用29893。两次exit0，loss311.808624对311.808617、failed ranks0，日志 `/tmp/fixed-router-native-fsdp.log`、`/tmp/fixed-router-custom-fsdp.log`。该脚本检查一次backward，不称多步/恢复验收；原32场景承担单步SGD证据。
- HEAD `0b812f4fc3f95d4e4eb735d55abe4486a867c2ad`。工作期间外部暂存了前批4个生产文件，已核对并保留；本批model_replacement校验在该暂存版本之上仍为未暂存改动，未擅自调整index。完整pytest：3589 passed、2 skipped、280 warnings、43.80s、exit0（`/tmp/fixed-router-entry-all-units.log`）。适用全仓pre-commit通过；随后新增FSDP测试脚本受影响pre-commit通过，初次日志行宽失败已修正。compileall/diff检查通过，无新增生产源码。
- 独立开放 FIXED-ROUTER-EP1-01：ParallelConfig允许EP1/fixed_router，apply_ep的EP1早退会静默忽略。验收应覆盖公开配置至实际固定选择，兼顾TP先切HF expert的现有布局；不能简单在TP之后局部swap并宣称TP1/EP1所有组合正确。下一动作审查该TP边界并实现单rank可达路径或一致的明确配置契约。Trainer对None梯度直接跳过的代码已阅读，但本批TP=1不执行其TP归约，不能称TP×fixed已验收。CUDA/NPU、MC2与其他迁移开放项继续。

### FIXED-ROUTER-EP1-01：本地固定路由可达，TP 特征分片适配仍开放

- P1 / `expert_parallel.py::apply_ep`、`parallel_config.py` 与 `tensor_parallel.py` 调用链。原EP1直接返回导致fixed_router=True静默无效。新增真实Qwen3MoE公开parallelize入口回归：原实现block身份未改变，1 failed/63 deselected、exit1（`/tmp/fixed-router-ep1-before.log`）。修复后EP1/TP1且fixed_router时执行已有本地MoE转换，显式ep_group=None，其他EP1模型继续原no-op；无新增路由实现或重排阶段。
- TP边界阅读：EP1的HF专家已feature shard，并登记TPMoeSequenceBoundary、sharded参数名称和checkpoint布局，不能在此后无条件swap而丢失这些契约。新增配置检查对fixed_router+EP1+TP>1明确报尚未适配，并保留EP>1/TP>1可构造。此拒绝只消除静默配置错误，**不关闭TP1以外的固定路由实现缺口**；FIXED-ROUTER-EP1-TP-01继续开放，验收需真实边界/梯度归约/checkpoint状态全部保留。
- 定向模型转换和配置套件349 passed、12.63s、exit0（`/tmp/fixed-router-ep1-targeted.log`）。本地模型初始专家shape(8,48,64)及公开flag断言，输出与HF原权重独立固定公式在1e-10匹配，确认非配置对象空验证。
- 两rank CPU/Gloo脚本扩展为EP1 local及EP2 alltoall/Domino，共48场景（各路径×AC开关×4种token/padding×aux0/0.03）。EP1故意向apply_ep传WORLD，仍要求8个本地专家、独立本rank梯度与更新；EP2继续检查4专家slice和参考梯度all-reduce。输出、输入/gate/专家梯度、SGD、计数及aux metric沿既定容差全部通过。命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29894 tests/integration_tests/fixed_router_ep_equivalence.py`，exit0，48条passed（`/tmp/fixed-router-local-and-ep.log`）。无FSDP/PP/CP或设备内核，不外推相关组合。
- HEAD `0b812f4fc3f95d4e4eb735d55abe4486a867c2ad` 加累计工作区。完整pytest：3591 passed、2 skipped、260 warnings、43.25s、exit0（`/tmp/fixed-router-ep1-all-units.log`）。全仓适用Git文件pre-commit、compileall/diff通过；新脚本文档行宽初次失败已修正。外部进一步暂存了本批生产文件，保留index和测试/文档工作区，不重置或提交。无新增生产文件。
- 当前仅关闭EP1/TP1公开选项静默忽略；TP feature-sharded fixed routing、EP1完整Trainer/FSDP恢复及目标设备矩阵仍开放。下一动作适配TP边界下fixed routing或推进其他可运行的必需迁移缺口；全局不完成，无commit/push。

### FIXED-ROUTER-TRAINER-RESUME-01：EP1 本地固定路由的完整训练恢复

- 验证缺口 / `Trainer` → parallelize → EP1本地转换 → native/custom FSDP2 → AdamW/EMA → DCP。`pp_checkpoint_equivalence.py` 新增LLMTUNER_TEST_FIXED_ROUTER公开配置选项及每步非空断言：所有decoder MoE确已转换、router.fixed_router=True、专家数8/EP degree。生产未改；先前EP1 block级数值证据沿用FIXED-ROUTER-EP1-01，不把连续/恢复互比当作独立数学参考。
- 两rank CPU/Gloo FP32，Qwen3MoE两层/E8/top2，PP1/TP1/CP1/EP1/DP shard2，分别native与custom FSDP；dynamic EMA、默认HF辅助loss、无AC/compile，同步DCP。full四步，第2步checkpoint；独立新进程resume前先逐位比较模型/AdamW/scheduler/trainer/EMA快照，双方pre-update failures=0，再训练两步比较连续轨迹和最终状态。两实现full/resume共四命令均exit0。连续loss均为[4.878663063049316, 4.86895751953125, 4.842203140258789, 4.858000755310059]，恢复后两步最大差异0；不放宽原逐位断言。
- 命令模板 `LLMTUNER_TEST_FIXED_ROUTER=1 LLMTUNER_TEST_MOE=1 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_EMA=dynamic PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=<port> tests/integration_tests/pp_checkpoint_equivalence.py <full|resume> <workdir>`。native端口29895/29896、目录/tmp/fixed-router-trainer-native；custom另加LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom、端口29897/29898、目录/tmp/fixed-router-trainer-custom。日志同目录名前缀-full.log/-resume.log。日志中的loss与状态检查摘要以上述记录长期保留，不仅依赖临时文件。
- HEAD `0b812f4fc3f95d4e4eb735d55abe4486a867c2ad` 加累计工作区；本批仅集成脚本和文档，相关pre-commit/diff通过，复用FIXED-ROUTER-EP1-01完整3591 passed/2 skipped门禁，无生产变化无需重跑pytest。没有开启LLMTUNER_TEST_RNG_STATE/LOADER_STATE快照选项，因此不声称逐项独立验证RNG/data state；已有轨迹和训练状态恢复已验收，设备、跨拓扑和编译未覆盖。
- TP适配继续开放：已阅读真实HF router输出协议、TPMoeSequenceBoundary、tp_sharded_param_names及root _tp_checkpoint_layouts注册；通用转换会把packed gate/up拆成两个参数、移动FQN，native aux在full-token gather后的复制注入还需考虑TP gate归约缩放。不得只移除配置限制。下一动作沿此契约实现TP适配，或推进另一独立未验迁移；全局不完成，本批无commit/push。

### LOCAL-INTERRUPT-COORDINATION-01：本地中断与退出参与错误协调

- P1 / `accelerator/collectives.py::synchronize_local_errors`，调用链从 CLI → Trainer → runtime/assembly 的模型构造等本地阶段进入。原实现仅捕获 Exception，KeyboardInterrupt/SystemExit 会跳过状态归约，使正常 rank 等待通信。单测在原实现 2 failed、1 passed，确认两类异常均未提交失败状态；修复改为捕获 BaseException，完成既有协调后原样重抛，不改变阶段顺序或增加 collective。
- 新单测覆盖 ValueError/KeyboardInterrupt/SystemExit 的失败状态与异常对象身份；扩展 `trainer_initialization_coordination.py`，在两个模型构造入口及各自失败 rank 注入三种异常，共六配置。真实 CPU/Gloo 两 rank 检查健康方 RuntimeError、失败方原异常、未进入并行变换、RNG/执行策略回滚及后续 barrier；既有 loader/checkpointer/metrics 失败清理及新 Trainer 一次更新全部通过。
- 环境 PyTorch 2.14.0、CPU，HEAD `1cacd3e52ba6a4947fc8ddd90fc006d0820557a4` 加已有文档及本批一个生产/两个测试文件改动。初次与独立缓存目录重跑均在最后训练更新遇到本机已有 Inductor PCH mtime 失效，exit1，日志 `/tmp/llmtuner-local-interrupt-coordination.log` 和 `-fresh.log`；六配置协调本身均通过，不把整脚本记为通过。
- 最终命令 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 TORCHINDUCTOR_CACHE_DIR=/tmp/llmtuner-interrupt-no-pch-20261016 PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone --nproc_per_node=2 tests/integration_tests/trainer_initialization_coordination.py`，exit0，日志 `/tmp/llmtuner-local-interrupt-coordination-no-pch.log`。仅禁用环境失效的 PCH 缓存，未修改训练实现、数值断言或容差。
- 修改前完整 CPU 基线 3591 passed/2 skipped；修改后 `pytest -q --disable-warnings` 3594 passed、2 skipped、266 warnings、38.84s、exit0，日志 `/tmp/llmtuner-local-interrupt-units.log`。全部适用 Git 文件 pre-commit（排除受保护示例和未跟踪演示文稿）、compileall、diff 检查通过。178 个源码路径均出现在逐文件台账中，该路径核对不代表本轮重新阅读全部文件。
- 此修复只覆盖仍可执行本地异常处理和后续协调的进程，不覆盖 SIGKILL、通信内部中断或设备故障恢复。未提交/推送，既有工作区改动保留。下一批继续沿数据与 train_step 的失败边界审查，TP fixed routing、KCP state/padding 与设备矩阵等开放项保持开放，全局未完成。

### DATA-FETCH-INTERRUPT-01：训练与验证读取中断参与状态归约

- P1 / `trainer/runtime/batch.py::fetch_batch`。继续从 CLI/Trainer 追踪数据读取，发现该函数独立实现 ready/exhausted/failed 协议，仅捕获 Exception；因此上批通用 synchronize_local_errors 修复不能覆盖它。原实现新增 KeyboardInterrupt/SystemExit 回归 2 failed，均因失败状态归约未执行。改为捕获 BaseException，沿用原有一次 MIN 归约及失败优先于耗尽的策略，源 rank 保留异常对象，不新增抽象或通信。
- `tests/unit_tests/cpu/test_trainer.py` 新增两项异常身份及归约状态回归。`tests/integration_tests/data_fetch_coordination.py` 扩展训练/验证各自的 interrupt/exit 场景：失败 rank 原异常，健康 rank 明确 RuntimeError，无不匹配 batch 产出，吞吐计数仍为首个成功 batch 的4 token，验证 loader 关闭且不发布结果；另两场景检查一方中断、另一方耗尽时失败优先，所有场景后 barrier 成功。原有异常、结构错误、正常耗尽、验证初始化/清理/结果错误场景继续通过。
- 两 rank CPU/Gloo 命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone --nproc_per_node=2 tests/integration_tests/data_fetch_coordination.py`，exit0，日志 `/tmp/llmtuner-fetch-interrupt-coordination.log`。仅在验证结束后将函数导入合并到模块导入区，行为未变。非真实数据源中断、目标设备或通信内部故障的验收。
- HEAD `1cacd3e52ba6a4947fc8ddd90fc006d0820557a4` 加既有文档/上批协调改动及本批一个生产/两个测试文件。完整 `pytest -q --disable-warnings`：3596 passed、2 skipped、274 warnings、45.99s、exit0，日志 `/tmp/llmtuner-fetch-interrupt-units.log`；全部适用 Git 文件 pre-commit（排除受保护示例及未跟踪演示文稿）、compileall、diff 检查通过。无新源码文件或名称迁移。
- 当前关闭仍可执行 Python 异常处理的读取中断协调缺口；不声称能处理 loader 阻塞、SIGKILL 或 collective 内部异常。下一批检查训练 iterator 初始化、token 计数与模型预处理的 rank 局部失败边界；TP fixed routing、KCP padding/state 和设备矩阵仍开放。全局目标未完成，未 commit/push。

### TRAIN-ITERATOR-INIT-01：训练数据迭代器初始化失败协调

- P1 / `trainer/trainer.py::train` → `data_iterator` → `iter(loader)`。fetch_batch 只保护 next，训练入口在 profiler/循环之前直接初始化 iterator；某 rank 的 __iter__ 失败会直接清理退出，其他 rank 继续运行。验证入口已有 Validation input initialization 保护，本批复用同一 synchronize_local_errors，在纯本地训练 iterator 初始化外围协调，不拆分 trainer.py、不把 fetch collective 放入 local-only 边界。
- 扩展 `tests/integration_tests/data_fetch_coordination.py`：真实 Trainer.train，rank0 loader.__iter__ 分别抛 ValueError/KeyboardInterrupt/SystemExit；双方 checkpoint 和 loader 各关闭一次，原 rank 保留异常对象，健康 rank 收到 Training input initialization failed on another rank，profiler 未构造，每场景 barrier 成功。既有读取/验证所有场景保持通过。
- 修复前脚本 exit1，日志 `/tmp/llmtuner-iterator-before.log`：健康 rank 已越过迭代器初始化进入 profiler 参数求值，因精简 fixture 没有 profiler 字段报 AttributeError；这证明越界，但不称真实 deadlock 复现。修复后 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone --nproc_per_node=2 tests/integration_tests/data_fetch_coordination.py` exit0，日志 `/tmp/llmtuner-iterator-after.log`，三个新场景两侧全部通过。随后仅绑定测试闭包循环变量以满足 Ruff，不改执行逻辑。
- 环境 CPU/Gloo、PyTorch 2.14.0，HEAD `1cacd3e52ba6a4947fc8ddd90fc006d0820557a4` 加既有累计工作区和本批 trainer/集成脚本。完整 pytest -q --disable-warnings：3596 passed、2 skipped、264 warnings、39.58s、exit0（`/tmp/llmtuner-iterator-units.log`）；全部适用 Git 文件 pre-commit（排除受保护示例/未跟踪演示文稿）、compileall/diff 通过。pre-commit 初次发现新测试三处循环闭包 B023，绑定后重跑通过。
- 新增每次 Trainer.train 一次状态协调，不是每步新增通信；不覆盖 loader 初始化内部通信、阻塞或进程强制退出。无源码新增/搬迁，现有工作区保留，未提交推送。下一项 token 计数阶段目前仍是 rank 本地运算后直接进入归约，需检查畸形元数据/计数异常传播；全局未完成。

### TOKEN-COUNT-METADATA-01：拒绝非法 loss 分母元数据并协调失败

- P1 / `trainer/runtime/batch.py::count_valid_tokens`、`fetch_batch`。collator 的 num_valid_tokens 是 Python int；原计数入口直接返回任意元数据，浮点值后续构建 int64 分母会截断，负数/布尔/越界也未拒绝。新增七项回归（True/False/-1/1.5/字符串/超过标签总数/列表），原实现全部未抛错，7 failed。现明确要求非 bool 的 Python int，且处于 [0, labels.numel()]；缺省或 None 仍重计数，合法 collator 元数据仍优先，不增加逐标签一致性扫描。
- 在 fetch_batch 已有 local try 中复用计数入口检查存在的元数据，因此训练/验证在进入 token reduction 前统一失败，沿用原 MIN 归约，无新增 collective、标签扫描或设备同步。正常 Batch 不在 fetch 重计数，字典缺省计数仍留到原消费阶段。已阅读文本/多模态/Pipeline collator，原生计数类型符合约束。
- 两 rank CPU/Gloo 集成新增训练及验证 rank0 提供1.5计数、rank1继续有效数据：原 rank ValueError含字段名，健康 rank RuntimeError，不产出第二个 batch、不增加吞吐计数、不发布验证指标，验证 loader 关闭，后续 barrier 成功。命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone --nproc_per_node=2 tests/integration_tests/data_fetch_coordination.py` exit0，日志 `/tmp/llmtuner-token-metadata-coordination.log`，所有既有错误/耗尽/初始化场景仍通过。
- HEAD `1cacd3e52ba6a4947fc8ddd90fc006d0820557a4` 加既有累计工作区和本批 batch.py/两个测试文件。完整 `pytest -q --disable-warnings`：3603 passed、2 skipped、283 warnings、43.34s、exit0（`/tmp/llmtuner-token-metadata-units.log`）。适用全仓 pre-commit 初次新增场景元组超长，由 formatter 修复后受影响检查通过；compileall/diff 通过。没有修改数值容差、正常计数语义或公共名称。
- 此项不证明计数元数据与 IGNORE_INDEX 分布逐位一致；仍按已有契约信任合法范围的 collator 计数。自定义 loader 改为提供上述整数或省略字段；numpy/张量标量不是内置 collator 的公开输出。剩余 count_routing_tokens、缺省计数/微批准备失败传播及模型预处理边界继续开放；全局未完成，未提交/推送。

### MICROBATCH-ACCOUNTING-01：统计失败前不提交 token 计数

- P2 / `trainer/runtime/batch.py::microbatch`。原实现先递增持久化 ntokens_seen 再执行有效/路由 token 统计；程序化 train_step 或直接 microbatch 调用不经过 fetch 校验，统计异常会留下错误状态。新增原实现回归失败：原17 token在有效计数报错后变21。现把计数提交移至两项统计成功之后，不新增包装、通信或状态字段。
- 两项回归覆盖非法有效计数 ValueError 和缺少 input 导致路由统计 KeyError，均要求原17 token不变。已有 canceled accumulation window 的已消费 token 语义保持（该窗口中成功准备的批仍计数），不把本项描述为完整 optimizer step 回滚。正常吞吐、恢复及CP/TP分片计数由现有测试复用。
- HEAD `1cacd3e52ba6a4947fc8ddd90fc006d0820557a4` 加已有累计工作区及本批 batch.py/单测。完整 pytest -q --disable-warnings：3605 passed、2 skipped、267 warnings、41.60s、exit0，日志 `/tmp/llmtuner-microbatch-accounting-units.log`；全部适用 Git 文件 pre-commit、compileall/diff 通过，排除受保护示例/未跟踪演示文稿。本项只有本地计数赋值顺序调整，无通信变化，不重复分布式数值实验。
- 工作期间外部暂存前批 collectives.py、batch.py、trainer.py，已核对并保留；本批计数顺序仍为 batch.py 未暂存差异，未操作 index/commit/push。HF preprocess 的字段长度检查已阅读，CP packed mask路径可能通信，不能直接把整个 preprocess 包进 local-only 协调；后续须区分纯本地校验与分片通信。全局审查及TP fixed/KCP/设备矩阵缺口继续开放。

### PREPROCESS-LOCAL-ERROR-01：模型执行前协调预处理错误

- P1 / `trainer/runtime/batch.py::preprocess`。此前数据读取只检查labels/计数，HF长度、媒体字段、mask构造等本地错误发生在其后；一侧退出，其他侧可进入模型collective。现Trainer共用预处理入口使用synchronize_local_errors保护搬运和HF预处理，模型forward仍位于边界之外；训练、验证和PP prepare_pipeline_inputs均调用该入口，不拆分HF或trainer实现。
- 修正前项MICROBATCH-ACCOUNTING-01关于“CP packed mask路径可能通信”的未确认判断：本机PyTorch2.14源码_context_parallel_shard→_context_parallel_buffers以distribute_tensor(src_data_rank=None)本地切片；BlockMask使用_create_cp_block_mask本地构造，未发现collective。已阅读本项目_shard_for_cp/_shard_for_tp/get_attention_masks及input_sharding调用链。此结论针对当前HF实现，自定义preprocess适配器必须保持无collective契约，后续升级私有torch API须重新核查。
- 扩展真实两rank CPU/Gloo脚本，离线Qwen3 HF模型，两种失败rank的labels长度错误均原侧ValueError、对侧明确RuntimeError，随后barrier；另实际CP2/TP2两种mesh本地输入/标签切片逐位匹配独立torch.chunk结果。既有训练读取、验证清理及初始化全部通过。命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone --nproc_per_node=2 tests/integration_tests/data_fetch_coordination.py` exit0，最终日志 `/tmp/llmtuner-preprocess-sharding.log`。不执行CP/TP模型forward，不据此宣称attention数值/目标设备组合验收。
- 初次测试fixture误用build_model_config签名（无seq_len、直接传架构字段）导致两rank TypeError，exit1（`/tmp/llmtuner-preprocess-coordination.log`）；按实际factory API改为seq_len+arch_overrides后脚本通过（`/tmp/llmtuner-preprocess-coordination-final.log`），再扩展CP/TP得到最终结果。不据fixture错误修改生产factory。
- HEAD `1cacd3e52ba6a4947fc8ddd90fc006d0820557a4` 加累计工作区及本批batch.py/集成脚本。完整pytest：3605 passed、2 skipped、274 warnings、42.53s、exit0（`/tmp/llmtuner-preprocess-units.log`）；全部适用Git文件pre-commit通过（初次测试行宽由formatter修复，最终脚本再次通过）、compileall/diff通过。未操作index/commit/push。
- 成本：每个消费组/PP微批新增world小标量状态归约；正确性门槛明确，目标设备开销未测。不恢复SIGKILL、搬运hang或通信故障。微批计数其他失败边界、PP微批拆分之前的局部错误、设备矩阵仍开放；全局未完成。

### PP-SPLIT-COORDINATION-01：预处理之前协调微批拆分失败

- P1 / `trainer/runtime/pipeline.py::prepare_pipeline_inputs`。沿上批预处理归约检查发现split_pipeline_microbatches先在每rank独立校验行数/布局；某侧失败时健康侧会进入第一个预处理状态归约，通信顺序不匹配。现复用synchronize_local_errors在纯本地拆分外围协调，再按既定微批数逐个预处理，不把内层collective包进local-only边界，无新包装或API。
- `data_fetch_coordination.py`增加首/末stage分别持3行、另一侧4行、微批数2的场景；原侧ValueError含divisible，健康侧Pipeline input splitting failed on another rank；双方preprocess均未调用，后续barrier成功。真实两rankCPU/Gloo脚本完整exit0（`/tmp/llmtuner-pp-split-coordination.log`），命令沿已有PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone --nproc_per_node=2。此场景测试PP输入准备，不把它当作真实schedule执行证据。
- 实际PP2 Trainer补充证据：两rankCPU/Gloo FP32、Qwen3两层、1F1B/4微批、其他模型轴1、无AC/compile/EMA，连续4步在step2保存DCP；独立新进程恢复前两侧pre-update state failures=0，后续两步与连续路径最大loss差0，最终模型/optimizer/scheduler/trainer检查全部通过。连续loss=[4.878459930419922,4.867842197418213,4.858003616333008,4.86264705657959]。
- 命令模板 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py <full|resume> /tmp/llmtuner-pp-preprocess-20261016`，两进程均exit0，日志`/tmp/llmtuner-pp-preprocess-full.log`、`-resume.log`。PCH开关沿前批本机缓存失效规避，未禁用训练目标机制，不外推设备或跨编译恢复。
- HEAD `1cacd3e52ba6a4947fc8ddd90fc006d0820557a4` 加累计工作区及本批pipeline.py/集成测试修改。完整pytest：3605 passed、2 skipped、264 warnings、40.39s、exit0（`/tmp/llmtuner-pp-split-units.log`）；适用全仓pre-commit、compileall、diff通过。既有暂存区保持，未提交推送。
- 新增每消费组一次world小标量归约，目标设备开销未测。本项确认当前统一num_pp_microbatches配置下正常PP调度/恢复；不证明不同rank私自更改微批数仍支持。全局目标与其他并行设备/迁移缺口保持开放，下一步继续微批计数阶段与公开组合矩阵。

### OPTIMIZER-POSTSTEP-REVIEW-01：scheduler/EMA 生命周期复核

- 本批继续逐文件审查 `components/optimizer/ema.py`、`lr_scheduler.py`、`config/ema_config.py`，并核对 `Trainer.train_step` 更新顺序和 `DistributedCheckpointManager._load_checkpoint` 的普通/Kimi跨布局EMA冷启动分支。未确认需要修改的生产缺陷，不为本轮审查制造重构。
- 当前契约：optimizer.step → scheduler.step → EMA.step(self.step)，日志学习率在更新前快照；scheduler共享last_epoch、按恢复optimizer.initial_lr重建每组学习率，拒绝重复optimizer、进度分歧及超长恢复。EMA按trainer step派生firing count，先校验全部跟踪组再更新，空state仅在模型恢复后重置；选择buffer支持alias去重，FQN冲突及参数身份/元数据变化显式拒绝。已有CPU回归覆盖独立手算lerp、动态衰减、门控、tensor LR身份和失败时不部分更新。
- 定向命令 `pytest -q tests/unit_tests/cpu/components/optimizer/test_ema.py tests/unit_tests/cpu/components/optimizer/test_lr_scheduler.py --disable-warnings`：116 passed、1.54s、exit0，日志 `/tmp/llmtuner-ema-scheduler-review.log`。HEAD `1cacd3e52ba6a4947fc8ddd90fc006d0820557a4` 加累计工作区，本轮无生产/测试改动；复用PP-SPLIT-COORDINATION-01完整3605 passed/2 skipped及真实PP恢复证据，不重复运行全量。
- 本轮限于源码复核与CPU回归，不代表所有dense/expert DTensor mesh、低精度EMA或目标设备buffer组合验收。路由mask长度检查已追踪至HF preprocess字段长度校验；没有把重复前置检查作为新修复。全局目标、并行迁移与设备矩阵缺口仍开放；文档按范围pre-commit/diff检查，已有暂存区保持，无commit/push。

### SEED-LOAD-ONLY-CONFLICT-01：拒绝不会产生seed产物的配置组合

- P2 / `config/checkpoint_config.py::__post_init__` → CLI seed分支 → BaseCheckpointManager._should_save。create_seed_checkpoint=True可与load_only=True同设；CLI只检查enable/world size，save因load_only返回False且入口不训练，可能成功退出但无seed产物。新增配置回归原实现1 failed（未抛ConfigError），确定组合未被拒绝。
- 直接在所属CheckpointConfig中拒绝create_seed_checkpoint+load_only，提示checkpoint.load_only=False，CLI及程序化构造复用同一校验，不新增入口分支。普通load_only、正常seed及enable首步提示保持原语义；有意收紧此前无产物组合，无持久化格式变更。
- 真实CLI `python -m llmtuner --enable --create_seed_checkpoint --load_only` exit1，明确ConfigError字段与条件，日志 `/tmp/llmtuner-seed-load-only-cli.log`。完整pytest -q --disable-warnings：3606 passed、2 skipped、282 warnings、42.00s、exit0（`/tmp/llmtuner-seed-load-only-units.log`），既有真实step0保存回归保持通过。
- HEAD `1cacd3e52ba6a4947fc8ddd90fc006d0820557a4` 加既有累计工作区及本批checkpoint_config.py/test_config.py。适用全仓pre-commit初次新增错误字符串89字符失败，拆分相邻字符串后受影响检查通过；compileall/diff通过。已有index保持，未commit/push。
- 本批另阅读base.py的load来源优先级及_should_save，不据此关闭全部异步/远程checkpoint风险。全局目标和设备/混合并行矩阵保持开放，继续核对其他可执行项。

### CHECKPOINT-LOAD-PREFLIGHT-01：恢复路径与状态选择异常协调

- P1 / `components/checkpointer/base.py::load`。源码确认_resolve_load_source执行本地storage扫描/元数据探测，_states_to_load执行排除项校验；原两者未协调，rank局部IO/状态选择异常会使其他rank进入_load_checkpoint/DCP。现在staging/saving等待后复用synchronize_local_errors，先完成路径解析与待恢复state选择，再在边界外加载，拒绝前不改模型。无路径优先级、状态类别或持久化格式变更。
- 扩展 `tests/integration_tests/checkpoint_save_coordination.py`，真实manager先写step1；_resolve_load_source和_states_to_load各对rank0/rank1注入OSError，四场景原rank保留异常对象、健康rank明确Checkpoint load preflight错误，_load_checkpoint未调用、模型逐位不变、barrier成功。解除故障后真实DCP load恢复保存权重。测试使用故障注入，不称真实远程IO故障验收。
- 命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone --nproc_per_node=2 tests/integration_tests/checkpoint_save_coordination.py` exit0，日志 `/tmp/llmtuner-load-preflight-coordination.log`；既有writer/stager/staging同步、线程异步与PROCESS失败重试场景全部通过。随后仅为测试闭包绑定循环变量、生产增加类型收窄assert，执行逻辑不变。
- HEAD `1cacd3e52ba6a4947fc8ddd90fc006d0820557a4` 加累计工作区及本批base.py/集成脚本。完整pytest：3606 passed、2 skipped、273 warnings、42.53s、exit0（`/tmp/llmtuner-load-preflight-units.log`）；适用全仓pre-commit、compileall/diff通过。未新增生产文件，原index保持，未提交推送。
- 成本为每次启用checkpoint load一次状态协调，不是每训练step通信。本项只协调抛异常的预检；不同rank在无异常情况下发现不同源/一侧None的存储可见性差异仍需独立检查，不能宣称全恢复发现协议一致性已完成。无硬件/远程传输验收，全局目标仍开放。

### CHECKPOINT-LOAD-CONSENSUS-01：恢复源与类别分歧统一拒绝

- P1 / `components/checkpointer/base.py::load`。前项只协调预检异常；原有一rank返回None、另一rank解析到有效source时，双方ready均为真，一侧直接return False、另一侧进入DCP。新增两rank回归在修复前exit1，双方AssertionError为missing: inconsistent restore accepted（`/tmp/llmtuner-load-consensus-before.log`），确认静默接受分歧而非通信超时推测。
- 现在预检成功后在默认组以all_gather_object交换小型内部选择描述（CheckpointLoadSource及有序状态类别），发现任意不一致则所有rank明确RuntimeError，之后才允许返回False或加载。包含路径、model_only、HF/量化标志和类别，不传模型张量、不选择任意rank覆盖其他rank的发现结果。单rank不新增通信；分布式每次load增加一次对象元数据交换，NCCL对象通信依赖既有正确current device，设备成本未测。
- 扩展真实manager集成：分别制造source缺失、path、model_only、格式、状态类别五类分歧，双方均拒绝且_load_checkpoint未调用、模型逐位未变、barrier成功；全部None一致返回False，之后原生真实DCP retry恢复权重。原异常预检与writer/stager/staging同步/异步/PROCESS故障场景全部通过。最终 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone --nproc_per_node=2 tests/integration_tests/checkpoint_save_coordination.py` exit0，日志 `/tmp/llmtuner-load-consensus-final.log`。首次修复后五类测试亦通过（`-after.log`），最终另补全None场景。
- HEAD `1cacd3e52ba6a4947fc8ddd90fc006d0820557a4` 加累计工作区。完整pytest：3606 passed、2 skipped、279 warnings、40.13s、exit0（`/tmp/llmtuner-load-consensus-units.log`）；全部适用Git文件pre-commit、compileall/diff通过；初次测试长行由formatter修正后重跑通过。工作期间外部暂存前项base.py及config修复，已保留index，本项consensus为其上的未暂存变更，未commit/push。
- 同一checkpoint的类别相同不证明rank内部张量内容/拓扑一致，本项不扩张为完整数据校验；也不保证不同路径映射同一底层文件的布局可自动识别。存储挂载与恢复配置必须一致，明确报错后由调用方修正。目标设备、远程部署和其他并行组合矩阵继续开放，全局未完成。

### CHECKPOINT-LOAD-CONSENSUS-PP-01：新恢复协议的实际PP兼容验证

- 证据缺口 / 前项CHECKPOINT-LOAD-CONSENSUS-01只用复制Linear模型；本批使用真实PP2本地不同参数parts，验证新source/state类别一致性检查不会把stage张量差异误判为恢复源分歧。生产/测试未新增改动。
- 两rank CPU/Gloo FP32、Qwen3两层、PP2/1F1B/4微批，TP/CP/EP1，无AC/compile/EMA，native配置。完整Trainer连续4步step2保存DCP；独立新进程恢复前两侧pre-update state failures=0，恢复后两步loss最大差0，最终model/optimizer/scheduler/trainer一致。连续loss=[4.878459930419922,4.867842197418213,4.858003616333008,4.86264705657959]。仅比较共用类别和路径，未要求不同stage的参数FQN相同。
- 命令模板 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py <full|resume> /tmp/llmtuner-load-consensus-pp-20261016`；两个session均exit0，日志`/tmp/llmtuner-load-consensus-pp-full.log`、`-resume.log`。PyTorch2.14.0，HEAD `1cacd3e52ba6a4947fc8ddd90fc006d0820557a4` 加当前累计工作区。PCH开关只规避已记录本机编译缓存问题。
- 继续阅读base.py保存节奏、load来源、purge队列/独立保留类别、close中断保留future及distributed_checkpoint.py staging/saving收尾；未确认新缺陷，不制造实现修改。`pytest -q tests/unit_tests/cpu/components/checkpointer --disable-warnings`：200 passed、105 warnings、1.61s、exit0（`/tmp/llmtuner-retention-review.log`）。复用前项完整3606 passed/2 skipped，本批仅补证据及文档，无需重跑全量。
- 不关闭远程存储可见性、NCCL/HCCL对象交换、PP×其他并行及异步load期间进程故障；全局未完成。暂存/未暂存内容保持，未提交推送，文档pre-commit/diff通过。

### DATA-RESTORE-CONTRACT-REVIEW-01：游标、配置身份与状态提交复核

- 本轮逐文件重新阅读datasets/synthetic.py、datasets/dataloader.py，追踪loader_builder.py::corpus_data_identity、Trainer.load_state_dict及tokenizer/huggingface.py和multimodal.py实际资源读取。未确认新生产缺陷，未修改实现或新增等价测试。
- 合成loader在提交_iterator/_position前校验字典、DP degree、batch_layout、steps范围和不后退约束；绝对batch索引使用独立CPU generator，不改变全局RNG，已有iterator读取新的_source iterator引用。Grain在set_state前检查version/DP/rank state/layout，内部set_state失败关闭iterator并保留原异常；这不是原地回滚保证，失败后需新建loader。Trainer两个非负整型counter全部校验后才共同赋值。
- 指纹排除本地挂载路径和prefetch深度，包含处理配置、seed/repeat、本地语料内容与tokenizer资源内容。核查当前tokenizer实际只加载tokenizer.json/vocab.txt/vocab.json/merges.txt/tokenizer_config.json/chat_template.jinja，均在现有列表；未因外部HF常见但本项目不读取的额外资源制造虚假遗漏。远程/注册recipe仍需声明data_identity，指纹不是运行期间文件变更锁。
- `pytest -q tests/unit_tests/cpu/datasets/test_data_pipeline.py tests/unit_tests/cpu/test_trainer.py -k 'state or resume or cursor or checkpoint' --disable-warnings`：96 passed、378 deselected、24 warnings、12.82s、exit0，日志 `/tmp/llmtuner-data-state-review.log`。仅状态相关子集，不称整个数据模块测试或全量重跑；复用前项3606 passed/2 skipped累计门禁。
- HEAD `1cacd3e52ba6a4947fc8ddd90fc006d0820557a4` 加现有累计工作区；本批仅台账。文档pre-commit/diff按范围验证，暂存区未操作，无commit/push。有限DP remainder策略、运行中源文件变化、远程snapshot与设备矩阵仍未由本批验证，全局继续。

### MEMORY-AC-REVIEW-01：重计算策略与环境说明复核

- 重新阅读memory/activation_checkpoint.py、remat_regions.py、config/activation_checkpoint_config.py并追踪调用契约：full/selective使用非重入wrapper；checkpoint_contexts捕获每次forward的MoE pending padding mask，重算后恢复当时live mask及ContextVar；full/selective先构造全部wrapper再替换layers，region按nn.Linear结构声明区域，memory_budget仅在compile开启时应用全局knob。
- 专项CPU `pytest -q tests/unit_tests/cpu/memory --disable-warnings`：96 passed、8.44s、exit0，日志 `/tmp/llmtuner-memory-review.log`。另外交互实验用三段不同shape的bias-free Linear/Sigmoid，多个待反向forward后逆序backward，对独立deepcopy未AC模型输出梯度观察到差0；未据选择策略counter疑虑修改代码，也不把该小实验当全部PP/设备重计算证明。
- 修正activation_checkpoint.py模块docstring的过时环境断言：此前称region因开发torch太旧不可执行；当前PyTorch2.14.0，实际importlib探测torch_remat不存在。改写为固定的torch>=2.10及安装可选包要求，不绑定开发环境版本；未改变任何运行实现。region真实backend本机仍未验证，单测fake不能替代它。
- HEAD `1cacd3e52ba6a4947fc8ddd90fc006d0820557a4` 加累计工作区，本批只有模块说明及台账。受影响pre-commit/diff通过；复用现有完整3606 passed/2 skipped门禁，无行为变化不重跑全量。无index/commit/push操作。模型依赖的region实际执行、内存预算编译效果、selective topk设备确定性仍需对应环境证据，全局目标未完成。

### COMPILE-MEMORY-BUDGET-ROLLBACK-01：装配失败恢复激活内存预算

- P2 / `parallel/model_compile.py::rollback_compile_settings_on_error` → Trainer.__init__。逐文件审查compile配置/应用、parallel/stages及AC设置链路，发现memory_budget AC设置functorch.activation_memory_budget后若后续装配失败，既有rollback只恢复scalar capture/_micro_pipeline_tp，导致下一模型继承失败运行预算。新增真实Trainer构造失败边界单测（装配阶段注入全局预算变化）原实现两类异常均失败，预算从1.0残留0.25。
- 在现有compile rollback上下文中快照/恢复activation_memory_budget，能力缺失则不写；不新增上下文或导入训练入口，不改变成功运行的预算。memory_budget模式本来要求compile，Trainer的既有compile设置上下文覆盖该生产路径。普通ValueError及KeyboardInterrupt均保留原异常对象并恢复原值。
- 完整 `pytest -q --disable-warnings`：3608 passed、2 skipped、279 warnings、38.33s、exit0，日志 `/tmp/llmtuner-memory-budget-rollback-units.log`。适用全仓pre-commit（排除受保护示例/未跟踪演示文稿）、compileall/diff通过；无collective/布局变更，本批不重复分布式实验。
- HEAD `1cacd3e52ba6a4947fc8ddd90fc006d0820557a4` 加累计工作区及本批model_compile.py/test_trainer.py，原index保持，未提交推送。此项仅闭合失败装配的预算泄漏，不恢复成功运行后的设置、不处理并发策略修改、symm_mem注册或Dynamo仅有setter无getter的LRU状态，不声称所有torch全局变量事务回滚。async TP真实设备性能与region后端等缺口继续开放，全局未完成。

### PROFILER-LIFECYCLE-REVIEW-02：训练 trace 与 allocator history 生命周期复核

- 重新逐文件阅读 `components/profiler.py`、`utils/monitoring.py`、`config/observability_config.py`，追踪 `Trainer.train` 的进入、step、异常退出及 finally close；检查对应 profiler 回归的断言。未确认新的生产缺陷，本批不修改实现或增加重复测试。
- 已核对：trace 部分启动失败尝试退出且保留原异常；memory 构造失败由外层停止已启动 trace；退出时 trace 导出失败仍尝试 OOM snapshot 和 memory close，训练原异常优先；history 停止失败保留 owner 供显式退出重试。相同容量的重叠 owner 共享 recorder，最后 owner 才停止；冲突容量在启动前拒绝。snapshot 用同目录临时文件及 replace 发布，失败删除临时文件并保留原成品。
- `pytest -q tests/unit_tests/cpu/components/test_profiler.py tests/unit_tests/cpu/test_monitoring.py --disable-warnings`：65 passed、1 warning、1.33s、exit0。包括真实 CPU trace gzip JSON 导出、恢复 step 的 schedule、部分 entry 失败后再次启动成功；设备 allocator history 和关闭失败使用替身，不等价于 CUDA/NPU allocator 实测。无需由本轮重复无关全量测试，累计行为改动的全量结果见前项 3608 passed/2 skipped。
- 当前 main/HEAD `1cacd3e52ba6a4947fc8ddd90fc006d0820557a4` 加累计工作区；上轮 prompt 精简属独立用户请求，未替代本目标。保留已有暂存及未暂存改动，未 commit/push。此批仅增加审查证据，文档 pre-commit 与 diff 按范围检查；多 rank profiler 局部 I/O 故障的全局传播、设备侧 trace/history 及性能仍未由此批证明，全局目标继续。

### VALIDATION-STATE-REDUCTION-REVIEW-02：验证隔离与四 rank 归约复核

- 逐文件重读 `trainer/runtime/validation.py`、`reductions.py`、`config/validation_config.py`、`utils/seeding.py`，并追踪 `accelerator/spmd.py` 的上下文恢复。核对 staging 完成后进入 eval、逐 module 混合模式恢复、共享 config 的 mask 属性存在性恢复、kernel packed 标记恢复，以及 Python/NumPy/CPU RNG 隔离。未确认新的实现缺陷，保留现有结构。
- `pytest -q tests/unit_tests/cpu/test_validate.py tests/unit_tests/cpu/test_pp_distinct_seed.py --disable-warnings`：74 passed、4.49s、exit0。已检查成功/失败恢复断言、零 batch/有效 token 拒绝、非有限 loss 拒绝、loader 分配/迭代/关闭失败的异常优先级。设备 RNG 由 torch fork_rng 契约承载，本批未运行设备 RNG 验证。
- `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone --nproc_per_node=4 tests/integration_tests/validation_reduction_equivalence.py`：exit0，输出 DP x CP、DP x TP 均匹配完整 batch loss。真实 CPU/Gloo，DP2×CP2 与 DP2×TP2 两种 mesh、两批不均匀有效 token、FP64 确定性本地投影，生产验证循环/collective 对照独立完整交叉熵，rtol/atol=1e-7；确认 loader closed、混合模式恢复及 ntokens_seen=123 不变。脚本显式切序列，不验证实际 attention、TP/FSDP 模型变换、PP 或设备性能；可选 jax profiler 缺失警告未影响退出。
- 当前 HEAD `1cacd3e52ba6a4947fc8ddd90fc006d0820557a4` 加累计工作区。本批仅台账，文档 pre-commit/diff 按范围检查；未操作暂存区、commit/push。多 rank 自定义 eval/setup 本地异常传播以及目标设备 RNG/模型变换矩阵未被本批闭合，全局审查继续。

### CLI-ASSEMBLY-REVIEW-03：入口、配置装配与失败清理复核

- 重新逐文件阅读 `trainer/cli.py`、`__main__.py`、`config/cli_config.py`、`config/llmtuner_config.py`、`trainer/runtime/assembly.py`，追踪 Trainer 构造/close 边界及 checkpointer 内部 services/writer 协调。`pyproject.toml` 的 llmtuner-train 与模块入口均指向同一 main。未确认新缺陷，不为本批制造结构改动。
- 核对 parser group/view 顺序和 strict zip、隐藏 programmatic 字段、from_groups 重建配置且不修改原组；实际 HF geometry 在数据构建前更新。模型在 optimizer 捕获参数前应用并行，PP 本地 parts 被 scheduler/EMA/checkpointer 共用；checkpointer 内部已含协调，未在外层包住可能包含 collective 的整个构造。CLI 拥有进程组销毁，程序化 Trainer.close 保留进程组；训练原错误优先于资源/组清理错误。
- `pytest -q tests/unit_tests/cpu/test_trainer.py -k 'cli or seed_checkpoint or failed_trainer_construction or failed_assembly' --disable-warnings`：31 passed、109 deselected、7.88s、exit0。包含真实子进程创建 `.metadata` 的 step-0 checkpoint 并确认未训练到 step5，显式 argv/隐式 args 文件、seed 配置前置拒绝、组清理错误优先级、部分构造释放及 host RNG/compile policy 回滚。`python -m llmtuner --help` 正常显示全部 parser flags，exit0。此处延迟的是 Trainer 导入，CLI 本身仍依赖 transformers/torch，不宣称入口无重依赖。
- HEAD `1cacd3e52ba6a4947fc8ddd90fc006d0820557a4` 加累计工作区；本批仅审查证据，文档 pre-commit/diff 检查，未操作 index/commit/push。未验证独立 rank 的配置解析失败或非对称任意自定义构造异常；已有两 rank 构造协调证据仍按各自覆盖范围使用，不扩展为全部失败路径保证。设备/多轴缺口仍开放，全局目标未完成。

### MOE-AUX-INDEPENDENT-ORACLE-01：辅助损失参考公式与实际输入契约

- 重新阅读 `models/common/aux_loss.py`、`models/common/moe/auxiliary_loss.py`、`bias_update.py`，追踪 layer/router 的 training、checkpoint replay、padding 过滤和 optimizer pre-hook。未确认新的生产缺陷。发现既有 `moe_aux_loss_grad_equivalence.py` 用同一生产辅助损失计算单进程参考，而且将有负值的 gate logits 传给要求非负 score 的公式；它能检查通信差异，却不能独立证明公式或真实 router 输入契约。
- 更新该脚本：生产路径传入 softmax 概率；参考分支直接用 E*counts/(K*T) 与概率和计算 raw sum，以 carrier.sum()+coeff*raw_sum 执行普通 autograd，不调用生产 AuxLoss/Injection。保持 CP2、FP32、全局分母1、固定种子、绝对误差1e-6，不弱化断言。仅改集成测试，未修改模型实现。
- 两 rank CPU/Gloo：`PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone --nproc_per_node=2 tests/integration_tests/moe_aux_loss_grad_equivalence.py` exit0；loss 32.080692 两侧一致，router gradient 最大绝对差4.172e-7，参考最大幅度0.6261，失败rank0。单独运行 `aux_loss_runtime_metrics.py` 同样exit0，当前模型层均值2、旧构造排除及空PP阶段参与均通过。两项独立机制验证不代表真实PP×CP模型组合已验证。
- 专项 `pytest -q tests/unit_tests/cpu/models/test_aux_loss.py tests/unit_tests/cpu/models/test_moe_bias_update.py tests/unit_tests/cpu/models/test_moe_padding_mask.py --disable-warnings`：97 passed、4.97s；完整 `pytest -q --disable-warnings`：3608 passed、2 skipped、268 warnings、37.32s、exit0，日志 `/tmp/llmtuner-aux-oracle-units.log`。适用全仓pre-commit（排除受保护示例/演示文稿）、compileall、diff通过，已审阅测试差异。
- HEAD `1cacd3e52ba6a4947fc8ddd90fc006d0820557a4` 加累计工作区；本批保留其他改动和index，无commit/push。此参考覆盖softmax、无padding、固定topk及CP2的公式/梯度；sigmoid/不同分母/真实EP组合的设备证据不能由此替代，原MC2/固定路由TP等开放项继续保留，全局目标未完成。

### PACKING-FINITE-DIAGNOSTIC-01：有限尾批契约与零 batch 诊断一致性

- 逐文件重读 `datasets/packing/build.py`、`conversions.py`、`iterators.py`、`collators.py`、`text/processors.py`、`contracts.py`。核对 processor 先 shift、packed 输入不再 shift、prompt 标签屏蔽、padding 标签掩码、上下文/文档 positions 重置、document-aware remainder 的恢复校验与失败回滚，以及 PP 部分 schedule 的空行补齐。未确认新的 packing 运行缺陷。
- 确认 `trainer/runtime/validation.py::finalize_validation` 的零 batch 诊断仍称 concat-then-split 丢弃不满批次，建议减小 batch；这与当前 repeat=False 保留尾批及已有回归冲突。修正 docstring/错误文字为检查空源及处理过滤后无可用样本，并说明有限文本尾批补齐；不修改异常类型或控制流。
- 专项数据 `pytest -q tests/unit_tests/cpu/datasets/test_data_pipeline.py --disable-warnings`：336 passed、10 warnings、7.14s；验证流程66 passed、3.61s。已有 `test_finite_packing_loader_preserves_all_targets` 覆盖两种packing×document cap，核对独立预期labels全集及valid count；未新增只匹配文案的测试。另用当前真实Grain FlatMapIterDataset将21token/context4拆成[4,4,4,4,4,1]，并读取其实现确认迭代路径不消费 max_fan_out，未将该字段疑虑当作已证实截断缺陷。
- 完整 `pytest -q --disable-warnings`：3608 passed、2 skipped、262 warnings、37.47s、exit0，日志 `/tmp/llmtuner-packing-review-units.log`；适用全仓pre-commit（排除受保护示例/演示文稿）、compileall、diff通过。本批仅说明文字及台账变更，未改collective/布局，不重复分布式实验。
- HEAD `1cacd3e52ba6a4947fc8ddd90fc006d0820557a4` 加累计工作区，已有index保持，未commit/push。本批不扩展为multimodal packing、远程源快照、设备attention边界或多轴数值证明，全局审查继续。

### MULTIMODAL-DATA-CONTRACT-REVIEW-03：媒体、patch 与 mRoPE 边界复核

- 逐文件重读 `datasets/multimodal/{placeholders,mrope,packing,collator,processors,image,video,recipes}.py`，追踪 `HFTransformerModel.preprocess_inputs` 消费边界。未确认新的实现缺陷，未进行无收益搬运或重复测试。占位符数量/非负计数有检查；图像处理失败整样本丢弃，文本/媒体对齐成功后才shift并屏蔽视觉目标；whole-document first-fit打包保留媒体顺序，padding不贡献有效标签。
- 核对 patch temporal 尾帧复制、block/raster两种空间排列、grid计数与merge整除；mRoPE检测相邻image/video转换及文档0位置重置，要求媒体run与grid数目和每run token长度一致。CPU工作线程契约不扩展为设备输入通用工具；视频后端av按需导入，元数据缺失时会全量解码以计帧，不声称有固定内存上限。
- `pytest -q tests/unit_tests/cpu/datasets/test_mrope.py tests/unit_tests/cpu/datasets/test_multimodal_build.py tests/unit_tests/cpu/datasets/test_multimodal_data.py tests/unit_tests/cpu/models/test_multimodal.py --disable-warnings`：210 passed、80 warnings、5.42s、exit0。包含数据构建/懒加载/几何及模型入口拒绝等专项；不是远程OBELICS/CC12M数据集或真实视频codec全链路实测，也不是混合并行VLM训练。
- 支持边界明确：现有HFTransformerModel只调用文本decoder，任何非None pixel_values/video/grid/mrope输入均显式UnsupportedCombinationError，不能把数据层能力宣称为端到端多模态训练；需要媒体消费模型adapter及其训练/恢复证据才可闭合。此为已有显式限制，本批不删除保护或静默丢弃媒体。
- HEAD `1cacd3e52ba6a4947fc8ddd90fc006d0820557a4` 加累计工作区；本批只补台账，复用上一批完整3608 passed/2 skipped质量结果，文档pre-commit/diff按范围检查，未修改index或commit/push。真实decoder/视觉encoder集成、远程媒体及设备矩阵仍保留缺口，全局目标继续。

### CHECKPOINT-PUBLICATION-REVIEW-03：generation 发布与保留契约复核

- 完整重读 `components/checkpointer/filesystem.py`、`storage.py`、`checkpoint_writer.py`、`hf_storage.py`，并专项检查 `base.py::_purge_stale_checkpoints/maybe_wait_for_saving/close` 及对应失败断言。未确认新的实现缺陷，不调整已有generation发布设计。optional torch_checkpointing文件仅检查相关保存/等待/关闭片段，本条不登记为其全部源码复审完成。
- native writer在plan中加入generation前缀，metadata serialization完成后本地os.replace或受支持远程whole-object pipe发布；file://走本地replace。HF先consolidate独立generation，再发布含generation相对路径的index；索引reader依据weight_map读实际分片。保留策略在发布前保护已有窗口、成功后计入新项，resumable与model-only类别独立，当前保存目录不作为abandoned删除；中断且live future仍在时保留owner供重试。
- `pytest -q tests/unit_tests/cpu/components/checkpointer/test_filesystem.py tests/unit_tests/cpu/components/checkpointer/test_checkpointer.py -k 'same_step or rewrite or publication or generation or retention or purge or filesystem' --disable-warnings`：38 passed、97 deselected、58 warnings、1.63s、exit0。已阅读同step原metadata字节保持、model/optimizer/scheduler精确恢复及retry断言；HF consolidation/publication失败后fresh模型读回旧权重；远程发布成功但ack丢失后读回完整新generation。真实local DCP/HF与fsspec memory/file I/O加失败注入，不能替代S3/GCS服务实测。
- HEAD `1cacd3e52ba6a4947fc8ddd90fc006d0820557a4` 加累计工作区。本批仅台账，复用最近完整3608 passed/2 skipped门禁，文档pre-commit/diff检查；未操作index、commit/push。generation旧文件随step目录保留策略清理，不提供并发writer排斥、power-loss durability或reader/retention租约；真实远程存储与optional backend运行缺口保持开放，全局继续。

### PARALLEL-GUARD-STAGES-REVIEW-03：组合拒绝与装配顺序契约

- 完整重读 `parallel/compatibility.py`、`stages.py`、`parallelize.py`、`expert_parallel/expert_parallel.py`，核对组合规则的触发职责与稠密/PP装配分支，未确认新的生产缺陷。config字段/组合校验仍由对应config负责；compatibility表只承载assembly/probe拒绝，不视为全部支持矩阵或完整能力枚举。
- 已检查EP group来自稀疏ep轴，Domino error group来自stage-local loss视图；ep1 fixed_router保持local无WORLD fallback，quantile仍明确要求ep>1。PP先split、各part运行共享STAGE_ORDER，FSDP pattern配置按part解析，变换后重新绑定stages[i].submod再构造schedule，并向last-stage loss传TP词表信息；稠密分支也使用相同stage循环。
- `pytest -q tests/unit_tests/cpu/parallel/test_compatibility.py tests/unit_tests/cpu/parallel/test_assembly_stages.py --disable-warnings`：14 passed、2.60s、exit0。已阅读断言：全部20条规则的异常类型/消息、AST guard路径及函数调用存在、公共入口解析与dense调用顺序（compile开/关）。AST存在性不证明trigger条件完备或正确；测试使用替身的stage调用顺序不等价于真实多轴模型训练。
- HEAD `1cacd3e52ba6a4947fc8ddd90fc006d0820557a4` 加累计工作区；本批只补台账，复用最近完整3608 passed/2 skipped结果，文档pre-commit/diff按范围检查，未操作index或commit/push。固定路由ep1×tp>1、MC2默认FP16梯度、KCP/设备通信及多轴硬件矩阵等既有开放项未由本批闭合；下一步仍需按实际实现与参考收敛，不能仅移除guard以宣称支持，全局未完成。

### FIXED-ROUTER-TP-CONTRACT-02：转换前后布局与基线数值核验

- 重新检查 `config/parallel_config.py` 的fixed_router×ep1×tp>1拒绝，阅读tensor_parallel.py的TPMoeSequenceBoundary/shard_experts_for_tp/realize_moe_blocks_tp/布局注册、tensor_parallel/checkpointing.py全文件，以及expert_parallel/model_replacement.py和block_conversion.py的替换/复制链。确认不能仅移除guard：TP先将HF gate/up逐半按F切片、down按F切片并通过block类mixin包序列gather/scatter；EP随后以新MoE替换整个block，将fused参数改为w1/w3/w2，原mixin及原FQN元数据不能自动保留。
- 后续适配的可核验契约：native MoE保留F-shard而不是复制完整专家；full-stream固定assignment在各TP rank一致；gather/scatter每边只发生一次；shared expert部分输出遵守同一sum边界；更新tp_sharded_param_names及checkpoint layouts到新w1/w3/w2路径，模型/optimizer/EMA全局保存和新运行时恢复；明确aux loss在复制router/完整token流上的梯度缩放。原拒绝保留，适配未实施且未声称支持。
- `pytest -q tests/unit_tests/cpu/parallel/test_tp_moe.py --disable-warnings`：37 passed、2.50s、exit0。两次真实CPU/Gloo：`PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone --nproc_per_node=2 tests/integration_tests/tp_moe_equivalence.py [--router-fp64]`均exit0，含生产TP边界、Trainer复制router梯度归约、两轮输出/梯度、SGD更新、DCP model/Adam状态恢复及后续Adam更新。
- 默认FP32 router的partial-vs-dense input grad最大差3.65791686e-4，脚本以独立partial梯度oracle验证TP，未断言该项与FP64 dense一致；显式FP64 router模式差3.63797881e-12，并通过其dense input grad严格断言。精度诊断只针对该小型HF式routed/shared block，不能作为生产FP32 dense梯度严格等价或fixed_router native转换后的证明；不放宽原断言、不改生产路由dtype。
- HEAD `1cacd3e52ba6a4947fc8ddd90fc006d0820557a4` 加累计工作区；本批仅证据及适配契约，文档pre-commit/diff，未改index/commit/push。现有TP基线可作为后续转换回归参考，fixed EP1×TP仍开放，设备与其他多轴未由本批闭合，全局继续。

### CHECKPOINT-PREFLIGHT-INTERRUPT-01：TP/EP本地准备中断传播

- P1 / `parallel/tensor_parallel/checkpointing.py::_prepare_state`及`expert_parallel/checkpointing.py::gather_expert_state`。本地参数索引/manifest准备只捕获Exception，KeyboardInterrupt/SystemExit绕过既有all_gather_object状态报告，其他rank可能停在metadata或后续tensor collective。新增CPU回归分别覆盖TP save/load和EP save×两类中断：原实现6 failed，报告调用列表均为空；异常对象本身仍抛出，缺陷是没有告知peer。
- 两处纯本地准备边界改为捕获BaseException，EP local_error注解同步。复用既有组状态交换后原样raise原异常，不在捕获区增加collective，不改变tensor布局/数值。通信本身失败、组发现阶段或第二次异步中断不承诺可恢复；EP load的独立DTensor重分发不在本修复范围。
- 扩展既有tp_checkpoint_coordination.py和ep_checkpoint_coordination.py，在任意一侧rank注入两类中断；原rank异常identity保持，peer收到明确preparation failed，EP确认未执行tensor gather；之后健康gather/load通过。命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone --nproc_per_node=2 tests/integration_tests/{tp,ep}_checkpoint_coordination.py` 分别运行均exit0，最终日志 `/tmp/llmtuner-tp-checkpoint-interrupt-final.log`、`/tmp/llmtuner-ep-checkpoint-interrupt.log`。TP首次扩展测试误给load本地shape导致对侧本地ValueError，已改为正确全局shape；首次失败日志 `/tmp/llmtuner-tp-checkpoint-interrupt.log` 保留，未弱化断言。
- 完整 `pytest -q --disable-warnings`：3614 passed、2 skipped、271 warnings、37.65s、exit0，日志 `/tmp/llmtuner-checkpoint-interrupt-units.log`。适用全仓pre-commit（排除受保护示例/演示文稿）、compileall、diff通过，已检查自动格式化。两rank CPU/Gloo仅验证这两种组内准备失败协议，不等价于跨PP/FSDP组或NCCL/HCCL故障恢复。
- HEAD `1cacd3e52ba6a4947fc8ddd90fc006d0820557a4` 加累计工作区。本批两处生产捕获边界、一个新CPU测试文件及两个集成扩展；保留原暂存/其他改动，无commit/push。固定路由TP等原开放项不关闭，全局继续。

### DOMINO-EXPERT-INTERRUPT-01：本地专家中断协调与传输清理

- P1 / `models/common/moe/experts.py`、`parallel/expert_parallel/domino_dispatch.py`。本地计算捕获及CPU/CUDA清理仅捕获Exception，KeyboardInterrupt/SystemExit可绕过posthook后协调或已启动传输的等待。新增回归原实现6 failed、3 passed；失败日志 `/tmp/llmtuner-domino-interrupt-before.log`。
- 在既有纯本地专家捕获及已启动传输清理边界捕获BaseException，错误容器注解同步；协调后保留原异常对象，清理失败作为cause，继续尝试其他传输。不增加collective，不扩大已知通信无关专家体的准入范围。专项33 passed，包含两类中断、两slice、模拟CUDA等待次序，以及清理自身中断后仍等待后续传输。
- 真实CPU/Gloo EP2：`PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone --nproc_per_node=2 tests/integration_tests/ep_domino_equivalence.py` exit0；两slice×两失败rank×ValueError/KeyboardInterrupt/SystemExit共12项，失败侧异常identity、peer明确错误、同runtime后续forward/backward均通过；原输出/梯度与AC数值矩阵仍通过。日志 `/tmp/llmtuner-domino-interrupt-integration-final.log`。首次扩展误改输入错误测试分支，已纠正，失败日志保留为 `/tmp/llmtuner-domino-interrupt-integration.log`，非生产问题。
- 完整 `pytest -q --disable-warnings`：3630 passed、2 skipped、270 warnings、37.78s、exit0，日志 `/tmp/llmtuner-domino-interrupt-units-final.log`。适用全仓pre-commit（排除受保护示例与演示文稿）、compileall、diff检查通过；已审阅生产与测试差异。
- CUDA仅模拟调度验证，未运行CUDA/NPU内核；不承诺通信失败、hook内部失败、第二次异步中断或任意用户专家体的恢复。全局开放项及多轴设备证据缺口仍保留。本批未改index、未commit/push。

### CHECKPOINT-STAGING-INTERRUPT-01：暂存中断完成Future与PyTorch回调兼容

- P1 / `components/checkpointer/distributed_checkpoint.py::CoordinatedStager`。同步stage抛出KeyboardInterrupt/SystemExit绕过协调；异步source.result重抛后仅捕获Exception，导致completion和upload Future未完成。新单测原实现4 failed、2 passed，日志 `/tmp/llmtuner-staging-interrupt-before-final.log`；第一次未将异步测试异常转换为AssertionError，pytest被KeyboardInterrupt中止，保留 `/tmp/llmtuner-staging-interrupt-before.log`，不是完整测试结果。
- 读取本机PyTorch2.14 async_save源码：staging回调只捕获Exception，直接将BaseException注入upload Future会从Future回调逃逸。现有stage和finish边界捕获BaseException；先完成训练线程completion并保留原异常identity，upload侧以RuntimeError并设置原中断为cause，普通Exception保持原对象。单进程也复用包装，避免遗漏回调兼容；不新增阶段collective或改变保存布局。
- 新单测覆盖同步/异步×OSError/KeyboardInterrupt/SystemExit，并以实际ThreadPoolExecutor与Future验证回调完成后线程还能执行下一任务；检查两个完成Future、原异常identity及cause。检查点专项206 passed；包含旧保存、恢复、远程发布回归。
- `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone --nproc_per_node=2 tests/integration_tests/checkpoint_save_coordination.py` exit0，CPU/Gloo、PROCESS、FP32、CPU shared-memory staging：两暂存方式×两失败rank×三异常12种情况，旧checkpoint精确加载、失败无metadata、同manager重试及精确加载通过；同时保留原writer、stager和load-preflight验证。日志 `/tmp/llmtuner-staging-interrupt-integration.log`。设备pinned-memory实际拷贝、NCCL/HCCL与通信内部失败未验证。
- 完整 `pytest -q --disable-warnings`：3636 passed、2 skipped、266 warnings、37.59s、exit0，日志 `/tmp/llmtuner-staging-interrupt-units.log`。适用全仓pre-commit（排除受保护示例/演示文稿）、compileall、diff通过；生产差异及既有集成改动已审阅，原index和其他工作区改动保留，未commit/push。全局目标仍未完成。

### OPTIMIZER-SCHEDULE-RESTORE-REVIEW-04：参数组、调度与EMA契约复核

- 完整重读 `components/optimizer/{optimizer,state_dict,lr_scheduler,ema}.py`及`config/lr_scheduler_config.py`，核对共享别名first-match与canonical FQN、跨part重复检测、Adam零lr材料化/失败回滚、AMSGrad布局、组重合并、scheduler恢复先验证全组后更新及tensor lr identity、EMA固定scope与动态年龄。未确认新的缺陷，不做无收益改名或拆分。
- 读取本地torchtitan HEAD `01b1b69e3881397db5dcd4f96d3c5ffa6a79384d` 的 `torchtitan/components/optim/lr_scheduler.py`。与历史来源路径不同，当前上游已用optim目录、HostLambdaLR缓存host lr，允许total_steps短于run并clamp到末值，默认decay_ratio=None即warmup后立即decay。本仓库默认无warmup/无decay、拒绝短schedule及超界恢复；这些是已有配置与测试约束，不按上游当前默认静默更改。两者均保持optimizer.step后推进scheduler和虚拟末step语义；本仓库从已恢复optimizer initial_lr重建并写回实际lr。
- `pytest -q tests/unit_tests/cpu/components/optimizer --disable-warnings`：264 passed、2 warnings、2.57s、exit0；日志 `/tmp/llmtuner-optimizer-contract-review.log`。阅读现有回归确认覆盖各曲线/边界、恢复后的lr和tensor identity、无部分恢复、AMSGrad双向切换、材料化NaN/overflow防护、EMA继续更新/冷启动/scope变化，并非仅依靠测试数目判定。
- `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone --nproc_per_node=4 tests/integration_tests/pp_optimizer_groups.py` exit0，PP2×DP2、CPU/Gloo/FP32，规则归属、独立AdamW一步更新逐位一致，局部planning/initialization及PP子组失败抵达所有world副本；日志 `/tmp/llmtuner-optimizer-pp-review.log`。该脚本不执行实际PP schedule或多步恢复。
- `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone --nproc_per_node=2 tests/integration_tests/fsdp_checkpoint_equivalence.py --process-save-cpu` exit0，CPU/Gloo/FP32、native/custom×shard/replicate保存恢复8种组合，真实PROCESS共享CPU暂存、两步checkpoint后失败同step覆盖、全新runtime恢复并继续至6步。已检查脚本使用独立dense AdamW与手算线性lr：恢复更新前model/完整optimizer state逐位一致；同布局继续训练atol0，跨布局及dense oracle atol1e-5；loss、梯度、权重、moments/counters和lr均检查。日志 `/tmp/llmtuner-optimizer-fsdp-review.log`。
- 本批仅审查证据与上游差异文档，不修改生产/测试及index；无commit/push。设备fused/capturable、host lr缓存性能、真实PP训练恢复及其他多轴仍需各自证据，不能由CPU标量lr或单FSDP轴推断。全局目标继续。

### VALIDATION-TRAINING-STATE-REVIEW-04：实际PP验证的训练状态隔离

- 完整复核 `trainer/runtime/validation.py`、`utils/seeding.py`，追踪validate等待staging、临时loader创建/关闭、fetch耗尽、DP有效token与loss mesh求和、PP eval分母1、模式/attention mask/flex packed恢复及RNG隔离。未确认新的实现缺陷；任意模型forward或hook通信失败不被本地错误协调泛化包裹。
- 现有 `pp_validation_equivalence.py`已比较生产Trainer的schedule.eval与合并stage权重后的非PP参考loss，但缺少更新状态不变断言。本批扩展该脚本：每次validate前deepcopy完整FQN Adam状态与scheduler进度、训练token计数、各参数现存gradient（含None），验证后精确比较（rtol0/atol0），保留原loss/模式检查。不修改生产逻辑、不更改容差。
- `pytest -q tests/unit_tests/cpu/test_validate.py --disable-warnings`：66 passed、6.11s、exit0，日志 `/tmp/llmtuner-validation-contract-review.log`。已检查覆盖成功/失败RNG隔离、混合子模块模式、mask恢复、空数据/全masked/非有限loss、loader部分初始化与失败清理；并非每个分布式故障的证明。
- 真实 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --standalone --nproc_per_node=2 tests/integration_tests/pp_validation_equivalence.py` exit0，CPU/Gloo、FP32、Qwen3、PP2/1F1B/2microbatch、其余轴1、AC/compile关闭，2训练step各验证2batch；新增状态不变断言通过。reported=[4.785974317981351,4.706551828692036]，独立full-model参考=[4.785974317981351,4.7065513364730345]，最大绝对差4.922e-7<1e-5。日志 `/tmp/llmtuner-validation-pp-state-review.log`；此前未扩展版复跑亦通过。环境变量用于本机Inductor PCH兼容，不代表启用model compile。
- 真实 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone --nproc_per_node=4 tests/integration_tests/validation_reduction_equivalence.py` exit0，DP2×CP2及DP2×TP2的显式sequence切片投影，验证循环归约与独立全batch CE符合rtol/atol1e-7；loader关闭、训练token不变和混合模式恢复通过。日志 `/tmp/llmtuner-validation-reduction-review.log`。该脚本未应用真实TP/CP attention变换，不能证明实际多轴模型数值。
- 全量 `pytest -q --disable-warnings`：3636 passed, 2 skipped, 259 warnings in 40.22s、exit0，日志 `/tmp/llmtuner-validation-state-units.log`；适用全仓pre-commit（排除受保护示例/演示文稿）、compileall、diff检查通过，已审阅测试差异。
- 本批仅扩展集成断言和台账，保留原index及其他工作区改动，无commit/push；设备、其他PP调度、EMA验证权重选择、通信内部失败与实际多轴仍不由本批闭合，全局继续。

### CLI-FRESH-PROCESS-RESUME-REVIEW-04：真实入口跨进程恢复

- 完整复核 `__main__.py`、`trainer/cli.py`、`config/cli_config.py`、`config/llmtuner_config.py`，核对console script指向同一main、禁用隐式args文件、隐藏嵌套字段及base config重建、seed输出前置检查、延迟Trainer导入与原始错误优先的资源/进程组清理。本批未确认新的入口缺陷，不修改生产行为。
- 新增 `tests/integration_tests/cli_checkpoint_equivalence.py`：用实际 `python -m llmtuner` 启动三个独立子进程，连续3步 vs 先2步/新进程恢复到3步；两侧total_steps3和decay_ratio1一致，避免首段schedule不同。离线Qwen3、vocab32/hidden16/intermediate32/2layers/2heads、seq8/batch2、seed42、deterministic、单CPU/eager、native DCP完整最终保存。不是Trainer对象模拟或同进程load。
- `PYTHONPATH=. python tests/integration_tests/cli_checkpoint_equivalence.py` exit0；三子进程均exit0，resume日志确认step2，最终模型/完整optimizer/scheduler/data cursor/train_state与解码RNG递归精确一致（tensor rtol0/atol0、metadata严格类型和值）。train_state={step:3,ntokens_seen:48}，scheduler={last_epoch:3}。最终日志 `/tmp/llmtuner-cli-checkpoint-equivalence-final.log`，完整命令各项见脚本，artifact目录 `/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-cli-resume-wd_6rwq_` 保留各子进程日志、checkpoint与转换后的pt。
- 首次手动比较发现RNG pickle字节长度11996 vs11994；原始编码含storage标识不能作随机流等价判定。改为pickle解码，NumPy uint32数组无损转tensor，与Python/CPU/device/topology元数据一起比较。第一次脚本又暴露torch.testing.assert_close不支持字符串元数据；已增加带差异路径的递归严格比较器，保留 `/tmp/llmtuner-cli-checkpoint-equivalence.log`。这些是比较方法错误，不作为恢复实现缺陷，也未丢弃RNG字段或放宽容差。
- 全量CPU `pytest -q --disable-warnings`：3636 passed, 2 skipped, 280 warnings in 41.35s、exit0，日志 `/tmp/llmtuner-cli-entry-units.log`；适用全仓pre-commit（排除受保护示例/演示文稿）、compileall、diff通过，已审阅新增脚本。
- CPU专用脚本明确拒绝非CPU自动选择环境，不声称GPU/NPU、分布式、EMA或compile已验证。本批仅新增集成脚本及台账；检测到外部操作将更多累计生产改动暂存，当前agent未改index、未commit/push，保留现状。全局目标继续。

### META-MATERIALIZATION-FREEZE-REVIEW-04：材料化别名与冻结权重复核

- 完整重读 `models/hf/materialization.py`、`factory.py`，追踪`trainer/runtime/assembly.py::_build_model/_apply_parallelism`：HF初始加载选择meta构造，既定并行阶段后to_empty分配本地存储，再重建RoPE/MoE非持久buffer、保留real buffer和共享别名，最后构造optimizer。未知非持久meta buffer先拒绝，不允许未初始化运行状态进入训练。本批未确认新生产缺陷。
- 现有材料化回归已检查精确RoPE/输出、persistent buffer checkpoint填充、real buffer保留、外部RoPE别名和共享parameter梯度。本批将共享parameter测试扩展为frozen=False/True，分别确认同对象、requires_grad保持、相同前向以及反向梯度（冻结时均None）；`pytest -q tests/unit_tests/cpu/models/test_hf_wrapper.py --disable-warnings`：81 passed、1 warning、11.39s、exit0，日志 `/tmp/llmtuner-materialization-review-units-final.log`。新增1种边界，未拆分或重命名生产模块。
- 实际Trainer HF路径：`PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 LLMTUNER_TEST_KIMI=1 LLMTUNER_TEST_KIMI_MOE=1 LLMTUNER_TEST_KIMI_LATENT=1 LLMTUNER_TEST_EP_SIZE=2 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom torchrun --standalone --nproc_per_node=2 tests/integration_tests/kimi_hf_initial_load.py /var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-meta-hf-review-qrgqtuqf` exit0。CPU/Gloo/FP32、EP2、dp_shard2/efsdp1、TP/PP/CP1、alltoall、latent projections、AC/compile关；日志 `/tmp/llmtuner-materialization-integration.log`。
- 已检查脚本实际断言：原始HF safetensors加载后导出所有权重逐位一致，logits与原始非并行模型rtol/atol1e-6，训练2步后native checkpoint与HF model-export并存，旧文件hash及完整运行状态不变，导出权重逐位一致、再前向对照及继续训练1步通过。这不是冻结共享权重×EP的组合证明，两个场景证据独立。设备RoPE/DTensor、未知架构、fused内核仍需各自验证。
- 完整CPU `pytest -q --disable-warnings`：3637 passed, 2 skipped, 275 warnings in 42.78s、exit0，日志 `/tmp/llmtuner-meta-freeze-units.log`。
- 本批保留已有工作区和外部index；未commit/push。适用全仓pre-commit重新初始化并安装配置中的环境后exit0（排除受保护示例/演示文稿），compileall和diff通过，已审阅测试改动。全局未完成。

### GRAD-NORM-PP-COMBINATION-REVIEW-04：跨PP的TP/EP范数独立对照

- 完整复核 `accelerator/collectives.py::clip_grad_norm_`、`trainer/runtime/reductions.py`以及Trainer的TP复制梯度归约/裁剪/finite判定/optimizer更新顺序。密集复制参数计一次、TP物理分片与EP专家分别归约，PP合并各stage的p次幂；低精度norm先提升FP32，DTensor norm full_tensor后裁剪。未确认本批新的生产缺陷。
- 扩展 `tests/integration_tests/parallel_grad_norm_equivalence.py`，保留原world2场景，增加world4的PP2×TP2/PP2×EP2：每stage一个内轴复制dense梯度与两个不同物理分片，以独立float64全向量[3,4,5,6,7,8]计算L1/L2/L∞范数，检查每个本地梯度裁剪系数；rtol1e-14/atol0，避免把复制参数重复计数隐藏在分片合成中。一个物理分片注入NaN，所有rank必须抛nonfinite且所有gradient不变（含NaN逐项比较）。
- `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone --nproc_per_node=4 tests/integration_tests/parallel_grad_norm_equivalence.py` exit0，两组合/三norm及NaN拒绝通过，日志 `/tmp/llmtuner-grad-norm-combinations.log`。同命令nproc2 exit0，原FP16大范数、foreach双路径、BF16/FP16真实DTensor和空stage/无参数peer全部通过，日志 `/tmp/llmtuner-grad-norm-single-axes.log`。
- `pytest -q tests/unit_tests/cpu/accelerator/test_collectives.py --disable-warnings`：13 passed、1.11s、exit0，日志 `/tmp/llmtuner-grad-norm-unit-review.log`。本批测试使用真实Gloo mesh与collective，但梯度由显式赋值产生，不能当作生产TP/EP/FSDP模型变换、Trainer全局finite机制、GPU/NPU kernel或通信故障恢复的证明。Trainer默认norm2及停止更新边界经静态追踪，不扩大此实验结论。
- 完整CPU `pytest -q --disable-warnings`：3637 passed, 2 skipped, 262 warnings in 40.06s、exit0，日志 `/tmp/llmtuner-grad-norm-combination-units.log`。
- 期间外部提交将累计实现/测试变更纳入HEAD，当前 `a488490e07704af32644d3ea444100c8a2498f67`；本agent仅编辑上述集成测试及文档，未执行stage/commit/push，保留外部提交。适用全仓pre-commit（排除受保护示例/演示文稿）、compileall、diff通过，已审阅测试差异。全局开放项继续。

### MC2-TP-CURRENT-STATE-RECHECK-02：当前提交仍未通过FP16梯度验收

- 关联MC2-TP-PUBLIC-01/MC2-TP-HF-GRADIENT-MAP-01，当前HEAD `a488490e07704af32644d3ea444100c8a2498f67`。重读 `parallel/expert_parallel/{mc2_dispatch,mc2_ops}.py`及token dispatcher本地reorder/count交换，核对FSDPTurbo HEAD `0af4eca242c14ebff254dab1a38045f1395fa520` 的 `fsdp_turbo/ops/npu/grouped_matmul_mc2.py`。权重转置、send/recv逆序、rank-major count按expert汇总、grouped权重梯度与上游一致；共享专家独立计算为既有适配差异。源码一致不足以证明数值或设备正确。
- 实际 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone --nproc_per_node=2 tests/integration_tests/mc2_tp_ep_equivalence.py` exit1，日志 `/tmp/llmtuner-mc2-tp-current-review.log`。先等待Inductor调用brew --prefix libomp产生的API下载，确认torchrun/worker/brew进程存活并持续poll原session36407，未重复启动；最终失败是既有逐参数梯度断言，不是启动环境错误。
- CPU/Gloo、TP2/EP2重叠轴、FP16、生产apply_tp/apply_ep和复制梯度归约、CPU MC2算子替身；rank0五键/rank1七键仍失败，包含embedding和第一层q/k/v/o，部分第二层投影。保留atol5e-4/rtol1e-2；例如rank0第一层q最大绝对差0.0023193359375，v为0.002593994140625。未修改实现/容差，未将前向通过或既有FP64平均误差分析当作关闭依据。
- `pytest -q tests/unit_tests/cpu/models/test_mc2_ops.py --disable-warnings`：9 passed、2.02s、exit0，日志 `/tmp/llmtuner-mc2-ops-review.log`；只证明adapter代数、空expert及依赖检查，不能替代本次失败或NPU验收。MC2-TP-PUBLIC-01继续开放；后续须取得真实Ascend融合内核的完整梯度/更新对照，并针对融合与分支低精度舍入差异建立独立数值契约，不能简单改成普通分支计算绕过融合机制。
- 本批仅更新台账和矩阵，不改生产/测试及index，无commit/push。目标仍未完成；其他可运行审查工作继续，缺NPU不作为整个目标阻塞。

### KCP-MLA-MASK-REVIEW-04：布尔与additive因果padding对照

- 完整重读 `parallel/context_parallel/kcp.py`，复查KDA本地输入校验、跨rank sequence/kernel/head几何一致性、runtime懒加载、共享cp_context、conv与chunk_kda参数，以及MLA Ulysses mask恢复/全局重建。KDA显式拒绝padding/cache/packed不是完整迁移；`KCP-PADDING-STATE-01`仍开放。本机无fla/Ascend KDA runtime，不能设计未经runtime契约确认的compact状态交换作为已实现支持。
- 扩展 `tests/integration_tests/kimi_mla_cp_equivalence.py`：原q_lora=None/4×gate关/开×padding关/开8场景再覆盖布尔mask与0/-inf additive mask，共16场景。两侧分别从同一有效位置构造局部/完整因果mask，独立full eager MLA参考，生产kcp_mla_attention执行真实SeqToHead/HeadToSeq与mask all_gather；保持原输出1e-6、input grad2e-6、参数grad3e-6的atol/rtol，不放宽断言。
- CPU/Gloo、FP32、CP2、其余并行轴1、AC/compile关：`PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone --nproc_per_node=2 tests/integration_tests/kimi_mla_cp_equivalence.py`通过首次16组合及原rank局部非法mask/KDA padding/不等长/几何不一致拒绝，日志 `/tmp/llmtuner-kcp-mla-mask-review.log`。去掉重复reference forward后以同命令复跑最终版本，最终exit0，日志 `/tmp/llmtuner-kcp-mla-mask-review-final.log`。
- `pytest -q tests/unit_tests/cpu/parallel/test_kcp_runtime_contract.py --disable-warnings`：6 passed、2.84s、exit0，日志 `/tmp/llmtuner-kcp-contract-review.log`；mock只证明kernel参数传递及拒绝前置，不证明KDA状态通信或实际设备内核。实际MLA padding可用不等价于混合KDA模型padding可用。
- 完整CPU `pytest -q --disable-warnings`：3637 passed、2 skipped、283 warnings、42.49s、exit0，日志 `/tmp/llmtuner-kcp-mask-full-units.log`；适用全仓pre-commit（排除受保护示例/演示文稿）、compileall及diff通过，已审阅新增测试差异。
- 本批仅集成mask覆盖与证据，未修改生产/index，无commit/push。NPU/CUDA、dropout随机性、KDA compact/state与实际多轴模型继续开放，全局目标未完成。

### PROFILER-LIFECYCLE-REVIEW-04：真实CLI trace与恢复步号

- 完整重读 `components/profiler.py`及`utils/garbage_collection.py`，复核`utils/monitoring.py` allocator history接口和Trainer循环Profiler作用域。检查trace部分启动失败清理、OOM cause/context循环检测、快照临时文件原子发布、memory history共享owner及失败关闭重试、GC多owner与恢复原状态。未确认本批新缺陷，不作无收益重组。
- `pytest -q tests/unit_tests/cpu/components/test_profiler.py --disable-warnings`：63 passed、1 warning、1.26s、exit0，日志 `/tmp/llmtuner-profiler-lifecycle-review.log`。阅读断言确认覆盖嵌套/reuse拒绝、初始化重试、snapshot频率/协议/失败不发布、trace导出、OOM保留与清理失败等；backend mock不当作设备证明。
- 实际两个独立 `python -m llmtuner` 运行，先steps2后从完整checkpoint恢复至steps4；共同参数 `--model_name_or_path qwen3 --vocab_size 32 --hidden_size 16 --intermediate_size 32 --num_hidden_layers 2 --num_attention_heads 2 --num_key_value_heads 2 --max_seq_len 8 --global_batch_size 2 --enable true --last_save_model_only false --total_steps 4 --enable_profiling true --profile_freq 2 --profiler_warmup 0 --profiler_active 1`，dump_folder为 `/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-profiler-review-c82s_8ko`。两子进程exit0，resume日志确认step2；日志与checkpoint/trace保留于该目录，路径索引 `/tmp/llmtuner-profiler-review-root.txt`。
- 解压并json解析实际 `profiling/traces/iteration_2/rank0_trace.json.gz`与iteration_4，均3869 traceEvents、2950 aten::算子事件，确认真实训练活动而非空文件，恢复步号未重置。CPU/eager、并行轴1，CPU allocator无history；不推断CUDA/XPU/NPU trace、设备OOM或快照可视化结果。profile_freq2/warmup0仅用于短实验，不是性能建议或基准。
- 本批仅审查台账变更，无生产/测试/index变更，无commit/push；纯文档检查内容、命令、格式与diff。全局开放项继续。

### LOADER-LIFECYCLE-REVIEW-05：数据加载游标、预取与恢复失败清理

- 范围：完整阅读 `llmtuner/datasets/dataloader.py`、
  `llmtuner/datasets/loader_builder.py`、`llmtuner/datasets/contracts.py`，
  续接 synthetic loader 审查；核对数据管线及 trainer 的相关恢复/清理测试。
- 当前实现：builder 负责排序、重复与 DP ownership，loader 只负责 batch、
  collation、预取及游标。恢复前验证版本、DP 度数、rank、batch layout 和可选
  corpus identity；元数据不匹配不触碰游标，底层 set_state 失败则关闭迭代器，
  清理失败保留原始异常。trainer 保留关闭失败的 loader 供再次清理。
- 依赖核对：实际安装的 Grain ThreadPrefetchDatasetIterator 在 __next__ 取出
  批次时更新消费者状态，get_state 返回该状态；set_state 先停止预取再恢复 parent。
  因此未将生产者已预取但尚未消费的位置作为恢复位置。close 停止预取并关闭 parent。
- 验证：`pytest -q tests/unit_tests/cpu/datasets/test_data_pipeline.py
  --disable-warnings`，退出码 0，336 passed、10 warnings、7.15 秒；日志
  `/tmp/llmtuner-loader-lifecycle-review.log`。既有测试覆盖后续批次一致、有限
  packing 目标完整、DCP layout/identity 拒绝及恢复失败时原始异常保留。
- 结论：本批未确认新增实现缺陷，未修改生产代码。结果限于 CPU 数据管线；
  repeat=False 的多 DP 数据源仍由明确 guard 拒绝，不据此宣称已支持分布式有限
  数据耗尽策略。未重新执行完整训练或设备组合验证。

### STREAM-DP-RESUME-REVIEW-05：HF 流式分片与窗口恢复

- 完整复核 `datasets/sources.py`、`datasets/composition.py`，以及已存在的
  HF cursor 生命周期、epoch/reset 失败与窗口 shuffle 恢复测试。JSONL 索引按
  pattern 顺序及每组排序建立 offset，拒绝重复 resolved path；map source 在
  全局 shuffle 后分片，HF streaming 在窗口 shuffle 前通过 HF 分片。
- 新实验使用实际 `datasets.IterableDataset.from_generator` 生成 id=0..22，
  调用实际 HF split_dataset_by_node，依次遍历 DP=2/3/4 的全部 rank 坐标。
  每个有限分片的排序结果严格等于独立参考 `range(rank,23,degree)`，所有
  分片合并后严格等于 0..22，无重复、无遗漏。
- 对每个坐标构造 repeat=True 的实际 HuggingFaceStreamingIterDataset 与
  Grain WindowShuffleIterDataset(window_size=5,seed=42)。消费 11 条后保存，
  新游标恢复，随后 41 条逐条严格等于连续读取结果，并全部属于该 rank 分片。
  共验证 9 个坐标，跨多个 repeat 边界；所有游标显式关闭。
- 命令：本地 `python` 实验，退出码 0；日志
  `/tmp/llmtuner-stream-dp-resume-review.log`，DP=2/3/4 均报告通过。
  未修改生产代码。本实验使用本地真实 HF/Grain，不依赖网络或 mock；在单进程
  逐个模拟数据坐标，没有进程组，不能证明 rank 间耗尽协调或远端数据精确恢复。
- 结论：本轮未确认新增实现缺陷。有限多 DP loader 的显式拒绝策略保持不变；
  上游自带 shuffle buffer、远端 source 状态覆盖及设备通信仍需分别验证。

### PACKING-STATE-MATRIX-REVIEW-05：文档容量与余量游标矩阵

- 完整复核 `datasets/packing/iterators.py`、`conversions.py` 和 `build.py`：
  文档/上下文切分、余量 parent state 与 offset、padding 标签、有限尾行以及
  恢复失败回滚。复用已有 malformed state/interrupt 测试证据，无新增确认缺陷。
- 新 CPU 实验：12 个确定性 seed，每组 9 个长度在 0..24 的文档，包含空文档；
  context=1/3/8、row tokens=1/5/16、document cap=1/2/5，共 324 个配置。
  使用实际 DocumentAwareConcatThenSplitIterDataset 与 Grain map iterator，
  将全部非 IGNORE_INDEX 标签与独立生成的原始标签序列逐项比较，确认保序且
  无漏/重复 token。实验覆盖行长小于及大于 context 的直接迭代器输入。
- 每配置在初始、中间、全部输出完成的位置保存，新游标恢复并比较全部后续
  input_ids/labels/positions/padding_mask，均为数组精确相等；共 972 次恢复。
  所有游标显式关闭，命令 `python` 实验最终退出码 0，日志
  `/tmp/llmtuner-packing-state-matrix.log`。
- 范围：这是本地 packing 状态机及目标序列完整性证据，未经过 tokenizer、
  collator、trainer，也不证明设备并行或 arbitrary upstream source 的恢复能力。
  未修改生产代码；first-fit 的上游 FlatMap 路径不消费 max_fan_out，已有实际
  长文档验证，不将该声明字段推断为截断缺陷。

### TEXT-SFT-LABEL-REVIEW-05：文本 shift 与 SFT 监督闭环

- 完整复核 `datasets/text/processors.py`、`text/recipes.py`、`text/renderer.py`
  及 `datasets/collators.py`：普通文本 shift、单轮 prompt token prefix 检查、
  EOS 去重/补齐、超长样本拒绝、renderer mask 与 labels 同步 shift、padding
  屏蔽、有效 token 统计和 PP 有限尾部补空行。未确认新增实现缺陷。
- 测试命令：`pytest -q tests/unit_tests/cpu/components/test_chat_renderer.py
  tests/unit_tests/cpu/datasets/test_data_pipeline.py --disable-warnings`，最终
  退出码 0，355 passed、10 warnings、7.27 秒；日志
  `/tmp/llmtuner-text-label-review.log`。
- 独立标签实验：本地 JSONL 三样本，响应分别 w2/w3、w5、空串，prompt 使用
  不同 token；实际本地 WordLevel tokenizer 与显式前缀模板，经 build_dataloader
  完成单轮处理、packing、collation、thread prefetch。concat_then_split/first_fit
  各结合 document cap=None/1/2，共 6 配置，有限遍历。有效标签多重集合严格
  等于独立参考 {w2,w3,w5,EOS,EOS,EOS}，每组有效数 6，padding 标签全为
  IGNORE_INDEX；所有 loader 显式 close。first-fit 允许重排，故不要求标签顺序。
- 实验最终退出码 0；日志 `/tmp/llmtuner-sft-label-oracle.log`，资产位于
  `/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-sft-oracle-pbpl45s4`。
  解释器退出阶段出现 Grain DatasetIterator.__del__ 模块销毁 AttributeError，
  不计作资源清理验证成功；本项标签断言均在退出前完成。
- 限制：renderers 实际依赖仍未安装，上述该路径单测使用明确标注的 fake 接口，
  不证明真实库 token/mask 等价；该缺口保持开放。本实验未执行 trainer backward
  或设备并行，也不将简单前缀模板推广为任意 BPE/chat template 可用。

### MULTIMODAL-PATCH-REVIEW-05：媒体预处理、packing 与独立卷积参考

- 完整复核 `datasets/multimodal/{image,video,placeholders,processors,recipes,
  packing,collator,mrope}.py`：媒体解码、预算缩放、时间补齐、视觉占位符、标签
  屏蔽、whole-document packing、媒体列表展开、grid/MRoPE 对齐与文档重置。
  本轮未确认新增缺陷，未改动生产实现。
- 专项命令 `pytest -q tests/unit_tests/cpu/datasets/test_multimodal_data.py
  --disable-warnings`，退出码 0，180 passed、20 warnings、2.54 秒；日志
  `/tmp/llmtuner-multimodal-review.log`。真实 PNG/CPU resize 与 packing 路径
  已执行；视频 decode 的部分测试通过 fake PyAV seam，不计真实视频解码证据。
- 新独立数值参考：frames=1/2/3/5，spatial patch=1/2，temporal patch=1/2/3，
  merge=1/2，block/raster 两顺序，共 96 场景。使用确定性 float64 整数像素和
  卷积权重，独立补齐最后帧后通过 torch.nn.functional.conv3d 计算参考；实际
  vision_to_patches 结果乘 flatten kernel，经显式空间坐标列表对应到参考。
  全部输出 atol=0/rtol=0，patch 数严格等于 vision token 数乘 merge²。
- 实验 `python` 最终退出码 0；日志 `/tmp/llmtuner-vision-conv-oracle.log`。
  该参考独立于 einops 的 reshape 表达式，验证 patch 向量 C/pt/ph/pw 排列、
  block/raster 序列顺序及时间补齐；不能据此证明 VLM trainer、设备 kernel、
  原始视频解码或网络媒体读取已验证。

### RNG-LIFECYCLE-REVIEW-05：默认随机流、嵌套上下文及中断

- 完整复核 `utils/seeding.py`、`components/checkpointer/random_state.py`、
  `trainer/runtime/validation.py`，追踪 assembly DTensor tracker 初始化与 trainer
  rollback 上下文调用。host checkpoint 在改变默认 RNG 前通过私有 generator
  验证状态；validation 保留默认随机流并恢复混合 module mode、mask 和 kernel
  packed 策略。未确认新增缺陷，未修改生产实现。
- 命令 `pytest -q tests/unit_tests/cpu/test_pp_distinct_seed.py
  tests/unit_tests/cpu/components/checkpointer/test_random_state.py
  tests/unit_tests/cpu/test_validate.py --disable-warnings`，最终退出码 0，
  120 passed、4 warnings、4.65 秒；日志 `/tmp/llmtuner-rng-lifecycle-review.log`。
- 新实际 CPU 实验：seed=0/-1/-2**63/2**64-1，各结合 RuntimeError、
  KeyboardInterrupt、SystemExit，共 12 场景。嵌套 preserve_rng_state 内消费
  Python/NumPy/Torch 默认流后中断，退出内层与外层分别严格对照保存状态的下一
  随机输出，浮点标量相等、13 元素 torch tensor bitwise 相等。使用实际
  RNGCheckpointState 保存/恢复参考；无 mock。最终退出码 0，日志
  `/tmp/llmtuner-rng-nested-oracle.log`。
- 范围：默认 host RNG 已验证，不覆盖 loader 私有 generator、设备 generator
  或真实 DTensor tracker；tracker 单测 mock 仅验证初始化参数，不能替代 CUDA/NPU
  运行证据。assembly rollback 的设备随机流不在当前文档契约内，不据本轮宣称
  已恢复设备状态。嵌套实验仅在独立子进程中运行，不改变工作进程策略。

### TENSOR-TREE-HF-MAPPING-01：修复 HF ModelOutput 搬运失败

- 优先级 P2；路径 `accelerator/tensor_tree.py::_move`，调用方为 collective
  staging 的 tensor tree 搬运与公开 accelerator API。完整复核该文件及
  `utils/{dtypes,module_names,lazy_exports}.py`，未为已清晰的工具额外拆分。
- 复现：真实 transformers BaseModelOutput(last_hidden_state=torch.ones(2))
  调用 move_tensor_tree(...,'meta')，在 result.update(items) 抛出
  `You cannot use update on a BaseModelOutput instance`。Mapping 输入符合
  声明契约，但现有实现假设所有 MutableMapping 支持 clear/update。
- 修复：保留浅副本原有键，通过逐键 __setitem__ 写入搬运后的值；既不调用
  受限 update，也让 ModelOutput 同步同名属性，无 transformers 生产依赖。
  原始容器及原始张量不变，defaultdict/namedtuple 既有契约保留。
- 回归 `test_huggingface_output_preserves_mapping_and_attribute_access` 使用
  实际 BaseModelOutput 与 CPU→meta 搬运，验证类型、嵌套 hidden_states、
  字典/属性身份一致、可选 attentions=None 和原始 CPU 张量身份未变化。
- 修复前专项工具测试 95 passed（1.20 秒）；修复后完整
  `pytest -q --disable-warnings` 最终退出码 0，3638 passed、2 skipped、
  273 warnings、40.38 秒，日志 `/tmp/llmtuner-mapping-fix-full.log`。
  全部适用文件 pre-commit 通过（排除受保护 examples/native_alltoall.py 与
  pptx），日志 `/tmp/llmtuner-mapping-fix-gates.log`；compileall 与 diff check
  通过，已检查自动格式化与源码差异。
- 状态：此缺陷已修复并通过 CPU 回归；meta 用于检验设备及结构变化，不证明
  accelerator collective 的设备传输或 autograd 组合正确。未 commit/push。

### ALL-REDUCE-STRIDED-01：修复非连续视图静默归约错误

- 优先级 P1，`accelerator/collectives.py::all_reduce`。完整复核 collectives
  与 distributed bootstrap/query/comm device；发现同设备 .to() 不会把 strided
  view 转为连续存储，底层 Gloo 归约以连续存储读取，导致错误值写入相邻元素。
- 修复前真实两 rank CPU/Gloo：各 rank 对 storage[:,0] 归约，输入分别 [1,2]
  与 [2,3]，独立预期 sum=[3,5]；实际第二元素 rank0 最大绝对差 3、rank1 差 2，
  进程最终退出码 1，日志 `/tmp/llmtuner-collective-contract-review.log`。
- 修复：comm device 搬运后显式 contiguous()，完成通信再通过原 view 回写，
  保留原 stride、storage 和未参与通信的相邻列。连续输入不额外复制。
  新 CPU 单测断言通信缓冲连续及相邻列不变，真实集成脚本
  `tests/integration_tests/collective_contract_equivalence.py` 固化该回归。
- 命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone
  --nproc_per_node=2 tests/integration_tests/collective_contract_equivalence.py`，
  修复后最终退出码 0；日志 `/tmp/llmtuner-collective-contract-repo.log`。
  sum/mean/product/min/max 使用 float64 独立固定参考，atol=0/rtol=0，相邻列
  严格保持 0；另验证 PartialToInvariant 前向值 3、反向梯度 6，以及 rank1
  ValueError/KeyboardInterrupt/SystemExit 的原异常身份、peer 错误和后续归约可用。
- 完整 CPU `pytest -q --disable-warnings`：3639 passed、2 skipped、
  284 warnings、40.26 秒，最终退出码 0；日志
  `/tmp/llmtuner-strided-reduce-full.log`。适用文件 pre-commit、compileall、
  diff check 通过；排除受保护 native_alltoall 与 pptx，自动变动已审阅。
- 状态：已修复并通过真实 CPU/Gloo 回归；不将 Gloo 结果推广为 NCCL/HCCL
  设备验证。无 collective 内故障恢复承诺。未 commit/push。

### LOGGING-CLOSED-HANDLER-01：修复关闭后重新配置导致文件日志丢失

- 优先级 P2，`utils/logging.py::{FileHandler,get_logger}`。完整复核日志实现
  与调用方，原有 closed handler 回归在关闭前 removeHandler，漏掉正常 close/
  shutdown 后 handler 仍挂在 logger 或共享 child 上的场景。
- 真实本地复现：get_logger(...,file_mode='w') 写 first，关闭 file handler
  但不移除，重新配置同路径后写 second；控制台有 second，文件无 second。
  原因是旧 handler 的 DestinationFilter 先标记已发送，但标准库关闭后的
  w handler 不再 emit，新 handler 又因去重跳过，记录被静默丢弃。
- 修复：自有 FileHandler 在已关闭时不运行目的地过滤器，避免占用去重标记，
  也避免 append handler 被标准库自动重新打开；get_logger 清理当前 logger
  已关闭的自有 file handler。其他 logger 仍持有的旧共享 handler 安全跳过，
  应用自有 handler 保持不变。
- 新增 a/w 两场景真实文件回归：parent/child 共享旧 handler，关闭后只重新
  配置 parent，child 发出的记录必须写入新目标一次；a 保留历史，w 截断历史，
  旧 stream 保持 None。专项 30 passed、0.96 秒，最终退出码 0，日志
  `/tmp/llmtuner-logging-closed-targeted.log`。
- 完整 `pytest -q --disable-warnings` 最终退出码 0，3641 passed、2 skipped、
  289 warnings、39.75 秒；日志 `/tmp/llmtuner-logging-closed-full.log`。
  全适用文件 pre-commit、compileall、diff check 通过，已审阅源码及自动修改；
  排除受保护 native_alltoall 与 pptx。未 commit/push。
- 状态：关闭后顺序重新配置契约已修复；不宣称 close 与 emit 任意并发交错或
  多进程共享同一文件的写入原子性已验证。

### OPTIMIZER-FQN-ORDER-REVIEW-05：参数重排后的完整更新等价

- 完整复核 `components/optimizer/state_dict.py` 与 `optimizer.py`：lazy state
  materialization 的梯度/学习率/eps/decay 回滚、FQN 重键与冲突检查、组配置
  合并、AMSGrad 状态布局、所有 inner state 预验证，以及共享/冻结参数分组。
  未确认新增实现缺陷；未为已有职责明确的 state_dict 工具额外拆分。
- 专项命令 `pytest -q tests/unit_tests/cpu/components/optimizer
  --disable-warnings`，最终退出码 0，264 passed、2 warnings、2.76 秒；日志
  `/tmp/llmtuner-optimizer-state-review.log`。
- 新实际 CPU float64 实验：Adam/AdamW × AMSGrad off/on × foreach off/on，
  共 8 场景。每组 3 个 trainable 参数及 1 个 requires_grad=False 参数，两个
  不同学习率 group。先连续更新 3 步，deepcopy flat checkpoint；新 optimizer
  反转 group 顺序和组内参数顺序，设置不同初始 lr，materialize 后按 FQN 恢复。
  随后三次更新与原 optimizer 参数逐步 atol=0/rtol=0，最终全部 flat state tensor
  精确相等、非 tensor 配置相等，key 集合一致。实验中的 tied 仅为参数名称，
  不据此宣称此实验覆盖共享参数；共享别名证据仍来自既有分组回归。
- 命令本地 `python`，最终退出码 0；日志
  `/tmp/llmtuner-optimizer-fqn-order-oracle.log`。这是实际 optimizer/FQN codec
  验证，不经过 DCP、trainer 或 EP/TP 适配，不替代设备/分布式恢复证据。
- 本轮仅补充台账；完整代码门禁复用前一批当前实现的 3641 passed/2 skipped，
  未重复执行全量测试。文档内容、pre-commit 与 diff check 检查通过。

### EMA-DCP-ALIAS-REVIEW-05：共享参数、复数 buffer 与恢复后的连续更新

- 完整复核 `components/optimizer/ema.py`、`config/ema_config.py`，追踪
  trainer.step 的调用顺序及 DCP/torch_checkpointing 冷启动逻辑。EMA 是独立
  state；当前 validation 不临时替换为 EMA 权重，未把不存在的 swap 路径列为
  已验证能力。未确认新增实现缺陷。
- `pytest -q tests/unit_tests/cpu/components/optimizer/test_ema.py
  --disable-warnings`，最终退出码 0，72 passed、1.57 秒；日志
  `/tmp/llmtuner-ema-review-units.log`。
- 新真实 CPU DCP 实验：decay=None/0.75 × update_every=1/3，共 4 配置；
  start_step=1、step_bias=2。float64 参数注册共享 alias，complex128 buffer
  注册 stats/stats_alias，pattern 只匹配后一个别名；state 始终采用 primary FQN。
  前五步以独立算式 average+(live-average)*(1-decay) 校验参数及复数 buffer，
  rtol/atol=1e-14。实际 dcp.save 写磁盘后，新 model/EMA 通过 dcp.load 恢复，
  随后四步全部 flat state tensor 与连续运行 bitwise 相等，非 tensor 项相等，
  参数和 buffer 别名 identity 保持不变。
- 命令本地 `python`，最终退出码 0，日志 `/tmp/llmtuner-ema-dcp-oracle.log`；
  资产 `/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-ema-dcp-7s9skcfs`。
  DCP 明确运行于未初始化 PG 的单进程模式，不推广为多 rank 或设备 EMA 验证；
  新 model/EMA 在同一进程构造，不宣称新进程恢复或 trainer 整体 checkpoint 等价。
- 本轮仅记录审查证据，无生产代码变动；文档 pre-commit、diff check 通过。

### LR-SCHEDULE-RESUME-REVIEW-05：实际更新学习率与中途恢复

- 完整复核 `components/optimizer/lr_scheduler.py` 与
  `config/lr_scheduler_config.py`，核对 trainer 的 optimizer.step → scheduler.step
  → EMA.step 顺序、LambdaLR 初始 epoch=0、虚拟末步、warmup/decay 缩短、
  多优化器共同 epoch 及恢复预验证。本轮未确认新增实现缺陷。
- `pytest -q tests/unit_tests/cpu/components/optimizer/test_lr_scheduler.py
  --disable-warnings` 最终退出码 0，44 passed、1.44 秒；日志
  `/tmp/llmtuner-lr-review-units.log`。
- 新实际 CPU 实验：linear/sqrt/cosine × warmup=0/2/7 × decay_ratio=0/0.5/1，
  total_steps=7、min_lr_factor=0.2，共 27 配置。两个独立 SGD optimizer 的
  base_lr=0.1/0.03，逐步以外部解析曲线检查 optimizer 实际应用前的 lr，
  rel_tol=1e-14/abs_tol=1e-15。第 3 次更新后保存参数、optimizer 和 schedule，
  新对象恢复后执行剩余 4 次更新，最终参数 bitwise 等于连续运行，scheduler
  epoch 与 lr metrics 严格一致。全 warmup 与零 decay 均覆盖。
- `python` 实验最终退出码 0，日志 `/tmp/llmtuner-lr-resume-matrix.log`。
  未使用生产 wsd_factor 生成参考，也没有忽略 optimizer state 的 initial_lr；
  本实验为单进程新对象恢复，不证明 DCP/新进程/设备整体训练恢复。
- 本轮无生产代码变动；仅补充审查证据，文档 pre-commit 和 diff check 通过。

### CHECKPOINT-PUBLICATION-REVIEW-05：路径与代际发布失败复核

- 完整复核 `components/checkpointer/{filesystem,storage,checkpoint_writer}.py`：
  pathlike/URI 保留、remote listing 自身 marker 排除、只忽略 FileNotFoundError、
  唯一 shard generation、metadata 预序列化及本地 os.replace/file URI 转换。
  未确认新增缺陷；保留既有明确限制：不提供并发 writer 排他或断电持久性。
- `pytest -q tests/unit_tests/cpu/components/checkpointer/test_filesystem.py
  --disable-warnings`，退出码 0，9 passed、1.20 秒；日志
  `/tmp/llmtuner-filesystem-review.log`。
- `pytest -q tests/unit_tests/cpu/components/checkpointer/test_checkpointer.py
  -k 'same_step or generation_writer or memory_checkpoint' --disable-warnings`，
  退出码 0，16 passed、110 deselected、42 warnings、1.68 秒；日志
  `/tmp/llmtuner-publication-review.log`。核对测试实际调用 DCP manager 写盘和读回，
  memory/file × sync/async × shards/publication 故障注入后比较旧 metadata
  字节不变，未修改 DCP reader 可读取旧权重，重试保存后读取新权重。
- 故障注入是实际文件系统/DCP 路径上的受控异常，不代表进程崩溃或远端网络
  故障恢复；S3/GCS 的 atomic whole-object replacement 尚无真实后端实测，
  不能把 memory/file 结果推广。此次未重新执行既有多 rank FSDP checkpoint 脚本。
- 本批仅更新审查台账；文档 pre-commit、diff check 通过，无生产代码变动。

### CHECKPOINT-LOAD-PLANNER-REVIEW-05：严格键与兼容迁移复核

- 完整复核 `components/checkpointer/load_planner.py`、`model_state.py`：
  compiled key canonicalization、歧义拒绝、AMSGrad destination 动态调整、
  legacy RNG/layout 的受限兼容，以及模型多 part key/shape 预校验与稳定存储。
  未确认新增实现缺陷。缺失 RNG 的 legacy 兼容不等于恢复随机流连续性；
  缺失 rank RNG 仍必须拒绝，不将其误判为旧 checkpoint。
- 核对实际 DCP 回归的断言：compiled tensor 与 byte/scalar 都回写 canonical
  destination；flatten=True/False 均保留原 keys；缺失模型 key、歧义 compiled
  name 和尺寸不匹配在复制前报错；AMSGrad 新分配状态通过 Stateful callback
  传给 optimizer；共享 buffer identity 与 offset view 在新模型恢复后保持。
- 命令 `pytest -q tests/unit_tests/cpu/components/checkpointer/test_model_state.py
  tests/unit_tests/cpu/components/checkpointer/test_random_state.py
  tests/unit_tests/cpu/components/checkpointer/test_checkpointer.py
  -k 'model or compiled or amsgrad or legacy or rng or layout' --disable-warnings`，
  最终退出码 0；结果：81 passed, 110 deselected, 39 warnings in 1.83s。
  日志 `/tmp/llmtuner-load-planner-review.log`。
- 此证据覆盖本地 CPU DCP 与 eager-backend compile 名称迁移，不推广为实际
  Inductor、设备 reshard 或所有 module 自定义 load hook 的原子性保证。
  结构变动后的 wrapper cache 明确不支持，需重新构造 wrapper。
- 本批无生产代码变动，仅更新台账；文档 pre-commit 与 diff check 通过。

### COMMON-NUMERICAL-PRIMITIVES-REVIEW-05

- 完整复核 `models/common/scatter_add.py`、`activation.py`、`cast_linear.py`，
  并追踪 `parallel/expert_parallel/token_dispatch.py::LocalTokenDispatcher.combine`
  和 `models/hf/model_setup.py::apply_cast_lm_head` 调用方；本批未确认新的生产缺陷。
- scatter 自定义算子不修改输入，前向恢复 deterministic enabled/warn-only 状态；
  backward 按 index gather，对大于 index 的 source 区域补零。现有测试比较原生
  scatter 的输出及梯度，覆盖较大 source 和 aot_eager fullgraph，以及默认
  compile 的前向。未将这些结果解释为 CUDA 原子归约的确定性验证。
- CastLinear 禁用外层 autocast 后按配置 dtype 计算；替换复用 weight/bias，
  保留训练模式、RNG、共享参数身份和 state-dict 键。测试涵盖 HF 装配入口、
  非法 dtype、梯度存储 dtype 及 CPU autocast。SituGLU 测试比较独立公式及梯度，
  并检查 Kimi dense/expert MLP 的实际接入。
- 验证：`pytest -q tests/unit_tests/cpu/models/test_scatter_add.py
  tests/unit_tests/cpu/models/test_cast_linear.py
  tests/unit_tests/cpu/models/test_situ_activation.py --disable-warnings`，
  macOS arm64 / PyTorch 2.14.0，退出 0，**34 passed in 5.18s**；
  日志 `/tmp/llmtuner-common-primitives-review.log`。
- 边界：本批为源码及现有 CPU 回归复核，未新增生产改动；不证明 CUDA/NPU、
  DTensor 或多轴组合正确性，也未验证并发切换进程全局 deterministic 标志。

### ROPE-EMBEDDING-FFN-REVIEW-05

- 完整阅读 `models/common/rope.py`、`embedding.py`、`feed_forward.py` 和
  `config/rope_config.py`，复核位置边界、两种旋转布局、YaRN/Llama 配置分支、
  embedding 本地词表契约及 FFN interleaved checkpoint hooks。本批未确认新缺陷。
- 现有 embedding 测试覆盖 padding 所属 shard、foreign token 对 max_norm 与
  scale_grad_by_freq 的影响、错误 shard shape、空 shard、非法全局 token，以及
  不均匀/空词表 shard 的输出求和与完整 embedding 梯度比较。其 rank/group 为
  单进程替身；组件不执行 collective，也未由 HF TP 自动安装，不作真实 TP 证据。
- 执行 `pytest -q tests/unit_tests/cpu/models/test_rope.py
  tests/unit_tests/cpu/models/test_embedding.py
  tests/unit_tests/cpu/models/test_models_common.py --disable-warnings`，退出 0，
  **176 passed in 1.83s**，日志 `/tmp/llmtuner-rope-embedding-ffn-review.log`。
- 独立 CPU 数值实验：Complex/CosSin 两种 RoPE × float32/bfloat16，位置
  `[7,0,7,2,31,1,4]`，以显式二维旋转公式比较输出及加权输入梯度；float32
  rtol/atol=1e-5/1e-6，bfloat16 输出相同容差、梯度 1e-2/1e-2，全部通过。
  FFN bias 开/关两例使用独立逻辑 w1/w3/w2 参数及 F.linear，float64 比较
  输出、输入及全部参数梯度（1e-13/1e-14），新模块加载 logical state 后输出
  与原模块精确一致。退出 0，日志 `/tmp/llmtuner-rope-ffn-oracle.log`。
- 临时实验未加入仓库测试，不能作为长期可重现回归。上述不证明设备内核、
  DTensor、多轴并行、meta materialization 或全局 dtype 转换后的缓存精度。
  后续继续 attention 公共构件及未解除的并行组合缺口。

### ATTENTION-QKV-MASK-REVIEW-05

- 完整阅读 `models/common/attention/{qkv,masks,__init__}.py`，追踪
  `models/common/tp_projections.py::AllGatherFusedQKVLinear._project` 和
  `models/hf/model.py::get_attention_masks`：QKV 按 KV group 保存 query heads、
  key、value；HF 根据 DSA 选择 dense additive mask，否则选择 flex mask。
  本批未确认新的生产缺陷，未进行无收益的文件重组。
- QKV 现有测试覆盖 GQA 几何、任意 leading axes 和空 token、连续输出、
  独立逻辑投影 GEMM 的输出/梯度、weight/bias checkpoint round-trip、
  fused/logical 冲突，以及 bias 转换失败前不消费源条目。
- 命令 `pytest -q tests/unit_tests/cpu/models/test_qkv.py
  tests/unit_tests/cpu/models/test_masks.py --disable-warnings`：退出 0，
  **93 passed in 1.31s**；日志 `/tmp/llmtuner-attention-common-review.log`。
- 独立 CPU 实验枚举 token 长度 1–8 的全部 **255** 种完整文档切分，用显式
  Python 文档归属及 `key <= query` 建立参考；dense 的零/-inf、generic
  document modifier 和 efficient modifier 与 causal 的组合全部一致。
  动态 metadata 的累计边界/最长段与参考精确一致；固定容量 metadata 的真实
  边界相同，剩余槽均为 terminal token count。退出 0，日志
  `/tmp/llmtuner-packed-mask-oracle.log`；临时实验未加入仓库回归测试。
- 边界：枚举覆盖以位置 0 开始、无 padding 的合法 packed segments；不证明
  任意非法输入契约、真实 flex/NPU kernel、CUDA graph 或 DTensor checkpoint
  redistribute。metadata 和 efficient modifier 当前无训练生产调用方；不能以其
  单测通过证明实际训练已使用对应机制。后续继续沿 HF attention backend 复核。

### HF-ATTENTION-BACKEND-REVIEW-05

- 完整阅读 `models/hf/attention_backend.py`，追踪 HF wrapper 构造、
  `uses_flex_attention`、mask 分支、CP backend guard/内核安装，以及
  `parallel/model_compile.py` 的 regional annotation。默认 CUDA 选 llmtuner
  flex adapter，其他设备选 SDPA；DeepSeek V4 特例为 eager。注册为 HF 的全局
  registry 项，adapter 按当前 attention module 的 kernel/annotation 标志执行。
- 执行 `pytest -q tests/unit_tests/cpu/models/test_hf_wrapper.py
  tests/unit_tests/cpu/parallel/test_model_compile.py
  tests/unit_tests/cpu/parallel/test_cp.py --disable-warnings`：退出 0，
  **160 passed, 2 skipped, 2 warnings in 15.53s**，日志
  `/tmp/llmtuner-hf-attention-backend-review.log`。两项 skip 是当前 torch 已有
  async-TP/regional-Inductor 能力时不可达的“缺能力报错”分支。
- 独立真实 CPU Qwen3 GQA4/2、float64 SDPA/eager 对比仍未验收：首次被本机
  Inductor PCH mtime 失效阻断，日志 `/tmp/llmtuner-hf-sdpa-eager-oracle.log`；
  关闭预编译头缓存后重跑退出 1，logits 160/192 元素未满足预设
  rtol=1e-10/atol=1e-12，最大绝对误差 1.4376955945194503e-08，最大相对
  误差 6.188302994018855e-06，日志
  `/tmp/llmtuner-hf-sdpa-eager-oracle-retry.log`。梯度比较尚未触达，不记通过。
- 已检查实际 transformers Qwen3 `eager_attention_forward` 源码，其 softmax
  显式 `dtype=torch.float32` 后转回 query dtype；因此整个模型 `.double()`
  不能使该 eager 路径成为 float64 attention oracle。本批未放宽容差或据此
  修改生产代码；下一步建立数学精度独立参考，区分 HF eager 舍入与 SDPA 差异。
- 本批未确认生产缺陷。CPU/mock adapter 和 guard 结果不证明 CUDA/NPU flex、
  CP 通信内核或多轴组合正确；独立数值验收保持开放。

### HF-ATTENTION-PRECISION-ORACLE-REVIEW-05

- 续接 HF-ATTENTION-BACKEND-REVIEW-05 的独立数值失败，未修改生产代码、
  测试断言或容差。读取当前 transformers Qwen3 attention/SDPA 实现，建立两个
  显式数学参考：repeat_interleave 展开 GQA，einsum 计算 QK，添加 HF 生成的
  causal mask，softmax，再 einsum 与 V 相乘；分别保留 float64 softmax 和
  显式 float32 softmax 后转回 query dtype。
- 真实 `HFTransformerModel` 构造 CPU Qwen3，GQA4/2、1层、hidden16、vocab32、
  dropout0、eval、float64、seed42、tokens `[1,4,2,7,8,3]`，深拷贝相同权重。
  SDPA/eager 两条生产路径不替换；另外两个参考副本仅替换 HF eager attention
  函数，保留其余装配/position/normalization/head 行为。此参考独立于 SDPA
  运算实现，但复用 HF 的非 attention 模块，不能证明整模型所有数学定义。
- SDPA vs 显式 float64：logits 最大绝对差 **0.0**；全部 **14** 个参数梯度
  张量通过 rtol=1e-9/atol=1e-11。HF eager vs 显式 float32-softmax：logits
  最大绝对差 **0.0**；全部 **14** 个参数梯度张量通过相同容差。
  两组 logits 均沿用 rtol=1e-10/atol=1e-12。SDPA/eager 间仍为
  1.4376955945194503e-08，复现原失败而非掩盖。
- 命令 `TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=.
  python /tmp/llmtuner_attention_precision_oracle.py`，退出 0，日志
  `/tmp/llmtuner-attention-precision-oracle.log`。脚本为临时实验，不是仓库回归。
  关闭 PCH 仅处理已记录的本机编译缓存错误，不更换 attention backend。
- 本例失败已由独立参考归因于 HF eager 的 float32 softmax 精度契约，不认定
  llmtuner 缺陷；仅解除本例的数值定位缺口。训练模式/dropout、其他模型/seed、
  compile、CUDA/NPU、真实 CP/TP 组合仍不在本例证明范围内，全局目标未完成。

### HF-RMS-NORM-DTENSOR-REVIEW-05

- 完整复核 `models/hf/normalization.py` 和
  `models/hf/model_setup.py::stabilize_residual_norm_gradients`：ATen custom op
  保留 HF FP32 归一化/输入 dtype cast 顺序，一阶反向及 fake 输出契约；
  仅接受 CPU fp16/bf16/fp32，backend 仅用于 Qwen3/Qwen3MoE。native 残差
  norm 使用共享存储 view 统一梯度合并边界，重复设置清理自有 hook/forward。
  本批未确认新生产缺陷，未重组合理实现。
- `pytest -q tests/unit_tests/cpu/models/test_normalization.py --disable-warnings`：
  退出 0，**38 passed in 4.62s**，日志 `/tmp/llmtuner-normalization-review.log`；
  覆盖三种 input × 三种 weight dtype × 四种 shape（含空/非连续），对实际 HF
  autograd 比较输出及梯度精确一致，另有 custom-op opcheck 与 float64 拒绝。
- `pytest -q tests/unit_tests/cpu/models/test_hf_wrapper.py -k norm_backend
  --disable-warnings`：退出 0，**5 passed, 76 deselected, 1 warning in 7.50s**，
  日志 `/tmp/llmtuner-normalization-lifecycle-review.log`。验证配置装配、参数身份、
  state keys、deepcopy、aot_eager 输出及重复 backend 切换保留用户 hook。
- 新增临时真实 2-rank CPU/Gloo DTensor 实验：shape `(6,4)` Shard(0)，
  `(2,6,4)` Shard(0)/Shard(1)，weight Replicate，fp32，seed123，随机输入、
  非恒定 weight 与上游梯度。调用生产 `rms_norm` 及其注册 sharding 规则；
  forward full_tensor 和输入梯度与完整 Qwen3RMSNorm 精确一致，权重梯度为
  `Partial(sum)`，full_tensor 后与参考通过 rtol/atol=1e-6/1e-6。
- 命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone
  --nproc_per_node=2 /tmp/llmtuner_norm_dtensor_oracle.py`，最终退出 0，日志
  `/tmp/llmtuner-norm-dtensor-oracle.log`。临时脚本不是仓库回归测试。
- 边界：证据覆盖实际 CPU collective 与单轴 DTensor 规则，不代表完整 FSDP/TP
  模型装配、混合精度分布式训练、组合 mesh、CUDA/NPU 或高阶梯度验证。

### MOE-AUX-INJECTION-LIFECYCLE-REVIEW-05

- 完整复核 `models/common/aux_loss.py` 与 `models/common/moe/auxiliary_loss.py`，
  追踪 trainer step denominator、metric collection 和 router 归一化调用契约。
  aux injection 为恒定单位反向、系数和 routed-token 分母预先乘入 raw loss；
  指标不乘 coeff。训练器统计包括 masked prompts、排除 padding，先覆盖累积
  窗口再 DP 求和；不能混用 CE 的有效 label 分母。token partial 使用 forward
  sum/identity backward；已全局一致的 balance value 仅按 batch mesh 汇总。
- 复核 runtime denominator/snapshot 隔离、失败 step scratch 清理、AC replay
  抑制指标写入、按当前实例去重、PP key/count 协调及无 loss stage 参与归约。
  本批未确认新的生产缺陷，未修改源码。
- `pytest -q tests/unit_tests/cpu/models/test_aux_loss.py --disable-warnings`：
  退出 0，**36 passed in 1.71s**，日志 `/tmp/llmtuner-aux-review.log`。
  `pytest -q tests/unit_tests/cpu/test_trainer.py -k 'routing or aux'
  --disable-warnings`：退出 0，**3 passed, 137 deselected in 3.73s**，日志
  `/tmp/llmtuner-aux-trainer-review.log`；包含 masked prompts/padding 分母区别。
- `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone --nproc_per_node=2
  tests/integration_tests/moe_aux_loss_grad_equivalence.py`：退出 0，cp2、tp1、
  ep1、pp1、CPU fp32、每 rank16tokens、E4/K2、coeff0.7。实际 SPMD CP 组调用
  生产 balance loss；完整 token 流独立公式参考比较 raw loss 和 router weight
  gradient，最大梯度差 **4.172e-07**，固定阈值 1e-6，失败 rank0个；日志
  `/tmp/llmtuner-aux-distributed-review.log`。另断言 partial reduce backward
  为 identity。脚本显式求和局部 router 梯度，不代表完整 FSDP 训练。
- 同样 torchrun 前缀运行 `tests/integration_tests/aux_loss_runtime_metrics.py`：
  退出 0，真实 PP2 Gloo collective 验证不同旧实例注册不影响当前 layer mean，
  并验证某 stage 无当前 AuxLoss 时 key/count/metric collective 一致参与；日志
  `/tmp/llmtuner-aux-runtime-review.log`。其 topology 为最小替身，非模型 PP 调度。
- 边界：不证明 TP×EP token ownership、真实 torch_remat、设备内核或完整混合
  并行梯度累积；CPU unit 的 region checkpoint 为替身证据，相关缺口仍保留。

### MODEL-FACTORY-RESOLUTION-REVIEW-05

- 复核 `models/hf/factory.py` 的 offline/Hub/local 分支、override 深拷贝、
  composite text context 扩展、runtime options 传递，以及模型类解析；阅读
  `models/{__init__,common/__init__,hf/__init__}.py` 与
  `utils/lazy_exports.py` 的索引职责和 leaf 解析。本批未确认生产缺陷。
- `pytest -q tests/unit_tests/cpu/models/test_factory_config.py
  tests/unit_tests/cpu/models/test_models_index.py --disable-warnings`：退出 0，
  **10 passed in 4.60s**，日志 `/tmp/llmtuner-model-factory-review.log`。
  覆盖 composite runtime options、实际本地 VL 配置 context 扩展、common
  export 一致性、未知名称 AttributeError 和子进程中 rope 不引入 MoE siblings。
- `pytest -q tests/unit_tests/cpu/models/test_hf_wrapper.py
  -k 'config or architecture or offline or local or resolution' --disable-warnings`：
  退出 0，**26 passed, 55 deselected in 3.82s**，日志
  `/tmp/llmtuner-factory-wrapper-review.log`。按实际筛选范围复核，不声称整个
  wrapper 由本命令覆盖；本批亦不声称 factory 导入仅依赖配置类。
- 临时实际安装 transformers 的六家族矩阵：llama/qwen2/qwen3/qwen3_moe/
  mistral/gemma2，分别构造小型离线配置；解析类匹配 Auto 映射；声明未知
  architecture 后仍回退正确类；声明正确类、保存 config.json 后重新加载，
  确认类及 vocab/hidden 不变、context 至少满足请求；原 override 不被修改。
  全部通过，退出 0，日志 `/tmp/llmtuner-model-resolution-matrix.log`。
- 边界：矩阵仅构造/保存/加载配置和解析类，未构建六种模型权重或运行训练，
  无网络 Hub、自定义 remote code、可选内核缺失环境或并行能力证明。
  临时矩阵不是仓库回归测试；全局训练及并行矩阵验收仍未完成。

### HF-PP-EXPORT-INDEX-REVIEW-05

- 完整复核 `models/hf/state_dict_adapter.py` 和
  `components/checkpointer/hf_storage.py`，检查 prefix/collision、tied head、
  indexed tensor ownership、generation 发布及分布式调用方。
- 确认并修复 **HF-PP-EXPORT-TEST-INDEX-01（P2，验证脚本）**：
  `tests/integration_tests/hf_checkpoint_pp.py` 在成功保存后仍 glob 导出根目录，
  与当前 generation 子目录/index 发布契约不一致，得到空 exported mapping。
  原始真实两 rank 运行退出 1，日志 `/tmp/llmtuner-hf-pp-review.log`。
  改为读取已发布 index，先断言 weight_map 完整键集合，再从每个声明文件读取
  归属张量；保留最终完整键集合及逐 tensor 比较。未改变生产保存机制或容差。
- `pytest -q tests/unit_tests/cpu/models/test_state_dict_adapter.py
  --disable-warnings`：**23 passed, 11 warnings in 4.14s**；
  `/tmp/llmtuner-hf-state-adapter-review.log`。checkpointer `-k hf_export`：
  **7 passed,119 deselected,12 warnings in 1.42s**，
  `/tmp/llmtuner-hf-export-review.log`，覆盖 roundtrip 和失败重写保留旧发布状态。
- 修复后命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone
  --nproc_per_node=2 tests/integration_tests/hf_checkpoint_pp.py` 最终退出 0，
  `/tmp/llmtuner-hf-pp-review-fixed.log`。实际 CPU Llama split 两 stages，
  索引 HF 初始加载、所有 local tensors 对比、最终 generation HF export
  及全部导出 tensors 对比通过；不包括 PP forward/backward 调度。
- 完整 CPU **3641 passed,2 skipped,277 warnings in 38.70s**，退出 0，
  `/tmp/llmtuner-hf-pp-index-full.log`。适用全仓 pre-commit 退出 0，
  `/tmp/llmtuner-hf-pp-index-gates.log`，排除受保护 native_alltoall 和 PPTX；
  compileall、diff HEAD check 通过，已审阅脚本 diff。
- 本批仅修复验证脚本；未确认新的生产缺陷。不证明远程存储、断电持久性、
  CUDA/NPU 或完整并行训练恢复，其他开放问题保留。

### HF-FSDP-PUBLICATION-MATRIX-REVIEW-05

- 续接 HF-PP-EXPORT-TEST-INDEX-01，搜索 integration scripts 的 safetensors
  glob/固定 shard 路径。`kimi_hf_initial_load.py` 已按发布 index 读取 generation
  文件；其固定 `model.safetensors` 仅用于创建初始 HF fixture，非导出假设。
  `hf_checkpoint_equivalence.py` 经正式 storage reader 加载，无相同失效。
- 完整阅读 `tests/integration_tests/hf_checkpoint_equivalence.py`，核对其
  helper topology 为真实 dp_shard2、dp_replicate1、其他轴1，生产 apply_fsdp
  native/custom，TinyTokenDecoder/DTensor、CPU/Gloo。不是 HF attention 模型。
- 运行 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone
  --nproc_per_node=2 tests/integration_tests/hf_checkpoint_equivalence.py`，最终
  退出 0，日志 `/tmp/llmtuner-hf-fsdp-publication-review.log`。
  native/custom × indexed false/true（单/双输出 shard）× consolidation/
  publication 故障共 **8** 个案例全部通过。
- 每例先创建完整训练 checkpoint 和同 step model-export，确认训练 checkpoint
  保持 resumable；修改参数后注入 rank1 consolidation 或 rank0 publication
  OSError，要求所有 rank 捕获 CheckpointException。新建不同 seed 模型及 manager
  加载旧发布权重，全部 full-state tensors 精确一致（rtol=atol=0）；随后成功
  重写、再次加载并与新权重精确一致。故障注入覆盖实际文件 I/O/collective，
  不能解释为仅 mock 测试。
- 本批未新增源码或测试修改，未确认新的生产缺陷；本轮仅记录已有仓库脚本的
  当前执行证据。它验证 HF model-only 发布恢复，不验证 optimizer/RNG/data
  完整恢复、模型 forward/backward、并发 writer 排他、远程存储或断电持久性。
  全局验收及其他并行组合缺口保持开放。

### FSDP-ADAMW-RESHARD-RESUME-REVIEW-05

- 完整阅读 `tests/integration_tests/fsdp_checkpoint_equivalence.py`，核对实际
  helper 模型、DCP manager、optimizer/scheduler 构造与失败注入。以当前工作树
  重新运行本地路径及 `--fsspec` 两组真实 CPU/Gloo 两 rank 矩阵。
- 命令前缀 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --standalone
  --nproc_per_node=2 tests/integration_tests/fsdp_checkpoint_equivalence.py`：
  默认和追加 `--fsspec` 均最终退出 0，日志分别
  `/tmp/llmtuner-fsdp-resume-current-review.log` 与
  `/tmp/llmtuner-fsdp-fsspec-current-review.log`。
- 每组 native/custom × shard/ddp source × shard/ddp destination **8** 例通过。
  使用生产 apply_fsdp、AdamW for-loop、decay scheduler、rank-distinct 数据；
  step2 保存，rank0 同 step 线程重写写入后 OSError，要求所有 rank 获知失败。
  新 seed123/base LR0.3 的新模型、optimizer、scheduler 仍加载原发布 checkpoint。
  后续更新前模型/完整 flat optimizer tensor state 精确一致，scheduler state 和
  当前 metrics 相等；随后四步 loss/全梯度、最终参数与 optimizer state 对齐。
- 同拓扑恢复 atol=0，跨拓扑 atol=1e-5，rtol=0；dense 独立 AdamW 参考使用
  两 rank 全数据，loss/梯度/最终参数及 moment/counter atol=1e-5。保留原容差，
  未用单个参数或最终 loss 替代完整 state 比较。`file://` 为真实本地 fsspec
  适配，不代表 S3/GCS 网络传输。
- 本批未修改源码/脚本，未确认新生产缺陷。该脚本 `states={}`、确定性 `_data`
  生成，不含 Trainer、loader cursor、EMA、RNG 或 dropout；因此这里只完成
  模型/AdamW/scheduler 的恢复证据，不能宣布完整训练状态恢复完成。设备内核、
  多轴组合及跨进程重启亦未由本例证明，原开放项继续保留。

### TRAINER-FSDP-RNG-LOADER-RESUME-REVIEW-05

- 复核 `tests/integration_tests/pp_checkpoint_equivalence.py` 的配置、full/resume、
  runtime snapshot 和逐状态比较，补充真实 Trainer 与独立进程重启证据。
- 两阶段共同环境：`PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0
  TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 LLMTUNER_TEST_PP_SIZE=1
  LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom LLMTUNER_TEST_EMA=dynamic
  LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1
  LLMTUNER_TEST_ACCUMULATION=2`；分别运行 `torchrun --standalone
  --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py
  full|resume /tmp/llmtuner-trainer-resume-review.xZdwC6`，两进程组均退出 0。
  日志 `/tmp/llmtuner-trainer-full-current.log`、
  `/tmp/llmtuner-trainer-resume-current.log`，保留 artifacts 供复核。
- CPU/Gloo Qwen3、custom dp_shard2、pp/tp/cp/ep1、fp32、eager、AC none、
  accumulation2、synthetic loader、动态 EMA；full 训练4步并于step2保存，
  resume 新进程 Trainer 加载 step2。两 rank 的 pre-update state failures 均0，
  覆盖模型、完整 optimizer、scheduler、trainer/token counters、EMA、Python/
  NumPy/Torch RNG snapshot、loader state；DTensor 校验 layout 与 local shard。
- 继续2步 loss 与连续训练尾段精确一致，max_abs=0；最终全部模型/optimizer/
  scheduler/trainer/token counters/EMA 比较通过。未修改生产代码或脚本。
- 明确验证缺口：虽然 final snapshot 保存 RNG/loader，当前脚本 final 比较仅
  手动枚举上述训练状态，未比较 final RNG/loader；本批只证明它们在恢复后首次
  更新前相同。模型无 dropout、数据为 synthetic，不能把 loss 一致当作随机消费
  或真实 corpus cursor 的充分证明。后续补齐该验收，再验证实际 corpus/compile。
  该例亦不证明其他组合或目标设备；全局目标保持未完成。

### TRAINER-FINAL-RNG-LOADER-CHECK-REVIEW-05

- 补齐 **TRAINER-RESUME-FINAL-STATE-01（P2，验证缺口）**：
  `tests/integration_tests/pp_checkpoint_equivalence.py::_phase_resume` 原先只比较
  保存点 RNG/loader，未比较后续训练终点的这两项已保存状态。现在对 final
  snapshot 的 `rng`/`dataloader` 使用已有递归比较器，精确比较 tensor、元数据
  和键集合；任一侧有键而另一侧缺失也计失败。不改变其他状态检查或数值容差。
- 沿用上一批独立 full 进程生成的 artifacts：
  `/tmp/llmtuner-trainer-resume-review.xZdwC6`。相同环境
  `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0
  TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 LLMTUNER_TEST_PP_SIZE=1
  LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom LLMTUNER_TEST_EMA=dynamic
  LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1
  LLMTUNER_TEST_ACCUMULATION=2` 下，运行 `torchrun --standalone
  --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py resume
  /tmp/llmtuner-trainer-resume-review.xZdwC6`，退出0，日志
  `/tmp/llmtuner-final-rng-loader-resume.log`。两 rank 更新前检查通过，恢复两步后
  final RNG/loader 与原连续训练精确一致，loss max_abs=0，其余最终状态仍通过。
- 完整 CPU **3641 passed,2 skipped,261 warnings in 37.18s**，退出0，
  `/tmp/llmtuner-final-rng-loader-full.log`。适用全仓 pre-commit 通过，
  `/tmp/llmtuner-final-rng-loader-gates.log`，排除受保护文件及PPTX；
  compileall、diff HEAD check 通过，已审阅实际脚本差异。
- 本批不修改生产代码。上述解除该 synthetic/custom-FSDP2/eager/动态EMA/
  accumulation2 案例的终点状态比较缺口，不证明 dropout 随机消费、真实语料
  cursor、compile 或其他并行组合。全局审查与矩阵验收仍在进行。

### TRAINER-PACKED-CORPUS-RESUME-REVIEW-05

- 续接最终 RNG/loader 比较增强，使用实际本地 JSONL、tokenizer、production
  packing/loader 和 Trainer 做独立进程 full/resume，而非 synthetic 数据。
  阅读 `_write_corpus` 和 `_cpu_packed_attention` 明确替代机制边界。
- 两阶段相同环境：`PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0
  TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 LLMTUNER_TEST_PP_SIZE=1
  LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom LLMTUNER_TEST_EMA=dynamic
  LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1
  LLMTUNER_TEST_ACCUMULATION=2`，命令 `torchrun --standalone
  --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py
  full|resume /tmp/llmtuner-packed-resume-review.3mrNYe --packed`，两阶段退出0。
  日志 `/tmp/llmtuner-packed-full-current.log`、
  `/tmp/llmtuner-packed-resume-current.log`；artifacts 保留。
- CPU/Gloo、Qwen3、custom dp_shard2、其他并行轴1、fp32、eager/AC none、
  accumulation2、动态EMA，shuffle false。本地生成多文档 JSONL，真实 loader
  checkpoint 恢复，无手工 fast-forward。step2 保存/新进程恢复，更新前全状态
  两 rank failures=0；后续两步 loss 精确一致（max_abs=0），最终模型、optimizer、
  scheduler、trainer/token counters、EMA、RNG、loader 状态均比较通过。
- attention 使用脚本的 CPU substitute：读取 production BlockMask.mask_mod，
  create_mask 展开 dense boolean mask，再交给 HF SDPA。这验证真实语料 packing/
  cursor、状态保存恢复和 CPU dense attention 路径，不代表生产 CUDA flex kernel
  或该 packed 配置在原生 CPU CLI 无替换时受支持。无 dropout/compile/streaming
  或多轴组合结论。本批未新增源码修改、未确认生产缺陷，全局目标未完成。

### TRAINER-COMPILED-FULL-STATE-RESUME-REVIEW-05

- 当前增强的 final RNG/loader 比较用于真实独立进程 compile 恢复。共同环境：
  `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0
  TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 LLMTUNER_TEST_PP_SIZE=1
  LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom LLMTUNER_TEST_EMA=dynamic
  LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1
  LLMTUNER_TEST_ACCUMULATION=2 LLMTUNER_TEST_NORM_BACKEND=aten
  LLMTUNER_TEST_COMPILE_BACKEND=aot_eager`；命令 `torchrun --standalone
  --nproc_per_node=2 tests/integration_tests/pp_checkpoint_equivalence.py
  full|resume /tmp/llmtuner-compiled-resume-review.uAOv3S`，两阶段最终退出0。
  日志 `/tmp/llmtuner-compiled-full-current.log` 和
  `/tmp/llmtuner-compiled-resume-current.log`，artifacts 保留。
- CPU/Gloo Qwen3、custom FSDP shard2、其他并行轴1、fp32、AC none、synthetic
  loader、动态 EMA、accumulation2；实际 production ATen norm 和 aot_eager
  whole-model compile。脚本逐步断言 OptimizedModule、unique_graphs>0，且
  Qwen norm.forward 为请求的 ATen 实现，避免配置未生效却通过。
- step2 保存，新进程加载前后检查：两个 rank pre-update failures=0；随后两步
  loss 与连续4步尾段精确一致（max_abs=0）；最终模型/optimizer/scheduler/
  trainer/token counters/EMA/RNG/loader 均通过原精确比较。未放宽断言。
- Dynamo 警告记录：`masks.py` 的 lru_cache wrapper 在 tracing 时被忽略、直接
  trace 底层函数；本例未观察数值差异，警告不作为已发现 bug 或完整图证明。
  whole-model 非 fullgraph compile 允许 graph breaks，本例只证明实际编译图参与
  与恢复等价，不宣称所有算子编译或性能收益。
- 本批只新增验证证据，未修改源码。无 Inductor/device kernel、dropout、真实
  packed corpus/compile 组合或多轴验证结论，全局目标仍未完成。

### GC-TRAINER-CLEANUP-REVIEW-05

- 完整复核 `utils/garbage_collection.py`，追踪 Trainer.train 中创建/run/finally
  和 Trainer.close 的资源释放。GC 使用进程级 owner count/lock，最后一个
  owner 关闭才恢复最初 enabled 状态；初始化失败捕获 BaseException 并释放
  ownership，close 幂等，closed.run 明确报错。本批未确认新生产缺陷。
- 阅读现有 GC 回归：非法配置不改变全局开关、初始 enabled/disabled、正反顺序
  关闭重叠 owner、失败嵌套 owner、RuntimeError/KeyboardInterrupt/SystemExit
  初始化失败。Trainer 关闭逐项尝试，保存首个清理错误；训练已有错误时保留原
  错误，失败 metrics/loader 和可重试 checkpoint owner 保留以便再次关闭。
- `pytest -q tests/unit_tests/cpu/utils/test_utils.py --disable-warnings`：退出0，
  **52 passed in 1.17s**，日志 `/tmp/llmtuner-gc-lifecycle-review.log`。
  此文件同时包含其他 utils 测试，不能将52项全部标为GC测试。
- `pytest -q tests/unit_tests/cpu/test_trainer.py -k 'close or cleanup or
  initialization' --disable-warnings`：退出0，**14 passed,126 deselected
  in 4.19s**，日志 `/tmp/llmtuner-trainer-cleanup-review.log`。
- 本批仅审查/验证与记录，无源码修改。测试包含失败注入及资源替身，不代表
  任意线程同时操作 GC、外部代码改写全局 GC 状态、真实媒体 worker 或设备
  通信故障均可恢复；不据此解除分布式故障恢复边界。全局目标仍在进行。

### DTYPE-CANONICAL-NAME-REVIEW-05

- 完整复核 `utils/dtypes.py`、`utils/module_names.py`，追踪 lm_head 配置校验
  与 `model_setup` 调用方、DCP load planner。dtype helper 的实际生产调用方
  先校验 compute_dtype；canonical_fqn 只删除完整 dotted wrapper segment，
  不对相似名称做 substring 删除，planner 对 compiled 名称归一后的冲突报错。
- `pytest -q tests/unit_tests/cpu/utils/test_module_names.py
  tests/unit_tests/cpu/components/checkpointer/test_model_state.py --disable-warnings`：
  退出0，**25 passed,2 warnings in 1.41s**，日志
  `/tmp/llmtuner-canonical-names-review.log`。范围为名称/模型状态，不声称覆盖
  所有 optimizer/load-planner 分支；此前 planner 专项证据保持独立。
- 临时纯逻辑检查12个 dtype alias、4个拒绝字符串，并确认配置允许的 lm_head
  dtype 均解析为浮点 dtype，退出0；`/tmp/llmtuner-dtype-contract-review.log`。
  本批未确认生产缺陷，未新增源码/回归测试，不因统一外观移动准确命名模块。
- 边界：typed helper 不承担任意对象输入校验；canonical wrapper 名是保留语义，
  不是任意真实子模块同名消歧机制。无设备转换、DTensor 或数值精度验证结论。
  全局目标仍未完成。

### ACCELERATOR-CAPABILITY-DEVICE-REVIEW-05

- 完整阅读 `accelerator/capabilities.py`、`accelerator/device.py`：核对 cached
  registry、未知名称立即报错、缺能力提示、PP microbatch driver 签名探测、
  grouped_mm 实际调用探测，以及设备优先级、vendor extension 导入、backend
  对应和 set_device 失败语义。本批未确认新的生产缺陷，未修改源码。
- `pytest -q tests/unit_tests/cpu/accelerator/test_capabilities.py
  tests/unit_tests/cpu/accelerator/test_device.py --disable-warnings`：退出0，
  **16 passed in 1.15s**，日志 `/tmp/llmtuner-capability-device-review.log`。
  包括 registry/cache/error 契约、缺失 PP eval、grouped_mm probe 的设备及
  expert weight stride；模拟 vendor/device 结果不算目标设备实际证据。
- 当前进程直接执行全部8个 registry probes，正常退出0；输出
  `/tmp/llmtuner-current-capability-probes.log`。实际选择 CPU/Gloo。
  属性/签名探测只证明符号或调用形式存在，不证明训练 kernel、autograd、
  dtype/shape 全域或通信 backend 可用；grouped_mm 的小型 bf16 probe 也仅为
  当前设备单个几何样本。不能把 symm_mem 符号存在解释为 CPU 可运行 CUDA
  symmetric-memory collectives。
- 缓存与 import-time device 选择为进程生命周期契约；运行中替换设备/扩展或
  动态修改 torch API 不属于本例验证。真实 CUDA/NPU/其他 vendor backend
  仍需目标硬件，不能由本机通过解除并行矩阵缺口。全局审查继续。

### ACCELERATOR-SPMD-NESTING-REVIEW-05

- 复读 `llmtuner/accelerator/spmd.py`、CPU SPMD 测试及 trainer、validation、pipeline 调用位置；本轮未确认生产缺陷，未修改运行时实现。
- `spmd_context(None)` 是已有 no-op 契约，嵌套时继承外层 mesh。实际安装的 `spmd_types.set_current_mesh(None)` 也表示复用当前 mesh，不能将其误解为清空。显式 `set_current_spmd_mesh(None)` 仅屏蔽本地查询、不清空上游类型上下文；当前生产调用方未使用此分支，不据此宣称支持完全禁用上游上下文。
- 真实单 rank CPU/Gloo DeviceMesh 验证通过：dense/sparse 注册、不同具名 mesh 嵌套、内层异常恢复、None 继承、单例 group 抑制、最终注册清理。日志 `/tmp/llmtuner-spmd-nesting-review.log`，退出 0。测试通过 `file://` rendezvous 初始化实际进程组，无伪造 mesh；不证明多 rank 通信、线程隔离或设备组合正确。
- `pytest -q tests/unit_tests/cpu/accelerator/test_spmd.py --disable-warnings`：6 passed，0.61s。无代码变更；全局完成条件及此前硬件验证缺口仍开放。

### MC2-FSDP-POST-UPDATE-LIFECYCLE-REVIEW-05

- 补强 `tests/integration_tests/mc2_fsdp_lifecycle.py`：原测试仅一次前向/反向，不能验证 optimizer 更新后的重新 unshard 或 hook 搬移后参数视图是否陈旧。新增真实 SGD(lr=0.1) 更新，逐参数 full_tensor 对照未分片 GroupedExperts，清空梯度后以新输入再次前向/反向并核对全部专家梯度；沿用 FP64 atol=rtol=1e-12。未修改生产代码、融合路径或原断言。
- 实际命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29889 tests/integration_tests/mc2_fsdp_lifecycle.py`：exit0，native、custom、custom hook_module=root 三场景均通过。日志 `/tmp/llmtuner-mc2-fsdp-update-localhost.log`。首次 standalone rendezvous 解析为198.18地址，TCPStore连接失败；确认launcher仍运行后终止PID54204并取得exit1，才改用显式localhost。失败日志 `/tmp/llmtuner-mc2-fsdp-update-review.log` 保留。
- 真实两rank CPU/Gloo FSDP；每rank相同数据、FP64、专家E2/hidden4/FFN3、counts=[2,3]，MC2算子是local world_size=1的CPU EmulatedOps。覆盖参数分片与更新生命周期，不覆盖EP传输、TP、真实NPU内核、异步性能、checkpoint或不均匀DP数据。
- 完整CPU测试3641 passed、2 skipped、277 warnings、41.15s（`/tmp/llmtuner-mc2-fsdp-update-full.log`）；compileall通过。MC2×TP原FP16失败与Ascend数值契约、KCP状态通信及Domino设备重叠仍开放。

### TRAINER-OPTIMIZER-SERVICES-ROLLBACK-REVIEW-05

- 复读 runtime assembly/batch/validation、collectives 的 synchronize_local_errors、optimizer 构造及 Kimi checkpoint 模板边界。所检查调用将已知通信放在本地异常协调外；未由局部阅读宣称全部调用方或第三方 hook 无 collective。
- 补强 `tests/integration_tests/trainer_initialization_coordination.py`：真实 Trainer/FSDP/optimizer 装配后，在 build_lr_scheduler 注入 failing_rank0/1 × ValueError/KeyboardInterrupt/SystemExit 共六场景；失败侧保留异常对象身份，健康侧收到 Optimizer services construction 阶段错误，均不得进入 dataloader。复用 failed_constructor 对照 Python/NumPy/Torch CPU RNG、deterministic/anomaly/cuDNN 开关及环境变量回滚；每场景 barrier 检查双方仍可通信，最后真实新 Trainer train_step 有限并关闭。
- 命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29890 tests/integration_tests/trainer_initialization_coordination.py`：exit0，日志 `/tmp/llmtuner-optimizer-services-coordination-no-pch.log`。首次同命令未设置 PCH 开关时 exit1，Inductor 头文件mtime不匹配，日志 `/tmp/llmtuner-optimizer-services-coordination.log`；确认终止后重跑，不修改生产配置或放宽断言。
- CPU/Gloo 两rank、默认 native FSDP shard2、其他并行轴1；注入仅替换本地scheduler构造，非模拟网络故障。不证明进程被杀、通信失败可恢复，亦不覆盖设备RNG或真实信号到达collective。未修改生产实现。
- 完整CPU3641 passed、2 skipped、284 warnings、41.33s，日志 `/tmp/llmtuner-optimizer-services-full.log`；compileall通过。全局迁移及设备验证开放项保持不变。

### CLI-CUSTOM-AC-PREFETCH-REVIEW-05

- 复读 `__main__.py`、trainer/cli.py、config/cli_config.py 与模型/训练字段，核对 pyproject console script 仍指向同一 main；prefetch 参数由 FSDPConfig 经 apply_fsdp 传至 set_fsdp_prefetch_windows。现有显式args隔离、禁用隐式args文件、seed前置拒绝和原始异常优先清理契约保持不变，本轮未确认新的生产缺陷。
- 实际生产入口命令：`PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29891 -m llmtuner --model_name_or_path qwen3 --vocab_size 32 --hidden_size 16 --intermediate_size 32 --num_hidden_layers 3 --num_attention_heads 2 --num_key_value_heads 2 --max_seq_len 8 --global_batch_size 2 --steps 3 --data_parallel_shard_size -1 --fsdp_implementation custom --fsdp_forward_prefetch_count 2 --fsdp_backward_prefetch_count 2 --activation_checkpoint_mode full --log_freq 1 --dump_folder /tmp/llmtuner-cli-custom-ac-prefetch-review`，exit0。真实CPU/Gloo custom shard2、其他轴1、FP32、eager/full AC；日志确认3层AC及3步完成，后两步loss3.47631/3.47194，grad_norm2.1441/2.1832，日志 `/tmp/llmtuner-cli-custom-ac-prefetch-review.log`。无算子模拟或训练入口替换。
- 此为公开CLI组合可运行性证据，未收集prefetch enqueue时间线、未对照独立数值参考或保存完整checkpoint，不证明异步重叠性能、恢复、设备内核或TP/PP/CP/EP组合。
- `pytest -q tests/unit_tests/cpu/test_trainer.py -k 'cli or seed_checkpoint' --disable-warnings`：19 passed、121 deselected、8.37s（`/tmp/llmtuner-cli-contract-review.log`）；FSDP contract110 passed、5.43s（`/tmp/llmtuner-fsdp-prefetch-contract-review.log`）。仅新增台账，本轮未修改代码；不重复已完成的全量CPU门禁。

### TRAINER-AC-PREFETCH-FULL-STATE-RESUME-REVIEW-05

- 补强 `tests/integration_tests/pp_checkpoint_equivalence.py` 配置入口：新增 LLMTUNER_TEST_FORWARD_PREFETCH、LLMTUNER_TEST_BACKWARD_PREFETCH（默认0）及 LLMTUNER_TEST_NUM_LAYERS（默认维持原schedule层数）。通过真实 FSDPConfig/ModelConfig 生效，保留所有既有状态及loss断言；不修改生产运行时。
- 新进程 full/resume 两阶段，真实CPU/Gloo两rank、custom FSDP shard2、TP/PP/CP/EP1、FP32、Qwen3三层、full AC、前后向窗口均2、累积2、dynamic EMA、RNG及loader检查开启。先前两层试跑亦通过，但窗口会按边界截短；最终使用三层，具备两个后续decoder单元。没有以窗口配置值推断实际设备重叠性能。
- 两阶段共同环境：`PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom LLMTUNER_TEST_FORWARD_PREFETCH=2 LLMTUNER_TEST_BACKWARD_PREFETCH=2 LLMTUNER_TEST_NUM_LAYERS=3 LLMTUNER_TEST_AC=full LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1 LLMTUNER_TEST_ACCUMULATION=2`；命令 `torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29893 tests/integration_tests/pp_checkpoint_equivalence.py full|resume /var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-ac-prefetch-three-layer-hpb_jrgy`。两者exit0；driver `/tmp/llmtuner-ac-prefetch-three-layer-driver.log`，阶段日志 `/tmp/llmtuner-ac-prefetch-three-layer-{full,resume}.log`。
- step2保存、连续训练4步对照新进程恢复后2步：两rank更新前完整状态failures=0，最终模型/AdamW/scheduler/EMA/counters及RNG/loader状态通过精确比较，tail loss最大差0。此为同配置连续运行参考，不等同独立dense数值oracle、改变窗口后的迁移或真实CUDA/NPU预取验证。
- 完整CPU3641 passed、2 skipped、260 warnings、39.62s（`/tmp/llmtuner-ac-prefetch-full-unit.log`），compileall通过。全局硬件与迁移缺口继续开放。

### HF-CUSTOM-AC-PREFETCH-DENSE-ORACLE-REVIEW-05

- 复用此前 fine_grained_fsdp_equivalence 的小模型dense/compile/outstanding证据，不重复其16场景。新增 `tests/integration_tests/hf_fsdp_ac_prefetch_equivalence.py`，以未并行、无AC的原始HF Qwen3 CausalLM作为参考，公开 apply_fsdp 与生产 wrap_full 作用于另一副本；初始全参数FQN集合与内容逐位相同。模型三层、vocab32/hidden16/FFN32/heads2、FP32、无dropout，两个rank使用不同8-token因果序列，loss按全局16tokens归一化。
- 真实CPU/Gloo custom FSDP shard2、其他轴1、full AC、前后向prefetch窗口2，连续三步SGD(lr0.05,momentum0.9)。每步对照本地输出、全量gather后的全部参数梯度与更新参数，atol1e-6/rtol1e-5，未放宽已有测试断言。oracle前向分别处理两个完整rank序列后合并loss，不把两个序列拼成一个因果上下文。
- 命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29894 tests/integration_tests/hf_fsdp_ac_prefetch_equivalence.py`，exit0，两rank三步全部通过，日志 `/tmp/llmtuner-hf-ac-prefetch-repository.log`。初始临时原型亦exit0，日志 `/tmp/llmtuner-hf-ac-prefetch-oracle.log`。仓库脚本另加CPU环境拒绝、初始参数集合及内容严格校验；最后调整仅返回注解与输出换行。
- 独立HF计算参考补充上一批同配置恢复证据；此脚本不覆盖Trainer数据预处理、padding/packing、EMA、checkpoint、compile或设备通信/重叠。完整CPU3641 passed、2 skipped、282 warnings、40.48s（`/tmp/llmtuner-hf-ac-prefetch-full-unit.log`）；compileall通过。

### CHECKPOINT-RETENTION-WORKER-REVIEW-05

- 复读 checkpointer/base.py 的 purge_thread、close、async retention completion及目录分类，distributed_checkpoint 的save/close与resumable检查、storage协议；当前本轮检查未确认新的生产缺陷。删除失败会记录并继续，保存成功后才把新恢复点计入horizon；不据此宣称并发独立manager或回退重写旧step的竞态已验证。
- 补强 `tests/integration_tests/checkpoint_save_coordination.py`：此前同步/异步 × failing_rank0/1 的真实writer初始化失败/重试最多保存两步，keep_latest_k=2从未触发删除。现在成功重试step2后修改bias、真实保存step3，等待完成并close；两rank barrier后断言purge线程已退出，目录精确为step2/step3，实际DCP加载step3逐位匹配模型。原失败时旧快照可加载和重试断言全部保留。
- 命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29895 tests/integration_tests/checkpoint_save_coordination.py` exit0（`/tmp/llmtuner-retention-worker-coordination.log`）。两rank CPU/Gloo、普通FP32 Linear/AdamW/LambdaLR、真实本地存储和后台线程，四种组合的completed retention通过；同脚本已有load/stager/staging异常恢复也通过。不是FSDP模型、远程存储或真实设备pinned copy证据。
- 完整CPU3641 passed、2 skipped、274 warnings、40.84s（`/tmp/llmtuner-retention-worker-full-unit.log`），compileall通过。仅修改集成验证与台账，未修改生产实现。

### DATA-LOADER-FULL-BATCH-PREFETCH-RESUME-REVIEW-05

- 复读 datasets/dataloader.py、sources.py 的 HF cursor 和 packing/iterators.py。Grain保存消费侧iterator state，source cursor含epoch及HF状态；packing remainder通过保存父游标并重放原样本恢复。未确认本批新的生产缺陷，不据此宣称所有外部HF source均可精确恢复。
- 补强 `test_loader_resume_reproduces_the_following_batches_exactly`：原只比较input列表，无法发现labels/positions/padding/有效token数错误。现逐字段完整batch精确比较，并参数化source/target预取深度2→2、1→4、4→1；恢复目标已消费一个batch并已启动迭代器，源消费两个batch，随后连续三个batch必须一致；finally关闭双方loader。预取深度不改变数据身份或batch布局。
- 针对恢复、stream repeated epoch、window shuffle的6项测试通过（332 deselected，4.03s）；完整CPU3643 passed、2 skipped、265 warnings、42.59s，日志 `/tmp/llmtuner-loader-full-batch-resume-units.log`。真实Grain线程预取、本地tokenizer/corpus、CPU单进程；不证明多进程worker、远程source或故障时自动回滚训练step。
- 仅修改数据回归与台账；compileall通过，原生产行为与既有工作区改动保留，全局目标及设备矩阵缺口保持开放。

### PACKING-RESTORE-ROLLBACK-ERROR-01（P2，已修复）

- 位置：datasets/packing/iterators.py::DocumentAwareConcatThenSplitIterator.set_state。原恢复异常后的 parent.set_state(previous_parent_state) 如果再次抛错，会覆盖原始ValueError/KeyboardInterrupt，并留下开放但父游标未知的迭代器。
- 两项故障回归先复现：原恢复抛指定异常对象，回滚抛OSError；修复前两项均因OSError覆盖失败，日志 `/tmp/llmtuner-packing-rollback-before.log`，exit1。修复后回滚错误记录日志并关闭迭代器；清理错误另行记录，原恢复异常仍原样抛出。回滚成功时保留已有继续消费契约。
- 回归断言原异常对象身份、两次set_state调用、父子迭代器关闭及后续next拒绝；普通ValueError和KeyboardInterrupt两种。针对packing/loader恢复57 passed、283 deselected、6 warnings、3.44s（`/tmp/llmtuner-packing-rollback-after.log`）；完整CPU3645 passed、2 skipped、260 warnings、42.77s（`/tmp/llmtuner-packing-rollback-full.log`）。compileall通过。
- 本批源码变更为iterators.py及上述回归；检测到外部工作将HEAD更新到34ac892并暂存iterators.py、utils/dtypes.py，未改动外部index或dtypes内容，未自行commit/push。故障注入针对真实Grain父子iterator边界，不外推远程source故障或rank协调。

### DATA-COMPOSITION-UNEVEN-EPOCH-REVIEW-05

- 完整复读 datasets/composition.py：map预处理/过滤→global shuffle→连续DP分片→repeat；stream先source ownership再window shuffle/处理；mix权重按最小值归一化并拒绝溢出，concat先关闭子节点shuffle/repeat/DP再全局应用。保留过滤后的可用行不等量及finite mix先耗尽即停止的既有契约。本轮未确认新缺陷，未调整合理结构。
- 新增 test_uneven_dp_slices_cover_each_repeated_epoch：真实Grain map图，7行/3rank与11行/4rank × shuffle开/关，验证分片大小总和及最大差1；每个rank独立repeat游标连续读取三个本地epoch，按rank合并后逐项等于未分片参考对应epoch，且每个源ID恰好一次。四项通过（340 deselected，4.19s）。这是单进程模拟DP ownership坐标的纯数据图验证，不是实际collective或过滤/packing/stream组合证明。
- 当前178个源码文件均可在台账找到路径，但路径出现不作为完成依据，仍需各机制证据及开放项验收。完整CPU3649 passed、2 skipped、272 warnings、43.09s（`/tmp/llmtuner-uneven-dp-epoch-full.log`）；compileall、受影响pre-commit和diff检查通过。仅补测试与记录，保留先前packing修复及外部暂存改动。

### TRAINER-FETCH-MALFORMED-EXHAUSTION-PRECEDENCE-REVIEW-05

- 完整复读 trainer/runtime/batch.py，核对fetch状态优先级（失败0、耗尽1、ready2，world MIN）、先验证count再token计数、CPU消费后搬运与preprocess本地错误边界。num_valid_tokens为collator提供的可信整数元数据，当前边界校验范围而不每批重算；不将其当作任意外部batch真实性证明。本轮未确认生产缺陷。
- 扩展 data_fetch_coordination.py：故障rank0/1 × 缺labels/非法负num_valid_tokens，与另一rank同时耗尽；失败侧保留KeyError/ValueError，另一侧必须报Data fetch failed，不得误返回DataloaderExhaustedError，每场景barrier。原脚本训练/验证初始化、preprocess、清理及结果错误协调断言保留。
- 实际 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29896 tests/integration_tests/data_fetch_coordination.py` exit0（`/tmp/llmtuner-fetch-exhaustion-precedence.log`），四场景通过。CPU/Gloo真实world状态归约，source为确定性小型generator；不覆盖blocked loader、模型collective内故障或硬件进程中断。格式检查指出循环闭包需显式绑定failing_rank，已修正，最终pre-commit通过。
- 完整CPU3649 passed、2 skipped、273 warnings、38.79s（`/tmp/llmtuner-fetch-precedence-full.log`），compileall/diff通过。仅测试与台账变更，生产代码及外部暂存改动保留。

### PARALLEL-MIGRATION-CLAIM-AUDIT-REVIEW-05

- 修复 distributed_implementation_comparison.md 正文过时断言：此前仍称MC2×TP显式拒绝，与当前ParallelConfig及apply_ep→block_conversion接线和后文失败记录矛盾。实际构造 TP2/EP2/dispatcher=mc2 配置成功；现准确写为配置/装配可达、原FP16梯度验收仍失败，不能宣称数值支持通过。没有删掉或放宽MC2-TP-PUBLIC-01失败。
- 同段补充KCP实际输入边界：非空等长local sequence，padding/compact状态交换未完成，拒绝配置不能视为完成迁移。Domino目标设备重叠及上游完整backward pipeline未实现的说明继续保留。
- 本机实查：CUDA不可用，importlib查找torch_npu、fla、triton_ascend_kernels、torch_remat均不存在。MC2实际融合、KCP实际状态通信、Domino CUDA流性能不能在此环境验收；当前仍有其他源码与调用链审查可推进，不标记全局blocked或complete。
- 仅修正文档声明，配置探针与源码阅读不是设备验证；文档pre-commit及diff检查通过，无需重复训练全量测试。

### OPTIMIZER-FQN-GROUP-REORDER-REVIEW-05

- 完整复读 components/optimizer/state_dict.py，核对lazy状态初始化保护grad/lr/eps/decay、失败清理、FQN重复拒绝、group选项一致性、Adam计数器/moment形状/AMSGrad恢复。本轮未确认新的生产缺陷。
- 新增Adam/AdamW两项test_fqn_restore_matches_update_after_parameter_group_reordering：源已更新一次且有不同LR参数组，目标以逆序组及不同LR构造；初始化后从独立flat快照恢复，全部flat状态精确相等，再对每个名字施加不同梯度，更新参数及全部flat状态继续逐位相同。验证按FQN而非整数位置恢复，不涉及冻结/共享参数的联合场景。
- 初次测试直接传入源state tensor，PyTorch同设备load保留storage，参考更新污染目标moment，导致两项失败；日志 `/tmp/llmtuner-optimizer-group-reorder.log`（2failed/63passed）及同夹具全量失败 `/tmp/llmtuner-optimizer-group-reorder-full.log` 保留。改用deepcopy模拟持久化独立快照，未改生产加载、容差或梯度；65 passed、2 warnings、0.99s（`/tmp/llmtuner-optimizer-group-reorder-fixed-fixture.log`）。这是测试别名缺陷，不作为生产bug。
- 最终完整CPU3651 passed、2 skipped、268 warnings、42.95s（`/tmp/llmtuner-optimizer-group-reorder-final-full.log`）；compileall和受影响pre-commit通过。仅测试与台账变更；CPU直接flat加载不替代分布式DCP或设备验证。

### EMA-SPARSE-FIRING-BUFFER-RESUME-REVIEW-05

- 完整复读 optimizer/ema.py、config/ema_config.py，并核对Trainer在optimizer及scheduler后调用EMA。固定tracking scope、身份/metadata预检、buffer别名去重、动态firing数从global step推导及cold-start契约保持；本轮未确认新生产缺陷。
- 新增test_dynamic_ema_sparse_firings_resume_parameter_and_buffer_against_formula：FP64标量参数和buffer，start_step2、间隔3、bias3、half_life0.5，连续14步live值变化；独立Python公式仅在step5/8/11/14更新，逐步对照atol/rtol1e-14。step7在两个firing之间创建新model/EMA并从独立快照恢复，后续每步全部flat状态与连续路径逐位相等，验证buffer与稀疏动态aging组合而非仅state round-trip。
- EMA suite73 passed、1.48s（`/tmp/llmtuner-ema-sparse-resume.log`）；完整CPU3652 passed、2 skipped、279 warnings、43.14s（`/tmp/llmtuner-ema-sparse-resume-full.log`）。最后仅显式初始化resumed_model并加非空断言，修复静态作用域检查，无数值逻辑改变。compileall、受影响pre-commit、diff通过。
- 仅测试和台账变更；CPU直接EMA状态恢复，不外推设备buffer/DTensor、DCP组合或训练失败后的事务回滚。全局目标仍开放。

### SCHEDULER-STATE-TYPE-01

- 优先级P2；位置 `components/optimizer/lr_scheduler.py::LRSchedulersContainer.load_state_dict`，由checkpoint恢复调用。非字典的假值（None、空列表、空字符串、0、False）此前被静默当作空状态，掩盖损坏的恢复数据；非空错误类型则产生偶然异常。
- 在空状态判断前显式校验dict，非法类型统一抛ValueError，保留空字典no-op契约。新增七项回归同时检查拒绝输入后步数和学习率不变；不改变持久化格式，无需迁移。
- 修复前7 failed（`/tmp/llmtuner-scheduler-state-type-before.log`）；修复后scheduler suite 51 passed（`/tmp/llmtuner-scheduler-state-type-after.log`）。完整CPU 3659 passed、2 skipped、267 warnings、40.20s（`/tmp/llmtuner-scheduler-state-type-full.log`，原session70793最终exit0）。
- 适用文件全量pre-commit通过（`/tmp/llmtuner-scheduler-state-type-final-gates.log`，排除受保护的examples/native_alltoall.py及PPTX）；compileall及git diff HEAD --check通过，自动检查无新增源码修改。
- 类型校验及状态不变验收已通过；不作为DCP跨rank故障协调、目标设备或混合并行的验证证据。全局审查仍未完成。

### MC2-DISTRIBUTED-MULTISTEP-REVIEW-05

- 完整复读expert_parallel/mc2_ops.py、mc2_dispatch.py及分布式算子适配脚本。当前MC2×TP的FP16失败继续保持开放，未修改融合公式或放宽容差；本轮未确认新的生产缺陷。
- 扩展tests/integration_tests/mc2_distributed_adapter.py：同一MC2RoutedExperts实例连续三次momentum SGD，每次独立构图、清梯度，与无EP的全量LocalTokenDispatcher/GroupedExperts参考比较输出、输入/score/专家梯度、更新参数及momentum_buffer。保留原atol5e-4/rtol1e-2。
- 覆盖不均匀输入、rank0无输入、rank1无接收，以及新增所有rank均无token四场景。实际 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29877 tests/integration_tests/mc2_distributed_adapter.py` exit0，全部场景三步通过（`/tmp/llmtuner-mc2-adapter-multistep.log`）。
- 进一步在普通场景step1将路由全部切到rank0、step2恢复，验证已有动量的rank1暂时无接收再恢复；以port29878重跑四场景exit0（`/tmp/llmtuner-mc2-adapter-multistep-dynamic.log`）。完整CPU3659 passed、2 skipped、279 warnings、40.50s（`/tmp/llmtuner-mc2-adapter-multistep-full.log`）；全量适用pre-commit及最终三文件pre-commit通过，compileall/diff通过。仅测试和文档变更。
- 证据为CPU/Gloo、FP16、EP2、TP/PP/CP1，无FSDP/AC/compile/EMA，真实all-to-all及生产MC2 autograd，融合核由DistributedOps模拟；全量参考复用基础专家实现，不是独立HF全模型。不能关闭MC2-TP-PUBLIC-01或Ascend融合内核验收。

### VALIDATION-TOKEN-COORDINATION-01

- 优先级P1；位置trainer/runtime/validation.py::count_validation_batch_tokens，由run_validation_loop在每个验证batch进入DP token归约前调用。labels.numel/metrics/count_valid_tokens/张量构造在本地执行，若单rank异常，其余rank直接进入DP all-reduce而挂起。fetch_batch只检查numel，标量Batch.labels可触发此路径；其他本地计数/metrics异常也存在同一边界。
- 将上述本地操作放入synchronize_local_errors("Validation token accounting")，DP collective保留在协调区间之外。故障侧原异常继续抛出，peer在token归约前报RuntimeError；不声称可恢复通信自身错误。
- 在data_fetch_coordination.py加入真实两rank DP mesh，分别rank0/1收到标量labels，验证故障侧IndexError、peer精确错误消息、barrier；随后正常batch全局有效token计数应为6，确认group仍可使用。原训练/验证数据、初始化、结果与cleanup故障断言保留。
- 修复前真实两进程exit1：peer token all-reduce与故障侧barrier均Gloo 10s timeout（`/tmp/llmtuner-validation-token-before.log`），不是外部环境失败。修复后原脚本以port29880运行exit0（`/tmp/llmtuner-validation-token-after.log`）；新增成功计数断言后以同命令port29881最终运行exit0（`/tmp/llmtuner-validation-token-final.log`），错误后正常token归约为6。
- 完整CPU3659 passed、2 skipped、279 warnings、39.71s（`/tmp/llmtuner-validation-token-full.log`）；适用文件全量pre-commit通过（`/tmp/llmtuner-validation-token-gates.log`），compileall及diff检查通过。保护文件与外部改动保留。
- CPU/Gloo、两rank DP、无模型前反向的故障回归；不替代PP/CP/TP模型collective错误恢复或设备验证。全局审查仍开放。

### VALIDATION-TOKEN-LOOP-LIFECYCLE-REVIEW-05

- 对VALIDATION-TOKEN-COORDINATION-01补充实际run_validation_loop调用链验收，避免只用直接helper调用证明loader生命周期。完整复读validation.py，并检查utils/seeding.py的validation RNG隔离契约；本轮未确认新的生产缺陷。
- 在data_fetch_coordination.py新增失败rank0/1 × count/metrics.add_tokens × ValueError/KeyboardInterrupt/SystemExit共12场景。使用真实fetch及token协调，故障侧异常对象身份不变，peer精确报Validation token accounting错误，preprocess/finalize均不得调用，每侧临时loader恰好close一次；每场景barrier确认无遗留未匹配通信。
- `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29882 tests/integration_tests/data_fetch_coordination.py` exit0（`/tmp/llmtuner-validation-token-loop.log`），含原脚本所有fetch/preprocess/validation/cleanup错误检查。仅测试与台账变更，生产代码保持上一批已验证版本。
- validation CPU单测66 passed、3.93s（`/tmp/llmtuner-validation-token-loop-units.log`），脚本pre-commit及diff通过。本轮仅增加显式torchrun脚本断言，复用上一批完整CPU3659 passed结果，不将66项单测视为全仓验收。
- CPU/Gloo DP2，错误发生于模型前本地操作；不证明collective中断、进程崩溃或设备恢复，也不替代Trainer.validate外层模式/RNG恢复的已有独立单测。全局审查保持开放。

### KCP-MLA-GEOMETRY-01

- 优先级P1；位置parallel/context_parallel/kcp.py::kcp_mla_attention。与KDA已有检查不同，MLA仅校验本地mask和heads整除，未保证CP rank的batch/head/sequence/QK与V维度相同；可进入不匹配的mask all_gather或Ulysses all-to-all。本轮模式切换检查未确认生产缺陷，继续审查CP时确认此通信前契约缺口。
- 新增本地Q/K/V形状对齐检查，并在payload交换前用固定五整数metadata和MIN归约核对跨rank几何，全部rank一致报ValueError。collective不放入本地错误协调区间，不引入不等长分片支持。
- 扩展kimi_mla_cp_equivalence.py：两rank分别合法但batch、heads、sequence、query_dim、value_dim之一不同，共五场景。真实metadata collective，payload all_gather设为必须不可达的探针，每场景后真实barrier。修复前exit1，两rank均触发payload探针（`/tmp/llmtuner-kcp-mla-geometry-before.log`，port29883）；修复后exit0（`/tmp/llmtuner-kcp-mla-geometry-after.log`，port29884），并保留原16组MLA数值、mask及KDA错误边界测试。
- 运行命令 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29884 tests/integration_tests/kimi_mla_cp_equivalence.py`。CPU/Gloo FP32、CP2、无TP/PP/EP/FSDP/AC/compile；有效路径为真实Ulysses通信，数值参考为全序列eager KimiLatentAttention，原输出/输入梯度/参数梯度容差1e-6/2e-6/3e-6不变。
- 完整CPU3659 passed、2 skipped、271 warnings、39.71s（`/tmp/llmtuner-kcp-mla-geometry-full.log`）；全量适用pre-commit通过（`/tmp/llmtuner-kcp-mla-geometry-gates.log`），compileall/diff通过。最终仅拆分长错误字符串以满足88列，无语义改变。
- 本修复只关闭MLA几何预检缺陷，不关闭KCP-PADDING-STATE-01，不证明KDA Ascend状态内核、性能或完整混合并行通过。

### TRAINER-CP-TP-SEQUENCE-MESH-01

- 优先级P1；trainer/runtime/reductions.py::resolve_training_reduction_meshes返回CP×TP二维sequence mesh，trainer.py::_step_metrics将其作为一维mesh调用get_group()，真实TP2×CP2首次日志汇总即报DeviceMesh二维错误。原单轴单测只使用object，无法揭示该契约错误。
- 对多轴sequence mesh显式_flatten("sequence_loss")，单轴保持原mesh；更新16组合单测检查flatten及返回归约mesh，17项通过。未修改模型计算、梯度或持久化格式。
- 修复前四rank实际Trainer full阶段exit1（`/tmp/llmtuner-tp-cp-restore-full.log`）。修复后full/resume均exit0（`/tmp/llmtuner-tp-cp-restore-fixed-{full,resume}.log`），四rank更新前state failures=0，恢复后两步loss差0，最终状态通过。连续loss=[4.9923529624938965,4.5286335945129395,4.273299217224121,4.131411552429199]。
- 命令：`PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 LLMTUNER_TEST_WORLD_SIZE=4 LLMTUNER_TEST_PP_SIZE=1 LLMTUNER_TEST_TP_SIZE=2 LLMTUNER_TEST_CP_SIZE=2 LLMTUNER_TEST_SEQ_LEN=256 LLMTUNER_TEST_ACCUMULATION=2 LLMTUNER_TEST_EMA=dynamic LLMTUNER_TEST_FSDP_IMPLEMENTATION=custom LLMTUNER_TEST_RNG_STATE=1 LLMTUNER_TEST_LOADER_STATE=1 torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29887 tests/integration_tests/pp_checkpoint_equivalence.py full WORKDIR --packed`；resume改phase及port29888。WORKDIR为`/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-tp-cp-restore-fixed-mgcnpgsl`。
- 完整CPU3659 passed、2 skipped、264 warnings、37.82s（`/tmp/llmtuner-sequence-mesh-full.log`）；适用全量pre-commit通过（`/tmp/llmtuner-sequence-mesh-gates.log`），compileall及diff通过，审阅自动检查无额外源码修改。
- CPU/Gloo FP32 Qwen3，TP2×KV-CP2、custom FSDP、packed local JSONL、累积2、动态EMA、AC/compile关；真实TP/CP/FSDP/DCP，attention数值使用cpu_cp_attention显式SDPA替身。更新前及最终比较模型/AdamW/scheduler/EMA/trainer/数据游标/Python、NumPy、CPU RNG。连续恢复一致不等于独立未并行模型正确性或设备flex验证；全局目标保持开放。

### TRAINING-METRIC-REFERENCE-REVIEW-05

- 复读Trainer._step_metrics及runtime/reductions，新增独立Python sum/max参考的真实四rank脚本training_metric_reduction_equivalence.py。使用生产ParallelTopology及归约函数，分别CP2×TP2、DP2×CP2、DP2×TP2，核对loss/max_loss/n_tokens_seen/grad_norm/LR；DP2另外覆盖一组全部标签masked，所有rank仍参与通信。全局无有效token不进入metrics，保持训练前置拒绝契约。
- 初版夹具误对单例DP调用get_mesh，按真实API改用可选mesh；随后10/3精确断言出现1ULP求和差，日志 `/tmp/llmtuner-training-metric-reduction{,-final}.log`保留。改用4/8有效计数使预期指标二进制精确可表示，保留精确断言，未改生产公式或放宽容差；不作为生产bug。
- `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29891 tests/integration_tests/training_metric_reduction_equivalence.py` exit0（`/tmp/llmtuner-training-metric-reduction-exact.log`），五项有效场景所有rank精确通过。CPU/Gloo FP64显式scalar shard losses，真实生产拓扑/collective，但不是模型前反向或目标设备证据。
- 仅新增显式integration脚本及台账；复用上一批生产版本全量CPU3659 passed结果，不因测试夹具变化重复训练suite。全局审查及完整并行矩阵仍开放。

### TRAINER-REDUCTION-FIELDS-01 / TP-CP-FULL-AC-RESTORE-REVIEW-05

- 当前工作区外部将TrainingReductionMeshes字段从dp/pp/loss/sequence重命名为*_mesh，Trainer及测试消费者尚未同步。本轮原TP2×CP2 full-AC运行在首步报AttributeError（`/tmp/llmtuner-tp-cp-ac-full.log`），发生于AC执行前，不作为重计算缺陷。保留新的命名及暂存状态，同步Trainer七处访问、CPU测试和独立指标脚本，不添加旧名兼容壳。
- 修复后完整CPU3659 passed、2 skipped、261 warnings、53.03s（`/tmp/llmtuner-mesh-fields-full.log`）。首次pre-commit指出长断言并自动格式化，最终全量适用pre-commit通过（`/tmp/llmtuner-mesh-fields-final-gates.log`）；compileall/diff通过。
- 实际四rank CPU/Gloo FP32 Qwen3 TP2×KV-CP2/custom FSDP、packed JSONL、累积2、动态EMA、full AC、compile关。沿用pp_checkpoint_equivalence.py full/resume --packed及上一批环境，新增LLMTUNER_TEST_AC=full，RNG_STATE/LOADER_STATE=1，ports29894/29895。WORKDIR=`/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-tp-cp-ac-final-ov6ovrdp`。
- full/resume均exit0（`/tmp/llmtuner-tp-cp-ac-final-{full,resume}.log`），四rank更新前模型/AdamW/scheduler/EMA/trainer/数据游标/Python、NumPy、CPU RNG比较failures0，恢复后两步loss差0，最终state通过。连续loss=[4.9923529624938965,4.5286335945129395,4.273299217224121,4.131411552429199]，与此前无AC轨迹相同；仅轨迹相同不能替代全部无AC梯度对照。
- 真实TP/CP/FSDP/AC/DCP路径；attention使用显式CPU SDPA替身，无目标设备flex/compile性能证据。本项不关闭其他并行矩阵缺口，全局审查仍开放。

### TP-CP-SELECTIVE-AC-RESTORE-REVIEW-05

- 复读memory/activation_checkpoint.py的逐op policy、mm形状、checkpoint context及装配顺序，并对照本地torchtitan当前activation_checkpoint.py。上游SelectiveAC已迁为RegionAC预设，现有upstream_map已记有意分叉；修正模块开头“全部四策略已移植”的笼统描述，明确本仓库selective/memory_budget保留历史实现，不改policy行为。
- 真实四rank CPU/Gloo FP32 Qwen3 TP2×KV-CP2/custom FSDP、packed JSONL、累积2、动态EMA、selective AC、compile关。使用pp_checkpoint_equivalence.py full/resume --packed，环境同TP-CP-FULL-AC-RESTORE-REVIEW-05，AC改selective，ports29901/29902，RNG_STATE/LOADER_STATE=1。
- WORKDIR=`/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-tp-cp-sac-lmkxq76g`。full/resume均exit0（`/tmp/llmtuner-tp-cp-sac-{full,resume}.log`），四rank更新前完整状态failures0，恢复后两步loss差0，最终状态通过；连续loss=[4.9923529624938965,4.5286335945129395,4.273299217224121,4.131411552429199]。模型/AdamW/scheduler/EMA/trainer/数据游标/Python、NumPy、CPU RNG均纳入快照比较。
- attention为显式CPU SDPA替身，真实TP/CP/FSDP/SAC/DCP路径；不证明设备flex/内核性能或与当前上游region selective语义等价。仅说明文字及证据变更，复用现有CPU全量结果，不重复训练suite；全局目标仍开放。

### COMPILE-CONFIG-ASSEMBLY-REVIEW-05

- 复读parallel/stages.py、config/cli_config.py、config/compile_config.py、parallel/model_compile.py，并追踪runtime/assembly.py::_apply_parallelism至parallelize.py：主模型与PP parts共用STAGE_ORDER，compile_config、TP mesh、AC选项传入实际变换；CLI明确隐藏嵌套programmatic-only选项，不假设存在对应标量flags。
- 核对backend注册解析、per-block身份保持、regional标记归属、MoE scalar capture、async TP能力/mesh检查，以及失败时标记/编译调用/global flag回滚。本轮未确认生产缺陷，不作无收益重命名或增加封装。
- 候选探针手动把CompileConfig(backend=eager)修改为async=True后可先进入async setup；正常__post_init__已禁止该组合，不能证明合法配置错误。探针已撤回，源码与测试无该项变更，失败日志`/tmp/llmtuner-compile-preflight-before.log`保留，不计入bug。
- `pytest -q tests/unit_tests/cpu/parallel/test_model_compile.py --disable-warnings` exit0：46 passed、2 skipped、2 warnings、9.71s（`/tmp/llmtuner-compile-config-review.log`）。两skip为当前torch具备相应能力，故不适用的“缺能力”反向场景；不是设备async overlap性能验证。本轮仅审查记录，完整目标仍开放。

### DATA-SOURCE-BYTE-OFFSETS-REVIEW-05

- 完整复读datasets/sources.py，核对惰性source描述、HF显式字段冲突/类型检查、DP streaming先分片、每cursor独立deepcopy、epoch恢复/失败关闭及JSONL紧凑byte offset；追踪loader_builder.py::corpus_data_identity的本地内容及tokenizer指纹、声明snapshot和prefetch深度排除。本轮未确认新生产缺陷；构造期指纹不是文件修改锁，不扩大其保证。
- 新增test_jsonl_byte_offsets_preserve_multifile_unicode_rows：两文件逆文件名顺序、UTF-8中文/emoji/日文/韩文/重音字符、CRLF/空白行/末行无换行，非顺序读取、负索引、越界及重新构造索引均与独立预期dict相同。确认偏移为字节而非字符，文件pattern顺序保留，不依赖已解码row缓存。
- `pytest -q tests/unit_tests/cpu/datasets/test_data_pipeline.py --disable-warnings` exit0，345 passed、10 warnings、9.80s（`/tmp/llmtuner-jsonl-byte-offset-review.log`）；受影响pre-commit及diff通过。仅CPU本地文件读取与索引契约，不作为完整DCP、远程数据变更或分布式恢复证明；全局目标仍开放。

### SFT-EXPLICIT-TOKEN-CONTRACT-REVIEW-05

- 完整复读datasets/text/processors.py、packing/conversions.py，并追踪text/recipes.py的single-turn/multiturn选择与post_filter。单轮模板显式关闭generation prompt，追加但不重复EOS，要求prompt精确token prefix，labels屏蔽len(prompt)-1；超长按已声明策略丢弃而非截断。renderer路径由外部loss_mask决定监督，不强推单轮prefix契约。本轮未确认生产缺陷。
- 新增test_sft_supervision_matches_explicit_token_contract四场景：独立Tokenizer固定prompt=[1,2,3]、response=[7,8]、EOS=9；已带/未带EOS × limit4/5。limit5精确检查inputs=[1,2,3,7,8]、labels=[IGNORE,IGNORE,7,8,9]，limit4必须丢弃。避免仅从生产tokenizer重新推导预期而遗漏首响应token或EOS重复问题。
- `pytest -q tests/unit_tests/cpu/datasets/test_data_pipeline.py --disable-warnings` exit0：349 passed、10 warnings、9.37s（`/tmp/llmtuner-sft-explicit-contract.log`）；受影响pre-commit与diff通过。仅测试及台账变更；不作为可选renderers包、真实BPE跨边界或完整分布式SFT训练验收。全局目标仍开放。

### MULTIMODAL-COLLATION-CONSUMER-REVIEW-06

- 重新读取目标文件并核对当前工作区；本轮 HEAD 为 `3063113`，已有文档、
  prompt 及外部文件改动保留，未修改 index、未 commit/push。
- 复核 `datasets/multimodal/{collator,packing,mrope}.py` 及
  `models/hf/model.py` 的实际输入消费边界：源 padding 和 packing padding
  合并后屏蔽标签；whole-document 媒体列表顺序保持；MRoPE 按文档重置，
  分别消费 image/video grid，并检查 placeholder 长度与空间整除。
- 当前 HF 文本 decoder 明确拒绝非空媒体、grid 与三轴 MRoPE 元数据，
  不存在本轮可确认的静默丢弃媒体问题。本轮未确认新生产缺陷，未修改实现，
  未新增与既有覆盖重复的测试。
- 验证命令：`pytest -q tests/unit_tests/cpu/datasets/test_mrope.py
  tests/unit_tests/cpu/datasets/test_multimodal_build.py
  tests/unit_tests/cpu/datasets/test_multimodal_data.py
  tests/unit_tests/cpu/models/test_multimodal.py --disable-warnings`。
  结果：210 passed、61 warnings、5.49s、exit 0；日志
  `/tmp/llmtuner-multimodal-continuation-review.log`。
- 此证据覆盖 CPU 数据处理和模型入口拒绝契约，不证明真实媒体编码器训练、
  目标设备内核或混合并行 VLM 数值等价。完整 VLM 消费端仍需专用模型适配，
  全局审查与未完成迁移继续保持开放。

### CHECKPOINT-STAGING-SUCCESS-CONTRACT-REVIEW-06

- 复核 `components/checkpointer/distributed_checkpoint.py` 的
  `CoordinatedStager`、DCP save 分派与 staging/upload 等待，
  `base.py` 的 close 重试及 retention 发布顺序，并阅读
  `checkpoint_writer.py` 和 `torch_checkpointing_backend.py` 相关实现。
  已有失败测试覆盖普通异常/中断、活跃 future 所有权及清理；未确认新生产缺陷。
- 新增 `test_staging_success_preserves_snapshot_and_completion_boundary` 三例，
  覆盖 stager 直接返回快照、已完成 Future、后台完成 Future。验证待完成时
  upload/completion 均未提前完成，上传结果保留快照对象身份、完成通知为 None，
  同步等待和底层 close 转发正确；不宣称证明真实设备快照隔离或跨 rank 通信。
- 修改前 checkpointer 专项 206 passed、105 warnings、2.12s；新增测试单独
  9 passed、1.11s；修改后完整 checkpointer 专项 209 passed、105 warnings、
  2.02s、exit 0。命令 `pytest -q tests/unit_tests/cpu/components/checkpointer
  --disable-warnings`；最终日志 `/tmp/llmtuner-checkpointer-final-review-06.log`。
- 受影响测试的 pre-commit 全部适用检查通过，日志
  `/tmp/llmtuner-staging-success-gates-06.log`；审阅 diff，无自动修改。
  本轮仅测试与台账变更，无生产实现改动，未重跑训练或全局 CPU suite。
  可选 torch_checkpointing 实际 I/O、设备 pinned staging 与多轴恢复缺口保留。

### FSSPEC-RETENTION-DRAIN-REVIEW-06

- 复核 `components/checkpointer/{filesystem,storage}.py` 与 `base.py`
  的发现、目录识别、retention 分类和删除线程；未确认新生产缺陷。
- 新增 `test_fsspec_retention_drains_deletions_and_keeps_latest_loadable`
  四例：memory/file URI × disabled/async。真实保存三个 step，使用合法
  keep_latest_k=2，close 后验证只剩 step-2/step-3、删除线程退出，
  原生 DCP reader 读取 step-3 并逐张量零容差比对最终权重。
  验证没有模拟删除队列，也没有依赖 sleep。
- 首轮 keep_latest_k=1 被现有配置检查拒绝：4 failed、209 passed，日志
  `/tmp/llmtuner-fsspec-retention-review-06.log`；属于测试夹具非法，修正为
  既有合法契约，未变更生产校验或降低断言。
- 最终 `pytest -q tests/unit_tests/cpu/components/checkpointer
  --disable-warnings`：213 passed、115 warnings、2.18s、exit 0，日志
  `/tmp/llmtuner-fsspec-retention-final-06.log`。受影响文件 pre-commit
  通过，日志 `/tmp/llmtuner-fsspec-retention-final-gates-06.log`。
- 仅测试及台账改动；证据为单进程真实 fsspec memory/local URI，
  不证明 S3/GCS 一致性、通信故障或跨拓扑完整训练恢复。全局目标保持开放。

### RNG-PREUPDATE-RESTORE-REVIEW-06

- 复核 `components/checkpointer/random_state.py`、`utils/seeding.py`，追踪
  DCP 恢复最后阶段 RNG 校验/应用和 validation 的随机数隔离；未确认新缺陷。
- 增强 `tests/integration_tests/rng_checkpoint_equivalence.py`：在保存点保留
  独立 model/optimizer/scheduler/RNG 快照，新运行时 load 后、首次更新前
  精确核对全部上述状态，再沿用后两步随机 loss/参数/optimizer 零容差比较。
- 首轮直接比较 pickle bytes 失败，日志
  `/tmp/llmtuner-rng-boundary-review-06.log`。张量存储的序列化字节不能作为
  RNG 状态等价判断，改为反序列化后逐字段精确比较，未放宽数值容差。
- 最终命令：`PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0
  TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=2
  --master_addr=127.0.0.1 --master_port=29924
  tests/integration_tests/rng_checkpoint_equivalence.py`，exit 0，DDP/native
  FSDP/custom FSDP × disabled/async 六种场景全部恢复边界与后续轨迹精确。
  同时保留既有 rank 局部 scheduler/model/RNG 故障协调测试。
- 最终日志 `/tmp/llmtuner-rng-boundary-final-06.log`；受影响文件 pre-commit
  全适用项通过，日志 `/tmp/llmtuner-rng-boundary-gates-06.log`。
  测试过程中外部暂存了该文件的部分改动，保留 index 原样，未 commit/push。
- 此为 CPU/Gloo 两 rank 默认流与新模型/optimizer 运行时，不证明设备 RNG、
  新 OS 进程恢复、loader 私有 generator、拓扑变化或多轴矩阵；缺口保持开放。

### CLI-FRESH-PROCESS-LIFECYCLE-REVIEW-06

- 当前 HEAD `3063113`；重新核对 `pyproject.toml` console script、
  `llmtuner/__main__.py`、`trainer/cli.py`、`trainer/runtime/assembly.py` 和
  Trainer 构造失败/close 边界。模块入口与 console script 均调用 cli.main；
  Trainer 清理拥有的训练资源，CLI 最终销毁进程组并保留主要异常。
  未确认新的生产缺陷，未增加重复封装或测试。
- `pytest -q tests/unit_tests/cpu/test_trainer.py -k
  'cli or failed_trainer_construction' --disable-warnings`：28 passed、
  112 deselected、3.93s、exit 0；日志
  `/tmp/llmtuner-cli-lifecycle-review-06.log`。
- `PYTHONPATH=. python tests/integration_tests/cli_checkpoint_equivalence.py`：
  exit 0；实际三个独立 Python CLI 进程执行连续三步、初始两步、恢复到三步，
  最终 model/optimizer/scheduler/loader/trainer/RNG 完整语义状态精确一致。
  检查脚本确实反序列化 RNG 并比较 NumPy/torch 内容，非 opaque pickle bytes。
  trainer 最终 step=3、ntokens_seen=48，scheduler last_epoch=3。
- 日志 `/tmp/llmtuner-cli-current-review-06.log`；真实运行产物保留在
  `/var/folders/lv/pf690zrn1k55fm2chddvpkx40000gn/T/llmtuner-cli-resume-261n4k1v`，
  包含三个 CLI 日志及独立 checkpoint。测试证明单 CPU eager Qwen3/synthetic
  的跨进程最终状态等价，不证明多轴、新进程首次更新前完整状态或设备内核。
- 本轮仅台账更新，无生产/测试修改；保留全部外部暂存内容，不 commit/push。
  全局审查及既有未完成迁移继续开放。

### TP-EP-CP-DECLARATION-REVIEW-06

- 复核 `parallel/compatibility.py` 的 assembly/probe verdict、实际调用位置，
  `parallel/expert_parallel/hf_inspection.py` 的布局/路由契约，以及
  `config/parallel_config.py` 的 dispatcher 与 fixed router 组合限制。
  未确认新缺陷；DeepSeek-V2 group_limited_greedy、GPT-OSS、shared gate 等
  拒绝仍对应实际缺失语义，未把拒绝视为完成迁移。
- `pytest -q tests/unit_tests/cpu/parallel/test_compatibility.py
  --disable-warnings`：6 passed、0.82s、exit 0；日志
  `/tmp/llmtuner-compatibility-review-06.log`。包含 AST 核对 guard 所指函数
  确实调用对应 verdict；不证明 guard 触发条件全部正确。
- 当前代码真实四 rank：`PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0
  TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=4
  --master_addr=127.0.0.1 --master_port=29925
  tests/integration_tests/tp_ep_cp_equivalence.py`，exit 0，all checks passed。
  Qwen3MoE/TP2/KVCP2/EP2、world4、seq256、8experts、float64，EP 复用
  现有 rank 轴而非额外相乘。TP 替换/专家持有数均检查，拼接 logits 与
  summed loss 对照独立未分片 HF 模型，原有容差 1e-6 保持。
- 日志 `/tmp/llmtuner-tp-ep-cp-current-06.log`。CPU eager flex 的前向证据，
  不包含 backward、参数更新、FSDP、PP 或设备性能，不能关闭完整三轴训练
  验收项。未修改生产或测试，仅更新台账；保留 index 和外部文件。

### TP-EP-CP-CE-TRAINING-REFERENCE-06

- 新增 `tests/integration_tests/tp_ep_cp_training_equivalence.py`，通过生产
  parallelize_hf_transformers 装配 TP2/KVCP2/EP2/FSDP，四 CPU/Gloo rank。
  独立未分片 HF Qwen3MoE 参考，FP64、seq256、2层、8 experts，三步 SGD
  momentum=0.9。输入归属独立 contiguous slicing；每 rank 27 参数逐一比较
  完整梯度、更新后权重及 momentum，logits 和全局 CE loss 同时核对，
  atol/rtol 均为 1e-6，未因失败放宽。DTensor.full_tensor 只重建 FSDP 维，
  TP/EP 参考切片由独立 HF 张量命名/布局映射决定。
- 显式使用 train 模式。最初共享 _build 默认 eval 的运行仅证明求导，不能
  当作 train 模式验收；切换 train 后缺少 AuxLoss step denominator 的测试
  夹具被现有检查拒绝，日志
  `/tmp/llmtuner-tp-ep-cp-training-{custom,native}-train-06.log`。
  本验证目标为 CE：明确在两侧配置 router_aux_loss_coef=0，而非修改生产
  机制或伪造辅助目标。带 aux 的三轴训练仍未验收，不将该缺口关闭。
- attention 运算复用 pp_cp_equivalence.cpu_cp_attention 的显式 dense-mask
  SDPA 替代；真实 CP mask/collective/autograd、TP、EP alltoall 和 FSDP
  保留，不能作为 production flex backward 或设备内核证明。
- 最终命令前缀 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0
  TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=4
  --master_addr=127.0.0.1`，脚本参数分别 custom/native；端口 29931/29932。
  两次 exit0，四 rank 各三步全部27个 gradients/updates/momentum passed；
  日志 `/tmp/llmtuner-tp-ep-cp-training-{custom,native}-ce-06.log`。
- 新脚本 pre-commit 全适用项通过（首轮 import/format 与 lambda 捕获提示
  已修正并复核），日志 `/tmp/llmtuner-tp-ep-cp-training-ce-gates-06.log`；
  diff 检查通过。本批无生产实现改动，不重复运行无关全局单测。
  未包含 Trainer orchestration、aux loss、AC/compile、PP 或完整恢复。

### TP-EP-CP-AUX-TRAINING-REFERENCE-06

- 在上一批 CE 场景上扩展 `tp_ep_cp_training_equivalence.py`：可传第二参数
  aux，配置 router_aux_loss_coef=0.01，显式 train 模式，按真实 Trainer
  契约设置全局 routed-token 分母并进入 spmd_context，使 CP/TP 统计归约生效。
- 独立参考由未分片 HF router 的 logits/selected experts 构造：FP32 softmax，
  全流 bincount，f=E*counts/(K*T)，p=mean(probabilities)，每层 sum(f*p)，
  CE + 0.01*sum(layer_aux) 直接求导。未复用生产 AuxLoss 或其归约/注入。
  每层生产 instance_acc 对照独立公式；参考辅助梯度的最大贡献要求 >1e-6，
  防止辅助项无效仍通过。所有27参数梯度、更新及momentum保持atol/rtol1e-6。
- 最终四 CPU/Gloo rank、TP2/KVCP2/EP2、FP64、两层Qwen3MoE、三步SGD：
  custom/native FSDP 均 exit0，四rank各三步全部27项梯度/更新/momentum通过。
  命令沿用前批 torchrun 前缀，端口29934/29935，脚本参数 custom aux /
  native aux；日志 `/tmp/llmtuner-tp-ep-cp-aux-{custom,native}-final-06.log`。
- 修改后原CE custom场景另行复验exit0，端口29936，日志
  `/tmp/llmtuner-tp-ep-cp-aux-ce-regression-06.log`。pre-commit适用项全部
  通过，日志 `/tmp/llmtuner-tp-ep-cp-aux-final-gates-06.log`；仅format自动
  调整，已审阅；diff检查通过。保留外部暂存版本，不修改index或commit/push。
- 本批补充上一批明确开放的“带aux三轴更新”具体CPU场景证据，未确认生产
  缺陷。仍使用显式dense-mask SDPA attention替代，不证明设备flex backward、
  梯度累积/packing/padding、AC/compile、Trainer完整编排、PP或恢复矩阵。

### TP-EP-CP-PADDING-DENOMINATOR-REFERENCE-06

- 扩展 `tp_ep_cp_training_equivalence.py` 的 aux padding 场景：尾部64 token
  和内部16:24标为padding，首16 token为masked prompt。独立断言路由有效
  数184、CE有效标签数168；cp1/tp1 的64个本地token全部padding且标签忽略。
- 并行端调用真实 HFTransformerModel.forward(padding_mask=...)；辅助分母
  为184，CE分母168。独立 HF 参考对全部token执行模型，但均衡公式只统计
  非padding的HF路由结果，prompt仍计入，避免把CE掩码当路由掩码。生产
  每层aux指标与独立公式、27参数的梯度/更新/momentum、logits/loss均核对。
- 四CPU/Gloo rank TP2/KVCP2/EP2，FP64，两层Qwen3MoE，aux系数0.01，
  native/custom FSDP各三步均exit0；保持atol/rtol1e-6。命令沿用既有前缀，
  脚本参数 custom aux padding / native aux padding，端口29937/29938。
  日志 `/tmp/llmtuner-tp-ep-cp-padding-{custom,native}-06.log`。
- 受影响脚本pre-commit全适用项通过（首轮仅格式调整，已审阅），日志
  `/tmp/llmtuner-tp-ep-cp-padding-final-gates-06.log`；diff检查通过。
  本批未确认生产缺陷，保留已有index及外部文件，未commit/push。
- 此为padding统计与分母区分的具体三轴训练证据：padding不代表token不派发，
  不证明物理空输入rank、真实packer、梯度累积、Trainer编排、AC/compile、
  设备flex或完整恢复；这些要求保持开放。

### CUMULATIVE-QUALITY-GATES-REVIEW-07

- 重新读取目标、核对当前工作区，保留外部暂存版本。本轮针对最近新增的
  staging成功、fsspec retention、RNG恢复边界、三轴训练脚本执行累计门禁，
  不以专项通过替代完整CPU回归。
- `pytest -q --disable-warnings`：3671 passed、2 skipped、283 warnings、
  41.28s、exit0，日志 `/tmp/llmtuner-cumulative-full-07.log`。此命令未输出
  skip明细，不将两个skip计作验证通过；现有compile测试包含环境能力已存在
  时跳过不可达拒绝路径的门控。未据此推断设备或分布式脚本已由pytest执行。
- `python -m compileall -q llmtuner` 与 `git diff HEAD --check` exit0。
  对全部已跟踪适用文件运行pre-commit，排除受保护
  examples/native_alltoall.py，全部检查通过，日志
  `/tmp/llmtuner-cumulative-final-gates-07.log`。
- 首轮将未跟踪文件也纳入检查，end-of-file-fixer自动调整外部pr.md与
  hyperparallel_pr.md末尾换行，日志 `/tmp/llmtuner-cumulative-gates-07.log`；
  已向用户说明，此后改用已跟踪文件集合，未继续处理外部文档内容。
  本轮未修改生产实现或测试，未修改index、未commit/push。
- 累计质量门禁通过不代表全局目标完成：真实设备、迁移机制、声明多轴矩阵
  与完整恢复等未验收项继续开放。后续针对具体未完成契约推进，避免重复
  无新增覆盖的全局测试。

### TP-EP-CP-AC-AUX-PADDING-REFERENCE-07

- 扩展三轴训练脚本的full/selective AC参数，生产parallelize入口应用AC，
  独立HF参考不加AC。canonical_fqn消除checkpoint包装路径差异；新增
  两层CheckpointWrapper断言，以及backward后aux指标不重复累加检查。
- TP2/KVCP2/EP2、四CPU/Gloo rank、aux0.01、padding/提示词双分母，
  native/custom FSDP full AC均三步exit0，原27参数梯度/权重/momentum与
  独立HF参考通过，atol/rtol1e-6。日志
  `/tmp/llmtuner-triaxis-ac-{custom,native}-07.log`，端口29939/29940。
- selective首轮未传配置，被既有检查拒绝，属于测试装配遗漏：
  `/tmp/llmtuner-triaxis-sac-{custom,native}-07.log`。修正为传递
  cfg.training.selective_ac，未绕过检查或替换重计算机制。
  最终两种引擎均三步exit0，包装数量/全部梯度/更新/momentum/aux指标通过，
  日志 `/tmp/llmtuner-triaxis-sac-{custom,native}-final-07.log`，端口29943/29944。
- 受影响脚本pre-commit通过，日志
  `/tmp/llmtuner-triaxis-ac-clean-gates-07.log`；diff检查通过。
  无生产代码改动，保留外部暂存内容，未commit/push。
- CPU显式SDPA attention替代保留，未覆盖设备flex、累积、compile、
  Trainer完整编排或checkpoint恢复。本结果仅补充具体AC训练组合证据。

### TP-EP-CP-AC-REPLAY-NONVACUITY-07

- 三轴独立参考脚本新增decoder body forward_pre_hook计数，hook位于
  checkpoint wrapper内部；每step记录增量，none必须每层一次，full/selective
  必须每层至少两次。结合既有包装数量断言和backward后aux指标检查，
  排除只安装包装但未重计算、或重复指标未被发现的弱证据。
- 四CPU/Gloo rank、TP2/KVCP2/EP2、custom FSDP、aux0.01、padding，
  full/selective各三步exit0，每rank每step实际layer calls=[2,2]；
  none对照三步exit0、calls=[1,1]。全部27参数梯度、更新、momentum及
  独立HF辅助指标仍保持atol/rtol1e-6通过。
- 命令沿用三轴脚本torchrun前缀，端口29945/29946/29947，参数分别
  custom aux padding full / selective / 不带AC；日志
  `/tmp/llmtuner-ac-replay-{full,selective,control}-07.log`。
- 首轮pre-commit发现打印行超长，拆行后适用项全部通过，日志
  `/tmp/llmtuner-ac-replay-final-gates-07.log`，diff检查通过。
  无生产实现修改，无index操作，保留外部暂存内容；未commit/push。
- 此次增加机制触达证据，不扩展此前CPU SDPA替代的设备覆盖；未证明
  native FSDP新计数断言、梯度累积、compile或完整恢复。相关缺口保持开放。

### TP-EP-CP-ACCUMULATION-AC-REFERENCE-07

- 三轴独立参考脚本新增末尾accumulation参数（默认1），每step使用不同seed
  的两个microbatch，optimizer只更新一次；生产aux分母仅step开始设置，
  为两批路由token总数368，CE分母336。参考CE和每批sequence-wise aux按
  step权重累计；指标逐microbatch对照累计独立公式，backward后再次比较。
- 四CPU/Gloo rank、TP2/KVCP2/EP2、FP64、aux0.01、padding/masked prompt，
  native/custom FSDP × full/selective AC四场景均三步exit0。每rank每step
  两层body各执行4次，全部27参数累计梯度、权重、momentum与独立HF参考
  通过，atol/rtol1e-6保持；参考辅助梯度贡献仍要求大于1e-6。
- 命令沿用三轴脚本torchrun前缀，参数末尾full 2/selective 2；
  custom端口29948/29949、native29950/29951。日志
  `/tmp/llmtuner-triaxis-accum-{full,selective}-07.log` 和
  `/tmp/llmtuner-triaxis-accum-native-{full,selective}-07.log`。
- 首轮仅pre-commit格式调整；最终全适用项通过，日志
  `/tmp/llmtuner-triaxis-accum-final-gates-07.log`。diff检查通过。
  未确认生产缺陷，无生产实现修改，不修改index、未commit/push。
- 本场景每批有效token数相同（路由184/CE168），不证明不同有效长度的
  加权累积、尾窗口耗尽、Trainer编排、compile、目标设备或完整恢复。
  CPU SDPA替代attention边界保持，未缩减原目标，相关验收继续开放。

### TP-EP-CP-UNEQUAL-TOKEN-ACCUMULATION-07

- 三轴训练脚本新增uneven参数：先构造整个step的batch列表，明确每批
  (routed,CE)分别(184,168)/(152,128)，总分母336/296。生产CE summed
  loss除step总标签数，aux step分母设为总路由数；独立参考分别按各批
  CE数/总CE、路由数/总路由加权，不再依赖有效数相同才成立的1/累积次数。
- 四CPU/Gloo rank TP2/KVCP2/EP2、FP64、aux0.01，custom/full AC与
  native/selective AC两种配置均三步exit0，27参数累计梯度/权重/momentum、
  每层累计aux指标、全局CE均对照独立HF通过，atol/rtol1e-6保持；
  每层body每step执行4次。参数末尾full 2 uneven / selective 2 uneven，
  端口29952/29953，日志 `/tmp/llmtuner-triaxis-uneven-{custom,native}-07.log`。
- 原等长custom无AC累积2回归exit0，每层calls=[2,2]，端口29954，日志
  `/tmp/llmtuner-triaxis-uneven-regression-07.log`；格式自动调整审阅后
  pre-commit全部适用项通过，日志
  `/tmp/llmtuner-triaxis-uneven-final-gates-07.log`，diff检查通过。
- 无生产实现修改，未确认新生产缺陷；保留外部暂存，不修改index或commit/push。
  此证据覆盖不等有效数的独立加权累计，不证明Trainer计数/尾窗口耗尽、
  真实packer、目标设备、compile或完整恢复。原全局要求继续开放。

### TRAINING-TOKEN-COORDINATION-01 — P1，本地计数失败传播

- 路径：`trainer/runtime/batch.py::microbatch`，调用方为
  `Trainer.train_step`。`fetch_batch` 已协调取数与标签校验，但后续有效
  token/路由 token 计数仍可能局部抛错，使其他 rank 继续进入分母 collective。
- 实现：将两个纯本地计数放入 `synchronize_local_errors`，成功后才推进
  `ntokens_seen`。协调区间内无 collective，不扩大到通信故障恢复。
- 回归：`tests/integration_tests/data_fetch_coordination.py` 分别在 rank0、
  rank1 删除 input；原 rank 保留 KeyError，其他 rank 收到协调错误，
  两侧计数均保持17且后续 barrier 成功。未运行修复前失败版本，根因依据
  当前调用链与异常路径；修复后真实两 rank 回归通过。
- 命令：`PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0 torchrun --nproc_per_node=2
  --master_addr=127.0.0.1 --master_port=29956
  tests/integration_tests/data_fetch_coordination.py`，macOS CPU/Gloo、
  PyTorch2.14；日志 `/tmp/llmtuner-training-token-fix-08.log`，退出0。
- 质量门禁：完整CPU套件3671 passed、2 skipped、288 warnings（52.36s），
  日志 `/tmp/llmtuner-training-token-full-08.log`；compileall、diff检查通过；
  适用文件pre-commit最终全部通过，日志
  `/tmp/llmtuner-training-token-final-gates-08.log`。
- 状态：此缺陷实现与上述回归验收完成；不证明所有训练前置操作的失败传播、
  目标设备、完整并行矩阵或通信故障恢复。相关全局验收保持开放。
  当前实现与回归已由外部暂存，本轮不操作index、未commit/push。

### TRAINER-UNEQUAL-TOKEN-SUMMED-REFERENCE-09

- 审查真实 `Trainer.train_step`：先读取完整累积窗口，再以step有效标签总数
  归一化各批loss；尾窗口耗尽已有真实Trainer/DCP测试，消耗计数包含取消窗口。
  未据此修改生产实现。
- 新增CPU回归 `test_trainer_unequal_token_accumulation_matches_summed_loss_reference`：
  两批有效标签数14/4，真实Trainer累积两批；独立参考直接使用相同模型前向、
  PyTorch summed CE，总loss除18后一次backward、clip和AdamW更新。
  对照全部模型状态、梯度、优化器状态、scheduler、loss及消耗计数32。
  状态/梯度容差rtol1e-5、atol1e-7，loss相对容差1e-6。
- 初始“合并整批”fixture失败：默认随机数据causal路径合并后改变attention
  上下文，不构成独立文档等价。尝试positions隔离时先遇本机Inductor PCH
  缓存错误，禁PCH重跑后生产SDPA明确拒绝packed；保留日志
  `/tmp/llmtuner-trainer-unequal-{09,final-09,nopch-09}.log`（首份实际文件为
  `/tmp/llmtuner-trainer-unequal-09.log`）。改用summed-graph参考，未放宽容差、
  未替换生产attention或绕过后端限制。
- 最终独立参考1 passed（9.01s），日志
  `/tmp/llmtuner-trainer-unequal-reference-09.log`；整个Trainer模块141 passed、
  33 warnings（14.25s），日志 `/tmp/llmtuner-trainer-all-09.log`。
  适用文件pre-commit全部通过，日志 `/tmp/llmtuner-trainer-unequal-gates-09.log`；
  diff检查通过。本轮仅测试/台账修改，未重复全CPU套件，不操作index或commit/push。
- 边界：参考复用生产HF前向与预处理，独立验证CE求和与Trainer累积编排，
  不证明attention自身正确、多轴Trainer、MoE aux、目标设备或完整恢复。
  上述全局验收缺口继续开放。

### TRAINER-TP-EP-CP-UNEQUAL-REFERENCE-10

- 新增 `tests/integration_tests/trainer_tp_ep_cp_equivalence.py`，补齐此前三轴
  手工训练loop未调用真实Trainer step的缺口。生产parallelize装配后，构造
  Trainer运行时，直接调用未替换的train_step、microbatch、preprocess、
  forward/backward、TP复制梯度归约、clip、有限检查与metrics。
- 四CPU/Gloo rank，TP2/KVCP2/EP2、FP64、native/custom FSDP、full AC、
  aux=0，累积2；每批有效标签168/128、step总分母296，包含padding与
  masked prompt。独立完整HF前向以两批summed CE除296，一次backward，
  SGD lr0.02/momentum0.9。三步全部27参数梯度、更新与loss通过1e-6
  绝对/相对容差；每rank消耗计数step*128，全局metrics为step*512。
- 命令前缀 `PYTHONPATH=. GLOO_SOCKET_IFNAME=lo0
  TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 torchrun --nproc_per_node=4
  --master_addr=127.0.0.1`，端口29958/custom、29959/native，脚本末尾参数
  为对应实现。两次均退出0，日志
  `/tmp/llmtuner-trainer-triaxis-custom-final-10.log`、
  `/tmp/llmtuner-trainer-triaxis-native-10.log`。
- 初次fixture错误地给example_model只读property赋值；修正为model字段后
  通过，原日志 `/tmp/llmtuner-trainer-triaxis-custom-10.log` 保留。
  首轮pre-commit两行超长，修正后全适用项通过，日志
  `/tmp/llmtuner-trainer-triaxis-final-gates-10.log`，diff检查通过。
- 边界：采用显式CPU dense-mask SDPA替代，保留生产CP通信/autograd、
  TP/EP/FSDP；运行时手动装配，scheduler/checkpointer为无操作fixture，
  输入直接传入迭代器，未覆盖完整Trainer构造/loader/train循环、aux、
  compile、恢复、目标设备内核或性能。本轮无生产修改，不操作index或提交。
  此证据不替代以上开放验收。

### TRAINER-TRIAXIS-MOMENTUM-REPLAY-11

- 强化 `trainer_tp_ep_cp_equivalence.py`：在checkpoint wrapper内部decoder
  body安装forward计数，每step两microbatch必须每层至少执行4次；同时将
  全部27参数的SGD momentum buffer按独立HF物理TP/EP切片映射逐一对照。
  原梯度、更新、loss与消耗token断言保留，atol/rtol1e-6未调整。
- 四CPU/Gloo rank TP2/KVCP2/EP2、native/custom FSDP、full AC、aux=0、
  不等有效数累积2，两种实现均三步exit0；每rank每step两层body calls=[4,4]，
  梯度/权重/动量全部通过。这排除了只安装AC wrapper未重计算，以及更新
  暂时一致但隐藏动量状态不同的弱证据。
- 命令沿用上一项torchrun前缀，端口29960/custom、29961/native，日志
  `/tmp/llmtuner-trainer-mechanisms-{custom,native}-11.log`。
  适用文件pre-commit全部通过，日志
  `/tmp/llmtuner-trainer-mechanisms-gates-11.log`；diff检查通过。
- 无生产实现修改，不操作index、未commit/push。仍为手动装配Trainer，
  scheduler/checkpointer无操作、显式CPU attention替代；不证明真实loader、
  aux、compile、完整恢复及目标设备。以上缺口继续开放。

### TRAINER-TRIAXIS-AUX-DENOMINATOR-12

- 三轴Trainer脚本增加aux参数，生产系数0.01，并注册生产optimizer aux
  metrics pre-hook；step分母由真实train_step设置，不在fixture提前设置。
  两批CE有效数168/128，总296；路由数184/152，总336。
- 参考使用原生HF gate hook提取logits/selected，独立FP32 softmax、bincount、
  专家频率与均值概率公式；各批按路由数/336加权。每层aux梯度贡献要求
  max abs>1e-6，排除CE-only偶然通过。两层均值对照生产mean指标，反向
  仍对照CE+0.01aux的全部27参数梯度、更新和SGD动量，atol/rtol1e-6。
- 四CPU/Gloo rank TP2/KVCP2/EP2、full AC、累积2，native/custom各三步
  exit0；每层body每step执行4次，aux指标与独立参考一致，排除重计算重复
  计数。命令沿用前项前缀，端口29962/custom、29963/native，末尾aux。
  日志 `/tmp/llmtuner-trainer-aux-{custom,native}-12.log`。
- 原aux关闭custom对照也三步exit0，端口29964，日志
  `/tmp/llmtuner-trainer-aux-control-12.log`。
- 首轮及最终适用pre-commit通过，最终日志
  `/tmp/llmtuner-trainer-aux-final-gates-12.log`。无生产修改，不操作index。
  仍为手动Trainer运行时、无操作scheduler/checkpointer、CPU attention替代，
  不证明完整装配、loader、compile、恢复或目标设备；全局验收继续开放。

### TRAINER-TRIAXIS-SCHEDULER-ORDER-13

- 静态核对 `trainer/runtime/assembly.py::_build_optimizer_stack`：生产构造
  OptimizersContainer、每inner optimizer的scheduler、EMA及容器级aux/MoE
  pre-hook；真实train_step的optimizer→scheduler→EMA顺序保持。
- 三轴Trainer脚本移除无操作scheduler，使用生产build_lr_scheduler：
  warmup2/total4/linear decay_ratio0.5、base lr0.02、训练3步。
  独立HF参考手工设置本步lr=[0.01,0.02,0.02]，不调用生产schedule；
  每步断言日志lr等于本步更新值、optimizer下步lr=[0.02,0.02,0.01]、
  last_epoch等于step，原梯度/权重/动量/aux/AC断言全部保留。
- 四CPU/Gloo rank TP2/KVCP2/EP2、full AC、aux0.01、不等token累积2，
  native/custom各三步exit0；全部27参数仍以atol/rtol1e-6通过。
  命令沿用前项前缀，端口29965/custom、29966/native，末尾aux；日志
  `/tmp/llmtuner-trainer-schedule-{custom,native}-13.log`。
  适用pre-commit通过 `/tmp/llmtuner-trainer-schedule-gates-13.log`，diff检查通过。
- 本轮无生产修改，不重复全CPU套件、不操作index或commit/push。
  scheduler已使用生产实现；optimizer仍为直接SGD，非生产容器，checkpointer
  无操作、EMA关闭、Trainer手动装配与CPU attention替代边界保持。
  本项不证明完整初始化、loader、EMA、compile、完整恢复或目标设备。

### TRAINER-TRIAXIS-EMA-POSTUPDATE-14

- 三轴Trainer脚本启用生产EMA(model_parts=[model], decay=0.75)，真实
  train_step推进EMA。独立完整HF参考在SGD更新后按0.75旧均值+0.25新参数
  计算，每个step将27个EMA状态映射到物理TP/EP切片，与FSDP full_tensor
  重建值逐一对照，atol/rtol1e-6。参考不调用生产EMA实现。
- 四CPU/Gloo rank TP2/KVCP2/EP2、full AC、aux0.01、有效标签不等累积2、
  生产WSD scheduler，native/custom各三步exit0，原梯度/更新/动量、aux指标、
  lr顺序、AC body=[4,4]断言继续通过。
- 命令沿用前项前缀，端口29967/custom、29968/native，末尾aux；日志
  `/tmp/llmtuner-trainer-ema-{custom,native}-14.log`。适用pre-commit通过，
  日志 `/tmp/llmtuner-trainer-ema-gates-14.log`，diff检查通过。
- 本轮无生产修改、不操作index或commit/push。EMA覆盖固定decay、每step
  更新和参数状态，不证明动态decay、延迟/间隔策略、buffer EMA或恢复。
  optimizer仍直接SGD、checkpointer无操作，完整构造、真实loader、compile
  及目标设备缺口继续开放；CPU attention替代边界保持。

### TRAINER-TRIAXIS-ADAMW-NUMERICS-15 — P1，开放

- 三轴Trainer脚本新增末尾adam选项，通过生产OptimizersContainer构造
  AdamW(for-loop, lr0.02, weight_decay0.1, betas0.9/0.999, eps1e-8)，
  world协调初始化，scheduler直接调度inner optimizer，aux hook注册在容器。
  参考使用独立torch.optim.AdamW，相同超参数。新增exp_avg/exp_avg_sq与step
  断言，保留SGD路径和所有原验收；未放宽原1e-6容差。
- native/custom真实四CPU/Gloo rank运行均exit1。custom诊断确认首步
  gradient=False的专家w1_EFD权重失败，而该参数梯度比较先通过1e-6：
  rank0/2最大绝对差3.0094034995e-6，rank1/3为6.2524247537e-6，
  每片12288元素中2至3个不满足。尚未定位到optimizer契约错误；
  可能涉及微小梯度数值差被Adam归一化放大，此为待验证假设。
- 命令沿用前项torchrun前缀，端口29969/custom、29970/native，末尾aux adam；
  日志 `/tmp/llmtuner-trainer-adam-{custom,native}-15.log`。custom明确
  step/gradient标注诊断端口29971，日志
  `/tmp/llmtuner-trainer-adam-diagnostic-15.log`，再次exit1。
- 原custom aux SGD控制端口29972，三步exit0，日志
  `/tmp/llmtuner-trainer-adam-sgd-control-15.log`。格式及lambda变量绑定
  修正后适用pre-commit全部通过，日志
  `/tmp/llmtuner-trainer-adam-final3-gates-15.log`；diff检查通过。
- 下一步：逐失败索引提取更新前参数、实际/参考梯度、m/v及Adam更新公式，
  分离梯度舍入、专家映射与optimizer实现；先以同一梯度独立验证更新公式，
  再判断根因及验收条件。不得将SGD成功或状态测试数量作为AdamW通过证据。
  当前三轴AdamW开放，无生产修复、不操作index、未commit/push。

### TRAINER-TRIAXIS-ADAMW-FORMULA-16 — 关联15，仍开放

- 三轴脚本保留首步更新前参数，失败前打印dtype、shape、最大差异索引、
  实际/参考梯度，以及同梯度独立AdamW首步公式残差。没有调整容差或跳过
  失败断言。native/custom两次仍exit1，端口29973/custom、29974/native，
  日志 `/tmp/llmtuner-trainer-adam-formula-16.log` 与
  `/tmp/llmtuner-trainer-adam-formula-native-16.log`。
- 两实现得到相同诊断：FP64专家片(4,48,64)，rank0/2 flat7008梯度
  -7.32688931748271e-9 vs -7.335928895135747e-9；rank1/3 flat135梯度
  -4.2637111619114876e-9 vs -4.251001722290967e-9。
- 使用更新前参数与实际梯度计算独立首步公式
  w*(1-lr*wd)-lr*g/(abs(g)+eps)，lr0.01/wd0.1/eps1e-8，
  全参数片最大公式残差1.3877787807814457e-17。
  将上述实际/参考梯度代入公式得到权重差3.0094034995248322e-6和
  6.252424753741548e-6，与实测3.0094034995254565e-6和
  6.252424753740854e-6吻合。
- 结论范围：首个失败权重可由微小梯度差经Adam归一化完整解释，现有证据
  不支持首步optimizer更新公式错误；尚未定位梯度差来自FP32路由、collective
  累加顺序或其他路径，也未验收后续Adam状态与恢复，15保持开放。
- 下一步比较路由概率及逐层前反向，分离aux、AC与布局的数值来源；不得
  调大eps或容差来替代原场景验收。格式自动修改审阅后适用pre-commit通过，
  日志 `/tmp/llmtuner-adam-formula-final-gates-16.log`，diff检查通过。
  无生产代码修改，不操作index、未commit/push。

### TRAINER-TRIAXIS-ADAMW-FACTOR-CONTROLS-17 — 关联15，开放

- 脚本增加显式full/none AC参数，默认full保持；none路径decoder每step
  必须执行2次，full至少4次。aux位置可传noaux，仍使用生产AdamW容器。
  原基线aux/full不变，原1e-6容差及失败断言保持。
- custom四CPU/Gloo rank三组诊断均exit1：noaux/full端口29975、aux/none
  29976、noaux/none29977。命令沿用前项前缀，脚本参数分别
  custom noaux adam full / custom aux adam none / custom noaux adam none。
  日志 `/tmp/llmtuner-adam-control-{noaux-full,aux-none,noaux-none}-17.log`。
- aux/none首步专家w1_EFD梯度、权重误差和同梯度公式残差与aux/full完全
  相同：3.0094034995e-6/6.2524247537e-6，公式残差1.39e-17。
  noaux/full与noaux/none均首步k_proj.weight失败，rank1/3物理片
  (32,64)的(15,33)单元素，最大绝对误差2.4428592840918667e-6；
  该参数梯度在前一断言通过。两者失败值相同。
- 结论边界：AC不是当前首步差异的必要条件；关闭aux仍有注意力权重失败，
  aux不是整体失败的唯一来源。不据此推断某个通信或路由实现有bug，
  后续检查基础前反向数值与Adam小梯度敏感性。保留原配置失败验收。
- 适用pre-commit通过 `/tmp/llmtuner-adam-controls-gates-17.log`，diff检查
  通过。无生产修改、不操作index、未commit/push。所有控制保留三轴、
  不等token累积、scheduler与EMA，未以关闭机制的测试代替原场景验收。

### TRAINER-TRIAXIS-ROUTER-DTYPE-CONTROL-18 — 关联15，开放

- 静态对照生产 `models/common/moe/gate.py::RouterGateLinearFunction` 与
  已安装transformers Qwen3MoeTopKRouter.forward：生产投影/反向固定FP32，
  FP64 HF参考使用FP64 linear后FP32 softmax。这是明确精度契约差异，
  不能仅依据模型参数dtype假定整个参考路径同精度。
- 脚本新增显式fp32_router诊断选项，HF gate用独立F.linear对输入和weight
  转FP32，再softmax/topk/归一化；不使用生产gate，不改变默认HF参考。
  此选项改变oracle契约，仅供归因，不替代原HF场景验收。
- custom noaux adam none fp32_router、四CPU/Gloo rank端口29978，exit1，
  日志 `/tmp/llmtuner-router-contract-18.log`。首步仍k_proj权重失败，
  rank1物理片(32,64)、flat993，梯度9.476207196712494e-8 vs
  9.442896889725813e-8，权重误差3.044763508742998e-6，实际梯度独立AdamW
  公式残差6.94e-18。原noaux参考误差2.44e-6，改gate精度未消除失败。
- 结论：gate精度是需记录的适配差异，但不足以解释或消除全部数值失败。
  不修改生产精度、eps或容差；后续仍需逐层激活/梯度及计算顺序对照。
  扩展同梯度公式诊断到k_proj；已排除首个注意力权重失败的首步更新公式
  错误，尚未证明整体并行梯度无缺陷。15保持开放。
- 适用pre-commit最终通过，日志
  `/tmp/llmtuner-router-contract-final-gates-18.log`，diff检查通过。
  无生产修改、不操作index或commit/push。

训练指标输入边界复核：`flops_per_token` 与其 attention 辅助函数现在拒绝
负数、布尔值和非整数序列长度；窗口大小必须为正整数。`MetricsProcessor`
同样拒绝负数、布尔值和非整数 `num_flops_per_token`，避免独立调用产生负
TFLOPs 或 MFU。训练配置原有的 `max_seq_len >= 1` 校验保持不变。新增边界
回归后完整 CPU 测试 2267 passed、2 skipped；Ruff、format、compileall 和
diff 检查通过。
