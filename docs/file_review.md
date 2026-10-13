# 逐文件代码审查

范围：`llmtuner/**/*.py`。状态仅代表实现阅读，不代表全部硬件组合已验证。静态检查不会自动将文件标为已审查。

已阅读文件仍可保留待核对项；其余文件明确列为待审查。

| 文件 | 状态 | 检查内容与待办 |
| --- | --- | --- |
| `llmtuner/__init__.py` | 已阅读 | 公开配置与 Trainer 懒加载导出；检查导出路径及导入副作用。 |
| `llmtuner/__main__.py` | 已阅读 | CLI 入口转发；检查 main 调用。 |
| `llmtuner/accelerator/__init__.py` | 已阅读 | 包导出与懒加载。 |
| `llmtuner/accelerator/capabilities.py` | 已阅读 | 能力缓存、签名探测、错误提示；修复缺失 eval 时崩溃。 |
| `llmtuner/accelerator/collectives.py` | 已阅读 | TP/EP/PP 梯度范数归约、DTensor、空梯度、通信设备转换；多设备等价性待验证。 |
| `llmtuner/accelerator/device.py` | 已阅读 | 设备发现、环境 rank、后端映射及设备设置。 |
| `llmtuner/accelerator/distributed.py` | 已阅读 | 三类 launcher、PG 查询和通信设备；修复 MPI LOCAL_RANK 缺失。 |
| `llmtuner/accelerator/spmd.py` | 已阅读 | 线程局部 mesh 栈、dense/sparse 注册、嵌套上下文恢复。 |
| `llmtuner/accelerator/tensor_tree.py` | 已阅读 | 统一张量树遍历、容器重建和 out 结构预校验；覆盖 namedtuple/defaultdict、键顺序、混合设备与失败前无写入。 |
| `llmtuner/components/__init__.py` | 已阅读 | 包入口及公开导出；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/checkpointer/__init__.py` | 已阅读 | __getattr__、__dir__；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/checkpointer/base.py` | 已阅读 | purge_thread、BaseCheckpointManager；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/checkpointer/checkpoint_keys.py` | 已阅读 | canonical_fqn；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/checkpointer/distributed_checkpoint.py` | 已阅读 | DistributedCheckpointAsyncMode、FilesystemCheckpointStorage、DistributedCheckpointManager；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/checkpointer/filesystem.py` | 已阅读 | is_remote、resolve、exists、isdir、isfile；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/checkpointer/model_state.py` | 已阅读 | ModelCheckpointState 聚合 PP/EP 状态并维护缓存；支持非张量 extra_state，按 shape/dtype/device 刷新张量缓存。 |
| `llmtuner/components/checkpointer/storage.py` | 已阅读 | CheckpointStorage 协议和 FilesystemCheckpointStorage；不依赖管理器或 DCP，保留本地及 fsspec 路径语义。 |
| `llmtuner/components/checkpointer/torch_checkpointing_backend.py` | 已阅读 | require_torch_checkpointing、BackendSymbols、BackendCheckpointStorage、init_subprocess_logging、item_specs；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/loss.py` | 已阅读 | vocab_shard_bounds、next_token_targets、cross_entropy_loss、shard_local_labels、LossParallelCrossEntropy；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/metrics.py` | 已阅读 | DeviceMemoryMonitor、build_device_memory_monitor、BaseLogger、TensorBoardLogger、WandBLogger；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/optimizer/__init__.py` | 已阅读 | 包入口及公开导出；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/optimizer/ema.py` | 已阅读 | EMAParamOptimizer、EMA；检查训练服务状态、错误路径、持久化及资源生命周期。 |
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
| `llmtuner/config/cli_config.py` | 已阅读 | cli_view、cli_groups；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/config/compile_config.py` | 已阅读 | CompileConfig；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/config/context_parallel_config.py` | 已阅读 | validate_context_parallel_config；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/config/data_config.py` | 已阅读 | DataloaderConfig；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/config/expert_parallel_config.py` | 已阅读 | validate_expert_parallel_config；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/config/field_checks_config.py` | 已阅读 | require_at_least；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/config/fsdp_config.py` | 已阅读 | validate_fsdp_config；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/config/llmtuner_config.py` | 已阅读 | LLMTunerConfig；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/config/model_config.py` | 已阅读 | ModelConfig；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/config/observability_config.py` | 已阅读 | MetricsConfig、ProfilerConfig；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/config/optimizer_config.py` | 已阅读 | LRSchedulerConfig、ParamGroupConfig、OptimizerConfig、EMAConfig；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/config/parallel_config.py` | 已阅读 | ParallelConfig；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/config/pipeline_parallel_config.py` | 已阅读 | validate_pipeline_parallel_config；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/config/tensor_parallel_config.py` | 已阅读 | validate_tensor_parallel_config；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/config/training_config.py` | 已阅读 | TrainingConfig；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/config/validation_config.py` | 已阅读 | ValidationConfig；检查默认值、字段约束、跨配置依赖和错误路径。 |
| `llmtuner/datasets/__init__.py` | 已阅读 | 包入口及公开导出；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/collators.py` | 已阅读 | Collator、TextCollator；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/contracts.py` | 已阅读 | require_positive、DatasetBuildContext、DatasetIterationPolicy、Batch；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/dataloader.py` | 已阅读 | DataloaderExhaustedError、require_same_dp_degree、BaseDataLoader、GrainDataLoader；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/loader_builder.py` | 已阅读 | _resolve_multimodal_recipe、_resolve_text_recipe、build_dataloader；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/composition.py` | 已阅读 | SingleDataset、DatasetMix、DatasetConcat、is_not_none、as_iter_dataset；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/multimodal/__init__.py` | 已阅读 | 包入口及公开导出；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/multimodal/collator.py` | 已阅读 | text_positions、vision_grid_positions、MultimodalCollator；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/multimodal/image.py` | 已阅读 | decode_image、smart_resize、resize_to_pixel_budget、resize_to_navit_patch_grid、process_image；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/multimodal/packing.py` | 已阅读 | build_multimodal_packing、multimodal_sample_to_packing_input、packing_output_to_multimodal_sample；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/multimodal/placeholders.py` | 已阅读 | insert_vision_placeholders；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/multimodal/processors.py` | 已阅读 | process_multimodal_sample、process_obelics_sample、process_cc12_wd_sample、MultimodalProcessor；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/multimodal/video.py` | 已阅读 | load_video、process_video；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/packing/__init__.py` | 已阅读 | 包入口及公开导出；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/packing/build.py` | 已阅读 | row_lengths、build_concat_then_split_packing、build_first_fit_packing；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/packing/conversions.py` | 已阅读 | packing_output_is_full、text_sequence_to_packing_input、packing_output_to_text_sequence；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/packing/iterators.py` | 已阅读 | DocumentAwareConcatThenSplitIterDataset、DocumentAwareConcatThenSplitIterator、SplitTextSequenceDocuments、next_document_chunk_end；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/sources.py` | 已阅读 | RandomAccessDataSource、IndexedJsonlSource、IndexedJsonlDataSource、reject_duplicated_hf_fields、HuggingFaceRandomAccessSource；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/synthetic.py` | 已阅读 | batch_iterator、RandomTokenSource、RandomTokenDataLoader；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/text/__init__.py` | 已阅读 | 包入口及公开导出；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/text/processors.py` | 已阅读 | read_text、TextProcessor、require_token_prefix、ChatProcessor、local_jsonl_recipe；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/datasets/text/renderer.py` | 已阅读 | RendererTokenizerWrapper、build_chat_renderer；检查样本/标签对齐、分片、打包、迭代终止及恢复路径。 |
| `llmtuner/errors.py` | 已阅读 | 异常层次及 ValueError/NotImplementedError 兼容。 |
| `llmtuner/memory/__init__.py` | 已阅读 | AC 公开入口懒加载。 |
| `llmtuner/memory/activation_checkpoint.py` | 已阅读 | 全量/选择性/区域/预算 AC、重算辅助损失标记；可选 remat 运行待验证。 |
| `llmtuner/memory/remat_regions.py` | 已阅读 | 线性区域选择、保存/重算优先级与未匹配模式。 |
| `llmtuner/models/__init__.py` | 已阅读 | 包入口及公开导出；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/__init__.py` | 已阅读 | __getattr__、__dir__；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/activation.py` | 已阅读 | ActivationFn、SwiGLU；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/tp_projections.py` | 已阅读 | warn_once_no_tp_overlap、validate_tp_projection_preconditions、AllGatherFusedQKVLinear、RowParallelLinear、TensorParallelFeedForward；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/attention/__init__.py` | 已阅读 | 包入口及公开导出；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/attention/masks.py` | 已阅读 | VarlenMetadata、round_up、get_causal_mask_mod、get_document_mask_mod、get_efficient_causal_mask_mod_for_packed_document；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/attention/qkv.py` | 已阅读 | local_head_split、QKVLinear；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/aux_loss.py` | 已阅读 | AuxLossInjection、AuxLoss、zero_aux_losses、collect_aux_loss_metrics、register_aux_loss_zero_hook；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/cast_linear.py` | 已阅读 | CastLinear、to_cast_linear；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/embedding.py` | 已阅读 | Embedding；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/feed_forward.py` | 已阅读 | compute_ffn_hidden_dim、FeedForward、SigmoidGatedFeedForward；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/linear.py` | 已阅读 | RouterGateLinearFunction、RouterGateLinear、PartialBiasRowwiseLinear；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/moe/__init__.py` | 已阅读 | 包入口及公开导出；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/moe/auxiliary_loss.py` | 已阅读 | PartialToInvariantAllReduce、MicrobatchWiseLoadBalanceLoss；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/moe/bias_update.py` | 已阅读 | update_expert_bias、register_moe_load_balancing_hook、update_quantile_expert_bias、register_moe_quantile_balancing_hook；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/moe/experts.py` | 已阅读 | grouped_mm_available、GroupedExperts、RoutedExperts、DominoRoutedExperts；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/moe/layer.py` | 已阅读 | MoE、iter_moe_layers；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/moe/routing.py` | 已阅读 | TokenChoiceTopKRouter、QuantileBalancedTopKRouter、QuantileBalancer；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/parallel/expert_parallel/token_dispatch.py` | 已阅读 | materialize、LocalDispatchMetadata、AllToAllDispatchMetadata、LocalTokenDispatcher、BaseEPTokenDispatcher；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/multimodal.py` | 已阅读 | get_vision_positions、build_vision_bank_indices、gather_vision_embeds、scatter_vision_embeds；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/rope.py` | 已阅读 | maybe_check_max_pos、yarn_inv_freq、RoPEConfig、RoPE、ComplexRoPE；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/common/scatter_add.py` | 已阅读 | deterministic_scatter_add、deterministic_scatter_add_fake、_backward、_setup_context；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/hf/__init__.py` | 已阅读 | 包入口及公开导出；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/hf/attention_backend.py` | 已阅读 | available_attention_backend、flex_attention_adapter、uses_deepseek_sparse_attention、register_attention_backend；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/hf/factory.py` | 已阅读 | build_model_config、build_model_config_for、num_flops_per_token、unwrap_text_config、resolve_model_class；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/hf/flops.py` | 已阅读 | quadratic_attention_flops_per_token、MoEGeometry、flops_per_token、int_attr、first_int_attr；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/hf/model.py` | 已阅读 | HFTransformerModel；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/hf/model_setup.py` | 已阅读 | first_present、validate_head_config、resolve_experts_implementation、apply_cast_lm_head；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/models/hf/state_dict_adapter.py` | 已阅读 | HFTransformerStateDictAdapter；检查张量形状、前向/反向、状态转换和模型兼容性边界。 |
| `llmtuner/parallel/__init__.py` | 已阅读 | __getattr__、__dir__；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/attention_heads.py` | 已阅读 | head_counts、require_heads_divisible_by；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/compatibility.py` | 已阅读 | Row、validation_once_requires_dp1、validation_once_requires_finite_corpus、chunked_loss_pp、chunked_loss_validation；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/context_parallel/__init__.py` | 已阅读 | 包入口及公开导出；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/context_parallel/attention.py` | 已阅读 | cp_all_to_all、reject_unrepresentable_attention_kwargs、run_flex、SeqToHead、HeadToSeq；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/context_parallel/context_parallel.py` | 已阅读 | require_ulysses_compatible、apply_cp；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/context_parallel/input_sharding.py` | 已阅读 | require_torch_cp、resolve_load_balancer、shard_batch_for_cp、shard_padding_mask_for_cp、shard_padding_mask_for_tp；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/context_parallel/kcp.py` | 已阅读 | _kcp_runtime、kcp_kda_forward、apply_kcp；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/expert_parallel/__init__.py` | 已阅读 | 包入口及公开导出；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/expert_parallel/block_conversion.py` | 已阅读 | restore_fp32_state_buffers、build_aux_loss、convert_block；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/expert_parallel/checkpointing.py` | 已阅读 | mark_experts_ep_sharded、expert_shard_map、named_params、matched_entries、gather_expert_state；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/expert_parallel/expert_parallel.py` | 已阅读 | apply_ep；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/expert_parallel/hf_inspection.py` | 已阅读 | router_of、FusedExperts、fused_experts_of、has_router_weight、is_hf_moe_block；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/expert_parallel/model_replacement.py` | 已阅读 | swap_hf_moe_blocks；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/fully_shard_parallel/__init__.py` | 已阅读 | 包入口及公开导出；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/fully_shard_parallel/fine_grained.py` | 已阅读 | fully_shard_fine_grained、NativeFSDPStrategy、CustomFSDPStrategy、get_fsdp_strategy；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/fully_shard_parallel/fully_shard_parallel.py` | 已阅读 | iter_transformer_layers、resolve_fsdp_mesh、resolve_sparse_fsdp_mesh、iter_fsdp_modules、disable_fsdp_gradient_division；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/fully_shard_parallel/plan.py` | 已阅读 | set_fsdp_prefetch_windows、_resolve_pattern_units、apply_fsdp_by_pattern；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/model_compile.py` | 已阅读 | apply_compile、maybe_enable_async_tp、maybe_regional_inductor_backend、maybe_regional_inductor；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/module_patterns.py` | 已阅读 | compile_module_pattern、module_name_matches、resolve_module_patterns；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/parallelize.py` | 已阅读 | parallelize_hf_transformers；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/pipeline_parallel/__init__.py` | 已阅读 | 包入口及公开导出；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/pipeline_parallel/partition.py` | 已阅读 | generate_llm_fqn_per_model_part、split_model_into_stages；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/pipeline_parallel/pipeline_parallel.py` | 已阅读 | PipelineParallelSetup、scalar_loss_fn、make_schedule_loss_fn、get_pipeline_metadata、prepend_first_stage_modules；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/stages.py` | 已阅读 | Stage、stage_enabled；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/tensor_parallel/__init__.py` | 已阅读 | 包入口及公开导出；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/tensor_parallel/linear_ops.py` | 已阅读 | ensure_symm_mem_ops、functional_collectives、all_gather_linear、linear_reduce_scatter、all_gather_along；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/tensor_parallel/tensor_parallel.py` | 已阅读 | shard_weight、ColumnParallelLinear、RowParallelLinear、ColwiseLinearNoGather、GatherSequenceFirst；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/parallel/topology.py` | 已阅读 | ParallelTopology、build_parallel_topology、build_mesh；检查 mesh/degree、参数转换、通信和不支持组合的拒绝路径；多设备数值待验证。 |
| `llmtuner/trainer/__init__.py` | 已阅读 | 包入口及公开导出；检查训练编排、分母/梯度归约、checkpoint、验证和清理流程。 |
| `llmtuner/trainer/cli.py` | 已阅读 | parse_config、main；检查训练编排、分母/梯度归约、checkpoint、验证和清理流程。 |
| `llmtuner/trainer/runtime/__init__.py` | 已阅读 | 包入口及公开导出；检查训练编排、分母/梯度归约、checkpoint、验证和清理流程。 |
| `llmtuner/trainer/runtime/assembly.py` | 已阅读 | seed_everything、_build_model、_apply_parallelism、_build_optimizer_stack、_build_checkpointer；检查训练编排、分母/梯度归约、checkpoint、验证和清理流程。 |
| `llmtuner/trainer/runtime/batch.py` | 已阅读 | dp_rank_world_size、batch_size_per_rank、build_dataloader、data_iterator、batch_generator；检查训练编排、分母/梯度归约、checkpoint、验证和清理流程。 |
| `llmtuner/trainer/runtime/pipeline.py` | 已阅读 | split_pipeline_microbatches、run_pipeline_step；检查训练编排、分母/梯度归约、checkpoint、验证和清理流程。 |
| `llmtuner/trainer/runtime/reductions.py` | 已阅读 | TrainingReductionMeshes、loss_reporting_meshes、training_reduction_meshes；检查训练编排、分母/梯度归约、checkpoint、验证和清理流程。 |
| `llmtuner/trainer/runtime/validation.py` | 已阅读 | build_validation_dataloader、count_validation_batch_tokens、finalize_validation、check_validation_feasibility、should_validate；检查训练编排、分母/梯度归约、checkpoint、验证和清理流程。 |
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
