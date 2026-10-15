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
- 复用 `contracts.require_positive`，在 splitter、document packing dataset
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
- 配置层复用 require_at_least 拒绝两字段负值，None、0 和正值仍有效，
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
