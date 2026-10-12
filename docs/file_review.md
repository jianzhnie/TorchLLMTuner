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
| `llmtuner/components/checkpointer/base.py` | 已阅读 | purge_thread、shares_storage、ModelWrapper、CheckpointStorage、BaseCheckpointManager；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/checkpointer/checkpoint_keys.py` | 已阅读 | canonical_fqn；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/checkpointer/distributed_checkpoint.py` | 已阅读 | DistributedCheckpointAsyncMode、FilesystemCheckpointStorage、DistributedCheckpointManager；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/checkpointer/filesystem.py` | 已阅读 | is_remote、resolve、exists、isdir、isfile；检查训练服务状态、错误路径、持久化及资源生命周期。 |
| `llmtuner/components/checkpointer/torch_checkpointing_backend.py` | 已阅读 | require_torch_checkpointing、BackendConfig、BackendCheckpointStorage、init_subprocess_logging、item_specs；检查训练服务状态、错误路径、持久化及资源生命周期。 |
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
| `llmtuner/utils/logging.py` | 已阅读 | rank 过滤、格式及初始化；后续检查祖先 logger 重复输出。 |
| `llmtuner/utils/monitoring.py` | 已阅读 | 设备容量、FLOPS、颜色、内存快照；修复设备 0 硬编码。 |
| `llmtuner/utils/seeding.py` | 已阅读 | Python/NumPy/Torch 种子及确定性设置。 |

## 已确认并修复

- 旧版 pipeline 缺少 eval 时能力探测崩溃；MPI LOCAL_RANK 未发布；设备监控固定查询设备 0。
- ParallelConfig 重复校验清理；未知 packing 名称拒绝。
- TP 显式 pattern 优先于默认 pattern；EP expert down projection 完整尺寸校验。
- torch_checkpointing HF 最后一次导出的 item_specs 名称遮蔽；优化器初始化异常时恢复 lr 和 grad。
- 合成数据种子归一化为 64 位；KCP 增加独立 degree，拒绝 packed、headtail 和混合 softmax 模型。

## 后续验证与限制

- 已逐个阅读 147/147 个 Python 文件；这是实现审查覆盖，不等于所有组合均已运行通过。
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
- 清单与当前源文件集合一致：147 条，无缺失、无重复。

## 本轮结构调整

- EP dispatch 实现整体迁移至 `parallel/expert_parallel/token_dispatch.py`。
- 优化器状态管理集中于 `components/optimizer/state_dict.py`。
- TP 模型构件改名为 `models/common/tp_projections.py`；前馈类使用
  `TensorParallelFeedForward`，前置检查使用 `validate_tp_projection_preconditions`。
- 删除 HF `input_processing.py`：reshape 直接放在唯一调用方
  `preprocess_inputs`，目标构造直接复用 `components.loss.next_token_targets`。
- 当前源文件数为 147；合并后的预处理逻辑继续在 `models/hf/model.py` 中审查。

- 设备树模块改为 `accelerator/tensor_tree.py`，公开函数改为
  `get_tensor_tree_device` 和 `move_tensor_tree`；旧名称删除并更新全部调用方。

- 数据契约集中在 `datasets/contracts.py`：`TextSequence` 和 `SampleProcessor`
  从构建模块迁入；配方组合与 Grain 构建改为 `datasets/composition.py`，
  dataloader 装配改为 `datasets/loader_builder.py`。旧模块删除，消费者直接引用定义模块。
