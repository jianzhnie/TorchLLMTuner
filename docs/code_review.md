# llmtuner 全局代码审查

本报告记录本轮与上一轮连续审查的结果。静态语法与 Ruff 检查覆盖
`llmtuner` 的全部 148 个 Python 文件；人工审查重点是训练调用链、参数转换、
恢复状态和异常清理路径。静态检查与 CPU 回归测试不能替代多设备数值等价测试，
本报告不将未运行的硬件组合记为已验证。

## 范围与职责

| 层 | Python 文件数 | 审查关注点 |
| --- | ---: | --- |
| 包入口 | 3 | 公共导出、CLI 入口 |
| `accelerator` | 7 | 设备选择、collective、嵌套 tensor 转移 |
| `components` | 19 | loss、优化器、EMA、checkpoint、metrics、profiler |
| `config` | 19 | 字段校验、CLI 配置组合、并行能力限制 |
| `datasets` | 22 | DP ownership、packing、标签对齐、多模态处理 |
| `memory` | 3 | activation checkpoint 与重计算区域 |
| `models` | 30 | HF 适配、attention、投影、MoE routing 与 dispatch |
| `parallel` | 29 | topology、阶段顺序、TP/FSDP pattern、CP/PP/EP 转换 |
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

质量门禁：全量 CPU pytest、`pre-commit run --all-files`、Ruff lint/format、
`python -m compileall -q llmtuner`、`git diff --check`。

本轮最终结果：`1091 passed, 10 skipped`。上述质量门禁均通过。

分布式测试需根据脚本要求选择 rank 数、设备和模型。全量 CPU 测试中的跳过项
以及尚未运行的 torchrun 脚本不能视为通过。本轮不发布或推送修改。
