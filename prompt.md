# TorchLLMTuner 代码优化

请优化 TorchLLMTuner

## 优化目标

1. **NPU 优先**：Ascend NPU 是一等公民，所有优化必须考虑其性能和支持情况，不得损害 NPU 表现；性能结论须实测。
2. **结构**：按职责重组，减少重复、冗余分层和过度封装。
3. **命名**：准确表达职责，避免无必要的下划线前缀，不批量改名。
4. **删减**：确认无引用且不属于公开入口、插件接口或动态调用后，再删除未使用代码。
5. **简化**：以直接、易维护的方案重构复杂实现；抽象必须有明确收益。
6. **质量**：遵循现有风格与分层，更新受影响的注释和文档。
7. **导入**：置于文件顶部；可选依赖、循环依赖或延迟初始化等例外须有理由。

## 执行要求

- 优先优化高收益代码，避免全仓库重写。
- 说明动机、行为影响和验证结果；范围外问题仅记录。
- 运行相关测试，以及 `ruff check llmtuner tests`、`ruff format --check llmtuner tests` 和 `python -m compileall -q llmtuner`。条件允许时运行完整 CPU 单元测试；无法执行时说明原因。
- 不提交 `outputs/`、缓存或其他生成文件。

## 环境与运行约束

| 项目 | 配置（运行前以实测为准） |
| --- | --- |
| 容器 | `hyper-parallel-env` |
| 硬件 / 软件 | 8×Ascend 910、CANN 9.1.0、torch / torch-npu 2.12 |
| 模型 | `/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-30B-A3B` |

从 `available_nodes.txt` 选择节点，在宿主机确认 8 卡空闲并核对挂载、设备和版本，不能仅凭容器内 `npu-smi` 判断占用。用 `docker exec hyper-parallel-env bash -lc '...'` 执行容器命令。离线依赖由宿主机下载至 `/home/jianzhnie/llmtuner/wheels`。
