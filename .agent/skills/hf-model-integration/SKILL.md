---
name: hf-model-integration
description: >
  把 HuggingFace 模型接入 TorchLLMTuner：llmtuner/config 配置对齐、经
  llmtuner/models/hf 构造 HF 模型、通过 llmtuner/parallel 各家族公共入口
  挂接并行适配、memory/ 特性兼容性检查、examples 与 tests 交付。适用于
  功能性接入；数值验收用 hf-model-precision-validation。
---

# HF 模型接入

用最小的模型侧实现保住 HF 模型的数学语义，同时复用 llmtuner 的通用
层次。范例如 `examples/train_qwen3.py`：`AutoConfig`/`AutoModelForCausalLM`
构造模型，`llmtuner.config` 描述配置，`Trainer` 跑训练。

## 加载

1. 编辑前读 `AGENTS.md` 与 `.agent/rules/code-style.md`。
2. [references/integration-workflow.md](references/integration-workflow.md)——
   逐步接入流程与检查清单。
3. 挂接并行适配时读 `.agent/rules/distributed.md`。

## 代码导向（真实路径）

| 关注点 | 位置 |
|--------|------|
| HF 模型构造 | `llmtuner/models/hf/model.py`、`factory.py`、`model_setup.py`、`materialization.py` |
| HF 构件 | `llmtuner/models/hf/`（`attention_backend.py`、`qkv_projection.py`、`normalization.py`、`state_dict_adapter.py`） |
| 共享数学组件 | `llmtuner/models/common/`（`feed_forward.py`、`embedding.py`、`rope.py`、`moe/`、`attention/`） |
| 配置面 | `llmtuner/config/model_config.py`、`llmtuner/config/llmtuner_config.py` |
| 并行入口 | `llmtuner/parallel/<family>/<family>.py`，及 `parallelize.py`、`topology.py` |
| 显存特性 | `llmtuner/memory/activation_checkpoint.py`（模式见 `config/activation_checkpoint_config.py`，挂接在 `training_config.py`） |

## 流程（摘要——细节见 references/）

1. **权威来源**：锁定 HF 模型类与 transformers 版本；确认
   `AutoConfig`/`AutoModelForCausalLM` 确实能构造发布的架构（配置能加载
   不等于架构受支持）。`tests/caps.py` 已有 `transformers_qwen3`、
   `transformers_gpt_oss` 等能力探针——新模型家族需在此新增探针。
2. **配置对齐**：把 HF 配置字段映射到 `ModelConfig`
   （`model_name_or_path`、vocab/hidden/layers/heads/kv-heads，模式见
   `examples/train_qwen3.py`）；冒烟用 `AutoConfig.for_model` 构造离线
   小配置。
3. **模型构造**：优先组合 `llmtuner/models/common/` 组件与 `models/hf/`
   辅助模块，而非新写模型代码；架构不变量放模型侧，权重名映射放适配侧
   （`state_dict_adapter.py` 管检查点 FQN 映射）。
4. **并行适配**：经公共入口与声明式 pattern 字段挂接 TP/CP/EP/FSDP
   （`tp_config.tp_colwise_modules` / `tp_rowwise_modules`、
   `fsdp_config.fsdp_apply_modules`），不在通用并行代码里加模型家族特判。
5. **显存兼容性**：验证各 `activation_checkpoint_mode` 与模型 block 结构
   的兼容性（`region` 模式从 block 的 `nn.Linear` 派生 region——确认模型
   暴露了它们）。
6. **示例**：新增 `examples/train_<model>.py`（NPU 多卡启动另配 `.sh`，
   参照 `train_qwen3_30b_npu.*`）；小配置离线可跑；不隐式下载资源。
7. **测试**：配置/纯逻辑补 CPU 单测（`tests/unit_tests/cpu/models/`）；
   并行正确性补 `torchrun` 等价性脚本（`tests/integration_tests/`），用
   `tests.caps.require_env` 门控。

## 完成门槛

接入完成的判定：模型在单卡离线可训练（示例可跑）、配置校验通过、模型
声明支持的并行度通过对应等价性脚本、检查点存取往返一致
（`state_dict_adapter` 的 FQN 与 HF 布局吻合）、通用框架路径不含模型
家族特判。随后移交 `hf-model-precision-validation` 做数值验收。
