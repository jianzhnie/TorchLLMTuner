---
name: code-style
description: TorchLLMTuner 全局代码风格与约定
---

# 代码风格

本规则集为全仓库默认风格；如有更具体的规则文件，以更具体者为准。

## 语言与格式

- Python 3.10 语法；四空格缩进；**88 字符行宽**。
- Ruff 强制规则为 `E`、`F`、`I`、`UP`、`B`——提交前运行
  `ruff check llmtuner tests` 与 `ruff format --check llmtuner tests`。
- 文件以换行符结尾；模块级定义之间空两行。
- 提交代码中不留含糊的 `TODO`；必须保留时写明具体可执行的内容。
- 注释解释"为什么"，不复述代码本身在做什么。

## 命名与类型标注

- 模块与函数用 `snake_case`；类用 `PascalCase`；常量用 `UPPER_CASE`。
- 私有名称加 `_leading_underscore`；不自定义双下划线开头结尾的名称。
- 所有公开函数签名必须有类型标注。
- 命名应表明职责，例如 `model_replacement.py`、`assemble_trainer_runtime`。

## 导入

- 实现代码中从**定义所在的叶子模块**导入（如
  `from llmtuner.parallel.topology import ...`），不经包门面中转。
- 包的 `__init__.py` 只暴露刻意设计的公共 API。
- 导入顺序：标准库、第三方、本仓库（ruff `I` 规则强制）。
- 导入放在模块顶部；例外仅限循环依赖、可选硬件依赖（如 `torch_npu`）、
  或 `TYPE_CHECKING` 仅类型符号——每种例外附简短注释说明原因。

## 错误处理与设计

- 在边界处校验输入；抛出带描述信息的 `ValueError`；业务逻辑不用 `assert`。
- 优先组合而非继承；避免死代码与过高复杂度。
- 路径处理用 `pathlib`/`os.path`，不手工拼接字符串。
- 库代码用 logging 框架而非 `print`。

## 提交规范

Conventional Commits：`<type>(<scope>): <subject>`，例如
`refactor(parallel): reorganize topology modules`、
`feat(tokenizer): add multimodal tokens`。

- 允许的 type：`feat`、`fix`、`refactor`、`docs`、`test`、`chore`。
- 主题行：祈使句，约 80 字符，结尾不加句号。
- 正文解释"为什么"而非"做了什么"。
- 不含 AI 助手或 IDE 署名尾注（如 `Co-authored-by:`、`Made-with:`）。
