#!/bin/bash
# PostToolUse 钩子：修改关键目录后提醒检查关联文件（exit 0，不阻断）

set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))" 2>/dev/null)

if [ -z "$FILE_PATH" ]; then
    exit 0
fi

case "$FILE_PATH" in
    *.py) ;;
    *) exit 0 ;;
esac

case "$FILE_PATH" in
    */llmtuner/parallel/*.py)
        echo "REMINDER: 你修改了并行相关代码 (llmtuner/parallel/)。"
        echo "请检查 tests/integration_tests/ 下对应的 torchrun 等价性脚本是否需要更新，"
        echo "并确认 llmtuner/parallel/ 同名公共入口文件（如 tensor_parallel/tensor_parallel.py）仍然导出一致。"
        ;;
    */llmtuner/config/*.py)
        echo "REMINDER: 你修改了配置定义 (llmtuner/config/)。"
        echo "请同步检查 examples/ 下的示例配置是否需要更新。"
        ;;
    */llmtuner/models/*.py)
        echo "REMINDER: 你修改了模型代码 (llmtuner/models/)。"
        echo "请检查 llmtuner/memory/ 中的内存优化与新模型行为是否兼容。"
        ;;
    */llmtuner/datasets/*.py)
        echo "REMINDER: 你修改了数据管线代码 (llmtuner/datasets/)。"
        echo "请检查 tests/unit_tests/cpu/ 下的数据管线单测是否需要更新。"
        ;;
esac

exit 0
