#!/bin/bash
# PostToolUse 钩子：文件写入后自动修复基础代码风格问题。

set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))" 2>/dev/null)

if [ -z "$FILE_PATH" ] || [ ! -f "$FILE_PATH" ]; then
    exit 0
fi

case "$FILE_PATH" in
    *.py) ;;
    *) exit 0 ;;
esac

# ruff 不在 PATH 时静默跳过
if ! command -v ruff >/dev/null 2>&1; then
    exit 0
fi

ruff check --fix "$FILE_PATH" >/dev/null 2>&1 || true
ruff format "$FILE_PATH" >/dev/null 2>&1 || true
