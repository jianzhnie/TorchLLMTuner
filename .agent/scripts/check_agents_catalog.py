#!/usr/bin/env python3
"""校验 .agent/agents/*.md 的 frontmatter 与内部 .agent 引用。

对每个 agent 文件检查：
  * 存在含 name / description / tools 的 YAML frontmatter
  * ``name`` 与文件名（去扩展名）一致
  * 文中引用的每个 ``.agent/rules/...``、``.agent/skills/...`` 路径真实存在

全部通过时退出码为 0；否则非零并输出简短报告。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
REF_RE = re.compile(r"\.agent/(?:rules|skills)/[\w./-]+")

REQUIRED_KEYS = ("name", "description", "tools")


def _parse_frontmatter(text: str) -> dict[str, str] | None:
    """解析 YAML frontmatter 的顶层键；块值（如列表）记为占位符。"""
    match = FRONTMATTER_RE.match(text)
    if not match:
        return None
    fields: dict[str, str] = {}
    lines = match.group(1).splitlines()
    for index, line in enumerate(lines):
        if re.match(r"^[a-z_]+:", line):
            key, _, value = line.partition(":")
            value = value.strip()
            if not value:  # 块值，例如后续行是 YAML 列表
                items = [
                    item for item in lines[index + 1 :] if item.startswith((" ", "-"))
                ]
                value = "list" if items else ""
            fields[key.strip()] = value
    return fields


def main() -> int:
    """运行全部检查并输出报告。"""
    root = Path(__file__).resolve().parents[2]
    agents_dir = root / ".agent" / "agents"
    errors: list[str] = []

    agent_files = sorted(agents_dir.glob("*.md"))
    if not agent_files:
        errors.append(f"{agents_dir} 下未找到 agent 文件")

    for path in agent_files:
        rel = path.relative_to(root)
        text = path.read_text(encoding="utf-8")
        fields = _parse_frontmatter(text)
        if fields is None:
            errors.append(f"{rel}：缺少 YAML frontmatter")
            continue
        for key in REQUIRED_KEYS:
            if key not in fields or not fields[key]:
                errors.append(f"{rel}：frontmatter 缺少 '{key}'")
        if fields.get("name") and fields["name"] != path.stem:
            errors.append(
                f"{rel}：name '{fields['name']}' 与文件名 '{path.stem}' 不一致"
            )
        for ref in sorted(set(REF_RE.findall(text))):
            if not (root / ref).exists():
                errors.append(f"{rel}：引用的路径不存在：{ref}")

    if errors:
        print("agents 目录校验失败：")
        for line in errors:
            print(" ", line)
        return 1
    print(f"agents 目录校验通过（共检查 {len(agent_files)} 个 agent）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
