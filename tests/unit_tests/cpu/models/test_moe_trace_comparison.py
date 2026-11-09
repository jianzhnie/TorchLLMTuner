"""Diagnostic counts use the same tolerance scale as the acceptance result."""

import ast
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]


def comparison():
    namespace = {"torch": torch, "Any": object, "ATOL": 0.005, "RTOL": 0.02}
    for file, name in (
        ("qwen35_cross_framework.py", "compare"),
        ("qwen35_moe_module_trace.py", "compare_branch"),
    ):
        path = ROOT / "tests" / "integration_tests" / file
        node = next(
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
            namespace,
        )
    return namespace["compare_branch"]


def test_failure_count_uses_acceptance_tolerance_scale():
    compare = comparison()
    a, b = torch.tensor([1.0]), torch.tensor([1.0253])
    result = compare(a, b)
    assert not result["passed"] and result["failed_elements"] == 1
    result = compare(b, a)
    assert result["passed"] and result["failed_elements"] == 0


def test_mismatch_has_no_misleading_element_count():
    compare = comparison()
    for value in (torch.ones(2), torch.ones(1, dtype=torch.bfloat16)):
        result = compare(torch.ones(1), value)
        assert not result["passed"] and not result["exact"]
        assert result["failed_elements"] is None


def test_nonfinite_elements_count_as_failures():
    result = comparison()(torch.tensor([float("nan"), 1.0]), torch.ones(2))
    assert not result["passed"] and result["failed_elements"] == 1
