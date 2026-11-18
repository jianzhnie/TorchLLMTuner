"""Keep generic training layers independent of concrete model implementations."""

import ast
from importlib.util import resolve_name
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "llmtuner"
MODEL_FAMILIES = {
    path.name
    for path in (SOURCE / "models").iterdir()
    if (path / "__init__.py").is_file() and path.name != "common"
}


def imported_modules(path):
    relative = path.relative_to(SOURCE.parent)
    package = ".".join(relative.parts[:-1])
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = resolve_name("." * node.level + module, package)
            yield module, node.lineno
            for alias in node.names:
                yield f"{module}.{alias.name}", node.lineno


@pytest.mark.parametrize("layer", ["models", "parallel", "components"])
def test_layer_imports_preserve_responsibility_boundaries(layer):
    violations = []
    for path in sorted((SOURCE / layer).rglob("*.py")):
        relative = path.relative_to(SOURCE.parent)
        for module, line in imported_modules(path):
            parts = module.split(".")
            if parts[0] != "llmtuner":
                continue
            forbidden = parts[1:2] == ["trainer"]
            if layer in ("parallel", "components"):
                forbidden |= (
                    len(parts) > 2
                    and parts[1] == "models"
                    and parts[2] in MODEL_FAMILIES
                )
            if forbidden:
                violations.append(f"{relative}:{line}: {module}")
    assert not violations, "Forbidden layer dependencies:\n" + "\n".join(violations)


def test_production_does_not_import_experiments():
    violations = [
        f"{path.relative_to(SOURCE.parent)}:{line}: {module}"
        for path in sorted(SOURCE.rglob("*.py"))
        for module, line in imported_modules(path)
        if module.split(".")[0] == "experiments"
    ]
    assert not violations, "Production imports experimental code:\n" + "\n".join(
        violations
    )
