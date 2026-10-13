# Repository Guidelines

## Project Structure & Module Organization

Source code lives in `llmtuner/`. Keep responsibilities within the existing
layers: `config/` defines configuration, `datasets/` builds data pipelines,
`models/` contains model behavior, `parallel/` applies FSDP/TP/PP/CP/EP,
`memory/` contains memory optimizations, `components/` provides reusable
training services, and `trainer/` orchestrates execution. Parallel families
use their same-named file as the public entry point, such as
`parallel/tensor_parallel/tensor_parallel.py`.

CPU unit tests live in `tests/unit_tests/cpu/`. Distributed equivalence scripts
live in `tests/integration_tests/` and are run explicitly with `torchrun`.
Documentation belongs in `docs/`; runnable configurations and examples belong
in `examples/`. Do not commit generated files from `outputs/`, caches, or
`__pycache__/`.

## Build, Test, and Development Commands

- `python -m pip install -e '.[dev]'`: install the package and development tools.
- `python -m llmtuner --steps 20`: run a short local training job.
- `pytest -q --disable-warnings`: run the CPU unit suite.
- `ruff check llmtuner tests`: check imports, style, and common Python errors.
- `ruff format --check llmtuner tests`: verify formatting.
- `pre-commit run --all-files`: run every repository quality gate.
- `python -m compileall -q llmtuner`: verify that all modules compile.

## Coding Style & Naming Conventions

Use Python 3.10 syntax, four-space indentation, type annotations, and an
88-character line limit. Ruff enforces `E`, `F`, `I`, `UP`, and `B` rules.
Use `snake_case` for modules and functions, `PascalCase` for classes, and
`UPPER_CASE` for constants. Choose names that state responsibility, such as
`model_replacement.py` or `assemble_trainer_runtime`. Import from the defining
leaf module inside implementation code; package `__init__.py` files should
expose only intentional public APIs.

## Testing Guidelines

Name tests `test_<behavior>.py` and test functions `test_<expected_behavior>`.
Add CPU tests for pure logic and targeted `torchrun` scripts for collective or
mesh behavior. Gate hardware-dependent tests with `tests.caps.require_env`.
Run the full unit suite and pre-commit before submitting changes.

## Commit & Pull Request Guidelines

History follows Conventional Commit style, for example `refactor(parallel):
reorganize topology modules` or `feat(tokenizer): add multimodal tokens`. Keep
commits focused. Pull requests should explain the problem, summarize structural
or behavioral changes, list validation commands and results, and link relevant
issues. Include logs for runtime changes and screenshots only for visual output.
