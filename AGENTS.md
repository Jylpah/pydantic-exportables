# AGENTS.md

## Project overview

This is a Python project. Prefer small, explicit changes. Keep existing architecture and naming unless the task clearly requires refactoring.

## Environment

Use `uv` for dependency management and command execution.

Do not use `pip install` directly. Do not create virtual environments manually unless asked.

## Common commands

Install dependencies:

```bash
uv sync --all-extras --dev
```

Run tests:

```bash
uv run pytest
```

Check syntax:

```bash
uv run ruff check src tests
uv run mypy src tests
```

Format Python code:

```bash
uv run ruff format .
```