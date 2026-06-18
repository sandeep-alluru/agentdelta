# Contributing to agentdelta

Thank you for your interest in contributing. This guide covers everything you need to go from zero to a merged PR.

## What we're looking for

| Contribution type | Notes |
|---|---|
| Bug fixes | Always welcome — open an issue first if it's non-obvious |
| New output formats | `--format html`, `--format junit`, etc. |
| New instrumentation adapters | OpenAI Agents SDK, Autogen, CrewAI, Agno |
| Alignment algorithm improvements | Better windowing, dynamic programming alignment |
| Performance improvements | Batched embedding, async support |
| Documentation | Examples, guides, translations |
| Tests | More edge cases, property-based tests |

## Quick start

```bash
git clone https://github.com/sandeep-alluru/agentdelta
cd agentdelta
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Running checks

```bash
make test       # run the full test suite
make lint       # ruff check + ruff format --check
make typecheck  # mypy
make all        # lint + typecheck + test
```

Or individually:

```bash
pytest tests/ -v
ruff check src/ tests/
mypy src/agentdelta/
```

## Adding a new output format

1. Add a `to_<format>(result: DiffResult) -> str` function in `src/agentdelta/report.py`
2. Add the format name to the `--format` choice in `src/agentdelta/cli.py`
3. Add a `test_to_<format>_*` test in `tests/test_report.py`
4. Add an example to `examples/` if the format is non-trivial
5. Document it in the `## Output formats` section of the README

## Adding a new instrumentation adapter

1. Create `src/agentdelta/<framework>_adapter.py`
2. Implement a callback/decorator/hook that emits `TraceNode` and `TraceEdge` objects
3. Export the class from `src/agentdelta/__init__.py`
4. Add tests in `tests/test_<framework>_adapter.py`
5. Add the framework to the `[project.optional-dependencies]` section in `pyproject.toml`

## Branch model

- Branch from `main`
- Name branches: `fix/describe-the-bug`, `feat/new-feature`, `docs/what-changed`
- Keep PRs focused — one logical change per PR

## PR requirements

- All tests must pass (`make test`)
- No new lint or type errors (`make lint && make typecheck`)
- New behaviour must have corresponding tests
- Update `CHANGELOG.md` under `[Unreleased]`
- Follow [Conventional Commits](https://www.conventionalcommits.org/) for the PR title:
  `fix:`, `feat:`, `docs:`, `refactor:`, `test:`, `chore:`, `ci:`

## Review timeline

PRs are reviewed within **5 business days**. If you haven't heard back, ping `@sandeep-alluru` in the PR comments.

## Code style

- Ruff for formatting and linting (configured in `pyproject.toml`)
- MyPy for type checking
- All public functions and classes require docstrings
- No `print()` in library code — use `rich.console.Console` or logging
- No silent failures — raise descriptive exceptions at boundaries

## Commit signing

We recommend signing commits (`git config commit.gpgsign true`) but do not require it.
