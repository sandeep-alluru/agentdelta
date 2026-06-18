# redline — Codex Developer Guide

> Read by OpenAI Codex CLI. Supplements AGENTS.md with Codex-specific conventions.

## What this project does

redline is a Python library and CLI for semantic diff of AI agent behavior. It captures LLM reasoning steps, tool calls, and tool returns as JSONL traces, embeds them locally with `all-MiniLM-L6-v2`, and finds the first step where two runs diverged.

Primary use case: behavioral regression testing in CI — detect when a prompt change, model upgrade, or tool swap silently changes *how* the agent reasons, not just what it outputs.

## Module map

```
src/redline/
├── trace.py        # TraceNode (content-addressed), TraceEdge, AgentTrace (JSONL I/O)
├── embed.py        # Thread-safe SentenceTransformer singleton + cosine alignment
├── diff.py         # diff_traces() → DiffResult, ForkPoint, StepDiff
├── instrument.py   # LangChain/LangGraph callback + record() context manager
├── report.py       # Rich terminal / JSON / GitHub PR Markdown output
├── cli.py          # Click CLI: redline diff, redline inspect
└── mcp_server.py   # MCP server (pip install redline[mcp])
```

## Build and test commands

```bash
# Full suite (run before any commit)
make all            # lint + typecheck + test

# Individual
make test           # pytest --cov=redline (43 tests, target ≥85% coverage)
make lint           # ruff check + ruff format --check
make typecheck      # mypy
make fmt            # ruff format (auto-fix)
```

## Running the CLI locally

```bash
pip install -e ".[dev]"
redline diff examples/baseline.jsonl examples/candidate.jsonl
redline inspect examples/baseline.jsonl
```

## Key invariants — never change without tests

1. `TraceNode.id` is `sha256("{node_type}:{content}")[:16]` — content-addressed, deterministic
2. `_get_model()` in `embed.py` is the only place `SentenceTransformer` is imported — thread-safe singleton
3. `fork_threshold=0.70`, `match_threshold=0.85` — calibrated defaults; changing them breaks existing CI integrations
4. JSONL wire format in `trace.py` is public API — any schema change requires a migration path
5. `has_regression` is `True` iff `fork_point is not None` — no additional state

## Code conventions

- Python 3.10+, fully type-annotated, mypy strict
- Ruff rules: E W F I UP B S N SIM RUF PT; ignore S101 (asserts in tests), N806 (numpy convention)
- No `print()` in library code — use `rich.console.Console`
- All public functions and classes require docstrings
- New output formats → `report.py` + `cli.py` --format choice + tests
- New framework adapters → `instrument_<framework>.py` + export from `__init__.py` + tests

## Adding a new framework adapter

1. Create `src/redline/instrument_<framework>.py`
2. Implement a context manager matching the `record()` API in `instrument.py`
3. Export from `__init__.py` and add to `__all__` alphabetically
4. Add tests in `tests/test_instrument_<framework>.py`
5. Add `[<framework>]` optional dependency to `pyproject.toml`

## What NOT to do

- Do not import `SentenceTransformer` outside of `embed.py`
- Do not commit `coverage.xml` — it is in `.gitignore`
- Do not push without running `make all` first
- Do not bump `pyproject.toml` version — releases are cut by the maintainer from CHANGELOG
