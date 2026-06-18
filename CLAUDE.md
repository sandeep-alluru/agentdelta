# agentdelta — Developer Guide

agentdelta is a semantic diff engine for AI agent behavior. It records step-by-step agent traces as JSONL, embeds nodes with `all-MiniLM-L6-v2`, aligns two traces by cosine similarity, and detects the first semantic fork point.

## Architecture

```
trace.py        Data model: NodeType, EdgeType, TraceNode, TraceEdge, AgentTrace
embed.py        sentence-transformers singleton + sliding-window alignment
diff.py         Fork detection → DiffResult, ForkPoint, StepDiff
instrument.py   LangChain callback (AgentdeltaCallback) + record() context manager
report.py       Rich terminal / JSON / Markdown output formatters
cli.py          Click CLI: `agentdelta diff` and `agentdelta inspect`
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for full data flow and algorithm details.

## Setup

```bash
git clone https://github.com/sandeep-alluru/agentdelta
cd agentdelta
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Development commands

```bash
make test       # pytest with branch coverage (fails under 85%)
make lint       # ruff check + format check
make typecheck  # mypy --strict
make fmt        # ruff format (auto-fix)
make all        # lint + typecheck + test
```

## Running tests

```bash
pytest                          # all 43 tests
pytest tests/test_diff.py -v    # single module
pytest -k "test_fork"           # by name
```

## Key invariants

- `TraceNode.id` is content-addressed (SHA-256[:16] of `{node_type}:{content}`). Same content = same ID across runs.
- `embed_trace()` mutates nodes in-place and is idempotent — safe to call twice.
- `align_traces()` uses greedy 1:1 matching, O(n·window). Default window=5.
- `fork_threshold=0.70`: first aligned pair below this becomes the `ForkPoint`.
- `match_threshold=0.85`: pairs above this are "match" (no change).
- `has_regression` is True iff `fork_point is not None`.

## Adding a new output format

1. Add a `to_<format>(result: DiffResult) -> str` function in `report.py`
2. Add the format name to the `--format` choice in `cli.py`
3. Add tests in `tests/test_report.py`

## Adding a new instrumentation adapter

1. Create `src/agentdelta/instrument_<framework>.py`
2. Implement a class that emits `TraceNode` objects to an `AgentTrace`
3. Export from `src/agentdelta/__init__.py`
4. Document in README under "Quick start"

## PR conventions

- One logical change per PR; keep diffs small and reviewable
- All PRs require passing CI (lint + typecheck + test on all matrix combinations)
- Update `CHANGELOG.md` under `[Unreleased]` for any user-visible change
- Do not bump `pyproject.toml` version — releases are cut from `CHANGELOG.md` by the maintainer

## Trace format

```jsonl
{"type": "trace_meta", "run_id": "v1.0"}
{"type": "node", "step": 1, "node_type": "start", "content": "...", ...}
{"type": "node", "step": 2, "node_type": "llm", "content": "...", ...}
{"type": "edge", "source_step": 1, "target_step": 2, "edge_type": "sequence", ...}
```

New node types must be added to `NodeType` in `trace.py` and handled in `_describe_fork()` in `diff.py`.
