# GitHub Copilot Instructions — agentdelta

agentdelta is a semantic diff engine for AI agent behavior. It records agent execution traces as JSONL, embeds each step with `all-MiniLM-L6-v2`, and detects the first semantic fork point between two runs.

## Architecture

| Module | Purpose |
|---|---|
| `trace.py` | Data model: `NodeType`, `EdgeType`, `TraceNode`, `TraceEdge`, `AgentTrace` |
| `embed.py` | Sentence-transformer singleton + `align_traces()` sliding-window alignment |
| `diff.py` | `diff_traces()` → `DiffResult`, `ForkPoint`, `StepDiff` |
| `instrument.py` | `AgentdeltaCallback` (LangChain) + `record()` context manager |
| `report.py` | `print_diff()`, `to_json()`, `to_markdown()` formatters |
| `cli.py` | Click CLI: `agentdelta diff` and `agentdelta inspect` |

## Key invariants

- `TraceNode.id` is SHA-256[:16] of `"{node_type}:{content}"` — content-addressed, deterministic
- `embed_trace()` mutates nodes in-place; idempotent
- `fork_threshold=0.70` — cosine similarity below this → ForkPoint
- `match_threshold=0.85` — cosine similarity above this → "match" (no change)
- `has_regression` is `True` iff `fork_point is not None`
- Node content is truncated: LLM ≤2000 chars, tool ≤500 chars

## Code style

- Python 3.10+, type-annotated, mypy strict
- Ruff rules: E, W, F, I, UP, B, S, N, SIM, RUF, PT; ignore S101 (assert in tests), N806
- No `print()` in library code — use `rich.console.Console`
- All public classes and functions must have docstrings
- Tests use `pytest`; CLI tests use `click.testing.CliRunner`

## Trace JSONL format

```jsonl
{"type": "trace_meta", "run_id": "v1.0"}
{"type": "node", "id": "...", "step": 1, "node_type": "start", "content": "...", "metadata": {}}
{"type": "edge", "source_step": 1, "target_step": 2, "edge_type": "sequence", "label": ""}
```

## Adding a new output format

Add `to_<format>(result: DiffResult) -> str` to `report.py`, add to `--format` choices in `cli.py`, add tests.

## Adding a new instrumentation adapter

Create `src/agentdelta/instrument_<framework>.py`. See `instrument.py` (LangChain) as reference. Export from `__init__.py`. Add tests.
