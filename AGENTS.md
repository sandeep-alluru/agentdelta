# agentdelta — Agent Context

This file is read by AI coding assistants to understand project architecture and conventions:
**OpenAI Codex CLI** · **Claude Code** · **GitHub Copilot** · **Cursor** · **Windsurf** · **Aider** · **Continue.dev**

For tool-specific developer guides see: `CLAUDE.md` (Claude Code) · `CODEX.md` (OpenAI Codex CLI)

## What this project does

agentdelta is a semantic diff engine for AI agent behavior. It records the step-by-step reasoning trace of an LLM agent (LLM calls, tool calls, tool returns) as a JSONL file, then compares two runs and finds the exact step where the agent's behavior diverged.

Primary use case: behavioral regression testing in CI/CD — detect when a model upgrade, prompt change, or tool swap silently changes how an agent reasons, not just what it outputs.

## Module map

```
src/agentdelta/
├── trace.py        # Data model: TraceNode, TraceEdge, AgentTrace (JSONL save/load)
├── embed.py        # SentenceTransformer embeddings + sliding-window alignment
├── diff.py         # Fork detection → DiffResult with ForkPoint
├── instrument.py   # LangChain callback + record() context manager
├── report.py       # Rich terminal / JSON / GitHub PR Markdown output
└── cli.py          # Click CLI: agentdelta diff, agentdelta inspect
```

## Key invariants

- `TraceNode.id` is content-addressed: SHA-256[:16] of `{node_type}:{content}`. Same reasoning step → same ID across runs.
- `embed.py:_get_model()` is thread-safe via double-checked locking with `threading.Lock`.
- `align_traces()` is greedy 1:1 — each trace_a node matches at most one trace_b node within ±window positions. O(n·window).
- `has_regression` is True iff `fork_point is not None` — i.e., at least one aligned pair fell below `fork_threshold`.

## Testing

```bash
make test        # 43 tests, ~20s (loads sentence-transformer on first run)
make lint        # ruff check + format
make typecheck   # mypy
```

HuggingFace model is cached at `~/.cache/huggingface` after first run.

## What NOT to change without careful thought

- `embed.py:align_traces()` window size default (5) — widening it fixes some edge cases but degrades performance on long traces
- `diff.py` threshold defaults (fork=0.70, match=0.85) — these are calibrated; changing breaks existing CI integrations
- JSONL format in `trace.py` — it's the public wire format; any change requires a migration path
