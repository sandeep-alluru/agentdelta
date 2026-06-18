# Architecture

agentdelta is a pure-Python library with five independent modules and a thin CLI layer.

## Module map

```
src/agentdelta/
├── trace.py        # Data model: TraceNode, TraceEdge, AgentTrace
├── embed.py        # Embedding + sliding-window alignment
├── diff.py         # Fork detection algorithm → DiffResult
├── instrument.py   # LangChain callback + record() context manager
├── report.py       # Rich / JSON / Markdown output formatters
└── cli.py          # Click CLI (thin wrapper over diff + report)
```

## Data flow

```
Agent run
    │
    ▼  (instrument.py)
AgentdeltaCallback
    │  hooks: on_llm_end, on_tool_start, on_tool_end,
    │         on_chain_start, on_chain_end
    ▼
AgentTrace (trace.py)
    │  list[TraceNode] + list[TraceEdge]
    │  saved as JSONL
    ▼
embed_trace() (embed.py)
    │  SentenceTransformer("all-MiniLM-L6-v2")
    │  each node.content → 384-dim float vector
    ▼
align_traces() (embed.py)
    │  sliding-window cosine similarity (window=5)
    │  greedy 1:1 matching
    ▼
diff_traces() (diff.py)
    │  per-pair status: match / changed / added / removed
    │  first pair below fork_threshold → ForkPoint
    ▼
DiffResult (diff.py)
    │
    ├── print_diff()   → Rich terminal table
    ├── to_json()      → JSON string (CI/CD)
    └── to_markdown()  → GitHub PR comment Markdown
```

## Trace format (JSONL)

Each trace is a `.jsonl` file — one JSON object per line.

```jsonl
{"type": "trace_meta", "run_id": "v1.0"}
{"type": "node", "id": "a3f8...", "step": 1, "node_type": "start",     "content": "What is the weather in Tokyo?"}
{"type": "node", "id": "b9c1...", "step": 2, "node_type": "llm",       "content": "I should look up the current weather."}
{"type": "node", "id": "d2e4...", "step": 3, "node_type": "tool_call", "content": "get_weather(location='Tokyo')"}
{"type": "node", "id": "f5a7...", "step": 4, "node_type": "tool_return","content": "{\"temp\": 22, \"condition\": \"sunny\"}"}
{"type": "node", "id": "c8b2...", "step": 5, "node_type": "llm",       "content": "The weather in Tokyo is 22°C and sunny."}
{"type": "node", "id": "e1d3...", "step": 6, "node_type": "end",       "content": "Tokyo: 22°C, sunny."}
{"type": "edge", "source_step": 1, "target_step": 2, "edge_type": "sequence",     "label": ""}
{"type": "edge", "source_step": 2, "target_step": 3, "edge_type": "llm_decision", "label": ""}
{"type": "edge", "source_step": 3, "target_step": 4, "edge_type": "tool_call",    "label": "get_weather"}
{"type": "edge", "source_step": 4, "target_step": 5, "edge_type": "tool_return",  "label": "tool_output"}
{"type": "edge", "source_step": 5, "target_step": 6, "edge_type": "sequence",     "label": "chain_end"}
```

Node IDs are content-addressed: `SHA-256[:16]` of `"{node_type}:{content}"`.
The same reasoning step always produces the same ID regardless of which run it appears in.

## Alignment algorithm

The sliding-window alignment in `embed.py:align_traces()` is a greedy 1:1 matcher:

1. For each node `na` in trace A at index `i`, search trace B nodes in the window `[i-window, i+window]`
2. Find the candidate `nb` maximising `cosine_similarity(na.embedding, nb.embedding)`
3. If `score >= threshold`: match `(na, nb, score)` and mark `nb` as used
4. Else: emit `(na, None, 0.0)` — node removed
5. After all of trace A is processed, append unmatched trace B nodes as `(None, nb, 0.0)` — nodes added

This is O(n·window) per trace pair — fast for typical agent traces (< 100 steps).

For very long traces (> 500 steps), a future optimization is full DTW (dynamic time warping) alignment.

## Fork detection

`diff_traces()` iterates the alignment and labels each pair:

| Condition | Status |
|---|---|
| `score >= match_threshold` (default 0.85) | `match` |
| `score >= fork_threshold` (default 0.70) | `changed` |
| `na is None` | `added` |
| `nb is None` | `removed` |

The first `changed` pair becomes the `ForkPoint`.

`has_regression` is `True` iff a `ForkPoint` exists — i.e., at least one step fell below `fork_threshold`.

## Embedding model

`all-MiniLM-L6-v2` (22M parameters, 384-dim output) is chosen for:
- Offline / local inference — no API key required
- Fast: ~5ms per step on CPU
- Good semantic clustering of short instruction-following text
- MIT license

The model is lazy-loaded on first call to `embed_trace()` and cached as a module-level singleton.
