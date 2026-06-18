# Trace Format

agentdelta traces are `.jsonl` files — one JSON object per line, human-readable, and `git diff`-able.

## Record types

### `trace_meta` (first line)

```json
{"type": "trace_meta", "run_id": "v1.0"}
```

| Field | Type | Description |
|---|---|---|
| `type` | `"trace_meta"` | Always `"trace_meta"` |
| `run_id` | `string` | Unique identifier for this run |

Any additional fields are stored as trace metadata.

### `node`

```json
{
  "type": "node",
  "id": "a3f8b2c1d4e5f678",
  "step": 3,
  "node_type": "tool_call",
  "content": "get_weather(location='Tokyo')",
  "metadata": {}
}
```

| Field | Type | Description |
|---|---|---|
| `type` | `"node"` | Always `"node"` |
| `id` | `string` | Content-addressed SHA-256[:16] of `"{node_type}:{content}"` |
| `step` | `integer` | 1-based sequential position |
| `node_type` | `string` | One of `start`, `llm`, `tool_call`, `tool_return`, `end` |
| `content` | `string` | Human-readable step content (truncated to 2000 chars for LLM, 500 for tools) |
| `metadata` | `object` | Framework-specific key/value pairs |

### `edge`

```json
{
  "type": "edge",
  "source_step": 3,
  "target_step": 4,
  "edge_type": "tool_call",
  "label": "get_weather"
}
```

| Field | Type | Description |
|---|---|---|
| `type` | `"edge"` | Always `"edge"` |
| `source_step` | `integer` | Step number of the source node |
| `target_step` | `integer` | Step number of the target node |
| `edge_type` | `string` | One of `llm_decision`, `tool_call`, `tool_return`, `sequence` |
| `label` | `string` | Optional human-readable label |

## Node types

| Type | When emitted | Content |
|---|---|---|
| `start` | First chain invocation | User input text |
| `llm` | After each LLM generation | Reasoning/response text (≤2000 chars) |
| `tool_call` | Before each tool execution | `tool_name(input)` (≤500 chars) |
| `tool_return` | After each tool execution | Tool output (≤500 chars) |
| `end` | Final chain output | Final agent output (≤500 chars) |

## Full example

```jsonl
{"type": "trace_meta", "run_id": "v1.0"}
{"type": "node", "id": "a3f8b2c1", "step": 1, "node_type": "start",      "content": "What is the weather in Tokyo?", "metadata": {}}
{"type": "node", "id": "b9c1d2e3", "step": 2, "node_type": "llm",        "content": "I should look up the current weather.", "metadata": {}}
{"type": "node", "id": "d2e4f5a6", "step": 3, "node_type": "tool_call",  "content": "get_weather(location='Tokyo')", "metadata": {}}
{"type": "node", "id": "f5a7b8c9", "step": 4, "node_type": "tool_return","content": "{\"temp\": 22, \"condition\": \"sunny\"}", "metadata": {}}
{"type": "node", "id": "c8b2a1d3", "step": 5, "node_type": "end",        "content": "Tokyo: 22°C, sunny.", "metadata": {}}
{"type": "edge", "source_step": 1, "target_step": 2, "edge_type": "sequence",     "label": ""}
{"type": "edge", "source_step": 2, "target_step": 3, "edge_type": "llm_decision", "label": ""}
{"type": "edge", "source_step": 3, "target_step": 4, "edge_type": "tool_call",    "label": "get_weather"}
{"type": "edge", "source_step": 4, "target_step": 5, "edge_type": "tool_return",  "label": "tool_output"}
```

## Writing traces from any framework

You don't need the LangChain callback. Emit nodes directly:

```python
from agentdelta import AgentTrace
from agentdelta.trace import TraceNode, TraceEdge, NodeType, EdgeType

trace = AgentTrace(run_id="my_run_v1")

trace.add_node(TraceNode(step=1, node_type=NodeType.START, content="user prompt here"))
trace.add_node(TraceNode(step=2, node_type=NodeType.LLM, content="reasoning text"))
trace.add_node(TraceNode(step=3, node_type=NodeType.TOOL_CALL, content="my_tool(input)"))
trace.add_node(TraceNode(step=4, node_type=NodeType.TOOL_RETURN, content="tool output"))
trace.add_node(TraceNode(step=5, node_type=NodeType.END, content="final answer"))

trace.add_edge(TraceEdge(1, 2, EdgeType.SEQUENCE))
trace.add_edge(TraceEdge(2, 3, EdgeType.LLM_DECISION))
trace.add_edge(TraceEdge(3, 4, EdgeType.TOOL_CALL, label="my_tool"))
trace.add_edge(TraceEdge(4, 5, EdgeType.TOOL_RETURN))

trace.save("my_run_v1.jsonl")
```
