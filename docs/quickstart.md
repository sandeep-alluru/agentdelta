# Quick Start

## Install

```bash
pip install agentdelta
```

With LangChain/LangGraph instrumentation:

```bash
pip install "agentdelta[langchain]"
```

Zero-install (no virtualenv):

```bash
pipx run agentdelta --help
```

## Step 1 — Record two runs

=== "LangChain / LangGraph"

    ```python
    from agentdelta import record

    # Baseline (before your change)
    with record("baseline.jsonl", run_id="v1.0") as cb:
        agent.invoke({"input": "..."}, config={"callbacks": [cb]})

    # Candidate (after your change)
    with record("candidate.jsonl", run_id="v1.1") as cb:
        agent.invoke({"input": "..."}, config={"callbacks": [cb]})
    ```

=== "Custom / Any Framework"

    ```python
    from agentdelta import AgentTrace
    from agentdelta.trace import TraceNode, NodeType

    trace = AgentTrace(run_id="my_run")
    trace.add_node(TraceNode(step=1, node_type=NodeType.START,       content="user input"))
    trace.add_node(TraceNode(step=2, node_type=NodeType.LLM,         content="reasoning..."))
    trace.add_node(TraceNode(step=3, node_type=NodeType.TOOL_CALL,   content="tool_name(args)"))
    trace.add_node(TraceNode(step=4, node_type=NodeType.TOOL_RETURN, content="tool result"))
    trace.add_node(TraceNode(step=5, node_type=NodeType.END,         content="final answer"))
    trace.save("my_run.jsonl")
    ```

## Step 2 — Diff the traces

```bash
agentdelta diff baseline.jsonl candidate.jsonl
```

Example output when a tool selection changed:

```
╭───────────────────────────────────────────────╮
│ agentdelta  v1.0 vs v1.1                      │
╰───────────────────────────────────────────────╯
  🔴 REGRESSION DETECTED  3/6 steps matched (50.0%)  1 changed  +1 added  -1 removed

╭────────────────────── Fork Point ──────────────────────╮
│ ⚡ First fork at step 3                               │
│ Tool selection changed: 'get_weather' → 'web_search'  │
╰────────────────────────────────────────────────────────╯

 Step   Status    Type          Detail
    3   CHANGED   🔧 tool_call  Tool selection changed: 'get_weather' → 'web_search'
    4   REMOVED   ↩ tool_return  - [tool_return] {"temp": 22, "condition": "sunny"}
    5   CHANGED   🧠 llm        Reasoning path diverged (similarity: 0.85)
```

Example output when traces are equivalent:

```
╭───────────────────────────────────────────────╮
│ agentdelta  v1.0 vs v1.1                      │
╰───────────────────────────────────────────────╯
  ✅ No regression  6/6 steps matched (100.0%)
```

## Step 3 — Use in CI

```bash
agentdelta diff baseline.jsonl candidate.jsonl --exit-code
# exits 1 if regression detected, 0 if clean
```

## Step 4 — Inspect a single trace

```bash
agentdelta inspect baseline.jsonl
```

## Python API

```python
from agentdelta import AgentTrace, diff_traces
from agentdelta.report import print_diff, to_json, to_markdown

trace_a = AgentTrace.load("baseline.jsonl")
trace_b = AgentTrace.load("candidate.jsonl")

result = diff_traces(trace_a, trace_b)

# Terminal output
print_diff(result)

# Programmatic access
if result.has_regression:
    fp = result.fork_point
    print(f"Fork at step {fp.step_a}: {fp.description}")
    print(f"Similarity: {fp.similarity:.2f}")

# CI/CD JSON
json_str = to_json(result)

# GitHub PR comment
markdown_str = to_markdown(result)
```
