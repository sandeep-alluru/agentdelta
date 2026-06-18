# Adding a Framework Adapter

redline ships with a LangChain/LangGraph adapter. This guide shows how to add support for another framework.

## Overview

An adapter is a thin shim that hooks into a framework's event system and emits `TraceNode` objects to an `AgentTrace`. The only contract is:

1. One `NodeType.START` node when the agent begins
2. One `NodeType.LLM` node for each LLM generation
3. One `NodeType.TOOL_CALL` + `NodeType.TOOL_RETURN` pair per tool execution
4. One `NodeType.END` node when the agent finishes
5. A `record_<framework>()` context manager that saves the trace on exit

## Step 1 — Create the adapter module

```python
# src/redline/instrument_myframework.py
from __future__ import annotations

import contextlib
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from redline.trace import AgentTrace, EdgeType, NodeType, TraceEdge, TraceNode


class MyFrameworkAdapter:
    """Records a my-framework agent run as an AgentTrace."""

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.trace = AgentTrace(run_id=self.run_id)
        self._step = 0

    def _next_step(self) -> int:
        self._step += 1
        return self._step

    def on_agent_start(self, inputs: dict[str, Any]) -> None:
        """Hook into the framework's agent-start event."""
        step = self._next_step()
        self.trace.add_node(
            TraceNode(step=step, node_type=NodeType.START, content=str(inputs)[:500])
        )

    def on_llm_response(self, text: str) -> None:
        """Hook into the framework's LLM-response event."""
        step = self._next_step()
        node = TraceNode(step=step, node_type=NodeType.LLM, content=text[:2000])
        self.trace.add_node(node)
        if step > 1:
            self.trace.add_edge(TraceEdge(step - 1, step, EdgeType.LLM_DECISION))

    def on_tool_call(self, tool_name: str, tool_input: str) -> None:
        """Hook into the framework's tool-call event."""
        step = self._next_step()
        content = f"{tool_name}({tool_input[:500]})"
        node = TraceNode(step=step, node_type=NodeType.TOOL_CALL, content=content)
        self.trace.add_node(node)
        if step > 1:
            self.trace.add_edge(TraceEdge(step - 1, step, EdgeType.TOOL_CALL, tool_name))

    def on_tool_result(self, output: str) -> None:
        """Hook into the framework's tool-result event."""
        step = self._next_step()
        node = TraceNode(step=step, node_type=NodeType.TOOL_RETURN, content=str(output)[:500])
        self.trace.add_node(node)
        if step > 1:
            self.trace.add_edge(TraceEdge(step - 1, step, EdgeType.TOOL_RETURN))

    def on_agent_finish(self, output: str) -> None:
        """Hook into the framework's agent-finish event."""
        step = self._next_step()
        node = TraceNode(step=step, node_type=NodeType.END, content=str(output)[:500])
        self.trace.add_node(node)
        if step > 1:
            self.trace.add_edge(TraceEdge(step - 1, step, EdgeType.SEQUENCE, "chain_end"))


@contextlib.contextmanager
def record_myframework(
    output_path: str | Path, run_id: str | None = None
) -> Iterator[MyFrameworkAdapter]:
    """Context manager that records a my-framework run and saves the trace on exit."""
    adapter = MyFrameworkAdapter(run_id=run_id)
    try:
        yield adapter
    finally:
        adapter.trace.save(output_path)
```

## Step 2 — Export from `__init__.py`

Add to `src/redline/__init__.py`:

```python
from redline.instrument_myframework import MyFrameworkAdapter, record_myframework

__all__ = [
    ...,
    "MyFrameworkAdapter",
    "record_myframework",
]
```

Keep `__all__` sorted alphabetically (ruff RUF022 will enforce this).

## Step 3 — Add optional dependency

In `pyproject.toml`:

```toml
[project.optional-dependencies]
myframework = ["myframework>=1.0"]
```

## Step 4 — Write tests

```python
# tests/test_instrument_myframework.py
from redline import AgentTrace
from redline.instrument_myframework import MyFrameworkAdapter, record_myframework
from redline.trace import NodeType


def test_adapter_captures_llm_node():
    adapter = MyFrameworkAdapter(run_id="test")
    adapter.on_agent_start({"input": "hello"})
    adapter.on_llm_response("I'll help with that.")
    assert len(adapter.trace.nodes) == 2
    assert adapter.trace.nodes[1].node_type == NodeType.LLM


def test_adapter_captures_tool_call():
    adapter = MyFrameworkAdapter(run_id="test")
    adapter.on_agent_start({"input": "hello"})
    adapter.on_tool_call("search", "query='test'")
    adapter.on_tool_result("result text")
    tool_nodes = [n for n in adapter.trace.nodes if n.node_type == NodeType.TOOL_CALL]
    assert len(tool_nodes) == 1
    assert "search" in tool_nodes[0].content


def test_record_saves_jsonl(tmp_path):
    output = tmp_path / "run.jsonl"
    with record_myframework(output, run_id="test") as adapter:
        adapter.on_agent_start({"input": "test"})
        adapter.on_agent_finish("done")
    loaded = AgentTrace.load(output)
    assert loaded.run_id == "test"
    assert len(loaded.nodes) == 2
```

## Step 5 — Update README

Add a snippet to the "Quick start" section in `README.md` showing how to use the new adapter, and add the framework to the features table.

## Tip: use `/project:add-adapter` in Claude Code

Type `/project:add-adapter <framework>` in Claude Code and it will scaffold all five steps above for you.
