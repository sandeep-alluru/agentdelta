"""LangChain/LangGraph callback handler for capturing agent traces."""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from agentdelta.trace import AgentTrace, EdgeType, NodeType, TraceEdge, TraceNode


class AgentdeltaCallback:
    """
    LangChain BaseCallbackHandler-compatible callback that records agent runs
    as AgentTrace objects.

    Usage:
        callback = AgentdeltaCallback()
        agent.invoke({"input": "..."}, config={"callbacks": [callback]})
        trace = callback.trace
        trace.save("run.jsonl")
    """

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.trace = AgentTrace(run_id=self.run_id)
        self._step = 0

    def _next_step(self) -> int:
        self._step += 1
        return self._step

    # ── LangChain callback interface ──────────────────────────────────────────

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        """No-op — input prompts are not captured; only LLM output is recorded."""

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Record an LLM generation as a ``NodeType.LLM`` trace node."""
        try:
            text = response.generations[0][0].text
        except (AttributeError, IndexError):
            text = str(response)
        step = self._next_step()
        node = TraceNode(step=step, node_type=NodeType.LLM, content=text[:2000])
        self.trace.add_node(node)
        if step > 1:
            self.trace.add_edge(TraceEdge(step - 1, step, EdgeType.LLM_DECISION, "llm_output"))

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """Record a tool invocation as a ``NodeType.TOOL_CALL`` trace node."""
        tool_name = serialized.get("name", "unknown_tool")
        step = self._next_step()
        content = f"{tool_name}({input_str[:500]})"
        node = TraceNode(step=step, node_type=NodeType.TOOL_CALL, content=content)
        self.trace.add_node(node)
        if step > 1:
            self.trace.add_edge(TraceEdge(step - 1, step, EdgeType.TOOL_CALL, tool_name))

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Record a tool result as a ``NodeType.TOOL_RETURN`` trace node."""
        step = self._next_step()
        node = TraceNode(step=step, node_type=NodeType.TOOL_RETURN, content=str(output)[:500])
        self.trace.add_node(node)
        if step > 1:
            self.trace.add_edge(TraceEdge(step - 1, step, EdgeType.TOOL_RETURN, "tool_output"))

    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        """Record the initial chain input as a ``NodeType.START`` node (first call only)."""
        if self._step == 0:
            step = self._next_step()
            node = TraceNode(
                step=step,
                node_type=NodeType.START,
                content=str(inputs.get("input", inputs))[:500],
            )
            self.trace.add_node(node)

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Record the final chain output as a ``NodeType.END`` node."""
        step = self._next_step()
        node = TraceNode(
            step=step,
            node_type=NodeType.END,
            content=str(outputs.get("output", outputs))[:500],
        )
        self.trace.add_node(node)
        if step > 1:
            self.trace.add_edge(TraceEdge(step - 1, step, EdgeType.SEQUENCE, "chain_end"))

    def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        """No-op — agent actions are captured via ``on_tool_start``."""

    def on_agent_finish(self, finish: Any, **kwargs: Any) -> None:
        """No-op — agent finish is captured via ``on_chain_end``."""

    # LangGraph compatibility shims (it checks for these via hasattr)
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """No-op error shim required for LangGraph compatibility."""

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        """No-op error shim required for LangGraph compatibility."""

    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        """No-op error shim required for LangGraph compatibility."""


@contextlib.contextmanager
def record(output_path: str | Path, run_id: str | None = None) -> Iterator[AgentdeltaCallback]:
    """
    Context manager that records an agent run and saves the trace on exit.

    Usage:
        with agentdelta.record("run_a.jsonl") as cb:
            agent.invoke({"input": "..."}, config={"callbacks": [cb]})
        # trace_a.jsonl is now saved
    """
    callback = AgentdeltaCallback(run_id=run_id)
    try:
        yield callback
    finally:
        callback.trace.save(output_path)
