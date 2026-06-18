"""Shared fixtures for agentdelta tests."""

from __future__ import annotations

import pytest

from agentdelta.trace import AgentTrace, EdgeType, NodeType, TraceEdge, TraceNode


def _make_trace(run_id: str, steps: list[tuple[NodeType, str]]) -> AgentTrace:
    trace = AgentTrace(run_id=run_id)
    for i, (ntype, content) in enumerate(steps, start=1):
        trace.add_node(TraceNode(step=i, node_type=ntype, content=content))
        if i > 1:
            trace.add_edge(TraceEdge(i - 1, i, EdgeType.SEQUENCE, ""))
    return trace


@pytest.fixture
def simple_trace_a() -> AgentTrace:
    return _make_trace(
        "run_a",
        [
            (NodeType.START, "What is the weather in London?"),
            (NodeType.LLM, "I should use the weather tool to answer this."),
            (NodeType.TOOL_CALL, "get_weather(location='London')"),
            (NodeType.TOOL_RETURN, '{"temp": 18, "condition": "cloudy"}'),
            (NodeType.LLM, "The weather in London is 18°C and cloudy."),
            (NodeType.END, "The weather in London is 18°C and cloudy."),
        ],
    )


@pytest.fixture
def simple_trace_b_match(simple_trace_a) -> AgentTrace:
    """Nearly identical trace — should produce no regression."""
    return _make_trace(
        "run_b",
        [
            (NodeType.START, "What is the weather in London?"),
            (NodeType.LLM, "I need to check the weather tool for London."),
            (NodeType.TOOL_CALL, "get_weather(location='London')"),
            (NodeType.TOOL_RETURN, '{"temp": 18, "condition": "cloudy"}'),
            (NodeType.LLM, "London weather: 18°C, cloudy skies."),
            (NodeType.END, "London weather: 18°C, cloudy skies."),
        ],
    )


@pytest.fixture
def simple_trace_b_fork() -> AgentTrace:
    """Trace that forks at step 3 — uses a different tool."""
    return _make_trace(
        "run_b_fork",
        [
            (NodeType.START, "What is the weather in London?"),
            (NodeType.LLM, "I should use the weather tool to answer this."),
            (NodeType.TOOL_CALL, "web_search(query='London weather today')"),
            (NodeType.TOOL_RETURN, "London: 18°C cloudy according to BBC Weather"),
            (NodeType.LLM, "The weather in London is 18°C and cloudy."),
            (NodeType.END, "The weather in London is 18°C and cloudy."),
        ],
    )
