"""Tests for agentdelta.mcp_server — import guard and helper functions."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agentdelta.trace import AgentTrace, EdgeType, NodeType, TraceEdge, TraceNode


def _make_trace(run_id: str, steps: list[tuple[NodeType, str]]) -> AgentTrace:
    trace = AgentTrace(run_id=run_id)
    for i, (ntype, content) in enumerate(steps, start=1):
        trace.add_node(TraceNode(step=i, node_type=ntype, content=content))
        if i > 1:
            trace.add_edge(TraceEdge(i - 1, i, EdgeType.SEQUENCE, ""))
    return trace


def test_has_mcp_flag_is_bool() -> None:
    """_HAS_MCP must be a boolean."""
    from agentdelta import mcp_server

    assert isinstance(mcp_server._HAS_MCP, bool)


def test_run_server_exits_when_mcp_missing() -> None:
    """run_server() should exit with code 1 when MCP is not available."""
    import agentdelta.mcp_server as mcp_mod

    with patch.object(mcp_mod, "_HAS_MCP", False):
        with pytest.raises(SystemExit) as exc_info:
            mcp_mod.run_server()
        assert exc_info.value.code == 1


def test_diff_helper_returns_dict() -> None:
    """_diff() helper should return a dict with 'summary' and 'steps' keys."""
    import tempfile
    from pathlib import Path

    from agentdelta.mcp_server import _diff

    steps = [
        (NodeType.START, "What is 2+2?"),
        (NodeType.LLM, "The answer is four."),
        (NodeType.END, "four"),
    ]
    a = _make_trace("a", steps)
    b = _make_trace("b", steps)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fa:
        a.save(fa.name)
        path_a = fa.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fb:
        b.save(fb.name)
        path_b = fb.name

    try:
        result = _diff(path_a, path_b, 0.70, 0.85)
        assert isinstance(result, dict)
        assert "summary" in result
        assert "steps" in result
    finally:
        Path(path_a).unlink(missing_ok=True)
        Path(path_b).unlink(missing_ok=True)


def test_inspect_helper_returns_dict() -> None:
    """_inspect() helper should return a dict with run_id and steps."""
    import tempfile
    from pathlib import Path

    from agentdelta.mcp_server import _inspect

    steps = [
        (NodeType.START, "Hello"),
        (NodeType.END, "World"),
    ]
    trace = _make_trace("inspect_run", steps)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        trace.save(f.name)
        tmp = f.name

    try:
        result = _inspect(tmp)
        assert isinstance(result, dict)
        assert result["run_id"] == "inspect_run"
        assert "steps" in result
        assert result["total_nodes"] == 2
    finally:
        Path(tmp).unlink(missing_ok=True)


def test_record_snippets_langchain() -> None:
    """_RECORD_SNIPPETS should have a 'langchain' entry with record() usage."""
    from agentdelta.mcp_server import _RECORD_SNIPPETS

    assert "langchain" in _RECORD_SNIPPETS
    assert "record" in _RECORD_SNIPPETS["langchain"]


def test_record_snippets_custom() -> None:
    """_RECORD_SNIPPETS should have a 'custom' entry with AgentTrace usage."""
    from agentdelta.mcp_server import _RECORD_SNIPPETS

    assert "custom" in _RECORD_SNIPPETS
    assert "AgentTrace" in _RECORD_SNIPPETS["custom"]
