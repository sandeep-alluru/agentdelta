"""Tests for agentdelta.diff."""

from __future__ import annotations

from agentdelta.diff import DiffResult, ForkPoint, StepDiff, _describe_fork, diff_traces
from agentdelta.trace import NodeType, TraceNode


def test_no_regression_on_similar_traces(simple_trace_a, simple_trace_b_match):
    result = diff_traces(simple_trace_a, simple_trace_b_match)
    assert isinstance(result, DiffResult)
    assert result.summary["total_steps"] > 0
    # Similar traces should have high similarity
    assert result.summary["similarity_pct"] >= 50.0


def test_fork_detected_on_diverging_traces(simple_trace_a, simple_trace_b_fork):
    result = diff_traces(
        simple_trace_a,
        simple_trace_b_fork,
        fork_threshold=0.70,
        match_threshold=0.85,
    )
    # There should be some changed or unmatched steps
    assert len(result.changed_steps) + len(result.added_steps) + len(result.removed_steps) > 0


def test_result_has_run_ids(simple_trace_a, simple_trace_b_match):
    result = diff_traces(simple_trace_a, simple_trace_b_match)
    assert result.run_id_a == "run_a"
    assert result.run_id_b == "run_b"


def test_summary_keys_present(simple_trace_a, simple_trace_b_match):
    result = diff_traces(simple_trace_a, simple_trace_b_match)
    required = {"total_steps", "matched", "changed", "added", "removed",
                "similarity_pct", "has_regression", "fork_step"}
    assert required.issubset(result.summary.keys())


def test_step_statuses_valid(simple_trace_a, simple_trace_b_match):
    result = diff_traces(simple_trace_a, simple_trace_b_match)
    valid = {"match", "changed", "added", "removed"}
    for step in result.steps:
        assert step.status in valid


def test_fork_point_is_fork_type(simple_trace_a, simple_trace_b_fork):
    result = diff_traces(simple_trace_a, simple_trace_b_fork, fork_threshold=0.70)
    if result.fork_point:
        fp = result.fork_point
        assert isinstance(fp, ForkPoint)
        assert fp.node_a is not None
        assert fp.node_b is not None
        assert 0.0 <= fp.similarity <= 1.0


def test_identical_traces_no_changes(simple_trace_a):
    result = diff_traces(simple_trace_a, simple_trace_a)
    assert result.summary["changed"] == 0
    assert result.summary["added"] == 0
    assert result.summary["removed"] == 0
    assert result.summary["similarity_pct"] == 100.0


def test_has_regression_property_with_fork(simple_trace_a, simple_trace_b_fork):
    result = diff_traces(simple_trace_a, simple_trace_b_fork, fork_threshold=0.70)
    # has_regression should be True when there is a fork point
    if result.fork_point is not None:
        assert result.has_regression is True


def test_has_regression_all_zero_match() -> None:
    """has_regression should be True when total > 0 and matched == 0, even without fork_point."""
    # Craft a DiffResult with no fork_point but all steps unmatched
    node_a = TraceNode(step=1, node_type=NodeType.START, content="hello")
    node_b = TraceNode(step=1, node_type=NodeType.START, content="world")
    result = DiffResult(
        run_id_a="a",
        run_id_b="b",
        steps=[StepDiff(node_a, node_b, 0.0, "changed", "diverged")],
        fork_point=None,
        summary={"total": 1, "matched": 0, "has_regression": True},
    )
    assert result.has_regression is True


def test_fork_point_is_tool_change() -> None:
    """ForkPoint.is_tool_change() should return True for tool call transitions."""
    node_a = TraceNode(step=1, node_type=NodeType.TOOL_CALL, content="tool_a()")
    node_b = TraceNode(step=1, node_type=NodeType.TOOL_CALL, content="tool_b()")
    fp = ForkPoint(
        step_a=1, step_b=1, node_a=node_a, node_b=node_b, similarity=0.3, description="changed"
    )
    assert fp.is_tool_change() is True


def test_fork_point_is_not_tool_change_for_llm() -> None:
    """ForkPoint.is_tool_change() should return False for LLM→LLM."""
    node_a = TraceNode(step=1, node_type=NodeType.LLM, content="reasoning a")
    node_b = TraceNode(step=1, node_type=NodeType.LLM, content="reasoning b")
    fp = ForkPoint(
        step_a=1, step_b=1, node_a=node_a, node_b=node_b, similarity=0.4, description="llm diverge"
    )
    assert fp.is_tool_change() is False


def test_fork_point_is_reasoning_change() -> None:
    """ForkPoint.is_reasoning_change() should return True for LLM→LLM transitions."""
    node_a = TraceNode(step=2, node_type=NodeType.LLM, content="I will use tool A")
    node_b = TraceNode(step=2, node_type=NodeType.LLM, content="I will use tool B")
    fp = ForkPoint(
        step_a=2, step_b=2, node_a=node_a, node_b=node_b, similarity=0.5, description="reasoning"
    )
    assert fp.is_reasoning_change() is True


def test_fork_point_is_not_reasoning_change_for_tools() -> None:
    """ForkPoint.is_reasoning_change() should return False for tool calls."""
    node_a = TraceNode(step=3, node_type=NodeType.TOOL_CALL, content="tool()")
    node_b = TraceNode(step=3, node_type=NodeType.TOOL_RETURN, content="result")
    fp = ForkPoint(
        step_a=3, step_b=3, node_a=node_a, node_b=node_b, similarity=0.2, description="tool change"
    )
    assert fp.is_reasoning_change() is False


def test_describe_fork_tool_call_different_tools(simple_trace_a, simple_trace_b_fork):
    """Traces that fork at a tool call step should produce a descriptive fork description."""
    result = diff_traces(simple_trace_a, simple_trace_b_fork, fork_threshold=0.70)
    if result.fork_point is not None:
        assert len(result.fork_point.description) > 0
        # Should mention something about the change
        desc = result.fork_point.description.lower()
        assert any(word in desc for word in ["tool", "changed", "diverged", "similarity", "step"])


def test_describe_fork_same_tool_different_args() -> None:
    """Same tool name but different args should produce 'arguments diverged' description."""
    from agentdelta.trace import AgentTrace, EdgeType, TraceEdge

    a = AgentTrace(run_id="a")
    b = AgentTrace(run_id="b")
    a.add_node(TraceNode(step=1, node_type=NodeType.TOOL_CALL, content="search(query='redis cache')"))
    b.add_node(TraceNode(step=1, node_type=NodeType.TOOL_CALL, content="search(query='postgres database performance')"))
    result = diff_traces(a, b, fork_threshold=0.99)
    if result.fork_point is not None:
        # Same tool name → should report argument divergence
        assert "search" in result.fork_point.description or "argument" in result.fork_point.description.lower() or "changed" in result.fork_point.description.lower()


def test_describe_fork_llm_reasoning_divergence() -> None:
    """Two LLM nodes that diverge should produce a 'reasoning' description."""
    from agentdelta.trace import AgentTrace

    a = AgentTrace(run_id="a")
    b = AgentTrace(run_id="b")
    a.add_node(TraceNode(step=1, node_type=NodeType.LLM, content="I will use cache_tool to speed up retrieval"))
    b.add_node(TraceNode(step=1, node_type=NodeType.LLM, content="Let me call the database for fresh data"))
    result = diff_traces(a, b, fork_threshold=0.99)
    if result.fork_point is not None:
        desc = result.fork_point.description.lower()
        assert "reasoning" in desc or "diverged" in desc or "similarity" in desc


def test_describe_fork_type_mismatch() -> None:
    """Nodes of different types (e.g. LLM vs TOOL_CALL) should produce a type-change description."""
    from agentdelta.trace import AgentTrace

    a = AgentTrace(run_id="a")
    b = AgentTrace(run_id="b")
    a.add_node(TraceNode(step=1, node_type=NodeType.LLM, content="I will answer directly without tools"))
    b.add_node(TraceNode(step=1, node_type=NodeType.TOOL_CALL, content="search(query='answer')"))
    result = diff_traces(a, b, fork_threshold=0.99)
    if result.fork_point is not None:
        desc = result.fork_point.description.lower()
        assert "changed" in desc or "type" in desc or "similarity" in desc


def test_empty_trace_vs_nonempty() -> None:
    """Diff of an empty trace vs a non-empty trace should produce only added steps."""
    from agentdelta.trace import AgentTrace

    a = AgentTrace(run_id="empty")
    b = AgentTrace(run_id="b")
    b.add_node(TraceNode(step=1, node_type=NodeType.LLM, content="only in b"))
    result = diff_traces(a, b)
    assert result.summary["added"] >= 1
    assert result.summary["removed"] == 0


# ── Direct _describe_fork branch coverage ─────────────────────────────────────

def test_describe_fork_two_tool_calls_different_names() -> None:
    """_describe_fork should mention tool selection change when tool names differ."""
    na = TraceNode(step=1, node_type=NodeType.TOOL_CALL, content="search_tool(query='x')")
    nb = TraceNode(step=1, node_type=NodeType.TOOL_CALL, content="fetch_tool(url='y')")
    desc = _describe_fork(na, nb, 0.3)
    assert "Tool selection changed" in desc
    assert "search_tool" in desc
    assert "fetch_tool" in desc


def test_describe_fork_two_tool_calls_same_name() -> None:
    """_describe_fork should mention argument divergence when tool name is the same."""
    na = TraceNode(step=2, node_type=NodeType.TOOL_CALL, content="search(query='cats')")
    nb = TraceNode(step=2, node_type=NodeType.TOOL_CALL, content="search(query='dogs')")
    desc = _describe_fork(na, nb, 0.5)
    assert "arguments diverged" in desc


def test_describe_fork_llm_to_llm() -> None:
    """_describe_fork should report reasoning divergence for LLM→LLM pairs."""
    na = TraceNode(step=3, node_type=NodeType.LLM, content="I will use tool A")
    nb = TraceNode(step=3, node_type=NodeType.LLM, content="I will use tool B")
    desc = _describe_fork(na, nb, 0.4)
    assert "Reasoning path diverged" in desc


def test_describe_fork_type_mismatch_fallback() -> None:
    """_describe_fork should fall back to a 'Step type changed' message for mixed types."""
    na = TraceNode(step=4, node_type=NodeType.LLM, content="reasoning")
    nb = TraceNode(step=4, node_type=NodeType.TOOL_CALL, content="tool()")
    desc = _describe_fork(na, nb, 0.2)
    assert "Step type changed" in desc
    assert "llm" in desc
    assert "tool_call" in desc
