"""Tests for agentdelta.score — RegressionScore and compute_score."""

from __future__ import annotations

import pytest

from agentdelta.diff import diff_traces
from agentdelta.score import RegressionScore, compute_score
from agentdelta.trace import AgentTrace, EdgeType, NodeType, TraceEdge, TraceNode


def _make_trace(run_id: str, steps: list[tuple[NodeType, str]]) -> AgentTrace:
    trace = AgentTrace(run_id=run_id)
    for i, (ntype, content) in enumerate(steps, start=1):
        trace.add_node(TraceNode(step=i, node_type=ntype, content=content))
        if i > 1:
            trace.add_edge(TraceEdge(i - 1, i, EdgeType.SEQUENCE, ""))
    return trace


@pytest.fixture
def identical_traces() -> tuple[AgentTrace, AgentTrace]:
    steps: list[tuple[NodeType, str]] = [
        (NodeType.START, "Query: what is 2+2?"),
        (NodeType.LLM, "The answer is four."),
        (NodeType.END, "four"),
    ]
    return _make_trace("a", steps), _make_trace("b", steps)


@pytest.fixture
def forked_traces() -> tuple[AgentTrace, AgentTrace]:
    a = _make_trace(
        "a",
        [
            (NodeType.START, "Summarise this document"),
            (NodeType.LLM, "I will use the summarize_tool to condense this"),
            (NodeType.TOOL_CALL, "summarize_tool(doc='...')"),
            (NodeType.TOOL_RETURN, "Summary: short text"),
            (NodeType.END, "short text"),
        ],
    )
    b = _make_trace(
        "b",
        [
            (NodeType.START, "Summarise this document"),
            (NodeType.LLM, "Let me search for relevant sections first"),
            (NodeType.TOOL_CALL, "web_search(query='document summary')"),
            (NodeType.TOOL_RETURN, "results..."),
            (NodeType.END, "some result"),
        ],
    )
    return a, b


def test_compute_score_returns_regression_score(identical_traces: tuple[AgentTrace, AgentTrace]) -> None:
    """compute_score must return a RegressionScore dataclass."""
    a, b = identical_traces
    diff = diff_traces(a, b)
    score = compute_score(diff)
    assert isinstance(score, RegressionScore)


def test_identical_traces_pass(identical_traces: tuple[AgentTrace, AgentTrace]) -> None:
    """Identical traces should yield a high score and PASS verdict."""
    a, b = identical_traces
    diff = diff_traces(a, b)
    score = compute_score(diff)
    assert score.verdict == "PASS"
    assert score.overall >= 80.0


def test_score_components_in_range(identical_traces: tuple[AgentTrace, AgentTrace]) -> None:
    """All score components must be in [0, 100]."""
    a, b = identical_traces
    diff = diff_traces(a, b)
    score = compute_score(diff)
    for attr in ("overall", "structural", "semantic", "tool_fidelity", "fork_penalty"):
        val = getattr(score, attr)
        assert 0.0 <= val <= 100.0, f"{attr}={val} out of range"


def test_forked_traces_lower_score(forked_traces: tuple[AgentTrace, AgentTrace]) -> None:
    """Traces with a divergent tool call should score lower than identical traces."""
    a_fork, b_fork = forked_traces
    a_id = _make_trace(
        "a",
        [
            (NodeType.START, "Summarise this document"),
            (NodeType.LLM, "I will use the summarize_tool to condense this"),
            (NodeType.TOOL_CALL, "summarize_tool(doc='...')"),
            (NodeType.TOOL_RETURN, "Summary: short text"),
            (NodeType.END, "short text"),
        ],
    )
    diff_id = diff_traces(a_id, a_id)
    score_id = compute_score(diff_id)

    diff_fork = diff_traces(a_fork, b_fork)
    score_fork = compute_score(diff_fork)
    assert score_fork.overall <= score_id.overall


def test_custom_thresholds_change_verdict(forked_traces: tuple[AgentTrace, AgentTrace]) -> None:
    """Setting a very high pass_threshold should cause WARN or FAIL for divergent traces."""
    a, b = forked_traces
    diff = diff_traces(a, b)
    # Normal threshold may pass or warn for forked traces
    normal = compute_score(diff, pass_threshold=80.0, warn_threshold=60.0)
    assert normal.threshold_pass == 80.0
    assert normal.threshold_warn == 60.0
    # Very strict threshold — anything below 100 must WARN or FAIL
    strict = compute_score(diff, pass_threshold=100.1, warn_threshold=99.9)
    assert strict.verdict in ("WARN", "FAIL")
    assert strict.threshold_pass == 100.1
    assert strict.threshold_warn == 99.9


def test_empty_trace_perfect_score() -> None:
    """Two empty traces should yield overall=100, PASS."""
    a = AgentTrace(run_id="empty_a")
    b = AgentTrace(run_id="empty_b")
    diff = diff_traces(a, b)
    score = compute_score(diff)
    assert score.overall == 100.0
    assert score.verdict == "PASS"


def test_fork_penalty_zero_for_no_fork(identical_traces: tuple[AgentTrace, AgentTrace]) -> None:
    """No fork → fork_penalty should be 100 (no deduction)."""
    a, b = identical_traces
    diff = diff_traces(a, b)
    score = compute_score(diff)
    assert score.fork_penalty == 100.0


def test_verdict_fail_threshold_exit(forked_traces: tuple[AgentTrace, AgentTrace]) -> None:
    """A heavily forked pair with very high threshold should FAIL."""
    a, b = forked_traces
    diff = diff_traces(a, b)
    score = compute_score(diff, pass_threshold=99.0, warn_threshold=98.0)
    assert score.verdict == "FAIL"


def test_str_representation(identical_traces: tuple[AgentTrace, AgentTrace]) -> None:
    """__str__ should include key fields."""
    a, b = identical_traces
    score = compute_score(diff_traces(a, b))
    s = str(score)
    assert "overall=" in s
    assert "verdict=" in s
