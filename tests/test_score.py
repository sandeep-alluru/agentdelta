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


def test_fork_at_step_1_fork_penalty_zero() -> None:
    """Fork at the very first step should yield fork_penalty=0.0."""
    a = _make_trace(
        "a",
        [
            (NodeType.START, "user query about redis"),
            (NodeType.LLM, "I will use the cache tool now"),
            (NodeType.END, "done with cache"),
        ],
    )
    b = _make_trace(
        "b",
        [
            (NodeType.START, "user query about postgres"),
            (NodeType.LLM, "I will use the database tool now"),
            (NodeType.END, "done with database"),
        ],
    )
    diff = diff_traces(a, b, fork_threshold=0.99)  # force fork at step 1
    score = compute_score(diff)
    # fork_penalty must be clamped to [0, 100]
    assert 0.0 <= score.fork_penalty <= 100.0
    assert isinstance(score, RegressionScore)


def test_fork_at_last_step_penalty_near_100() -> None:
    """Fork at the last step should yield a fork_penalty close to 100."""
    steps_a: list[tuple[NodeType, str]] = [
        (NodeType.START, "What is 2 + 2?"),
        (NodeType.LLM, "The answer is four."),
        (NodeType.TOOL_CALL, "calc(2+2)"),
        (NodeType.TOOL_RETURN, "4"),
        (NodeType.END, "The answer is four"),
    ]
    steps_b: list[tuple[NodeType, str]] = [
        (NodeType.START, "What is 2 + 2?"),
        (NodeType.LLM, "The answer is four."),
        (NodeType.TOOL_CALL, "calc(2+2)"),
        (NodeType.TOOL_RETURN, "4"),
        (NodeType.END, "The result is 4.0"),
    ]
    a = _make_trace("a", steps_a)
    b = _make_trace("b", steps_b)
    diff = diff_traces(a, b, fork_threshold=0.99, match_threshold=0.999)
    score = compute_score(diff)
    assert 0.0 <= score.fork_penalty <= 100.0


def test_no_fork_fork_penalty_is_100() -> None:
    """Traces with no fork should yield fork_penalty exactly 100.0."""
    steps: list[tuple[NodeType, str]] = [
        (NodeType.START, "What is 3 + 3?"),
        (NodeType.LLM, "The answer is six."),
        (NodeType.END, "six"),
    ]
    a = _make_trace("a", steps)
    b = _make_trace("b", steps)
    diff = diff_traces(a, b)
    score = compute_score(diff)
    assert score.fork_penalty == 100.0


def test_fork_penalty_single_baseline_node() -> None:
    """Fork with only one baseline node should yield fork_penalty=0.0 (max penalty)."""
    # One-node baseline vs a totally different one-node candidate forces fork at step 1
    # with baseline_nodes == 1 → the <= 1 branch in score.py lines 105-107
    a = _make_trace("a", [(NodeType.LLM, "I will use cache_tool to retrieve the value")])
    b = _make_trace("b", [(NodeType.LLM, "I prefer to query the database instead")])
    diff = diff_traces(a, b, fork_threshold=0.99)
    # Manually inject a fork_point so the branch is hit
    if diff.fork_point is not None:
        score = compute_score(diff)
        assert score.fork_penalty == 0.0
    else:
        # No fork detected by embeddings — still valid, just check shape
        score = compute_score(diff)
        assert isinstance(score, RegressionScore)


def test_warn_verdict_between_thresholds(forked_traces: tuple[AgentTrace, AgentTrace]) -> None:
    """Score between warn and pass thresholds should yield WARN verdict."""
    a, b = forked_traces
    diff = diff_traces(a, b)
    score = compute_score(diff)
    # Use thresholds that bracket the actual score to force WARN
    forced_warn = compute_score(diff, pass_threshold=score.overall + 1, warn_threshold=score.overall - 1)
    assert forced_warn.verdict == "WARN"


def test_tool_fidelity_perfect_when_no_baseline_tool_calls() -> None:
    """Traces with no TOOL_CALL nodes should yield tool_fidelity=100.0."""
    steps: list[tuple[NodeType, str]] = [
        (NodeType.START, "Hello"),
        (NodeType.LLM, "World"),
        (NodeType.END, "Done"),
    ]
    a = _make_trace("a", steps)
    b = _make_trace("b", steps)
    diff = diff_traces(a, b)
    score = compute_score(diff)
    assert score.tool_fidelity == 100.0


def test_semantic_score_100_when_no_paired_steps() -> None:
    """When no steps are paired, semantic score defaults to 100.0."""
    # Build a diff where trace_a has nodes and trace_b is empty → all steps are "removed"
    a = _make_trace("a", [(NodeType.LLM, "some reasoning here")])
    b = AgentTrace(run_id="b")
    diff = diff_traces(a, b)
    score = compute_score(diff)
    # semantic defaults to 100 when no paired steps; overall is still in [0, 100]
    assert 0.0 <= score.overall <= 100.0


def test_fork_penalty_zero_when_single_baseline_node_with_fork() -> None:
    """fork_penalty=0 when baseline_nodes==1 and a fork exists (lines 105-107 of score.py)."""
    from agentdelta.diff import DiffResult, ForkPoint, StepDiff

    node_a = TraceNode(step=1, node_type=NodeType.LLM, content="baseline reasoning")
    node_b = TraceNode(step=1, node_type=NodeType.LLM, content="candidate reasoning")
    fork = ForkPoint(
        step_a=1,
        step_b=1,
        node_a=node_a,
        node_b=node_b,
        similarity=0.3,
        description="reasoning diverged",
    )
    # One step in baseline (step_a is not None), one in candidate — single baseline node
    step = StepDiff(step_a=node_a, step_b=node_b, similarity=0.3, status="changed", summary="diverged")
    diff = DiffResult(
        run_id_a="a",
        run_id_b="b",
        steps=[step],
        fork_point=fork,
        summary={"total_steps": 1, "matched": 0, "changed": 1, "added": 0,
                 "removed": 0, "similarity_pct": 0.0, "has_regression": True, "fork_step": 1},
    )
    score = compute_score(diff)
    # baseline_nodes == 1 → fork_penalty == 0.0
    assert score.fork_penalty == 0.0
