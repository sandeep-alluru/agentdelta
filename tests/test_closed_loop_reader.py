"""Closed-loop reader/gate - empty/wrong output must fail loudly (L1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentdelta.closed_loop import (
    ClosedLoopError,
    GateOutcome,
    assert_no_regression,
    gate_traces,
)
from agentdelta.trace import AgentTrace, EdgeType, NodeType, TraceEdge, TraceNode


def _make_trace(run_id: str, steps: list[tuple[NodeType, str]]) -> AgentTrace:
    trace = AgentTrace(run_id=run_id)
    for i, (ntype, content) in enumerate(steps, start=1):
        trace.add_node(TraceNode(step=i, node_type=ntype, content=content))
        if i > 1:
            trace.add_edge(TraceEdge(i - 1, i, EdgeType.SEQUENCE, ""))
    return trace


def test_empty_baseline_fails_loud() -> None:
    baseline = AgentTrace(run_id="empty_a")
    candidate = _make_trace(
        "b",
        [(NodeType.START, "q"), (NodeType.END, "a")],
    )
    out = gate_traces(baseline, candidate)
    assert isinstance(out, GateOutcome)
    assert out.ok is False
    assert out.verdict == "FAIL_LOUD"
    assert out.exit_code == 2
    assert out.score is None
    assert "empty baseline" in out.reason.lower()


def test_empty_candidate_fails_loud() -> None:
    baseline = _make_trace(
        "a",
        [(NodeType.START, "q"), (NodeType.END, "a")],
    )
    candidate = AgentTrace(run_id="empty_b")
    out = gate_traces(baseline, candidate)
    assert out.verdict == "FAIL_LOUD"
    assert out.exit_code == 2
    assert "empty candidate" in out.reason.lower()


def test_missing_file_fails_loud(tmp_path: Path) -> None:
    missing = tmp_path / "nope.jsonl"
    baseline = _make_trace(
        "a",
        [(NodeType.START, "q"), (NodeType.END, "a")],
    )
    out = gate_traces(baseline, missing)
    assert out.verdict == "FAIL_LOUD"
    assert out.exit_code == 2
    assert "not found" in out.reason.lower()


def test_to_dict_serialisable() -> None:
    out = gate_traces(AgentTrace(run_id="a"), AgentTrace(run_id="b"))
    payload = out.to_dict()
    assert payload["verdict"] == "FAIL_LOUD"
    assert payload["ok"] is False
    assert payload["score"] is None


def test_identical_traces_pass_via_gate() -> None:
    steps: list[tuple[NodeType, str]] = [
        (NodeType.START, "What is 2+2?"),
        (NodeType.LLM, "The answer is four."),
        (NodeType.END, "four"),
    ]
    a = _make_trace("baseline", steps)
    b = _make_trace("candidate", steps)
    out = gate_traces(a, b)
    assert out.ok is True
    assert out.verdict == "PASS"
    assert out.exit_code == 0
    assert out.score is not None
    assert out.score.overall >= 80.0
    assert out.has_regression is False
    payload = out.to_dict()
    assert payload["score"] is not None
    assert payload["score"]["verdict"] == "PASS"


def test_tool_path_regression_not_silent_pass() -> None:
    """Divergent tool selection must not produce a silent ok=True PASS."""
    a = _make_trace(
        "baseline",
        [
            (NodeType.START, "Check weather in London"),
            (NodeType.LLM, "I will call the weather tool"),
            (NodeType.TOOL_CALL, "get_weather(location='London')"),
            (NodeType.TOOL_RETURN, '{"temp": 18}'),
            (NodeType.END, "18C cloudy"),
        ],
    )
    b = _make_trace(
        "candidate",
        [
            (NodeType.START, "Check weather in London"),
            (NodeType.LLM, "I will search the web instead"),
            (NodeType.TOOL_CALL, "web_search(query='London weather')"),
            (NodeType.TOOL_RETURN, "BBC says cloudy"),
            (NodeType.END, "18C cloudy"),
        ],
    )
    out = gate_traces(a, b, pass_threshold=95.0, warn_threshold=90.0, warn_is_ok=False)
    # Must not be a silent success: either FAIL/WARN-as-not-ok or FAIL_LOUD
    assert out.verdict in {"FAIL", "WARN", "FAIL_LOUD"}
    assert out.ok is False
    assert out.exit_code in {1, 2}
    assert out.reason


def test_assert_no_regression_raises_on_empty() -> None:
    with pytest.raises(ClosedLoopError, match="FAIL_LOUD"):
        assert_no_regression(AgentTrace(run_id="a"), AgentTrace(run_id="b"))


def test_assert_no_regression_returns_on_match() -> None:
    steps: list[tuple[NodeType, str]] = [
        (NodeType.START, "ping"),
        (NodeType.END, "pong"),
    ]
    out = assert_no_regression(_make_trace("a", steps), _make_trace("b", steps))
    assert out.ok is True
